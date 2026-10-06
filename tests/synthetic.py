"""Small synthetic PTA for unit tests, plus a dense brute-force reference likelihood."""

from __future__ import annotations

import numpy as np

from ptagwb.basis import fourier_basis, free_spectrum, powerlaw
from ptagwb.data import Pulsar, get_tspan
from ptagwb.likelihood import TIMING_PRIOR_VARIANCE, timing_basis
from ptagwb.noise import build_white_noise


def make_pta(n_psr: int = 5, seed: int = 0, n_epochs: int = 25, signal: float = 0.0) -> tuple[list[Pulsar], dict]:
    """Pulsars with multi-channel epochs on two backends, a quadratic+offset+sinusoid timing
    model and white residuals (optionally plus a common red process of rms ``signal`` s)."""
    rng = np.random.default_rng(seed)
    psrs, nd = [], {}
    t0 = 53000.0 * 86400.0
    for a in range(n_psr):
        name = f"J{1000 + a:04d}+0000"
        start = t0 + rng.uniform(0, 2) * 365.25 * 86400
        span = rng.uniform(6, 10) * 365.25 * 86400
        epochs = np.sort(start + rng.uniform(0, span, n_epochs))
        toas, flags = [], []
        for e in epochs:
            be = "430_PUPPI" if rng.random() < 0.5 else "L-wide_ASP"
            nch = rng.integers(1, 5)  # singletons happen -> no ECORR for them (nmin=2)
            toas.extend(e + 0.1 * np.arange(nch) + rng.uniform(0, 0.05, nch))
            flags.extend([be] * nch)
        toas = np.array(toas)
        isort = np.argsort(toas, kind="mergesort")
        toas, flags = toas[isort], np.array(flags)[isort]
        n = len(toas)
        err = rng.uniform(0.3, 2.0, n) * 1e-6
        tt = (toas - toas.mean()) / span
        M = np.column_stack([np.ones(n), tt, tt**2, np.sin(2 * np.pi * toas / 3.15576e7), rng.normal(size=n) * 1e-3])
        res = err * rng.normal(size=n)
        if signal:
            res = res + signal * np.sin(2 * np.pi * (toas - t0) / (8 * 365.25 * 86400) + a)
        pos = rng.normal(size=3)
        pos /= np.linalg.norm(pos)
        psrs.append(
            Pulsar(
                name=name,
                toas=toas,
                stoas=toas,
                residuals=res,
                toaerrs=err,
                freqs=np.full(n, 1400.0),
                freqs_topo=np.full(n, 1400.0),
                backend_flags=flags.astype("U"),
                telescope=np.array(["ao"] * n),
                Mmat=M,
                fitpars=["Offset", "F0", "F1", "PX", "X"],
                pos=pos,
            )
        )
        for be in ("430_PUPPI", "L-wide_ASP"):
            nd[f"{name}_{be}_efac"] = rng.uniform(0.8, 1.5)
            nd[f"{name}_{be}_log10_t2equad"] = rng.uniform(-7.5, -6.0)
            nd[f"{name}_{be}_log10_ecorr"] = rng.uniform(-7.0, -6.0)
    return psrs, nd


def dense_loglike(
    psrs: list[Pulsar],
    nd: dict,
    T: float,
    n_modes: int,
    n_common: int,
    rn_log10_A,
    rn_gamma,
    Gamma: np.ndarray | None,
    log10_A=None,
    gamma=None,
    log10_rho=None,
    timing: str = "svd",
    nmin: int = 2,
) -> float:
    """Brute force: full TOA-space covariance of the whole PTA, timing model marginalised
    by the projector formula, all dense numpy float64 (chain convention constants)."""
    P = len(psrs)
    Ns, Fs, Ms, rs = [], [], [], []
    for p in psrs:
        Ns.append(build_white_noise(p, nd, nmin=nmin).dense())
        f, df, F = fourier_basis(p.toas, n_modes, T)
        Fs.append(F)
        Ms.append(timing_basis(p.Mmat, timing))
        rs.append(p.residuals)
    sizes = [len(r) for r in rs]
    off = np.concatenate([[0], np.cumsum(sizes)])
    n = off[-1]
    C = np.zeros((n, n))
    for a in range(P):
        C[off[a] : off[a + 1], off[a] : off[a + 1]] += Ns[a]
    phi_rn = [powerlaw(f, df, rn_log10_A[a], rn_gamma[a]) for a in range(P)]
    nc2 = 2 * n_common
    if log10_rho is not None:
        phic = free_spectrum(np.asarray(log10_rho))
    else:
        phic = powerlaw(f[:nc2], df[:nc2], log10_A, gamma)
    G = np.eye(P) if Gamma is None else Gamma
    for a in range(P):
        for b in range(P):
            cov = np.zeros(2 * n_modes)
            if a == b:
                cov = cov + phi_rn[a]
            cov[:nc2] += G[a, b] * phic
            C[off[a] : off[a + 1], off[b] : off[b + 1]] += (Fs[a] * cov) @ Fs[b].T
    M = np.zeros((n, sum(m.shape[1] for m in Ms)))
    c = 0
    for a in range(P):
        M[off[a] : off[a + 1], c : c + Ms[a].shape[1]] = Ms[a]
        c += Ms[a].shape[1]
    r = np.concatenate(rs)
    Ci = np.linalg.inv(C)
    MCM = M.T @ Ci @ M
    Pm = Ci - Ci @ M @ np.linalg.solve(MCM, M.T @ Ci)
    return float(
        -0.5
        * (
            r @ Pm @ r
            + np.linalg.slogdet(C)[1]
            + np.linalg.slogdet(MCM)[1]
            + M.shape[1] * np.log(TIMING_PRIOR_VARIANCE)
        )
    )


def tspan(psrs):
    return get_tspan(psrs)
