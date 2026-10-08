"""Dense small-system oracle for the general (multi-block, multi-leg) likelihood (M3a, gate G5).

Builds the full TOA-space covariance of the whole array explicitly,

    C = blockdiag(N_a) + sum_a [sampled + fixed blocks of a] + sum_ab Gamma_ab F^c_a phi_c F^c_b^T

(white noise with overlapping ECORR epochs as dense additive terms, every GP block with its own
grid / scaling / selection), and marginalises the timing model with the projector formula

    -2 lnL = r^T C^-1 r - r^T C^-1 M (M^T C^-1 M)^-1 M^T C^-1 r + log|C| + log|M^T C^-1 M| + m log 1e40

entirely in long double (64-bit mantissa) with a hand-written Cholesky, so the reference is
~1e3 x more precise than any float64 implementation. Shares no code with ``ptagwb.combined``
beyond the block bases (``FourierBlock.basis``, which is tested separately) and the timing basis.
"""

from __future__ import annotations

import numpy as np

from ptagwb.basis import FYR
from ptagwb.likelihood import TIMING_PRIOR_VARIANCE, timing_basis

LD = np.longdouble


def chol_ld(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=LD)
    n = S.shape[0]
    L = np.zeros_like(S)
    for j in range(n):
        d = S[j, j] - L[j, :j] @ L[j, :j]
        if not d > 0:
            raise np.linalg.LinAlgError(f"not positive definite at {j}: {d}")
        L[j, j] = np.sqrt(d)
        L[j + 1 :, j] = (S[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def fsub_ld(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve L X = B (L lower triangular), long double."""
    B = np.asarray(B, dtype=LD)
    X = np.zeros_like(B)
    for i in range(L.shape[0]):
        X[i] = (B[i] - L[i, :i] @ X[:i]) / L[i, i]
    return X


def powerlaw_ld(f, df, log10_A, gamma):
    f, df = np.asarray(f, dtype=LD), np.asarray(df, dtype=LD)
    A = LD(10) ** (2 * LD(log10_A))
    return A / (12 * LD(np.pi) ** 2) * LD(FYR) ** (LD(gamma) - 3) * f ** (-LD(gamma)) * df


def dense_loglike(psrs, wns, models, params, Gamma=None, common="powerlaw", timing="svd") -> float:
    """psrs: objects with toas/freqs/backend_flags/residuals/Mmat; wns: GeneralWhiteNoise (or
    WhiteNoise) per pulsar; models: combined.PulsarGPModel per pulsar; params as for
    GeneralPTALikelihood (process arrays over the pulsars that carry the process)."""
    P = len(psrs)
    sizes = [len(p.toas) for p in psrs]
    off = np.concatenate([[0], np.cumsum(sizes)])
    n = int(off[-1])
    C = np.zeros((n, n), dtype=LD)
    counters: dict[str, int] = {}
    Fc = []
    for a, (p, wn, mdl) in enumerate(zip(psrs, wns, models, strict=True)):
        sl = slice(off[a], off[a + 1])
        C[sl, sl] += np.asarray(wn.dense(), dtype=LD)
        for nm, blk in mdl.sampled.items():
            i = counters.get(nm, 0)
            counters[nm] = i + 1
            f, df = blk.frequencies()
            phi = powerlaw_ld(f, df, np.asarray(params[f"{nm}_log10_A"])[i], np.asarray(params[f"{nm}_gamma"])[i])
            F = np.asarray(blk.basis(p.toas, p.freqs, p.backend_flags), dtype=LD)
            C[sl, sl] += (F * phi) @ F.T
        for blk, phi in mdl.fixed:
            F = np.asarray(blk.basis(p.toas, p.freqs, p.backend_flags), dtype=LD)
            C[sl, sl] += (F * np.asarray(phi, dtype=LD)) @ F.T
        Fc.append(None if mdl.common is None else np.asarray(mdl.common.basis(p.toas, p.freqs, p.backend_flags), dtype=LD))
    if Fc[0] is not None:
        f, df = models[0].common.frequencies()
        if common == "powerlaw":
            phic = powerlaw_ld(f, df, params["log10_A"], params["gamma"])
        else:
            phic = np.repeat(LD(10) ** (2 * np.asarray(params["log10_rho"], dtype=LD)), 2)
        G = np.eye(P) if Gamma is None else np.asarray(Gamma)
        for a in range(P):
            for b in range(P):
                if G[a, b] != 0:
                    C[off[a] : off[a + 1], off[b] : off[b + 1]] += LD(G[a, b]) * (Fc[a] * phic) @ Fc[b].T
    Ms = [timing_basis(p.Mmat, timing) for p in psrs]
    m = sum(x.shape[1] for x in Ms)
    M = np.zeros((n, m), dtype=LD)
    c = 0
    for a, x in enumerate(Ms):
        M[off[a] : off[a + 1], c : c + x.shape[1]] = x
        c += x.shape[1]
    r = np.concatenate([np.asarray(p.residuals, dtype=LD) for p in psrs])
    L = chol_ld(C)
    Y = fsub_ld(L, np.column_stack([M, r]))
    YM, yr = Y[:, :m], Y[:, m]
    Gm = YM.T @ YM
    Lm = chol_ld(Gm)
    z = fsub_ld(Lm, YM.T @ yr)
    quad = yr @ yr - z @ z
    logdetC = 2 * np.sum(np.log(np.diag(L)))
    logdetM = 2 * np.sum(np.log(np.diag(Lm)))
    return float(-0.5 * (quad + logdetC + logdetM + m * np.log(LD(TIMING_PRIOR_VARIANCE))))


def fd_gradient(f, params: dict, h: float = 1e-4) -> dict:
    """4th-order central differences of a long-double function of a parameter dict."""
    out = {}
    for k, v in params.items():
        flat = np.atleast_1d(np.asarray(v, dtype=np.float64)).copy()
        g = np.zeros_like(flat)
        for i in range(flat.size):
            vals = []
            for s in (-2, -1, 1, 2):
                x = flat.copy()
                x[i] += s * h
                vals.append(f(dict(params, **{k: x.reshape(np.shape(v))})))
            g[i] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
        out[k] = g.reshape(np.shape(v))
    return out
