"""Independent long-double arbiter for G5-PTA on the EPTA DR2new model (docs/M3B_PLAN.md Sec. 4.4,
6.2), extending the M3a arbiter (``tests/m3a_arbiter.py``) to the EPTA blocks and the dip (N3).

Shares no numerics with ``ptagwb.epta`` / ``ptagwb.combined``:

* own Fourier columns per block: sin/cos(2 pi k t / T_a), chromatic (1400/nu)^idx and the
  TempoNest factor sqrt(12) pi / (1400^2 2.41e-4), evaluated in long double from the float64
  arrays; the common block on the array span;
* white noise N_ii = efac^2 sigma_i^2 + 10^(2 log10_tnequad) in long double (diagonal: EPTA has no
  ECORR);
* timing marginalisation by the normal-equation projector in long double on an orthonormal
  (Euclidean) basis of span(M) (production: QR of the whitened SVD basis);
* the dip delay and its derivatives in long double, subtracted from the residuals before the
  long-double Gram matrices are formed;
* CURN: per-pulsar Sigma_a = Phi_a^-1 + A_a by a hand-written long-double Cholesky; HD: exact
  Schur elimination of every pulsar's noise columns, then the joint common system
  (Gamma (x) diag(phi_c))^-1 + A'_cc in long double;
* gradients (CURN): analytic trace formula for the power-law hyperparameters and
  d lnL / d theta_dip = r~^T K dd/dtheta (K the GP- and timing-marginalised inverse covariance);
  HD gradients by long-double central differences (``fd_ld``).

lnL is returned without the parameter-independent constants (timing-basis and white-noise
log-determinants): compare *shapes* (differences between points) and the cross-model difference
lnL_HD - lnL_CURN at the same point, in which every remaining model-dependent term (GP and common
log-determinants, ORF) is included.
"""

from __future__ import annotations

import numpy as np
from dense_oracle import chol_ld, fsub_ld

LD = np.longdouble
FYR = LD(1) / (LD(365.25) * LD(86400))
TN = LD(np.sqrt(12.0)) * LD(np.pi) / (LD(1400) ** 2 * LD("2.41e-4"))
LN10 = np.log(LD(10))


def _cols(toas, freqs, n, T, idx=0.0, tn=False):
    t = np.asarray(toas, dtype=LD)
    k = np.arange(1, n + 1, dtype=LD)
    arg = 2 * LD(np.pi) * t[:, None] * (k / LD(T))[None, :]
    F = np.empty((len(t), 2 * n), dtype=LD)
    F[:, 0::2], F[:, 1::2] = np.sin(arg), np.cos(arg)
    if idx:
        s = (LD(1400) / np.asarray(freqs, dtype=LD)) ** LD(idx)
        F *= (s * TN if tn else s)[:, None]
    return F, np.repeat(k / LD(T), 2), LD(1) / LD(T)


def _phi(f, df, log10_A, gamma):
    return LD(10) ** (2 * LD(log10_A)) / (12 * LD(np.pi) ** 2) * FYR ** (LD(gamma) - 3) * f ** (-LD(gamma)) * df


class ArbiterPulsar:
    """Stage 1 of one EPTA pulsar in long double. ``blocks``: manifest block specs; ``dip``: the
    manifest dip spec if this is the dip pulsar."""

    def __init__(self, psr, wn_values: dict, blocks: list, T_common: float, n_common: int, dip: dict | None = None):
        self.name = psr.name
        T_a = LD(psr.toas.max()) - LD(psr.toas.min())
        cols, self.groups = [], []
        for b in blocks:
            F, f, df = _cols(psr.toas, psr.freqs, int(b["modes"]), T_a, b["chrom_idx"], b["norm"] == "temponest_dm")
            cols.append(F)
            self.groups.append((b["name"], f, df))
        Fc, fc, dfc = _cols(psr.toas, psr.freqs, n_common, T_common)
        cols.append(Fc)
        self.groups.append(("common", fc, dfc))
        self.F = np.hstack(cols)
        self.nc = Fc.shape[1]
        sig = np.asarray(psr.toaerrs, dtype=LD)
        nd = np.empty(len(sig), dtype=LD)
        for bk in np.unique(psr.backend_flags):
            m = psr.backend_flags == bk
            nd[m] = LD(wn_values[f"{psr.name}_{bk}_efac"]) ** 2 * sig[m] ** 2 + \
                LD(10) ** (2 * LD(wn_values[f"{psr.name}_{bk}_log10_tnequad"]))
        self.ninv = 1 / nd
        Mn = psr.Mmat / np.linalg.norm(psr.Mmat, axis=0)
        U, _, _ = np.linalg.svd(Mn, full_matrices=False)
        self.M = np.asarray(U, dtype=LD)
        self.r = np.asarray(psr.residuals, dtype=LD)
        self.dip = dip
        if dip is not None:
            self.t = np.asarray(psr.toas, dtype=LD)
            self.chrom = (LD(1400) / np.asarray(psr.freqs, dtype=LD)) ** LD(dip["chrom_idx"])
        # parts of the Gram matrix that do not depend on the residual
        W = self.ninv[:, None]
        Gmm = self.M.T @ (W * self.M)
        self.Gmf = self.M.T @ (W * self.F)
        self.Gff = self.F.T @ (W * self.F)
        self.Lm = chol_ld(Gmm)
        self.Zf = fsub_ld(self.Lm, self.Gmf)
        self.A = self.Gff - self.Zf.T @ self.Zf
        self._fixed = None if dip is not None else self._resid_terms(self.r, [])

    def delay(self, x_dip):
        """(d, [dd/dlog10_Amp, dd/dlog10_tau]) in long double; x_dip = (log10_Amp, log10_tau, t0)."""
        a, lt, t0 = (LD(v) for v in x_dip)
        tau = LD(10) ** lt * LD(86400)
        dt = self.t - t0 * LD(86400)
        on = dt >= 0
        wf = np.where(on, np.exp(-np.where(on, dt, 0) / tau), LD(0))
        d = LD(self.dip["sign"]) * LD(10) ** a * wf * self.chrom
        return d, [LN10 * d, d * np.where(on, dt, 0) / tau * LN10]

    def _resid_terms(self, r, gs):
        """(b, s, [b_g], [s_rg]) for residual r and extra vectors g: b = F^T P r, s = r^T P r with
        P the timing-marginalised N^-1 (normal equations, long double)."""
        X = np.column_stack([r] + list(gs)) if gs else r[:, None]
        W = self.ninv[:, None]
        Gmx = self.M.T @ (W * X)
        Gfx = self.F.T @ (W * X)
        Gxx = X.T @ (W * X)
        zx = fsub_ld(self.Lm, Gmx)
        B = Gfx - self.Zf.T @ zx
        S = Gxx - zx.T @ zx
        return B[:, 0], S[0, 0], [B[:, i] for i in range(1, X.shape[1])], [S[0, i] for i in range(1, X.shape[1])]

    def terms(self, x_dip=None, grads=False):
        if self.dip is None:
            return self._fixed
        d, gd = self.delay(x_dip)
        return self._resid_terms(self.r - d, gd if grads else [])

    def phi(self, hyper: dict):
        """Diagonal prior variances (noise + common) and their derivatives w.r.t. (log10_A, gamma)."""
        out = []
        for nm, f, df in self.groups:
            A, g = hyper[nm]
            ph = _phi(f, df, A, g)
            out.append((nm, ph, 2 * LN10 * ph, ph * np.log(FYR / f)))
        return out


def _hyper(arb, man, x, names, orf):
    ix = {n: i for i, n in enumerate(names)}
    h = {}
    for b in man["blocks"][arb.name]:
        h[b["name"]] = (x[ix[f"{arb.name}_{b['name']}_log10_A"]], x[ix[f"{arb.name}_{b['name']}_gamma"]])
    h["common"] = (x[ix[f"gw_{orf}_log10_A"]], x[ix[f"gw_{orf}_gamma"]])
    return h


def _dipx(man, x, names):
    pre = man["dip"]["param_prefix"]
    ix = {n: i for i, n in enumerate(names)}
    return [x[ix[f"{pre}_{k}"]] for k in ("log10_Amp", "log10_tau", "t0")]


def curn_logL_grad(arbs, man, x, names):
    """(lnL shape, gradient over ``names`` except the dip t0 (NaN there))."""
    ix = {n: i for i, n in enumerate(names)}
    ll = LD(0)
    grad = np.zeros(len(names), dtype=LD)
    for a in arbs:
        b, s, bg, sg = a.terms(_dipx(man, x, names), grads=True) if a.dip is not None else a.terms()
        parts = a.phi(_hyper(a, man, x, names, "crn"))
        phi = np.concatenate([p[1] for p in parts])
        S = a.A + np.diag(1 / phi)
        L = chol_ld(S)
        y = fsub_ld(L, b[:, None])[:, 0]
        Linv = fsub_ld(L, np.eye(len(phi), dtype=LD))
        xx = Linv.T @ y
        ll += -s / 2 + (y @ y) / 2 - np.sum(np.log(phi)) / 2 - np.sum(np.log(np.diag(L)))
        gk = (xx * xx + np.sum(Linv * Linv, axis=0)) / phi**2 / 2 - 1 / phi / 2
        o = 0
        for nm, ph, dA, dg in parts:
            sl = slice(o, o + len(ph))
            o += len(ph)
            if nm == "common":
                grad[ix["gw_crn_log10_A"]] += np.sum(gk[sl] * dA)
                grad[ix["gw_crn_gamma"]] += np.sum(gk[sl] * dg)
            else:
                grad[ix[f"{a.name}_{nm}_log10_A"]] += np.sum(gk[sl] * dA)
                grad[ix[f"{a.name}_{nm}_gamma"]] += np.sum(gk[sl] * dg)
        if a.dip is not None:
            pre = man["dip"]["param_prefix"]
            for k, bgi, sgi in zip(("log10_Amp", "log10_tau"), bg, sg, strict=True):
                # d lnL / d theta = r~^T K dd/dtheta = s_rg - b^T Sigma^-1 b_g
                grad[ix[f"{pre}_{k}"]] = sgi - xx @ bgi
            grad[ix[f"{pre}_t0"]] = LD(np.nan)
    return float(ll), grad.astype(np.float64)


def hd_logL(arbs, man, x, names, Gamma):
    """HD lnL shape by exact Schur elimination of each pulsar's noise columns, then the joint common
    system in long double."""
    P = len(arbs)
    nc = arbs[0].nc
    Acc = np.zeros((P * nc, P * nc), dtype=LD)
    bc = np.zeros(P * nc, dtype=LD)
    ll = LD(0)
    phic = None
    for i, a in enumerate(arbs):
        b, s, _, _ = a.terms(_dipx(man, x, names)) if a.dip is not None else a.terms()
        parts = a.phi(_hyper(a, man, x, names, "hd"))
        phin = np.concatenate([p[1] for p in parts[:-1]]) if len(parts) > 1 else np.zeros(0, dtype=LD)
        phic = parts[-1][1]
        k = len(phin)
        A = a.A
        bn, bcc = b[:k], b[k:]
        ll += -s / 2 - np.sum(np.log(phin)) / 2
        if k:
            Snn = A[:k, :k] + np.diag(1 / phin)
            L = chol_ld(Snn)
            yn = fsub_ld(L, bn[:, None])[:, 0]
            Z = fsub_ld(L, A[:k, k:])
            ll += (yn @ yn) / 2 - np.sum(np.log(np.diag(L)))
            Acc[i * nc:(i + 1) * nc, i * nc:(i + 1) * nc] = A[k:, k:] - Z.T @ Z
            bc[i * nc:(i + 1) * nc] = bcc - Z.T @ yn
        else:
            Acc[i * nc:(i + 1) * nc, i * nc:(i + 1) * nc] = A
            bc[i * nc:(i + 1) * nc] = bcc
    G = np.asarray(Gamma, dtype=LD)
    Phic = np.kron(G, np.diag(phic))
    Lp = chol_ld(Phic)
    Lpinv = fsub_ld(Lp, np.eye(P * nc, dtype=LD))
    S = Lpinv.T @ Lpinv + Acc
    L = chol_ld(S)
    y = fsub_ld(L, bc[:, None])[:, 0]
    ll += (y @ y) / 2 - np.sum(np.log(np.diag(Lp))) - np.sum(np.log(np.diag(L)))
    return float(ll)


def hd_gamma_ld(pos):
    """HD ORF in long double from float64 unit vectors (diagonal 1)."""
    p = np.asarray(pos, dtype=LD)
    c = np.clip(p @ p.T, -1, 1)
    xx = (1 - c) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        G = LD(1.5) * np.where(xx > 0, xx * np.log(np.where(xx > 0, xx, 1)), 0) - xx / 4 + LD(0.5)
    np.fill_diagonal(G, 1)
    return G


def fd_ld(f, x, idx, h=1e-5):
    """Central differences of f (returning float, computed in long double internally) at the
    coordinates ``idx`` of x."""
    g = {}
    for i in idx:
        xp, xm = np.array(x, dtype=np.float64), np.array(x, dtype=np.float64)
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def build(psrs, man):
    from ptagwb.epta import array_span

    T = array_span(psrs)
    return [ArbiterPulsar(p, man["white_noise"]["values"], man["blocks"][p.name], T, int(man["common"]["modes"]),
                          man["dip"] if p.name == man["dip"]["pulsar"] else None) for p in psrs]
