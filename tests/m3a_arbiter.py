"""Independent long-double arbiter for gate G5 (review M3a #6).

CURN likelihood (shape) and its gradient for fixed white noise, power-law IRN / DM / common blocks
on distinct grids, computed by a path that shares no numerics with ``ptagwb.combined``:

* own Fourier columns (sin/cos on f_k = k / T, chromatic scale (fref/nu)^idx);
* the white-noise covariance rebuilt from its epoch tables and whitened component by component
  with a hand-written long-double Cholesky (production: float64 Cholesky/whitening and QR);
* timing marginalisation by the *normal-equation projector* in long double,
  A = F^T N^-1 F - (M^T N^-1 F)^T (M^T N^-1 M)^-1 (M^T N^-1 F), b likewise (production: QR of the
  whitened timing basis and orthogonal projection, float64);
* the reduced covariance Sigma = Phi^-1 + A factorised by a hand-written long-double Cholesky
  (production: Householder QR of [R_F Phi^1/2; I] in float64);
* the gradient from the analytic trace derivative
  d lnL / d phi_k = 1/2 phi_k^-2 (x_k^2 + [Sigma^-1]_kk) - 1/2 phi_k^-1,  x = Sigma^-1 b,
  chained with d phi / d log10_A = 2 ln10 phi and d phi / d gamma = phi ln(f_yr / f)
  (production: the analytic VJP of the square-root reducer).

lnL is returned without its parameter-independent constant: compare *shapes* (differences between
points), which are invariant to likelihood constant conventions.
"""

from __future__ import annotations

import numpy as np
from dense_oracle import chol_ld, fsub_ld

LD = np.longdouble
FYR = LD(1) / (LD(365.25) * LD(86400))


def _cols(toas, freqs, n, T, idx=0.0, fref=1400.0):
    t = np.asarray(toas, dtype=np.float64)
    k = np.arange(1, n + 1)
    arg = 2 * np.pi * t[:, None] * (k / T)[None, :]
    F = np.empty((len(t), 2 * n))
    F[:, 0::2], F[:, 1::2] = np.sin(arg), np.cos(arg)
    if idx:
        F *= ((fref / np.asarray(freqs)) ** idx)[:, None]
    f = np.repeat(np.asarray(k, dtype=LD) / LD(T), 2)
    return F, f, LD(1) / LD(T)


def _phi(f, df, log10_A, gamma):
    return LD(10) ** (2 * LD(log10_A)) / (12 * LD(np.pi) ** 2) * FYR ** (LD(gamma) - 3) * f ** (-LD(gamma)) * df


def _gram_ld(wn, X):
    """X^T N^-1 X in long double, with N rebuilt from the white-noise description (diagonal +
    ECORR epochs) and whitened component by component with a long-double Cholesky. Independent of
    GeneralWhiteNoise's factorisation (only its epoch tables are read)."""
    n = len(wn.ndiag)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    epochs = [wn.ep_toa[wn.ep_ptr[e]:wn.ep_ptr[e + 1]] for e in range(len(wn.ep_var))]
    for idx in epochs:
        r0 = find(int(idx[0]))
        for i in idx[1:]:
            ri = find(int(i))
            if ri != r0:
                parent[ri] = r0
    roots = np.array([find(i) for i in range(n)])
    comp = {}
    for i, rt in enumerate(roots):
        comp.setdefault(int(rt), []).append(i)
    ep_of = {}
    for e, idx in enumerate(epochs):
        ep_of.setdefault(int(roots[idx[0]]), []).append(e)
    Xl = np.asarray(X, dtype=LD)
    G = np.zeros((X.shape[1], X.shape[1]), dtype=LD)
    single = np.array([v[0] for v in comp.values() if len(v) == 1], dtype=np.int64)
    if len(single):
        dvar = np.asarray(wn.ndiag, dtype=LD)[single].copy()
        pos = {int(i): j for j, i in enumerate(single)}
        for rt in (int(roots[i]) for i in single):
            for e in ep_of.get(rt, []):
                dvar[pos[int(epochs[e][0])]] += LD(wn.ep_var[e])
        Y = Xl[single] / np.sqrt(dvar)[:, None]
        G += Y.T @ Y
    for rt, members in comp.items():
        if len(members) == 1:
            continue
        idx = np.array(members)
        loc = {int(i): j for j, i in enumerate(idx)}
        C = np.diag(np.asarray(wn.ndiag, dtype=LD)[idx])
        for e in ep_of.get(rt, []):
            li = np.array([loc[int(i)] for i in epochs[e]])
            C[np.ix_(li, li)] += LD(wn.ep_var[e])
        L = chol_ld(C)
        Y = fsub_ld(L, Xl[idx])
        G += Y.T @ Y
    return G


class PulsarArbiter:
    """Stage 1 of one pulsar in long double (normal-equation projector)."""

    def __init__(self, psr, wn, model):
        rn, dm, cm = model.sampled["rn"], model.sampled["dm"], model.common
        Fr, self.fr, self.dfr = _cols(psr.toas, psr.freqs, rn.n_modes, rn.T)
        Fd, self.fd, self.dfd = _cols(psr.toas, psr.freqs, dm.n_modes, dm.T, dm.chrom_idx, dm.fref)
        Fc, self.fc, self.dfc = _cols(psr.toas, psr.freqs, cm.n_modes, cm.T)
        self.nr, self.nd, self.nc = Fr.shape[1], Fd.shape[1], Fc.shape[1]
        F = np.hstack([Fr, Fd, Fc])
        M = np.asarray(psr.Mmat, dtype=np.float64)
        # orthonormal (Euclidean) basis of span(M): the normal equations below square the condition
        # number of the whitened timing matrix, which for raw columns exceeds long-double range
        M, _, _ = np.linalg.svd(M / np.linalg.norm(M, axis=0), full_matrices=False)
        r = np.asarray(psr.residuals, dtype=np.float64)
        X = np.column_stack([M, F, r])
        G = _gram_ld(wn, X)  # long-double Gram matrix of [M, F, r] in the N^-1 metric
        m, K = M.shape[1], F.shape[1]
        Gmm, Gmf, Gmr = G[:m, :m], G[:m, m:m + K], G[:m, m + K]
        L = chol_ld(Gmm)
        Zf, zr = fsub_ld(L, Gmf), fsub_ld(L, Gmr[:, None])[:, 0]
        self.A = G[m:m + K, m:m + K] - Zf.T @ Zf
        self.b = G[m:m + K, m + K] - Zf.T @ zr

    def phi_and_derivs(self, p):
        out = []
        for f, df, A, g in ((self.fr, self.dfr, p["rn_log10_A"], p["rn_gamma"]),
                            (self.fd, self.dfd, p["dm_log10_A"], p["dm_gamma"]),
                            (self.fc, self.dfc, p["log10_A"], p["gamma"])):
            ph = _phi(f, df, A, g)
            out.append((ph, 2 * np.log(LD(10)) * ph, ph * np.log(FYR / f)))
        return out

    def logL_grad(self, p):
        parts = self.phi_and_derivs(p)
        phi = np.concatenate([x[0] for x in parts])
        S = self.A + np.diag(1 / phi)
        L = chol_ld(S)
        y = fsub_ld(L, self.b[:, None])[:, 0]
        Linv = fsub_ld(L, np.eye(len(phi), dtype=LD))
        x = Linv.T @ y
        Sinv_diag = np.sum(Linv * Linv, axis=0)
        ll = 0.5 * (y @ y) - 0.5 * np.sum(np.log(phi)) - np.sum(np.log(np.diag(L)))
        gk = 0.5 * (x * x + Sinv_diag) / phi**2 - 0.5 / phi
        sl = [slice(0, self.nr), slice(self.nr, self.nr + self.nd), slice(self.nr + self.nd, None)]
        g = {}
        for (ph, dA, dg), s, nm in zip(parts, sl, ("rn", "dm", "gw"), strict=True):
            g[f"{nm}_log10_A"] = np.sum(gk[s] * dA)
            g[f"{nm}_gamma"] = np.sum(gk[s] * dg)
        return ll, g


def arbiter_curn(arbs, p):
    """Sum over pulsars: (lnL shape, gradient dict in GeneralPTALikelihood's parameter layout)."""
    ll, grads = LD(0), {"rn_log10_A": [], "rn_gamma": [], "dm_log10_A": [], "dm_gamma": [], "log10_A": LD(0),
                        "gamma": LD(0)}
    for i, a in enumerate(arbs):
        pi = {"rn_log10_A": p["rn_log10_A"][i], "rn_gamma": p["rn_gamma"][i], "dm_log10_A": p["dm_log10_A"][i],
              "dm_gamma": p["dm_gamma"][i], "log10_A": p["log10_A"], "gamma": p["gamma"]}
        l, g = a.logL_grad(pi)
        ll += l
        for k in ("rn_log10_A", "rn_gamma", "dm_log10_A", "dm_gamma"):
            grads[k].append(g[k])
        grads["log10_A"] += g["gw_log10_A"]
        grads["gamma"] += g["gw_gamma"]
    return float(ll), {k: np.asarray(v, dtype=np.float64) for k, v in grads.items()}


def arbiter_correlated(arbs, p, Gamma):
    """lnL shape for a correlated common process (any ORF Gamma), joint over pulsars in long
    double: Phi has Gamma_ab phi_c on the common columns of pulsars a, b; lnL = 1/2 b^T (Phi^-1 +
    A)^-1 b - 1/2 log|Phi| - 1/2 log|Phi^-1 + A| (block-diagonal A, b from each PulsarArbiter)."""
    sizes = [a.nr + a.nd + a.nc for a in arbs]
    off = np.concatenate([[0], np.cumsum(sizes)])
    n = int(off[-1])
    Phi = np.zeros((n, n), dtype=LD)
    A = np.zeros((n, n), dtype=LD)
    b = np.zeros(n, dtype=LD)
    phic = None
    for i, a in enumerate(arbs):
        pi = {"rn_log10_A": p["rn_log10_A"][i], "rn_gamma": p["rn_gamma"][i], "dm_log10_A": p["dm_log10_A"][i],
              "dm_gamma": p["dm_gamma"][i], "log10_A": p["log10_A"], "gamma": p["gamma"]}
        parts = a.phi_and_derivs(pi)
        s0 = off[i]
        Phi[s0:s0 + a.nr, s0:s0 + a.nr] = np.diag(parts[0][0])
        Phi[s0 + a.nr:s0 + a.nr + a.nd, s0 + a.nr:s0 + a.nr + a.nd] = np.diag(parts[1][0])
        phic = parts[2][0]
        A[s0:off[i + 1], s0:off[i + 1]] = a.A
        b[s0:off[i + 1]] = a.b
    for i, ai in enumerate(arbs):
        for j, aj in enumerate(arbs):
            ci, cj = off[i] + ai.nr + ai.nd, off[j] + aj.nr + aj.nd
            Phi[ci:ci + ai.nc, cj:cj + aj.nc] += LD(Gamma[i, j]) * np.diag(phic)
    Lp = chol_ld(Phi)
    Lpinv = fsub_ld(Lp, np.eye(n, dtype=LD))
    S = Lpinv.T @ Lpinv + A
    L = chol_ld(S)
    y = fsub_ld(L, b[:, None])[:, 0]
    return float(0.5 * (y @ y) - np.sum(np.log(np.diag(Lp))) - np.sum(np.log(np.diag(L))))
