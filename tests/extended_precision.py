"""Extended-precision (np.longdouble, 64-bit mantissa on x86) per-pulsar reference likelihood.

Same fp64 inputs (residuals, TOA errors, design matrix, Fourier basis) as the float64 code, but
every subsequent operation -- whitening, Gram-Schmidt projection off the timing model,
Cholesky of the coefficient-space system -- in long double. Used to arbitrate float64
round-off differences between our likelihood and the oracles in extreme corners of the prior.
"""

from __future__ import annotations

import numpy as np

from ptagwb.basis import fourier_basis, powerlaw
from ptagwb.likelihood import TIMING_PRIOR_VARIANCE, timing_basis
from ptagwb.noise import build_white_noise

LD = np.longdouble


def _whiten_ld(wn, X):
    X = X.astype(LD)
    dm12 = (1 / np.sqrt(wn.ndiag.astype(LD)))[:, None]
    Y = X * dm12
    m = wn.epoch >= 0
    if wn.n_epoch:
        w2 = np.zeros(wn.n_epoch, dtype=LD)
        np.add.at(w2, wn.epoch[m], (dm12[m, 0] ** 2))
        J = wn.ecorr_var.astype(LD)
        alpha = (1 / np.sqrt(1 + J * w2) - 1) / w2
        wy = np.zeros((wn.n_epoch, X.shape[1]), dtype=LD)
        np.add.at(wy, wn.epoch[m], Y[m] * dm12[m])
        Y[m] += alpha[wn.epoch[m], None] * wy[wn.epoch[m]] * dm12[m]
    s = np.zeros(wn.n_epoch, dtype=LD)
    np.add.at(s, wn.epoch[m], 1 / wn.ndiag[m].astype(LD))
    logdet = np.sum(np.log(wn.ndiag.astype(LD))) + np.sum(np.log1p(wn.ecorr_var.astype(LD) * s))
    return Y, logdet


def _mgs(W):
    """Modified Gram-Schmidt with one re-orthogonalisation pass: Q, |diag R|."""
    _, m = W.shape
    Q = np.empty_like(W)
    rdiag = np.empty(m, dtype=LD)
    for j in range(m):
        v = W[:, j].copy()
        for _ in range(2):
            for i in range(j):
                v -= (Q[:, i] @ v) * Q[:, i]
        nrm = np.sqrt(v @ v)
        rdiag[j] = nrm
        Q[:, j] = v / nrm
    return Q, rdiag


def _chol_ld(S):
    n = S.shape[0]
    L = np.zeros_like(S)
    for j in range(n):
        d = S[j, j] - L[j, :j] @ L[j, :j]
        L[j, j] = np.sqrt(d)
        L[j + 1 :, j] = (S[j + 1 :, j] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
    return L


def _forward_ld(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def pulsar_contractions_ld(psr, noisedict, T, n_modes=30, timing="svd", nmin=2):
    wn = build_white_noise(psr, noisedict, nmin=nmin)
    Mt = timing_basis(psr.Mmat, timing)
    _, _, F = fourier_basis(psr.toas, n_modes, T)
    X = np.column_stack([Mt, F, psr.residuals])
    W, logdet_N = _whiten_ld(wn, X)
    m = Mt.shape[1]
    Q, rdiag = _mgs(W[:, :m])
    R = W[:, m:]
    for _ in range(2):
        R = R - Q @ (Q.T @ R)
    Fp, rp = R[:, :-1], R[:, -1]
    return {
        "A": Fp.T @ Fp,
        "b": Fp.T @ rp,
        "s": rp @ rp,
        "const": logdet_N + 2 * np.sum(np.log(rdiag)) + m * np.log(LD(TIMING_PRIOR_VARIANCE)),
    }


def curn_loglike_ld(c, T, rn_log10_A, rn_gamma, log10_A, gamma, n_modes=30, n_common=14):
    f, df, _ = fourier_basis(np.zeros(1), n_modes, T)
    phi = powerlaw(f.astype(LD), df.astype(LD), LD(rn_log10_A), LD(rn_gamma))
    phi[: 2 * n_common] += powerlaw(f[: 2 * n_common].astype(LD), df[: 2 * n_common].astype(LD), LD(log10_A), LD(gamma))
    sq = np.sqrt(phi)
    S = np.eye(len(phi), dtype=LD) + sq[:, None] * c["A"] * sq[None, :]
    L = _chol_ld(S)
    y = _forward_ld(L, sq * c["b"])
    return -0.5 * (c["s"] - y @ y + 2 * np.sum(np.log(np.diag(L))) + c["const"])
