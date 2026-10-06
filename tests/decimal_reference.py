"""Arbitrary-precision (``decimal``, 50 significant digits) coefficient-space reference.

Takes the float64 stage-1 contractions (s, b, A per pulsar; treated as exact) and evaluates
the CURN or correlated (HD) likelihood by the *joint* enterprise-style formula

    -2 log L = sum_a s_a - b^T Sigma^-1 b + log|Phi| + log|Sigma|,  Sigma = Phi^-1 + blockdiag(A_a)

with Phi the full (P*60)-dimensional coefficient prior (intrinsic RN on the diagonal plus
Gamma_ab phi^CP on the common modes). No Woodbury reduction, no cancellation-avoiding tricks:
the 50-digit arithmetic absorbs the ~1e20 condition numbers of the prior corners. Gradients by
central differences in the same precision. Slow (pure Python): use for a few pulsars.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np

getcontext().prec = 50
D = Decimal
YEAR = D(31557600)
PI = D("3.14159265358979323846264338327950288419716939937510")
TEN = D(10)


def _dec(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    if isinstance(x, np.longdouble):
        return D(np.format_float_scientific(x, unique=True))
    return D(repr(float(x)))


def _pow(x: Decimal, y: Decimal) -> Decimal:
    return (y * x.ln()).exp()


def powerlaw_dec(fk: Decimal, T: Decimal, log10_A: Decimal, gamma: Decimal) -> Decimal:
    fyr = 1 / YEAR
    A2 = _pow(TEN, 2 * log10_A)
    return A2 / (12 * PI * PI) * _pow(fyr, gamma - 3) * _pow(fk, -gamma) / T


def _chol(M):
    n = len(M)
    L = [[D(0)] * n for _ in range(n)]
    for j in range(n):
        s = M[j][j] - sum(L[j][k] * L[j][k] for k in range(j))
        L[j][j] = s.sqrt()
        for i in range(j + 1, n):
            L[i][j] = (M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
    return L


def _fwd(L, b):
    y = []
    for i in range(len(b)):
        y.append((b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i])
    return y


def _inv_spd(M):
    n = len(M)
    L = _chol(M)
    # inverse via solving L L^T X = I column by column
    cols = []
    for j in range(n):
        e = [D(int(i == j)) for i in range(n)]
        y = _fwd(L, e)
        x = [D(0)] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
        cols.append(x)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def loglike_dec(terms, T, rn_log10_A, rn_gamma, log10_A, gamma, Gamma=None, n_modes=30, n_common=14,
                include_const=False):
    """Joint-system log-likelihood in 50-digit arithmetic. ``Gamma=None`` means CURN."""
    P = len(terms)
    m = 2 * n_modes
    nc2 = 2 * n_common
    Td = D(repr(float(T)))
    fk = [D(k + 1) / Td for k in range(n_modes)]
    phi_rn = [[powerlaw_dec(fk[k // 2], Td, _dec(rn_log10_A[a]), _dec(rn_gamma[a]))
               for k in range(m)] for a in range(P)]
    phi_c = [powerlaw_dec(fk[k // 2], Td, _dec(log10_A), _dec(gamma)) for k in range(nc2)]
    G = np.eye(P) if Gamma is None else np.asarray(Gamma)
    N = P * m
    # Phi couples pulsars only mode by mode: invert / log-det the P x P block of every mode.
    logdet_phi = D(0)
    Phinv = [[D(0)] * N for _ in range(N)]
    for k in range(m):
        blk = [[(phi_rn[a][k] if a == b else D(0)) + (_dec(G[a, b]) * phi_c[k] if k < nc2 else D(0))
                for b in range(P)] for a in range(P)]
        Lk = _chol(blk)
        logdet_phi += 2 * sum(Lk[i][i].ln() for i in range(P))
        inv = _inv_spd(blk)
        for a in range(P):
            for b in range(P):
                Phinv[a * m + k][b * m + k] = inv[a][b]
    Sig = [row[:] for row in Phinv]
    bvec = []
    s_tot = D(0)
    for a, t in enumerate(terms):
        if getattr(t, "RA", None) is not None:
            # exact A = RA^T RA and b = RA^T c from the square-root representation
            RA = [[_dec(x) for x in row] for row in t.RA]
            c = [_dec(x) for x in t.c]
            Aa = [[sum(RA[k][i] * RA[k][j] for k in range(m)) for j in range(m)] for i in range(m)]
            ba = [sum(RA[k][i] * c[k] for k in range(m)) for i in range(m)]
            sa = _dec(t.s_perp) + sum(x * x for x in c)
        else:
            Aa = [[_dec(x) for x in row] for row in t.A]
            ba = [_dec(x) for x in t.b]
            sa = _dec(t.s)
        for i in range(m):
            for j in range(m):
                Sig[a * m + i][a * m + j] += Aa[i][j]
        bvec += ba
        s_tot += sa
    Ls = _chol(Sig)
    y = _fwd(Ls, bvec)
    val = s_tot - sum(v * v for v in y) + logdet_phi + 2 * sum(Ls[i][i].ln() for i in range(N))
    if include_const:
        val += sum(D(repr(t.const())) for t in terms)
    return -val / 2


def grad_fd_dec(terms, T, params: dict, which: list[tuple[str, int | None]], Gamma=None, h="1e-12", **kw):
    """Central differences of ``loglike_dec`` (50 digits) for selected parameters."""
    hd = D(h)
    out = {}
    for key, idx in which:
        vals = []
        for sgn in (1, -1):
            p = dict(params)
            if idx is None:
                p[key] = _dec(params[key]) + sgn * hd
            else:
                arr = [_dec(x) for x in np.atleast_1d(params[key])]
                arr[idx] += sgn * hd
                p[key] = arr
            vals.append(loglike_dec(terms, T, p["rn_log10_A"], p["rn_gamma"], p["log10_A"], p["gamma"], Gamma, **kw))
        out[(key, idx)] = float((vals[0] - vals[1]) / (2 * hd))
    return out
