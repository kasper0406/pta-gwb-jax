"""Optimal statistic: coefficient-space implementation vs brute-force TOA-space reference,
the release's loop implementation of the pair covariance, and binned estimator identities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from synthetic import make_pta, tspan

from ptagwb.basis import fourier_basis, powerlaw
from ptagwb.likelihood import PTALikelihood, precompute, timing_basis
from ptagwb.noise import build_white_noise
from ptagwb.optstat import OptimalStatistic, binned_correlations, hd_curve

NM, NC = 6, 3


@pytest.fixture(scope="module")
def pta():
    psrs, nd = make_pta(6, seed=3, signal=4e-7)
    T = tspan(psrs)
    terms = precompute(psrs, nd, T, n_modes=NM)
    like = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf="curn")
    return psrs, nd, T, like


def _params(P, seed):
    rng = np.random.default_rng(seed)
    return {
        "rn_log10_A": rng.uniform(-15, -13, P),
        "rn_gamma": rng.uniform(1, 6, P),
        "log10_A": rng.uniform(-14.5, -13.5),
        "gamma": rng.uniform(2, 5),
    }


def _dense_components(psrs, nd, T, p):
    """X_a = F_c^T P_a r_a, Z_a = F_c^T P_a F_c with the timing-projected inverse covariance."""
    Xs, Zs = [], []
    nc2 = 2 * NC
    for a, ps in enumerate(psrs):
        N = build_white_noise(ps, nd).dense()
        f, df, F = fourier_basis(ps.toas, NM, T)
        phi = powerlaw(f, df, p["rn_log10_A"][a], p["rn_gamma"][a])
        phi[:nc2] += powerlaw(f[:nc2], df[:nc2], p["log10_A"], p["gamma"])
        C = N + (F * phi) @ F.T
        Ci = np.linalg.inv(C)
        M = timing_basis(ps.Mmat)
        Pm = Ci - Ci @ M @ np.linalg.solve(M.T @ Ci @ M, M.T @ Ci)
        Fc = F[:, :nc2]
        Xs.append(Fc.T @ Pm @ ps.residuals)
        Zs.append(Fc.T @ Pm @ Fc)
    return np.array(Xs), np.array(Zs)


def test_components_vs_dense(pta):
    psrs, nd, T, like = pta
    os_ = OptimalStatistic(like)
    for seed in range(3):
        p = _params(len(psrs), seed)
        X, Z = (np.asarray(a) for a in os_.components({k: jnp.asarray(v) for k, v in p.items()}))
        Xr, Zr = _dense_components(psrs, nd, T, p)
        assert np.allclose(X, Xr, rtol=1e-8, atol=1e-10 * np.abs(Xr).max())
        assert np.allclose(Z, Zr, rtol=1e-8, atol=1e-10 * np.abs(Zr).max())


def _loop_os(X, Z, ph, pos):
    """Straight port of enterprise_extensions OptimalStatistic.compute_os (pair loop)."""
    rho, sig, orf = [], [], []
    P = len(X)
    for i in range(P):
        for j in range(i + 1, P):
            top = X[i] @ (ph * X[j])
            bot = np.trace((Z[i] * ph[None, :]) @ (Z[j] * ph[None, :]))
            rho.append(top / bot)
            sig.append(1 / np.sqrt(bot))
            orf.append(hd_curve(np.arccos(np.dot(pos[i], pos[j]))))
    rho, sig, orf = map(np.array, (rho, sig, orf))
    return rho, sig, np.sum(rho * orf / sig**2) / np.sum(orf**2 / sig**2), 1 / np.sqrt(np.sum(orf**2 / sig**2))


def test_os_vs_loop(pta):
    psrs, _nd, _T, like = pta
    os_ = OptimalStatistic(like)
    p = _params(len(psrs), 7)
    pj = {k: jnp.asarray(v) for k, v in p.items()}
    X, Z = (np.asarray(a) for a in os_.components(pj))
    ph = np.asarray(os_.phihat(p["gamma"]))
    rho_r, sig_r, a2_r, s_r = _loop_os(X, Z, ph, like.pos)
    res = os_.os(pj)
    assert np.allclose(res["rho"], rho_r, rtol=1e-12)
    assert np.allclose(res["sig"], sig_r, rtol=1e-12)
    assert np.isclose(res["A2"], a2_r, rtol=1e-12) and np.isclose(res["sigma"], s_r, rtol=1e-12)
    # batched noise-marginalised path agrees with the single-point path
    batch = {k: np.repeat(np.atleast_1d(v)[None], 3, axis=0).squeeze(-1) if np.ndim(v) == 0 else np.repeat(v[None], 3, axis=0) for k, v in p.items()}
    nm = os_.noise_marginalized(batch, batch=2)
    assert np.allclose(nm["A2"], res["A2"], rtol=1e-10) and np.allclose(nm["snr"], res["snr"], rtol=1e-10)


def _release_gw_corr(Z, ph, A, pos, sigmas):
    """The NG15 release's ``OS.gw_corr`` loop (data_release/figure_1/optimal_statistic_covariances.py),
    with its effective ``_tracedot`` (the second definition, which ignores the ORF)."""
    M = [Zi * ph[None, :] for Zi in Z]
    P = len(Z)
    pairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
    orf = lambda a, b: hd_curve(np.arccos(np.dot(pos[a], pos[b])))

    def td(*idx):
        r = np.identity(len(ph))
        for i in idx:
            r = r @ M[i]
        return np.trace(r)

    S = np.zeros((len(pairs), len(pairs)))
    for ij, (i, j) in enumerate(pairs):
        for kl in range(ij, len(pairs)):
            k, l = pairs[kl]
            if ij == kl:
                v = td(i, j) + A**4 * td(i, j, i, j) * orf(i, j) ** 2
            elif i == k and j != l:
                v = A**2 * td(i, j, l) * orf(j, l) + A**4 * td(i, j, i, l) * orf(i, l) * orf(i, j)
            elif i != k and j == l:
                v = A**2 * td(j, k, i) * orf(i, k) + A**4 * td(i, j, k, j) * orf(i, j) * orf(k, j)
            elif i != k and j != l and i != l and j != k:
                v = A**4 * orf(i, l) * orf(k, j) * td(i, j, k, l) + A**4 * orf(i, k) * orf(j, l) * td(i, j, l, k)
            elif j == k:
                v = A**4 * orf(i, j) * orf(j, l) * td(j, i, j, l) + A**2 * orf(i, l) * td(j, i, l)
            else:
                v = 0.0
            S[ij, kl] = S[kl, ij] = v * sigmas[ij] ** 2 * sigmas[kl] ** 2
    return S


def test_pair_covariance_vs_release_loop(pta):
    psrs, _nd, _T, like = pta
    os_ = OptimalStatistic(like)
    p = _params(len(psrs), 11)
    pj = {k: jnp.asarray(v) for k, v in p.items()}
    _, Z = (np.asarray(a) for a in os_.components(pj))
    ph = np.asarray(os_.phihat(p["gamma"]))
    _, sig = os_.rho_sigma(pj)
    ref = _release_gw_corr(Z, ph, 10 ** p["log10_A"], like.pos, sig)
    ours = os_.pair_covariance(pj)
    assert np.allclose(ours, ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())


def test_binned_single_pair_bins_and_uncorrelated_limit():
    rng = np.random.default_rng(0)
    n = 40
    xi = np.sort(rng.uniform(0.05, 3.1, n))
    u = hd_curve(xi)
    rho = u * 2.0 + rng.normal(size=n) * 0.1
    sig = rng.uniform(0.05, 0.2, n)
    C = np.diag(sig**2)
    edges = np.array([0.0, 1.0, 2.0, np.pi])
    out = binned_correlations(xi, rho, u, C, edges, a2_norm=2.0)
    # diagonal covariance -> weighted least squares of rho = A^2 u within each bin
    for b in range(3):
        m = (xi >= edges[b]) & (xi < edges[b + 1])
        a2 = np.sum(rho[m] * u[m] / sig[m] ** 2) / np.sum(u[m] ** 2 / sig[m] ** 2)
        assert np.isclose(out["rho_bin"][b], hd_curve(out["xi_mean"][b]) * a2)
    assert np.allclose(out["cov"], np.diag(np.diag(out["cov"])))
    assert out["chi2"] > 0
