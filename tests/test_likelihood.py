"""Reduced (Woodbury / coefficient-space) likelihood vs brute-force dense covariance."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthetic import dense_loglike, make_pta, tspan

from ptagwb import orf as orfs
from ptagwb.likelihood import PTALikelihood, precompute

NM, NC = 6, 3


@pytest.fixture(scope="module")
def pta():
    psrs, nd = make_pta(5, seed=0, signal=3e-7)
    T = tspan(psrs)
    return psrs, nd, T, precompute(psrs, nd, T, n_modes=NM)


def _params(P, seed, common="powerlaw"):
    rng = np.random.default_rng(seed)
    p = {"rn_log10_A": rng.uniform(-15, -12.5, P), "rn_gamma": rng.uniform(1, 6, P)}
    if common == "powerlaw":
        p.update(log10_A=rng.uniform(-15, -13), gamma=rng.uniform(2, 6))
    else:
        p["log10_rho"] = rng.uniform(-8, -6, NC)
    return p


def _orf(name, pos):
    return None if name == "curn" else orfs.ORFS[name](pos)


@pytest.mark.parametrize("orf_name", ["curn", "hd", "monopole", "dipole"])
@pytest.mark.parametrize("common", ["powerlaw", "freespec"])
def test_reduced_vs_dense(pta, orf_name, common):
    psrs, nd, T, terms = pta
    pos = np.stack([p.pos for p in psrs])
    methods = ["sigma"] if orf_name == "curn" else ["sigma", "B"]
    for method in methods:
        L = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf=orf_name, common=common, method=method)
        for seed in range(3):
            p = _params(len(psrs), seed, common)
            ours = float(L.logL({k: jnp.asarray(v) for k, v in p.items()}))
            ref = dense_loglike(psrs, nd, T, NM, NC, p["rn_log10_A"], p["rn_gamma"], _orf(orf_name, pos),
                                log10_A=p.get("log10_A"), gamma=p.get("gamma"), log10_rho=p.get("log10_rho"))
            assert abs(ours - ref) <= 1e-8 * abs(ref), (orf_name, common, method, seed, ours, ref, ours - ref)


def test_identity_orf_recovers_curn(pta):
    psrs, _nd, T, terms = pta
    curn = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf="curn")
    for method in ("sigma", "B"):
        eye = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf=np.eye(len(psrs)), method=method)
        for seed in range(5):
            p = {k: jnp.asarray(v) for k, v in _params(len(psrs), seed).items()}
            assert abs(float(curn.logL(p)) - float(eye.logL(p))) < 1e-9 * abs(float(curn.logL(p)))
            g1, g2 = jax.grad(curn._logL)(p), jax.grad(eye._logL)(p)
            for k in g1:
                np.testing.assert_allclose(g2[k], g1[k], rtol=1e-7, atol=1e-9)


def test_nmin1_changes_only_singletons(pta):
    psrs, nd, T, _ = pta
    t1 = precompute(psrs, nd, T, n_modes=NM, nmin=1)
    p = _params(len(psrs), 0)
    L1 = PTALikelihood(t1, T, n_modes=NM, n_common=NC, orf="hd")
    ref = dense_loglike(psrs, nd, T, NM, NC, p["rn_log10_A"], p["rn_gamma"], orfs.hd(np.stack([q.pos for q in psrs])),
                        log10_A=p["log10_A"], gamma=p["gamma"], nmin=1)
    assert abs(float(L1.logL({k: jnp.asarray(v) for k, v in p.items()})) - ref) <= 1e-8 * abs(ref)


@pytest.mark.parametrize("orf_name", ["curn", "hd"])
@pytest.mark.parametrize("common", ["powerlaw", "freespec"])
def test_gradients_vs_finite_differences(pta, orf_name, common):
    psrs, _nd, T, terms = pta
    L = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf=orf_name, common=common)
    p0 = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in _params(len(psrs), 7, common).items()}
    g = jax.grad(L._logL)(p0)
    f = lambda p: float(L.logL(p))
    h = 1e-5
    for k, v in p0.items():
        flat = np.atleast_1d(np.asarray(v))
        for i in range(flat.size):
            e = np.zeros_like(flat)
            e[i] = h
            pp = dict(p0, **{k: jnp.asarray((flat + e).reshape(np.shape(v)))})
            pm = dict(p0, **{k: jnp.asarray((flat - e).reshape(np.shape(v)))})
            fd = (f(pp) - f(pm)) / (2 * h)
            an = float(np.atleast_1d(np.asarray(g[k]))[i])
            assert abs(fd - an) <= 1e-5 * max(1.0, abs(an)), (k, i, fd, an)


def test_mixed_precision_gradient_close(pta):
    psrs, _nd, T, terms = pta
    ex = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf="hd")
    mx = PTALikelihood(terms, T, n_modes=NM, n_common=NC, orf="hd", grad_precision="mixed")
    p = {k: jnp.asarray(v) for k, v in _params(len(psrs), 3).items()}
    (v1, g1), (v2, g2) = ex.value_and_grad(p), mx.value_and_grad(p)
    assert float(v1) == float(v2)
    scale = max(float(jnp.max(jnp.abs(g1[k]))) for k in g1)
    for k in g1:
        np.testing.assert_allclose(g2[k], g1[k], atol=1e-3 * scale, rtol=1e-3)


def test_convention_constant(pta):
    psrs, _nd, T, terms = pta
    a = PTALikelihood(terms, T, n_modes=NM, n_common=NC, convention="chain")
    b = PTALikelihood(terms, T, n_modes=NM, n_common=NC, convention="enterprise")
    p = {k: jnp.asarray(v) for k, v in _params(len(psrs), 0).items()}
    ntoa = sum(p_.ntoa for p_ in psrs)
    np.testing.assert_allclose(float(a.logL(p)) - float(b.logL(p)), 0.5 * ntoa * np.log(2 * np.pi), rtol=1e-12)
