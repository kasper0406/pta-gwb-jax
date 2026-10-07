"""Performance variants (``ptagwb.perf_likelihood``, docs/PERF.md) must reproduce the production
likelihood: value (without the parameter-independent constant) to <= 1e-9 absolute and every
gradient component to <= 1e-8 relative (to max(|g|, 1)), single and vmapped."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthetic import make_pta, tspan

from ptagwb import orf as orfs
from ptagwb.likelihood import PTALikelihood, _tri_inv_lower, precompute
from ptagwb.perf_likelihood import FastPTALikelihood, _hh_bottom, _tri_inv_levels

NM, NC = 10, 5


def _strip(like):
    like.const_total = 0.0
    like.value_and_grad = jax.jit(jax.value_and_grad(like._logL))
    return like


def _flat(g):
    return np.concatenate([np.atleast_1d(np.asarray(g[k])) for k in sorted(g)])


def _check(prod, fast, pts, tol_v=1e-9, tol_g=1e-8):
    batched = jax.jit(jax.vmap(jax.value_and_grad(fast._logL)))
    vb, gb = batched({k: jnp.stack([jnp.asarray(p[k], dtype=jnp.float64) for p in pts]) for k in pts[0]})
    for i, p in enumerate(pts):
        v0, g0 = prod.value_and_grad(p)
        for v1, g1 in (fast.value_and_grad(p), (vb[i], {k: gb[k][i] for k in gb})):
            assert abs(float(v1) - float(v0)) <= tol_v, (i, float(v1) - float(v0))
            a, b = _flat(g0), _flat(g1)
            err = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))
            assert err <= tol_g, (i, err)


def test_hh_bottom_equals_qr():
    rng = np.random.default_rng(1)
    n, m = 12, 25
    top = np.triu(rng.standard_normal((n, n))) * np.logspace(-6, 6, n)[None, :]
    top = np.concatenate([top, rng.standard_normal((n, m - n))], axis=1)
    R = np.asarray(_hh_bottom(jnp.asarray(top)))
    R0 = np.linalg.qr(np.concatenate([top, np.concatenate([np.eye(n), np.zeros((n, m - n))], 1)]), mode="r")[:n]
    s = np.sign(np.diag(R)) * np.sign(np.diag(R0))
    np.testing.assert_allclose(R * s[:, None], R0, rtol=1e-10, atol=1e-12 * np.abs(R0).max())


@pytest.mark.parametrize("n", [37, 64, 130, 300])
def test_tri_inv_levels(n):
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n))
    L = jnp.linalg.cholesky(jnp.asarray(A @ A.T / n + np.eye(n)))
    for depth in (0, 3):
        X = _tri_inv_levels(L, leaf=16, depth=depth)
        np.testing.assert_allclose(np.asarray(X), np.asarray(_tri_inv_lower(L)), atol=1e-12 * float(jnp.abs(X).max()))


@pytest.fixture(scope="module")
def pta():
    psrs, nd = make_pta(5, seed=0, signal=3e-7)
    T = tspan(psrs)
    return T, precompute(psrs, nd, T, n_modes=NM)


@pytest.mark.parametrize("orf_name", ["curn", "hd"])
@pytest.mark.parametrize("variant", [("hh", "levels"), ("hh", "recursive"), ("prod", "levels")])
def test_synthetic_matches_production(pta, orf_name, variant):
    T, terms = pta
    P = len(terms)
    orf = "curn" if orf_name == "curn" else orfs.hd(np.stack([t.pos for t in terms]))
    kw = {"n_modes": NM, "n_common": NC, "orf": orf}
    prod = _strip(PTALikelihood(terms, T, **kw))
    fast = _strip(FastPTALikelihood(terms, T, reduce=variant[0], tri_inv=variant[1], **kw))
    rng = np.random.default_rng(0)
    pts = [{"rn_log10_A": rng.uniform(-15, -12.5, P), "rn_gamma": rng.uniform(1, 6, P),
            "log10_A": rng.uniform(-15, -13), "gamma": rng.uniform(2, 6)} for _ in range(4)]
    pts += [{"rn_log10_A": np.full(P, a), "rn_gamma": np.full(P, g), "log10_A": np.asarray(la), "gamma": np.asarray(gc)}
            for a, g, la, gc in itertools.product((-20.0, -11.0), (0.0, 7.0), (-18.0, -11.0), (0.0, 7.0))]
    _check(prod, fast, pts)


@pytest.mark.oracle
@pytest.mark.slow
def test_ng15_matches_production(ours, noisedict):
    """All 67 NG15 pulsars, HD and CURN, the 16 prior corners + 2 reviewer points of
    test_corners.py and random prior draws (the full table is bench/check_exact.py)."""
    from ptagwb.data import get_tspan

    T = get_tspan(ours)
    terms = precompute(ours, noisedict, T, position="enterprise")
    P = len(terms)
    rng = np.random.default_rng(3)
    corners = list(itertools.product((-20.0, -11.0), (0.0, 7.0), (-18.0, -11.0), (0.0, 7.0)))
    corners += [(-11.1, 6.9, -11.1, 6.9), (-11.0, 7.0, -11.0, 7.0)]
    pts = [{"rn_log10_A": np.full(P, a), "rn_gamma": np.full(P, g), "log10_A": np.asarray(la), "gamma": np.asarray(gc)}
           for a, g, la, gc in corners]
    pts += [{"rn_log10_A": rng.uniform(-20, -11, P), "rn_gamma": rng.uniform(0, 7, P),
             "log10_A": np.asarray(rng.uniform(-18, -11)), "gamma": np.asarray(rng.uniform(0, 7))} for _ in range(4)]
    for orf in ("hd", "curn"):
        _check(_strip(PTALikelihood(terms, T, orf=orf)), _strip(FastPTALikelihood(terms, T, orf=orf)), pts)
