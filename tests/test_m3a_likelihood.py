"""M3a general likelihood (``ptagwb.combined``) against the long-double dense oracle (gate G5,
"our dense oracle"): unequal grids, a DM block, a fixed band block, overlapping ECORR, TN and T2
EQUAD legs, staggered multi-leg spans, padding (pulsars with different K), CURN and HD, power-law
and free-spectrum common processes. Tolerances (fixed in advance, docs/M3A_VALIDATION.md):
value |ours - dense| <= 1e-9 max(1, |lnL|) on the absolute value (same constant convention);
gradient |ours - fd(dense)| <= 1e-8 max(1, |g|) per component.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dense_oracle import dense_loglike, fd_gradient
from synthetic_m3 import make_array, random_params

from ptagwb import orf as orfs
from ptagwb.combined import ColumnLayout, GeneralPTALikelihood, PulsarGPModel, precompute_general
from ptagwb.gp import FourierBlock
from ptagwb.noise import general_from_white_noise

VAL_TOL = 1e-9
GRAD_TOL = 1e-8


def _jp(p):
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p.items()}


@pytest.fixture(scope="module")
def arr():
    psrs, wns, models, _T = make_array(seed=1)
    terms = [precompute_general(p, w, m) for p, w, m in zip(psrs, wns, models, strict=True)]
    return psrs, wns, models, terms


def test_layout_distinct_vs_prefix():
    blk = FourierBlock("red_noise", 30, 5e8)
    lay = ColumnLayout(PulsarGPModel(sampled={"rn": blk}, common=FourierBlock("gw", 14, 5e8)))
    assert lay.K == 60 and np.array_equal(lay.common_cols, np.arange(28))  # M1 prefix identity
    lay2 = ColumnLayout(PulsarGPModel(sampled={"rn": blk}, common=FourierBlock("gw", 14, 5.2e8)))
    assert lay2.K == 88 and np.array_equal(lay2.common_cols, np.arange(60, 88))  # distinct grids
    lay3 = ColumnLayout(PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 10, 5e8)}, common=FourierBlock("gw", 14, 5e8)))
    assert lay3.K == 28 and np.array_equal(lay3.common_cols, np.arange(28))  # common longer than IRN


@pytest.mark.parametrize("orf_name", ["curn", "hd"])
@pytest.mark.parametrize("common", ["powerlaw", "freespec"])
def test_value_vs_dense(arr, orf_name, common):
    psrs, wns, models, terms = arr
    G = None if orf_name == "curn" else orfs.hd(np.stack([p.pos for p in psrs]))
    like = GeneralPTALikelihood(terms, orf=orf_name, common=common)
    rng = np.random.default_rng(5)
    for _ in range(3):
        p = random_params(rng, models, common)
        ours = float(like.logL(_jp(p)))
        ref = dense_loglike(psrs, wns, models, p, Gamma=G, common=common)
        assert abs(ours - ref) <= VAL_TOL * max(1.0, abs(ref)), (ours, ref, ours - ref)


@pytest.mark.parametrize("orf_name", ["curn", "hd"])
def test_gradient_vs_dense(arr, orf_name):
    psrs, wns, models, terms = arr
    G = None if orf_name == "curn" else orfs.hd(np.stack([p.pos for p in psrs]))
    like = GeneralPTALikelihood(terms, orf=orf_name)
    p = random_params(np.random.default_rng(11), models)
    g = jax.grad(like._logL)(_jp(p))
    gd = fd_gradient(lambda q: dense_loglike(psrs, wns, models, q, Gamma=G), p)
    for k in p:
        a, b = np.atleast_1d(np.asarray(g[k])), np.atleast_1d(gd[k])
        assert np.all(np.abs(a - b) <= GRAD_TOL * np.maximum(1.0, np.abs(b))), (k, a, b)


@pytest.mark.parametrize("reduce,tri_inv", [("hh", "levels"), ("prod", "levels")])
def test_fast_variants_agree(arr, reduce, tri_inv):
    _, _, models, terms = arr
    ref = GeneralPTALikelihood(terms, orf="hd")
    fast = GeneralPTALikelihood(terms, orf="hd", reduce=reduce, tri_inv=tri_inv)
    for s in range(3):
        p = _jp(random_params(np.random.default_rng(s), models))
        (v1, g1), (v2, g2) = ref.value_and_grad(p), fast.value_and_grad(p)
        assert abs(float(v1) - float(v2)) <= 1e-9 * max(1, abs(float(v1)))
        for k in g1:
            np.testing.assert_allclose(np.asarray(g2[k]), np.asarray(g1[k]), rtol=1e-8, atol=1e-8)


def test_padding_is_exact(arr):
    """Pulsar B (fewer columns) alone vs padded inside the array, noise-only likelihood."""
    _, _, models, terms = arr
    both = GeneralPTALikelihood(terms, orf=None, common=None)
    alone = [GeneralPTALikelihood([t], orf=None, common=None) for t in terms]
    assert terms[1].K < terms[0].K
    p = random_params(np.random.default_rng(3), models)
    p.pop("log10_A"), p.pop("gamma")
    pa = {"rn_log10_A": p["rn_log10_A"][:1], "rn_gamma": p["rn_gamma"][:1], "dm_log10_A": p["dm_log10_A"], "dm_gamma": p["dm_gamma"]}
    pb = {"rn_log10_A": p["rn_log10_A"][1:], "rn_gamma": p["rn_gamma"][1:]}
    v = float(both.logL(_jp(p)))
    va, vb = float(alone[0].logL(_jp(pa))), float(alone[1].logL(_jp(pb)))
    assert abs(v - (va + vb)) <= 1e-10 * abs(v)
    g = jax.grad(both._logL)(_jp(p))
    gb = jax.grad(alone[1]._logL)(_jp(pb))
    np.testing.assert_allclose(np.asarray(g["rn_gamma"])[1], np.asarray(gb["rn_gamma"])[0], rtol=1e-11)


def test_general_white_noise_matches_disjoint():
    """GeneralWhiteNoise (Cholesky per component) == WhiteNoise (M1 symmetric whitening)."""
    from synthetic import make_pta

    from ptagwb.noise import build_white_noise

    psrs, nd = make_pta(3, seed=4)
    rng = np.random.default_rng(0)
    for p in psrs:
        w1 = build_white_noise(p, nd)
        w2 = general_from_white_noise(w1)
        np.testing.assert_allclose(w2.dense(), w1.dense(), rtol=0, atol=0)
        X = rng.normal(size=(p.ntoa, 3))
        W1, W2 = w1.whiten(X), w2.whiten(X)
        np.testing.assert_allclose(W1.T @ W1, W2.T @ W2, rtol=1e-10)
        assert abs(w1.logdet() - w2.logdet()) <= 1e-12 * abs(w1.logdet())
        np.testing.assert_allclose(w2.solve(X), w1.solve(X), rtol=1e-10)


def test_overlapping_ecorr_dense_ops(arr):
    _psrs, wns, _, _ = arr
    w = wns[0]
    assert w.max_component > 1 and len(set(w.ep_label.tolist())) > 2
    N = w.dense()
    X = np.random.default_rng(2).normal(size=(len(N), 4))
    np.testing.assert_allclose(w.solve(X), np.linalg.solve(N, X), rtol=1e-9, atol=0)
    Y = w.whiten(np.eye(len(N)))
    np.testing.assert_allclose(Y.T @ Y, np.linalg.inv(N), rtol=1e-8, atol=1e-8 * np.abs(np.linalg.inv(N)).max())
    assert abs(w.logdet() - np.linalg.slogdet(N)[1]) <= 1e-11 * abs(w.logdet())


def test_tn_vs_t2_equad():
    from ptagwb.noise import white_variance

    s = np.array([1e-6, 2e-6])
    np.testing.assert_allclose(white_variance(s, 1.3, -6.5, "t2"), 1.3**2 * (s**2 + 10**-13))
    np.testing.assert_allclose(white_variance(s, 1.3, -6.5, "tn"), 1.3**2 * s**2 + 10**-13)
    # Q_TN = EFAC * Q_T2 gives the same covariance
    np.testing.assert_allclose(white_variance(s, 1.3, -6.5 + np.log10(1.3), "tn"), white_variance(s, 1.3, -6.5, "t2"))
    with pytest.raises(ValueError):
        white_variance(s, 1.0, -6.0, "x")


def test_namespacing_separates_ptas(arr):
    psrs, wns, _, _ = arr
    sys_ = set(psrs[0].backend_flags.tolist())
    assert {"X:BE", "Y:BE"} <= sys_  # equal raw labels, different parameters
    assert set(wns[0].systems) == sys_


def test_close_spans_conditioning():
    """IRN on a 19.1-yr grid and common on a 20.1-yr grid (nearly collinear blocks): the
    square-root path stays accurate (same oracle tolerance)."""
    rng = np.random.default_rng(7)
    psrs, wns, models, Tarr = make_array(seed=2, distinct=False, fixed=False)
    for m in models:
        m.sampled["rn"] = FourierBlock("red_noise", 5, Tarr * 19.1 / 20.1)
    terms = [precompute_general(p, w, m) for p, w, m in zip(psrs, wns, models, strict=True)]
    like = GeneralPTALikelihood(terms, orf="hd")
    G = orfs.hd(np.stack([p.pos for p in psrs]))
    for _ in range(2):
        p = random_params(rng, models)
        ours, ref = float(like.logL(_jp(p))), dense_loglike(psrs, wns, models, p, Gamma=G)
        assert abs(ours - ref) <= VAL_TOL * max(1.0, abs(ref)), (ours, ref)


def test_projector_reproduces_stage1(arr):
    psrs, wns, models, _ = arr
    t = precompute_general(psrs[0], wns[0], models[0], projector=True)
    c, s = t.projector(psrs[0].residuals)
    np.testing.assert_allclose(c, t.c, rtol=1e-12, atol=1e-14 * np.abs(t.c).max())
    assert abs(s - t.s_perp) <= 1e-10 * t.s_perp
