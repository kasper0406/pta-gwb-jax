"""Exactness of the hybrid kernel (``ptagwb.hybrid``): block-MH independence jumps leave the target
invariant (1-D peak + shelf, 2-D amplitude/slope-like block), a naive acceptance without the
proposal ratio is detected by the same test, and NUTS + jumps through the production driver
reproduces a small Gaussian-process (free-spectrum-like) posterior computed on a grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from ptagwb.diagnostics import mcse_quantile
from ptagwb.hybrid import BlockProposals, fit_proposals, log_q1, make_sweep, sample_q1
from ptagwb.sampling import BoxTransform, Posterior

LO, HI = -15.5, -1.0


def _peak_shelf_logpdf(x):
    """1-D target: 95% narrow peak at -7, 5% flat shelf below -10 (unnormalised log density)."""
    peak = jnp.log(0.95) - 0.5 * ((x + 7.0) / 0.1) ** 2 - jnp.log(0.1 * jnp.sqrt(2 * jnp.pi))
    shelf = jnp.where(x < -10.0, jnp.log(0.05 / (-10.0 - LO)), -jnp.inf)
    return jnp.logaddexp(peak, shelf)


def _grid_sampler(logpdf, lo, hi, n=400_001):
    g = np.linspace(lo, hi, n)
    p = np.exp(np.asarray(jax.vmap(logpdf)(jnp.asarray(g))))
    c = np.cumsum(p)
    c /= c[-1]
    return g, c


def _exact_draws(g, c, size, rng):
    return np.interp(rng.uniform(size=size), c, g)


def _ks_p(x, g, c):
    return stats.kstest(x, lambda t: np.interp(t, g, c)).pvalue


@pytest.fixture(scope="module")
def one_d():
    tr = BoxTransform(("r",), np.array([LO]), np.array([HI]))
    post = Posterior.generic(tr, lambda x: _peak_shelf_logpdf(x[0]))
    g, c = _grid_sampler(_peak_shelf_logpdf, LO, HI)
    rng = np.random.default_rng(0)
    # deliberately imperfect proposal: fitted to shifted peak-only draws (+ the prior component)
    fit = rng.normal(-6.8, 0.2, (5000, 1))
    prop = fit_proposals(fit, ["r"], tr.lo, tr.hi, ["r"], [], w_prior=0.2, k1=20)
    return tr, post, prop, g, c


def _apply_sweeps(tr, post, prop, x0, n_sweeps, seed=1):
    sweep = make_sweep(prop, tr, post.logL_x_raw, sweeps=n_sweeps)
    z0 = jnp.asarray(tr.to_unconstrained(x0))
    ll0 = jax.vmap(post.logL_x_raw)(tr.to_constrained(z0))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(x0))
    z, ll, acc = jax.jit(jax.vmap(sweep))(keys, z0, ll0)
    x = np.asarray(tr.to_constrained(z))
    np.testing.assert_allclose(np.asarray(ll), np.asarray(jax.vmap(post.logL_x_raw)(jnp.asarray(x))), atol=1e-8)
    return x, np.asarray(acc)


def test_proposal_density_normalised_and_sampler_consistent(one_d):
    tr, _, prop, _, _ = one_d
    e, lo, hi, w = jnp.asarray(prop.edges1[0]), float(prop.lo1[0]), float(prop.hi1[0]), prop.w_prior
    g = np.linspace(lo, hi, 2_000_001)
    q = np.exp(np.asarray(jax.vmap(lambda t: log_q1(t, e, lo, hi, w))(jnp.asarray(g))))
    assert abs(np.trapezoid(q, g) - 1.0) < 2e-3
    xs = np.asarray(jax.vmap(lambda k: sample_q1(k, e, lo, hi, w))(jax.random.split(jax.random.PRNGKey(3), 50_000)))
    assert np.all((xs > lo) & (xs < hi))
    cq = np.cumsum(q)
    cq /= cq[-1]
    assert _ks_p(xs, g, cq) > 1e-3


def test_block_mh_invariance_1d(one_d):
    """Exact draws in, one and five sweeps applied: still exact (KS) and the shelf mass is kept."""
    tr, post, prop, g, c = one_d
    rng = np.random.default_rng(1)
    x0 = _exact_draws(g, c, 40_000, rng)[:, None]
    for n_sweeps in (1, 5):
        x, acc = _apply_sweeps(tr, post, prop, x0, n_sweeps)
        assert 0.05 < acc.mean() / n_sweeps < 0.95  # the kernel actually moves
        crossed = np.mean((x0[:, 0] < -10) != (x[:, 0] < -10))
        assert crossed > 0.005  # and moves between the regions
        assert _ks_p(x[:, 0], g, c) > 1e-3, n_sweeps
        p_shelf = np.mean(x[:, 0] < -10)
        assert abs(p_shelf - 0.05) < 5 * np.sqrt(0.05 * 0.95 / len(x))


def test_naive_acceptance_is_detected(one_d):
    """Power check: dropping the proposal ratio q(x)/q(x') (a common bug) visibly breaks
    invariance under the same test."""
    tr, post, prop, g, c = one_d
    e, lo, hi, w = jnp.asarray(prop.edges1[0]), float(prop.lo1[0]), float(prop.hi1[0]), prop.w_prior

    def naive(key, x):
        k1, k2 = jax.random.split(key)
        xn = sample_q1(k1, e, lo, hi, w)
        la = _peak_shelf_logpdf(xn) - _peak_shelf_logpdf(x)
        return jnp.where(jnp.log(jax.random.uniform(k2)) < la, xn, x)

    x0 = _exact_draws(g, c, 40_000, np.random.default_rng(2))
    keys = jax.random.split(jax.random.PRNGKey(5), len(x0))
    x = np.asarray(jax.jit(jax.vmap(naive))(keys, jnp.asarray(x0)))
    assert _ks_p(x, g, c) < 1e-6


def _ab_logpdf(v):
    """2-D (log10_A, gamma)-like target on [-20,-11] x [0,7]: 90% correlated peak, 10% low-amplitude
    shelf with gamma unconstrained."""
    a, gm = v[0], v[1]
    u = (a + 13.5) / 0.15
    w = (gm - 3.0 - 2.0 * (a + 13.5)) / 0.3
    peak = jnp.log(0.9) - 0.5 * (u**2 + w**2) - jnp.log(2 * jnp.pi * 0.15 * 0.3)
    shelf = jnp.where(a < -15.0, jnp.log(0.1 / (5.0 * 7.0)), -jnp.inf)
    return jnp.logaddexp(peak, shelf)


def test_block_mh_invariance_2d():
    lo, hi = np.array([-20.0, 0.0]), np.array([-11.0, 7.0])
    tr = BoxTransform(("A", "g"), lo, hi)
    post = Posterior.generic(tr, _ab_logpdf)
    # exact draws from a fine 2-D grid (cell-uniform jitter)
    na, ng = 1800, 1400
    ga, gg = np.linspace(lo[0], hi[0], na + 1), np.linspace(lo[1], hi[1], ng + 1)
    ca, cg = 0.5 * (ga[1:] + ga[:-1]), 0.5 * (gg[1:] + gg[:-1])
    A, G = np.meshgrid(ca, cg, indexing="ij")
    P = np.exp(np.asarray(jax.vmap(_ab_logpdf)(jnp.asarray(np.stack([A.ravel(), G.ravel()], 1)))))
    P /= P.sum()
    rng = np.random.default_rng(3)
    cells = rng.choice(P.size, size=30_000, p=P)
    x0 = np.stack([A.ravel()[cells], G.ravel()[cells]], 1)
    x0 += rng.uniform(-0.5, 0.5, x0.shape) * np.array([ga[1] - ga[0], gg[1] - gg[0]])
    # imperfect proposal fitted from a biased sample of the peak only
    fit = np.stack([rng.normal(-13.3, 0.3, 4000), rng.normal(3.5, 0.8, 4000)], 1)
    prop = fit_proposals(fit, ["A", "g"], lo, hi, [], [("A", "g")], w_prior=0.2, k2a=8, k2b=6)
    x, acc = _apply_sweeps(tr, post, prop, x0, 3, seed=4)
    assert 0.05 < acc.mean() / 3 < 0.95
    assert np.mean((x0[:, 0] < -15) != (x[:, 0] < -15)) > 0.01
    # marginal CDFs from the grid
    pa, pg = P.reshape(na, ng).sum(1), P.reshape(na, ng).sum(0)
    for j, (gr, pm) in enumerate(((ga, pa), (gg, pg))):
        cdf = np.concatenate([[0.0], np.cumsum(pm)])
        assert stats.kstest(x[:, j], lambda t, gr=gr, cdf=cdf: np.interp(t, gr, cdf)).pvalue > 1e-3, j
    m = x[:, 0] < -15
    assert abs(m.mean() - 0.1) < 5 * np.sqrt(0.09 / len(x))
    # a property of the joint, not only marginals: gamma | peak region
    pk = x[:, 0] > -14.5
    assert abs(np.corrcoef(x[pk, 0], x[pk, 1])[0, 1] - np.corrcoef(x0[x0[:, 0] > -14.5].T)[0, 1]) < 0.03


def test_proposals_json_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    lo, hi = -np.full(3, 10.0), np.full(3, 10.0)
    p = fit_proposals(X, ["a", "b", "c"], lo, hi, ["a"], [("b", "c")])
    p.to_json(tmp_path / "p.json")
    q = BlockProposals.from_json(tmp_path / "p.json", ["a", "b", "c"])
    for k in ("idx1", "edges1", "idx2", "edges2a", "edges2b", "lo2", "hi2"):
        np.testing.assert_array_equal(getattr(p, k), getattr(q, k))
    with pytest.raises(ValueError, match="different parameter vector"):
        BlockProposals.from_json(tmp_path / "p.json", ["a", "b", "x"])


# ---------------------------------------------------------------------- full hybrid NUTS on a GP toy


def _gp_toy():
    """y ~ N(0, I + sum_k 10^(2 rho_k) F_k F_k^T), two 'frequency bins' (sin/cos columns), uniform
    priors rho_k in [-3, 1]: bin 0 detected, bin 1 marginal (peak + low-power shelf)."""
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 50)
    F = [np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], 1) for f in (1.0, 2.0)]
    rho_true = (0.3, -0.8)
    C = np.eye(len(t)) + sum(10 ** (2 * r) * Fk @ Fk.T for r, Fk in zip(rho_true, F))
    y = np.linalg.cholesky(C) @ rng.standard_normal(len(t))
    Fj, yj = [jnp.asarray(f) for f in F], jnp.asarray(y)

    def logL(x):
        C = jnp.eye(len(t)) + sum(10 ** (2 * x[k]) * Fj[k] @ Fj[k].T for k in range(2))
        L = jnp.linalg.cholesky(C)
        a = jax.scipy.linalg.solve_triangular(L, yj, lower=True)
        return -0.5 * a @ a - jnp.sum(jnp.log(jnp.diagonal(L)))

    return logL


def test_hybrid_nuts_gp_toy(tmp_path, monkeypatch):
    from ptagwb import sampling
    from ptagwb.sampling import RunConfig, load_run, run_nuts

    logL = _gp_toy()
    lo, hi = np.array([-3.0, -3.0]), np.array([1.0, 1.0])
    tr = BoxTransform(("gw_log10_rho_0", "gw_log10_rho_1"), lo, hi)
    post = Posterior.generic(tr, logL)
    # exact marginals on a grid
    n = 600
    g = np.linspace(-3, 1, n + 1)
    c = 0.5 * (g[1:] + g[:-1])
    R0, R1 = np.meshgrid(c, c, indexing="ij")
    lp = np.asarray(jax.jit(jax.vmap(logL))(jnp.asarray(np.stack([R0.ravel(), R1.ravel()], 1)))).reshape(n, n)
    P = np.exp(lp - lp.max())
    P /= P.sum()
    p1 = P.sum(0)
    p_low = float(p1[c < -1.5].sum())
    assert 0.05 < p_low < 0.95  # the toy really has a low-power region of bin 1
    # proposals fitted from a crude (biased) pilot: draws near the peak only
    rng = np.random.default_rng(0)
    fit = np.stack([rng.normal(0.3, 0.2, 3000), rng.normal(-0.5, 0.2, 3000)], 1)
    prop = fit_proposals(fit, list(tr.names), lo, hi, list(tr.names), [], w_prior=0.2, k1=15)
    monkeypatch.setattr(sampling, "RUNS_DIR", tmp_path)
    prop.to_json(tmp_path / "prop.json")
    cfg = RunConfig(name="gp", model={}, num_chains=4, num_warmup=300, num_samples=1500, block=750, dense_mass=False,
                    seed=2, init_radius=2.0, jumps=str(tmp_path / "prop.json"))
    run_nuts(cfg, post, log=lambda s: None)
    r = load_run("gp")
    x = r["x"]
    assert r["jump_accept"].shape == (4, 1500, 2) and r["jump_accept"].sum() > 100
    for j, pm in enumerate((P.sum(1), p1)):
        cdf = np.cumsum(pm)
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            q_ref = np.interp(p, cdf, c)
            q, se = np.quantile(x[..., j], p), mcse_quantile(x[..., j], p)
            assert abs(q - q_ref) < 5 * se + 0.01, (j, p, q, q_ref, se)
    ind = (x[..., 1] < -1.5).astype(float)
    from ptagwb.diagnostics import ess

    se = np.sqrt(p_low * (1 - p_low) / ess(ind))
    assert abs(ind.mean() - p_low) < 5 * se, (ind.mean(), p_low, se)
