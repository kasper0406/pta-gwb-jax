"""Cached one-bin conditional (``ptagwb.conditional``) vs the production HD likelihood across the
prior, at corners and at high common power (poor conditioning of S), on a synthetic PTA."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthetic import make_pta, tspan

from ptagwb.conditional import make_bin_conditional
from ptagwb.likelihood import PTALikelihood, precompute

NM, NC = 8, 8


@pytest.fixture(scope="module")
def like():
    psrs, nd = make_pta(6, seed=2, n_epochs=60, signal=3e-7)
    T = tspan(psrs)
    return PTALikelihood(precompute(psrs, nd, T, n_modes=NM), T, n_modes=NM, n_common=NC, orf="hd", common="freespec")


def _params(P, rng, corner=None):
    if corner is None:
        u = 1 / (1 + np.exp(-rng.uniform(-4, 4, 2 * P + NC)))
    else:
        u = np.array([corner[0]] * P + [corner[1]] * P + [corner[2]] * NC, float)
    return {"rn_log10_A": -20 + 9 * u[:P], "rn_gamma": 7 * u[P:2 * P], "log10_rho": -15.5 + 14.5 * u[2 * P:]}


def _check(like, p, k, grid):
    build, cond = make_bin_conditional(like, k)
    cache = build(p)
    c = np.array([float(cond(cache, g)) for g in grid])
    full = []
    for g in grid:
        q = dict(p, log10_rho=np.asarray(p["log10_rho"]).copy())
        q["log10_rho"][k] = g
        full.append(float(like.logL(q)) + 0.5 * like.const_total)
    full = np.array(full)
    return np.max(np.abs((c - c[0]) - (full - full[0]))), np.ptp(full), c, full


@pytest.mark.parametrize("k", [0, 3, NC - 1])
def test_conditional_matches_production_across_prior(like, k):
    """Points whose other common bins carry at most moderate power (log10_rho <= -6, the posterior
    regime of the real data): <= 1e-9 relative to the logL range. Anywhere in the prior, including box
    corners: <= 1e-5 absolute. With several common bins near the maximal power (log10_rho -> -1),
    B = I + L^T A L is very ill conditioned (entries ~1e12) and the dense Schur form loses precision
    (measured <= 6e-6 vs a long-double dense reference, while production stays within 2e-9). The
    conditional only builds proposals; MH acceptance uses the production likelihood, so this affects
    efficiency, not correctness."""
    rng = np.random.default_rng(k)
    grid = np.linspace(-15.49, -1.01, 25)
    for i in range(6):
        p = _params(like.P, rng)
        p["log10_rho"] = rng.uniform(-15.5, -6.0, NC)
        err, rng_l, c, full = _check(like, p, k, grid)
        assert np.all(np.isfinite(c))
        assert err <= 1e-9 * max(1.0, rng_l), (i, err, rng_l)
        assert abs(c[0] - full[0]) <= 1e-9 * max(1.0, abs(full[0]))
    pts = [_params(like.P, rng) for _ in range(6)]
    pts += [_params(like.P, rng, c) for c in itertools.product((1e-3, 1 - 1e-3), repeat=3)]
    for p in pts:
        err, rng_l, c, full = _check(like, p, k, grid)
        assert np.all(np.isfinite(c))
        assert err <= 1e-5 + 1e-9 * rng_l, err


def test_cache_independent_of_own_bin(like):
    rng = np.random.default_rng(5)
    p = _params(like.P, rng)
    build, _ = make_bin_conditional(like, 2)
    c1 = build(p)
    p2 = dict(p, log10_rho=np.asarray(p["log10_rho"]).copy())
    p2["log10_rho"][2] = -3.0
    c2 = build(p2)
    for key in c1:
        np.testing.assert_array_equal(np.asarray(c1[key]), np.asarray(c2[key]))


def test_rejects_unsupported(like):
    with pytest.raises(ValueError):
        make_bin_conditional(like, NC)
    psrs, nd = make_pta(4, seed=1, n_epochs=30)
    T = tspan(psrs)
    curn = PTALikelihood(precompute(psrs, nd, T, n_modes=NM), T, n_modes=NM, n_common=NC, orf="curn", common="freespec")
    with pytest.raises(ValueError):
        make_bin_conditional(curn, 0)


def test_jit_vmap(like):
    rng = np.random.default_rng(9)
    ps = [_params(like.P, rng) for _ in range(3)]
    batch = {k: jnp.asarray(np.stack([p[k] for p in ps])) for k in ps[0]}
    build, cond = make_bin_conditional(like, 1)
    caches = jax.jit(jax.vmap(build))(batch)
    v = jax.jit(jax.vmap(lambda c: jax.vmap(lambda r: cond(c, r))(jnp.linspace(-15, -2, 7))))(caches)
    for i, p in enumerate(ps):
        c0 = build(p)
        ref = np.array([float(cond(c0, r)) for r in np.linspace(-15, -2, 7)])
        np.testing.assert_allclose(np.asarray(v[i]), ref, rtol=0, atol=1e-8 * max(1, np.abs(ref).max()))


# ---------------------------------------------------------------------- conditional-grid move (ptagwb.hybrid)


def test_grid_density_normalised_and_sampler_consistent():
    from scipy import stats

    from ptagwb.hybrid import grid_density, grid_logpdf, grid_sample

    g = jnp.asarray(np.sort(np.r_[np.linspace(-15, -1, 9), np.linspace(-8, -6, 5), -7.0]))  # incl. a duplicate node
    lv = jnp.asarray(-0.5 * ((np.asarray(g) + 7) / 0.6) ** 2 + np.where(np.asarray(g) < -10, -3.0, 0.0))
    logm, logZ, l = grid_density(g, lv)
    xx = np.linspace(-15, -1, 400_001)
    q = np.exp(np.asarray(jax.vmap(lambda t: grid_logpdf(t, g, l, logZ))(jnp.asarray(xx))))
    assert abs(np.trapezoid(q, xx) - 1) < 1e-6
    xs = np.asarray(jax.vmap(lambda k: grid_sample(k, g, l, logm))(jax.random.split(jax.random.PRNGKey(0), 40_000)))
    cq = np.cumsum(q)
    cq /= cq[-1]
    assert stats.kstest(xs, lambda t: np.interp(t, xx, cq)).pvalue > 1e-3


def test_grid_move_leaves_conditional_invariant(like):
    """Exact draws of rho_k from its conditional (fine grid of the production likelihood, other
    parameters fixed) stay exact after the conditional-grid move, also with a deliberately coarse grid
    (proposal != target); dropping the q ratio is detected by the same test."""
    from scipy import stats

    from ptagwb.hybrid import make_grid_moves
    from ptagwb.sampling import ModelSpec, Posterior

    spec = ModelSpec(orf="hd", common="freespec", n_common=NC, n_modes=NM)
    post = Posterior(like, spec)
    k = 2
    rng = np.random.default_rng(4)
    p = _params(like.P, rng)
    p["log10_rho"] = rng.uniform(-12, -7, NC)
    x0 = np.concatenate([p["rn_log10_A"], p["rn_gamma"], p["log10_rho"]])
    j = 2 * like.P + k
    xx = np.linspace(-15.5, -1.0, 20_001)[1:-1]
    X = np.repeat(x0[None], len(xx), 0)
    X[:, j] = xx
    ll = np.asarray(jax.jit(jax.vmap(post.logL_x_raw))(jnp.asarray(X)))
    pdf = np.exp(ll - ll.max())
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    n = 20_000
    draws = np.interp(rng.uniform(size=n), cdf, xx)
    Xd = np.repeat(x0[None], n, 0)
    Xd[:, j] = draws
    lld = jax.jit(jax.vmap(post.logL_x_raw))(jnp.asarray(Xd))
    keys = jax.random.split(jax.random.PRNGKey(1), n)
    for kw in ({"n_coarse": 6, "n_fine": 3, "half_width": 2.0}, {}):
        mv = make_grid_moves(post, [k], **kw)
        x1, _, _ = jax.jit(jax.vmap(mv))(keys, jnp.asarray(Xd), lld)
        x1 = np.asarray(x1)
        assert np.mean(x1[:, j] != draws) > 0.3  # it moves
        np.testing.assert_array_equal(np.delete(x1, j, axis=1), np.delete(Xd, j, axis=1))
        assert stats.kstest(x1[:, j], lambda t: np.interp(t, xx, cdf)).pvalue > 1e-3, kw
    # power check: an acceptance without the proposal ratio, with a coarse proposal, is biased
    from ptagwb import hybrid

    orig = hybrid.grid_logpdf
    try:
        hybrid.grid_logpdf = lambda *a: jnp.zeros(())  # q ratio = 1 -> wrong kernel
        mv = jax.jit(jax.vmap(make_grid_moves(post, [k], n_coarse=6, n_fine=3, half_width=2.0, w_uniform=0.5)))
        x1, l1 = jnp.asarray(Xd), lld
        for r_ in range(3):  # three applications amplify the bias of the wrong kernel
            x1, l1, _ = mv(jax.random.split(jax.random.PRNGKey(10 + r_), n), x1, l1)
    finally:
        hybrid.grid_logpdf = orig
    assert stats.kstest(np.asarray(x1)[:, j], lambda t: np.interp(t, xx, cdf)).pvalue < 1e-6


def test_hybrid_nuts_with_grid_moves_runs(like, tmp_path, monkeypatch):
    from ptagwb import sampling
    from ptagwb.sampling import ModelSpec, Posterior, RunConfig, load_run, run_nuts

    post = Posterior(like, ModelSpec(orf="hd", common="freespec", n_common=NC, n_modes=NM))
    monkeypatch.setattr(sampling, "RUNS_DIR", tmp_path)
    cfg = RunConfig(name="g", model={}, num_chains=2, num_warmup=10, num_samples=20, block=10, dense_mass=False,
                    seed=0, init_radius=1.0, grid_bins=[0, 2], step_size=0.05)
    run_nuts(cfg, post, log=lambda s: None)
    r = load_run("g")
    assert r["jump_accept"].shape == (2, 20, 2)
    assert np.all(np.isfinite(r["x"]))


@pytest.fixture(scope="module")
def joint_ref(like):
    """Draws from the joint conditional of (rho_k, one pulsar's IRN pair) given the rest (fine 3-D grid
    of the production likelihood, cell-uniform jitter), the grid marginals, and a frozen 2-D proposal."""
    from ptagwb.hybrid import fit_proposals
    from ptagwb.sampling import ModelSpec, Posterior, build_transform

    spec = ModelSpec(orf="hd", common="freespec", n_common=NC, n_modes=NM)
    post = Posterior(like, spec)
    tr = build_transform(spec, like.names)
    k, a = 2, 0
    rng = np.random.default_rng(7)
    p = _params(like.P, rng)
    p["log10_rho"] = rng.uniform(-12, -7, NC)
    x0 = np.concatenate([p["rn_log10_A"], p["rn_gamma"], p["log10_rho"]])
    jr, ja, jg = 2 * like.P + k, a, like.P + a
    gr, gA, gG = np.linspace(-15.5, -1, 241), np.linspace(-20, -11, 61), np.linspace(0, 7, 51)
    cr, cA, cG = (0.5 * (v[1:] + v[:-1]) for v in (gr, gA, gG))
    R_, A_, G_ = np.meshgrid(cr, cA, cG, indexing="ij")
    X = np.repeat(x0[None], R_.size, 0)
    X[:, jr], X[:, ja], X[:, jg] = R_.ravel(), A_.ravel(), G_.ravel()
    f = jax.jit(jax.vmap(post.logL_x_raw))
    ll = np.concatenate([np.asarray(f(jnp.asarray(X[i:i + 50_000]))) for i in range(0, len(X), 50_000)])
    pr = np.exp(ll - ll.max())
    pr /= pr.sum()
    n = 12_000
    cells = rng.choice(len(pr), n, p=pr)
    Xd = X[cells].copy()
    for jcol, gg in ((jr, gr), (ja, gA), (jg, gG)):
        Xd[:, jcol] += rng.uniform(-0.5, 0.5, n) * (gg[1] - gg[0])
    lld = f(jnp.asarray(Xd))
    fitX = np.repeat(x0[None], 3000, 0)
    fitX[:, ja], fitX[:, jg] = rng.normal(-14, 1.0, 3000).clip(-19.9, -11.1), rng.uniform(0.5, 6.5, 3000)
    prop = fit_proposals(fitX, list(tr.names), tr.lo, tr.hi, [], [(tr.names[ja], tr.names[jg])], k2a=6, k2b=5)
    psr = tr.names[ja][: -len("_red_noise_log10_A")]
    pmar = pr.reshape(len(cr), len(cA), len(cG))
    margs = ((jr, gr, pmar.sum((1, 2))), (ja, gA, pmar.sum((0, 2))), (jg, gG, pmar.sum((0, 1))))
    return post, k, psr, prop, Xd, lld, margs


def _joint_ks(x1, margs):
    from scipy import stats

    out = []
    for jcol, gg, m in margs:
        cdf = np.concatenate([[0.0], np.cumsum(m)])
        out.append(stats.kstest(x1[:, jcol], lambda t, gg=gg, cdf=cdf: np.interp(t, gg, cdf)).pvalue)
    return out


@pytest.mark.slow  # ~1e6 likelihood evaluations for the 3-D reference (shared by the next test)
def test_companion_grid_move_leaves_joint_conditional_invariant(joint_ref):
    """Joint move of rho_k and one pulsar's IRN (log10_A, gamma): draws from the joint conditional
    stay distributed as before (marginal KS against the discretised reference)."""
    from ptagwb.hybrid import make_grid_moves

    post, k, psr, prop, Xd, lld, margs = joint_ref
    keys = jax.random.split(jax.random.PRNGKey(3), len(Xd))
    mv = jax.jit(jax.vmap(make_grid_moves(post, [k], companions={k: [psr]}, proposals=prop)))
    x1, _, acc = mv(keys, jnp.asarray(Xd), lld)
    assert 0.05 < float(np.mean(acc)) < 0.98
    assert min(_joint_ks(np.asarray(x1), margs)) > 1e-3


@pytest.mark.slow
def test_companion_ratio_negative_control(joint_ref, monkeypatch):
    """Negative control: with the companion proposal density replaced by a constant (i.e. the
    companion ratio q2(c) / q2(c') dropped from the acceptance; the companion draws are unchanged),
    the same test detects the bias."""
    from ptagwb import hybrid

    post, k, psr, prop, Xd, lld, margs = joint_ref
    monkeypatch.setattr(hybrid, "log_q2", lambda *a, **kw: jnp.zeros(()))
    mv = jax.jit(jax.vmap(hybrid.make_grid_moves(post, [k], companions={k: [psr]}, proposals=prop)))
    x1, l1 = jnp.asarray(Xd), lld
    for r_ in range(3):
        x1, l1, _ = mv(jax.random.split(jax.random.PRNGKey(20 + r_), len(Xd)), x1, l1)
    assert min(_joint_ks(np.asarray(x1), margs)) < 1e-6


def test_repeated_companions_rejected(like):
    """A companion listed twice (or overlapping the bin) is rejected; repeating a bin as separate
    sequential moves stays allowed."""
    from ptagwb.hybrid import fit_proposals, make_grid_moves
    from ptagwb.sampling import ModelSpec, Posterior, build_transform

    spec = ModelSpec(orf="hd", common="freespec", n_common=NC, n_modes=NM)
    post = Posterior(like, spec)
    tr = build_transform(spec, like.names)
    rng = np.random.default_rng(0)
    X = tr.lo + (tr.hi - tr.lo) * rng.uniform(0.2, 0.8, (400, tr.dim))
    pairs = [(tr.names[a], tr.names[like.P + a]) for a in (0, 1)]
    prop = fit_proposals(X, list(tr.names), tr.lo, tr.hi, [], pairs, k2a=4, k2b=3)
    p0, p1 = (n[: -len("_red_noise_log10_A")] for n, _ in pairs)
    with pytest.raises(ValueError, match="repeated companion"):
        make_grid_moves(post, [2], companions={2: [p0, p0]}, proposals=prop)
    make_grid_moves(post, [2], companions={2: [p0, p1]}, proposals=prop)  # distinct: fine
    make_grid_moves(post, [2, 2, 3], companions={2: [p0]}, proposals=prop)  # repeated bin moves: fine
