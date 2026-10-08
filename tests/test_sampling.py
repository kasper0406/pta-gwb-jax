"""Sampler building blocks: box transform + Jacobian, NUTS on toy targets with known answers
(including a deterministic approximate gradient), and the convergence diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from ptagwb.diagnostics import ess, ess_bulk, ess_tail, mcse_quantile, rhat, summarize
from ptagwb.sampling import BoxTransform, ModelSpec, Posterior, build_transform, unpack


def _box(D=3, seed=0):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-20, -5, D)
    return BoxTransform(tuple(f"p{i}" for i in range(D)), lo, lo + rng.uniform(0.5, 10, D))


def test_transform_roundtrip_and_jacobian():
    tr = _box(5)
    z = np.random.default_rng(1).normal(size=5) * 3
    x = tr.to_constrained(z)
    assert np.all((x > tr.lo) & (x < tr.hi))
    assert np.allclose(tr.to_unconstrained(x), z, rtol=1e-10, atol=1e-10)
    J = jax.jacfwd(lambda zz: tr.to_constrained(zz))(jnp.asarray(z))
    assert np.allclose(np.asarray(J), np.diag(np.diag(np.asarray(J))))  # elementwise map
    assert np.isclose(float(tr.log_jacobian(jnp.asarray(z))), np.linalg.slogdet(np.asarray(J))[1], rtol=1e-12)
    # extreme z: finite log Jacobian (no 0 * inf)
    zz = jnp.array([-60.0, 60.0, 0.0, 700.0, -700.0])
    assert np.all(np.isfinite(np.asarray(tr.log_jacobian(zz))))
    with pytest.raises(ValueError):
        tr.to_unconstrained(tr.hi)


def test_flat_likelihood_density_integrates_to_one():
    """exp(log pi(x(z)) + log|J|) is a normalised density on R (1-D quadrature per box)."""
    tr = _box(1, seed=3)
    zs = np.linspace(-40, 40, 200001)
    dens = np.exp(np.asarray(jax.vmap(lambda z: tr.log_jacobian(z[None]) - tr.log_volume)(jnp.asarray(zs))))
    assert abs(np.trapezoid(dens, zs) - 1.0) < 1e-8


def test_model_layout():
    names = ["A", "B", "C"]
    for spec, D in ((ModelSpec(gamma=13 / 3), 7), (ModelSpec(), 8), (ModelSpec(common="freespec", n_common=5), 11)):
        tr = build_transform(spec, names)
        assert tr.dim == D
        x = jnp.asarray(tr.to_constrained(np.zeros(D)))
        p = unpack(x, spec, 3)
        assert p["rn_log10_A"].shape == (3,) and p["rn_gamma"].shape == (3,)
    tr = build_transform(ModelSpec(gamma=13 / 3), names)
    assert (tr.lo[-1], tr.hi[-1]) == (-18.0, -11.0)  # resolved production prior (M2)
    tr = build_transform(ModelSpec(gamma=13 / 3, prior_overrides={"gw_log10_A_fixed_gamma": (-18, -14)}), names)
    assert tr.hi[-1] == -14.0


def _nuts(post, D, seed=0, n=1500, chains=4, warm=500):
    from numpyro.infer import MCMC, NUTS

    m = MCMC(NUTS(potential_fn=post.potential_fn, dense_mass=True), num_warmup=warm, num_samples=n,
             num_chains=chains, chain_method="vectorized", progress_bar=False)
    z0 = np.random.default_rng(seed).uniform(-2, 2, (chains, D))
    m.run(jax.random.PRNGKey(seed), init_params=jnp.asarray(z0))
    z = np.asarray(m.get_samples(group_by_chain=True))
    return np.asarray(post.transform.to_constrained(z))


def _check_truncnorm(x, tr, mu, sd):
    for j in range(len(mu)):
        a, b = (tr.lo[j] - mu[j]) / sd[j], (tr.hi[j] - mu[j]) / sd[j]
        ref = stats.truncnorm(a, b, loc=mu[j], scale=sd[j])
        for p in (0.05, 0.5, 0.95):
            q, se = np.quantile(x[..., j], p), mcse_quantile(x[..., j], p)
            assert abs(q - ref.ppf(p)) < 5 * se + 1e-3 * sd[j], (j, p, q, ref.ppf(p), se)
        assert rhat(x[..., j]) < 1.02


def test_nuts_truncated_gaussian():
    """Gaussian likelihood cut by the box: marginals must be truncated normals."""
    tr = BoxTransform(("a", "b", "c"), np.array([0.0, -1.0, 2.0]), np.array([1.0, 3.0, 9.0]))
    mu, sd = np.array([0.9, -0.5, 4.0]), np.array([0.3, 1.0, 0.05])
    post = Posterior.generic(tr, lambda x: -0.5 * jnp.sum(((x - mu) / sd) ** 2))
    _check_truncnorm(_nuts(post, 3), tr, mu, sd)


def test_nuts_exact_with_deterministic_approximate_gradient():
    """Leapfrog with a deterministic, position-only force field is reversible and volume
    preserving; NUTS's multinomial sampling uses the exact energy, so a perturbed gradient
    (here +5% x sin(3 z) relative) must not bias the samples (justifies grad_precision="mixed")."""
    tr = BoxTransform(("a", "b"), np.array([-3.0, 0.0]), np.array([3.0, 7.0]))
    mu, sd = np.array([0.5, 3.2]), np.array([0.7, 0.4])

    @jax.custom_vjp
    def loglike(x):
        return -0.5 * jnp.sum(((x - mu) / sd) ** 2)

    def fwd(x):
        return loglike(x), x

    def bwd(x, g):
        exact = -(x - mu) / sd**2
        return (g * exact * (1.0 + 0.05 * jnp.sin(3.0 * x)),)

    loglike.defvjp(fwd, bwd)
    post = Posterior.generic(tr, loglike)
    _check_truncnorm(_nuts(post, 2, seed=4), tr, mu, sd)


# ---------------------------------------------------------------------- diagnostics


def _ar1(phi, n, chains, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((chains, n))
    e = rng.normal(size=(chains, n)) * np.sqrt(1 - phi**2)
    x[:, 0] = rng.normal(size=chains)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def test_ess_iid_and_ar1():
    x = np.random.default_rng(0).normal(size=(4, 5000))
    assert 0.85 * x.size < ess(x) < 1.15 * x.size
    assert 0.85 * x.size < ess_bulk(x) < 1.15 * x.size
    assert 0.8 * x.size < ess_tail(x) < 1.2 * x.size
    assert rhat(x) < 1.01
    phi = 0.9
    y = _ar1(phi, 20000, 4, 1)
    expect = y.size * (1 - phi) / (1 + phi)
    assert 0.8 * expect < ess(y) < 1.2 * expect


def test_rhat_detects_offset_chains():
    x = np.random.default_rng(2).normal(size=(4, 2000))
    x[0] += 1.0
    assert rhat(x) > 1.05


def test_mcse_quantile_calibrated():
    """Spread of the median over replicate iid chains matches the MCSE estimate."""
    rng = np.random.default_rng(5)
    meds, ses = [], []
    for _ in range(200):
        x = rng.normal(size=(1, 2000))
        meds.append(np.median(x))
        ses.append(mcse_quantile(x, 0.5))
    assert 0.8 < np.std(meds) / np.mean(ses) < 1.25
    s = summarize(rng.normal(size=(4, 1000)))
    assert {"q05", "q50", "q95", "q50_mcse", "rhat", "ess_bulk", "ess_tail"} <= set(s)


def test_run_config_likelihood_impl_default_and_validation():
    from ptagwb.config import REPO_ROOT
    from ptagwb.sampling import RunConfig

    cfg = RunConfig.from_json(REPO_ROOT / "configs/m2/hd_fs30_v2.json")
    assert cfg.likelihood_impl == "production"  # default unchanged for existing configs
    assert RunConfig(name="t", model={}, likelihood_impl="fast").likelihood_impl == "fast"
    with pytest.raises(ValueError, match="likelihood_impl"):
        RunConfig(name="t", model={}, likelihood_impl="bogus")


@pytest.mark.parametrize("common,nc,gp", [("freespec", 8, "mixed"), ("powerlaw", 5, "float64")])
def test_make_likelihood_fast_matches_production(common, nc, gp):
    """``likelihood_impl="fast"`` gives a FastPTALikelihood whose posterior potential and gradient
    agree with production (budgets as tests/test_perf_likelihood.py: 2e-9 abs, 1e-8 rel; mixed
    precision only perturbs the gradient, so its budget is looser there)."""
    from synthetic import make_pta, tspan

    from ptagwb.likelihood import PTALikelihood, precompute
    from ptagwb.perf_likelihood import FastPTALikelihood
    from ptagwb.sampling import make_likelihood

    psrs, nd = make_pta(4, seed=3, n_epochs=40, signal=3e-7)
    T = tspan(psrs)
    terms = precompute(psrs, nd, T, n_modes=10)
    spec = ModelSpec(orf="hd", common=common, n_common=nc, n_modes=10, grad_precision=gp)
    prod, fast = make_likelihood(terms, T, spec), make_likelihood(terms, T, spec, "fast")
    assert type(prod) is PTALikelihood and isinstance(fast, FastPTALikelihood)
    assert (fast.reduce_name, fast.tri_inv) == ("hh", "levels")
    with pytest.raises(ValueError, match="likelihood_impl"):
        make_likelihood(terms, T, spec, "bogus")
    pp, pf = Posterior(prod, spec), Posterior(fast, spec)
    vg_p, vg_f = jax.jit(jax.value_and_grad(pp.potential_fn)), jax.jit(jax.value_and_grad(pf.potential_fn))
    rng = np.random.default_rng(0)
    for z in rng.uniform(-2, 2, (4, pp.transform.dim)):
        v0, g0 = vg_p(jnp.asarray(z))
        v1, g1 = vg_f(jnp.asarray(z))
        assert abs(float(v0) - float(v1)) <= 2e-9
        tol = 1e-8 if gp == "float64" else 1e-4
        assert float(jnp.max(jnp.abs(g0 - g1))) <= tol * max(1.0, float(jnp.max(jnp.abs(g0))))
