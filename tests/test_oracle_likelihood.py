"""Full-67-pulsar CURN / HD log-likelihoods vs discovery, enterprise and the released chains.

Agreement targets (M1): likelihood differences between points <= 1e-6 absolute, and absolute
values once the constant conventions are matched ("chain" = discovery / enterprise 3.3.1,
"enterprise" = enterprise >= 3.4 with the n log 2pi term). Test points are posterior samples
from the released chains plus prior draws with IRN log10_A <= -13. For IRN amplitudes near
the top of the prior (log10_A -> -11, gamma -> 7) the oracles' normal-equation algebra loses
digits (up to ~2e-4 on the full PTA); there an extended-precision reference arbitrates
(``test_prior_edge_vs_extended_precision``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from oracle_helpers import (
    HAVE_DISCOVERY,
    HAVE_ENTERPRISE,
    discovery_model,
    enterprise_pta,
    load_chain,
    parameter_points,
    to_discovery_params,
    to_discovery_pulsar,
)

from ptagwb.data import get_tspan
from ptagwb.likelihood import PTALikelihood, precompute

pytestmark = [pytest.mark.oracle, pytest.mark.slow]
CHAIN = {"curn": "m2a", "hd": "m3a"}


@pytest.fixture(scope="module")
def terms_feathers(feathers, noisedict):
    T = get_tspan(feathers)
    return precompute(feathers, noisedict, T), T


@pytest.fixture(scope="module")
def terms_ours(ours, noisedict):
    T = get_tspan(ours)
    return precompute(ours, noisedict, T), T


def _points(names, orf, seed):
    return parameter_points(names, load_chain(CHAIN[orf]), 6, 6, seed=seed, rn_log10_A_range=(-20.0, -13.0))


def _discovery_values(psrs, noisedict, T, orf, pts):
    m = discovery_model([to_discovery_pulsar(p, noisedict) for p in psrs], orf, T, ecorr_enterprise=True)
    f = jax.jit(m.logL)
    vals = []
    for pt in pts:
        pd = to_discovery_params(pt, orf)
        vals.append(float(f({k: jnp.asarray(pd[k]) for k in m.logL.params})))
    return np.array(vals)


@pytest.mark.parametrize("source", ["feathers", "ours"])
@pytest.mark.parametrize("orf", ["curn", "hd"])
def test_vs_discovery(request, noisedict, orf, source):
    if not HAVE_DISCOVERY:
        pytest.skip("discovery not installed")
    psrs = request.getfixturevalue(source)
    terms, T = request.getfixturevalue(f"terms_{source}")
    pts = _points([p.name for p in psrs], orf, seed=1)
    L = PTALikelihood(terms, T, orf=orf)
    ours = np.array([float(L.logL(L.params_from_named(pt))) for pt in pts])
    ref = _discovery_values(psrs, noisedict, T, orf, pts)
    d = ours - ref
    print(f"{orf}/{source}: max |ours - discovery| = {np.abs(d).max():.2e}, spread {np.ptp(d):.2e}")
    assert np.ptp(d) <= 1e-6
    assert np.abs(d).max() <= 1e-6


@pytest.mark.parametrize("orf", ["curn", "hd"])
def test_vs_enterprise(feathers, noisedict, terms_feathers, orf):
    if not HAVE_ENTERPRISE:
        pytest.skip("enterprise not installed")
    terms, T = terms_feathers
    pts = _points([p.name for p in feathers], orf, seed=2)[1:11]  # 5 chain + 5 prior
    pta = enterprise_pta(feathers, noisedict, T, orf)
    L = PTALikelihood(terms, T, orf=orf, convention="enterprise")
    d = np.array(
        [
            float(L.logL(L.params_from_named(pt))) - pta.get_lnlikelihood({k: v for k, v in pt.items() if k[0] != "_"})
            for pt in pts
        ]
    )
    print(f"{orf}: max |ours - enterprise| = {np.abs(d).max():.2e}, spread {np.ptp(d):.2e}")
    assert np.ptp(d) <= 1e-6
    assert np.abs(d).max() <= 1e-6


@pytest.mark.parametrize("orf", ["curn", "hd"])
def test_reproduces_chain_logl(feathers, terms_feathers, ours, terms_ours, orf):
    """Released production chains (enterprise 3.3.1) store logl as float32 (ulp 0.5 at 8e6)."""
    chain = load_chain(CHAIN[orf])
    rows = chain.iloc[np.random.default_rng(3).choice(len(chain), 8, replace=False)]
    for terms, T, tag in ((*terms_feathers, "feathers"), (*terms_ours, "ours")):
        L = PTALikelihood(terms, T, orf=orf)
        d = np.array([float(L.logL(L.params_from_named(r.to_dict()))) - float(r["logl"]) for _, r in rows.iterrows()])
        print(f"{orf}/{tag}: ours - chain logl: mean {d.mean():+.3f}, max |.-mean| {np.abs(d - d.mean()).max():.3f}")
        if tag == "feathers":
            assert np.abs(d).max() <= 0.3
        else:  # PINT 1.1.7 front end + ICRS positions: constant offset, tiny spread
            assert np.abs(d - d.mean()).max() <= 0.4


def test_prior_edge_vs_extended_precision(feathers, noisedict):
    """Where discovery/enterprise deviate (huge IRN), ours matches a long-double reference."""
    from extended_precision import curn_loglike_ld, pulsar_contractions_ld

    T = get_tspan(feathers)
    fmap = {p.name: p for p in feathers}
    cases = [("J1705-1903", -11.78, 6.80), ("J1751-2857", -11.24, 6.39), ("J2214+3000", -11.60, 6.66)]
    for name, la, g in cases:
        p = fmap[name]
        L = PTALikelihood(precompute([p], noisedict, T), T, orf="curn")
        pt = {f"{name}_red_noise_log10_A": la, f"{name}_red_noise_gamma": g, "gw_log10_A": -17.33, "gw_gamma": 5.64}
        ref = curn_loglike_ld(pulsar_contractions_ld(p, noisedict, T), T, la, g, -17.33, 5.64)
        d = float(L.logL(L.params_from_named(pt))) - float(ref)
        print(f"{name}: ours - long double = {d:+.2e}")
        assert abs(d) <= 3e-8  # discovery: up to 2e-4 at these points


def test_identity_orf_equals_curn_full(terms_ours, ours):
    terms, T = terms_ours
    curn = PTALikelihood(terms, T, orf="curn")
    eye = PTALikelihood(terms, T, orf=np.eye(len(terms)))
    for pt in parameter_points([p.name for p in ours], load_chain("m2a"), 3, 2, seed=4, rn_log10_A_range=(-20, -13)):
        a, b = float(curn.logL(curn.params_from_named(pt))), float(eye.logL(eye.params_from_named(pt)))
        assert abs(a - b) <= 1e-7, (a, b)


def test_full_pta_gradient_fd(terms_ours, ours):
    terms, T = terms_ours
    L = PTALikelihood(terms, T, orf="hd")
    pt = parameter_points([p.name for p in ours], load_chain("m3a"), 1, 0, seed=6)[0]
    p0 = L.params_from_named(pt)
    g = jax.grad(L._logL)(p0)
    h = 1e-3  # 5-point stencil: truncation O(h^4), round-off ~1e-8 / h
    checks = [("log10_A", None), ("gamma", None), ("rn_log10_A", 0), ("rn_gamma", 30), ("rn_log10_A", 50)]
    for k, i in checks:
        v = np.array(p0[k], dtype=np.float64)
        e = np.zeros_like(v)
        if i is None:
            e = np.float64(h)
        else:
            e[i] = h
        f = {s_: float(L.logL(dict(p0, **{k: jnp.asarray(v + s_ * e)}))) for s_ in (-2, -1, 1, 2)}
        fd = (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * h)
        an = float(g[k] if i is None else g[k][i])
        assert abs(fd - an) <= 1e-4 * max(1.0, abs(an)), (k, i, fd, an)
