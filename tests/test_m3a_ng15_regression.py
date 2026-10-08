"""Gate G9: the M2 NG15 likelihood through the new M3a path (single leg per pulsar, general
stage 1 ``combined.precompute_general`` with the M1 white noise, IRN block on the array span,
common block on the same grid -> prefix identity, ``GeneralPTALikelihood``) must reproduce
``likelihood.PTALikelihood`` (production) and ``FastPTALikelihood`` bit-for-bit, or within the M1
budget (value <= 1e-9 absolute, gradient <= 1e-8 relative to max(|g|, 1)).

Also checks that wrapping each NG15 pulsar as a one-leg ``MultiLegPulsar`` (namespaced systems,
per-leg timing columns) leaves the stage-1 terms unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from oracle_helpers import load_chain, parameter_points

from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
from ptagwb.data import get_tspan
from ptagwb.gp import FourierBlock
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.noise import build_white_noise
from ptagwb.perf_likelihood import FastPTALikelihood

pytestmark = [pytest.mark.slow]
VAL_TOL, GRAD_TOL = 1e-9, 1e-8


@pytest.fixture(scope="module")
def ng15(ours, noisedict):
    T = get_tspan(ours)
    m1 = precompute(ours, noisedict, T)
    model = PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 30, T)}, common=FourierBlock("gw", 14, T))
    gen = [precompute_general(p, build_white_noise(p, noisedict), model) for p in ours]
    return ours, T, m1, gen


def _points(names, orf):
    try:
        chain = load_chain("m2a" if orf == "curn" else "m3a")
    except Exception:  # noqa: BLE001
        pytest.skip("released NG15 chains not fetched")
    pts = parameter_points(names, chain, 4, 4, seed=3, rn_log10_A_range=(-20.0, -12.0))
    out = []
    for pt in pts:
        out.append({
            "rn_log10_A": jnp.array([pt[f"{n}_red_noise_log10_A"] for n in names]),
            "rn_gamma": jnp.array([pt[f"{n}_red_noise_gamma"] for n in names]),
            "log10_A": jnp.asarray(pt["gw_log10_A"]), "gamma": jnp.asarray(pt["gw_gamma"]),
        })
    return out


def test_stage1_bit_identical(ng15):
    _, _, m1, gen = ng15
    for a, b in zip(m1, gen, strict=True):
        assert np.array_equal(a.RA, b.RA) and np.array_equal(a.c, b.c), a.name
        assert a.s_perp == b.s_perp and a.logdet_N == b.logdet_N and a.logdet_MNM == b.logdet_MNM
        assert b.K == 60 and np.array_equal(b.layout.common_cols, np.arange(28))


@pytest.mark.parametrize("orf", ["curn", "hd"])
def test_value_and_gradient_vs_production(ng15, orf):
    psrs, T, m1, gen = ng15
    prod = PTALikelihood(m1, T, orf=orf)
    fast = FastPTALikelihood(m1, T, orf=orf)
    g_prod = GeneralPTALikelihood(gen, orf=orf)
    g_fast = GeneralPTALikelihood(gen, orf=orf, reduce="hh", tri_inv="levels")
    names = [p.name for p in psrs]
    worst = {"bit_identical_values": 0, "n": 0, "max_dvalue": 0.0, "max_dgrad_rel": 0.0}
    for p in _points(names, orf):
        for ref, new in ((prod, g_prod), (fast, g_fast)):
            (v1, g1), (v2, g2) = ref.value_and_grad(p), new.value_and_grad(p)
            worst["n"] += 1
            worst["bit_identical_values"] += float(v1) == float(v2)
            dv = abs(float(v1) - float(v2))
            worst["max_dvalue"] = max(worst["max_dvalue"], dv)
            assert dv <= VAL_TOL, (orf, dv)
            for k in g1:
                a, b = np.asarray(g1[k]), np.asarray(g2[k])
                rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0)))
                worst["max_dgrad_rel"] = max(worst["max_dgrad_rel"], rel)
                assert rel <= GRAD_TOL, (orf, k, rel)
    print(f"G9 {orf}: {worst}")


def test_freespec_vs_production(ng15):
    psrs, T, m1, _ = ng15
    prod = PTALikelihood(m1, T, orf="hd", common="freespec", n_common=30)
    rng = np.random.default_rng(0)
    gen_terms = ng15[3]
    for t in gen_terms:  # same stage 1; a 30-mode common block re-uses all 60 IRN columns
        t.layout.model.common = FourierBlock("gw", 30, T)
    from ptagwb.combined import ColumnLayout

    for t in gen_terms:
        t.layout = ColumnLayout(t.layout.model)
    g = GeneralPTALikelihood(gen_terms, orf="hd", common="freespec")
    for _ in range(3):
        p = {"rn_log10_A": jnp.asarray(rng.uniform(-16, -13, len(psrs))), "rn_gamma": jnp.asarray(rng.uniform(1, 6, len(psrs))),
             "log10_rho": jnp.asarray(rng.uniform(-9, -6, 30))}
        (v1, g1), (v2, g2) = prod.value_and_grad(p), g.value_and_grad(p)
        assert abs(float(v1) - float(v2)) <= VAL_TOL
        for k in g1:
            a, b = np.asarray(g1[k]), np.asarray(g2[k])
            assert np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0)) <= GRAD_TOL
    for t in gen_terms:  # restore for other tests in the module
        t.layout.model.common = FourierBlock("gw", 14, T)
        t.layout = ColumnLayout(t.layout.model)



def test_multileg_wrapper_is_transparent(ng15, noisedict):
    from ptagwb.multileg import stack_legs
    from ptagwb.noise import namespace

    psrs, T, m1, _ = ng15
    model = PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 30, T)}, common=FourierBlock("gw", 14, T))
    for p, t in list(zip(psrs, m1, strict=True))[:5]:
        mp = stack_legs(p.name, [p], ["NG15"], timing="per_leg")
        nd = {k.replace(f"{p.name}_", f"{p.name}_{namespace('NG15', '')}", 1) if k.startswith(p.name + "_") else k: v
              for k, v in noisedict.items()}
        wn = build_white_noise(mp, nd)
        g = precompute_general(mp, wn, model)
        np.testing.assert_allclose(g.RA.T @ g.RA, t.RA.T @ t.RA, rtol=1e-10, atol=1e-10 * np.abs(t.RA.T @ t.RA).max())
        assert abs(g.s_perp + g.c @ g.c - (t.s_perp + t.c @ t.c)) <= 1e-9 * (t.s_perp + t.c @ t.c)
        assert abs(g.const() - t.const()) <= 1e-8 * abs(t.const())
