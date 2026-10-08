"""M3a multi-leg container and the combination gates on real validation pulsars:

* option C (YA-v3 / MetaPulsar-0.9.3 consistent rewrite) and option B (composite) against
  MetaPulsar v0.9.3 itself (gate G4 vs MetaPulsar): TOA-matched residuals and weighted timing
  column space;
* G5 on a small combined system (J1022+1001, 5 legs; J0437-4715, 3 legs; B and C): our likelihood
  vs enterprise (HD values) and discovery (CURN values and JAX gradients);
* G6 reference-model invariance (J1909-3744, references NG15 vs PPTA, clock/ephemeris held fixed);
* G8 signal/null injections on the real TOA sampling (score test, information equality).

Every multi-leg build runs each leg in a fresh spawned process. Results are also produced, with
more points/realisations, by ``scripts/m3a_validate.py`` (docs/M3A_VALIDATION.md).
"""

from __future__ import annotations

import multiprocessing

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from m3a_oracles import HAVE_DISCOVERY_ENTERPRISE, have_metapulsar

from ptagwb.combined import GeneralPTALikelihood, precompute_general

pytestmark = [pytest.mark.slow, pytest.mark.oracle]


def _legs(name):
    from ptagwb.m3data import validation_set

    return [tuple(x) for x in validation_set()["multileg"][name]]


def _build(name, timing, ref=None, force=None):
    from ptagwb.multileg import build_multileg
    from ptagwb.profiles import COMBINED, YA_V3_CLOCKS

    tag = f"test-{timing}-{ref}-" + "-".join(f"{k}{v}" for k, v in sorted((force or {}).items()))
    clk = COMBINED if timing == "per_leg" else YA_V3_CLOCKS
    try:
        mp, _ = build_multileg(name, _legs(name), timing=timing, clock=clk, reference=ref, tag=tag, jobs=5, force=force)
    except FileNotFoundError:
        pytest.skip("M3 data not fetched")
    return mp


@pytest.fixture(scope="module")
def systems():
    out = {(n, t): _build(n, t) for n in ("J1022+1001", "J0437-4715") for t in ("shared", "per_leg")}
    # option C with per-leg DM (MetaPulsar-main configuration): phase-connected for both pulsars,
    # unlike the v0.9.3 shared-DM rule (docs/M3A_VALIDATION.md Sec. 6)
    for n in ("J1022+1001", "J0437-4715"):
        out[(n, "localDM")] = _build(n, "shared", force={"local_dm": True})
    return out


def _mp_compare(name, strategy, ours):
    """MetaPulsar build in a fresh process (one clock profile per process)."""
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(_mp_worker, (name, strategy, ours))


def _mp_worker(name, strategy, ours):
    import os
    import warnings

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    warnings.simplefilter("ignore")
    import m3a_oracles as O
    import pint.logging

    from ptagwb.m3data import pta_of
    from ptagwb.multileg import choose_reference
    from ptagwb.profiles import COMBINED, SITE_PROFILE_OF_PTA, YA_V3_CLOCKS

    pint.logging.setup(level="ERROR")
    legs = _legs(name)
    ptas = [pta_of(d) for d, _ in legs]
    ref = choose_reference(ptas)
    order = [ref] + [p for p in ptas if p != ref]
    legs = [legs[ptas.index(p)] for p in order]
    mp, _ = O.metapulsar_build(legs, order, strategy, YA_V3_CLOCKS if strategy == "consistent" else COMBINED,
                               SITE_PROFILE_OF_PTA)
    return O.compare_with_metapulsar(ours, mp)


@pytest.mark.parametrize("timing,strategy", [("shared", "consistent"), ("per_leg", "composite")])
def test_container_vs_metapulsar(systems, timing, strategy):
    if not have_metapulsar():
        pytest.skip("MetaPulsar v0.9.3 source not fetched")
    ours = systems[("J0437-4715", timing)]
    r = _mp_compare("J0437-4715", strategy, ours)
    assert r["n_ours"] == r["n_metapulsar"] and r["ncol_ours"] == r["ncol_metapulsar"]
    assert all(v["max_abs_dres_ns"] < 1e-3 for v in r["residuals"].values()), r["residuals"]
    assert r["colspace_max_sin"] < 1e-6


def test_c_merges_shared_columns(systems):
    c, b = systems[("J1022+1001", "shared")], systems[("J1022+1001", "per_leg")]
    assert c.reference == "NG15" and c.Mmat.shape[1] < b.Mmat.shape[1]
    assert {"F0", "F1", "PB", "A1", "DM", "DM1", "DM2"} <= set(c.fitpars)
    assert all(f"Offset:{lg.pta}" in c.fitpars for lg in c.legs)
    assert len({s.split(":")[0] for s in c.backend_flags}) == 5  # namespaced systems


@pytest.mark.parametrize("timing", ["localDM", "per_leg"])
def test_g5_vs_enterprise_and_discovery(systems, timing):
    if not HAVE_DISCOVERY_ENTERPRISE:
        pytest.skip("oracle group not installed")
    import m3a_oracles as O

    mps = [systems[("J1022+1001", timing)], systems[("J0437-4715", timing)]]
    wns, models, nd, Tarr = O.g5_setup(mps, seed=7)
    terms = [precompute_general(p, w, m) for p, w, m in zip(mps, wns, models, strict=True)]
    hd, curn = GeneralPTALikelihood(terms, orf="hd"), GeneralPTALikelihood(terms, orf="curn")
    # arbiter for gradient disagreements with discovery: CURN through the correlated-ORF path
    # (identity ORF: diagonal split + Sigma' core + its custom VJP), numerically independent of the
    # separable CURN path (docs/M3A_VALIDATION.md Sec. 5)
    eye = GeneralPTALikelihood(terms, orf=np.eye(len(terms)))
    dl = O.discovery_g5(mps, nd, models, Tarr, 5)
    fd, gd = jax.jit(dl.logL), jax.jit(jax.grad(dl.logL))
    pta = O.enterprise_g5(mps, nd, models, Tarr, 5)
    vals = []
    for p in O.g5_params(mps, np.random.default_rng(11), 4):
        named = O._named(mps, p)
        jp = {k: jnp.asarray(v) for k, v in p.items()}
        vals.append([float(curn.logL(jp)), float(fd({k: jnp.asarray(named[k]) for k in dl.logL.params})),
                     float(hd.logL(jp)), float(pta.get_lnlikelihood({k: named[k] for k in pta.param_names}))])
        go = jax.grad(curn._logL)(jp)
        ge = jax.grad(eye._logL)(jp)
        gdd = gd({k: jnp.asarray(named[k]) for k in dl.logL.params})
        pairs = [(float(np.asarray(go[f"{proc}_{par}"])[i]), float(np.asarray(ge[f"{proc}_{par}"])[i]),
                  float(gdd[f"{m.name}_{proc}_{par}"])) for i, m in enumerate(mps) for proc in ("rn", "dm")
                 for par in ("log10_A", "gamma")]
        pairs += [(float(go[par]), float(ge[par]), float(gdd[key])) for par, key in
                  (("log10_A", "gw_log10_A"), ("gamma", "gw_gamma"))]
        for a, a_eye, b in pairs:
            sc = max(abs(b), 1.0)
            # pass: within 1e-8 of discovery, or (arbitrated) our two independent paths agree to
            # 1e-11 and discovery is within 1e-7
            assert abs(a - b) <= 1e-8 * sc or (abs(a - a_eye) <= 1e-11 * sc and abs(a - b) <= 1e-7 * sc), (a, a_eye, b)
    v = np.array(vals)
    d = v - v[0]
    tol = np.maximum(1e-6, 1e-9 * np.abs(v[:, 0]))
    assert np.all(np.abs(d[:, 0] - d[:, 1]) <= tol), d
    assert np.all(np.abs(d[:, 2] - d[:, 3]) <= tol), d


@pytest.fixture(scope="module")
def j1909_swap():
    base = _build("J1909-3744", "shared", "NG15")
    other = _build("J1909-3744", "shared", "PPTA", force={"clock": "TT(BIPM2019)", "ephem": "DE440"})
    return base, other


def test_g6_linearisation(j1909_swap):
    """The residual difference between references is linear (whitened norm <= 0.1) and every column
    except SINI agrees to < 1e-3 (the documented diagnosis of the open G6 gate)."""
    import m3a_oracles as O

    base, other = j1909_swap
    key = {(a, b): i for i, (a, b) in enumerate(zip(other.flags["pta"], other.flags["name"], strict=True))}
    idx = np.array([key[(a, b)] for a, b in zip(base.flags["pta"], base.flags["name"], strict=True)])
    W = 1 / base.toaerrs
    e = O.weighted_projector_complement(np.hstack([base.Mmat * W[:, None], other.Mmat[idx] * W[:, None]]))(
        (base.residuals - other.residuals[idx]) * W)
    assert np.linalg.norm(e) <= 0.1
    B = other.Mmat[idx] * W[:, None]
    QB, _ = np.linalg.qr(B / np.linalg.norm(B, axis=0))
    for j, nm in enumerate(base.fitpars):
        c = base.Mmat[:, j] * W
        c = c / np.linalg.norm(c)
        mis = np.linalg.norm(c - QB @ (QB.T @ c))
        assert mis < 1e-3 or nm == "SINI", (nm, mis)


@pytest.mark.xfail(strict=True, reason="gate G6 (exit condition E7) is open: the SINI column of the nearly "
                   "edge-on J1909-3744 depends on the reference's nominal SINI (docs/M3A_VALIDATION.md Sec. 6)")
def test_g6_reference_swap(j1909_swap):
    import m3a_oracles as O

    base, other = j1909_swap
    key = {(a, b): i for i, (a, b) in enumerate(zip(other.flags["pta"], other.flags["name"], strict=True))}
    idx = np.array([key[(a, b)] for a, b in zip(base.flags["pta"], base.flags["name"], strict=True)])
    W = 1 / base.toaerrs
    e = O.weighted_projector_complement(np.hstack([base.Mmat * W[:, None], other.Mmat[idx] * W[:, None]]))(
        (base.residuals - other.residuals[idx]) * W)
    assert np.linalg.norm(e) <= 0.1
    assert O.principal_sines(base.Mmat * W[:, None], other.Mmat[idx] * W[:, None])[0] <= 1e-3


def test_g8_injections(systems):
    import m3a_injection as I

    C = [systems[("J1022+1001", "localDM")], systems[("J0437-4715", "localDM")]]
    B = [systems[("J1022+1001", "per_leg")], systems[("J0437-4715", "per_leg")]]
    out = I.run(C, B, R=300, log10_A_true=-14.0)
    assert out["pass"], out["results"]


def test_shared_dm_breaks_phase_connection(systems):
    """The v0.9.3 shared-DM rule leaves per-leg DM offsets as nu^-2 delays the shared DM column cannot
    absorb: J1022+1001 legs wrap (rms of hundreds of us), the local-DM variant stays at us level."""
    sh, loc = systems[("J1022+1001", "shared")], systems[("J1022+1001", "localDM")]
    rms = lambda m, i: float(np.std(m.residuals[m.leg == i]))
    assert max(rms(sh, i) for i in range(len(sh.legs))) > 1e-4
    assert max(rms(loc, i) for i in range(1, len(loc.legs))) < 5e-5
