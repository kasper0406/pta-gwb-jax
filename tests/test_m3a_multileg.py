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


def _build(name, timing, ref=None, force=None, allow_inadmissible=False):
    from ptagwb.multileg import build_multileg
    from ptagwb.profiles import COMBINED, YA_V3_CLOCKS

    tag = f"test-{timing}-{ref}-" + "-".join(f"{k}{v}" for k, v in sorted((force or {}).items()))
    clk = COMBINED if timing == "per_leg" else YA_V3_CLOCKS
    try:
        mp, _ = build_multileg(name, _legs(name), timing=timing, clock=clk, reference=ref, tag=tag, jobs=5, force=force,
                               allow_inadmissible=allow_inadmissible)
    except FileNotFoundError:
        pytest.skip("M3 data not fetched")
    return mp


@pytest.fixture(scope="module")
def systems():
    # the YA-v3 shared-DM C builds of these two pulsars are inadmissible (phase wraps): built here
    # only as explicitly marked diagnostics (MetaPulsar comparison, wrap test)
    out = {(n, t): _build(n, t, allow_inadmissible=(t == "shared")) for n in ("J1022+1001", "J0437-4715")
           for t in ("shared", "per_leg")}
    # option C with per-leg DM (MetaPulsar-main configuration): phase-connected for both pulsars,
    # unlike the v0.9.3 shared-DM rule (docs/M3A_VALIDATION.md Sec. 6)
    for n in ("J1022+1001", "J0437-4715"):
        out[(n, "localDM")] = _build(n, "shared", force={"local_dm": True}, allow_inadmissible=True)
    out[("J1909-3744", "shared")] = _build("J1909-3744", "shared")  # admissible YA-v3 build
    out[("J1909-3744", "per_leg")] = _build("J1909-3744", "per_leg")
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


@pytest.mark.parametrize("which", ["C", "B"])
def test_g5_vs_arbiter_enterprise_discovery(systems, which):
    """Gating (fixed in advance, constant-invariant): ours vs the independent long-double arbiter,
    lnL shape differences <= 1e-6 nats (CURN and HD) and gradients <= 1e-8 relative. The float64
    oracles (discovery CURN, enterprise HD) must agree with ours to 1e-6 nats on the B system; on the
    87k-TOA C system they deviate from the arbiter by up to ~1e-4 nats themselves
    (docs/M3A_VALIDATION.md Sec. 5), so there they must only be farther from the arbiter than we are."""
    if not HAVE_DISCOVERY_ENTERPRISE:
        pytest.skip("oracle group not installed")
    import m3a_oracles as O
    from m3a_arbiter import PulsarArbiter, arbiter_correlated, arbiter_curn

    from ptagwb import orf as orfs

    mps = ([systems[("J1909-3744", "shared")], systems[("J0437-4715", "per_leg")]] if which == "C"
           else [systems[("J1022+1001", "per_leg")], systems[("J0437-4715", "per_leg")]])
    assert all(m.meta["admissible"] for m in mps)
    wns, models, nd, Tarr = O.g5_setup(mps, seed=7)
    terms = [precompute_general(p, w, m) for p, w, m in zip(mps, wns, models, strict=True)]
    hd, curn = GeneralPTALikelihood(terms, orf="hd"), GeneralPTALikelihood(terms, orf="curn")
    arbs = [PulsarArbiter(p, w, m) for p, w, m in zip(mps, wns, models, strict=True)]
    G = orfs.hd(np.stack([m.pos for m in mps]))
    dl = O.discovery_g5(mps, nd, models, Tarr, 5)
    fd = jax.jit(dl.logL)
    pta = O.enterprise_g5(mps, nd, models, Tarr, 5)
    vals = []
    for p in O.g5_params(mps, np.random.default_rng(99), 3):
        named = O._named(mps, p)
        jp = {k: jnp.asarray(v) for k, v in p.items()}
        la, ga = arbiter_curn(arbs, p)
        vals.append([float(curn.logL(jp)), la, float(hd.logL(jp)), arbiter_correlated(arbs, p, G),
                     float(fd({k: jnp.asarray(named[k]) for k in dl.logL.params})),
                     float(pta.get_lnlikelihood({k: named[k] for k in pta.param_names}))])
        go = jax.grad(curn._logL)(jp)
        for k in ga:
            a, b = np.atleast_1d(np.asarray(go[k], dtype=float)), np.atleast_1d(ga[k])
            assert np.all(np.abs(a - b) <= 1e-8 * np.maximum(1.0, np.abs(b))), (k, a, b)
    d = np.array(vals) - np.array(vals)[0]
    ours_curn, ours_hd = np.max(np.abs(d[:, 0] - d[:, 1])), np.max(np.abs(d[:, 2] - d[:, 3]))
    assert ours_curn <= 1e-6 and ours_hd <= 1e-6, (ours_curn, ours_hd)
    disc, ent = np.max(np.abs(d[:, 4] - d[:, 1])), np.max(np.abs(d[:, 5] - d[:, 3]))
    if which == "B":
        assert np.max(np.abs(d[:, 0] - d[:, 4])) <= 1e-6 and np.max(np.abs(d[:, 2] - d[:, 5])) <= 1e-6
    else:
        assert disc >= ours_curn and ent >= ours_hd, (disc, ours_curn, ent, ours_hd)


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

    C, B = [systems[("J1909-3744", "shared")]], [systems[("J1909-3744", "per_leg")]]
    out = I.run(C, B, R=300, log10_A_true=-14.0)
    assert out["pass"], out["results"]


def test_shared_dm_breaks_phase_connection(systems):
    """The v0.9.3 shared-DM rule leaves per-leg DM offsets as nu^-2 delays the shared DM column cannot
    absorb: J1022+1001 legs wrap (rms of hundreds of us), the local-DM variant stays at us level."""
    sh, loc = systems[("J1022+1001", "shared")], systems[("J1022+1001", "localDM")]
    rms = lambda m, i: float(np.std(m.residuals[m.leg == i]))
    assert max(rms(sh, i) for i in range(len(sh.legs))) > 1e-4
    assert max(rms(loc, i) for i in range(1, len(loc.legs))) < 5e-5


def test_inadmissible_build_rejected_and_marked(systems):
    from ptagwb.combined import PulsarGPModel
    from ptagwb.gp import FourierBlock
    from ptagwb.multileg import CONFIG_LOCAL_DM, CONFIG_YA, InadmissibleBuildError
    from ptagwb.noise import build_general_white_noise

    sh, loc = systems[("J1022+1001", "shared")], systems[("J1022+1001", "localDM")]
    assert sh.meta["config"] == CONFIG_YA and sh.meta["admissible"] is False and sh.meta["diagnostic_override"]
    assert sh.meta["admissibility"]["InPTA"]["linearisation_rms_whitened"] > 10  # phase wraps
    assert loc.meta["config"] == CONFIG_LOCAL_DM and loc.meta["admissible"] is False  # PPTA DDH vs NG15 DD
    good = systems[("J1909-3744", "shared")]
    assert good.meta["config"] == CONFIG_YA and good.meta["admissible"] is True
    assert max(a["linearisation_rms_whitened"] for a in good.meta["admissibility"].values()) < 0.01
    assert all("admissibility" in lg.meta and "profile" in lg.meta for lg in sh.legs)
    with pytest.raises(InadmissibleBuildError):
        _build("J1022+1001", "shared")
    sy = sorted(set(sh.backend_flags.tolist()))
    wn = build_general_white_noise(sh.toas, sh.toaerrs, sh.backend_flags, {s: (1.0, -9.0) for s in sy})
    mdl = PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 5, 3e8)})
    with pytest.raises(ValueError, match="inadmissible"):
        precompute_general(sh, wn, mdl)


def test_meta_survives_serialisation(systems, tmp_path):
    from ptagwb.multileg import load_multileg, save_multileg

    sh = systems[("J1022+1001", "shared")]
    save_multileg(sh, tmp_path / "x.npz")
    back = load_multileg(tmp_path / "x.npz")
    assert back.meta["config"] == sh.meta["config"] and back.meta["admissible"] is False
    assert back.legs[0].meta["profile"] == sh.legs[0].meta["profile"]
    assert back.legs[0].meta["admissibility"]["admissible"] == sh.legs[0].meta["admissibility"]["admissible"]


def test_option_c_keeps_reference_harmonic_count():
    """Review M3a #1: the shared-model rewrite must not restore PINT's forced NHARMS = 7."""
    from ptagwb.m3data import quarantine
    from ptagwb.multileg import build_multileg
    from ptagwb.profiles import YA_V3_CLOCKS

    legs = [("epta_dr2new", "J0751+1807"), ("inpta_dr2", "J0751+1807")]
    mp, res = build_multileg("J0751+1807", legs, timing="shared", clock=YA_V3_CLOCKS, tag="test-nharms-C", jobs=2,
                             allow_inadmissible=True)
    for r in res.values():
        assert r.meta["nharms_used"] == 4 and r.meta["nharms_setup"] == 7, r.meta.get("nharms_used")
        assert "NHARMS 4" in r.meta["par_as_loaded"].replace("  ", " ") or any(
            ln.split()[:2] == ["NHARMS", "4"] for ln in r.meta["par_as_loaded"].splitlines())
    assert all(lg.meta["ell1h_nharms_config"] == "tempo2" for lg in mp.legs)
    assert ("epta_dr2new", "J0751+1807") not in quarantine({"ell1h_nharms": "tempo2"})
