"""Pilot-v2 plumbing of the EPTA run driver (review of pilot v1): optional config fields, the fixed
reference-derived metric and frozen block proposals loaded from committed files, the pre-registered
in-run stop rules, checkpoints; the tuning-artifact generator; the pilot report's release rules.
No GPU, no sampling."""

from __future__ import annotations

import copy
import json
import sys
from collections import namedtuple

import numpy as np
import pytest

from ptagwb.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import m3b_pilot_report as P
import m3b_run_epta as R
import m3b_tuning_artifacts as G

V1 = REPO_ROOT / "configs" / "m3b" / "run_configs" / "epta_pilot_curn_freegamma.json"
RULES = {"stop_on_post_warmup_divergence": True, "check_at_transition": 150, "max_elapsed_s_at_check": 2400.0,
         "min_post_warmup_per_chain": 300, "projection_deadline_s": 5040.0, "projection_window_chunks": 3}


def v2() -> dict:
    c = json.loads(V1.read_text())
    c.update(run_id="epta_pilot_curn_freegamma_v2", num_warmup=150, max_transitions=600, max_tree_depth=8,
             adapt_mass_matrix=False, seed=20261011, target_accept_prob=0.9,
             inverse_mass_matrix="file:configs/m3b/metrics/epta_curn_ref_z_v1.npz",
             blocks={"proposals": "file:configs/m3b/proposals/epta_curn_ref_v1.json"}, stop_rules=dict(RULES))
    return c


def test_v1_configs_keep_their_behaviour():
    for f in sorted((REPO_ROOT / "configs" / "m3b" / "run_configs").glob("*.json")):
        cfg = R.validate_config(json.loads(f.read_text()))
        if "v1" in cfg["run_id"]:
            assert R.effective(cfg) == {"inverse_mass_matrix": None, "target_accept_prob": 0.8, "stop_rules": None}


def test_v2_draft_validates():
    cfg = R.validate_config(v2())
    assert R.effective(cfg)["target_accept_prob"] == 0.9


@pytest.mark.parametrize("change", [
    {"target_accept_prob": 0.3}, {"target_accept_prob": 1.0}, {"target_accept_prob": 1},
    {"adapt_mass_matrix": True}, {"dense_mass": False},
    {"inverse_mass_matrix": "file:configs/m3b/run_configs/x.npz"},
    {"inverse_mass_matrix": "file:configs/m3b/metrics/../x.npz"},
    {"inverse_mass_matrix": "configs/m3b/metrics/x.npz"},
    {"blocks": {"proposals": "file:configs/m3b/metrics/x.json"}},
    {"blocks": {"proposals": "prior", "extra": 1}},
    {"stop_rules": {k: v for k, v in RULES.items() if k != "projection_window_chunks"}},
    {"stop_rules": {**RULES, "check_at_transition": 155}},
    {"stop_rules": {**RULES, "max_elapsed_s_at_check": 2400}},
    {"surprise": 1},
])
def test_v2_validation_rejects(change):
    with pytest.raises(R.ConfigError):
        R.validate_config({**v2(), **change})


def test_block_layout_is_32_amplitude_pairs_and_the_dip():
    from ptagwb import epta

    man = epta.load_manifest()
    names, _, _, t0 = G.layout(man)
    lay = R.block_layout(man, names)
    assert len(lay) == 33 and lay[-1][0] == t0 and all(b[0].endswith("_log10_A") for b in lay[:-1])
    assert ("gw_crn_log10_A", "gw_crn_gamma") in lay


class _Model:
    def __init__(self, man):
        self.param_names, self.lo, self.hi, _ = G.layout(man)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Proposal/metric files in the driver's format: the staged reference-derived ones if present
    (exact recipe), else synthetic draws inside the box (format only)."""
    from ptagwb import epta

    man = epta.load_manifest()
    stage = G.STAGING
    if (stage / G.PROPOSAL_NAME).exists() and (stage / G.METRIC_NAME).exists():
        return man, stage / G.METRIC_NAME, stage / G.PROPOSAL_NAME
    names, lo, hi, t0 = G.layout(man)
    rng = np.random.default_rng(0)
    rows = lo + (hi - lo) * rng.uniform(0.01, 0.99, (2000, len(lo)))
    cont = [i for i, n in enumerate(names) if n != t0]
    d = tmp_path_factory.mktemp("art")
    np.savez(d / G.METRIC_NAME, inverse_mass_matrix=G.regularised_metric(G.box_logit(rows[:, cont], lo[cont], hi[cont])),
             names=np.array([names[i] for i in cont]))
    ix = {n: i for i, n in enumerate(names)}
    blocks = [{"params": list(b), "lo": [float(lo[ix[n]]) for n in b], "hi": [float(hi[ix[n]]) for n in b],
               "w_prior": 0.2, "edges": [G.equal_mass_edges(rows[:, ix[n]], lo[ix[n]], hi[ix[n]]) for n in b]}
              for b in R.block_layout(man, names)]
    (d / G.PROPOSAL_NAME).write_text(json.dumps({"blocks": blocks}))
    return man, d / G.METRIC_NAME, d / G.PROPOSAL_NAME


def _cfg_with(path, monkeypatch):
    monkeypatch.setattr(R, "file_ref", lambda v, kind: None if v in (None, "prior") else path)
    return {"blocks": {"proposals": "file:configs/m3b/proposals/epta_curn_ref_v1.json"}}


def test_build_blocks_from_a_proposal_file(artifacts, monkeypatch):
    man, _, pp = artifacts
    M = _Model(man)
    bl = R.build_blocks(man, M, _cfg_with(pp, monkeypatch))
    assert len(bl) == 33 and all(b.w_prior == 0.2 and len(b.edges) == len(b.idx) for b in bl)
    assert all(b.edges[0][0] == b.lo[0] and b.edges[0][-1] == b.hi[0] for b in bl)


@pytest.mark.parametrize("tamper", ["drop", "reorder", "box", "edges", "w0", "edge_count"])
def test_build_blocks_rejects_a_tampered_file(artifacts, monkeypatch, tmp_path, tamper):
    man, _, pp = artifacts
    d = json.loads(pp.read_text())
    b = d["blocks"]
    if tamper == "drop":
        b.pop(3)
    elif tamper == "reorder":
        b[0], b[1] = b[1], b[0]
    elif tamper == "box":
        b[0]["lo"][0] -= 1.0
    elif tamper == "edges":
        b[0]["edges"][0][5] = b[0]["edges"][0][4]
    elif tamper == "w0":
        b[0]["w_prior"] = 0.0
    else:
        b[0]["edges"].pop()
    f = tmp_path / "p.json"
    f.write_text(json.dumps(d))
    with pytest.raises(R.ConfigError):
        R.build_blocks(man, _Model(man), _cfg_with(f, monkeypatch))


def test_metric_loading_checks_names_symmetry_and_definiteness(artifacts, tmp_path):
    man, mp, _ = artifacts
    names, _, _, t0 = G.layout(man)
    cont = [n for n in names if n != t0]
    imm = R.load_metric(mp, cont)
    assert imm.shape == (66, 66) and np.linalg.eigvalsh(imm)[0] > 0
    with pytest.raises(R.ConfigError):
        R.load_metric(mp, cont[1:] + cont[:1])
    for bad in (imm + np.triu(np.full_like(imm, 1e-3), 1), imm - 10 * np.eye(66)):
        f = tmp_path / "m.npz"
        np.savez(f, inverse_mass_matrix=bad, names=np.array(cont))
        with pytest.raises(R.ConfigError):
            R.load_metric(f, cont)


def _chunks(secs, el0=0.0, L=10):
    out, el = [], el0
    for i, s in enumerate(secs):
        el += s
        out.append({"n": i, "seconds": s, "elapsed_s": el, "transitions_done": (i + 1) * L})
    return out


def test_stop_rules():
    assert R.stop_check(None, 150, 10, _chunks([100] * 15), 150, 5) is None
    assert "divergence" in R.stop_check(RULES, 150, 10, _chunks([10] * 16), 160, 1)
    assert R.stop_check(RULES, 150, 10, _chunks([10] * 16), 160, 0) is None
    # at 150: elapsed 15 x 170 = 2550 s > 2400 s
    assert "elapsed" in R.stop_check(RULES, 150, 10, _chunks([170] * 15), 150, 0)
    # elapsed 1500 s; slowest of the last 3 chunks 120 s / 10 = 12 s per transition -> 295 < 300
    secs = [100] * 12 + [100, 120, 80]
    assert "projected 295" in R.stop_check(RULES, 150, 10, _chunks(secs), 150, 0)
    # 10 s per transition -> 354 >= 300
    assert R.stop_check(RULES, 150, 10, _chunks([100] * 15), 150, 0) is None


def test_checkpoint_round_trip(tmp_path):
    import jax.numpy as jnp

    S = namedtuple("S", "z adapt")
    tree = (S(jnp.arange(6.0).reshape(2, 3), {"step": jnp.array([0.1, 0.2])}), jnp.array([1.0, 2.0]),
            jnp.array([7, 8], jnp.uint32))
    R.save_checkpoint(tmp_path / "c.npz", tree, {"transitions_done": 40})
    back, extra = R.load_checkpoint(tmp_path / "c.npz", tree)
    assert extra == {"transitions_done": 40}
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(__import__("jax").tree_util.tree_leaves(tree),
                                                                             __import__("jax").tree_util.tree_leaves(back)))


def test_generator_recipe():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4))
    c = np.cov(z, rowvar=False)
    m = G.regularised_metric(z)
    assert np.allclose(m, 0.95 * c + 0.05 * np.diag(np.diag(c)) + 1e-6 * np.eye(4))
    e = G.equal_mass_edges(rng.uniform(2, 3, 1000), 0.0, 5.0)
    assert len(e) == 21 and e[0] == 0.0 and e[-1] == 5.0 and np.all(np.diff(e) > 0)
    with pytest.raises(ValueError):
        G.equal_mass_edges(np.full(100, 2.0), 0.0, 5.0)
    with pytest.raises(ValueError):
        G.box_logit(np.array([[0.0]]), np.array([0.0]), np.array([1.0]))


def test_staged_artifacts_reproduce():
    """The staged (or committed) reference-derived artifacts are exactly what the generator builds."""
    from ptagwb import epta

    found = [d for d in (G.STAGING, REPO_ROOT / "configs" / "m3b") if (d / G.METRIC_NAME).exists()
             or (d / "metrics" / G.METRIC_NAME).exists()]
    if not found:
        pytest.skip("no tuning artifacts generated in this checkout")
    metric, props, prov = G.build(epta.load_manifest())
    for mp, pp in ((G.STAGING / G.METRIC_NAME, G.STAGING / G.PROPOSAL_NAME),
                   (REPO_ROOT / "configs/m3b/metrics" / G.METRIC_NAME, REPO_ROOT / "configs/m3b/proposals" / G.PROPOSAL_NAME)):
        if not mp.exists():
            continue
        with np.load(mp) as z:
            assert np.array_equal(z["inverse_mass_matrix"], metric["inverse_mass_matrix"])
            assert list(z["names"]) == list(metric["names"])
        assert json.loads(pp.read_text())["blocks"] == props["blocks"]
    assert prov["reference"]["rows_used"] == 22493 and prov["reference"]["burn_in_rows"] == 7497


def _per(rhat=1.002, eb=500.0, et=500.0, names=("a", "gw_crn_log10_A", "gw_crn_gamma")):
    return {n: {"rhat": rhat, "ess_bulk": eb, "ess_tail": et} for n in names}


def test_release_rules():
    items = [{"param": "x", "in_U": False, "status": "PASS"}, {"param": "y", "in_U": True, "status": "INCONCLUSIVE"}]
    ok = P.evaluate_release(450, 0, _per(), P.TARGETS, items)
    assert ok["screening_pass"] and ok["failed"] == []
    assert P.evaluate_release(299, 0, _per(), P.TARGETS, items)["failed"] == ["sufficient_evidence"]
    assert P.evaluate_release(450, 1, _per(), P.TARGETS, items)["failed"] == ["divergences"]
    assert P.evaluate_release(450, 0, _per(rhat=1.01), P.TARGETS, items)["failed"] == ["rhat"]
    assert "ess_all" in P.evaluate_release(450, 0, _per(et=99), P.TARGETS, items)["failed"]
    per = _per()
    per["gw_crn_gamma"]["ess_tail"] = 150
    assert P.evaluate_release(450, 0, per, P.TARGETS, items)["failed"] == ["ess_targets"]
    bad = copy.deepcopy(items)
    bad[0]["status"] = "INCONCLUSIVE"
    assert P.evaluate_release(450, 0, _per(), P.TARGETS, bad)["failed"] == ["transport_outside_U"]


def test_projection_arithmetic():
    reqs = [{"id": "floor", "factor": 2.0}, {"id": "E-3", "factor": 3.0}]
    p = P.projection(1200, 4, 8.0, 1800.0, reqs, 0.0069, 0.483)
    assert p["most_demanding"]["id"] == "E-3" and p["chain_transitions_needed"] == 3600
    assert p["run_A_breakdown"]["sampling_x2_h"] == pytest.approx(2 * 900 * 8.0 / 3600)
    assert p["run_A_gpu_h"] == pytest.approx(4.0 + 0.5 + 3600 * 0.0069 / 3600)
    assert p["remaining_total_h"] == pytest.approx(12 - 0.21 - 0.483) and p["run_A_fits"]
    assert not P.projection(1200, 4, 8.0, 1800.0, [{"id": "x", "factor": float("inf")}], 0.0, 0.5)["run_A_fits"]
