"""M3b-0E gates on the real EPTA DR2new data (docs/M3B_PLAN.md Sec. 4.1, 4.4, 4.5, 6.2, 6.5):
manifest regeneration and prior volume, T1 on the complete roster, the chain fingerprint, G5-PTA
(arbiter, enterprise, cross-model), the frozen acceptance files, plus unit tests of the T1 and
fingerprint helpers. The real-data tests need the release, the EPTA fork env
(scripts/setup_epta_fork_env.sh) and the tempo2 env; they skip otherwise (fail in strict mode).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from ptagwb import fingerprint as fp
from ptagwb.config import REPO_ROOT
from ptagwb.t1 import compare

sys.path.insert(0, str(REPO_ROOT / "scripts"))

EF_ENV = Path(os.environ.get("EF_ENV", Path.home() / ".local" / "opt" / "epta-fork-env"))
T2_ENV = Path(os.environ.get("T2_ENV", Path.home() / ".local" / "opt" / "tempo2-env"))
BASE = REPO_ROOT / "data" / "processed" / "m3b" / "epta"


def _need_envs():
    if not (EF_ENV / "bin" / "python").exists() or not (T2_ENV / "bin" / "python").exists():
        pytest.skip("EPTA fork env / tempo2 env not installed")
    if not (BASE / "canonical" / "prepare.json").exists() or not (BASE / "t2runtime" / "runtime.json").exists():
        pytest.skip("run scripts/m3b_epta_prepare.py first")


# ---------------------------------------------------------------------- unit tests


def _export(n=50, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(5e9, 5.3e9, n))
    M = np.column_stack([np.ones(n), t - t.mean(), (t - t.mean()) ** 2 / 1e8])
    return {"name": np.array("Jtest"), "toas": t, "stoas": t - 400.0, "residuals": rng.normal(0, 1e-6, n),
            "toaerrs": np.full(n, 1e-6), "freqs": rng.uniform(1e3, 3e3, n), "backend_flags": np.array(["A"] * n),
            "flags_json": np.array(json.dumps({"group": ["A"] * n, "pta": ["EPTA"] * n})), "fitpars": np.array(["Offset", "F0", "F1"]),
            "Mmat": M, "pos": np.array([1.0, 0, 0]), "pos_t": np.ones((n, 3)), "sunssb": np.ones((n, 6)),
            "planetssb": np.ones((n, 9, 6)), "deleted": np.zeros(n, bool)}


def test_t1_compare_detects_each_difference():
    a = _export()
    assert compare(a, dict(a))["ok"]
    for key, mutate in [("residuals", lambda d: d["residuals"].__setitem__(3, d["residuals"][3] + 2e-12)),
                        ("toas", lambda d: d["toas"].__setitem__(0, d["toas"][0] + 2e-6)),
                        ("freqs", lambda d: d["freqs"].__setitem__(1, d["freqs"][1] * (1 + 1e-11)))]:
        b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
        mutate(b)
        r = compare(a, b)
        assert not r["ok"] and any(key in f for f in r["fail"]), (key, r["fail"])
    b = dict(a, Mmat=a["Mmat"] + np.outer(np.sin(np.arange(50.0)), [0, 1e-6 * a["Mmat"][:, 1].std(), 0]))
    assert any("column space" in f for f in compare(a, b)["fail"])
    b = dict(a, backend_flags=np.array(["A"] * 49 + ["B"]))
    assert any("backend" in f for f in compare(a, b)["fail"])
    fl = json.loads(str(a["flags_json"]))
    b = dict(a, flags_json=np.array(json.dumps(fl | {"to": ["0"] * 50, ".cal": [""] * 50})))
    r = compare(a, b)
    assert r["ok"] and r["notes"]  # listed canonicalisation rules only
    b = dict(a, flags_json=np.array(json.dumps(fl | {"pta": ["X"] * 50})))
    assert not compare(a, b)["ok"]
    b = dict(a, deleted=np.r_[True, np.zeros(49, bool)])
    assert any("deletion" in f for f in compare(a, b)["fail"])


def test_fingerprint_helpers():
    assert np.allclose(fp.stored_precision(["1.25", "-13.5000", "57510.1452873427800867", "3e-5"]),
                       [0.005, 0.00005, 0.5e-16, 0.5e-5])
    rng = np.random.default_rng(0)
    ref = rng.normal(5e5, 10, 60)
    ok = fp.evaluate(ref + 3.0 + rng.uniform(-5e-7, 5e-7, 60), ref, np.zeros(60))
    assert ok.passed and abs(ok.mean - 3.0) < 1e-6
    bad = fp.evaluate(ref + rng.normal(0, 1e-5, 60), ref, np.zeros(60))
    assert not bad.passed and bad.chi2 > 50
    with pytest.raises(ValueError):
        fp.evaluate(ref[:10], ref[:10], np.zeros(10))
    assert fp.discriminate(ok, {"alt": bad})["resolved"] is False  # chi2 50-100 is not decisive
    assert len(fp.spread_rows(29990, 7497, 60)) == 60


# ---------------------------------------------------------------------- real-data gates


@pytest.fixture(scope="module")
def epta_data():
    from ptagwb import epta

    try:
        man = epta.load_manifest()
        psrs = epta.load_pulsars()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return man, psrs


@pytest.mark.slow
@pytest.mark.oracle
def test_manifest_and_prior_volume():
    from ptagwb import epta

    if not (REPO_ROOT / "data" / "raw" / "epta_fork" / "epta_enterprise_extensions").exists():
        pytest.skip("EPTA fork clones not fetched (data/raw/epta_fork)")
    import m3b_manifest_epta as mm

    man = mm.generate()
    assert json.loads(epta.MANIFEST_PATH.read_text()) == json.loads(json.dumps(man)), "committed manifest is stale"
    assert len(man["pulsars"]) == 25 and man["toas"]["n_total"] == 45428
    assert man["common"]["modes"] == 9 and man["dip"]["chrom_idx"] == 1.0
    pv = mm.prior_volume_check(man)
    assert pv["pass"], pv


@pytest.mark.slow
@pytest.mark.oracle
def test_t1_complete_roster():
    _need_envs()
    import m3b_t1

    res = m3b_t1.run(jobs=8, reuse=False)
    assert res["roster_complete"]
    for k, c in res["comparisons"].items():
        assert c["n_pass"] == 25, (k, [r["fail"] for r in c["rows"] if r["fail"]])
    assert res["T1_pass"]


def test_dip_terms_reduce_to_stage1_without_dip(epta_data):
    from ptagwb import epta

    man, psrs = epta_data
    m = epta.EPTAModel(psrs, man, "crn")
    x = np.array([(q["bounds"][0] + q["bounds"][1]) / 2 for q in man["parameters"] if "crn" in q["models"]])
    ia = m._dip_idx[0]
    x[ia] = -60.0  # 10^-60 s: the dip vanishes
    c, s = m.dip_terms(x)
    t = m.terms[m.dip.index]
    assert np.max(np.abs(np.asarray(c) - t.c)) < 1e-9 * max(1.0, np.max(np.abs(t.c)))
    assert abs(float(s) - t.s_perp) < 1e-9 * t.s_perp
    # the dip changes the likelihood when it is on
    x2 = x.copy()
    x2[ia] = -5.9
    assert abs(float(m.logL(x2)) - float(m.logL(x))) > 1.0


@pytest.mark.slow
@pytest.mark.oracle
def test_fingerprint_gate():
    _need_envs()
    import m3b_fingerprint

    res = m3b_fingerprint.run(n=60, with_enterprise=True)
    for k, v in res["models"].items():
        assert v["pass"], (k, v["result"])
        assert v["discrimination"]["resolved"], (k, v["discrimination"])
        assert v["ours_vs_enterprise_max_abs"] < 1e-6
    assert res["c_diff_consistent"]


@pytest.mark.slow
@pytest.mark.oracle
def test_g5_pta():
    _need_envs()
    import m3b_g5

    out = m3b_g5.run()
    assert out["G5_PTA_pass"], {k: c["fails"] for k, c in out["configs"].items()}


def test_acceptance_files_frozen(epta_data):
    from ptagwb import acceptance as acc

    a = json.loads((REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json").read_text())
    assert a["frozen_before_any_production_run"] and a["version"] == 3
    ex = acc.d9_exclusions(a)  # validates every field (fail closed)
    assert len(ex) == 24 and a["d9"]["reference_draws_in_U"] == {"crn_pl": 38, "hd_pl": 78}
    assert all(set(e["params"]) == {"crn_pl", "hd_pl"} for e in ex)  # one common domain
    assert all(set(e["reference_cases"].values()) & {"zero-visit", "few-event"} for e in ex)
    assert a["d9"]["unconditional_verdict"] == "INCONCLUSIVE" and a["d9"]["template"] == acc.D9_TEMPLATE
    assert "epsilon" not in json.dumps(a["d9"]).replace("no epsilon_m", "")
    head = {(r["id"], r["quantile"]): r["headline"] for r in a["quantities"]}
    assert sum(head.values()) == 8 and not head[("E-2", 0.95)]
    for r in a["quantities"]:  # the uniform rule
        assert r["headline"] == (1.645 * r["mcse_ref_D"] <= r["m"] / 2 + 1e-15)
    assert a["headline"][-1] == "E-6" and len(a["headline"]) == 9
    e6 = a["E6"]  # same-domain reference; the published BF is context only
    assert e6["reference_lnB_D"] == pytest.approx(4.22451, abs=1e-4) and e6["mcse_ref"] == pytest.approx(0.0371, abs=1e-3)
    assert "target_lnbf" not in e6 and e6["context_not_target"]["published_lnBF_unrestricted"] == pytest.approx(np.log(60))
    assert a["E5"]["role"].startswith("context only")
    import m3b_freeze_acceptance as fa

    acc_new, rel_new = fa.build()
    assert json.loads(json.dumps(acc_new)) == a, "acceptance file differs from a fresh generation"
    assert json.loads(json.dumps(rel_new)) == json.loads(fa.REL.read_text())


@pytest.mark.slow
def test_conditional_acceptance_end_to_end_on_the_frozen_schema(epta_data):
    """The complete conditional path on the actual frozen acceptance file: common-domain
    indicators on ordered chains, conditional CURN quantiles, HD quantiles and ln B_D by
    reweighting (weights w I_D, paired MCSE), classification of every headline row with
    ``classify_from_frozen`` / ``classify_e6_from_frozen``, and D9 eligibility with the file's
    inventories. 'Ours' = the released CURN chain split into 4 contiguous chains (a schema test;
    the CURN rows must come out EQUIVALENT against themselves)."""
    from ptagwb import acceptance as acc
    from ptagwb import epta
    from ptagwb import reweight as rw

    man, _ = epta_data
    a = json.loads((REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json").read_text())
    ex = acc.d9_exclusions(a)
    npz = REPO_ROOT / a["E6"]["source"]["file"]
    if not npz.exists():
        pytest.skip("reference_weights.npz missing (scripts/m3b_reference_weights.py)")
    names, X, burn = epta.load_reference("crn_pl", man)
    z = np.load(npz)
    rows = z["curn_rows_thin"]
    R = X[rows, :67]
    lw = z["lnl_hd_at_crn"] - z["lnl_crn"]
    k = 4
    split = lambda v: np.array_split(v, k)  # noqa: E731
    col = lambda n: split(R[:, names.index(n)])  # noqa: E731
    ind = acc.domain_indicator(col, ex, "crn_pl")
    status = {}
    for row in a["quantities"]:
        if not row["headline"]:
            continue
        p = row["param"].replace("gw_hd_", "gw_crn_")
        x = col(p)
        r = acc.conditional_quantile(x, ind, row["quantile"], log_w=split(lw) if row["model"] == "hd_pl" else None)
        c = acc.classify_from_frozen(row, r["q"], r["mcse"])
        status[f"{row['id']} q{row['quantile']}"] = c.status
        if row["model"] == "crn_pl":
            assert c.status == acc.EQUIVALENT, (row["id"], row["quantile"], c)
    assert set(status) == set(a["headline"][:-1])
    ar = rw.accept_reweighting(split(lw), {"gw_crn_log10_A": col("gw_crn_log10_A")}, (0.5,),
                               {("gw_crn_log10_A", 0.5): None}, mask=ind, n_boot=200)
    assert ar["lnbf"]["value"] == pytest.approx(a["E6"]["reference_lnB_D"], abs=1e-12)
    se = max(ar["lnbf"]["mcse_obm"], ar["lnbf"]["mcse_bootstrap"])
    status["E-6"] = acc.classify_e6_from_frozen(a["E6"], ar["lnbf"]["value"], se).status
    assert status["E-6"] == acc.EQUIVALENT
    res = acc.d9_eligibility({c: acc.PASS for c in a["d9"]["required_checks"]}, status,
                             required_checks=a["d9"]["required_checks"], required_headline=a["headline"],
                             excluded_draws={"crn_pl": {"reference": 38, "ours": int(sum((~m).sum() for m in ind))},
                                             "hd_pl": {"reference": 78, "ours": 0}},
                             required_models=a["d9"]["models"])
    assert res.unconditional_verdict == acc.INCONCLUSIVE
    assert res.eligible == all(v == acc.EQUIVALENT for v in status.values())

@pytest.mark.slow
def test_buckets_and_hh_reducer_are_exact(epta_data):
    """N8 size buckets with the structured-Householder reducer (the benchmarked production
    configuration) equal the padded M1 reducer: value to rounding, gradients to 1e-11 relative."""
    from ptagwb import epta

    man, psrs = epta_data
    names, X, burn = epta.load_reference("crn_pl", man)
    for orf in ("crn", "hd"):
        base = epta.EPTAModel(psrs, man, orf)
        fast = epta.EPTAModel(psrs, man, orf, reduce="hh", buckets=epta.BUCKETS)
        for x in X[[burn, burn + 9000], :67]:
            v0, g0 = base.value_and_grad(x)
            v1, g1 = fast.value_and_grad(x)
            assert abs(float(v1 - v0)) < 1e-8
            g0, g1 = np.asarray(g0), np.asarray(g1)
            assert np.max(np.abs(g1 - g0) / np.maximum(1.0, np.abs(g0))) < 1e-11


# ---------------------------------------------------------------------- fail-closed negatives


def test_fingerprint_overall_pass_requires_every_predicate():
    good = {"pass": True, "t0_margin_ok": True, "discrimination": {"resolved": True}}
    assert fp.overall_pass({"crn_pl": good, "hd_pl": good}, True)
    assert not fp.overall_pass({"crn_pl": good, "hd_pl": {**good, "discrimination": {"resolved": False}}}, True)
    assert not fp.overall_pass({"crn_pl": good, "hd_pl": {**good, "discrimination": {}}}, True)
    assert not fp.overall_pass({"crn_pl": good}, False)
    assert not fp.overall_pass({"crn_pl": good}, None)
    assert not fp.overall_pass({"crn_pl": {**good, "t0_margin_ok": None}}, True)
    assert not fp.overall_pass({}, True)


def test_binding_rejects_stale_missing_and_failed_results(tmp_path):
    from ptagwb.binding import StaleEvidenceError, require_bound

    cur = {"source": "a", "scripts": "b", "configs": "c", "runtime": "d", "exports": "e", "n_exports": 25}
    (tmp_path / "ok.json").write_text(json.dumps({"pass": True, "binding": cur}))
    assert require_bound({"ok.json": "pass"}, tmp_path, cur)
    (tmp_path / "stale.json").write_text(json.dumps({"pass": True, "binding": {**cur, "source": "old"}}))
    (tmp_path / "failed.json").write_text(json.dumps({"pass": False, "binding": cur}))
    (tmp_path / "unbound.json").write_text(json.dumps({"pass": True}))
    (tmp_path / "nonbool.json").write_text(json.dumps({"pass": "yes", "binding": cur}))
    for f in ("stale.json", "failed.json", "unbound.json", "nonbool.json", "absent.json"):
        with pytest.raises(StaleEvidenceError):
            require_bound({"ok.json": "pass", f: "pass"}, tmp_path, cur)


def test_fingerprint_inventory_and_discrimination_negatives():
    good = {"pass": True, "t0_margin_ok": True, "discrimination": {"resolved": True}}
    assert not fp.overall_pass({"crn_pl": good}, True)  # HD omitted
    assert not fp.overall_pass({"crn_pl": good, "hd_pl": good, "extra": good}, True)
    rng = np.random.default_rng(1)
    ref = rng.normal(5e5, 10, 60)
    ok = fp.evaluate(ref + rng.uniform(-5e-7, 5e-7, 60), ref, np.zeros(60))
    bad = fp.evaluate(ref + rng.normal(0, 1.0, 60), ref, np.zeros(60))
    assert not fp.discriminate(ok, {})["resolved"]  # empty alternatives
    assert not fp.discriminate(ok, {"common_modes_8": bad})["resolved"]  # incomplete inventory
    assert fp.discriminate(ok, {"common_modes_8": bad, "common_modes_10": bad})["resolved"]
