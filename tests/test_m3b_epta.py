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
    assert a["frozen_before_any_production_run"]
    regs = acc.d9_excluded_regions(a)  # validates every field (fail closed)
    assert {r["model"] for r in regs} == {"crn_pl", "hd_pl"}
    assert all(r["reference_case"] == "zero-visit" for r in regs)
    assert a["d9"]["unconditional_verdict"] == "INCONCLUSIVE" and a["d9"]["template"] == acc.D9_TEMPLATE
    dec = {(r["id"], r["quantile"]): r["decidable"] for r in a["quantities"]}
    assert not dec[("E-1", 0.05)] and not dec[("E-3", 0.05)] and not dec[("E-4", 0.95)]
    assert sum(dec.values()) == 9
    import m3b_freeze_acceptance as fa

    acc_new, rel_new = fa.build()
    assert json.loads(json.dumps(acc_new)) == a, "acceptance file differs from a fresh generation"
    assert json.loads(json.dumps(rel_new)) == json.loads(fa.REL.read_text())


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
