"""Free-spectrum acceptance gate: separate convergence / reproduction-agreement verdicts, input
validation (the reviewer's repros), exit codes, the fail-closed region criteria (relevance,
entries / exits / sojourns, occupancy MCSE), and the existing (unconverged) hd_fs30 draws."""

from __future__ import annotations

import numpy as np
import pytest

from ptagwb.diagnostics import (
    GATE_DEFAULTS,
    derive_relevant_regions,
    freespec_gate,
    gate_exit_code,
    region_events,
    region_states,
)

NB, NIRN = 30, 134


def _bimodal(rng, size, w_low):
    """'power present' peak N(-7.5, 0.3) mixed with a low-power plateau U(-15.5, -9.5)."""
    low = rng.random(size) < w_low
    return np.where(low, rng.uniform(-15.5, -9.5, size), rng.normal(-7.5, 0.3, size))


def _synthetic(seed=0, C=8, N=3000):
    rng = np.random.default_rng(seed)
    names = [f"irn_{i}" for i in range(NIRN)] + [f"gw_log10_rho_{k}" for k in range(NB)]
    w = np.linspace(0.0, 0.9, NB)
    x = np.empty((C, N, NIRN + NB))
    x[..., :NIRN] = rng.normal(size=(C, N, NIRN))
    for k in range(NB):
        x[..., NIRN + k] = _bimodal(rng, (C, N), w[k])
    ref = {f"gw_log10_rho_{k}": _bimodal(rng, 50000, w[k]) for k in range(NB)}
    return x, names, ref


def test_well_mixed_passes_both_verdicts():
    x, names, ref = _synthetic()
    g = freespec_gate(x, names, reference=ref, expected_names=names)
    assert g["input_errors"] == []
    assert g["convergence"] == "PASS", g["convergence_failures"][:3]
    assert g["reproduction_agreement"] == "PASS", g["agreement_failures"][:3]
    assert gate_exit_code(g) == 0
    # bin 0 has occupancy 0: its indicator diagnostics are unavailable, not precise
    assert g["bins"]["gw_log10_rho_0"]["indicator"]["available"] is False
    assert g["reference_diagnostics"]["gw_log10_rho_5"]["n"] == 50000
    # relevance derived from the reference: bin 0 has only the high region, later bins both
    assert g["relevance_source"].startswith("derived from reference")
    assert g["relevant_regions"]["gw_log10_rho_0"] == ["high"]
    assert g["relevant_regions"]["gw_log10_rho_5"] == ["low", "high"]
    r5 = g["regions"]["gw_log10_rho_5"]["low"]
    assert r5["pass"] and r5["transport_assessed"] and r5["entries"] >= 100 and r5["chains_with_events"] == 8
    assert g["n_regions_failing"] == 0 and g["no_exploration_evidence"] == []


def test_stuck_chain_fails_convergence():
    x, names, ref = _synthetic(seed=1)
    x[0, :, NIRN + 20] = np.random.default_rng(2).normal(-7.5, 0.3, x.shape[1])  # never leaves the peak
    g = freespec_gate(x, names, reference=ref, expected_names=names)
    assert g["convergence"] == "FAIL" and not g["bins"]["gw_log10_rho_20"]["indicator_pass"]
    assert gate_exit_code(g) == 1


def test_converged_but_disagreeing_reference_is_an_agreement_failure_only():
    x, names, ref = _synthetic(seed=3)
    shifted = {k: v - 2.0 for k, v in ref.items()}  # reference moved, run unchanged
    # relevance predeclared from the unshifted target (derived from the shifted reference, bin 0's
    # low region would be relevant and unvisited: a convergence FAIL, by design)
    g = freespec_gate(x, names, reference=shifted, expected_names=names, relevant_regions=derive_relevant_regions(ref))
    assert g["convergence"] == "PASS" and g["reproduction_agreement"] == "FAIL"
    assert gate_exit_code(g) == 1 and gate_exit_code(g, require_reproduction=False) == 0


@pytest.mark.parametrize("case", ["no_irn", "empty_ref", "wrong_ref", "inf_draw", "missing_bin", "duplicate"])
def test_invalid_inputs_are_rejected(case):
    x, names, ref = _synthetic(seed=4, N=200)
    expected = list(names)
    if case == "no_irn":  # the reviewer's repro: all 134 IRN parameters removed
        x, names = x[..., NIRN:], names[NIRN:]
    elif case == "empty_ref":
        ref = {}
    elif case == "wrong_ref":  # e.g. the released HD^13/3 chain: no free-spectrum bins
        ref = {"gw_log10_A": np.random.default_rng(0).normal(-14.6, 0.07, 5000)}
    elif case == "inf_draw":
        x = x.copy()
        x[0, 0, 3] = np.inf
    elif case == "missing_bin":
        x, names = x[..., :-1], names[:-1]
    elif case == "duplicate":
        names = names[:-1] + [names[0]]
    g = freespec_gate(x, names, reference=ref, expected_names=expected)
    assert g["input_errors"], case
    assert g["convergence"] == "UNAVAILABLE" and g["reproduction_agreement"] == "UNAVAILABLE"
    assert gate_exit_code(g) == 2


def test_no_reference_means_agreement_unavailable():
    x, names, ref = _synthetic(seed=5)
    g = freespec_gate(x, names, expected_names=names, relevant_regions=derive_relevant_regions(ref))
    assert g["reproduction_agreement"] == "UNAVAILABLE" and g["convergence"] == "PASS", g["convergence_failures"][:3]
    assert gate_exit_code(g, require_reproduction=True) == 2
    assert gate_exit_code(g, require_reproduction=False) == 0


def test_existing_hd_fs30_fails_convergence():
    from ptagwb.sampling import load_run

    try:
        run = load_run("hd_fs30")
    except FileNotFoundError:
        pytest.skip("runs/hd_fs30 not present (git-ignored production output)")
    g = freespec_gate(run["x"], run["names"])
    assert g["input_errors"] == [] and g["convergence"] == "FAIL"
    assert g["n_parameters_failing"] >= 50
    assert not g["parameters"]["gw_log10_rho_17"]["pass"] and not g["parameters"]["gw_log10_rho_19"]["pass"]
    assert not g["bins"]["gw_log10_rho_7"]["indicator_pass"]
    assert gate_exit_code(g, require_reproduction=False) == 1


def test_cli_schema_matches_production_run():
    """The independently derived 164-name schema equals the production run's names."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        from m2_freespec_diag import evaluate, expected_names

        from ptagwb.sampling import load_run

        run = load_run("hd_fs30")
        names = expected_names()
    except FileNotFoundError:
        pytest.skip("release or runs/hd_fs30 not present")
    assert len(names) == 164 and run["names"] == names
    code, g = evaluate("hd_fs30", "")
    assert code == 1 and g["convergence"] == "FAIL"
    assert evaluate("no_such_run", "")[0] == 2


# ---------------------------------------------------------------------- round-4 regressions


def test_sparse_bin_has_nonzero_se_and_agrees():
    """One low-power draw per chain in an otherwise zero-occupancy bin (reviewer's repro): both
    indicators are near-constant; the sparse-count SE must be > 0 and the bin must not fail."""
    x, names, ref = _synthetic(seed=0)
    x[:, 0, NIRN] = -10.0
    g = freespec_gate(x, names, reference=ref, expected_names=names)
    a = g["bins"]["gw_log10_rho_0"]["agreement"]
    assert a["se_ours"] > 0 and a["se_ref"] > 0 and np.isfinite(a["z"])
    assert g["convergence"] == "PASS" and g["reproduction_agreement"] == "PASS"
    assert gate_exit_code(g) == 0


def test_stored_run_zero_occupancy_bins_have_nonzero_se():
    from ptagwb.sampling import load_run

    try:
        run = load_run("hd_fs30")
    except FileNotFoundError:
        pytest.skip("runs/hd_fs30 not present")
    ref = {f"gw_log10_rho_{k}": run["x"][..., run["names"].index(f"gw_log10_rho_{k}")].ravel() for k in range(NB)}
    g = freespec_gate(run["x"], run["names"], reference=ref)
    zero = [n for n, b in g["bins"].items() if b["occupancy"] == 0.0]
    assert zero  # e.g. f_2, f_3 (bins 1, 2)
    for n in zero:
        assert g["bins"][n]["agreement"]["se_ours"] > 0 and np.isfinite(g["bins"][n]["agreement"]["z"])


@pytest.mark.parametrize("thr", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_threshold_is_invalid(thr):
    x, names, ref = _synthetic(seed=0, N=300)
    shifted = {k: v - 2.0 for k, v in ref.items()}
    g = freespec_gate(x, names, reference=shifted, expected_names=names, threshold=thr)
    assert g["input_errors"] and gate_exit_code(g) == 2


def _cli():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import m2_freespec_diag as cli

        names = cli.expected_names()
    except FileNotFoundError:
        pytest.skip("NG15 release not present")
    return cli, names


def _fixture_run(names):
    x, _, ref = _synthetic(seed=0)
    model = {"orf": "hd", "common": "freespec", "n_common": 30}
    return {"x": x, "names": list(names), "meta": {"config": {"model": model}}}, ref


@pytest.mark.parametrize("case", ["ok", "nan_threshold", "inf_threshold", "orf_curn", "orf_missing",
                                  "common_wrong", "n_common_wrong", "missing_x", "model_null",
                                  "nonnumeric_x", "loader_raises", "reference_not_dict"])
def test_cli_evaluate_validation(case):
    cli, names = _cli()
    run, ref = _fixture_run(names)
    thr = -9.0
    load_ref = lambda key: ref
    if case == "nan_threshold":
        thr = float("nan")
    elif case == "inf_threshold":
        thr = float("inf")
    elif case == "orf_curn":
        run["meta"]["config"]["model"]["orf"] = "curn"
    elif case == "orf_missing":
        del run["meta"]["config"]["model"]["orf"]
    elif case == "common_wrong":
        run["meta"]["config"]["model"]["common"] = "powerlaw"
    elif case == "n_common_wrong":
        run["meta"]["config"]["model"]["n_common"] = 14
    elif case == "missing_x":
        del run["x"]
    elif case == "model_null":
        run["meta"]["config"]["model"] = None
    elif case == "nonnumeric_x":
        run["x"] = np.full(run["x"].shape, "a", dtype=object)
    elif case == "reference_not_dict":
        load_ref = lambda key: [1, 2, 3]

    def load(name):
        if case == "loader_raises":
            raise FileNotFoundError(name)
        return run

    code, g = cli.evaluate("fixture", "ref", thr, load=load, load_reference=load_ref)
    if case == "ok":
        assert code == 0, (g.get("input_errors"), g.get("convergence_failures", [])[:3], g.get("agreement_failures"))
    else:
        assert code == 2 and g["input_errors"], case


# ---------------------------------------------------------------------- round-5 regressions


BAD_VALUES = [np.float32("inf"), np.float64("inf"), np.float32("nan"), np.float64("nan"),
              -np.float64("inf"), "3.5", None, True, np.bool_(True), [1.0]]


@pytest.fixture(scope="module")
def shifted_fixture():
    x, names, ref = _synthetic(seed=0, N=300)
    return x, names, {k: v - 2.0 for k, v in ref.items()}  # agreement would FAIL if evaluated


@pytest.mark.parametrize("key", [*GATE_DEFAULTS, "threshold", "n_bins"])
@pytest.mark.parametrize("bad", BAD_VALUES, ids=repr)
def test_every_criterion_rejects_invalid_values(shifted_fixture, key, bad):
    x, names, ref = shifted_fixture
    kw = {key: bad}
    g = freespec_gate(x, names, reference=ref, expected_names=names, **kw)
    assert g["input_errors"] and gate_exit_code(g) == 2, (key, bad)
    assert g["convergence"] == "UNAVAILABLE"


def test_numpy_scalar_criteria_accepted_and_unknown_keys_rejected(shifted_fixture):
    x, names, ref = shifted_fixture
    g = freespec_gate(x, names, reference=ref, expected_names=names, agreement_z_tol=np.float32(3.5),
                      min_minority_count=np.int64(10), threshold=np.float64(-9.0))
    assert not g["input_errors"] and g["reproduction_agreement"] == "FAIL"
    g = freespec_gate(x, names, reference=ref, expected_names=names, agreement_ztol=100.0)
    assert g["input_errors"] and gate_exit_code(g) == 2
    for probs in ((0.05, 1.5), ("0.5",), ()):
        g = freespec_gate(x, names, reference=ref, expected_names=names, probs=probs)
        assert g["input_errors"] and gate_exit_code(g) == 2


# ---------------------------------------------------------------------- fail-closed region criteria


def _small(seed=0, C=8, N=750, w=(0.05, 0.1, 0.0077), nirn=4):
    """Few-parameter i.i.d. fixture (passes parameter R-hat / ESS): bins 0..len(w)-1 bimodal with
    low-plateau weight w[k]; reference 50000 i.i.d. draws per bin."""
    rng = np.random.default_rng(seed)
    nb = len(w)
    names = [f"irn_{i}" for i in range(nirn)] + [f"gw_log10_rho_{k}" for k in range(nb)]
    x = np.empty((C, N, nirn + nb))
    x[..., :nirn] = rng.normal(size=(C, N, nirn))
    for k in range(nb):
        x[..., nirn + k] = _bimodal(rng, (C, N), w[k])
    ref = {f"gw_log10_rho_{k}": _bimodal(rng, 50000, w[k]) for k in range(nb)}
    return x, names, ref, nb, nirn


def test_region_event_counting_on_hand_made_sequence():
    #       undef  low   (keep) high  (keep) low   low   high  (keep) high
    seq = [-9.0, -11.0, -9.5, -7.0, -9.0, -12.0, -12.0, -7.5, -8.5, -6.0]
    assert region_states(np.array(seq), -10.0, -8.0).tolist() == [[-1, 0, 0, 1, 1, 0, 0, 1, 1, 1]]
    ev = region_events(np.array([seq, [-9.0] * 10]), -10.0, -8.0)  # second chain: never enters a region
    lo, hi = ev["low"], ev["high"]
    assert lo["entries_per_chain"] == [1, 0] and lo["exits_per_chain"] == [2, 0]  # first entry from undefined not counted
    assert hi["entries_per_chain"] == [2, 0] and hi["exits_per_chain"] == [1, 0]
    assert lo["sojourn_lengths"] == [2, 2] and hi["sojourn_lengths"] == [2, 3]
    assert lo["draws_in_state"] == 4 and hi["draws_in_state"] == 5
    assert lo["chains_with_events"] == 1 and hi["chains_with_events"] == 1
    assert hi["sojourns"] == {"count": 2, "mean": 2.5, "max": 3, "max_frac": 0.6}
    assert lo["sojourns"]["max_frac"] == 0.5


def test_unvisited_relevant_region_fails_closed():
    """Reviewer's loophole: 8 x 750 draws that never enter a region the reference puts 0.7% in.
    Parameter R-hat / ESS pass, but convergence must FAIL with 'no exploration evidence'."""
    x, names, ref, nb, nirn = _small()
    rng = np.random.default_rng(7)
    x[..., nirn + 2] = rng.normal(-7.5, 0.3, x.shape[:2])  # bin 2: never below -10
    m = 50000
    r = rng.normal(-7.5, 0.3, m)
    r[: int(0.007 * m)] = rng.uniform(-15.5, -10.5, int(0.007 * m))
    ref["gw_log10_rho_2"] = rng.permutation(r)
    g = freespec_gate(x, names, n_bins=nb, reference=ref, expected_names=names)
    assert g["parameters"]["gw_log10_rho_2"]["pass"]  # the ordinary diagnostics are fooled
    assert g["relevant_regions"]["gw_log10_rho_2"] == ["low", "high"]
    assert g["convergence"] == "FAIL" and gate_exit_code(g, require_reproduction=False) == 1
    assert "gw_log10_rho_2/low" in g["no_exploration_evidence"]
    assert any(f.startswith("gw_log10_rho_2 region low") and "no exploration evidence" in f
               for f in g["convergence_failures"])
    assert all("gw_log10_rho_2" in f for f in g["convergence_failures"]), g["convergence_failures"]
    assert not g["bins"]["gw_log10_rho_2"]["indicator_pass"]  # unavailable < -9 indicator: FAIL, not a pass
    # the same run passes when bin 2's low region is (explicitly) declared irrelevant
    g2 = freespec_gate(x, names, n_bins=nb, expected_names=names,
                       relevant_regions={"gw_log10_rho_0": ["low", "high"], "gw_log10_rho_1": ["low", "high"],
                                         "gw_log10_rho_2": ["high"]})
    assert g2["convergence"] == "PASS", g2["convergence_failures"]
    assert g2["relevance_source"] == "declared"


def test_single_long_sojourn_fails():
    """A relevant region visited only by one long excursion of one chain carries no transport
    evidence: chains-with-events and sojourn criteria fail."""
    x, names, ref, nb, nirn = _small(w=(0.05, 0.1, 0.05))
    rng = np.random.default_rng(3)
    v = rng.normal(-7.5, 0.3, x.shape[:2])
    v[0, 200:500] = rng.uniform(-15.5, -10.5, 300)  # one 300-draw excursion in chain 0 (5% of all draws)
    x[..., nirn + 2] = v
    g = freespec_gate(x, names, n_bins=nb, reference=ref, expected_names=names)
    rr = g["regions"]["gw_log10_rho_2"]["low"]
    assert rr["entries"] == 1 and rr["exits"] == 1 and rr["chains_with_events"] == 1
    assert rr["sojourns"]["max_frac"] == 1.0 and rr["pass"] is False
    msg = next(f for f in g["convergence_failures"] if f.startswith("gw_log10_rho_2 region low"))
    assert "longest sojourn" in msg and "chains with an entry/exit 1 < 2" in msg and "entries 1 < 10" in msg
    assert g["convergence"] == "FAIL"


def test_undeclared_relevance_is_inconclusive():
    x, names, ref, nb, _ = _small()
    g = freespec_gate(x, names, n_bins=nb, expected_names=names)
    assert g["convergence_failures"] == [] and g["convergence"] == "INCONCLUSIVE"
    assert g["relevance_source"] == "undeclared" and g["relevant_regions"] is None
    assert "not declared" in g["convergence_inconclusive"][0]
    assert gate_exit_code(g, require_reproduction=False) == 1
    # declaring it (here derived from the reference, labelled) lets the same run pass
    rel = derive_relevant_regions(ref, nb)
    g = freespec_gate(x, names, n_bins=nb, expected_names=names, relevant_regions=rel, relevance_source="test ref")
    assert g["convergence"] == "PASS" and g["relevance_source"] == "test ref"
    assert gate_exit_code(g, require_reproduction=False) == 0


def test_imprecise_agreement_is_inconclusive():
    x, names, ref, nb, _ = _small()
    rel = derive_relevant_regions(ref, nb)
    # our run too short for precision (8 x 50 draws), reference precise: INCONCLUSIVE
    g = freespec_gate(x[:, :50], names, n_bins=nb, reference=ref, expected_names=names, relevant_regions=rel)
    assert g["agreement_failures"] == [] and g["agreement_imprecise"]
    assert all("our SE" in m for m in g["agreement_imprecise"])
    assert g["reproduction_agreement"] == "INCONCLUSIVE" and gate_exit_code(g) == 1
    # a large |z| still makes it FAIL, even when imprecise
    g = freespec_gate(x[:, :50], names, n_bins=nb, reference={k: v - 2.0 for k, v in ref.items()},
                      expected_names=names, relevant_regions=rel)
    assert g["reproduction_agreement"] == "FAIL"
    # our run precise, reference imprecise (300 draws): PASS, the reference's imprecision flagged
    short = {k: v[:300] for k, v in ref.items()}
    g = freespec_gate(x, names, n_bins=nb, reference=short, expected_names=names, relevant_regions=rel)
    assert g["convergence"] == "PASS" and g["agreement_imprecise"] == []
    assert g["reproduction_agreement"] == "PASS", g["agreement_failures"]
    assert g["n_reference_imprecise"] > 0 and g["max_reference_se"]["se"] > GATE_DEFAULTS["agreement_se_max"]
    flagged = [rr["agreement"]["reference_imprecise"] for rb in g["regions"].values() for rr in rb.values()
               if "agreement" in rr] + [bb["agreement"]["reference_imprecise"] for bb in g["bins"].values()]
    assert any(flagged)
    assert gate_exit_code(g) == 0
    # precise reference: PASS, nothing flagged
    g = freespec_gate(x, names, n_bins=nb, reference=ref, expected_names=names)
    assert g["reproduction_agreement"] == "PASS", (g["agreement_failures"], g["agreement_imprecise"])
    assert g["n_reference_imprecise"] == 0 and gate_exit_code(g) == 0


@pytest.mark.parametrize("kw", [
    {"region_low": -8.5}, {"region_high": -9.5}, {"region_low": -9.0}, {"threshold": -7.0},
    {"region_min_events": 0}, {"region_min_events": 2.5}, {"region_min_chains": 0},
    {"region_min_mass": 0.0}, {"region_min_mass": 1.5}, {"region_max_sojourn_frac": 0.0},
    {"region_max_sojourn_frac": 1.01}, {"occupancy_mcse_max": 0.0}, {"occupancy_mcse_max": -0.01},
    {"agreement_se_max": 0.0},
    {"relevant_regions": [["low"]]}, {"relevant_regions": {"gw_log10_rho_0": ["low"]}},
    {"relevant_regions": {**{f"gw_log10_rho_{k}": [] for k in range(3)}, "gw_log10_rho_9": []}},
    {"relevant_regions": {**{f"gw_log10_rho_{k}": [] for k in range(3)}, "gw_log10_rho_1": ["mid"]}},
    {"relevant_regions": {**{f"gw_log10_rho_{k}": [] for k in range(3)}, "gw_log10_rho_1": ["low", "low"]}},
    {"relevant_regions": {**{f"gw_log10_rho_{k}": [] for k in range(3)}, "gw_log10_rho_1": "low"}},
    {"relevance_source": 3},
], ids=repr)
def test_new_criteria_and_relevance_validation(kw):
    x, names, ref, nb, _ = _small(N=100)
    g = freespec_gate(x, names, n_bins=nb, reference=ref, expected_names=names, **kw)
    assert g["input_errors"] and g["convergence"] == "UNAVAILABLE" and gate_exit_code(g) == 2, kw


def test_new_criteria_valid_values_accepted():
    x, names, ref, nb, _ = _small(N=100)
    g = freespec_gate(x, names, n_bins=nb, reference=ref, expected_names=names, region_low=np.float64(-12.0),
                      region_high=-7, region_min_mass=1.0, region_max_sojourn_frac=1.0, region_min_events=np.int64(1),
                      region_min_chains=1, occupancy_mcse_max=0.5, agreement_se_max=0.5,
                      relevant_regions={f"gw_log10_rho_{k}": [] for k in range(nb)})
    assert g["input_errors"] == [] and g["criteria"]["region_low"] == -12.0 and g["criteria"]["region_min_events"] == 1
    assert g["relevance_source"] == "declared"


def test_existing_hd_fs30_fails_on_unvisited_f3_low_region():
    """M2 run hd_fs30: zero draws below -10 at gw_log10_rho_2 (f_3) while the released core has
    0.7% there -> convergence FAIL naming that bin and region (relevance from the released core,
    also in the CLI's convergence-only mode)."""
    cli, _ = _cli()
    try:
        from m2_common import released

        from ptagwb.sampling import load_run

        run = load_run("hd_fs30")
        ref = {k: v for k, v in released("hd_fs30").items() if k.startswith("gw_log10_rho_")}
    except FileNotFoundError:
        pytest.skip("runs/hd_fs30 or the released core not present")
    g = freespec_gate(run["x"], run["names"], reference=ref)
    assert g["convergence"] == "FAIL" and "gw_log10_rho_2/low" in g["no_exploration_evidence"]
    assert any(f.startswith("gw_log10_rho_2 region low") and "no exploration evidence" in f
               for f in g["convergence_failures"])
    assert g["reference_diagnostics"]["gw_log10_rho_2"]["region_mass"]["low"] > 0.005
    code, g2 = cli.evaluate("hd_fs30", "")  # convergence only; relevance still from the released core
    assert code == 1 and "gw_log10_rho_2/low" in g2["no_exploration_evidence"]
    assert g2["relevance_source"].startswith("derived from released hd_fs30")
    code, g3 = cli.evaluate("hd_fs30", "", relevance_key="")  # undeclared: still not a PASS
    assert code == 1 and g3["relevance_source"] == "undeclared"
