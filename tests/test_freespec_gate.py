"""Free-spectrum acceptance gate: separate convergence / reproduction-agreement verdicts, input
validation (the reviewer's repros), exit codes, and the existing (unconverged) hd_fs30 draws."""

from __future__ import annotations

import numpy as np
import pytest

from ptagwb.diagnostics import freespec_gate, gate_exit_code

NB, NIRN = 30, 134


def _bimodal(rng, size, w_low):
    """'power present' peak N(-7.5, 0.3) mixed with a low-power plateau U(-15.5, -9.5)."""
    low = rng.random(size) < w_low
    return np.where(low, rng.uniform(-15.5, -9.5, size), rng.normal(-7.5, 0.3, size))


def _synthetic(seed=0, C=8, N=1000):
    rng = np.random.default_rng(seed)
    names = [f"irn_{i}" for i in range(NIRN)] + [f"gw_log10_rho_{k}" for k in range(NB)]
    w = np.linspace(0.0, 0.9, NB)
    x = np.empty((C, N, NIRN + NB))
    x[..., :NIRN] = rng.normal(size=(C, N, NIRN))
    for k in range(NB):
        x[..., NIRN + k] = _bimodal(rng, (C, N), w[k])
    ref = {f"gw_log10_rho_{k}": _bimodal(rng, 20000, w[k]) for k in range(NB)}
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
    assert g["reference_diagnostics"]["gw_log10_rho_5"]["n"] == 20000


def test_stuck_chain_fails_convergence():
    x, names, ref = _synthetic(seed=1)
    x[0, :, NIRN + 20] = np.random.default_rng(2).normal(-7.5, 0.3, x.shape[1])  # never leaves the peak
    g = freespec_gate(x, names, reference=ref, expected_names=names)
    assert g["convergence"] == "FAIL" and not g["bins"]["gw_log10_rho_20"]["indicator_pass"]
    assert gate_exit_code(g) == 1


def test_converged_but_disagreeing_reference_is_an_agreement_failure_only():
    x, names, ref = _synthetic(seed=3)
    shifted = {k: v - 2.0 for k, v in ref.items()}  # reference moved, run unchanged
    g = freespec_gate(x, names, reference=shifted, expected_names=names)
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
    x, names, _ = _synthetic(seed=5, N=600)
    g = freespec_gate(x, names, expected_names=names)
    assert g["reproduction_agreement"] == "UNAVAILABLE"
    assert gate_exit_code(g, require_reproduction=True) == 2
    assert gate_exit_code(g, require_reproduction=False) == (0 if g["convergence"] == "PASS" else 1)


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
