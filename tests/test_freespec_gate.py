"""Free-spectrum acceptance gate: passes on synthetic well-mixed draws, fails on stuck chains,
missing bins and on the existing (unconverged) hd_fs30 production draws."""

from __future__ import annotations

import numpy as np
import pytest

from ptagwb.diagnostics import freespec_gate

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
    rel = {f"gw_log10_rho_{k}": _bimodal(rng, 20000, w[k]) for k in range(NB)}
    return x, names, rel


def test_gate_passes_well_mixed():
    x, names, rel = _synthetic()
    g = freespec_gate(x, names, released=rel)
    assert g["pass"], g["failures"][:5]
    assert g["n_bins_failing"] == 0 and len(g["bins"]) == NB


def test_gate_fails_stuck_chain_missing_bin_and_short_run():
    x, names, rel = _synthetic(seed=1)
    stuck = x.copy()
    k = NIRN + 20  # occupancy ~0.6: chain 0 never leaves the 'power present' peak
    stuck[0, :, k] = np.random.default_rng(2).normal(-7.5, 0.3, x.shape[1])
    g = freespec_gate(stuck, names, released=rel)
    assert not g["pass"] and not g["bins"]["gw_log10_rho_20"]["pass"]
    assert not g["bins"]["gw_log10_rho_20"]["checks"]["indicator"]
    assert not g["bins"]["gw_log10_rho_20"]["checks"]["tail_stability"]
    g = freespec_gate(x[..., :-1], names[:-1], released=rel)  # bin 29 missing
    assert not g["pass"] and g["bins"]["gw_log10_rho_29"].get("missing")
    g = freespec_gate(x[:, :40], names, released=rel)  # too few draws: ESS minimums fail
    assert not g["pass"] and g["n_parameters_failing"] > 0


def test_gate_fails_on_existing_hd_fs30():
    from ptagwb.sampling import load_run

    try:
        run = load_run("hd_fs30")
    except FileNotFoundError:
        pytest.skip("runs/hd_fs30 not present (git-ignored production output)")
    g = freespec_gate(run["x"], run["names"])
    assert not g["pass"]
    assert g["n_bins_failing"] >= 10
    # the review's examples: bin 17/19 parameters (R-hat 1.011 / 1.014) and bin 8 occupancy
    assert not g["bins"]["gw_log10_rho_17"]["pass"] and not g["bins"]["gw_log10_rho_19"]["pass"]
    assert not g["bins"]["gw_log10_rho_7"]["checks"]["indicator"]
