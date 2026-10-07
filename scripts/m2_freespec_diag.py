"""Free-spectrum acceptance gate for M2 (CLI for ``ptagwb.diagnostics.freespec_gate``).

    uv run --no-sync python scripts/m2_freespec_diag.py [--run hd_fs30] [--released hd_fs30 | --released '']

Verdicts, reported separately (thresholds in ``diagnostics.GATE_DEFAULTS``):

* **convergence** PASS/FAIL: every one of the 164 parameters (30 free-spectrum bins and the 134
  IRN parameters) has rank-normalised split R-hat < 1.01 and bulk/tail ESS >= 400; every bin's
  occupancy indicator 1[log10_rho < -9], when not near-constant, has R-hat < 1.01 and ESS >= 400.
* **reproduction agreement** PASS/FAIL/UNAVAILABLE: per-bin occupancy vs the released core,
  |z| <= 3.5 with conservative, never-zero standard errors (the larger of a batch-means SE, our
  chains as units / contiguous batches of the reference, and the binomial SE from the indicator
  ESS; for near-constant indicators a Jeffreys-smoothed binomial SE with the ESS capped at the
  number of chains / batches); a conventional threshold, not a calibrated test. UNAVAILABLE only
  when no reference is requested (convergence-only mode). The reference's own diagnostics are printed
  and saved. Disagreement of a converged run is a finding about model / reference, not a sampler
  failure.
* **heuristic warnings**: tail-stability contrasts and near-constant indicators. Informational
  only; not calibrated; never affect the exit status.

Input contract (else exit 2): finite --threshold; the run's stored model states explicitly
orf = "hd", common = "freespec", n_common = 30 (no defaults); the run has numeric ``x`` and its parameter
names equal, in order, the names derived independently from that model spec and the 67-pulsar
GWB list of the release (par files minus ``data.EXPECTED_EXCLUDED``); names unique; all draws
finite; if a reference is requested, it is complete (all 30 bins) and finite.

Exit status: 0 = convergence PASS and agreement PASS (reproduction acceptance; with
``--released ''`` only convergence is required); 1 = diagnostic failure (which verdict failed is
printed); 2 = missing or invalid input (run, bins, reference). Writes
outputs/m2/freespec_gate_<run>.json when the inputs are valid.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from m2_common import ROOT, released, save_json

from ptagwb.data import EXPECTED_EXCLUDED, find_par_tim
from ptagwb.diagnostics import freespec_gate, gate_exit_code
from ptagwb.sampling import ModelSpec, load_run, parameter_names

N_BINS = 30


def expected_names() -> list[str]:
    """The M2 HD^free schema, derived from the model spec and the release's pulsar list (not from
    the run being checked)."""
    psrs = sorted(n for n in find_par_tim() if n not in EXPECTED_EXCLUDED)
    if len(psrs) != 67:
        raise RuntimeError(f"expected 67 GWB pulsars, found {len(psrs)}")
    return parameter_names(ModelSpec(orf="hd", common="freespec", n_common=N_BINS), psrs)


REQUIRED_MODEL = {"orf": "hd", "common": "freespec", "n_common": N_BINS}


def evaluate(run_name: str, reference_key: str, threshold: float = -9.0,
             load=load_run, load_reference=released) -> tuple[int, dict]:
    """(exit status, result). All loading, extraction and conversion happens inside the validation
    boundary, so missing or malformed inputs map to status 2 with a structured ``input_errors``.
    ``load`` / ``load_reference`` are injectable for tests."""

    def invalid(msg):
        return 2, {"input_errors": [msg], "convergence": "UNAVAILABLE", "reproduction_agreement": "UNAVAILABLE"}

    if not (isinstance(threshold, (int, float)) and math.isfinite(threshold)):
        return invalid(f"threshold must be finite, got {threshold!r}")
    try:
        run = load(run_name)
        if not isinstance(run, dict):
            return invalid(f"run {run_name!r}: loader returned {type(run).__name__}")
        meta = run.get("meta")
        model = meta.get("config", {}).get("model") if isinstance(meta, dict) else None
        if not isinstance(model, dict):
            return invalid(f"run {run_name!r}: no model specification in meta.config.model")
        # the stored model must state these explicitly: no defaulting (ModelSpec's default orf is CURN)
        for key, want in REQUIRED_MODEL.items():
            if key not in model:
                return invalid(f"run model lacks {key!r} (required: {want!r})")
            if model[key] != want:
                return invalid(f"run model {key} = {model[key]!r}, required {want!r}")
        if "x" not in run or "names" not in run:
            return invalid(f"run {run_name!r}: missing 'x' or 'names'")
        x = np.asarray(run["x"], dtype=np.float64)
        names = [str(n) for n in run["names"]]
        exp = expected_names()
        ref = None
        if reference_key:
            r = load_reference(reference_key)
            if not isinstance(r, dict):
                return invalid(f"reference {reference_key!r}: loader returned {type(r).__name__}")
            ref = {k: v for k, v in r.items() if str(k).startswith("gw_log10_rho_")}
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
        return invalid(f"{type(e).__name__}: {e}")
    g = freespec_gate(x, names, n_bins=N_BINS, threshold=threshold, reference=ref, expected_names=exp)
    return gate_exit_code(g, require_reproduction=bool(reference_key)), g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hd_fs30")
    ap.add_argument("--threshold", type=float, default=-9.0)
    ap.add_argument("--released", default="hd_fs30", help="released reference key ('' = convergence only)")
    args = ap.parse_args()
    code, g = evaluate(args.run, args.released, args.threshold)
    if g.get("input_errors"):
        print("GATE: INVALID INPUT (exit 2)")
        for e in g["input_errors"][:20]:
            print("  -", e)
        return 2
    print(f"{args.run}: occupancy = P(log10_rho < {args.threshold})")
    print("bin | per-chain occupancy | pooled | ref | ind R-hat | ind ESS | rho R-hat | bulk/tail ESS | agree z | conv")
    for n, b in g["bins"].items():
        s, st = g["parameters"][n], b["indicator"]
        pc = " ".join(f"{v:.2f}" for v in b["occupancy_per_chain"])
        ir = f"{st['rhat']:.3f}" if st["available"] else "n/a"
        ie = f"{st['ess']:.0f}" if st["available"] else "n/a"
        ag = b.get("agreement", {})
        print(f"{n.split('_')[-1]:>3} | {pc} | {b['occupancy']:.3f} | {ag.get('reference_occupancy', float('nan')):.3f} | "
              f"{ir} | {ie} | {s['rhat']:.3f} | {s['ess_bulk']:.0f}/{s['ess_tail']:.0f} | {ag.get('z', float('nan')):.1f} | "
              f"{'ok' if b['parameter_pass'] and b['indicator_pass'] else 'FAIL'}")
    if g["reference_diagnostics"]:
        rd = g["reference_diagnostics"].values()
        print(f"reference ({args.released}): max split R-hat {max(r['split_rhat'] for r in rd):.4f}, "
              f"min bulk/tail ESS {min(r['ess_bulk'] for r in rd):.0f}/{min(r['ess_tail'] for r in rd):.0f}")
    print(f"CONVERGENCE: {g['convergence']} ({g['n_parameters_failing']} parameters, "
          f"{g['n_bins_failing_convergence']} bins failing)")
    print(f"REPRODUCTION AGREEMENT: {g['reproduction_agreement']} ({g['n_bins_disagreeing']} bins disagreeing)")
    print(f"HEURISTIC WARNINGS (informational, not calibrated): {len(g['heuristic_warnings'])}")
    for f in (g["convergence_failures"] + g["agreement_failures"])[:15]:
        print("  -", f)
    save_json(g, ROOT / "outputs" / "m2" / f"freespec_gate_{args.run}.json")
    print(f"GATE exit status {code} ({'PASS' if code == 0 else 'FAIL'})")
    return code


if __name__ == "__main__":
    sys.exit(main())
