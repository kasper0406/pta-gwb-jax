"""Free-spectrum acceptance gate for M2 (CLI for ``ptagwb.diagnostics.freespec_gate``).

    uv run --no-sync python scripts/m2_freespec_diag.py [--run hd_fs30] [--released hd_fs30 | --released '']
                                                        [--relevance-ref hd_fs30 | --relevance-ref '']

The gate fails closed: a run that never explores a scientifically relevant region cannot pass.
Verdicts, reported separately (thresholds in ``diagnostics.GATE_DEFAULTS``):

* **region relevance** (predeclared): each bin has a "low" region log10_rho < -10 and a "high"
  region log10_rho > -8 (hysteresis state per chain, kept in between). A region is relevant when
  the released core (``--relevance-ref``, default hd_fs30, used also in convergence-only mode)
  puts >= 0.5% of its draws in it. ``--relevance-ref ''``: derived from ``--released`` if given,
  else undeclared, and convergence is then INCONCLUSIVE, never PASS.
* **convergence** PASS/FAIL/INCONCLUSIVE: every one of the 164 parameters (30 free-spectrum bins
  and the 134 IRN parameters) has rank-normalised split R-hat < 1.01 and bulk/tail ESS >= 400;
  every bin's occupancy 1[log10_rho < -9] has conservative MCSE <= 0.01 and, when not
  near-constant, indicator R-hat < 1.01 and ESS >= 400 (near-constant is a FAIL in a bin whose
  regions are both relevant, a warning otherwise); every relevant region needs >= 10 pooled
  entries and >= 10 exits, >= 2 chains with an entry or exit, no single sojourn holding > 50% of
  its draws, an available region indicator 1[x in R] with R-hat < 1.01 and ESS >= 400, and
  occupancy MCSE <= 0.01 (a bin's only relevant region: >= 2 chains visiting it, MCSE, indicator
  R-hat/ESS when available). Zero draws or zero events in a relevant region: FAIL, "no
  exploration evidence".
* **reproduction agreement** PASS/FAIL/INCONCLUSIVE/UNAVAILABLE: per bin the < -9 occupancy and
  the occupancy of each relevant region vs the released core, z = diff / sqrt(SE_ours^2 +
  SE_ref^2) with conservative, never-zero standard errors (the larger of a batch-means SE, our
  chains as units / contiguous batches of the reference, and the binomial SE from the indicator
  ESS; for near-constant indicators a Jeffreys-smoothed binomial SE with the ESS capped at the
  number of chains / batches). FAIL if any |z| > 3.5; else INCONCLUSIVE if our SE of any compared
  occupancy > 0.01; else PASS. The reference SE enters z only; comparisons where it exceeds 0.01
  are flagged and the max reference SE is printed (the released core's own occupancy SE, up to
  ~0.010, limits how precisely agreement can be established). A conventional threshold, not a calibrated test. UNAVAILABLE only when no reference is
  requested (convergence-only mode). The reference's own diagnostics (incl. whether the single
  sequence meets the region event criteria) are printed and saved. Disagreement of a converged run
  is a finding about model / reference, not a sampler failure.
* **heuristic warnings**: tail-stability contrasts and near-constant indicators. Informational
  only; not calibrated; never affect the exit status.

Input contract (else exit 2): finite --threshold; the run's stored model states explicitly
orf = "hd", common = "freespec", n_common = 30 (no defaults); the run has numeric ``x`` and its parameter
names equal, in order, the names derived independently from that model spec and the 67-pulsar
GWB list of the release (par files minus ``data.EXPECTED_EXCLUDED``); names unique; all draws
finite; a requested reference and relevance reference are complete (all 30 bins) and finite.

Exit status: 0 = convergence PASS and agreement PASS (reproduction acceptance; with
``--released ''`` only convergence is required); 1 = a verdict is FAIL or INCONCLUSIVE (printed);
2 = missing or invalid input (run, bins, reference). Writes outputs/m2/freespec_gate_<run>.json
when the inputs are valid.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from m2_common import ROOT, released, save_json

from ptagwb.data import EXPECTED_EXCLUDED, find_par_tim
from ptagwb.diagnostics import (
    GATE_DEFAULTS,
    GateInputError,
    derive_relevant_regions,
    freespec_gate,
    gate_exit_code,
)
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


def _bins_only(r, key):
    if not isinstance(r, dict):
        raise GateInputError(f"reference {key!r}: loader returned {type(r).__name__}")
    return {k: v for k, v in r.items() if str(k).startswith("gw_log10_rho_")}


def evaluate(run_name: str, reference_key: str, threshold: float = -9.0,
             load=load_run, load_reference=released, relevance_key: str = "hd_fs30") -> tuple[int, dict]:
    """(exit status, result). All loading, extraction and conversion happens inside the validation
    boundary, so missing or malformed inputs map to status 2 with a structured ``input_errors``.
    Region relevance is derived from ``relevance_key`` (a released chain; '' = from the reference
    if one is given, else undeclared). ``load`` / ``load_reference`` are injectable for tests."""

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
        ref = _bins_only(load_reference(reference_key), reference_key) if reference_key else None
        rel, source = None, None
        if relevance_key and relevance_key != reference_key:
            rel = derive_relevant_regions(_bins_only(load_reference(relevance_key), relevance_key), N_BINS)
            source = (f"derived from released {relevance_key} "
                      f"(reference mass >= {GATE_DEFAULTS['region_min_mass']})")
        elif relevance_key:
            source = (f"derived from released {relevance_key} "
                      f"(reference mass >= {GATE_DEFAULTS['region_min_mass']})")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as e:
        return invalid(f"{type(e).__name__}: {e}")
    g = freespec_gate(x, names, n_bins=N_BINS, threshold=threshold, reference=ref, expected_names=exp,
                      relevant_regions=rel, relevance_source=source)
    return gate_exit_code(g, require_reproduction=bool(reference_key)), g


def _f(v, fmt):
    return "n/a" if v is None else format(v, fmt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hd_fs30")
    ap.add_argument("--threshold", type=float, default=-9.0)
    ap.add_argument("--released", default="hd_fs30", help="released reference key ('' = convergence only)")
    ap.add_argument("--relevance-ref", default="hd_fs30",
                    help="released chain from which region relevance is derived ('' = from --released if given, "
                         "else undeclared -> convergence INCONCLUSIVE)")
    args = ap.parse_args()
    code, g = evaluate(args.run, args.released, args.threshold, relevance_key=args.relevance_ref)
    if g.get("input_errors"):
        print("GATE: INVALID INPUT (exit 2)")
        for e in g["input_errors"][:20]:
            print("  -", e)
        return 2
    print(f"{args.run}: occupancy = P(log10_rho < {args.threshold})")
    print("bin | per-chain occupancy | pooled | MCSE | ref | ind R-hat | ind ESS | rho R-hat | bulk/tail ESS | agree z | conv")
    for n, b in g["bins"].items():
        s, st = g["parameters"][n], b["indicator"]
        pc = " ".join(f"{v:.2f}" for v in b["occupancy_per_chain"])
        ag = b.get("agreement", {})
        print(f"{n.split('_')[-1]:>3} | {pc} | {b['occupancy']:.3f} | {b['occupancy_mcse']:.4f} | "
              f"{ag.get('reference_occupancy', float('nan')):.3f} | {_f(st['rhat'], '.3f')} | {_f(st['ess'], '.0f')} | "
              f"{s['rhat']:.3f} | {s['ess_bulk']:.0f}/{s['ess_tail']:.0f} | {ag.get('z', float('nan')):.1f} | "
              f"{'ok' if b['convergence_pass'] else 'FAIL'}")
    c = g["criteria"]
    print(f"\nregions: low = log10_rho < {c['region_low']}, high = log10_rho > {c['region_high']} (hysteresis); "
          f"relevance: {g['relevance_source']}")
    print("bin | region | relevant | occupancy | ref | entries/exits | chains w/ events | sojourns n/mean/max | "
          "max sojourn frac | ind R-hat | ind ESS | MCSE | agree z | verdict")
    for n, rb in g["regions"].items():
        for R, rr in rb.items():
            st, sj, ag = rr["indicator"], rr["sojourns"], rr.get("agreement", {})
            rel = {None: "?", True: "yes", False: "no"}[rr["relevant"]]
            verdict = {None: "-", True: "ok", False: "FAIL"}[rr["pass"]]
            print(f"{n.split('_')[-1]:>3} | {R:>4} | {rel:>3} | {rr['occupancy']:.4f} | "
                  f"{_f(ag.get('reference_occupancy'), '.4f')} | {rr['entries']}/{rr['exits']} | "
                  f"{rr['chains_with_events']}/{len(rr['entries_per_chain'])} | {sj['count']}/{sj['mean']:.1f}/{sj['max']} | "
                  f"{_f(sj['max_frac'], '.2f')} | {_f(st['rhat'], '.3f')} | {_f(st['ess'], '.0f')} | {rr['mcse']:.4f} | "
                  f"{_f(ag.get('z'), '.1f')} | {verdict}")
    if g["reference_diagnostics"]:
        rd = g["reference_diagnostics"]
        print(f"\nreference ({args.released}): max split R-hat {max(r['split_rhat'] for r in rd.values()):.4f}, "
              f"min bulk/tail ESS {min(r['ess_bulk'] for r in rd.values()):.0f}/"
              f"{min(r['ess_tail'] for r in rd.values()):.0f}")
        miss = [f"{n.split('_')[-1]}/{R}" for n, r in rd.items() for R, rr in r["regions"].items()
                if rr["meets_event_criteria"] is False]
        print(f"reference relevant regions not meeting the event criteria itself (informational): "
              f"{', '.join(miss) if miss else 'none'}")
    print(f"\nCONVERGENCE: {g['convergence']} ({g['n_parameters_failing']} parameters, "
          f"{g['n_bins_failing_convergence']} bins, {g['n_regions_failing']} relevant regions failing)")
    for reason in g["convergence_inconclusive"]:
        print("  inconclusive:", reason)
    print(f"  no exploration evidence: {', '.join(g['no_exploration_evidence']) or 'none'}")
    print(f"REPRODUCTION AGREEMENT: {g['reproduction_agreement']} ({g['n_bins_disagreeing']} bins disagreeing, "
          f"{len(g['agreement_imprecise'])} comparisons with our SE > {c['agreement_se_max']})")
    if g.get("max_reference_se"):
        m = g["max_reference_se"]
        print(f"  reference precision (informational): max reference SE {m['se']:.4f} ({m['quantity']}); "
              f"{g['n_reference_imprecise']} comparisons with reference SE > {c['agreement_se_max']}")
    print(f"HEURISTIC WARNINGS (informational, not calibrated): {len(g['heuristic_warnings'])}")
    region_or_bin = [f for f in g["convergence_failures"] if " region " in f or "occupancy" in f]
    other = [f for f in g["convergence_failures"] if f not in region_or_bin]
    print(f"convergence failures: {len(other)} parameter, {len(region_or_bin)} occupancy / region")
    for f in region_or_bin[:60] + other[:10] + g["agreement_failures"][:15]:
        print("  -", f)
    for f in g["agreement_imprecise"][:5]:
        print("  - imprecise:", f)
    save_json(g, ROOT / "outputs" / "m2" / f"freespec_gate_{args.run}.json")
    print(f"GATE exit status {code} ({'PASS' if code == 0 else 'FAIL/INCONCLUSIVE'})")
    return code


if __name__ == "__main__":
    sys.exit(main())
