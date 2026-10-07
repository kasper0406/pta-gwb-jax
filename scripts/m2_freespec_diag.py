"""Free-spectrum acceptance gate (CLI for ``ptagwb.diagnostics.freespec_gate``).

    uv run --no-sync python scripts/m2_freespec_diag.py [--run hd_fs30] [--threshold -9]

Exit status 0 only if the run passes **every** check; 1 on any failure; 2 if the run or a
diagnostic is missing. Checks (thresholds in ``diagnostics.GATE_DEFAULTS``):

* all expected bins present (``--n-bins``, default 30);
* every parameter (free-spectrum bins *and* the 134 IRN parameters): rank-normalised split R-hat
  < 1.01, bulk ESS >= 400, tail ESS >= 400;
* per bin, the occupancy indicator 1[log10_rho < threshold] (the "power absent" region of the
  bimodal marginals): R-hat < 1.01 and ESS >= 400, and pooled occupancy within 3.5 binomial-ESS
  standard errors of the released core's;
* tail stability per bin, at the pooled 5/50/95% quantiles q: the fraction of draws <= q in each
  chain vs the other chains, and in the first vs second half of all chains; z = difference /
  sqrt(var_a + var_b), binomial variances from each subset's own indicator ESS (i.e. the
  uncertainty of the difference, not a pooled MCSE); fail if any |z| > 4.5 (~0.5% family-wise
  false-alarm rate for ~810 tests). Quantile-value differences are reported descriptively.

The between-chain occupancy chi^2 p-value is printed as supplementary information only.
Sampler-agnostic: needs only ``runs/<run>/samples.npz`` (``x`` of shape (chains, draws, D)) and the
parameter names. Writes outputs/m2/freespec_gate_<run>.json.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from m2_common import ROOT, released, save_json

from ptagwb.diagnostics import freespec_gate
from ptagwb.sampling import load_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hd_fs30")
    ap.add_argument("--threshold", type=float, default=-9.0)
    ap.add_argument("--n-bins", type=int, default=30)
    ap.add_argument("--released", default="hd_fs30", help="released chain key ('' to skip the comparison)")
    args = ap.parse_args()
    try:
        run = load_run(args.run)
    except FileNotFoundError as e:
        print(f"GATE ERROR: run {args.run!r} not found ({e})")
        return 2
    rel = None
    if args.released:
        r = released(args.released)
        rel = {k.replace("gw_hd_log10_rho", "gw_log10_rho"): v for k, v in r.items()}
    g = freespec_gate(run["x"], run["names"], n_bins=args.n_bins, threshold=args.threshold, released=rel)
    C, N, D = run["x"].shape
    print(f"{args.run}: {C} chains x {N} draws, {D} parameters; occupancy = P(log10_rho < {args.threshold})")
    print("bin | per-chain occupancy | pooled | released | ind R-hat | ind ESS | rho R-hat | bulk/tail ESS | max|tail z| | chi2 p (suppl.) | pass")
    for n, b in g["bins"].items():
        if b.get("missing"):
            print(f"{n}: MISSING")
            continue
        s = g["parameters"][n]
        pc = " ".join(f"{v:.2f}" for v in b["occupancy_per_chain"])
        tz = max((abs(t["z"]) if np.isfinite(t["z"]) else np.inf) for t in b["tail_tests"])
        print(f"{n.split('_')[-1]:>3} | {pc} | {b['occupancy']:.3f} | {b.get('released_occupancy', float('nan')):.3f} | "
              f"{b['indicator_rhat']:.3f} | {b['indicator_ess']:.0f} | {s['rhat']:.3f} | {s['ess_bulk']:.0f}/{s['ess_tail']:.0f} | "
              f"{tz:.1f} | {b.get('supplementary_chi2_p', float('nan')):.2g} | {'ok' if b['pass'] else 'FAIL'}")
    print(f"parameters failing R-hat/ESS: {g['n_parameters_failing']} / {D}; bins failing: {g['n_bins_failing']} / {args.n_bins}")
    save_json(g, ROOT / "outputs" / "m2" / f"freespec_gate_{args.run}.json")
    if g["pass"]:
        print("GATE: PASS")
        return 0
    print(f"GATE: FAIL ({g['n_failures']} failures); first ones:")
    for f in g["failures"][:15]:
        print("  -", f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
