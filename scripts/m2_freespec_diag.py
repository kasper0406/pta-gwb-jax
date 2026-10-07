"""Free-spectrum convergence diagnostics: per-bin mode occupancy and tail-quantile stability.

    uv run --no-sync python scripts/m2_freespec_diag.py [--run hd_fs30] [--threshold -9]

The free-spectrum marginals of partially constrained bins are bimodal: a "power present" peak
near log10_rho ~ -7.5 and a prior-dominated low-power plateau down to the prior edge -15.5.
Matching medians does not show that the chains weight the two regions correctly, so for each bin:

* occupancy: fraction of draws below ``--threshold`` (default -9, the lower edge of the Fig. 1a
  histograms) per chain, pooled, and in the released core; a between-chain chi^2 test of equal
  occupancy using effective counts (chain length scaled by the indicator's ESS / N), and the
  split-R-hat and bulk ESS of the indicator itself;
* tail stability: 5% / 50% / 95% quantiles per chain and per half of the run, their MCSEs, and
  the released values.

A bin passes if the indicator R-hat < 1.01, its ESS > 100, the between-chain test has p > 0.01,
and the pooled occupancy agrees with the released one within 3 combined binomial-ESS errors.
Works on any run directory (sampler-agnostic: needs only ``x`` (chains, draws, D) and names).
Writes outputs/m2/freespec_diag_<run>.json and prints a table.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from m2_common import ROOT, released, run_draws, save_json
from scipy import stats

from ptagwb.diagnostics import ess, ess_bulk, mcse_quantile, rhat
from ptagwb.sampling import load_run


def occupancy(x: np.ndarray, thr: float) -> dict:
    """x: (chains, n) draws of one bin."""
    ind = (x < thr).astype(float)
    C = ind.shape[0]
    per_chain = ind.mean(axis=1)
    pooled = float(ind.mean())
    e = ess(ind) if np.ptp(ind) > 0 else float(ind.size)
    n_eff_chain = max(e / C, 1.0)
    # chi^2 homogeneity test on effective counts
    if 0 < pooled < 1 and C > 1:
        chi2 = float(np.sum((per_chain - pooled) ** 2 * n_eff_chain / (pooled * (1 - pooled))))
        p = float(stats.chi2(C - 1).sf(chi2))
    else:
        chi2, p = 0.0, 1.0
    se = float(np.sqrt(max(pooled * (1 - pooled), 1.0 / ind.size) / max(e, 1.0)))
    return {
        "per_chain": per_chain.tolist(),
        "pooled": pooled,
        "pooled_se_ess": se,
        "indicator_ess": float(e),
        "indicator_rhat": rhat(ind) if np.ptp(ind) > 0 else 1.0,
        "between_chain_chi2": chi2,
        "between_chain_p": p,
    }


def tails(x: np.ndarray) -> dict:
    n = x.shape[1]
    h = n // 2
    out = {}
    for p in (0.05, 0.5, 0.95):
        k = f"q{round(100 * p):02d}"
        out[k] = {
            "pooled": float(np.quantile(x, p)),
            "mcse": mcse_quantile(x, p),
            "per_chain": np.quantile(x, p, axis=1).tolist(),
            "first_half": float(np.quantile(x[:, :h], p)),
            "second_half": float(np.quantile(x[:, h:], p)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hd_fs30")
    ap.add_argument("--threshold", type=float, default=-9.0)
    ap.add_argument("--released", default="hd_fs30")
    args = ap.parse_args()
    run = load_run(args.run)
    rel = released(args.released) if args.released else None
    names = [n for n in run["names"] if n.startswith("gw_log10_rho_")]
    out = {"run": args.run, "threshold": args.threshold, "draws": list(run["x"].shape[:2]), "bins": {}}
    print(f"{args.run}: {run['x'].shape[0]} chains x {run['x'].shape[1]} draws; occupancy = P(log10_rho < {args.threshold})")
    print("bin | per-chain occupancy | pooled +- se | released | chi2 p | ind R-hat | ind ESS | rho R-hat | bulk ESS | q05 pooled (per chain) | q05 rel | pass")
    n_fail = 0
    for n in names:
        x = run_draws(run, n)
        o = occupancy(x, args.threshold)
        t = tails(x)
        b = {"occupancy": o, "tails": t, "rhat": rhat(x), "ess_bulk": ess_bulk(x)}
        if rel is not None:
            r = rel[n]
            ro = float(np.mean(r < args.threshold))
            r_ess = ess((r < args.threshold).astype(float)[None]) if 0 < ro < 1 else float(len(r))
            b["released_occupancy"] = ro
            b["released_occupancy_se"] = float(np.sqrt(max(ro * (1 - ro), 1.0 / len(r)) / max(r_ess, 1.0)))
            b["released_q05"] = float(np.quantile(r, 0.05))
            z = (o["pooled"] - ro) / np.hypot(o["pooled_se_ess"], b["released_occupancy_se"])
            b["occupancy_z"] = float(z)
        ok = (o["indicator_rhat"] < 1.01 and o["indicator_ess"] > 100 and o["between_chain_p"] > 0.01
              and abs(b.get("occupancy_z", 0.0)) < 3)
        b["pass"] = bool(ok)
        n_fail += not ok
        out["bins"][n] = b
        pc = " ".join(f"{v:.2f}" for v in o["per_chain"])
        qc = " ".join(f"{v:.1f}" for v in t["q05"]["per_chain"])
        print(f"{n.split('_')[-1]:>3} | {pc} | {o['pooled']:.3f} +- {o['pooled_se_ess']:.3f} | "
              f"{b.get('released_occupancy', float('nan')):.3f} | {o['between_chain_p']:.2g} | "
              f"{o['indicator_rhat']:.3f} | {o['indicator_ess']:.0f} | {b['rhat']:.3f} | {b['ess_bulk']:.0f} | "
              f"{t['q05']['pooled']:.2f} ({qc}) | {b.get('released_q05', float('nan')):.2f} | {'ok' if ok else 'FAIL'}")
    out["n_bins_failing"] = n_fail
    print(f"bins failing: {n_fail} / {len(names)}")
    save_json(out, ROOT / "outputs" / "m2" / f"freespec_diag_{args.run}.json")


if __name__ == "__main__":
    sys.exit(main())
