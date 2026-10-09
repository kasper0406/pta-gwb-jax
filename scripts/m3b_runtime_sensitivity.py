"""Evaluator-runtime sensitivity of the reproduction (review round 2 of M3b-0E, item 6). CPU, no
sampling: uses the paired likelihoods of ``scripts/m3b_reference_weights.py`` (pinned fork, chain
runtime ``epta-dr2-chain-runtime-v1`` vs the tempo2 2026 bundle runtime) at the released draws.

For each model (CURN: every retained draw; HD: every 5th) and each common-process parameter:
* both runtime summaries: the 5/50/95 % quantiles under the chain runtime (the released draws)
  and under the 2026 runtime (the same draws reweighted by w = L_2026 / L_chain);
* their shifts with the **paired** displacement MCSE (``reweight.paired_quantile_shift``) and,
  for comparison, the weighted-endpoint MCSE;
* overlap diagnostics (k-hat, Kish ESS) and the lnL-difference summary;
* the same conditional on the common domain D (acceptance file d9.exclusions).

This is a sensitivity of the reproduction to the evaluator runtime: the reproduction holds under
the pinned chain runtime and is not runtime-insensitive. It does not widen any margin.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_runtime_sensitivity.py
Writes data/processed/m3b/epta/results/runtime_sensitivity.json.
"""

from __future__ import annotations

import json

import numpy as np

from ptagwb import acceptance as acc
from ptagwb import epta
from ptagwb import reweight as rw
from ptagwb.binding import evidence_binding
from ptagwb.config import REPO_ROOT

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
PROBS = (0.05, 0.5, 0.95)


def main():
    man = epta.load_manifest()
    a = json.loads((REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json").read_text())
    ex = acc.d9_exclusions(a)
    meta = json.loads((RES / "reference_weights.json").read_text())
    z = np.load(RES / "reference_weights.npz")
    out = {"label": "sensitivity of the reproduction to the evaluator runtime (not a posterior result)", "models": {}}
    for key, thin in (("crn_pl", 1), ("hd_pl", meta["thin"])):
        names, X, burn = epta.load_reference(key, man)
        R = X[burn:][::thin]
        l0, l1 = z[f"{key}_lnl_chain_rt"], z[f"{key}_lnl_2026_rt"]
        if len(l0) != len(R):
            raise SystemExit(f"{key}: draws do not match the saved likelihoods")
        lw = l1 - l0
        ind = acc.domain_indicator(lambda n: [R[:, names.index(n)]], ex, key)[0]
        orf = key.split("_")[0]
        res = {"n": int(len(lw)), "thin": thin, "dlnl": {"mean": float(lw.mean()), "sd": float(lw.std(ddof=1)),
                                                         "min": float(lw.min()), "max": float(lw.max())},
               "overlap": {"khat": rw.psis_khat(lw), "kish_ess": rw.kish_ess(lw),
                           "khat_D": rw.psis_khat(lw[ind]), "kish_ess_D": rw.kish_ess(lw[ind])},
               "params": {}}
        for p in (f"gw_{orf}_log10_A", f"gw_{orf}_gamma"):
            x = R[:, names.index(p)]
            res["params"][p] = {}
            for dom, m in (("all", None), ("D", [ind])):
                rows = {}
                for q in PROBS:
                    s = rw.paired_quantile_shift([x], [lw], q, mask=m)
                    rows[str(q)] = {"chain_runtime": s["q_unweighted"], "runtime_2026": s["q_weighted"],
                                    "shift": s["shift"], "mcse_shift_paired": s["mcse_shift_paired"],
                                    "endpoint_mcse_weighted": s["endpoint_mcse_weighted"]}
                res["params"][p][dom] = rows
        out["models"][key] = res
        print(key, json.dumps(res["dlnl"]), json.dumps(res["overlap"]))
        for p, d in res["params"].items():
            for dom in ("all", "D"):
                print("  ", p, dom, {q: (round(v["shift"], 4), round(v["mcse_shift_paired"], 4)) for q, v in d[dom].items()})
    out["binding"] = evidence_binding()
    (RES / "runtime_sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
