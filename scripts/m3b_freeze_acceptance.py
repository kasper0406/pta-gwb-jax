"""N11 + plan Sec. 6.5: reference products of the released EPTA DR2new chains and the frozen
acceptance / relevance files, written BEFORE any production run.

* Appendix A reproduced: quantiles, bulk/tail ESS and quantile MCSEs of the common-process
  parameters of crn_pl and hd_pl (first 25 % of chain_1.txt discarded).
* Sec. 6.2 table: q_ref, MCSE_ref, margin m (D2, provisional: 0.15 sigma68_ref for medians, 0.25
  sigma68_ref for outer quantiles; sigma68_ref = (q84 - q16) / 2 of the reference), decidability
  (1.645 MCSE_ref < m), max_our_MCSE, role; E-5 (secondary) and E-6 (ln BF) rules.
* Relevance / transport declarations (Sec. 5.3): class (a) shelf-prone amplitudes (every noise and
  common log10_A, and the dip log10_Amp; shelf S = [lower, lower + 1 dex], peak P = the rest),
  class (b) the dip epoch t0 (inter-TOA gaps of J1713+0747 inside the window holding >= 1 %
  reference mass plus "rest"), class (c) everything else; the reference occupancy of every region
  with its case (estimable / few-event / zero-visit / all-visit) and status.
* D9 (Sec. 6.6): the excluded-region list U = every class-(a)/(b) region with **zero retained
  reference visits**, per model, with exact boundaries, reference file, sha256 and burn-in; few-
  event regions are listed separately (not eligible: separate review).

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_freeze_acceptance.py [--write]
Without --write the committed files are regenerated in memory and compared (fail if different).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from ptagwb import acceptance as acc
from ptagwb import epta
from ptagwb.config import REPO_ROOT
from ptagwb.diagnostics import ess_bulk, ess_tail, mcse_quantile

ACC = REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json"
REL = REPO_ROOT / "configs" / "m3b" / "relevance" / "epta.json"
PROBS = (0.05, 0.16, 0.5, 0.84, 0.95)
HEADLINE = {  # Sec. 6.2: (ID, model, parameter, quantile) -> role before decidability
    ("E-1", "hd_pl", "gw_hd_log10_A"): (0.05, 0.5, 0.95),
    ("E-2", "hd_pl", "gw_hd_gamma"): (0.05, 0.5, 0.95),
    ("E-3", "crn_pl", "gw_crn_log10_A"): (0.05, 0.5, 0.95),
    ("E-4", "crn_pl", "gw_crn_gamma"): (0.05, 0.5, 0.95),
}


def git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def reference_summary(man):
    out = {}
    for key in ("crn_pl", "hd_pl"):
        names, X, burn = epta.load_reference(key, man)
        R = X[burn:, :67]
        orf = key.split("_")[0]
        rows = {}
        for p in (f"gw_{orf}_log10_A", f"gw_{orf}_gamma"):
            v = R[:, names.index(p)]
            rows[p] = {"q": {str(q): float(np.quantile(v, q)) for q in PROBS},
                       "ess_bulk": float(ess_bulk(v[None, :])), "ess_tail": float(ess_tail(v[None, :])),
                       "mcse": {str(q): float(mcse_quantile(v[None, :], q)) for q in PROBS}}
        out[key] = {"n_rows": int(len(X)), "burn_in_rows": burn, "n_retained": int(len(R)),
                    "lnpost_minus_lnlike_mean": float(np.mean(X[:, 67] - X[:, 68])),
                    "lnlike_range": [float(X[burn:, 68].min()), float(X[burn:, 68].max())], "params": rows}
    return out


def decision_table(ref):
    rows = []
    for (eid, key, p), qs in HEADLINE.items():
        r = ref[key]["params"][p]
        s68 = (r["q"]["0.84"] - r["q"]["0.16"]) / 2
        for q in qs:
            m = (0.15 if q == 0.5 else 0.25) * s68
            mref = r["mcse"][str(q)]
            dec = acc.decidable(m, mref)
            rows.append({"id": eid, "model": key, "param": p, "quantile": q, "q_ref": r["q"][str(q)], "mcse_ref": mref,
                         "sigma68_ref": s68, "m": m, "decidable": bool(dec),
                         "max_our_mcse": acc.max_our_mcse(m, mref),
                         "role": "headline" if dec else "reported (reference-limited)"})
    return rows


def occupancy_dict(o: acc.Occupancy) -> dict:
    return {"case": o.case, "status": o.status, "p_hat": o.p_hat, "p_lo": o.p_lo, "p_hi": o.p_hi,
            "n_entries": o.n_entries, "n_exits": o.n_exits, "tau": o.tau, "n_eff": o.n_eff, "mcse": o.mcse,
            "support": acc.support_class(o)}


def relevance(man, psrs):
    out = {"class_a": [], "class_b": {}, "class_c": []}
    U, few = [], []
    for key in ("crn_pl", "hd_pl"):
        names, X, burn = epta.load_reference(key, man)
        R = X[burn:, :67]
        ref = man["reference_chains"][key]
        for q in man["parameters"]:
            if q["name"] not in names:
                continue
            v = R[:, names.index(q["name"])]
            lo, hi = q["bounds"]
            if q["shelf_prone"]:
                ind = acc.shelf_peak_indicators(v, lo)
                regs = acc.shelf_peak_regions(lo, hi)
                occ = {r: acc.occupancy(ind[r]) for r in ("S", "P")}
                ent = {"model": key, "param": q["name"], "regions": {r: list(regs[r]) for r in regs},
                       "reference": {r: occupancy_dict(occ[r]) for r in occ}}
                out["class_a"].append(ent)
                for r in ("S", "P"):
                    o = occ[r]
                    reg = {"name": f"{q['name']}:{r}", "model": key, "param": q["name"], "region": r,
                           "boundaries": list(regs[r]), "closed": "S: [lo, lo + 1]; P: (lo + 1, hi]",
                           "reference_file": ref["tar"] + "::" + key + "/chain_1.txt", "sha256": ref["chain_sha256"],
                           "burn_in": ref["burn_in_rows"], "reference_case": o.case}
                    if o.case == "zero-visit":
                        U.append(reg)
                    elif o.case == "few-event":
                        few.append(reg)
            elif q["kernel"] == "mh":
                pass  # class (b), below
            elif key == "crn_pl" or q["name"].startswith("gw_hd"):
                out["class_c"].append(q["name"])
        # class (b): t0 of the dip
        d = man["dip"]
        t0n = f"{d['param_prefix']}_t0"
        t0 = R[:, names.index(t0n)]
        dip = psrs[[p.name for p in psrs].index(d["pulsar"])]
        window = tuple(d["window_mjd"])
        ints = acc.event_intervals(dip.toas / 86400.0, window, t0)
        ind = acc.event_interval_indicators(t0, ints, window)
        occ = {k: acc.occupancy(v) for k, v in ind.items()}
        n_toa_in = int(np.sum((dip.toas / 86400.0 > window[0]) & (dip.toas / 86400.0 < window[1])))
        out["class_b"][key] = {"param": t0n, "window_mjd": list(window), "toa_time_scale": "BAT (TCB) MJD, enterprise psr.toas / 86400",
                               "n_toas_in_window": n_toa_in, "intervals": [list(i) for i in ints],
                               "reference": {k: occupancy_dict(o) for k, o in occ.items()}}
        for k, o in occ.items():
            bounds = list(ints[int(k[1:])]) if k != "rest" else "complement of the listed intervals within and outside the window"
            reg = {"name": f"{t0n}:{k}", "model": key, "param": t0n, "region": k, "boundaries": bounds,
                   "reference_file": ref["tar"] + "::" + key + "/chain_1.txt", "sha256": ref["chain_sha256"],
                   "burn_in": ref["burn_in_rows"], "reference_case": o.case}
            if o.case == "zero-visit":
                U.append(reg)
            elif o.case == "few-event":
                few.append(reg)
    return out, U, few


def build():
    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    ref = reference_summary(man)
    table = decision_table(ref)
    rel, U, few = relevance(man, psrs)
    acc_file = {
        "pta": "EPTA DR2new", "version": 1,
        "frozen_before_any_production_run": True,
        "plan": "docs/M3B_PLAN.md (approved revision 6ef14c8; user decisions D1-D9 of 2026-10-09 recorded in 4ab41dd)",
        "manifest": man["manifest"],
        "margins_status": "provisional (D2): changed only by a reviewed revision before the run it affects",
        "decision_rule": {"interval": "D +- 1.645 SE_D, SE_D = sqrt(MCSE_ours^2 + MCSE_ref^2)",
                          "EQUIVALENT": "inside [-m, m]", "INCOMPATIBLE": "entirely outside", "INCONCLUSIVE": "otherwise",
                          "reference_limited": "1.645 MCSE_ref >= m", "max_our_mcse": "sqrt((m/1.645)^2/4 - MCSE_ref^2), floor 0.2 m/1.645"},
        "reference": {k: {"file": man["reference_chains"][k]["tar"], "tar_sha256": man["reference_chains"][k]["tar_sha256"],
                          "chain_sha256": man["reference_chains"][k]["chain_sha256"],
                          "burn_in_rows": man["reference_chains"][k]["burn_in_rows"]} for k in ("crn_pl", "hd_pl")},
        "reference_summary": ref,
        "quantities": table,
        "E5": {"role": "secondary", "target": "HD log10_A at gamma = 13/3 (reweighted from a fixed-gamma CURN run)",
               "q_ref_paper": {"0.5": -14.61, "0.05": -14.73, "0.95": -14.50}, "m": {"0.5": 0.035, "0.05": 0.03, "0.95": 0.03},
               "reference_rounding_se": 0.005, "label": "reference uncertainty incomplete"},
        "E6": {"role": "headline", "target_lnbf": float(np.log(60.0)), "m": 0.30, "max_se": 0.10,
               "se": "MCSE(ln BF) (Sec. 5.4) and the spread of our estimators in quadrature",
               "context": "EPTA re-estimates 56-66"},
        "reweighting": {"estimator": "raw only", "khat_max": 0.5, "khat_scope": "pooled and every chain",
                        "max_mcse_lnbf": 0.10, "stability_p_min": 0.01, "max_chain_weight_share": 0.5},
        "convergence": {"rhat_max": 1.01, "ess_min_all": 400, "ess_min_targets": 1000, "divergences": 0,
                        "occupancy": "Sec. 5.3 rules, UNRESOLVED precedence (ptagwb.acceptance)",
                        "relevance_file": "configs/m3b/relevance/epta.json"},
        "d9": {
            "claim_class": acc.D9_CLASS, "template": acc.D9_TEMPLATE,
            "excluded_regions": U,
            "union_definition": "U_m = union of every listed region of model m (crn_pl: CURN; hd_pl: HD)",
            "few_event_regions_not_eligible": few,
            "p_grid": list(acc.D9_P_GRID), "bf_grid": list(acc.D9_BF_GRID),
            "epsilon_rule": "largest grid value <= min over the model's headline quantities of the lower 90 % MC bound of p*; class unavailable below 0.001",
            "unconditional_verdict": "INCONCLUSIVE",
        },
    }
    rel_file = {"pta": "EPTA DR2new", "version": 1, "frozen_before_any_production_run": True,
                "manifest": man["manifest"], **rel}
    return acc_file, rel_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    acc_file, rel_file = build()
    if a.write:
        REL.parent.mkdir(parents=True, exist_ok=True)
        ACC.write_text(json.dumps(acc_file, indent=1))
        REL.write_text(json.dumps(rel_file, indent=1))
    else:
        for path, new in ((ACC, acc_file), (REL, rel_file)):
            if json.loads(path.read_text()) != json.loads(json.dumps(new)):
                sys.exit(f"{path.name} differs from a fresh generation")
    U = acc_file["d9"]["excluded_regions"]
    print(f"U: {len(U)} regions ({sum(r['model'] == 'crn_pl' for r in U)} CURN, {sum(r['model'] == 'hd_pl' for r in U)} HD); "
          f"few-event (not eligible): {len(acc_file['d9']['few_event_regions_not_eligible'])}")
    for r in acc_file["quantities"]:
        print(f"{r['id']} {r['model']} q{r['quantile']}: q_ref {r['q_ref']:.3f} mcse {r['mcse_ref']:.4f} m {r['m']:.3f} "
              f"decidable {r['decidable']} max_our {r['max_our_mcse']:.4f}")


if __name__ == "__main__":
    main()
