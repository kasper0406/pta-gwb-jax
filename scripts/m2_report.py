"""Print the markdown tables of docs/M2_RESULTS.md from outputs/m2/{compare,bayes_factors,optstat}.json.

    uv run --no-sync python scripts/m2_report.py > outputs/m2/tables.md
"""

from __future__ import annotations

import json
import sys

import numpy as np
from m2_common import ROOT

OUT = ROOT / "outputs" / "m2"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else {}


def q3(s, exp=False):
    if exp:
        f = lambda v: 10**v
        return f"{f(s['q50']) * 1e15:.3f} [{f(s['q05']) * 1e15:.3f}, {f(s['q95']) * 1e15:.3f}]e-15"
    return f"{s['q50']:.3f} [{s['q05']:.3f}, {s['q95']:.3f}]"


def mcse3(s):
    return f"+-({s['q05_mcse']:.3f}, {s['q50_mcse']:.3f}, {s['q95_mcse']:.3f})"


def main():
    cmp_, bf, os_ = load("compare.json"), load("bayes_factors.json"), load("optstat.json")
    print("## Parameter comparison (5/50/95%; z = difference / combined quantile MCSE)\n")
    print("| run | released | param | ours | MCSE ours | released | MCSE rel | z(5,50,95) | KS D (p_ESS) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in cmp_.get("pairs", []):
        for p, v in e["params"].items():
            o, r = v["ours"], v["released"]
            z = f"{v['z_q05']:+.1f}, {v['z_q50']:+.1f}, {v['z_q95']:+.1f}"
            print(f"| {e['run']} | {e['released']} | {p} | {q3(o)} | {mcse3(o)} | {q3(r)} | {mcse3(r)} | {z} | "
                  f"{v['ks']['D']:.3f} ({v['ks']['p_ess']:.2f}) |")
    print("\n2-D (gamma, log10_A) energy distance, ours vs released:\n")
    for e in cmp_.get("pairs", []):
        if "energy_2d" in e:
            ed = e["energy_2d"]
            print(f"* {e['run']} vs {e['released']}: {ed['energy']:.2e} (MC floors: our chains 1-2 vs 3-4 "
                  f"{ed.get('ours_split_floor', float('nan')):.2e}; released halves {ed['released_split_floor']:.2e})")
    print("\n## ICRS minus enterprise positions (same model)\n")
    print("| run | param | enterprise | ICRS | shift of median | z(5,50,95) |")
    print("|---|---|---|---|---|---|")
    for rn, d in cmp_.get("icrs_shift", {}).items():
        for p, v in d.items():
            print(f"| {rn} | {p} | {q3(v['released'])} | {q3(v['ours'])} | {v['d_q50']:+.4f} | "
                  f"{v['z_q05']:+.1f}, {v['z_q50']:+.1f}, {v['z_q95']:+.1f} |")
    print("\n## IRN spot checks\n")
    print("| run | released | param | ours | released | z(5,50,95) | KS D |")
    print("|---|---|---|---|---|---|---|")
    for v in cmp_.get("irn", []):
        print(f"| {v['run']} | {v['released_chain']} | {v['param']} | {q3(v['ours'])} | {q3(v['released'])} | "
              f"{v['z_q05']:+.1f}, {v['z_q50']:+.1f}, {v['z_q95']:+.1f} | {v['ks']['D']:.3f} |")
    print("\n## Diagnostics and cost\n")
    print("| run | chains x draws | max R-hat (param) | min bulk / tail ESS | common bulk ESS | divergences | mean tree depth (steps) | accept | wall warmup + sampling [min] | grad evals | common ESS/s | SHA |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, d in cmp_.get("diagnostics", {}).items():
        ce = ", ".join(f"{k.replace('gw_', '')} {v:.0f}" for k, v in d["common_ess_bulk"].items())
        if len(d["common_ess_bulk"]) > 3:
            ce = f"min {min(d['common_ess_bulk'].values()):.0f}"
        print(f"| {n} | {d['chains']} x {d['draws_per_chain']} | {d['max_rhat']:.3f} ({d['argmax_rhat']}) | "
              f"{d['min_ess_bulk']:.0f} / {d['min_ess_tail']:.0f} | {ce} | {d['divergences']} | "
              f"{d['mean_tree_depth']:.1f} ({d['mean_steps']:.0f}) | {d['mean_accept']:.2f} | "
              f"{d['warmup_seconds'] / 60:.0f} + {d['sampling_seconds'] / 60:.0f} | {d['grad_evals']:,} | "
              f"{d['ess_per_s_common_sampling']:.2f} | {d['git_sha'][:7]}{'*' if d['git_dirty'] else ''} |")
    tot = sum(d["wall_seconds"] for d in cmp_.get("diagnostics", {}).values())
    print(f"\nTotal production wall time (warmup + sampling, excluding setup/compile): {tot / 3600:.2f} h")
    print("\n## Bayes factors HD vs CURN\n")
    print("| setup | estimator | BF | bootstrap sd | 16-84% | Kish ESS / n | block |")
    print("|---|---|---|---|---|---|---|")
    for k, r in bf.items():
        for est in ("reweight", "reverse_reweight", "bridge", "self_check_identity_orf"):
            if est in r:
                v = r[est]
                kish = f"{v['kish_ess']:.0f} / {v['n']}" if np.isfinite(v.get("kish_ess", np.nan)) else f"- / {v['n']}"
                print(f"| {k} | {est} | {v['bf']:.1f} | {v['bf_sd']:.1f} | {v['bf_q16_q84'][0]:.1f}-{v['bf_q16_q84'][1]:.1f} | {kish} | {v['block']} |")
    print("\n## Optimal statistic\n")
    for k, v in os_.items():
        if isinstance(v, dict) and "snr_mean" in v:
            print(f"* {k}: S/N {v['snr_mean']:.2f} +- {v['snr_std']:.2f}, A2 {v['A2_mean']:.3e} +- {v['A2_std']:.3e} (n={v['n']})")
        elif isinstance(v, dict) and "snr" in v and "A2" in v:
            print(f"* {k}: A2 {v['A2']:.4e} +- {v['sigma']:.4e}, S/N {v['snr']:.3f}")
        elif isinstance(v, dict) and "chi2" in v:
            print(f"* {k}: chi2 {v['chi2']:.2f}, p(chi2_15) {v['p_chi2_15dof']:.3f}, n_pairs {v['n_pairs']}")
    for k in ("pair_cov_vs_released_max_rel", "binned_at_released_ml_with_released_cov_chi2", "map_g433_params", "map_vg_params"):
        if k in os_:
            print(f"* {k}: {os_[k]}")


if __name__ == "__main__":
    sys.exit(main())
