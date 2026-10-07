"""Table of the matched HD^13/3 NUTS comparison (bench/results/samplers/matched_hd_*.json).

    JAX_PLATFORMS=cpu uv run --no-sync python bench/hd_matched_table.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def main():
    rows = []
    for f in sorted(glob.glob(str(BENCH / "results" / "samplers" / "matched_hd_*.json"))):
        d = json.loads(Path(f).read_text())
        x = d["extra"]
        tc, ta = x.get("target_common"), x.get("target_common_and_max_rhat")
        rows.append({
            "run": Path(f).stem, "sampler": {"np_nuts": "NumPyro NUTS", "bj_nuts2": "BlackJAX NUTS"}[d["sampler"]],
            "seed": d["seed"], "step_sizes": [round(s, 4) for s in x["step_size"]],
            "compile_s": d["t_compile"], "warmup_s": d["t_warm"], "warmup_grads": d["grads_warmup"],
            "draws": d["draws_per_chain"], "grads_per_draw": d["grads_per_draw_mean"],
            "lockstep": d["lockstep_efficiency"], "ms_per_exec_step": d["ms_per_lockstep_step"],
            "accept": d["accept_mean"],
            "ess_common": d["ess_bulk_common"], "ess_min_bulk": d["ess_bulk_min"], "ess_min_tail": d["ess_tail_min"],
            "rhat_common": d["rhat_common"], "rhat_max": d["rhat_max"],
            "ess_per_s_common": d["ess_per_s_bulk_common"], "ess_per_kgrad_common": d["ess_per_kgrad_bulk_common"],
            "ess_per_s_min_bulk": d["ess_per_s_bulk_min"], "ess_per_s_min_tail": d["ess_per_s_tail_min"],
            "target_common_draws": tc["draws"] if tc else None, "target_common_total_s": tc["t_total"] if tc else None,
            "target_all_draws": ta["draws"] if ta else None, "target_all_total_s": ta["t_total"] if ta else None,
            "t_total_s": d["t_total"], "divergences": d["divergences"],
        })
    (BENCH / "results" / "hd_matched_summary.json").write_text(json.dumps(rows, indent=1))
    print("| sampler | seed | final step sizes (4 chains) | warmup [min] (grads) | compile [s] | grads/draw | lockstep eff. | ms / executed step | draws/chain | ESS common / min bulk / min tail | R-hat common / max | ESS/s common | ESS/kgrad common | **time to ESS_common >= 400 & R-hat_common < 1.01, incl. compile + warmup [min]** | ... & max R-hat < 1.01 |")
    print("|" + "---|" * 15)
    for r in rows:
        tt = f"{r['target_common_total_s'] / 60:.1f} ({r['target_common_draws']} draws)" if r["target_common_total_s"] else f"not reached in {r['draws']} draws ({r['t_total_s'] / 60:.0f} min)"
        ta = f"{r['target_all_total_s'] / 60:.1f}" if r["target_all_total_s"] else "not reached"
        print(f"| {r['sampler']} | {r['seed']} | {', '.join(f'{s:.4f}' for s in r['step_sizes'])} | {r['warmup_s'] / 60:.1f} ({r['warmup_grads'] / 1e3:.0f}k) | "
              f"{r['compile_s']:.0f} | {r['grads_per_draw']:.0f} | {r['lockstep']:.2f} | {r['ms_per_exec_step']:.1f} | {r['draws']} | "
              f"{r['ess_common']:.0f} / {r['ess_min_bulk']:.0f} / {r['ess_min_tail']:.0f} | {r['rhat_common']:.3f} / {r['rhat_max']:.3f} | "
              f"{r['ess_per_s_common']:.3f} | {r['ess_per_kgrad_common']:.2f} | {tt} | {ta} |")


if __name__ == "__main__":
    main()
