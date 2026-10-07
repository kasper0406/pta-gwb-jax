"""Markdown table of bench/results/exact_ng15_{cpu,gpu}.json."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
tabs = {b: json.loads((R / f"exact_ng15_{b}.json").read_text())["results"] for b in ("cpu", "gpu") if (R / f"exact_ng15_{b}.json").exists()}
print("| configuration | budget dv / dg | " + " | ".join(f"{b.upper()}: single dv / dg; batched dv / dg; production floor dv / dg" for b in tabs) + " |")
print("|---|---|" + "---|" * len(tabs))
for k in next(iter(tabs.values())):
    t0 = next(iter(tabs.values()))[k]
    cells = [f"{t[k]['dv_single']:.1e} / {t[k]['dg_single']:.1e}; {t[k]['dv_batched']:.1e} / {t[k]['dg_batched']:.1e}; "
             f"{t[k]['floor_dv']:.1e} / {t[k]['floor_dg']:.1e}" + ("" if t[k]["within_budget"] else " **OUTSIDE**") for t in tabs.values()]
    print(f"| {k.replace('/hh+levels', '')} | {t0['budget'][0]:.0e} / {t0['budget'][1]:.0e} | " + " | ".join(cells) + " |")
