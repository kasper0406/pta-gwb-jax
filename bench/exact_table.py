"""Markdown table of bench/results/exact_ng15_{cpu,gpu}.json."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
tabs = {b: json.loads((R / f"exact_ng15_{b}.json").read_text())["results"] for b in ("cpu", "gpu") if (R / f"exact_ng15_{b}.json").exists()}
print("| configuration | " + " | ".join(f"{b.upper()} dv / dg (prod floor dv / dg)" for b in tabs) + " |")
print("|---|" + "---|" * len(tabs))
for k in next(iter(tabs.values())):
    cells = [f"{t[k]['dv']:.1e} / {t[k]['dg']:.1e} ({t[k]['floor_dv']:.1e} / {t[k]['floor_dg']:.1e})" + ("" if t[k]["within_tolerance"] else " **OUTSIDE**") for t in tabs.values()]
    print(f"| {k.replace('/hh+levels', '')} | " + " | ".join(cells) + " |")
