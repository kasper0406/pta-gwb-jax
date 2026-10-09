"""Gate T1 for the complete EPTA DR2new roster (docs/M3B_PLAN.md Sec. 4.1; ``ptagwb.t1``).

1. our tempo2 export (``scripts/m3b_t2_export.py`` in the plain tempo2 env) of the canonical files
   -> data/processed/m3b/epta/export/ours_canonical/ (the arrays our pipeline consumes), and of the
   released files -> ours_original/;
2. the pinned enterprise (EPTA fork env) built from the released and from the canonical files
   (``scripts/m3b_epta_oracle.py export``) -> enterprise_original/, enterprise_canonical/;
3. comparisons: (a) ours(canonical) vs enterprise(original) [the gate: what we consume vs what the
   PTA's code consumed], (b) enterprise(original) vs enterprise(canonical) [released vs canonical
   inputs], (c) ours(canonical) vs enterprise(canonical) [exporter vs enterprise on equal inputs],
   (d) ours(original) vs enterprise(original).

All with the runtime epta-dr2-chain-runtime-v1 (data/processed/m3b/epta/t2runtime).
Writes data/processed/m3b/epta/results/t1.json. Usage:
    PYTHONPATH=src python scripts/m3b_t1.py [--jobs 8] [--reuse]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ptagwb.config import REPO_ROOT
from ptagwb.t1 import compare

BASE = REPO_ROOT / "data" / "processed" / "m3b" / "epta"
RUNTIME = BASE / "t2runtime"


def export_ours(variant: str, jobs: int, reuse: bool) -> dict:
    prep = json.loads((BASE / "canonical" / "prepare.json").read_text())
    out = BASE / "export" / f"ours_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TEMPO2_OVERRIDE=str(RUNTIME))

    def one(item):
        psr, r = item
        dst = out / f"{psr}.npz"
        if reuse and dst.exists():
            return psr, 0
        par, tim = (r["canon_par"], r["canon_tim"]) if variant == "canonical" else (r["par"], r["tim"])
        p = subprocess.run([str(REPO_ROOT / "scripts" / "t2py"), str(REPO_ROOT / "scripts" / "m3b_t2_export.py"),
                            par, tim, str(dst)], env=env, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"{psr}: {p.stderr[-2000:]}")
        return psr, 1

    with ThreadPoolExecutor(jobs) as ex:
        return dict(ex.map(one, sorted(prep.items())))


def export_enterprise(variant: str, reuse: bool) -> None:
    d = BASE / "export" / f"enterprise_{variant}"
    if reuse and d.exists() and len(list(d.glob("*.npz"))) == 25:
        return
    env = dict(os.environ, TEMPO2_OVERRIDE=str(RUNTIME))
    p = subprocess.run([str(REPO_ROOT / "scripts" / "eptapy"), str(REPO_ROOT / "scripts" / "m3b_epta_oracle.py"),
                        "export", "--variant", variant], env=env, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])


def run(jobs: int = 8, reuse: bool = False) -> dict:
    for v in ("canonical", "original"):
        export_ours(v, jobs, reuse)
        export_enterprise(v, reuse)
    runtime = json.loads((RUNTIME / "runtime.json").read_text())
    pairs = {"ours_canonical_vs_enterprise_original": ("ours_canonical", "enterprise_original"),
             "enterprise_original_vs_enterprise_canonical": ("enterprise_original", "enterprise_canonical"),
             "ours_canonical_vs_enterprise_canonical": ("ours_canonical", "enterprise_canonical"),
             "ours_original_vs_enterprise_original": ("ours_original", "enterprise_original")}
    res = {"runtime": runtime["runtime"], "pin_verified": runtime.get("pin_verified"), "comparisons": {}}
    for key, (x, y) in pairs.items():
        rows = []
        for f in sorted((BASE / "export" / y).glob("*.npz")):
            with np.load(BASE / "export" / x / f.name) as a, np.load(f) as b:
                rows.append(compare(dict(a), dict(b)))
        res["comparisons"][key] = {"n_pulsars": len(rows), "n_toas": int(sum(r["n"][0] for r in rows)),
                                   "n_pass": int(sum(r["ok"] for r in rows)), "rows": rows,
                                   "max": {k: max(float(r.get(k, 0.0)) for r in rows)
                                           for k in ("max_abs_toas", "max_abs_stoas", "max_abs_residuals",
                                                     "max_abs_toaerrs", "max_rel_freqs", "mmat_max_sin",
                                                     "max_abs_pos", "max_rel_pos_t", "max_rel_sunssb",
                                                     "max_rel_planetssb")}}
    gate = res["comparisons"]["ours_canonical_vs_enterprise_original"]
    canon = res["comparisons"]["enterprise_original_vs_enterprise_canonical"]
    res["roster_complete"] = gate["n_pulsars"] == 25 and gate["n_toas"] == 45428
    res["T1_pass"] = bool(res["roster_complete"] and gate["n_pass"] == 25 and canon["n_pass"] == 25)
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "t1.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    res = run(a.jobs, a.reuse)
    for k, c in res["comparisons"].items():
        print(f"{k}: {c['n_pass']}/{c['n_pulsars']} pass, {c['n_toas']} TOAs; max {json.dumps(c['max'])}")
        for r in c["rows"]:
            if r["fail"] or r["notes"]:
                print("   ", r["name"], r["fail"], r["notes"])
    print("T1 PASS" if res["T1_pass"] else "T1 FAIL", "| complete roster:", res["roster_complete"])


if __name__ == "__main__":
    main()
