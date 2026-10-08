"""Fit and FREEZE block proposals for the hybrid kernel (``ptagwb.hybrid``) from pilot draws.

    uv run --no-sync python scripts/fs_fit_proposals.py --runs hd_fs30,hd_fs30_v2_pilotE \
        --out configs/m2/proposals/hd_fs30_v1.json

One 1-D block per free-spectrum bin and one 2-D (log10_A, gamma) block per listed pulsar. Draws
come only from our own runs (never from the released chains). The file is committed before any
run that uses it, so the proposal is fixed before measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_common import ROOT

from ptagwb.hybrid import fit_proposals
from ptagwb.sampling import git_state, load_run

DEFAULT_PAIRS = "J0610-2100,J2234+0611,J0437-4715,J1853+1303,J0645+5158,J1713+0747"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names with identical parameter vectors")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS, help="pulsars whose (log10_A, gamma) get a joint block")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-prior", type=float, default=0.2)
    ap.add_argument("--k1", type=int, default=40)
    ap.add_argument("--k2a", type=int, default=12)
    ap.add_argument("--k2b", type=int, default=8)
    args = ap.parse_args()
    runs = [load_run(r) for r in args.runs.split(",")]
    names = runs[0]["names"]
    lo, hi = np.array(runs[0]["meta"]["lo"]), np.array(runs[0]["meta"]["hi"])
    for r in runs[1:]:
        if r["names"] != names or r["meta"]["lo"] != list(lo) or r["meta"]["hi"] != list(hi):
            raise ValueError("runs differ in parameters or prior bounds")
    X = np.concatenate([r["x"].reshape(-1, len(names)) for r in runs])
    bins = [n for n in names if n.startswith("gw_log10_rho_")]
    pairs = [(f"{p}_red_noise_log10_A", f"{p}_red_noise_gamma") for p in args.pairs.split(",") if p]
    prop = fit_proposals(X, names, lo, hi, bins, pairs, w_prior=args.w_prior, k1=args.k1, k2a=args.k2a, k2b=args.k2b)
    meta = {"source_runs": args.runs.split(","), "source_shas": [r["meta"]["git"]["sha"] for r in runs], "n_draws": len(X),
            "pairs": args.pairs.split(","), "k1": args.k1, "k2a": args.k2a, "k2b": args.k2b, "fitted_at_sha": git_state()["sha"]}
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    prop.to_json(out, meta)
    print(json.dumps(meta), f"-> {out} ({prop.n_blocks} blocks)")


if __name__ == "__main__":
    main()
