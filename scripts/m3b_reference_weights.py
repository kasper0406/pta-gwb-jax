"""CPU reweighting analyses on the released EPTA DR2new draws (no sampling; review of 2ee1bb7,
items 2 and 4). Likelihoods come from the pinned fork enterprise (equal to ours to <= 5e-10 nats,
docs/M3B0_VALIDATION.md Sec. 6.2), evaluated in the fork env.

1. ``hd_over_curn``: log w = lnL_HD - lnL_CURN at the retained CURN reference draws (every
   ``--thin``-th), chain runtime. Gives the importance-sampling efficiency of HD-from-CURN
   reweighting (k-hat, Kish ESS, weighted-quantile MCSE per CURN draw), the input of the per-run
   cost projection (``scripts/m3b_projection.py``).
2. ``paired_runtime``: log w = lnL(2026 tempo2 bundle runtime) - lnL(chain runtime) at the
   retained CURN draws (all) and HD draws (every ``--thin``-th): the displacement of the headline
   quantiles if the D1 2026 runtime had been used, with overlap diagnostics (k-hat, Kish ESS).
   A sensitivity of the reproduction to the evaluator runtime, not a posterior.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_reference_weights.py [--thin 5]
Writes data/processed/m3b/epta/results/reference_weights.npz and .json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ptagwb import epta
from ptagwb import reweight as rw
from ptagwb.config import REPO_ROOT

BASE = REPO_ROOT / "data" / "processed" / "m3b" / "epta"
RUNTIMES = {"chain": BASE / "t2runtime", "d1_2026_bundle": BASE / "t2runtime-d1-2026bundle"}
HEAD = {"crn_pl": ("gw_crn_log10_A", "gw_crn_gamma"), "hd_pl": ("gw_hd_log10_A", "gw_hd_gamma")}


def fork_lnl(orf: str, X: np.ndarray, names: list[str], runtime: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        pts, out = Path(td) / "pts.npz", Path(td) / "out.npz"
        np.savez(pts, x=X, names=np.array(names))
        env = dict(os.environ, TEMPO2_OVERRIDE=str(runtime))
        p = subprocess.run([str(REPO_ROOT / "scripts" / "eptapy"), str(REPO_ROOT / "scripts" / "m3b_epta_oracle.py"),
                            "like", "--orf", orf, "--points", str(pts), "--out", str(out)], env=env,
                           capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-3000:])
        with np.load(out) as z:
            return np.asarray(z["lnlike"])


def summary(x, lw, probs=(0.05, 0.5, 0.95)):
    out = {"khat": rw.psis_khat(lw), "kish_ess": rw.kish_ess(lw), "n": int(len(lw)), "quantiles": {}}
    for q in probs:
        r = rw.weighted_quantile_mcse(x, lw, q)
        out["quantiles"][str(q)] = {"unweighted": float(np.quantile(x, q)), "weighted": r["q"], "mcse": r["mcse"],
                                    "shift": r["q"] - float(np.quantile(x, q))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thin", type=int, default=5)
    a = ap.parse_args()
    man = epta.load_manifest()
    res, arrays = {"thin": a.thin}, {}
    # ---- 1. HD/CURN weights at CURN draws
    names_c, Xc, burn = epta.load_reference("crn_pl", man)
    Rc = Xc[burn:][:: a.thin]
    names_h = [n.replace("gw_crn", "gw_hd") for n in names_c]
    l_c = fork_lnl("crn", Rc[:, :67], names_c, RUNTIMES["chain"])
    l_h = fork_lnl("hd", Rc[:, :67], names_h, RUNTIMES["chain"])
    lw = l_h - l_c
    arrays.update(curn_rows_thin=np.arange(burn, len(Xc))[:: a.thin], lnl_crn=l_c, lnl_hd_at_crn=l_h,
                  stored_lnl_crn=Rc[:, 68])
    hd = {}
    for p in HEAD["crn_pl"]:
        hd[p] = summary(Rc[:, names_c.index(p)], lw)
    lnbf = rw.mcse_lnbf_obm(lw)
    res["hd_over_curn"] = {"lnbf_raw": rw.raw_bf(lw), "lnbf_mcse_obm_thinned": lnbf["mcse"], "khat": rw.psis_khat(lw),
                           "kish_ess": rw.kish_ess(lw), "n": int(len(lw)), "per_param": hd,
                           "max_abs_stored_minus_fork_crn": float(np.max(np.abs(Rc[:, 68] - l_c)))}
    print("HD/CURN", json.dumps({k: v for k, v in res["hd_over_curn"].items() if k != "per_param"}), flush=True)
    # ---- 2. paired runtime
    pr = {}
    for key, orf, thin in (("crn_pl", "crn", 1), ("hd_pl", "hd", a.thin)):
        names, X, burn = epta.load_reference(key, man)
        R = X[burn:][::thin]
        l0 = fork_lnl(orf, R[:, :67], names, RUNTIMES["chain"])
        l1 = fork_lnl(orf, R[:, :67], names, RUNTIMES["d1_2026_bundle"])
        d = l1 - l0
        arrays[f"{key}_lnl_chain_rt"], arrays[f"{key}_lnl_2026_rt"] = l0, l1
        pr[key] = {"thin": thin, "n": int(len(d)), "dlnl_mean": float(d.mean()), "dlnl_sd": float(d.std(ddof=1)),
                   "dlnl_range": [float(d.min()), float(d.max())],
                   "params": {p: summary(R[:, names.index(p)], d) for p in HEAD[key]}}
        print(key, json.dumps({k: v for k, v in pr[key].items() if k != "params"}), flush=True)
    res["paired_runtime"] = pr
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    np.savez(BASE / "results" / "reference_weights.npz", **arrays)
    (BASE / "results" / "reference_weights.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
