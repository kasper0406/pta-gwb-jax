"""Compare sampling kernels on free-spectrum runs (docs/FS_PILOT.md Sec. 7): completed region
excursions per bin, occupancy MCSE, worst-parameter ESS, ESS per wall-clock hour, jump acceptance.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_compare_kernels.py curn_fs30_ab_nuts curn_fs30_ab_hybrid
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_compare_kernels.py hd_fs30_v2_pilotE hd_fs30_hybrid_pilot --draws 50

Regions and events as in ``diagnostics.region_events`` (low < -10, high > -8, hysteresis). A
"completed excursion" = entries + exits into/out of the minority region of the bin (pooled over
chains). ``--draws N`` truncates every run to its first N draws per chain (matched comparison).
Writes outputs/m2/fs_compare_<runs>.json.
"""

from __future__ import annotations

import argparse

import numpy as np
from m2_common import ROOT, save_json

from ptagwb.diagnostics import (
    GATE_DEFAULTS,
    _conservative_se,
    _indicator_stats,
    ess_bulk,
    ess_tail,
    region_events,
    rhat,
)
from ptagwb.sampling import load_run

LO, HI = GATE_DEFAULTS["region_low"], GATE_DEFAULTS["region_high"]


def summarize(name: str, n_draws: int | None, minority: dict) -> dict:
    r = load_run(name)
    x, meta, names = r["x"], r["meta"], r["names"]
    if n_draws:
        x = x[:, :n_draws]
    C, N, _ = x.shape
    ns = r["num_steps"][:, :N].astype(float)
    frac = N / r["x"].shape[1]
    samp_h = meta["sampling_seconds"] * frac / 3600
    par = {n: {"rhat": rhat(x[..., j]), "ess_bulk": ess_bulk(x[..., j]), "ess_tail": ess_tail(x[..., j])} for j, n in enumerate(names)}
    mins = {n: min(s["ess_bulk"], s["ess_tail"]) for n, s in par.items()}
    worst = min(mins, key=mins.get)
    bins = {}
    for k in range(30):
        n = f"gw_log10_rho_{k}"
        v = x[..., names.index(n)]
        ev = region_events(v, LO, HI)
        occ = {R: float(((v < LO) if R == "low" else (v > HI)).mean()) for R in ("low", "high")}
        mino = minority[n]
        ind = (v < -9.0).astype(float)
        st = _indicator_stats(ind, GATE_DEFAULTS["min_minority_count"])
        rind = ((v < LO) if mino == "low" else (v > HI)).astype(float)
        rst = _indicator_stats(rind, GATE_DEFAULTS["min_minority_count"])
        bins[n] = {
            "minority": mino, "occ_low": occ["low"], "occ_high": occ["high"],
            "excursions": ev[mino]["entries"] + ev[mino]["exits"],
            "entries_per_chain": ev[mino]["entries_per_chain"],
            "chains_with_events": ev[mino]["chains_with_events"],
            "max_sojourn_frac": ev[mino]["sojourns"]["max_frac"],
            "occ_lt9": float(ind.mean()), "mcse_lt9": _conservative_se(ind, st, 20),
            "mcse_minority": _conservative_se(rind, rst, 20), "ind_ess_minority": rst["ess"],
            "rhat": par[n]["rhat"], "min_ess": mins[n],
        }
    out = {
        "run": name, "C": C, "N": N, "sha": meta["git"]["sha"], "sampling_hours": samp_h,
        "warmup_seconds": meta.get("warmup_seconds"), "mean_steps": float(ns.mean()),
        "lockstep_steps": float(ns.max(0).mean()), "accept": float(r["accept_prob"][:, :N].mean()),
        "divergences": int(r["diverging"][:, :N].sum()), "step_size": meta.get("step_size"),
        "worst_param": worst, "worst_min_ess": mins[worst], "worst_rhat": max(s["rhat"] for s in par.values()),
        "n_rhat_fail": sum(s["rhat"] >= 1.01 for s in par.values()),
        "worst_ess_per_hour": mins[worst] / samp_h if samp_h > 0 else None,
        "median_min_ess": float(np.median(list(mins.values()))),
        "bins": bins,
    }
    if "jump_accept" in r:
        ja = r["jump_accept"][:, :N]
        out["jump_accept_rate_per_block"] = ja.mean(axis=(0, 1)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--draws", type=int, default=None)
    args = ap.parse_args()
    # minority region per bin from the pooled draws of all compared runs (same for every run)
    pooled = [load_run(r) for r in args.runs]
    minority = {}
    for k in range(30):
        n = f"gw_log10_rho_{k}"
        v = np.concatenate([p["x"][..., p["names"].index(n)].ravel() for p in pooled])
        minority[n] = "low" if (v < LO).mean() <= (v > HI).mean() else "high"
    res = [summarize(r, args.draws, minority) for r in args.runs]
    print("| | " + " | ".join(r["run"] for r in res) + " |")
    print("|---|" + "---|" * len(res))
    for k, lab, fmt in (("C", "chains", "{}"), ("N", "draws per chain", "{}"), ("sampling_hours", "sampling wall [h]", "{:.2f}"),
                        ("mean_steps", "mean leapfrog steps", "{:.0f}"), ("lockstep_steps", "lockstep steps", "{:.0f}"),
                        ("accept", "accept", "{:.3f}"), ("divergences", "divergences", "{}"),
                        ("worst_rhat", "max R-hat", "{:.3f}"), ("n_rhat_fail", "params R-hat >= 1.01", "{}"),
                        ("worst_param", "worst parameter (min bulk/tail ESS)", "{}"), ("worst_min_ess", "its ESS", "{:.0f}"),
                        ("worst_ess_per_hour", "worst ESS per sampling hour", "{:.0f}"), ("median_min_ess", "median param min ESS", "{:.0f}")):
        print(f"| {lab} | " + " | ".join(fmt.format(r[k]) if r[k] is not None else "-" for r in res) + " |")
    print()
    print("| bin | " + " | ".join(f"{r['run']}: minority, occ, excursions (chains), max sojourn, MCSE(<-9), R-hat" for r in res) + " |")
    print("|---|" + "---|" * len(res))
    for k in range(30):
        n = f"gw_log10_rho_{k}"
        cells = []
        for r in res:
            b = r["bins"][n]
            occ = b["occ_low"] if b["minority"] == "low" else b["occ_high"]
            msf = "-" if b["max_sojourn_frac"] is None else f"{b['max_sojourn_frac']:.2f}"
            cells.append(f"{b['minority']} {100 * occ:.1f}%, {b['excursions']} ({b['chains_with_events']}), {msf}, "
                         f"{b['mcse_lt9']:.3f}, {b['rhat']:.3f}")
        print(f"| f_{k + 1} | " + " | ".join(cells) + " |")
    for r in res:
        if "jump_accept_rate_per_block" in r:
            a = np.array(r["jump_accept_rate_per_block"])
            print(f"{r['run']}: jump acceptance per block: bins {np.round(a[:30], 2).tolist()}, IRN pairs {np.round(a[30:], 2).tolist()}")
    tag = "_".join(args.runs) + (f"_N{args.draws}" if args.draws else "")
    save_json(res, ROOT / "outputs" / "m2" / f"fs_compare_{tag}.json")


if __name__ == "__main__":
    main()
