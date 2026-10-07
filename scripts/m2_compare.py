"""Compare M2 runs with the released NG15 chains (MCSE-aware) and collect run diagnostics.

    uv run --no-sync python scripts/m2_compare.py

For every (our run, released chain) pair: 5/50/95% quantiles with Monte-Carlo standard errors,
z = (ours - released) / sqrt(mcse_ours^2 + mcse_rel^2) per quantile, two-sample KS with an
ESS-based p-value, and for 2-D (gamma, log10_A) the energy distance with a permutation-free
reference (energy distance between two halves of the released chain, i.e. the MC-noise floor).
Diagnostics per run: max R-hat, min bulk/tail ESS over all parameters, divergences, tree depth,
step size, wall time, gradient evaluations, ESS/s. Writes outputs/m2/compare.json.
"""

from __future__ import annotations

import sys

import numpy as np
from m2_common import ROOT, released, run_draws, save_json

from ptagwb.diagnostics import energy_distance, ks_ess, summarize
from ptagwb.sampling import load_run

# (our run, released chain key, parameters)
PAIRS = [
    ("curn_g433_14f", "curn_g433_tut", ["gw_log10_A"]),
    ("curn_vg_14f", "curn_vg_m2a", ["gw_gamma", "gw_log10_A"]),
    ("curn_vg_14f", "curn_vg_hm", ["gw_gamma", "gw_log10_A"]),
    ("hd_g433_14f", "hd_g433", ["gw_log10_A"]),
    ("hd_g433_14f", "hd_g433_tut", ["gw_log10_A"]),
    ("hd_vg_14f", "hd_vg", ["gw_gamma", "gw_log10_A"]),
    ("hd_vg_14f", "hd_vg_m3a", ["gw_gamma", "gw_log10_A"]),
    ("hd_vg_14f", "hd_vg_hm", ["gw_gamma", "gw_log10_A"]),
    ("hd_fs30", "hd_fs30", [f"gw_log10_rho_{i}" for i in range(10)]),
]
ICRS = {"hd_g433_14f": "hd_g433_14f_icrs", "hd_vg_14f": "hd_vg_14f_icrs"}
IRN_SPOT = ["B1937+21", "J1713+0747", "J1012+5307", "J0610-2100", "J1909-3744", "J1903+0327"]
IRN_PAIRS = [("curn_vg_14f", "curn_vg_m2a"), ("hd_vg_14f", "hd_vg_m3a")]


def _rel_summary(x):
    return summarize(np.asarray(x)[None, :])


def compare_param(ours: np.ndarray, rel: np.ndarray) -> dict:
    so, sr = summarize(ours), _rel_summary(rel)
    out = {"ours": so, "released": sr}
    for q in ("q05", "q50", "q95"):
        d = so[q] - sr[q]
        se = np.hypot(so[q + "_mcse"], sr[q + "_mcse"])
        out[f"z_{q}"] = float(d / se)
        out[f"d_{q}"] = float(d)
    out["ks"] = ks_ess(ours, rel, so["ess_bulk"], sr["ess_bulk"])
    return out


def run_diagnostics(run: dict) -> dict:
    x = run["x"]
    C, N, D = x.shape
    summ = [summarize(x[..., j]) for j in range(D)]
    m = run["meta"]
    tot_grad = m.get("warmup_grad_evals", 0) + m.get("sampling_grad_evals", 0)
    secs = m.get("warmup_seconds", 0) + m.get("sampling_seconds", 0)
    common = [j for j, n in enumerate(run["names"]) if n.startswith("gw_")]
    ess_common = min(summ[j]["ess_bulk"] for j in common)
    return {
        "chains": C,
        "draws_per_chain": N,
        "dim": D,
        "max_rhat": float(max(s["rhat"] for s in summ)),
        "argmax_rhat": run["names"][int(np.argmax([s["rhat"] for s in summ]))],
        "min_ess_bulk": float(min(s["ess_bulk"] for s in summ)),
        "min_ess_tail": float(min(s["ess_tail"] for s in summ)),
        "common_rhat": {run["names"][j]: summ[j]["rhat"] for j in common},
        "common_ess_bulk": {run["names"][j]: summ[j]["ess_bulk"] for j in common},
        "common_ess_tail": {run["names"][j]: summ[j]["ess_tail"] for j in common},
        "divergences": int(np.sum(run["diverging"])),
        "warmup_divergences": m.get("warmup_divergences"),
        "max_tree_depth_hits": int(np.sum(run["num_steps"] >= 2 ** m["config"]["max_tree_depth"] - 1)),
        "mean_tree_depth": float(np.mean(run["tree_depth"])),
        "mean_steps": float(np.mean(run["num_steps"])),
        "mean_accept": float(np.mean(run["accept_prob"])),
        "step_size": m.get("step_size"),
        "warmup_seconds": m.get("warmup_seconds"),
        "sampling_seconds": m.get("sampling_seconds"),
        "grad_evals": tot_grad,
        "wall_seconds": secs,
        "ess_per_s_common_sampling": float(ess_common / m["sampling_seconds"]) if m.get("sampling_seconds") else None,
        "ess_per_1000_grads_common": float(1000 * ess_common / max(m.get("sampling_grad_evals", 1), 1)),
        "git_sha": m["git"]["sha"],
        "git_dirty": m["git"]["dirty_files"],
    }


def energy2d(ours, rel, seed=0, ours_halves=None):
    rng = np.random.default_rng(seed)
    e = energy_distance(ours, rel, seed=seed)
    own = energy_distance(*ours_halves, seed=seed) if ours_halves is not None else float("nan")
    # MC noise floor: two disjoint random halves of the released chain, thinned like ours
    idx = rng.permutation(len(rel))
    h = len(rel) // 2
    floor = energy_distance(rel[idx[:h]], rel[idx[h:]], seed=seed)
    return {"energy": e, "released_split_floor": floor, "ours_split_floor": own}


def main():
    out = {"pairs": [], "diagnostics": {}, "icrs_shift": {}, "irn": []}
    runs = {}

    def get(name):
        if name not in runs:
            try:
                runs[name] = load_run(name)
            except FileNotFoundError:
                runs[name] = None
        return runs[name]

    for rn, key, pars in PAIRS:
        r = get(rn)
        if r is None:
            continue
        rel = released(key)
        entry = {"run": rn, "released": key, "params": {}}
        for p in pars:
            entry["params"][p] = compare_param(run_draws(r, p), rel[p])
        if "gw_gamma" in pars:
            ours2 = np.column_stack([run_draws(r, "gw_gamma").ravel(), run_draws(r, "gw_log10_A").ravel()])
            rel2 = np.column_stack([rel["gw_gamma"], rel["gw_log10_A"]])
            g3, a3 = run_draws(r, "gw_gamma"), run_draws(r, "gw_log10_A")
            h = g3.shape[0] // 2
            halves = tuple(np.column_stack([g3[sl].ravel(), a3[sl].ravel()]) for sl in (slice(0, h), slice(h, None)))
            entry["energy_2d"] = energy2d(ours2, rel2, ours_halves=halves)
        out["pairs"].append(entry)
        print(rn, key, {p: (round(v["z_q05"], 2), round(v["z_q50"], 2), round(v["z_q95"], 2)) for p, v in entry["params"].items()})
    for rn, ri in ICRS.items():
        a, b = get(rn), get(ri)
        if a is None or b is None:
            continue
        d = {}
        for p in [n for n in a["names"] if n.startswith("gw_")]:
            d[p] = compare_param(run_draws(b, p), run_draws(a, p).ravel())  # 'released' slot = enterprise-pos run
        out["icrs_shift"][rn] = d
    for rn, key in IRN_PAIRS:
        r = get(rn)
        if r is None:
            continue
        rel = released(key)
        for psr in IRN_SPOT:
            for par in ("log10_A", "gamma"):
                n = f"{psr}_red_noise_{par}"
                c = compare_param(run_draws(r, n), rel[n])
                out["irn"].append({"run": rn, "released_chain": key, "param": n, **c})
    for name, r in runs.items():
        if r is not None:
            out["diagnostics"][name] = run_diagnostics(r)
    for name in ["curn_vg_5f", "hd_vg_5f", "hd_g433_14f_icrs", "hd_vg_14f_icrs", "curn_fs30"]:
        r = get(name)
        if r is not None and name not in out["diagnostics"]:
            out["diagnostics"][name] = run_diagnostics(r)
    save_json(out, ROOT / "outputs" / "m2" / "compare.json")
    for n, d in out["diagnostics"].items():
        print(n, {k: d[k] for k in ("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "mean_tree_depth", "wall_seconds")})


if __name__ == "__main__":
    sys.exit(main())
