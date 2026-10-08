"""HD^free pilot diagnostics (docs/FS_PILOT.md): mode crossings, IRN funnel indicators, sampler cost
and a length projection for the full hd_fs30_v2 run.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_pilot_diag.py --run hd_fs30_v2_pilot

Needs ``outputs/m2/freespec_gate_<run>.json`` (scripts/m2_freespec_diag.py) for the released-core
occupancies. Writes ``outputs/m2/fs_pilot_<run>.json`` and prints the tables.

Definitions
-----------
* occupancy: fraction of draws with log10_rho < -9 (the gate's indicator).
* crossings at -9: number of changes of that indicator along a chain (counts jitter at -9).
* transitions (hysteresis): completed moves between "low" (log10_rho < -10) and "signal"
  (log10_rho > -8); the state is kept while the chain is in between. This is the count that
  matters for mixing between the two regions.
* indicator ESS per draw implied by a two-state chain with occupancy p (released mass below -10
  relative to below -10 plus above -8) and one-way transition rate
  f per draw (f = transitions / 2 / draws): ESS/draw ~= f / (2 p (1 - p)) for small rates, so an
  indicator ESS of 400 needs ~ 800 p (1 - p) / f draws in total.
* "relevant" bins: the released core has >= 0.5% of its draws both below -10 and above -8 (both
  regions carry posterior mass, so a converged run must move between them).
* bins are 0-based (gw_log10_rho_k is frequency bin f_{k+1}).
* funnel indicators per pulsar (z = unconstrained coordinate, eps = adapted step size, m = the
  diagonal inverse metric): mobility = mean |dz_gamma| per draw divided by the conditional sd of
  z_gamma, in the lowest vs highest log10_A tercile; and the Spearman correlation of the
  step-size-normalised move |dz_gamma| / (eps sqrt(m)) with log10_A.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from m2_common import ROOT, released, save_json
from scipy import stats

from ptagwb.diagnostics import ess_bulk, ess_tail, rhat
from ptagwb.sampling import load_run

LOW, HIGH, THR = -10.0, -8.0, -9.0


def transitions(x: np.ndarray, lo=LOW, hi=HIGH) -> int:
    state, n = None, 0
    for v in x:
        s = 0 if v < lo else (1 if v > hi else state)
        if state is not None and s != state:
            n += 1
        state = s
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hd_fs30_v2_pilot")
    ap.add_argument("--full-chains", type=int, default=8)
    ap.add_argument("--full-draws", type=int, default=750)
    args = ap.parse_args()
    r = load_run(args.run)
    meta, names = r["meta"], r["names"]
    x, z = r["x"], r["z"]
    C, N, D = x.shape
    ns = r["num_steps"].astype(float)
    eps = np.asarray(r["step_size"], float)
    imm = np.asarray(r["inverse_mass_matrix"], float)
    if imm.ndim == 3:
        imm = np.diagonal(imm, axis1=1, axis2=2)
    gate = json.loads((ROOT / "outputs" / "m2" / f"freespec_gate_{args.run}.json").read_text())
    m2 = json.loads((ROOT / "outputs" / "m2" / "freespec_gate_hd_fs30.json").read_text())
    out = {"run": args.run, "C": C, "N": N, "sha": meta["git"]["sha"]}

    # ---------------- sampler cost
    lock = ns.max(axis=0)  # vectorised chains: every chain pays the longest tree of the iteration
    grads = float(ns.sum())
    samp_s, warm_s = meta.get("sampling_seconds", np.nan), meta.get("warmup_seconds", np.nan)
    dh = np.unique(r["tree_depth"], return_counts=True)
    cost = {
        "warmup_seconds": warm_s,
        "warmup_grad_evals": meta.get("warmup_grad_evals"),
        "warmup_divergences": meta.get("warmup_divergences"),
        "warmup_steps_per_chain_iter": meta.get("warmup_grad_evals", np.nan) / (C * meta["config"]["num_warmup"]),
        "warmup_seconds_per_iter": warm_s / meta["config"]["num_warmup"],
        "sampling_seconds": samp_s,
        "sampling_seconds_per_iter": samp_s / N,
        "seconds_per_lockstep_step": samp_s / lock.sum(),
        "mean_steps": ns.mean(),
        "mean_lockstep_max_steps": lock.mean(),
        "lockstep_efficiency": ns.mean() / lock.mean(),
        "tree_depth_hist": {int(k): int(v) for k, v in zip(*dh)},
        "steps_per_chain_mean": ns.mean(axis=1).tolist(),
        "divergences": int(r["diverging"].sum()),
        "accept_mean": float(r["accept_prob"].mean()),
        "accept_per_chain": r["accept_prob"].mean(axis=1).tolist(),
        "step_size": eps.tolist(),
        "max_depth_hits": int((r["tree_depth"] >= meta["config"]["max_tree_depth"] + 1).sum() + (ns >= 1023).sum()),
    }
    out["cost"] = cost
    print(f"== cost: warmup {warm_s / 60:.1f} min ({cost['warmup_steps_per_chain_iter']:.0f} steps/chain-iter, "
          f"{cost['warmup_seconds_per_iter']:.1f} s/iter, {cost['warmup_divergences']} div); sampling {samp_s / 60:.1f} min "
          f"({cost['sampling_seconds_per_iter']:.1f} s/iter, {cost['seconds_per_lockstep_step'] * 1e3:.0f} ms per 8-chain step)")
    print(f"   steps/draw mean {ns.mean():.0f}, lockstep max {lock.mean():.0f} (efficiency {cost['lockstep_efficiency']:.2f}); "
          f"depth hist {cost['tree_depth_hist']}; div {cost['divergences']}; accept {cost['accept_mean']:.3f}; "
          f"step sizes {np.round(eps, 4).tolist()}")

    # ---------------- per-parameter diagnostics
    par = {}
    for j, n in enumerate(names):
        a = x[:, :, j]
        par[n] = {"rhat": rhat(a), "ess_bulk": ess_bulk(a), "ess_tail": ess_tail(a)}
    out["parameters"] = par
    fails = [n for n, s in par.items() if not (s["rhat"] < 1.01 and s["ess_bulk"] >= 400 and s["ess_tail"] >= 400)]
    worst_rhat = sorted(par, key=lambda n: -par[n]["rhat"])[:12]
    print(f"== parameters failing R-hat<1.01 / ESS>=400: {len(fails)} of {D}; max R-hat {par[worst_rhat[0]]['rhat']:.3f}")
    min_ess = {n: min(s["ess_bulk"], s["ess_tail"]) for n, s in par.items()}
    out["ess_per_grad"] = {n: min_ess[n] / grads for n in names}
    out["min_ess_per_draw"] = {n: min_ess[n] / (C * N) for n in names}

    # ---------------- free-spectrum bins: occupancy and crossings
    rel = released("hd_fs30")
    bins = []
    for k in range(30):
        n = f"gw_log10_rho_{k}"
        j = names.index(n)
        a = x[:, :, j]
        occ = (a < THR).mean(axis=1)
        cr = [int(np.sum(np.diff((c < THR).astype(int)) != 0)) for c in a]
        tr = [transitions(c) for c in a]
        ref = gate["bins"][n].get("agreement", {}).get("reference_occupancy", np.nan)
        m2occ = m2["bins"][n]["occupancy_per_chain"]
        g = gate["bins"][n]
        pooled = float((a < THR).mean())
        rk = np.asarray(rel[n], float)
        rel_lo, rel_hi = float((rk < LOW).mean()), float((rk > HIGH).mean())
        f = sum(tr) / 2 / (C * N)  # one-way transitions per draw
        p = rel_lo / (rel_lo + rel_hi) if rel_lo + rel_hi > 0 else 0.0  # two-state occupancy (released)
        need = 800 * p * (1 - p) / f if f > 0 else np.inf
        # with zero transitions: 95% Poisson upper bound on the rate (3 events) -> lower bound on the length
        need_lb = 800 * p * (1 - p) / (max(sum(tr), 3) / 2 / (C * N))
        ind = g["indicator"]
        need_ind_gate = 400 / (ind["ess"] / (C * N)) if ind.get("available") and ind.get("ess") else np.inf
        need_par = 400 / (min(par[n]["ess_bulk"], par[n]["ess_tail"]) / (C * N))
        bins.append({
            "bin": k, "released_frac_below_-10": rel_lo, "released_frac_above_-8": rel_hi,
            "both_regions_relevant": bool(min(rel_lo, rel_hi) >= 0.005), "occupancy_per_chain": occ.tolist(), "pooled": pooled, "released": ref, "m2_per_chain": m2occ,
            "crossings_at_-9": cr, "transitions": tr, "transitions_total": int(sum(tr)),
            "chains_with_transition": int(sum(t > 0 for t in tr)),
            "chains_in_both_regions": int(sum((c < LOW).any() and (c > HIGH).any() for c in a)),
            "indicator": g["indicator"], "rhat": par[n]["rhat"], "ess_bulk": par[n]["ess_bulk"], "ess_tail": par[n]["ess_tail"],
            "one_way_rate_per_draw": f, "draws_needed_two_state_ess400": need, "draws_needed_two_state_or_bound": need_lb,
            "draws_needed_gate_indicator_ess400": need_ind_gate, "draws_needed_param_ess400": need_par,
            "min_x": float(a.min()), "max_x": float(a.max()),
        })
    out["bins"] = bins
    print("== bins: k | occupancy per chain (<-9) | pooled | released | M2 per chain | transitions per chain (-10/-8) | "
          "crossings@-9 total | R-hat | bulk/tail ESS")
    for b in bins:
        print(f"{b['bin']:2d} | {' '.join(f'{v:.2f}' for v in b['occupancy_per_chain'])} | {b['pooled']:.3f} | "
              f"{b['released']:.3f} | {' '.join(f'{v:.2f}' for v in b['m2_per_chain'])} | {' '.join(map(str, b['transitions']))} "
              f"(chains {b['chains_with_transition']}) | {sum(b['crossings_at_-9'])} | {b['rhat']:.3f} | "
              f"{b['ess_bulk']:.0f}/{b['ess_tail']:.0f} | released <-10 / >-8: {b['released_frac_below_-10']:.3f} / "
              f"{b['released_frac_above_-8']:.3f}")

    # ---------------- IRN funnel indicators
    P = (D - 30) // 2
    psrs = [n[: -len("_red_noise_log10_A")] for n in names[:P]]
    funnel = {}
    for i, p in enumerate(psrs):
        jA, jg = i, P + i
        A = x[:, :, jA]
        dzg = np.abs(np.diff(z[:, :, jg], axis=1))
        dza = np.abs(np.diff(z[:, :, jA], axis=1))
        A0 = A[:, :-1]
        norm = (eps[:, None] * np.sqrt(imm[:, jg])[:, None])
        rho_s = stats.spearmanr((dzg / norm).ravel(), A0.ravel()).statistic
        q1, q2 = np.quantile(A0, [1 / 3, 2 / 3])
        lo_m, hi_m = A0 <= q1, A0 >= q2
        zg = z[:, :-1, jg]
        sd_lo, sd_hi = zg[lo_m].std(), zg[hi_m].std()
        mob_lo = dzg[lo_m].mean() / max(sd_lo, 1e-12)
        mob_hi = dzg[hi_m].mean() / max(sd_hi, 1e-12)
        nsr = stats.spearmanr(ns[:, 1:].ravel(), A0.ravel()).statistic
        rho_j = [names.index(f"gw_log10_rho_{k}") for k in range(30)]
        cb = [stats.spearmanr(A.ravel(), x[:, :, jj].ravel()).statistic for jj in rho_j]
        kb = int(np.nanargmax(np.abs(cb)))
        funnel[p] = {
            "max_abs_spearman_log10A_vs_bin": float(cb[kb]), "coupled_bin": kb,
            "log10_A": {**par[names[jA]]}, "gamma": {**par[names[jg]]},
            "spearman_normmove_gamma_vs_log10A": float(rho_s),
            "sd_zgamma_lowA": float(sd_lo), "sd_zgamma_highA": float(sd_hi),
            "width_ratio_low_high": float(sd_lo / max(sd_hi, 1e-12)),
            "mobility_lowA": float(mob_lo), "mobility_highA": float(mob_hi),
            "mobility_A_lowA": float(dza[lo_m].mean() / max(z[:, :-1, jA][lo_m].std(), 1e-12)),
            "spearman_treesteps_vs_log10A": float(nsr),
            "log10A_range": [float(A.min()), float(A.max())],
            "log10A_median_per_chain": np.median(A, axis=1).tolist(),
        }
    out["funnel"] = funnel
    worst = sorted(psrs, key=lambda p: -max(funnel[p]["log10_A"]["rhat"], funnel[p]["gamma"]["rhat"]))[:10]
    print("== worst IRN (by R-hat): psr | R-hat A/gamma | min ESS A/gamma | width ratio sd(z_g|lowA)/sd(z_g|highA) | "
          "mobility low/high A | rho(norm move, log10A) | rho(tree steps, log10A) | mobility of z_A in low-A tercile | median log10A per chain | most correlated bin (Spearman)")
    for p in worst:
        f = funnel[p]
        print(f"{p:12s} | {f['log10_A']['rhat']:.3f}/{f['gamma']['rhat']:.3f} | "
              f"{min(f['log10_A']['ess_bulk'], f['log10_A']['ess_tail']):.0f}/{min(f['gamma']['ess_bulk'], f['gamma']['ess_tail']):.0f} | "
              f"{f['width_ratio_low_high']:.2f} | {f['mobility_lowA']:.3f}/{f['mobility_highA']:.3f} | "
              f"{f['spearman_normmove_gamma_vs_log10A']:+.2f} | {f['spearman_treesteps_vs_log10A']:+.2f} | "
              f"{f['mobility_A_lowA']:.3f} | {' '.join(f'{v:.1f}' for v in f['log10A_median_per_chain'])} | "
              f"bin {f['coupled_bin']} rho {f['max_abs_spearman_log10A_vs_bin']:+.2f}")
    out["worst_irn"] = worst

    # ---------------- projection
    min_ess_draw = min(out["min_ess_per_draw"].values())
    worst_par = min(out["min_ess_per_draw"], key=out["min_ess_per_draw"].get)
    s_iter = cost["sampling_seconds_per_iter"]
    Cf = args.full_chains
    need_par = 400 / min_ess_draw / Cf  # draws per chain for ESS >= 400 at the measured ESS per draw
    finite_need = [b["draws_needed_two_state_ess400"] for b in bins if np.isfinite(b["draws_needed_two_state_ess400"])]
    relevant = [b for b in bins if b["both_regions_relevant"]]
    need_ind = max([b["draws_needed_two_state_or_bound"] for b in relevant], default=0) / Cf
    proj = {
        "min_ess_per_draw": min_ess_draw, "worst_parameter": worst_par,
        "draws_per_chain_for_param_ess400": need_par,
        "draws_per_chain_for_indicator_ess400_lower_bound": need_ind,
        "max_finite_indicator_draws_per_chain": (max(finite_need) / Cf) if finite_need else None,
        "zero_transition_relevant_bins": [b["bin"] for b in relevant if b["transitions_total"] == 0],
        # sampling phase only: the pilots' warmup cost is not representative of a v2 warmup (docs/FS_PILOT.md)
        "planned_draws_sampling_hours": args.full_draws * s_iter / 3600,
        "hours_per_1000_draws": 1000 * s_iter / 3600,
    }
    out["projection"] = proj
    print("== projection:", json.dumps(proj, indent=1, default=float))
    save_json(out, ROOT / "outputs" / "m2" / f"fs_pilot_{args.run}.json")


if __name__ == "__main__":
    main()
