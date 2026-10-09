"""EPTA pilot report: mixing, transport and the production projection against the pre-registered
pilot-v2 release rules (review of pilot v1, ``review_epta_pilot1.out``; docs/M3B0_VALIDATION.md
Sec. 9b). CPU only; reads a finished run directory, never samples.

Rules (frozen before the v2 run; ``RULES``):
* evidence: >= 300 post-warmup transitions per chain, else "insufficient pilot evidence";
* release screening: zero post-warmup divergences; rank-normalised/folded split R-hat < 1.01 for every
  parameter; bulk and tail ESS >= 100 for every parameter and >= 200 for the targets; every
  Sec. 5.3 transport item outside the frozen exclusions U PASS (region-indicator evidence: entries,
  exits, chains, sojourn, MCSE; ESS or MH acceptance never substitute);
* projection: per required output (headline quantities and E-6 under their frozen max_our_MCSE,
  the production floors ESS 400 all / 1,000 targets), the post-warmup chain-transitions needed =
  N_pilot (measured MCSE / allowed MCSE)^2 or N_pilot (ESS floor / measured ESS); the most demanding
  one, sampling time doubled, plus the measured warmup and the HD-reweighting cost; HD quantities
  and ln B_D use log w = lnL_HD - lnL_CURN on our pilot draws, on the common domain D.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_pilot_report.py RUN_ID [--hd] [--workers N]
Writes data/processed/m3b/epta/runs/RUN_ID/pilot_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ptagwb import diagnostics as D
from ptagwb.acceptance import (
    Occupancy,
    aggregate,
    conditional_quantile,
    d9_exclusions,
    domain_indicator,
    event_interval_indicators,
    occupancy,
    shelf_peak_indicators,
    support_class,
    transport_gate,
)
from ptagwb.config import REPO_ROOT

EPTA = REPO_ROOT / "data" / "processed" / "m3b" / "epta"
RUNS, RES = EPTA / "runs", EPTA / "results"
ACC = REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json"
REL = REPO_ROOT / "configs" / "m3b" / "relevance" / "epta.json"
TARGETS = ("gw_crn_log10_A", "gw_crn_gamma")
RULES = {"min_post_warmup_per_chain": 300, "rhat_max": 1.01, "ess_min_all": 100, "ess_min_targets": 200,
         "divergences_post_warmup": 0, "production_floor_ess_all": 400, "production_floor_ess_targets": 1000,
         "e6_max_se": 0.10, "sampling_safety_factor": 2.0, "cap_total_h": 12.0, "d3_benchmark_h": 0.21,
         "cap_production_h": 8.0}


def ref_occ(d: dict) -> Occupancy:
    return Occupancy(p_hat=d["p_hat"], p_lo=d["p_lo"], p_hi=d["p_hi"], status=d["status"], case=d["case"],
                     n_entries=d["n_entries"], n_exits=d["n_exits"], n_draws=0, n_chains=0, n_chains_visiting=0,
                     tau=d["tau"], n_eff=d["n_eff"], mcse=d["mcse"], longest_sojourn_frac=None)


def occ_dict(o: Occupancy) -> dict:
    return {k: getattr(o, k) for k in ("p_hat", "p_lo", "p_hi", "status", "case", "n_entries", "n_exits",
                                       "n_chains_visiting", "mcse", "longest_sojourn_frac")}


def load(run_dir):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    arr = {}
    for f in sorted(run_dir.glob("chunk_*.npz")):
        with np.load(f) as z:
            for k in z.files:
                if k != "first_transition":
                    arr.setdefault(k, []).append(z[k])
    return meta, {k: np.concatenate(v) for k, v in arr.items()}


def mixing(xs: np.ndarray, names: list[str]) -> dict:
    """xs (chains, draws, params) -> per-parameter R-hat, bulk/tail ESS."""
    return {nm: {"rhat": float(D.rhat(xs[:, :, j])), "ess_bulk": float(D.ess_bulk(xs[:, :, j])),
                 "ess_tail": float(D.ess_tail(xs[:, :, j]))} for j, nm in enumerate(names)}


def transport(xs, names, model_key, rel, excluded_params):
    items = []
    for ca in rel["class_a"]:
        if ca["model"] != model_key or ca["param"] not in names:
            continue
        j = names.index(ca["param"])
        ind = shelf_peak_indicators(list(xs[:, :, j]), ca["regions"]["S"][0])
        ours = (occupancy(ind["S"]), occupancy(ind["P"]))
        ref = (ref_occ(ca["reference"]["S"]), ref_occ(ca["reference"]["P"]))
        g = transport_gate(ref, ours)
        items.append({"param": ca["param"], "in_U": ca["param"] in excluded_params, "status": g.status,
                      "ref_support": [support_class(o) for o in ref],
                      "ours": {"S": occ_dict(ours[0]), "P": occ_dict(ours[1])}, "reasons": list(g.reasons)})
    return items


def dip(xs, names, rel, model_key):
    cb = rel["class_b"][model_key]
    j = names.index(cb["param"])
    ind = event_interval_indicators(list(xs[:, :, j]), [tuple(i) for i in cb["intervals"]], tuple(cb["window_mjd"]))
    return {"param": cb["param"], "occupancy": {k: occ_dict(occupancy(v)) for k, v in ind.items()},
            "t0_changes": int(np.sum(np.diff(xs[:, :, j], axis=1) != 0)),
            "note": "the reference t0 'rest' bin is in U; on D the dip epoch is in I0 by construction"}


def evaluate_release(n_post_per_chain: int, div_post: int, per: dict, targets, transport_items, rules=RULES) -> dict:
    """The pre-registered pilot-v2 screening rules (pure; unit-tested)."""
    out = {"sufficient_evidence": n_post_per_chain >= rules["min_post_warmup_per_chain"]}
    out["divergences"] = div_post <= rules["divergences_post_warmup"]
    out["rhat"] = bool(per) and all(v["rhat"] < rules["rhat_max"] for v in per.values())
    out["ess_all"] = bool(per) and all(min(v["ess_bulk"], v["ess_tail"]) >= rules["ess_min_all"] for v in per.values())
    out["ess_targets"] = all(t in per and min(per[t]["ess_bulk"], per[t]["ess_tail"]) >= rules["ess_min_targets"]
                             for t in targets)
    outside = [i for i in transport_items if not i["in_U"]]
    out["transport_outside_U"] = bool(outside) and all(i["status"] == "PASS" for i in outside)
    out["failed"] = [k for k, v in out.items() if v is False]
    out["screening_pass"] = not out["failed"]
    return out


def projection(n_post_chain_tr: int, chains: int, s_per_transition: float, warmup_s: float, requirements: list,
               hd_s_per_draw: float, pilot_used_h: float, rules=RULES) -> dict:
    """requirements: dicts with 'id' and 'factor' = (needed / pilot) chain-transition ratio."""
    worst = max(requirements, key=lambda r: r["factor"])
    n_req = worst["factor"] * n_post_chain_tr
    samp_h = rules["sampling_safety_factor"] * (n_req / chains) * s_per_transition / 3600.0
    run_a = samp_h + warmup_s / 3600.0 + n_req * hd_s_per_draw / 3600.0
    remaining_total = rules["cap_total_h"] - rules["d3_benchmark_h"] - pilot_used_h
    return {"most_demanding": worst, "chain_transitions_needed": n_req, "run_A_gpu_h": run_a,
            "run_A_breakdown": {"sampling_x2_h": samp_h, "warmup_h": warmup_s / 3600.0,
                                "hd_reweighting_h": n_req * hd_s_per_draw / 3600.0},
            "remaining_total_h": remaining_total, "production_phase_cap_h": rules["cap_production_h"],
            "run_A_fits": run_a <= min(rules["cap_production_h"], remaining_total),
            "run_B_note": "fixed gamma 13/3 was not piloted (its pilot does not fit the pilot phase); assuming "
                          "run A's efficiency it would cost about the same, within what remains after run A",
            "A_plus_B_same_efficiency_h": 2 * run_a,
            "A_plus_B_fits": 2 * run_a <= min(rules["cap_production_h"], remaining_total)}


_W = {}


def _init():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from ptagwb import epta
    from ptagwb.fastcond import FastEPTA

    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    Mc, Mh = epta.EPTAModel(psrs, man, "crn"), epta.EPTAModel(psrs, man, "hd")
    _W.update(Fc=FastEPTA(Mc), Fh=FastEPTA(Mh), cmap=[Mc.param_names.index(n.replace("gw_hd_", "gw_crn_"))
                                                     for n in Mh.param_names])


def _logw(rows):
    return [_W["Fh"].logL(x[_W["cmap"]]) - _W["Fc"].logL(x) for x in rows]


def hd_log_weights(x_rows: np.ndarray, workers: int) -> np.ndarray:
    import multiprocessing as mp

    chunks = np.array_split(np.arange(len(x_rows)), max(1, workers * 4))
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        parts = pool.map(_logw, [x_rows[c] for c in chunks if c.size])
    return np.array([v for p in parts for v in p])


def report(run_id: str, hd: bool = False, workers: int = 8, run_dir=None) -> dict:
    run_dir = RUNS / run_id if run_dir is None else run_dir
    meta, a = load(run_dir)
    cfg, names = meta["config"], meta["param_names"]
    W, L = cfg["num_warmup"], cfg["chunk_transitions"]
    x = a["x"]
    T, C, _ = x.shape
    xs = np.transpose(x[W:], (1, 0, 2))
    N = xs.shape[1]
    led = next(e for e in json.loads((RUNS / "ledger.json").read_text()) if e["run_id"] == run_id)
    chunks = meta["chunks"]
    nw = W // L
    warm_s = chunks[nw - 1]["elapsed_s"] if len(chunks) >= nw else chunks[-1]["elapsed_s"]
    samp_s = float(sum(c["seconds"] for c in chunks[nw:]))
    s_tr = samp_s / N if N else None  # seconds per 4-chain (lockstep) post-warmup transition
    rel, acc = json.loads(REL.read_text()), json.loads(ACC.read_text())
    excl = d9_exclusions(acc)
    out = {"run_id": run_id, "head": meta["provenance"]["head"], "config_sha256": meta["provenance"]["config_sha256"],
           "stop_reason": meta.get("stop_reason"), "rules": RULES,
           "ledger": {"status": led["status"], "charged_s": led["seconds"], "charged_gpu_h": led["seconds"] / 3600},
           "transitions": {"total": int(T), "warmup": W, "post_warmup_per_chain": int(N), "chains": C},
           "timing": {"warmup_and_compile_s": warm_s, "sampling_s": samp_s, "s_per_transition_all_chains": s_tr},
           "divergences": {"post_warmup": int(a["diverging"][W:].sum()), "warmup": int(a["diverging"][:W].sum())},
           "t0_mh_accept": float(a["t0_accept"][W:].mean() / cfg["n_t0_mh"]) if N else None,
           "block_mh_accept": [{"block": "+".join(names[i] for i in idx), "accept": float(r)}
                               for idx, r in zip(meta["block_idx"], a["block_accept"][W:].mean(axis=(0, 1)))]
           if N else None,
           "nuts": {"accept_prob_mean": float(a["accept_prob"][W:].mean()) if N else None,
                    "num_steps_median": float(np.median(a["num_steps"][W:])) if N else None,
                    "step_size_final": a["step_size"][-1].tolist() if "step_size" in a else None}}
    if N < 8:
        out["release"] = {"sufficient_evidence": False, "screening_pass": False, "failed": ["sufficient_evidence"]}
        return out
    per = mixing(xs, names)
    excluded_params = {e["params"]["crn_pl"] for e in excl if e["type"] == "shelf"}
    items = transport(xs, names, "crn_pl", rel, excluded_params) if cfg["gamma_common"] is None else []
    hours = samp_s / 3600.0
    out["mixing"] = {"rhat_max": max(v["rhat"] for v in per.values()),
                     "ess_min_all": min(min(v["ess_bulk"], v["ess_tail"]) for v in per.values()),
                     "targets": {t: {**per[t], "ess_per_sampling_gpu_h": min(per[t]["ess_bulk"], per[t]["ess_tail"]) / hours,
                                     "ess_per_chain_transition": min(per[t]["ess_bulk"], per[t]["ess_tail"]) / (N * C)}
                                 for t in TARGETS if t in per},
                     "per_param": per}
    out["transport"] = {"counts_outside_U": {s: sum(i["status"] == s for i in items if not i["in_U"])
                                             for s in ("PASS", "FAIL", "INCONCLUSIVE")},
                        "aggregate_outside_U": aggregate([i["status"] for i in items if not i["in_U"]]) if items else None,
                        "items": items}
    out["dip"] = dip(xs, names, rel, "crn_pl")
    out["release"] = evaluate_release(N, out["divergences"]["post_warmup"], per, TARGETS, items)
    # projection requirements (factor = chain-transitions needed / pilot chain-transitions)
    col = {nm: list(xs[:, :, j]) for j, nm in enumerate(names)}
    ind = domain_indicator(lambda n: col[n], excl, "crn_pl")
    reqs = [{"id": "floor_all", "factor": RULES["production_floor_ess_all"] / out["mixing"]["ess_min_all"]}]
    for t in TARGETS:
        if t in per:
            e = min(per[t]["ess_bulk"], per[t]["ess_tail"])
            reqs.append({"id": f"floor_target {t}", "factor": RULES["production_floor_ess_targets"] / e})
    lw = None
    if hd:
        rows = x[W:].reshape(-1, x.shape[2])  # transition-major, chain-minor
        lw_flat = hd_log_weights(rows, workers)
        lw = list(lw_flat.reshape(N, C).T)
    for q in acc["quantities"]:
        if not q["headline"]:
            continue
        if q["model"] == "crn_pl":
            r = conditional_quantile(col[q["param"]], ind, q["quantile"])
        elif lw is not None:
            r = conditional_quantile(col[q["param"].replace("gw_hd_", "gw_crn_")], ind, q["quantile"], log_w=lw)
        else:
            continue
        reqs.append({"id": f"{q['id']} q{q['quantile']}", "mcse": r["mcse"], "max_our_mcse": q["max_our_mcse"],
                     "factor": (r["mcse"] / q["max_our_mcse"]) ** 2})
    if lw is not None:
        from ptagwb.reweight import mcse_lnbf_obm, psis_khat

        e6 = mcse_lnbf_obm(lw, mask=ind)
        reqs.append({"id": "E-6 ln B_D", "mcse": e6["mcse"], "max_our_mcse": RULES["e6_max_se"],
                     "factor": (e6["mcse"] / RULES["e6_max_se"]) ** 2})
        out["hd_reweighting"] = {"khat": float(psis_khat(lw)), "lnB_D_mcse": e6["mcse"]}
    hd_cost = json.loads((RES / "projection.json").read_text())["hd_value_seconds_per_draw"]
    out["projection"] = projection(N * C, C, s_tr, warm_s, reqs, hd_cost, led["seconds"] / 3600.0)
    out["projection"]["requirements"] = reqs
    out["projection"]["hd_included"] = bool(hd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--hd", action="store_true", help="HD reweighting of the pilot draws (CPU, FastEPTA)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-dir", type=Path, default=None, help="read the run from here (default runs/RUN_ID)")
    a = ap.parse_args()
    run_dir = a.run_dir or RUNS / a.run_id
    out = report(a.run_id, a.hd, a.workers, run_dir)
    (run_dir / "pilot_report.json").write_text(json.dumps(out, indent=1, default=str))
    show = {k: v for k, v in out.items() if k not in ("transport", "mixing")}
    if "mixing" in out:
        show["mixing"] = {k: v for k, v in out["mixing"].items() if k != "per_param"}
        show["transport"] = {k: v for k, v in out["transport"].items() if k != "items"}
    print(json.dumps(show, indent=1, default=str))


if __name__ == "__main__":
    main()
