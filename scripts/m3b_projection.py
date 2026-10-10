"""Per-run EPTA production cost projection (review of 2ee1bb7, item 4). CPU only: combines the D3
benchmark timings (results/bench.json), the frozen acceptance file and the reweighting
efficiencies measured on the released CURN draws (results/reference_weights.json).

**Provisional scenarios, not an allocation.** ESS per transition is bracketed (0.5 / 0.25 / 0.1
for the slowest target) until the pilot measures it with the complete kernel; the required ESS
values are projections from the released chains.

Runs (plan Sec. 5.1, 5.4):
* **A: CURN^gamma (free gamma)**, with HD^gamma and BF(HD/CURN) by reweighting its draws.
  Targets (frozen acceptance file v2: headline quantities under the revised rule 1.645 MCSE_ref <=
  m/2, conditional on the common domain D): the CURN quantities (E-3, E-4) directly; the HD
  quantities (E-1, E-2) as raw-weighted quantiles; E-6 with MCSE(ln BF) <= 0.10; and the convergence floor of
  1,000 bulk/tail ESS for target parameters.
  * direct quantity: ESS_req = ESS_ref (MCSE_ref / max_our_MCSE)^2 (reference ESS: bulk for the
    median, tail for outer quantiles);
  * reweighted HD quantity: ESS_req = ESS_ref,CURN (MCSE_rw / max_our_MCSE)^2, with MCSE_rw the
    ratio-estimator MCSE of the weighted quantile on the released CURN draws (every 5th retained
    draw; ESS_ref,CURN of the same parameter, bulk or tail). This prices the reweighting
    inefficiency, which a direct HD reference ESS does not;
  * E-6: ESS_req = ESS_ref,CURN(log10_A, bulk) (MCSE_ref(ln BF) / 0.10)^2.
* **B: CURN gamma = 13/3 (fixed gamma)**, with HD 13/3 by reweighting (E-5, secondary). No
  released fixed-gamma chain: an **illustrative** ESS of 1,000 (the convergence floor), to be
  replaced by the pilot.

Complete kernel cost per chain-transition (4 vectorised chains): the measured NUTS + 2 t0-MH +
refresh update (bench.json) plus the planned frozen block-MH jumps (plan Sec. 5.1): one value-only
evaluation per block per transition for 31 noise (log10_A, gamma) pairs, the common pair and the
joint (t0, log10_tau, log10_Amp) dip block = 33 evaluations at B = 4 (measured value-only cost).
Warmup adds 25 %. Reweighting: one HD value per retained draw (B = 8 cost). Pilot: 2 GPU-h.

Usage: PYTHONPATH=src python scripts/m3b_projection.py
Writes data/processed/m3b/epta/results/projection.json.
"""

from __future__ import annotations

import json

from ptagwb.config import REPO_ROOT

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
ACC = REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json"
EFF = {"optimistic": 0.5, "central": 0.25, "pessimistic": 0.1}
N_BLOCKS = 33
PILOT_H = 2.0
ESS_FLOOR = 1000.0
FIXED_GAMMA_ESS_ILLUSTRATIVE = 1000.0


def main():
    bench = json.loads((RES / "bench.json").read_text())
    acc = json.loads(ACC.read_text())
    rw = json.loads((RES / "reference_weights.json").read_text())["hd_over_curn"]
    ref = acc["reference_summary_unconditional"]
    t = bench["timings"]
    C = bench["nuts"]["chains"]
    s_nuts = bench["nuts"]["seconds_per_transition_all_chains"] / C
    s_blocks = N_BLOCKS * t["curn_value_B4"] / 4
    s_tr = s_nuts + s_blocks
    s_hd = t["hd_value_B8"] / 8
    reqA = []
    for row in acc["quantities"]:
        if not row["headline"]:  # revised headline rule (1.645 MCSE_ref <= m/2)
            continue
        q, p = row["quantile"], row["param"]
        kind = "bulk" if q == 0.5 else "tail"
        if row["model"] == "crn_pl":
            r = ref["crn_pl"]["params"][p]
            ess = (r["ess_bulk"] if kind == "bulk" else r["ess_tail"]) * (row["mcse_ref_D"] / row["max_our_mcse"]) ** 2
            how = "direct CURN"
        else:
            pc = p.replace("gw_hd", "gw_crn")
            rc = ref["crn_pl"]["params"][pc]
            mcse_rw = rw["per_param"][pc]["quantiles"][str(q)]["mcse"]
            ess = (rc["ess_bulk"] if kind == "bulk" else rc["ess_tail"]) * (mcse_rw / row["max_our_mcse"]) ** 2
            how = f"HD by reweighting (MCSE_rw {mcse_rw:.4f} on the released CURN draws)"
        reqA.append({"id": row["id"], "quantile": q, "ess_required": float(ess), "how": how})
    ess_bf = ref["crn_pl"]["params"]["gw_crn_log10_A"]["ess_bulk"] * (rw["lnbf_mcse_obm_thinned"] / 0.10) ** 2
    reqA.append({"id": "E-6", "quantile": None, "ess_required": float(ess_bf),
                 "how": f"MCSE(ln BF) <= 0.10 (reference reweighting MCSE {rw['lnbf_mcse_obm_thinned']:.3f})"})
    essA = max([ESS_FLOOR] + [r["ess_required"] for r in reqA])
    runs = {"A_free_gamma": essA, "B_fixed_gamma": FIXED_GAMMA_ESS_ILLUSTRATIVE}
    table = {}
    for label, eff in EFF.items():
        row = {}
        tot = PILOT_H
        for run, ess in runs.items():
            n_tr = ess / eff
            sampling = n_tr * s_tr * 1.25 / 3600
            reweight = n_tr * s_hd / 3600
            row[run] = {"ess": ess, "transitions": n_tr, "sampling_gpu_h": sampling, "reweighting_gpu_h": reweight}
            tot += sampling + reweight
        row["total_gpu_h_incl_pilot"] = tot
        row["production_gpu_h"] = tot - PILOT_H
        row["fits_D4_production_cap_8h"] = bool(tot - PILOT_H <= 8.0)
        table[label] = row
    out = {"status": "provisional scenarios (efficiency bracketed until the pilot)",
           "seconds_per_chain_transition": {"nuts_and_t0_mh_measured": s_nuts, "amplitude_blocks_projected": s_blocks,
                                            "total": s_tr}, "hd_value_seconds_per_draw": s_hd,
           "requirements_run_A": reqA, "ess_required_run_A": essA,
           "ess_run_B_illustrative": FIXED_GAMMA_ESS_ILLUSTRATIVE,
           "reweighting_efficiency": {"khat": rw["khat"], "kish_ess": rw["kish_ess"], "n": rw["n"],
                                      "lnbf_raw": rw["lnbf_raw"]}, "scenarios": table}
    (RES / "projection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "scenarios"}, indent=1))
    for k, v in table.items():
        print(k, json.dumps(v, default=lambda o: round(o, 2)))


if __name__ == "__main__":
    main()
