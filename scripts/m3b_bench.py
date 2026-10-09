"""D3 benchmark (one, <= 1 GPU-h) of the EPTA DR2new model and projection of the production cost
(docs/M3B_PLAN.md Sec. 5.5, decisions D3/D4). Runs only if the exact-model checks passed:
T1, the manifest prior-volume check, the fingerprint and G5-PTA (results/*.json; fail closed).

Measured (GPU, float64; XLA_PYTHON_CLIENT_PREALLOCATE=false):
* CURN value+gradient and value-only at B in {1, 4, 8} (vmapped chains), HD value-only (the
  reweighting cost) at B in {1, 8}; median of 30 calls after compilation; peak device memory;
* the full update of the production kernel (``ptagwb.eventmh.EventMHNUTS``: one NUTS transition
  on the 66 continuous coordinates + 2 exact MH updates of t0 + cache refresh), 4 vectorised
  chains, after a 100-step step-size adaptation, 50 timed transitions; steps per transition,
  acceptance, divergences, t0 acceptance.

Benchmark expedients (NOT production settings, which the plan fixes from our own pilot): the dense
metric is the covariance of the released CURN chain's retained draws in the unconstrained
coordinates and the chains start at released draws. They only make the measured steps per
transition representative of a tuned run.

Projection: required ESS per run = max(1000, max over decidable headline quantities of
ESS_ref (MCSE_ref / max_our_MCSE)^2); transitions = ESS / (ESS per transition), bracketed by
0.1-0.5 (central 0.25) for the slowest parameter; plus HD reweighting (value-only per retained
draw) and the plan's <= 2 GPU-h pilot. Runs: CURN^gamma and CURN gamma = 13/3.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src python scripts/m3b_bench.py
Writes data/processed/m3b/epta/results/bench.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ptagwb import epta
from ptagwb.config import REPO_ROOT
from ptagwb.eventmh import EventMHNUTS, T0Proposal

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
ACC = REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json"


def preconditions() -> dict:
    need = {"t1.json": "T1_pass", "prior_volume.json": "pass", "fingerprint.json": "pass", "g5_pta.json": "G5_PTA_pass"}
    st = {}
    for f, k in need.items():
        p = RES / f
        st[f] = bool(p.exists() and json.loads(p.read_text()).get(k))
    if not all(st.values()):
        sys.exit(f"D3: exact-model checks not all passed: {st}")
    return st


def timeit(f, *args, n=30):
    out = f(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def mem_peak():
    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", -1))
    except Exception:  # noqa: BLE001
        return -1


def main():
    pre = preconditions()
    t_start = time.time()
    dev = jax.devices()[0]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    names, X, burn = epta.load_reference("crn_pl", man)
    R = X[burn:, :67]
    out = {"preconditions": pre, "device": str(dev), "nvidia_smi_before": smi, "timings": {}}
    Mc = epta.EPTAModel(psrs, man, "crn")
    Mh = epta.EPTAModel(psrs, man, "hd")
    rng = np.random.default_rng(1)
    for B in (1, 4, 8):
        xb = jnp.asarray(R[rng.choice(len(R), B, replace=False)])
        vg = jax.jit(jax.vmap(jax.value_and_grad(Mc._logL)))
        v = jax.jit(jax.vmap(Mc._logL))
        out["timings"][f"curn_value_grad_B{B}"] = timeit(vg, xb)
        out["timings"][f"curn_value_B{B}"] = timeit(v, xb)
        if B in (1, 8):
            out["timings"][f"hd_value_B{B}"] = timeit(jax.jit(jax.vmap(Mh._logL)), xb)
        out["timings"][f"peak_bytes_after_B{B}"] = mem_peak()
        print(B, {k: v for k, v in out["timings"].items() if k.endswith(f"B{B}")}, flush=True)

    # ---- full update: EventMHNUTS, 4 vectorised chains
    i0 = Mc.t0_index
    lo, hi = Mc.lo, Mc.hi
    prop = T0Proposal(lo[i0], hi[i0], w_uniform=0.2, rw_scale=2.0)
    ker = EventMHNUTS(Mc._logL, lo, hi, i0, prop, n_mh=2, max_tree_depth=10)
    cont = ker.cont
    Z = np.stack([ker.to_z(x) for x in R[:: max(1, len(R) // 4000)]])
    imm = np.cov(Z.T)
    C = 4
    x0 = R[rng.choice(len(R), C, replace=False)]
    keys = jax.random.split(jax.random.PRNGKey(0), C)
    n_warm, n_time = 100, 50
    init = jax.jit(jax.vmap(lambda k, x: ker.init(k, x, n_warm, step_size=0.05, inverse_mass_matrix=jnp.asarray(imm),
                                                   adapt_mass_matrix=False)))
    st, t0 = init(keys, jnp.asarray(x0))
    step = jax.jit(jax.vmap(ker.step))
    t = time.perf_counter()
    k = jax.random.PRNGKey(1)
    for i in range(n_warm):
        k, kk = jax.random.split(k)
        st, t0, info = step(jax.random.split(kk, C), st, t0)
    jax.block_until_ready(st.z)
    t_warm = time.perf_counter() - t
    steps, acc_p, divs, mh = [], [], [], []
    t = time.perf_counter()
    for i in range(n_time):
        k, kk = jax.random.split(k)
        st, t0, info = step(jax.random.split(kk, C), st, t0)
        steps.append(np.asarray(info[3]))
        acc_p.append(np.asarray(info[2]))
        divs.append(np.asarray(info[1]))
        mh.append(np.asarray(info[0]))
    jax.block_until_ready(st.z)
    t_time = time.perf_counter() - t
    steps = np.array(steps)
    out["nuts"] = {"chains": C, "warmup_steps": n_warm, "timed_transitions": n_time, "warmup_seconds": t_warm,
                   "timed_seconds": t_time, "seconds_per_transition_all_chains": t_time / n_time,
                   "steps_per_transition_mean": float(steps.mean()), "steps_per_transition_max": int(steps.max()),
                   "accept_prob_mean": float(np.mean(acc_p)), "divergences": int(np.sum(divs)),
                   "mh_t0_accept_rate": float(np.sum(mh) / (2 * C * n_time)),
                   "step_size": np.asarray(st.adapt_state.step_size).tolist(),
                   "seconds_per_gradient_batched": t_time / n_time / max(1.0, float(steps.mean())),
                   "n_continuous": int(len(cont))}
    out["timings"]["peak_bytes_end"] = mem_peak()
    print(out["nuts"], flush=True)

    # ---- projection
    accf = json.loads(ACC.read_text())
    ref = accf["reference_summary"]
    need = 1000.0
    per_q = []
    for row in accf["quantities"]:
        if not row["decidable"]:
            continue
        p = ref[row["model"]]["params"][row["param"]]
        ess_ref = p["ess_bulk"] if row["quantile"] == 0.5 else p["ess_tail"]
        e = ess_ref * (row["mcse_ref"] / row["max_our_mcse"]) ** 2
        per_q.append({"id": row["id"], "q": row["quantile"], "ess_needed": e})
        need = max(need, e)
    sec_tr = out["nuts"]["seconds_per_transition_all_chains"] / C  # per chain-transition, vectorised
    sec_tr_b1 = out["nuts"]["steps_per_transition_mean"] * out["timings"]["curn_value_grad_B1"]
    proj = {"ess_needed_per_run": need, "per_quantity": per_q, "runs": ["CURN^gamma", "CURN gamma=13/3"],
            "seconds_per_chain_transition_vectorised": sec_tr, "seconds_per_transition_B1_estimate": sec_tr_b1}
    cases = {}
    for label, eff in (("optimistic", 0.5), ("central", 0.25), ("pessimistic", 0.1)):
        n_tr = need / eff
        run_h = n_tr * sec_tr / 3600 * 1.25  # +25 % warmup
        hd_h = (n_tr * out["timings"]["hd_value_B8"] / 8) / 3600
        cases[label] = {"ess_per_transition": eff, "transitions_per_run": n_tr, "gpu_h_per_run": run_h,
                        "hd_reweighting_gpu_h_per_run": hd_h,
                        "total_gpu_h_two_runs_plus_pilot": 2 * (run_h + hd_h) + 2.0}
    proj["cases"] = cases
    proj["caveats"] = ["ESS per transition is not measured here (50 transitions); it is bracketed",
                       "shelf transport (Sec. 5.3) can require longer runs than the ESS targets; with zero-visit "
                       "shelves the unconditional verdict is INCONCLUSIVE regardless of run length (D9)",
                       "benchmark metric and inits from the released chain (expedient; production uses our pilot)"]
    out["projection"] = proj
    out["wall_seconds"] = time.time() - t_start
    out["gpu_hours_used"] = out["wall_seconds"] / 3600
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "bench.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(proj["cases"], indent=1))
    print(f"wall {out['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
