"""EPTA DR2new run driver (pilot and production; docs/M3B_PLAN.md Sec. 5, decision D4 of
2026-10-09, second set). **Not run in M3b-0E**: the pilot starts only after the independent review
and the coordinator's confirmation.

Fail-closed rules, all checked before the GPU is touched:
* the run config is a committed file under ``configs/m3b/run_configs/`` and the work tree has no changes
  in ``src/``, ``scripts/``, ``tests/`` or ``configs/`` (no deviation from committed configs;
  the HEAD sha, the config sha256 and the evidence binding are recorded with the run);
* every exact-model gate result is current, passing and bound to this configuration
  (``ptagwb.binding.require_bound``: T1, manifest/prior volume, fingerprint, G5-PTA, t0
  conditional);
* the GPU-time ledger (``ptagwb.budget``, locked) admits the run's ``max_gpu_hours`` within its
  phase and the 12 GPU-h total, reserving it in full while the run is open; the exclusive GPU lock
  is held for the whole run and ``nvidia-smi`` shows no other compute process.

During the run (review round 3: deadline enforcement independent of the ledger): this process is
the **supervisor**; it admits the run, then starts the sampling **worker** (``--worker``) and
SIGKILLs it at allocation - 2 grace, using only ``CLOCK_BOOTTIME``, ``waitpid`` and ``kill``. The
worker registers PR_SET_PDEATHSIG and verifies that its parent is still the supervisor before it is
executed (review round 4). Inside the worker a lock-free ``HardDeadline`` timer calls ``os._exit`` at
allocation - 3 grace, and the chunk planner stops between chunks before allocation - 4 grace (the
previous chunk's time must fit). Heartbeats are best-effort ledger writes. After the reap the
supervisor writes its lock-free record with the duration it measured on ``CLOCK_BOOTTIME`` (never
wall-clock timestamps) and closes the ledger entry with that duration rounded up (best-effort: if
the write fails, the entry stays open and charged in full).

The control plane (``ptagwb.budget``, ``ptagwb.binding``, this driver, the evidence rebind script)
is excluded from the gate-evidence binding (it cannot change a gate's numbers) and is bound
separately here: it must be committed and clean, and its file hashes (``control_binding``) are
recorded with the run. Stop rules (fixed in the config): the deadline, the
number of transitions, a non-finite log-likelihood. The driver never changes a setting; a pilot
that suggests a change ends the work and is reported.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src python scripts/m3b_run_epta.py CONFIG [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ptagwb.config import REPO_ROOT

RUNS = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "runs"
RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
LEDGER = RUNS / "ledger.json"
GPU_LOCK = Path.home() / ".cache" / "ptagwb" / "gpu0.lock"
PRECONDITIONS = {"t1.json": "T1_pass", "prior_volume.json": "pass", "fingerprint.json": "pass",
                 "g5_pta.json": "G5_PTA_pass", "t0_conditional.json": "pass",
                 "t2.json": "completed", "conditional_occupancy.json": "completed"}
REQUIRED = {"run_id": str, "phase": str, "model": str, "gamma_common": (float, type(None)), "chains": int,
            "num_warmup": int, "max_transitions": int, "chunk_transitions": int, "max_tree_depth": int,
            "step_size_init": float, "dense_mass": bool, "adapt_mass_matrix": bool, "seed": int,
            "init": str, "t0_proposal": dict, "n_t0_mh": int, "blocks": dict, "max_gpu_hours": float,
            "watchdog_grace_s": float}


class ConfigError(RuntimeError):
    pass


def validate_config(cfg: dict) -> dict:
    for k, t in REQUIRED.items():
        if k not in cfg:
            raise ConfigError(f"config lacks {k!r}")
        if not isinstance(cfg[k], t) or (t is int and isinstance(cfg[k], bool)):
            raise ConfigError(f"config {k!r} has type {type(cfg[k]).__name__}")
    extra = sorted(set(cfg) - set(REQUIRED) - {"comment"})
    if extra:
        raise ConfigError(f"unknown config keys {extra} (no silent settings)")
    if cfg["phase"] not in ("pilot", "production", "contingency") or cfg["model"] not in ("crn",):
        raise ConfigError("phase must be pilot/production/contingency and model crn (HD by reweighting)")
    if cfg["init"] not in ("prior_central",) and not cfg["init"].startswith("file:"):
        raise ConfigError("init must be prior_central or file:<committed npz>")
    if cfg["blocks"].get("proposals") not in ("prior",) and not str(cfg["blocks"].get("proposals")).startswith("file:"):
        raise ConfigError("blocks.proposals must be 'prior' or file:<committed json>")
    if not (0 < cfg["chunk_transitions"] <= cfg["max_transitions"]):
        raise ConfigError("bad chunk_transitions")
    return cfg


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True).stdout


def committed_and_clean(cfg_path: Path) -> dict:
    rel = str(cfg_path.resolve().relative_to(REPO_ROOT))
    try:
        git("ls-files", "--error-unmatch", rel)
    except subprocess.CalledProcessError as e:
        raise ConfigError(f"{rel} is not a committed file") from e
    dirty = git("status", "--porcelain", "--", "src", "scripts", "tests", "configs").strip()
    if dirty:
        raise ConfigError(f"uncommitted changes (no deviation from committed configs):\n{dirty}")
    from ptagwb.binding import control_binding

    # the control plane (budget, binding, this driver) is outside the gate-evidence binding; the run
    # binds it separately: committed and clean (checked above), and its file hashes are recorded
    return {"head": git("rev-parse", "HEAD").strip(), "config": rel,
            "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "control": control_binding()}


def gpu_free() -> str:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if out:
        raise ConfigError(f"GPU busy: {out}")
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()


def build_blocks(man, model, cfg):
    from ptagwb.eventmh import BlockProposal

    if cfg["blocks"]["proposals"] != "prior":
        raise ConfigError("frozen histogram proposals are loaded in production only (not implemented before the pilot)")
    ix = {n: i for i, n in enumerate(model.param_names)}
    out = []
    for nm in model.param_names:
        if nm.endswith("_log10_A"):
            g = nm[: -len("_log10_A")] + "_gamma"
            idx = (ix[nm],) + ((ix[g],) if g in ix else ())
            out.append(BlockProposal(idx=idx, lo=tuple(model.lo[list(idx)]), hi=tuple(model.hi[list(idx)])))
    pre = man["dip"]["param_prefix"]
    idx = tuple(ix[f"{pre}_{k}"] for k in ("t0", "log10_tau", "log10_Amp"))
    out.append(BlockProposal(idx=idx, lo=tuple(model.lo[list(idx)]), hi=tuple(model.hi[list(idx)])))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true", help="all checks, no ledger entry, no sampling")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a.config)
    cfg_path = Path(a.config)
    cfg = validate_config(json.loads(cfg_path.read_text()))
    prov = committed_and_clean(cfg_path)
    from ptagwb.binding import evidence_binding, require_bound
    from ptagwb.budget import GpuLock, Ledger, conservative_charge_s, supervise

    binding = evidence_binding()
    require_bound(PRECONDITIONS, RES, binding)
    ledger = Ledger(LEDGER)
    rem = ledger.remaining_h(cfg["phase"])
    if cfg["max_gpu_hours"] > rem:
        raise ConfigError(f"max_gpu_hours {cfg['max_gpu_hours']} > remaining {rem:.3f} GPU-h of {cfg['phase']}")
    with GpuLock(GPU_LOCK):  # exclusive for the whole run (released at exit, also on a crash)
        smi = gpu_free()
        if a.dry_run:
            print(json.dumps({"ok": True, "provenance": prov, "remaining_h": rem, "gpu": smi}, indent=1))
            return
        out = RUNS / cfg["run_id"]
        if out.exists():
            raise ConfigError(f"{out} exists (runs are never resumed or repeated silently)")
        out.mkdir(parents=True)
        # admission reserves the full allocation; a crash stays charged in full until reconciled
        # from the supervisor record
        ledger.open(cfg["run_id"], cfg["phase"], cfg["max_gpu_hours"], {**prov, "binding": binding, "gpu": smi})
        alloc = cfg["max_gpu_hours"] * 3600.0
        grace = cfg["watchdog_grace_s"]
        (out / "worker_env.json").write_text(json.dumps({"provenance": prov, "binding": binding}))
        env = dict(os.environ, M3B_RUN_LIMIT_S=str(alloc - 4 * grace), M3B_HARD_EXIT_S=str(alloc - 3 * grace))

        def on_exit(rec):  # best-effort terminal write; on failure the entry stays open (charged)
            charge = conservative_charge_s(rec["elapsed_s"], cfg["max_gpu_hours"])  # CLOCK_BOOTTIME duration
            status = "killed_at_deadline" if rec["killed_at_deadline"] else (
                "completed" if rec["returncode"] == 0 else f"worker_exit_{rec['returncode']}")
            ledger.close(cfg["run_id"], charge, status, {"supervisor": rec})

        # the supervisor kills the worker at allocation - 2 grace, whatever it is doing (no ledger,
        # no lock on the kill path); the worker's own hard exit fires one grace earlier
        rec = supervise([sys.executable, str(Path(__file__).resolve()), str(cfg_path), "--worker"],
                        run_id=cfg["run_id"], kill_after_s=alloc - 2 * grace, record_path=out / "supervisor.json",
                        env=env, on_exit=on_exit)
        print(json.dumps({k: rec[k] for k in ("returncode", "killed_at_deadline", "elapsed_s")}))


def worker(cfg_path: str) -> None:
    """The sampling process (started only by the supervisor in ``main``)."""
    from ptagwb.budget import Deadline, HardDeadline, Ledger, Watchdog

    cfg = validate_config(json.loads(Path(cfg_path).read_text()))
    out = RUNS / cfg["run_id"]
    meta = json.loads((out / "worker_env.json").read_text())
    t0 = time.monotonic()
    HardDeadline(t0 + float(os.environ["M3B_HARD_EXIT_S"])).start()  # os._exit, lock-free
    dl = Deadline(t0, float(os.environ["M3B_RUN_LIMIT_S"]), cfg["watchdog_grace_s"])
    ledger = Ledger(LEDGER)
    Watchdog(dl, on_beat=lambda el: ledger.heartbeat(cfg["run_id"], el), beat_s=cfg["watchdog_grace_s"]).start()
    reason = run(cfg, out, dl, ledger, meta["provenance"], meta["binding"])
    print(reason)


def run(cfg, out, dl, ledger, prov, binding) -> str:
    import jax
    import jax.numpy as jnp

    from ptagwb import epta
    from ptagwb.eventmh import EventMHNUTS, T0Proposal

    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    M = epta.EPTAModel(psrs, man, cfg["model"], reduce="hh", buckets=epta.BUCKETS, gamma_common=cfg["gamma_common"])
    i0 = M.t0_index
    tp = cfg["t0_proposal"]
    prop = T0Proposal(M.lo[i0], M.hi[i0], w_uniform=float(tp["w_uniform"]), rw_scale=float(tp["rw_scale_days"]))
    blocks = build_blocks(man, M, cfg)
    ker = EventMHNUTS(M._logL, M.lo, M.hi, i0, prop, n_mh=cfg["n_t0_mh"], max_tree_depth=cfg["max_tree_depth"],
                      dense_mass=cfg["dense_mass"], blocks=blocks)
    C = cfg["chains"]
    rng = np.random.default_rng(cfg["seed"])
    if cfg["init"] == "prior_central":
        x0 = M.lo + (M.hi - M.lo) * (0.25 + 0.5 * rng.random((C, len(M.lo))))
    else:
        raise ConfigError("file inits are for production (after the pilot)")
    keys = jax.random.split(jax.random.PRNGKey(cfg["seed"]), C)
    init = jax.jit(jax.vmap(lambda k, x: ker.init(k, x, cfg["num_warmup"], step_size=cfg["step_size_init"],
                                                  adapt_mass_matrix=cfg["adapt_mass_matrix"])))
    st, t0 = init(keys, jnp.asarray(x0))
    L = cfg["chunk_transitions"]

    def body(carry, kk):
        st, t0 = carry
        st, t0, info = jax.vmap(ker.step)(jax.random.split(kk, C), st, t0)
        x = jax.vmap(ker.physical)(st, t0)
        return (st, t0), (x, jax.vmap(M._logL)(x), *info)

    chunk = jax.jit(lambda c, k: jax.lax.scan(body, c, jax.random.split(k, L)))
    key = jax.random.PRNGKey(cfg["seed"] + 1)
    done, pred, n_chunk = 0, 0.0, 0
    meta = {"config": cfg, "provenance": prov, "binding": binding, "param_names": M.param_names,
            "block_idx": [list(b.idx) for b in blocks], "chunks": []}
    while done < cfg["max_transitions"]:
        if not dl.allows(pred):
            reason = "stopped: deadline (GPU-time cap)"
            break
        key, kk = jax.random.split(key)
        t = time.monotonic()
        (st, t0), rec = chunk((st, t0), kk)
        jax.block_until_ready(st.z)
        dt = time.monotonic() - t
        pred = dt
        xs, ll, acc_t0, div, ap, ns, bacc = (np.asarray(r) for r in rec)
        np.savez(out / f"chunk_{n_chunk:05d}.npz", x=xs, logL=ll, t0_accept=acc_t0, diverging=div, accept_prob=ap,
                 num_steps=ns, block_accept=bacc, first_transition=done)
        done += L
        meta["chunks"].append({"n": n_chunk, "seconds": dt, "transitions_done": done, "elapsed_s": dl.elapsed,
                               "divergences": int(div.sum())})
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        try:  # best-effort; accounting is the supervisor's
            ledger.heartbeat(cfg["run_id"], dl.elapsed)
        except Exception:  # noqa: BLE001
            pass
        n_chunk += 1
        if not np.all(np.isfinite(ll)):
            reason = "stopped: non-finite log-likelihood"
            break
    else:
        reason = "completed: max_transitions"
    meta["stop_reason"] = reason
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return reason


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        sys.exit(f"refused: {e}")
