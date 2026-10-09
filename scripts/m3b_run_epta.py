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
# optional keys (review of pilot v1): type and the value used when absent (= the v1 behaviour)
OPTIONAL = {"inverse_mass_matrix": ((str, type(None)), None), "target_accept_prob": (float, 0.8),
            "stop_rules": ((dict, type(None)), None)}
# pre-registered in-run stop rules (pilot v2): all keys required when stop_rules is given
STOP_RULE_KEYS = {"stop_on_post_warmup_divergence": bool, "check_at_transition": int,
                  "max_elapsed_s_at_check": float, "min_post_warmup_per_chain": int,
                  "projection_deadline_s": float, "projection_window_chunks": int}
FILE_DIRS = {"inverse_mass_matrix": ("configs/m3b/metrics/", ".npz"), "proposals": ("configs/m3b/proposals/", ".json")}


class ConfigError(RuntimeError):
    pass


def effective(cfg: dict) -> dict:
    """The optional settings actually used (absent -> documented default)."""
    return {k: cfg.get(k, d) for k, (_, d) in OPTIONAL.items()}


def file_ref(value, kind: str) -> Path | None:
    """``file:<repo-relative path>`` -> absolute path, restricted to its committed directory."""
    if value is None or value == "prior":
        return None
    d, suffix = FILE_DIRS[kind]
    if not isinstance(value, str) or not value.startswith("file:"):
        raise ConfigError(f"{kind}: expected 'file:{d}...{suffix}', got {value!r}")
    rel = value[len("file:"):]
    if not rel.startswith(d) or not rel.endswith(suffix) or ".." in Path(rel).parts:
        raise ConfigError(f"{kind}: {rel!r} must be a {suffix} file under {d}")
    return REPO_ROOT / rel


def validate_config(cfg: dict) -> dict:
    for k, t in REQUIRED.items():
        if k not in cfg:
            raise ConfigError(f"config lacks {k!r}")
        if not isinstance(cfg[k], t) or (t is int and isinstance(cfg[k], bool)):
            raise ConfigError(f"config {k!r} has type {type(cfg[k]).__name__}")
    extra = sorted(set(cfg) - set(REQUIRED) - set(OPTIONAL) - {"comment"})
    if extra:
        raise ConfigError(f"unknown config keys {extra} (no silent settings)")
    for k, (t, _) in OPTIONAL.items():
        if k in cfg and (not isinstance(cfg[k], t) or isinstance(cfg[k], bool)):
            raise ConfigError(f"config {k!r} has type {type(cfg[k]).__name__}")
    eff = effective(cfg)
    file_ref(eff["inverse_mass_matrix"], "inverse_mass_matrix")
    if not 0.5 <= eff["target_accept_prob"] < 1.0:
        raise ConfigError("target_accept_prob must be in [0.5, 1)")
    if eff["inverse_mass_matrix"] is not None and (cfg["adapt_mass_matrix"] or not cfg["dense_mass"]):
        raise ConfigError("a supplied inverse_mass_matrix is a fixed dense metric: dense_mass true, "
                          "adapt_mass_matrix false")
    rules = eff["stop_rules"]
    if rules is not None:
        if set(rules) != set(STOP_RULE_KEYS):
            raise ConfigError(f"stop_rules needs exactly {sorted(STOP_RULE_KEYS)}")
        for k, t in STOP_RULE_KEYS.items():
            if not isinstance(rules[k], t) or (t is int and isinstance(rules[k], bool)):
                raise ConfigError(f"stop_rules.{k} has type {type(rules[k]).__name__}")
        if rules["check_at_transition"] % cfg["chunk_transitions"] or rules["projection_window_chunks"] < 1:
            raise ConfigError("stop_rules.check_at_transition must be a chunk boundary; window >= 1 chunk")
    if cfg["phase"] not in ("pilot", "production", "contingency") or cfg["model"] not in ("crn",):
        raise ConfigError("phase must be pilot/production/contingency and model crn (HD by reweighting)")
    if cfg["init"] not in ("prior_central",) and not cfg["init"].startswith("file:"):
        raise ConfigError("init must be prior_central or file:<committed npz>")
    if set(cfg["blocks"]) != {"proposals"}:
        raise ConfigError("blocks must hold exactly 'proposals'")
    file_ref(cfg["blocks"]["proposals"], "proposals")
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
    cfg = json.loads(cfg_path.read_text())
    files = {}
    for kind, v in (("inverse_mass_matrix", cfg.get("inverse_mass_matrix")),
                    ("proposals", cfg.get("blocks", {}).get("proposals"))):
        f = file_ref(v, kind)
        if f is None:
            continue
        r = str(f.relative_to(REPO_ROOT))
        try:
            git("ls-files", "--error-unmatch", r)
        except subprocess.CalledProcessError as e:
            raise ConfigError(f"{r} is not a committed file") from e
        files[r] = hashlib.sha256(f.read_bytes()).hexdigest()
    from ptagwb.binding import control_binding

    # the control plane (budget, binding, this driver) is outside the gate-evidence binding; the run
    # binds it separately: committed and clean (checked above), and its file hashes are recorded
    return {"head": git("rev-parse", "HEAD").strip(), "config": rel,
            "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "files": files,
            "control": control_binding()}


def gpu_free() -> str:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if out:
        raise ConfigError(f"GPU busy: {out}")
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()


def block_layout(man, param_names) -> list[tuple[str, ...]]:
    """The block structure (plan Sec. 5.1): every (log10_A[, gamma]) amplitude pair in model order,
    then the joint (t0, log10_tau, log10_Amp) dip block."""
    ix = set(param_names)
    out = []
    for nm in param_names:
        if nm.endswith("_log10_A"):
            g = nm[: -len("_log10_A")] + "_gamma"
            out.append((nm, g) if g in ix else (nm,))
    pre = man["dip"]["param_prefix"]
    out.append(tuple(f"{pre}_{k}" for k in ("t0", "log10_tau", "log10_Amp")))
    return out


def build_blocks(man, model, cfg):
    from ptagwb.eventmh import BlockProposal

    names = list(model.param_names)
    ix = {n: i for i, n in enumerate(names)}
    layout = block_layout(man, names)
    src = cfg["blocks"]["proposals"]
    if src == "prior":
        return tuple(BlockProposal(idx=tuple(ix[n] for n in b), lo=tuple(model.lo[[ix[n] for n in b]]),
                                   hi=tuple(model.hi[[ix[n] for n in b]])) for b in layout)
    d = json.loads(file_ref(src, "proposals").read_text())
    blocks = d.get("blocks")
    if not isinstance(blocks, list) or [tuple(b.get("params", ())) for b in blocks] != layout:
        raise ConfigError("proposal file: blocks must match the block layout (names and order) exactly")
    out = []
    for b in blocks:
        idx = tuple(ix[n] for n in b["params"])
        lo, hi = [float(v) for v in model.lo[list(idx)]], [float(v) for v in model.hi[list(idx)]]
        if [float(v) for v in b["lo"]] != lo or [float(v) for v in b["hi"]] != hi:
            raise ConfigError(f"proposal file: {b['params']}: box differs from the prior box")
        w = float(b["w_prior"])
        if not 0.0 < w <= 1.0:
            raise ConfigError(f"proposal file: {b['params']}: w_prior {w} not in (0, 1]")
        edges = b["edges"]
        if len(edges) != len(idx):
            raise ConfigError(f"proposal file: {b['params']}: one edge array per coordinate")
        for e, l_, h_ in zip(edges, lo, hi):
            e = np.asarray(e, np.float64)
            if e.size < 2 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0) or e[0] != l_ or e[-1] != h_:
                raise ConfigError(f"proposal file: {b['params']}: edges must increase strictly from lo to hi")
        out.append(BlockProposal(idx=idx, lo=tuple(lo), hi=tuple(hi), w_prior=w,
                                 edges=tuple(tuple(float(v) for v in e) for e in edges)))
    return tuple(out)


def load_metric(path: Path, cont_names: list[str]) -> np.ndarray:
    """Fixed dense inverse mass matrix (covariance of z, NOT its inverse) for the continuous
    coordinates in sampler order; names must match exactly."""
    with np.load(path, allow_pickle=False) as z:
        imm = np.asarray(z["inverse_mass_matrix"], np.float64)
        names = [str(n) for n in z["names"]]
    if names != list(cont_names):
        raise ConfigError("metric: parameter names/order differ from the sampler's continuous coordinates")
    n = len(cont_names)
    if imm.shape != (n, n) or not np.all(np.isfinite(imm)):
        raise ConfigError(f"metric: shape {imm.shape} != ({n}, {n}) or non-finite")
    if np.max(np.abs(imm - imm.T)) > 1e-12 * np.max(np.abs(imm)):
        raise ConfigError("metric: not symmetric")
    try:
        np.linalg.cholesky(imm)
    except np.linalg.LinAlgError as e:
        raise ConfigError("metric: not positive definite") from e
    return imm


def stop_check(rules: dict | None, num_warmup: int, L: int, chunks: list[dict], done: int,
               post_warmup_divergences: int) -> str | None:
    """Pre-registered in-run stop rules (pilot v2), evaluated after every chunk."""
    if rules is None:
        return None
    if rules["stop_on_post_warmup_divergence"] and post_warmup_divergences > 0:
        return f"stopped: {post_warmup_divergences} post-warmup divergence(s) (pre-registered rule)"
    if done == rules["check_at_transition"]:
        el = chunks[-1]["elapsed_s"]
        if el > rules["max_elapsed_s_at_check"]:
            return (f"stopped: elapsed {el:.0f} s at transition {done} > {rules['max_elapsed_s_at_check']:.0f} s "
                    "(pre-registered rule)")
        slowest = max(c["seconds"] for c in chunks[-rules["projection_window_chunks"]:]) / L
        proj = max(done - num_warmup, 0) + int((rules["projection_deadline_s"] - el) // slowest)
        if proj < rules["min_post_warmup_per_chain"]:
            return (f"stopped: projected {proj} post-warmup transitions per chain < "
                    f"{rules['min_post_warmup_per_chain']} by {rules['projection_deadline_s']:.0f} s "
                    "(pre-registered rule)")
    return None


def save_checkpoint(path: Path, tree, extra: dict) -> None:
    """Sampler checkpoint (all leaves of ``tree``), atomic; diagnostics only (runs are never
    resumed without a reviewed decision)."""
    import tempfile

    import jax

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **{f"leaf_{i:03d}": np.asarray(v) for i, v in enumerate(leaves)},
                 treedef=np.array(str(treedef)), extra=np.array(json.dumps(extra)))
    os.replace(tmp, path)


def load_checkpoint(path: Path, like):
    import jax

    _, treedef = jax.tree_util.tree_flatten(like)
    with np.load(path, allow_pickle=False) as z:
        if str(z["treedef"]) != str(treedef):
            raise ConfigError("checkpoint structure differs")
        leaves = [z[f"leaf_{i:03d}"] for i in range(treedef.num_leaves)]
        extra = json.loads(str(z["extra"]))
    return jax.tree_util.tree_unflatten(treedef, leaves), extra


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
    eff = effective(cfg)
    ker = EventMHNUTS(M._logL, M.lo, M.hi, i0, prop, n_mh=cfg["n_t0_mh"], max_tree_depth=cfg["max_tree_depth"],
                      dense_mass=cfg["dense_mass"], blocks=blocks, target_accept_prob=eff["target_accept_prob"])
    cont_names = [M.param_names[i] for i in ker.cont]
    mf = file_ref(eff["inverse_mass_matrix"], "inverse_mass_matrix")
    imm = None if mf is None else jnp.asarray(load_metric(mf, cont_names))
    C = cfg["chains"]
    rng = np.random.default_rng(cfg["seed"])
    if cfg["init"] == "prior_central":
        x0 = M.lo + (M.hi - M.lo) * (0.25 + 0.5 * rng.random((C, len(M.lo))))
    else:
        raise ConfigError("file inits are for production (after the pilot)")
    keys = jax.random.split(jax.random.PRNGKey(cfg["seed"]), C)
    init = jax.jit(jax.vmap(lambda k, x: ker.init(k, x, cfg["num_warmup"], step_size=cfg["step_size_init"],
                                                  inverse_mass_matrix=imm,
                                                  adapt_mass_matrix=cfg["adapt_mass_matrix"])))
    st, t0 = init(keys, jnp.asarray(x0))
    W = cfg["num_warmup"]
    metric0 = np.asarray(st.adapt_state.inverse_mass_matrix)
    np.save(out / "metric_initial.npy", metric0)  # the metric actually used from transition 0
    last_metric = metric0
    L = cfg["chunk_transitions"]

    def body(carry, kk):
        st, t0 = carry
        st, t0, info = jax.vmap(ker.step)(jax.random.split(kk, C), st, t0)
        x = jax.vmap(ker.physical)(st, t0)
        return (st, t0), (x, jax.vmap(M._logL)(x), *info)

    chunk = jax.jit(lambda c, k: jax.lax.scan(body, c, jax.random.split(k, L)))
    key = jax.random.PRNGKey(cfg["seed"] + 1)
    done, pred, n_chunk = 0, 0.0, 0
    meta = {"config": cfg, "effective": eff, "provenance": prov, "binding": binding, "param_names": M.param_names,
            "continuous_names": cont_names, "block_idx": [list(b.idx) for b in blocks],
            "metric_source": "identity (adapted per the NumPyro schedule)" if mf is None else
            {"file": str(mf.relative_to(REPO_ROOT)), "sha256": prov.get("files", {}).get(str(mf.relative_to(REPO_ROOT)))},
            "metrics_saved_after_chunk": [], "chunks": []}
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
        step_size = np.asarray(st.adapt_state.step_size)
        np.savez(out / f"chunk_{n_chunk:05d}.npz", x=xs, logL=ll, t0_accept=acc_t0, diverging=div, accept_prob=ap,
                 num_steps=ns, block_accept=bacc, first_transition=done, step_size=step_size)
        metric = np.asarray(st.adapt_state.inverse_mass_matrix)
        if not np.array_equal(metric, last_metric):
            np.save(out / f"metric_after_chunk_{n_chunk:05d}.npy", metric)
            meta["metrics_saved_after_chunk"].append(n_chunk)
            last_metric = metric
        post = np.arange(done, done + L) >= W
        div_post = int(div[post].sum())
        done += L
        meta["chunks"].append({"n": n_chunk, "seconds": dt, "transitions_done": done, "elapsed_s": dl.elapsed,
                               "divergences": int(div.sum()), "divergences_post_warmup": div_post,
                               "step_size": step_size.tolist(),
                               "num_steps_max_per_transition": ns.max(axis=1).tolist()})
        try:  # best-effort diagnostics: never resumed without a reviewed decision
            save_checkpoint(out / "checkpoint.npz", (st, t0, key), {"transitions_done": done, "n_chunk": n_chunk})
        except Exception as e:  # noqa: BLE001
            meta["chunks"][-1]["checkpoint_error"] = repr(e)
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        try:  # best-effort; accounting is the supervisor's
            ledger.heartbeat(cfg["run_id"], dl.elapsed)
        except Exception:  # noqa: BLE001
            pass
        n_chunk += 1
        if not (np.all(np.isfinite(ll)) and np.all(np.isfinite(xs))):
            reason = "stopped: non-finite state or log-likelihood"
            break
        why = stop_check(eff["stop_rules"], W, L, meta["chunks"], done, div_post)
        if why is not None:
            reason = why
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
