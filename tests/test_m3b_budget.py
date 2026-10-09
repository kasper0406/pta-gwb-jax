"""D4 (2026-10-09, second set): the mechanical GPU-time cap (``ptagwb.budget``) and the run
driver's config validation (``scripts/m3b_run_epta.py``). No GPU, no sampling."""

from __future__ import annotations

import json
import sys
import time

import pytest

from ptagwb.budget import (CAPS_H, BudgetExceeded, Deadline, GpuLock, HardDeadline, Ledger, SupervisorRecord,
                           Watchdog, conservative_charge_s, supervise)
from ptagwb.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_caps_are_the_user_decision():
    assert CAPS_H == {"total": 12.0, "pilot": 2.0, "production": 8.0, "contingency": 2.0}


def test_ledger_admits_and_refuses(tmp_path):
    L = Ledger(tmp_path / "ledger.json")
    L.open("p1", "pilot", 1.5, {})
    assert L.remaining_h("pilot") == pytest.approx(0.5)  # an open run reserves its full allocation
    with pytest.raises(BudgetExceeded):
        L.open("p2", "pilot", 0.6, {})
    L.close("p1", 1.2 * 3600, "completed")
    assert L.used_h("pilot") == pytest.approx(1.2)
    with pytest.raises(BudgetExceeded):  # 0.8 h left in the pilot phase
        L.open("p2", "pilot", 0.9, {})
    L.open("p2", "pilot", 0.8, {})
    with pytest.raises(BudgetExceeded):  # never repeated / resumed
        L.open("p2", "pilot", 0.1, {})
    L.close("p2", 0.8 * 3600, "completed")
    with pytest.raises(BudgetExceeded):  # closed entries cannot be reopened or rewritten
        L.heartbeat("p2", 10.0)
    L.open("prod", "production", 8.0, {})
    L.close("prod", 8.0 * 3600, "completed")
    assert L.remaining_h("contingency") == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        L.open("c", "contingency", 2.5, {})
    with pytest.raises(BudgetExceeded):
        L.open("z", "production", 0.1, {})
    with pytest.raises(BudgetExceeded):
        L.open("neg", "contingency", 0.0, {})


def _try_open(args):
    path, run_id, barrier_path = args
    import time as _t
    from pathlib import Path as _P

    from ptagwb.budget import BudgetExceeded as _B
    from ptagwb.budget import Ledger as _L

    while not _P(barrier_path).exists():  # start together
        _t.sleep(0.001)
    try:
        _L(path).open(run_id, "pilot", 1.5, {})
        return run_id, True
    except _B:
        return run_id, False


def test_concurrent_admission_reserves_allocations(tmp_path):
    """Three simultaneous 1.5 GPU-h pilot admissions against the 2 GPU-h pilot cap: exactly one."""
    import multiprocessing as mp

    path, barrier = str(tmp_path / "ledger.json"), str(tmp_path / "go")
    ctx = mp.get_context("spawn")
    with ctx.Pool(3) as pool:
        res = pool.map_async(_try_open, [(path, f"p{i}", barrier) for i in range(3)])
        time.sleep(2.0)
        (tmp_path / "go").write_text("")
        out = res.get(60)
    assert sum(ok for _, ok in out) == 1
    assert len(Ledger(path).entries()) == 1


def test_crash_between_heartbeats_is_charged_in_full(tmp_path):
    """Review round 3 repro: 2 GPU-h allocation, last heartbeat 7,080 s, death at 7,139 s. Process
    death alone never lowers the charge; a supervisor record does, conservatively."""
    L = Ledger(tmp_path / "ledger.json")
    L.open("crash", "pilot", 2.0, {})
    L.heartbeat("crash", 7080.0)
    e = json.loads((tmp_path / "ledger.json").read_text())
    e[0]["pid"] = 2**22 + 12345  # the (dead) worker
    (tmp_path / "ledger.json").write_text(json.dumps(e))
    assert L.remaining_h("pilot") == pytest.approx(0.0)
    with pytest.raises(BudgetExceeded):  # no supervisor record: refused, full charge kept
        L.reconcile("crash", tmp_path / "absent.json", operator_note="process gone")
    rec = tmp_path / "supervisor.json"
    SupervisorRecord.write(rec, {"run_id": "crash", "start_unix": 1000.0})  # supervisor died: no end
    with pytest.raises(BudgetExceeded):
        L.reconcile("crash", rec, operator_note="no reaped end")
    assert L.used_h("pilot") == pytest.approx(2.0)
    SupervisorRecord.write(rec, {"run_id": "crash", "start_unix": 1000.0, "end_unix": 1000.0 + 7139.0})
    charge = L.reconcile("crash", rec, operator_note="reaped by the supervisor")
    assert charge == 7200.0  # ceil(7139 / 60) + 1 minutes, capped at the allocation
    with pytest.raises(BudgetExceeded):  # the reviewer's 120 s are not admitted
        L.open("next", "pilot", 120 / 3600, {})
    assert conservative_charge_s(0.0, 59.0, 2.0) == 120.0 and conservative_charge_s(0.0, 7100.0, 2.0) == 7200.0


def _hold_lock(path, hold_s, ready):
    import fcntl

    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        ready.set()
        time.sleep(hold_s)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def test_hard_deadline_fires_while_the_ledger_lock_is_held(tmp_path):
    """The ledger lock is held (by another thread) and the heartbeat blocks on it: the lock-free
    hard deadline still fires on time."""
    import threading

    L = Ledger(tmp_path / "ledger.json")
    L.open("r", "pilot", 1.0, {})
    ready = threading.Event()
    threading.Thread(target=_hold_lock, args=(L.lock_path, 2.0, ready), daemon=True).start()
    ready.wait(5)
    t0 = time.monotonic()
    fired = {}
    Watchdog(Deadline(t0, 100.0), on_beat=lambda el: L.heartbeat("r", el), beat_s=0.0, poll_s=0.01,
             exit_fn=lambda c: fired.setdefault("watchdog", c)).start()  # blocks in flock
    HardDeadline(t0 + 0.2, exit_fn=lambda c: fired.setdefault("hard", (c, time.monotonic() - t0))).start()
    time.sleep(0.6)
    assert fired["hard"][0] == 4 and 0.2 <= fired["hard"][1] <= 0.35
    assert "watchdog" not in fired  # still blocked on the lock, which did not matter


def test_failing_heartbeat_ends_the_worker():
    out = {}

    def bad(el):
        raise OSError("disk full")

    Watchdog(Deadline(time.monotonic(), 100.0), on_beat=bad, beat_s=0.0, poll_s=0.01,
             exit_fn=lambda c: out.setdefault("code", c)).start()
    time.sleep(0.2)
    assert out["code"] == 3


def test_supervisor_kills_on_time_with_ledger_lock_held_and_on_exit_blocked(tmp_path):
    """The supervisor SIGKILLs the worker at the deadline while the ledger lock is held and its
    terminal ledger write (on_exit) is blocked; the entry stays charged in full until the write
    completes, then it is charged the conservative supervisor span."""
    import os
    import sys
    import threading

    L = Ledger(tmp_path / "ledger.json")
    L.open("w", "pilot", 1.0, {})
    ready = threading.Event()
    threading.Thread(target=_hold_lock, args=(L.lock_path, 3.0, ready), daemon=True).start()
    ready.wait(5)
    res = {}

    def on_exit(rec):
        L.close("w", conservative_charge_s(rec["start_unix"], rec["end_unix"], 1.0), "killed_at_deadline")

    th = threading.Thread(target=lambda: res.update(supervise(
        [sys.executable, "-c", "import time; time.sleep(60)"], run_id="w", kill_after_s=0.5,
        record_path=tmp_path / "supervisor.json", on_exit=on_exit)), daemon=True)
    t0 = time.monotonic()
    th.start()
    time.sleep(1.2)  # deadline 0.5 s; the lock is still held, on_exit is blocked
    rec = SupervisorRecord.read(tmp_path / "supervisor.json")
    assert rec["killed_at_deadline"] and rec["elapsed_monotonic"] <= 0.5 + 0.3
    with pytest.raises(ProcessLookupError):
        os.kill(rec["worker_pid"], 0)  # the worker is gone (reaped), on time
    assert th.is_alive()  # the supervisor is still blocked in the best-effort terminal write
    th.join(10)
    assert time.monotonic() - t0 >= 3.0 - 0.1
    e = L.entries()[0]
    assert e["status"] == "killed_at_deadline" and e["seconds"] == 120.0  # ceil(span / 60 s) + 1 min


def test_supervisor_terminal_write_failure_keeps_full_charge(tmp_path):
    import sys

    L = Ledger(tmp_path / "ledger.json")
    L.open("x", "pilot", 1.0, {})

    def bad(rec):
        raise OSError("ledger unwritable")

    rec = supervise([sys.executable, "-c", "pass"], run_id="x", kill_after_s=10.0,
                    record_path=tmp_path / "s.json", on_exit=bad)
    assert rec["returncode"] == 0 and "on_exit_error" in rec
    assert L.entries()[0]["status"] == "running" and L.remaining_h("pilot") == pytest.approx(1.0)


def test_gpu_lock_is_exclusive(tmp_path):
    with GpuLock(tmp_path / "gpu.lock"):
        with pytest.raises(BudgetExceeded):
            GpuLock(tmp_path / "gpu.lock").__enter__()
    with GpuLock(tmp_path / "gpu.lock"):
        pass


def test_run_configs_validate_and_reject_unknown_settings():
    import m3b_run_epta as R

    for f in sorted((REPO_ROOT / "configs" / "m3b" / "run_configs").glob("*.json")):
        cfg = R.validate_config(json.loads(f.read_text()))
        assert cfg["phase"] == "pilot" and cfg["max_gpu_hours"] <= CAPS_H["pilot"]
    total = sum(json.loads(f.read_text())["max_gpu_hours"] for f in (REPO_ROOT / "configs" / "m3b" / "run_configs").glob("*.json"))
    assert total <= CAPS_H["pilot"]
    cfg = json.loads((REPO_ROOT / "configs" / "m3b" / "run_configs" / "epta_pilot_curn_freegamma.json").read_text())
    for bad in ({**cfg, "extra_knob": 1}, {k: v for k, v in cfg.items() if k != "seed"}, {**cfg, "phase": "test"},
                {**cfg, "chains": 4.0}, {**cfg, "init": "reference_chain"}):
        with pytest.raises(R.ConfigError):
            R.validate_config(bad)
