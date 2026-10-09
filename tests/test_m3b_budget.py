"""D4 (2026-10-09, second set): the mechanical GPU-time cap (``ptagwb.budget``) and the run
driver's config validation (``scripts/m3b_run_epta.py``). No GPU, no sampling."""

from __future__ import annotations

import json
import sys
import time

import pytest

from ptagwb.budget import CAPS_H, BudgetExceeded, Deadline, GpuLock, Ledger, Watchdog
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
    L = Ledger(tmp_path / "ledger.json")
    L.open("crash", "pilot", 1.98, {})
    L.heartbeat("crash", 7080.0)  # last heartbeat, then the process dies at 7,139 s
    assert json.loads((tmp_path / "ledger.json").read_text())[0]["status"] == "running"
    assert L.remaining_h("pilot") == pytest.approx(0.02)  # the full 1.98 h stay charged
    with pytest.raises(BudgetExceeded):  # the reviewer's case: 120 s must not be available
        L.open("next", "pilot", 120 / 3600, {})
    assert Ledger(tmp_path / "ledger.json").used_h() == pytest.approx(1.98)


def test_watchdog_kills_on_deadline_and_on_heartbeat_failure(tmp_path):
    d = Deadline(time.monotonic(), limit_s=0.2, grace_s=0.1)
    assert d.allows(0.1) and not d.allows(1.0)
    killed = {}
    L = Ledger(tmp_path / "ledger.json")
    L.open("r", "pilot", 1.0, {})

    def on_kill(el, reason):
        L.close("r", el, "killed: " + reason)
        killed["el"], killed["reason"] = el, reason

    wd = Watchdog(d, on_kill, poll_s=0.02, exit_fn=lambda code: killed.setdefault("code", code),
                  on_beat=lambda el: L.heartbeat("r", el), beat_s=0.05).start()
    time.sleep(0.6)  # a "chunk" that overruns the deadline
    wd.stop()
    assert killed["code"] == 3 and killed["el"] > 0.3 and killed["reason"] == "deadline overrun"
    e = L.entries()[0]
    assert e["status"].startswith("killed") and e["seconds"] > 0.3

    # a failing heartbeat (ledger unwritable) must kill the run even though on_kill also fails
    d2 = Deadline(time.monotonic(), limit_s=100.0, grace_s=1.0)
    k2 = {}

    def bad_beat(el):
        raise OSError("disk full")

    def bad_kill(el, reason):
        k2["reason"] = reason
        raise OSError("disk full")

    wd2 = Watchdog(d2, bad_kill, poll_s=0.01, exit_fn=lambda code: k2.setdefault("code", code), on_beat=bad_beat,
                   beat_s=0.0).start()
    time.sleep(0.3)
    wd2.stop()
    assert k2["code"] == 3 and k2["reason"].startswith("heartbeat failed")


def test_reconcile_only_dead_processes(tmp_path):
    L = Ledger(tmp_path / "ledger.json")
    L.open("live", "pilot", 1.0, {})  # pid = this (live) process
    with pytest.raises(BudgetExceeded):
        L.reconcile("live", operator_note="should refuse")
    e = json.loads((tmp_path / "ledger.json").read_text())
    e[0]["pid"] = 2**22 + 12345  # a pid that does not exist
    (tmp_path / "ledger.json").write_text(json.dumps(e))
    L.reconcile("live", operator_note="process gone, verified")
    assert L.entries()[0]["status"] == "crashed_reconciled" and L.used_h("pilot") == 0.0


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
