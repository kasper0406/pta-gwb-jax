"""D4 (2026-10-09, second set): the mechanical GPU-time cap (``ptagwb.budget``) and the run
driver's config validation (``scripts/m3b_run_epta.py``). No GPU, no sampling."""

from __future__ import annotations

import json
import sys
import time

import pytest

from ptagwb.budget import (
    CAPS_H,
    CLOCK,
    BudgetExceeded,
    Deadline,
    GpuLock,
    HardDeadline,
    Ledger,
    SupervisorRecord,
    Watchdog,
    conservative_charge_s,
    pdeathsig_hook,
    supervise,
)
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
    assert L.reconcile("crash", tmp_path / "absent.json", operator_note="process gone") == 7200.0
    e = L.entries()[0]
    assert e["status"] == "reconciled_full_allocation_no_reliable_bound" and L.used_h("pilot") == pytest.approx(2.0)


def _crash(tmp_path):
    L = Ledger(tmp_path / "ledger.json")
    L.open("crash", "pilot", 2.0, {})
    L.heartbeat("crash", 6000.0)
    return L, tmp_path / "supervisor.json"


def test_reconcile_round3_repro_charges_the_full_allocation(tmp_path):
    """Round 3 repro: 2 h allocation, heartbeat at 7,080 s, death at 7,139 s. Neither the heartbeat
    nor process death is a bound: no 120 s are released."""
    L, rec = _crash(tmp_path)
    SupervisorRecord.write(rec, {"run_id": "crash", "clock": CLOCK, "start_unix_info": 1000.0, "reaped": False})
    assert L.reconcile("crash", rec, operator_note="supervisor died: no reap") == 7200.0
    with pytest.raises(BudgetExceeded):
        L.open("next", "pilot", 120 / 3600, {})


@pytest.mark.parametrize("record", [
    {"run_id": "crash", "start_unix": 1000.0, "end_unix": 7400.0},  # wall-clock only (round-3 format)
    {"run_id": "crash", "clock": CLOCK, "reaped": True},  # no duration
    {"run_id": "crash", "clock": "CLOCK_REALTIME", "reaped": True, "elapsed_s": 10.0},  # wrong clock
    {"run_id": "crash", "clock": CLOCK, "reaped": False, "elapsed_s": 10.0},  # not reaped
    {"run_id": "crash", "clock": CLOCK, "reaped": True, "elapsed_s": float("nan")},
    {"run_id": "crash", "clock": CLOCK, "reaped": True, "elapsed_s": -1.0},
    {"run_id": "other", "clock": CLOCK, "reaped": True, "elapsed_s": 10.0},
])
def test_reconcile_without_a_reliable_monotonic_bound_keeps_the_full_allocation(tmp_path, record):
    L, rec = _crash(tmp_path)
    SupervisorRecord.write(rec, record)
    assert L.reconcile("crash", rec, operator_note="x") == 7200.0


def test_backward_wall_clock_step_releases_nothing(tmp_path):
    """Round 4 repro: 7,000 s consumed, heartbeat at 6,000 s, the wall clock steps back 600 s. A
    wall-clock span would charge 6,480 s and admit 720 s more; the supervisor's CLOCK_BOOTTIME
    duration charges 7,080 s and the 720 s are refused."""
    L, rec = _crash(tmp_path)
    SupervisorRecord.write(rec, {"run_id": "crash", "clock": CLOCK, "reaped": True, "start_unix_info": 1000.0,
                                 "end_unix_info": 1000.0 + 7000.0 - 600.0, "elapsed_s": 7000.0})
    assert L.reconcile("crash", rec, operator_note="reaped") == 7080.0
    with pytest.raises(BudgetExceeded):
        L.open("next", "pilot", 720 / 3600, {})
    assert conservative_charge_s(59.0, 2.0) == 120.0 and conservative_charge_s(7100.0, 2.0) == 7200.0
    assert conservative_charge_s(None, 2.0) == 7200.0 and conservative_charge_s(float("inf"), 2.0) == 7200.0


def test_supervisor_charges_its_monotonic_duration_through_a_wall_clock_step(tmp_path, monkeypatch):
    """A live supervisor whose wall clock steps back 600 s mid-run: the record's wall span is
    negative, the charged duration is the true one."""
    import ptagwb.budget as B

    real = time.time
    calls = {"n": 0}

    def stepped():
        calls["n"] += 1
        return real() - (600.0 if calls["n"] > 1 else 0.0)

    L = Ledger(tmp_path / "ledger.json")
    L.open("s", "pilot", 1.0, {})
    monkeypatch.setattr(B.time, "time", stepped)  # first call in supervise: start; later: -600 s
    t0 = time.monotonic()
    rec = supervise([sys.executable, "-c", "import time; time.sleep(1.0)"], run_id="s", kill_after_s=30.0,
                    record_path=tmp_path / "s.json",
                    on_exit=lambda r: L.close("s", conservative_charge_s(r["elapsed_s"], 1.0), "completed"))
    took = time.monotonic() - t0
    assert rec["end_unix_info"] - rec["start_unix_info"] < 0  # the wall clock went backwards
    assert took - 0.1 <= rec["elapsed_s"] <= took and rec["elapsed_s"] >= 1.0 and rec["clock"] == CLOCK
    assert L.entries()[0]["seconds"] == 120.0


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
        L.close("w", conservative_charge_s(rec["elapsed_s"], 1.0), "killed_at_deadline")

    th = threading.Thread(target=lambda: res.update(supervise(
        [sys.executable, "-c", "import time; time.sleep(60)"], run_id="w", kill_after_s=0.5,
        record_path=tmp_path / "supervisor.json", on_exit=on_exit)), daemon=True)
    t0 = time.monotonic()
    th.start()
    time.sleep(1.2)  # deadline 0.5 s; the lock is still held, on_exit is blocked
    rec = SupervisorRecord.read(tmp_path / "supervisor.json")
    assert rec["killed_at_deadline"] and rec["elapsed_s"] <= 0.5 + 0.3
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


class _FakeLibc:
    def __init__(self, set_rc=0, get_value=None):
        self.set_rc, self.get_value, self.value = set_rc, get_value, 0

    def prctl(self, op, arg, *rest):
        import ctypes

        if op == 1:
            if self.set_rc == 0:
                self.value = arg
            return self.set_rc
        ctypes.cast(arg, ctypes.POINTER(ctypes.c_int))[0] = self.value if self.get_value is None else self.get_value
        return 0


def test_pdeathsig_hook_requires_success_and_the_supervisor_as_parent():
    import os

    pdeathsig_hook(os.getppid(), libc=_FakeLibc(), getppid=os.getppid)()  # registered, parent verified
    with pytest.raises(OSError, match="PR_SET_PDEATHSIG"):
        pdeathsig_hook(os.getppid(), libc=_FakeLibc(set_rc=-1))()
    with pytest.raises(OSError, match="did not take effect"):
        pdeathsig_hook(os.getppid(), libc=_FakeLibc(get_value=0))()
    with pytest.raises(RuntimeError, match="supervisor died"):
        pdeathsig_hook(12345, libc=_FakeLibc(), getppid=lambda: 1)()  # re-parented


_SPAWNER = """
import os, subprocess, sys, time
sys.path.insert(0, {src!r})
from ptagwb.budget import pdeathsig_hook
mode, ready, mark = sys.argv[1:4]
hook = pdeathsig_hook(os.getpid())
def pre():
    time.sleep(1.0)  # the supervisor is SIGKILLed in this window, before registration
    if mode == "round4":
        hook()
    else:  # round 3: prctl without checking the parent
        import ctypes
        ctypes.CDLL(None).prctl(1, 9, 0, 0, 0)
open(ready, "w").close()
subprocess.Popen([sys.executable, "-c", "open(%r, 'w').write('worker ran')" % mark], preexec_fn=pre)
time.sleep(60)
"""


@pytest.mark.parametrize("mode", ["round3", "round4"])
def test_supervisor_dies_before_the_worker_registers(tmp_path, mode):
    """The supervisor dies after forking the worker but before PR_SET_PDEATHSIG is registered (the
    signal is not retroactive). Round 3's hook lets the orphan run (control: the race is real); the
    round-4 hook detects the re-parenting and the worker never executes."""
    import signal
    import subprocess

    ready, mark = tmp_path / "ready", tmp_path / "worker_ran"
    script = tmp_path / "spawner.py"
    script.write_text(_SPAWNER.format(src=str(REPO_ROOT / "src")))
    sup = subprocess.Popen([sys.executable, str(script), mode, str(ready), str(mark)])
    t = time.monotonic()
    while not ready.exists() and time.monotonic() - t < 30:
        time.sleep(0.02)
    assert ready.exists()
    time.sleep(0.3)
    sup.send_signal(signal.SIGKILL)
    sup.wait()
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and not mark.exists():
        time.sleep(0.05)
    assert mark.exists() == (mode == "round3")


_LIVE = """
import sys
sys.path.insert(0, {src!r})
from ptagwb.budget import supervise
supervise([sys.executable, "-c", "import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % sys.argv[1]],
          run_id="z", kill_after_s=120.0, record_path=sys.argv[2])
"""


def test_worker_dies_with_its_supervisor(tmp_path):
    import os
    import signal
    import subprocess

    pidf, script = tmp_path / "worker.pid", tmp_path / "live.py"
    script.write_text(_LIVE.format(src=str(REPO_ROOT / "src")))
    sup = subprocess.Popen([sys.executable, str(script), str(pidf), str(tmp_path / "s.json")])
    t = time.monotonic()
    while not (pidf.exists() and pidf.read_text()) and time.monotonic() - t < 30:
        time.sleep(0.02)
    wpid = int(pidf.read_text())
    sup.send_signal(signal.SIGKILL)
    sup.wait()
    t = time.monotonic()
    while time.monotonic() - t < 5:
        try:
            with open(f"/proc/{wpid}/stat") as f:
                if f.read().split(")")[-1].split()[0] == "Z":
                    break  # killed, waiting to be reaped by the new parent
        except FileNotFoundError:
            break
        time.sleep(0.02)
    else:
        os.kill(wpid, signal.SIGKILL)
        pytest.fail("the worker outlived its supervisor")


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
