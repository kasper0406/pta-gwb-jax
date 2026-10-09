"""Mechanical GPU-time cap for the EPTA runs (user decision D4 of 2026-10-09, second set; review
round 2 of M3b-0E).

* **Ledger** (``data/processed/m3b/epta/runs/ledger.json``): one entry per run. Every read-modify-
  write (admission, heartbeat, close) holds an exclusive ``fcntl`` lock on ``ledger.json.lock``
  and writes through a unique temporary file in the same directory followed by an atomic rename.
* **Charging (conservative):** a *closed* entry (``status`` not ``"running"``) is charged its
  recorded time, which the supervisor sets from its own timestamps (``conservative_charge_s``),
  never from the worker's heartbeats; an *open* entry (``"running"``: active, or crashed without
  reconciliation) is charged its **full requested allocation**.
* **Caps**: total 12 GPU-h; per phase pilot <= 2, production (incl. reweighting) <= 8,
  contingency <= 2. A run is admitted only if its allocation fits in what is left of its phase and
  of the total, under the charging rule above.
* **Exclusive GPU lock**: ``GpuLock`` holds an ``fcntl`` lock on a per-device lock file for the
  whole run; a second run (of this project) cannot start while it is held. ``nvidia-smi`` must also
  show no other compute process (other users' processes do not take the lock).
* **Deadlines, independent of the ledger** (review round 3): the run is a worker process started
  by a supervisor (``supervise``), which SIGKILLs it at the hard deadline using only monotonic time,
  ``waitpid`` and ``kill`` (no lock, no ledger, no file I/O on the kill path). Inside the worker, a
  ``HardDeadline`` timer thread calls ``os._exit`` slightly earlier, also without I/O; the chunk
  planner (``Deadline``) stops between chunks earlier still. Heartbeats (``Watchdog``) only write
  the ledger; a failing heartbeat ends the worker. Every ledger write after admission is
  best-effort: if the terminal write does not happen, the entry stays open and is charged in full.
* **Charging clock** (review round 4): every charge comes from a *duration* the supervisor measures
  itself with ``CLOCK_BOOTTIME`` (monotonic, immune to wall-clock steps, and it keeps counting
  through a suspend): the time from before the worker was started to after it was reaped. Wall-clock
  timestamps are recorded for information only and never enter a charge.
* **Reconciliation** of an open entry reads the supervisor's lock-free record. With the measured
  duration (written only after the reap), the charge is that duration rounded up to whole minutes
  plus one minute; without it (e.g. the supervisor died), the entry is closed at its **full
  allocation**. Process death alone never reduces the charge.
* **Parent-death registration** (review round 4): the worker's pre-exec hook (``pdeathsig_hook``)
  must set PR_SET_PDEATHSIG = SIGKILL successfully (read back with PR_GET_PDEATHSIG) and then
  verify that its parent is still the supervisor PID captured before spawning; otherwise the worker
  is never executed (the hook raises, the child exits with status 255).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

CAPS_H = {"total": 12.0, "pilot": 2.0, "production": 8.0, "contingency": 2.0}
OPEN = "running"
CLOCK = "CLOCK_BOOTTIME"
PR_SET_PDEATHSIG, PR_GET_PDEATHSIG = 1, 2


def clock_s() -> float:
    """The charging clock: CLOCK_BOOTTIME (monotonic; not affected by wall-clock steps; counts
    suspend)."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


class BudgetExceeded(RuntimeError):
    """A run would exceed (or has exceeded) its phase or the total GPU-time cap."""


class Ledger:
    def __init__(self, path: Path | str, caps: dict | None = None):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.caps = dict(CAPS_H if caps is None else caps)
        if set(self.caps) != set(CAPS_H):
            raise ValueError(f"caps must define {sorted(CAPS_H)}")

    # ------------------------------------------------------------- locked I/O
    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        d = json.loads(self.path.read_text())
        if not isinstance(d, list):
            raise BudgetExceeded("corrupt ledger")
        return d

    def _write(self, entries: list[dict]) -> None:
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def entries(self) -> list[dict]:
        with self._locked():
            return self._read()

    # ------------------------------------------------------------- accounting
    @staticmethod
    def charge_s(e: dict) -> float:
        """Seconds charged to an entry: its full allocation while open (active or unreconciled
        crash), its recorded elapsed time once closed."""
        if e["status"] == OPEN:
            return float(e["max_gpu_hours"]) * 3600.0
        return float(e["seconds"])

    @classmethod
    def _used_h(cls, entries, phase=None) -> float:
        return sum(cls.charge_s(e) for e in entries if phase is None or e["phase"] == phase) / 3600.0

    def used_h(self, phase: str | None = None) -> float:
        return self._used_h(self.entries(), phase)

    def _remaining_h(self, entries, phase) -> float:
        if phase not in self.caps or phase == "total":
            raise ValueError(f"unknown phase {phase!r}")
        return min(self.caps[phase] - self._used_h(entries, phase), self.caps["total"] - self._used_h(entries))

    def remaining_h(self, phase: str) -> float:
        return self._remaining_h(self.entries(), phase)

    # ------------------------------------------------------------- transitions
    def open(self, run_id: str, phase: str, max_gpu_hours: float, meta: dict) -> None:
        """Atomically admit a run (reserving its full allocation) or raise BudgetExceeded."""
        with self._locked():
            e = self._read()
            if any(x["run_id"] == run_id for x in e):
                raise BudgetExceeded(f"run {run_id!r} already in the ledger (runs are never repeated or resumed)")
            rem = self._remaining_h(e, phase)
            if not (max_gpu_hours > 0 and max_gpu_hours <= rem + 1e-12):
                raise BudgetExceeded(f"run {run_id}: max_gpu_hours {max_gpu_hours} exceeds what is left of {phase!r} "
                                     f"and the total ({rem:.3f} GPU-h, active allocations reserved in full)")
            e.append({"run_id": run_id, "phase": phase, "max_gpu_hours": float(max_gpu_hours), "status": OPEN,
                      "started_unix": time.time(), "seconds": 0.0, "pid": os.getpid(), "meta": meta})
            self._write(e)

    def heartbeat(self, run_id: str, seconds: float) -> None:
        self._set(run_id, seconds, None, None)

    def close(self, run_id: str, seconds: float, status: str, info: dict | None = None) -> None:
        if status == OPEN:
            raise ValueError("close needs a terminal status")
        self._set(run_id, seconds, status, info)

    def _set(self, run_id, seconds, status, info):
        with self._locked():
            e = self._read()
            for x in e:
                if x["run_id"] == run_id:
                    if x["status"] != OPEN:
                        raise BudgetExceeded(f"run {run_id!r} is already closed")
                    x["seconds"] = float(max(x["seconds"], seconds))
                    if status is not None:
                        x["status"] = status
                    if info:
                        x.setdefault("info", {}).update(info)
                    self._write(e)
                    return
            raise BudgetExceeded(f"run {run_id!r} not in the ledger")

    def reconcile(self, run_id: str, record_path: Path | str, *, operator_note: str) -> float:
        """Close a still-open entry from the **supervisor record** (``SupervisorRecord``) written by
        the launching supervisor process, never from the worker's heartbeats or from wall-clock
        timestamps. If the record holds the duration the supervisor measured on ``CLOCK_BOOTTIME``
        from before the worker started to after it was reaped (``elapsed_s``, written only after the
        reap), the charge is that duration rounded up to whole minutes plus one minute, capped at the
        allocation. Otherwise (no record, another run, no reap: e.g. the supervisor itself died)
        there is no reliable bound and the entry is closed at its **full allocation**. Returns the
        charged seconds."""
        rec = SupervisorRecord.read(record_path)
        el = rec.get("elapsed_s")
        bounded = (rec.get("run_id") == run_id and rec.get("clock") == CLOCK and rec.get("reaped") is True
                   and isinstance(el, (int, float)) and not isinstance(el, bool) and math.isfinite(el) and el >= 0)
        with self._locked():
            e = self._read()
            for x in e:
                if x["run_id"] == run_id and x["status"] == OPEN:
                    full = float(x["max_gpu_hours"]) * 3600.0
                    charge = conservative_charge_s(el, x["max_gpu_hours"]) if bounded else full
                    x["status"] = ("reconciled_from_supervisor_record" if bounded
                                   else "reconciled_full_allocation_no_reliable_bound")
                    x["seconds"] = charge
                    x.setdefault("info", {})["reconcile"] = {"note": operator_note, "record": rec, "bounded": bounded}
                    self._write(e)
                    return charge
            raise BudgetExceeded(f"no open entry {run_id!r}")


def conservative_charge_s(elapsed_s: float, max_gpu_hours: float) -> float:
    """Upper bound of the consumption from the supervisor's measured ``CLOCK_BOOTTIME`` duration:
    rounded up to whole minutes plus one minute, never more than the allocation. A missing or
    invalid duration is charged the full allocation."""
    full = float(max_gpu_hours) * 3600.0
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)) or not math.isfinite(elapsed_s) \
            or elapsed_s < 0:
        return full
    return float(min(60.0 * (math.ceil(elapsed_s / 60.0) + 1), full))


class SupervisorRecord:
    """Lock-free record of a run, written only by its supervisor (atomic temp file + rename): before
    the worker is started, and after it was reaped, with the duration measured on ``CLOCK_BOOTTIME``
    (``elapsed_s``). Wall-clock fields (``*_unix``) are informational."""

    @staticmethod
    def write(path: Path | str, rec: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def read(path: Path | str) -> dict:
        p = Path(path)
        return json.loads(p.read_text()) if p.exists() else {}


class GpuLock:
    """Exclusive, non-blocking per-device lock held for a run's lifetime."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.f = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            self.f.close()
            raise BudgetExceeded(f"GPU lock {self.path} is held by another run") from e
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        self.f.close()
        return False


@dataclass
class Deadline:
    """Monotonic deadline of one run (seconds since ``start``)."""

    start: float
    limit_s: float
    grace_s: float = 60.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def left(self) -> float:
        return self.limit_s - self.elapsed

    def allows(self, predicted_s: float) -> bool:
        """True if a block predicted to take ``predicted_s`` ends before the deadline."""
        return predicted_s <= self.left


class HardDeadline:
    """Lock-free hard deadline inside the worker: a dedicated timer thread that does nothing but
    sleep until ``deadline_monotonic`` and call ``exit_fn(4)`` (``os._exit``: no ledger, no file
    I/O, no locks, no cleanup). It cannot be blocked by heartbeats or ledger writes. The
    supervisor's SIGKILL (``supervise``) is the outer, independent line."""

    def __init__(self, deadline_monotonic: float, exit_fn=os._exit):
        self.deadline, self.exit_fn = deadline_monotonic, exit_fn
        self.fired = None
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            left = self.deadline - time.monotonic()
            if left <= 0:
                self.fired = time.monotonic()
                self.exit_fn(4)
                return
            self._stop.wait(min(left, 0.05))

    def start(self):
        self.t.start()
        return self

    def stop(self):
        self._stop.set()


class Watchdog:
    """Heartbeats only: calls ``on_beat(elapsed)`` every ``beat_s``; if it raises, ``exit_fn(3)``.
    Deadline enforcement is NOT done here (a heartbeat may block on the ledger lock or the file
    system): see ``HardDeadline`` and ``supervise``."""

    def __init__(self, deadline: Deadline, on_beat, beat_s: float = 60.0, poll_s: float = 1.0, exit_fn=os._exit):
        self.d, self.on_beat, self.beat_s, self.poll_s, self.exit_fn = deadline, on_beat, beat_s, poll_s, exit_fn
        self._last = 0.0
        self.reason = None
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.poll_s):
            if self.d.elapsed - self._last >= self.beat_s:
                self._last = self.d.elapsed
                try:
                    self.on_beat(self.d.elapsed)
                except Exception as e:  # noqa: BLE001
                    self.reason = f"heartbeat failed: {e!r}"
                    self.exit_fn(3)
                    return

    def start(self):
        self.t.start()
        return self

    def stop(self):
        self._stop.set()


def pdeathsig_hook(supervisor_pid: int, *, libc=None, getppid=os.getppid):
    """Return the worker's pre-exec hook. Everything that might allocate or import is prepared here,
    in the supervisor, before the fork. In the child the hook (1) sets PR_SET_PDEATHSIG = SIGKILL
    and requires success, reading the value back with PR_GET_PDEATHSIG; (2) then checks that its
    parent is still ``supervisor_pid`` (captured before spawning). If the supervisor died before
    step (1) took effect, the child was re-parented and the check fails. Any failure raises, so the
    worker is never executed (``subprocess`` reports the error and the child exits with 255)."""
    import ctypes
    import signal

    if libc is None:
        libc = ctypes.CDLL(None, use_errno=True)
    out = ctypes.c_int(-1)
    sig = int(signal.SIGKILL)

    def hook():
        if libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if libc.prctl(PR_GET_PDEATHSIG, ctypes.byref(out), 0, 0, 0) != 0 or out.value != sig:
            raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG did not take effect")
        if getppid() != supervisor_pid:
            raise RuntimeError("the supervisor died before the worker registered its parent-death signal")

    return hook


def supervise(cmd: list[str], *, run_id: str, kill_after_s: float, record_path: Path | str, env=None,
              poll_s: float = 0.05, on_exit=None) -> dict:
    """Run ``cmd`` as a worker and SIGKILL it ``kill_after_s`` seconds (``CLOCK_BOOTTIME``) after
    starting it, whatever it is doing. The kill path uses only the clock, ``waitpid`` and ``kill``:
    no locks, no ledger. The supervisor record is written before the start and after the reap
    (lock-free, atomic); the second write adds ``elapsed_s``, the duration from before ``Popen`` to
    after the reap on ``CLOCK_BOOTTIME``, which is the only quantity charges are computed from. Then
    ``on_exit(record)`` (e.g. the best-effort ledger close) is called; its failure leaves the
    allocation open (charged in full). The worker registers PR_SET_PDEATHSIG = SIGKILL and verifies
    its parent before it is executed (``pdeathsig_hook``), so it dies with the supervisor."""
    import subprocess

    sup_pid = os.getpid()
    hook = pdeathsig_hook(sup_pid)
    rec = {"run_id": run_id, "clock": CLOCK, "supervisor_pid": sup_pid, "start_unix_info": time.time(),
           "kill_after_s": kill_after_s, "cmd": cmd, "reaped": False}
    SupervisorRecord.write(record_path, rec)
    t0 = clock_s()
    # the hook was fully prepared before the fork and only calls prctl/getppid in the child; the
    # driver's supervisor runs no other threads
    proc = subprocess.Popen(cmd, env=env, preexec_fn=hook)  # noqa: PLW1509
    rec["worker_pid"] = proc.pid
    killed = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if clock_s() - t0 >= kill_after_s:
            proc.kill()
            killed = True
            rc = proc.wait()
            break
        time.sleep(poll_s)
    rec.update({"elapsed_s": clock_s() - t0, "reaped": True, "end_unix_info": time.time(), "returncode": rc,
                "killed_at_deadline": killed})
    try:
        SupervisorRecord.write(record_path, rec)
    except Exception as e:  # noqa: BLE001
        rec["record_write_error"] = repr(e)
    if on_exit is not None:
        try:
            on_exit(rec)
        except Exception as e:  # noqa: BLE001
            rec["on_exit_error"] = repr(e)
    return rec
