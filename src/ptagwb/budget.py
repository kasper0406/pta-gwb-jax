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
* **Reconciliation** of an open entry needs the supervisor's lock-free record (time before the
  worker started, time after it was reaped): the charge is that span rounded up to whole minutes
  plus one minute. Process death alone never reduces the charge.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

CAPS_H = {"total": 12.0, "pilot": 2.0, "production": 8.0, "contingency": 2.0}
OPEN = "running"


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
        the launching supervisor process, never from the worker's heartbeats: the record holds the
        wall-clock time taken *before* the worker was started and the time *after* it was reaped
        (``waitpid``), so end - start bounds the consumption from above. The charge is that span
        rounded up to whole minutes, plus one minute, capped at the allocation. Without a complete
        record (e.g. the supervisor itself died), nothing changes: the full allocation stays
        charged. Returns the charged seconds."""
        rec = SupervisorRecord.read(record_path)
        if rec.get("run_id") != run_id or "start_unix" not in rec or "end_unix" not in rec:
            raise BudgetExceeded(f"run {run_id!r}: no complete supervisor record (start and reaped end); "
                                 "the full allocation stays charged")
        with self._locked():
            e = self._read()
            for x in e:
                if x["run_id"] == run_id and x["status"] == OPEN:
                    charge = conservative_charge_s(rec["start_unix"], rec["end_unix"], x["max_gpu_hours"])
                    x["status"] = "reconciled_from_supervisor_record"
                    x["seconds"] = charge
                    x.setdefault("info", {})["reconcile"] = {"note": operator_note, "record": rec}
                    self._write(e)
                    return charge
            raise BudgetExceeded(f"no open entry {run_id!r}")


def conservative_charge_s(start_unix: float, end_unix: float, max_gpu_hours: float) -> float:
    """Upper bound of the consumption from supervisor timestamps: (end - start) rounded up to whole
    minutes plus one minute, never more than the allocation."""
    import math

    if not (end_unix >= start_unix):
        raise BudgetExceeded("supervisor record: end before start")
    return float(min(60.0 * (math.ceil((end_unix - start_unix) / 60.0) + 1), max_gpu_hours * 3600.0))


class SupervisorRecord:
    """Lock-free record of a run, written only by its supervisor (atomic temp file + rename): the
    wall-clock time before the worker was started, and after it was reaped."""

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


def supervise(cmd: list[str], *, run_id: str, kill_after_s: float, record_path: Path | str, env=None,
              poll_s: float = 0.05, on_exit=None) -> dict:
    """Run ``cmd`` as a worker and SIGKILL it ``kill_after_s`` seconds after starting it, whatever it
    is doing. The kill path uses only ``time.monotonic``, ``waitpid`` and ``kill``: no locks, no
    ledger. The supervisor record (start before ``Popen``; end after the reap) is written before
    and after (lock-free, atomic); then ``on_exit(record)`` (e.g. the best-effort ledger close)
    is called, and its failure leaves the allocation open (charged in full). The worker gets
    PR_SET_PDEATHSIG = SIGKILL, so it dies with the supervisor."""
    import signal
    import subprocess

    def _pdeathsig():
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except Exception:  # noqa: BLE001
            pass

    rec = {"run_id": run_id, "start_unix": time.time(), "kill_after_s": kill_after_s, "cmd": cmd}
    SupervisorRecord.write(record_path, rec)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, preexec_fn=_pdeathsig)
    rec["worker_pid"] = proc.pid
    killed = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.monotonic() - t0 >= kill_after_s:
            proc.kill()
            killed = True
            rc = proc.wait()
            break
        time.sleep(poll_s)
    rec.update({"end_unix": time.time(), "elapsed_monotonic": time.monotonic() - t0, "returncode": rc,
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
