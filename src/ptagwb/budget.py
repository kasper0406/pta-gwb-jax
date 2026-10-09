"""Mechanical GPU-time cap for the EPTA runs (user decision D4 of 2026-10-09, second set; review
round 2 of M3b-0E).

* **Ledger** (``data/processed/m3b/epta/runs/ledger.json``): one entry per run. Every read-modify-
  write (admission, heartbeat, close) holds an exclusive ``fcntl`` lock on ``ledger.json.lock``
  and writes through a unique temporary file in the same directory followed by an atomic rename.
* **Charging (conservative):** a *closed* entry (``status`` not ``"running"``) is charged its
  recorded elapsed time; an *open* entry (``"running"``: active, or crashed without
  reconciliation) is charged its **full requested allocation**. Admission therefore reserves the
  whole allocation of every active run, and a crash is charged in full until a human reconciles it
  (``reconcile``, which can only lower the charge to the last recorded heartbeat if the process is
  verifiably dead and the operator says so).
* **Caps**: total 12 GPU-h; per phase pilot <= 2, production (incl. reweighting) <= 8,
  contingency <= 2. A run is admitted only if its allocation fits in what is left of its phase and
  of the total, under the charging rule above.
* **Exclusive GPU lock**: ``GpuLock`` holds an ``fcntl`` lock on a per-device lock file for the
  whole run; a second run (of this project) cannot start while it is held. ``nvidia-smi`` must also
  show no other compute process (other users' processes do not take the lock).
* **Inside blocks**: ``Deadline`` gives the run's monotonic deadline; the driver sizes its chunks so
  that the predicted end of the next chunk stays before it; a ``Watchdog`` thread terminates the
  process if a chunk overruns, and *also* if a heartbeat (ledger write) fails: termination never
  depends on a successful ledger write.
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

    def reconcile(self, run_id: str, *, operator_note: str) -> None:
        """Close a crashed (still open) entry at its last heartbeat. Only for a process that is
        verifiably gone (its pid no longer exists); requires an operator note, recorded."""
        with self._locked():
            e = self._read()
            for x in e:
                if x["run_id"] == run_id and x["status"] == OPEN:
                    try:
                        os.kill(int(x["pid"]), 0)
                        raise BudgetExceeded(f"run {run_id!r}: process {x['pid']} still exists")
                    except ProcessLookupError:
                        pass
                    x["status"] = "crashed_reconciled"
                    x.setdefault("info", {})["reconcile_note"] = operator_note
                    self._write(e)
                    return
            raise BudgetExceeded(f"no open entry {run_id!r}")


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


class Watchdog:
    """Terminates the process (``exit_fn(3)``) if the deadline is overrun by more than
    ``grace_s``, or if a heartbeat (``on_beat``) raises. ``on_kill`` (best-effort ledger close) is
    attempted first; its failure does not prevent the exit."""

    def __init__(self, deadline: Deadline, on_kill, poll_s: float = 5.0, exit_fn=os._exit, on_beat=None,
                 beat_s: float = 60.0):
        self.d, self.on_kill, self.poll_s, self.exit_fn = deadline, on_kill, poll_s, exit_fn
        self.on_beat, self.beat_s, self._last_beat = on_beat, beat_s, 0.0
        self.reason = None
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _terminate(self, reason: str):
        self.reason = reason
        try:
            self.on_kill(self.d.elapsed, reason)
        except Exception:  # noqa: BLE001  termination must not depend on the ledger
            pass
        finally:
            self.exit_fn(3)

    def _run(self):
        while not self._stop.wait(self.poll_s):
            try:
                if self.on_beat is not None and self.d.elapsed - self._last_beat >= self.beat_s:
                    self._last_beat = self.d.elapsed
                    self.on_beat(self.d.elapsed)
            except Exception as e:  # noqa: BLE001
                self._terminate(f"heartbeat failed: {e!r}")
                return
            if self.d.elapsed > self.d.limit_s + self.d.grace_s:
                self._terminate("deadline overrun")
                return

    def start(self):
        self.t.start()
        return self

    def stop(self):
        self._stop.set()
