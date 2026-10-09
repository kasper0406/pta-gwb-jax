"""Mechanical GPU-time cap for the EPTA runs (user decision D4 of 2026-10-09, second set).

* **Ledger** (``data/processed/m3b/epta/runs/ledger.json``, append-only): every run appends an entry
  when it starts (``status="running"``, so a crashed run still counts up to its last heartbeat)
  and updates it at every heartbeat and at the end with the elapsed wall time. On the single GPU,
  GPU-hours = wall-hours of a process holding the GPU (compilation included).
* **Caps**: total 12 GPU-h; per phase pilot <= 2, production (incl. reweighting) <= 8,
  contingency <= 2. A run may start only if its own ``max_gpu_hours`` fits in what is left of its
  phase and of the total.
* **Inside blocks**: ``Deadline`` gives the absolute monotonic deadline of a run; the driver
  sizes its chunks so that the predicted end of the next chunk stays before it, checks it after
  every chunk, and a watchdog thread terminates the process (after writing the ledger) if a chunk
  overruns the deadline by more than ``grace_s``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

CAPS_H = {"total": 12.0, "pilot": 2.0, "production": 8.0, "contingency": 2.0}


class BudgetExceeded(RuntimeError):
    """A run would exceed (or has exceeded) its phase or the total GPU-time cap."""


class Ledger:
    def __init__(self, path: Path | str, caps: dict | None = None):
        self.path = Path(path)
        self.caps = dict(CAPS_H if caps is None else caps)
        if set(self.caps) != set(CAPS_H):
            raise ValueError(f"caps must define {sorted(CAPS_H)}")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        d = json.loads(self.path.read_text())
        if not isinstance(d, list):
            raise BudgetExceeded("corrupt ledger")
        return d

    def _write(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1))
        os.replace(tmp, self.path)

    def used_h(self, phase: str | None = None) -> float:
        return sum(e["seconds"] for e in self.entries() if phase is None or e["phase"] == phase) / 3600.0

    def remaining_h(self, phase: str) -> float:
        if phase not in self.caps or phase == "total":
            raise ValueError(f"unknown phase {phase!r}")
        return min(self.caps[phase] - self.used_h(phase), self.caps["total"] - self.used_h())

    def open(self, run_id: str, phase: str, max_gpu_hours: float, meta: dict) -> int:
        """Register a run; raises BudgetExceeded if its allocation does not fit."""
        if any(e["run_id"] == run_id for e in self.entries()):
            raise BudgetExceeded(f"run {run_id!r} already in the ledger (runs are never repeated or resumed silently)")
        rem = self.remaining_h(phase)
        if not (max_gpu_hours > 0 and max_gpu_hours <= rem + 1e-12):
            raise BudgetExceeded(f"run {run_id}: max_gpu_hours {max_gpu_hours} exceeds what is left of {phase!r} "
                                 f"and the total ({rem:.3f} GPU-h)")
        e = self.entries()
        e.append({"run_id": run_id, "phase": phase, "max_gpu_hours": max_gpu_hours, "status": "running",
                  "started_unix": time.time(), "seconds": 0.0, "meta": meta})
        self._write(e)
        return len(e) - 1

    def update(self, run_id: str, seconds: float, status: str = "running", info: dict | None = None) -> None:
        e = self.entries()
        for x in e:
            if x["run_id"] == run_id:
                x["seconds"] = float(max(x["seconds"], seconds))
                x["status"] = status
                if info:
                    x.setdefault("info", {}).update(info)
                self._write(e)
                return
        raise BudgetExceeded(f"run {run_id!r} not in the ledger")


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
    """Terminates the process if the deadline is overrun by more than ``grace_s`` (e.g. a chunk
    that runs long). Before exiting it calls ``on_kill(elapsed_s)`` (writes the ledger)."""

    def __init__(self, deadline: Deadline, on_kill, poll_s: float = 5.0, exit_fn=os._exit, on_beat=None,
                 beat_s: float = 60.0):
        self.d, self.on_kill, self.poll_s, self.exit_fn = deadline, on_kill, poll_s, exit_fn
        self.on_beat, self.beat_s, self._last_beat = on_beat, beat_s, 0.0
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.poll_s):
            if self.on_beat is not None and self.d.elapsed - self._last_beat >= self.beat_s:
                self._last_beat = self.d.elapsed
                self.on_beat(self.d.elapsed)  # ledger heartbeat: a crashed run still counts
            if self.d.elapsed > self.d.limit_s + self.d.grace_s:
                try:
                    self.on_kill(self.d.elapsed)
                finally:
                    self.exit_fn(3)
                return

    def start(self):
        self.t.start()
        return self

    def stop(self):
        self._stop.set()
