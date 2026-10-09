"""D4 (2026-10-09, second set): the mechanical GPU-time cap (``ptagwb.budget``) and the run
driver's config validation (``scripts/m3b_run_epta.py``). No GPU, no sampling."""

from __future__ import annotations

import json
import sys
import time

import pytest

from ptagwb.budget import CAPS_H, BudgetExceeded, Deadline, Ledger, Watchdog
from ptagwb.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_caps_are_the_user_decision():
    assert CAPS_H == {"total": 12.0, "pilot": 2.0, "production": 8.0, "contingency": 2.0}


def test_ledger_admits_and_refuses(tmp_path):
    L = Ledger(tmp_path / "ledger.json")
    L.open("p1", "pilot", 1.5, {})
    L.update("p1", 1.2 * 3600, "completed")
    assert L.used_h("pilot") == pytest.approx(1.2)
    with pytest.raises(BudgetExceeded):  # 0.8 h left in the pilot phase
        L.open("p2", "pilot", 0.9, {})
    L.open("p2", "pilot", 0.8, {})
    with pytest.raises(BudgetExceeded):  # never repeated / resumed silently
        L.open("p2", "pilot", 0.1, {})
    L.update("p2", 0.8 * 3600, "completed")
    L.open("prod", "production", 8.0, {})
    L.update("prod", 8.0 * 3600, "completed")
    # total 12: pilot 2 + production 8 used -> contingency 2 left, nothing more
    assert L.remaining_h("contingency") == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        L.open("c", "contingency", 2.5, {})
    with pytest.raises(BudgetExceeded):
        L.open("z", "production", 0.1, {})
    with pytest.raises(BudgetExceeded):
        L.open("neg", "contingency", 0.0, {})


def test_crashed_run_still_counts(tmp_path):
    L = Ledger(tmp_path / "ledger.json")
    L.open("crash", "pilot", 1.0, {})
    L.update("crash", 1800.0)  # heartbeat, then the process dies
    assert json.loads((tmp_path / "ledger.json").read_text())[0]["status"] == "running"
    assert L.remaining_h("pilot") == pytest.approx(1.5)


def test_deadline_and_watchdog_kill_inside_a_block(tmp_path):
    d = Deadline(time.monotonic(), limit_s=0.2, grace_s=0.1)
    assert d.allows(0.1) and not d.allows(1.0)
    killed = {}
    L = Ledger(tmp_path / "ledger.json")
    L.open("r", "pilot", 1.0, {})

    def on_kill(el):
        L.update("r", el, "killed_at_cap")
        killed["el"] = el

    wd = Watchdog(d, on_kill, poll_s=0.02, exit_fn=lambda code: killed.setdefault("code", code),
                  on_beat=lambda el: L.update("r", el), beat_s=0.05).start()
    time.sleep(0.6)  # a "chunk" that overruns the deadline
    wd.stop()
    assert killed["code"] == 3 and killed["el"] > 0.3
    e = L.entries()[0]
    assert e["status"] == "killed_at_cap" and e["seconds"] > 0.3


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
