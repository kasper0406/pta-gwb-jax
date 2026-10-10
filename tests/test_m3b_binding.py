"""Evidence binding scheme 2 (review round 4): the control plane is excluded from the gate-evidence
binding, provably cannot enter a gate computation, and legacy results are only re-stamped by a
verified migration (``scripts/m3b_rebind_evidence.py``)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from ptagwb import binding as B
from ptagwb.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

CONTROL_MODULES = {"ptagwb.budget", "m3b_run_epta", "m3b_rebind_evidence", "m3b_tuning_artifacts", "m3b_pilot_report"}
RUN_PLANE_PATHS = ("run_configs", "configs/m3b/metrics", "configs/m3b/proposals", "tuning_staging")
# names of ptagwb.binding a gate computation may import: stamping and checking only
BINDING_NAMES_ALLOWED = {"evidence_binding", "require_bound", "StaleEvidenceError", "runtime_file_hashes"}


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, None
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:  # relative import inside src/ptagwb
                mod = "ptagwb." + mod if mod else "ptagwb"
            for a in node.names:
                yield mod, a.name


def test_control_plane_is_not_imported_by_any_bound_file():
    sets = B.code_file_sets(B.worktree_listing())
    bound = sets["source"] + sets["scripts"] + sets["oracles"]
    assert bound and not set(bound) & set(B.CONTROL_PLANE)
    for rel in bound:
        for mod, name in _imports(REPO_ROOT / rel):
            full = f"{mod}.{name}" if name else mod
            assert mod not in CONTROL_MODULES and full not in CONTROL_MODULES, f"{rel} imports {full}"
            if mod == "ptagwb.binding" or full == "ptagwb.binding":
                assert name in BINDING_NAMES_ALLOWED, f"{rel} imports binding.{name}"
        text = (REPO_ROOT / rel).read_text()
        assert "importlib" not in text or rel.endswith("binding.py"), rel
        assert not any(t in text for t in RUN_PLANE_PATHS), f"{rel} reads a run-plane path"


def test_control_plane_files_exist_and_are_excluded():
    for r in B.CONTROL_PLANE:
        assert (REPO_ROOT / r).exists(), r
    listing = B.worktree_listing()
    assert set(B.CONTROL_PLANE) <= set(listing)
    full = B.code_file_sets(listing, scheme=1)
    assert set(B.CONTROL_PLANE) <= set(full["source"] + full["scripts"])
    now = B.code_file_sets(listing)
    assert not any(r.startswith(B.RUN_CONFIG_PREFIXES) for r in now["configs"])
    assert any(r.startswith("configs/m3b/run_configs/") for r in full["configs"])


def test_scheme2_ignores_control_plane_content_but_not_numerics():
    listing = ["src/ptagwb/budget.py", "src/ptagwb/epta.py", "scripts/m3b_run_epta.py", "scripts/m3b_t1.py",
               "scripts/m3b_pilot_report.py", "scripts/m3b_tuning_artifacts.py", "configs/m3b/acceptance_epta.json",
               "configs/m3b/run_configs/p.json", "configs/m3b/proposals/q.json", *B.ORACLES]
    files = {r: r.encode() for r in listing}

    def b(**changes):
        d = {**files, **changes}
        return B.code_binding(listing, lambda r: d[r], REPO_ROOT)

    ref = b()
    assert b(**{"src/ptagwb/budget.py": b"changed"}) == ref
    assert b(**{"scripts/m3b_run_epta.py": b"changed"}) == ref
    for f in ("scripts/m3b_pilot_report.py", "scripts/m3b_tuning_artifacts.py", "configs/m3b/run_configs/p.json",
              "configs/m3b/proposals/q.json"):
        assert b(**{f: b"changed"}) == ref, f
    assert b(**{"src/ptagwb/epta.py": b"changed"}) != ref
    assert b(**{"scripts/m3b_t1.py": b"changed"}) != ref
    assert b(**{"configs/m3b/acceptance_epta.json": b"{}"}) != ref
    assert b(**{"tests/m3b_arbiter.py": b"changed"}) != ref


def test_legacy_digest_is_reproduced():
    """code_binding with no exclusion equals the scheme-1 formula (globs + _group)."""
    root = REPO_ROOT
    legacy = {"code": {"source": B._group(sorted((root / "src" / "ptagwb").glob("*.py"))),
                       "scripts": B._group(sorted((root / "scripts").glob("m3b_*.py"))),
                       "oracles": B._group([root / r for r in B.ORACLES])},
              "configs": B._group(sorted((root / "configs" / "m3b").rglob("*.json")))}
    assert B.code_binding(B.worktree_listing(), lambda r: (root / r).read_bytes(), root, scheme=1) == legacy


@pytest.fixture
def rebind(monkeypatch, tmp_path):
    import m3b_rebind_evidence as R

    by_scheme = {1: {"code": "L1", "inputs": "I"}, 2: {"scheme": 2, "code": "L2", "inputs": "I"},
                 3: {"scheme": 3, "code": "C", "inputs": "I"}}
    current = dict(by_scheme[3])
    state = {"target": dict(current), "changed": ["src/ptagwb/budget.py", "configs/m3b/run_configs/v2.json"]}
    monkeypatch.setattr(R, "_git", lambda *a, **k: b"0" * 40 + b"\n")
    monkeypatch.setattr(R, "external_binding", lambda: {"inputs": "I"})
    monkeypatch.setattr(R, "binding_at", lambda c, ext, k: state["target"] if k == 3 else by_scheme[k])
    monkeypatch.setattr(R, "evidence_binding", lambda: current)
    monkeypatch.setattr(R, "changed_since", lambda c: state["changed"])
    old = {"from_commit": "f" * 40, "legacy_binding": by_scheme[1], "files_changed_since": ["src/ptagwb/budget.py"],
           "rule": "r4"}
    files = {"s1.json": {"binding": by_scheme[1]}, "s2.json": {"binding": by_scheme[2]},
             "s2_migrated.json": {"binding": by_scheme[2], "binding_migration": old},
             "stale.json": {"binding": {"scheme": 2, "code": "older", "inputs": "I"}}}
    for name, extra in files.items():
        (tmp_path / name).write_text(json.dumps({"pass": True, "x": 1.5, **extra}))
    return R, state, tmp_path, by_scheme, current


def test_rebind_restamps_only_verified_results(rebind):
    R, _, d, by_scheme, current = rebind
    rep = R.migrate("C", d)
    assert rep["files"] == {"s1.json": "rebound (scheme 1 -> 3)", "s2.json": "rebound (scheme 2 -> 3)",
                            "s2_migrated.json": "rebound (scheme 2 -> 3)",
                            "stale.json": "STALE: recorded binding is not the scheme-2 binding of the commit (recompute)"}
    g = json.loads((d / "s2_migrated.json").read_text())
    assert g["binding"] == current and g["x"] == 1.5 and "binding_migration" not in g
    h = g["binding_migrations"]
    assert [(m["from_scheme"], m["to_scheme"]) for m in h] == [(1, 2), (2, 3)]
    assert h[0]["from_binding"] == by_scheme[1] and h[0]["to_binding"] == by_scheme[2] == h[1]["from_binding"]
    assert h[1]["to_binding"] == current
    assert json.loads((d / "stale.json").read_text())["binding"]["code"] == "older"  # untouched


@pytest.mark.parametrize("changed", [["src/ptagwb/epta.py"], ["src/ptagwb/budget.py", "configs/m3b/x.json"],
                                     ["scripts/m3b_t1.py"]])
def test_rebind_refuses_if_a_bound_file_changed(rebind, changed):
    R, state, d, *_ = rebind
    state["changed"] = changed
    before = (d / "s2.json").read_text()
    rep = R.migrate("C", d)
    assert "refused" in rep and (d / "s2.json").read_text() == before


def test_rebind_refuses_if_the_current_scheme_of_the_commit_differs(rebind):
    R, state, d, *_ = rebind
    state["target"] = {"scheme": 3, "code": "other", "inputs": "I"}
    assert "refused" in R.migrate("C", d)


def test_migrated_gate_results_reverify():
    """Every migration recorded on a gate result re-verifies from git: its from/to bindings are the
    from/to-scheme bindings of the named commit (with today's inputs), and the last one is the
    current binding."""
    import m3b_rebind_evidence as R

    res = R.RES
    if not res.exists():
        pytest.skip("no gate results in this checkout")
    ext = B.external_binding()
    cur = B.evidence_binding()
    cache = {}
    for p in sorted(res.glob("*.json")):
        r = json.loads(p.read_text())
        assert "binding_migration" not in r or r.get("binding", {}).get("scheme") == 2, p.name
        hist = r.get("binding_migrations")
        if not hist:
            continue
        for m in hist:
            for k in (m["from_scheme"], m["to_scheme"]):
                if (m["from_commit"], k) not in cache:
                    cache[(m["from_commit"], k)] = R.binding_at(m["from_commit"], ext, k)
            assert m["from_binding"] == cache[(m["from_commit"], m["from_scheme"])], p.name
            assert m["to_binding"] == cache[(m["from_commit"], m["to_scheme"])], p.name
            assert all(R.excluded_now(f) or m["to_scheme"] < B.SCHEME for f in m["files_changed_since"]), p.name
        assert hist[-1]["to_binding"] == r["binding"] == cur, p.name
