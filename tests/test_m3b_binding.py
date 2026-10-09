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

CONTROL_MODULES = {"ptagwb.budget", "m3b_run_epta", "m3b_rebind_evidence"}
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
        assert "importlib" not in (REPO_ROOT / rel).read_text() or rel.endswith("binding.py"), rel


def test_control_plane_files_exist_and_are_excluded():
    for r in B.CONTROL_PLANE:
        assert (REPO_ROOT / r).exists(), r
    listing = B.worktree_listing()
    assert set(B.CONTROL_PLANE) <= set(listing)
    full = B.code_file_sets(listing, exclude=())
    assert set(B.CONTROL_PLANE) <= set(full["source"] + full["scripts"])


def test_scheme2_ignores_control_plane_content_but_not_numerics():
    listing = ["src/ptagwb/budget.py", "src/ptagwb/epta.py", "scripts/m3b_run_epta.py", "scripts/m3b_t1.py",
               "configs/m3b/acceptance_epta.json", *B.ORACLES]
    files = {r: r.encode() for r in listing}

    def b(**changes):
        d = {**files, **changes}
        return B.code_binding(listing, lambda r: d[r], REPO_ROOT)

    ref = b()
    assert b(**{"src/ptagwb/budget.py": b"changed"}) == ref
    assert b(**{"scripts/m3b_run_epta.py": b"changed"}) == ref
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
    assert B.code_binding(B.worktree_listing(), lambda r: (root / r).read_bytes(), root, exclude=()) == legacy


@pytest.fixture
def rebind(monkeypatch, tmp_path):
    import m3b_rebind_evidence as R

    legacy = {"code": "L", "inputs": "I"}
    current = {"scheme": 2, "code": "C", "inputs": "I"}
    state = {"scheme2_then": dict(current), "changed": ["src/ptagwb/budget.py"]}
    monkeypatch.setattr(R, "_git", lambda *a, **k: b"0" * 40 + b"\n")
    monkeypatch.setattr(R, "external_binding", lambda: {"inputs": "I"})
    monkeypatch.setattr(R, "bindings_at", lambda c, ext: (legacy, state["scheme2_then"]))
    monkeypatch.setattr(R, "evidence_binding", lambda: current)
    monkeypatch.setattr(R, "changed_since", lambda c: state["changed"])
    for name, b in (("good.json", legacy), ("stale.json", {"code": "older", "inputs": "I"})):
        (tmp_path / name).write_text(json.dumps({"pass": True, "x": 1.5, "binding": b}))
    return R, state, tmp_path, legacy, current


def test_rebind_restamps_only_verified_legacy_results(rebind):
    R, _, d, legacy, current = rebind
    rep = R.migrate("C", d)
    assert rep["files"] == {"good.json": "rebound", "stale.json": "STALE: recorded binding is not the legacy "
                                                                  "binding of the commit (recompute)"}
    g = json.loads((d / "good.json").read_text())
    assert g["binding"] == current and g["x"] == 1.5 and g["binding_migration"]["legacy_binding"] == legacy
    assert json.loads((d / "stale.json").read_text())["binding"]["code"] == "older"  # untouched


@pytest.mark.parametrize("changed", [["src/ptagwb/epta.py"], ["src/ptagwb/budget.py", "configs/m3b/x.json"]])
def test_rebind_refuses_if_a_bound_file_changed(rebind, changed):
    R, state, d, _, _ = rebind
    state["changed"] = changed
    before = (d / "good.json").read_text()
    rep = R.migrate("C", d)
    assert "refused" in rep and (d / "good.json").read_text() == before


def test_rebind_refuses_if_scheme2_of_the_commit_differs(rebind):
    R, state, d, *_ = rebind
    state["scheme2_then"] = {"scheme": 2, "code": "other", "inputs": "I"}
    assert "refused" in R.migrate("C", d)


def test_migrated_gate_results_reverify():
    """Every result re-stamped by the migration: its recorded legacy binding is the legacy binding
    of the named commit (recomputed from git), and that commit's scheme-2 binding is the current one
    (only control-plane files changed)."""
    import m3b_rebind_evidence as R

    res = R.RES
    if not res.exists():
        pytest.skip("no gate results in this checkout")
    ext = B.external_binding()
    cur = B.evidence_binding()
    seen = 0
    for p in sorted(res.glob("*.json")):
        r = json.loads(p.read_text())
        m = r.get("binding_migration")
        if m is None:
            continue
        legacy, scheme2_then = R.bindings_at(m["from_commit"], ext)
        assert m["legacy_binding"] == legacy, p.name
        assert r["binding"] == cur == scheme2_then, p.name
        assert all(f in B.CONTROL_PLANE for f in m["files_changed_since"]), p.name
        seen += 1
    assert seen or not any("binding_migration" in p.read_text() for p in res.glob("*.json"))
