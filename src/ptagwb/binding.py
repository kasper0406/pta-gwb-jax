"""Bind gate results to the exact evaluated configuration (review of 2ee1bb7, item 5).

Every M3b gate result file records ``binding = evidence_binding()``: sha256 of the evaluated
source (``src/ptagwb/*.py``, ``scripts/m3b_*.py``, ``tests/m3b_arbiter.py``), the committed M3b
configs (manifest, runtime pin, acceptance and relevance files), the tempo2 runtime's pin
verification record and every exported input array our model consumes. A consumer of gate
results (the D3 benchmark, the production runner) calls ``require_bound`` and fails closed if any
result is missing, failed, or was produced from a different configuration (a stale artifact).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import REPO_ROOT

EXPORTS = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "export" / "ours_canonical"
RUNTIME_RECORD = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "t2runtime" / "runtime.json"


class StaleEvidenceError(RuntimeError):
    """A gate result is missing, failed, or does not match the current configuration."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _group(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(Path(p).relative_to(REPO_ROOT)).encode())
        h.update(_sha(Path(p)).encode())
    return h.hexdigest()


def evidence_binding(root: Path = REPO_ROOT) -> dict:
    src = sorted((root / "src" / "ptagwb").glob("*.py"))
    scripts = sorted((root / "scripts").glob("m3b_*.py")) + [root / "tests" / "m3b_arbiter.py"]
    configs = sorted((root / "configs" / "m3b").rglob("*.json"))
    exports = sorted(EXPORTS.glob("*.npz"))
    if not exports:
        raise StaleEvidenceError(f"no exported inputs in {EXPORTS}")
    if not RUNTIME_RECORD.exists() or not json.loads(RUNTIME_RECORD.read_text()).get("pin_verified"):
        raise StaleEvidenceError("tempo2 runtime not pin-verified (scripts/m3b_epta_prepare.py)")
    return {"source": _group(src), "scripts": _group(scripts), "configs": _group(configs),
            "runtime": _sha(RUNTIME_RECORD), "exports": _group(exports), "n_exports": len(exports)}


def require_bound(results: dict[str, str], directory: Path, binding: dict | None = None) -> dict:
    """``results``: result file name -> name of its boolean pass field. Raises
    ``StaleEvidenceError`` unless every file exists, its pass field is True, and its recorded
    binding equals the current one. Returns the per-file status."""
    cur = binding if binding is not None else evidence_binding()
    status = {}
    for fname, key in results.items():
        p = Path(directory) / fname
        if not p.exists():
            raise StaleEvidenceError(f"{fname}: missing")
        r = json.loads(p.read_text())
        if r.get(key) is not True:
            raise StaleEvidenceError(f"{fname}: {key} is {r.get(key)!r}")
        b = r.get("binding")
        if b is None:
            raise StaleEvidenceError(f"{fname}: no binding recorded")
        diff = sorted(k for k in set(cur) | set(b) if cur.get(k) != b.get(k))
        if diff:
            raise StaleEvidenceError(f"{fname}: stale (differs in {diff})")
        status[fname] = True
    return status
