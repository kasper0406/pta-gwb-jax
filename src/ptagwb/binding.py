"""Bind gate results to the exact evaluated configuration (reviews of 2ee1bb7 and 50f8d68).

Every M3b gate result file records ``binding = evidence_binding()``: sha256 of the evaluated code
(``src/ptagwb/*.py``, ``scripts/m3b_*.py``, the oracles ``tests/m3b_arbiter.py`` and
``tests/dense_oracle.py``), the committed M3b configs, the exported input arrays our model
consumes (immutable evidence), the installed tempo2 runtime verified file by file against the
committed pin (runtime evidence), the numerical-library versions of the evaluating env and the
package versions of the external oracle envs. A consumer of gate
results (the D3 benchmark, the production runner) calls ``require_bound`` and fails closed if any
result is missing, failed, or was produced from a different configuration (a stale artifact).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import REPO_ROOT

EXPORTS = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "export" / "ours_canonical"
RUNTIME_DIR = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "t2runtime"
RUNTIME_PIN = REPO_ROOT / "configs" / "m3b" / "t2runtime_epta.json"
LIBRARIES = ("numpy", "scipy", "jax", "jaxlib", "numpyro")


class StaleEvidenceError(RuntimeError):
    """A gate result is missing, failed, or does not match the current configuration."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _group(paths, root=REPO_ROOT) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(Path(p).relative_to(root)).encode())
        h.update(_sha(Path(p)).encode())
    return h.hexdigest()


def runtime_file_hashes(rt: Path = RUNTIME_DIR) -> dict:
    """sha256 of every tempo2 runtime data file the evaluator reads (clock/, earth/, observatory/,
    the DE440 ephemeris), from the files themselves (symlinks followed)."""
    out = {}
    for sub in ("clock", "earth", "observatory"):
        for f in sorted((rt / sub).rglob("*")):
            if f.is_file():
                out[f"{sub}/{f.relative_to(rt / sub)}"] = _sha(f)
    out["ephemeris/DE440.1950.2050"] = _sha(rt / "ephemeris" / "DE440.1950.2050")
    return out


def verify_runtime(rt: Path = RUNTIME_DIR, pin: Path = RUNTIME_PIN) -> str:
    """Compare the installed runtime's actual file contents with the committed pin; return the
    hash of the verified file set or raise StaleEvidenceError."""
    if not pin.exists():
        raise StaleEvidenceError(f"no committed runtime pin {pin}")
    want = json.loads(pin.read_text())["files"]
    have = runtime_file_hashes(rt)
    bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    if bad:
        raise StaleEvidenceError(f"installed tempo2 runtime differs from the pin: {bad[:8]} ({len(bad)} files)")
    return hashlib.sha256(json.dumps(have, sort_keys=True).encode()).hexdigest()


def _env_versions(env: Path) -> dict:
    """Versions of the external oracle envs (conda-meta + dist-info names), for the record."""
    out = {}
    for f in sorted((env / "conda-meta").glob("*.json")):
        n = f.stem
        if n.split("-")[0] in ("tempo2", "python", "numpy", "scipy", "scikit", "libblas", "liblapack", "openblas", "suitesparse"):
            out[n] = True
    for d in sorted(env.glob("lib/python3*/site-packages/*.dist-info")):
        if d.name.split("-")[0].lower() in ("libstempo", "numpy", "scipy", "healpy", "ptmcmcsampler", "scikit_sparse"):
            out[d.name] = True
    return {"env": str(env), "packages": sorted(out)}


def evidence_binding(root: Path = REPO_ROOT) -> dict:
    """The configuration a gate result was computed from. Two kinds of evidence are separated:
    ``inputs`` binds the immutable exported arrays our model consumes (and the committed configs);
    ``runtime`` records that the installed tempo2 runtime's actual files equal the committed pin
    (needed by evidence that re-evaluates tempo2 / the fork oracle: T1, fingerprint, G5-PTA)."""
    import importlib.metadata as md
    import os

    src = sorted((root / "src" / "ptagwb").glob("*.py"))
    scripts = sorted((root / "scripts").glob("m3b_*.py"))
    oracles = [root / "tests" / "m3b_arbiter.py", root / "tests" / "dense_oracle.py"]
    configs = sorted((root / "configs" / "m3b").rglob("*.json"))
    exports = sorted(EXPORTS.glob("*.npz"))
    if not exports:
        raise StaleEvidenceError(f"no exported inputs in {EXPORTS}")
    home = Path.home() / ".local" / "opt"
    envs = {k: _env_versions(Path(os.environ.get(v, home / d)))
            for k, v, d in (("fork_env", "EF_ENV", "epta-fork-env"), ("tempo2_env", "T2_ENV", "tempo2-env"))}
    return {"code": {"source": _group(src, root), "scripts": _group(scripts, root), "oracles": _group(oracles, root)},
            "configs": _group(configs, root),
            "inputs": {"exports": _group(exports, root), "n_exports": len(exports)},
            "runtime": {"verified_files": verify_runtime(), "pin": _sha(RUNTIME_PIN)},
            "libraries": {n: md.version(n) for n in LIBRARIES},
            "oracle_envs": hashlib.sha256(json.dumps(envs, sort_keys=True).encode()).hexdigest()}


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
