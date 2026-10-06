"""M0 sanity tests: environment, package hygiene, data manifest."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLE_MODULES = {"enterprise", "enterprise_extensions", "discovery"}


def test_package_never_imports_oracles():
    """src/ptagwb must not depend on the reference implementations."""
    for py in (ROOT / "src" / "ptagwb").rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                assert n.split(".")[0] not in ORACLE_MODULES, f"{py}: imports oracle {n}"


def test_jax_float64_cholesky():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    a = jax.random.normal(jax.random.PRNGKey(0), (256, 256), dtype=jnp.float64)
    m = a @ a.T + 256 * jnp.eye(256)
    L = jnp.linalg.cholesky(m)
    assert L.dtype == jnp.float64
    assert float(jnp.max(jnp.abs(L @ L.T - m)) / jnp.max(jnp.abs(m))) < 1e-13


def test_jax_sees_gpu():
    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU visible (CPU-only machine)")
    assert jax.default_backend() == "gpu"


def test_manifest_well_formed():
    man = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    ds = man["datasets"]
    assert ds["ng15_v2.1.0"]["status"] == "fetched"
    for name, e in ds.items():
        assert e["status"] in {"fetched", "not_fetched"}, name
        if e["status"] == "fetched":
            for f in e["files"]:
                assert len(f["sha256"]) == 64
                assert f["checksum_verified"] in (True, None), (name, f["name"])
        else:
            assert e.get("reason"), name


def test_raw_files_match_manifest():
    """If raw data are present locally, their sizes match the manifest (cheap check)."""
    man = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    checked = 0
    for e in man["datasets"].values():
        for f in e.get("files", []):
            p = ROOT / f["path"]
            if p.exists():
                assert p.stat().st_size == f["size"], f["path"]
                checked += 1
    if checked == 0:
        pytest.skip("data/raw not populated; run scripts/fetch_data.py")
