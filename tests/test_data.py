"""Pulsar container, flag resolution, cache I/O and (if ingested) the NG15 selection."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import make_pta

from ptagwb.data import (
    CACHE_DIR,
    Pulsar,
    cache_path,
    find_par_tim,
    get_tspan,
    resolve_backend_flags,
)


def test_backend_flag_precedence():
    flags = {
        "fe": np.array(["430", "L-wide", "L-wide", ""]),
        "be": np.array(["ASP", "PUPPI", "PUPPI", "GUPPI"]),
        "f": np.array(["430_ASP", "", "Lw_PUPPI", ""]),
        "group": np.array(["", "", "grp", ""]),
    }
    out = resolve_backend_flags(flags, 4)
    assert list(out) == ["430_ASP", "L-wide_PUPPI", "grp", ""]


def test_npz_roundtrip(tmp_path):
    psrs, _ = make_pta(1)
    p = psrs[0]
    p.flags = {"f": p.backend_flags.copy()}
    p.meta = {"input_hash": "abc"}
    p.save(tmp_path / "x.npz")
    q = Pulsar.load(tmp_path / "x.npz")
    for k in ("toas", "residuals", "toaerrs", "Mmat", "pos", "backend_flags"):
        np.testing.assert_array_equal(getattr(q, k), getattr(p, k))
    assert q.name == p.name and q.fitpars == p.fitpars and q.meta == p.meta
    np.testing.assert_array_equal(q.flags["f"], p.flags["f"])


def test_release_files():
    try:
        base = find_par_tim()
        split = find_par_tim(split_only=True)
    except FileNotFoundError:
        pytest.skip("NG15 release not fetched")
    assert len(base) == 68 and "J0614-3329" in base
    assert sorted(split) == sorted(
        ["B1937+21ao", "B1937+21gbt", "J1600-3053gbt", "J1643-1224gbt", "J1713+0747ao",
         "J1713+0747gbt", "J1903+0327ao", "J1909-3744gbt"]
    )


def test_cached_gwb_selection():
    """If the default cache is populated: 67 pulsars, Tspan equals the collaboration's value."""
    names = sorted(n for n in find_par_tim())
    if not all(cache_path(n).exists() for n in names):
        pytest.skip(f"pulsar cache not populated under {CACHE_DIR}; run scripts/ingest.py")
    from ptagwb.data import load_pulsars

    psrs = load_pulsars(verbose=False)
    assert len(psrs) == 67
    # 505861299.1401644 s is hard-coded in the collaboration's figure code
    assert abs(get_tspan(psrs) - 505861299.1401644) < 1e-3
    for p in psrs:
        assert np.all(np.diff(p.toas) >= 0)
        assert p.Mmat.shape == (p.ntoa, len(p.fitpars))
        assert abs(np.linalg.norm(p.pos) - 1) < 1e-12
