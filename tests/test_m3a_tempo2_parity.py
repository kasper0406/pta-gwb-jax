"""Gates G3/G4 (PINT vs tempo2) on one validation leg, EPTA J1744-1134 (INCLUDE tree, TIME offsets,
-padd, LEAP, TCB par): the regression envelope of the achieved agreement, and the pre-registered
strict tolerances (1 ns and 0.01 sigma projected rms; column-space sine 1e-6), which are an open
exit condition (E8, docs/M3A_VALIDATION.md Sec. 4) and therefore xfail(strict=True).
"""

from __future__ import annotations

import pytest
from m3a_oracles import g3_g4, have_tempo2, run_tempo2

pytestmark = [pytest.mark.slow, pytest.mark.oracle]


@pytest.fixture(scope="module")
def j1744():
    if not have_tempo2():
        pytest.skip("tempo2/libstempo oracle env not installed (scripts/setup_tempo2_env.sh)")
    from ptagwb.legs import load_legs
    from ptagwb.m3data import leg_files
    from ptagwb.profiles import published_profile

    try:
        par, tim = leg_files("epta_dr2new")["J1744-1134"]
    except (FileNotFoundError, KeyError):
        pytest.skip("M3 data not fetched")
    (leg,) = load_legs([("epta_dr2new", "J1744-1134")], jobs=1, tag="test-t2")
    t2 = run_tempo2(par, tim, published_profile("EPTA"), "test/epta_dr2new/J1744-1134")
    return g3_g4(leg.psr, t2)


def test_envelope(j1744):
    assert j1744["n_pint"] == j1744["n_tempo2"]
    assert j1744["rms_diff_proj_ns"] < 3.0 and j1744["rms_diff_proj_sigma"] < 0.01
    assert j1744["ncol_pint"] == j1744["ncol_tempo2"] and j1744["g4_max_sin"] < 1e-4


@pytest.mark.xfail(strict=True, reason="E8 open: 1-5 ns PINT/tempo2 engine differences (Sec. 4)")
def test_strict_g3_g4(j1744):
    assert j1744["g3_ok"] and j1744["g4_ok"]
