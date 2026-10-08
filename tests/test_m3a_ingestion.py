"""M3a leg ingestion (gates G1, G2) on representative validation legs, each in a fresh process:
TOA identity (count, arrival time < 2 ns, uncertainty, frequency, observatory, every flag, -padd),
classified warnings (none unexplained), pinned clock files (every file PINT used comes from the
pinned directory with the manifest hash), and the canonicalisation rules.
"""

from __future__ import annotations

from collections import Counter

import pytest

from ptagwb.legs import (
    apply_indicator_flags,
    canonical_par,
    classify_warning,
    multivalued_mask_flags,
)
from ptagwb.profiles import COMBINED, published_profile
from ptagwb.timfile import TimRecord

pytestmark = [pytest.mark.slow]

LEGS = [
    ("mpta", "J1909-3744"),  # MeerKAT, TCB par
    ("epta_dr2new", "J1744-1134"),  # INCLUDE tree, TIME offsets, -padd, LEAP
    ("ppta_dr3_gh", "J1909-3744"),  # multi-valued -j mask flags, DE436 -> DE440 profile
    ("inpta_dr2", "J1614-2230"),  # valueless flags, DMX_0001 template, 'value uncertainty' lines
    ("ng15", "J1022+1001"),  # PINT-native release
]


@pytest.fixture(scope="module")
def legs():
    from ptagwb.legs import load_legs
    from ptagwb.m3data import leg_files

    try:
        for ds, lab in LEGS:
            leg_files(ds)[lab]
    except (FileNotFoundError, KeyError):
        pytest.skip("M3 data not fetched")
    return {(r.dataset, r.label): r for r in load_legs(LEGS, jobs=5, tag="test")}


@pytest.mark.parametrize("leg", LEGS)
def test_g1_g2(legs, leg):
    r = legs[leg]
    assert r.ok, r.error
    m = r.meta
    assert m["g1_ok"], {k: v for k, v in m.items() if k.startswith("g1")}
    assert m["g1_n_pint"] == m["g1_n_records"]
    assert m["g1_max_dt_ns"] < 2.0
    assert not m["g2_unexplained"], m["warnings"]
    prof = published_profile(r.pta, r.dataset)
    man = prof.manifest()["files"]
    assert m["clock_files"] and all(man[k] == v for k, v in m["clock_files"].items())


def test_multivalued_jump_flags_kept(legs):
    m = legs[("ppta_dr3_gh", "J1909-3744")].meta
    assert m["multivalued_mask_flags"] == ["-j"]
    jumps = [ln for ln in m["par_as_loaded"].splitlines() if ln.startswith("JUMP") and "-j__MEDUSA" in ln]
    assert len(jumps) >= 2 and all(" 1 " in ln or ln.split()[3] == "1" for ln in jumps)
    assert legs[("ppta_dr3_gh", "J1909-3744")].psr.Mmat.shape[1] == len(legs[("ppta_dr3_gh", "J1909-3744")].psr.fitpars)


def test_inpta_value_uncertainty_lines_frozen(legs):
    psr = legs[("inpta_dr2", "J1614-2230")].psr
    assert not any(p.startswith("DMX_") for p in psr.fitpars)


def test_signed_h3_stays_quarantined():
    from ptagwb.legs import load_legs
    from ptagwb.m3data import quarantine

    q = quarantine()
    assert q[("mpta", "J1825-0319")]["class"] == "signed-H3"
    (r,) = load_legs([("mpta", "J1825-0319")], jobs=1, tag="test")
    assert not r.ok and "M2 cannot be negative" in r.error


def test_quarantine_inventory():
    from ptagwb.m3data import quarantine

    q = quarantine()
    classes = Counter(v["class"] for v in q.values())
    assert classes == {"signed-H3": 1, "residual-excess": 4}
    from ptagwb.m3data import load_json_config

    lifted = load_json_config("quarantine.json")["lifted"]
    assert len(lifted) == 10 and {e["class"] for e in lifted} == {"ELL1H-H3H4-NHARMS"}


def test_canonical_par_rules():
    fixes = Counter()
    txt = canonical_par("PSRJ X\nEPHVER 5\nPB 1.5 1\nFB1 1e-20 1\nDMX_0002 1e-4 2e-4\nDMXR1_0002 55000\n"
                        "DMXR2_0002 55001\nCHI2R 1.2 6000\nTRACK -2\nPEPOCH 59000 5.29e-4943\n", fixes)
    assert "UNITS TCB" in txt and "TRACK" not in txt
    assert "FB0" in txt and fixes["par:PB->FB0"] == 1
    assert "DMX_0002 1e-4 0 2e-4" in txt  # tempo2: 'value X' with X not 0/1/2 -> frozen
    assert "CHI2R 1.2 6000" in txt  # bookkeeping keywords untouched
    assert "PEPOCH 59000 0 5.29e-4943" in txt
    assert "DMX_0001 0 0" in txt  # PINT DMX template range, explicitly frozen


def test_indicator_flags():
    recs = [TimRecord("f", 1, "a", 1400.0, "55000.1", 1.0, "pks", [("-j", "A"), ("-j", "B"), ("-f", "x")], 0.0),
            TimRecord("f", 2, "b", 1400.0, "55000.2", 1.0, "pks", [("-j", "A")], 0.0)]
    par = "JUMP -j A 1e-6 1\nJUMP -j B 2e-6 1\nJUMP -f x 0 1\n"
    flags = multivalued_mask_flags(recs, par)
    assert flags == {"-j"}
    new, ptxt = apply_indicator_flags(recs, par, flags, Counter())
    assert new[0].flags == [("-j__A", "1"), ("-j__B", "1"), ("-f", "x")]
    assert "JUMP -j__A 1 1e-6 1" in ptxt and "JUMP -j__B 1 2e-6 1" in ptxt and "JUMP -f x 0 1" in ptxt


def test_profiles_apply():
    txt, ch = published_profile("PPTA").apply_to_par("PSRJ X\nCLK TT(BIPM2020)\nEPHEM DE436\n")
    assert "EPHEM DE440" in txt and any("DE436 -> DE440" in c for c in ch)
    txt, _ = COMBINED.apply_to_par("PSRJ X\nCLOCK TT(BIPM2019)\n")
    assert "CLOCK TT(BIPM2023)" in txt and "EPHEM DE440" in txt


def test_warning_classifier():
    assert classify_warning("UserWarning: PINT does not support 'DILATEFREQ Y'")[0] == "unsupported-tempo2-setting"
    assert classify_warning("something new")[0] == "UNCLASSIFIED!"
