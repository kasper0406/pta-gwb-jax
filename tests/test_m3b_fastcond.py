"""``ptagwb.fastcond`` (numpy evaluator of the conditional diagnostics and T2) equals
``EPTAModel.logL`` (CURN and HD, incl. the dip and the one-pulsar swap), and runtime binding."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ptagwb.config import REPO_ROOT


@pytest.fixture(scope="module")
def data():
    from ptagwb import epta

    try:
        man, psrs = epta.load_manifest(), epta.load_pulsars()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return man, psrs


@pytest.mark.slow
def test_fastcond_equals_model(data):
    from ptagwb import epta
    from ptagwb.fastcond import FastEPTA

    man, psrs = data
    names, X, b = epta.load_reference("crn_pl", man)
    for orf in ("crn", "hd"):
        M = epta.EPTAModel(psrs, man, orf)
        F = FastEPTA(M)
        for r in (b, len(X) - 1):
            x = X[r, :67].copy()
            assert abs(F.logL(x) - float(M.logL(x))) < 1e-8
            x[M.t0_index] -= 3.0  # dip-onset change
            x[F.ix["J1713+0747_J1713+0747_dmexp_1_log10_Amp"]] -= 0.5
            assert abs(F.logL(x) - float(M.logL(x))) < 1e-8
        if orf == "hd":
            x = X[b, :67].copy()
            parts = [F.hd_reduce(a, x) for a in range(F.P)]
            cache = F.hd_prepare(parts, x)
            a = F.names.index("J0900-3144")
            x[F.ix["J0900-3144_red_noise_log10_A"]] -= 2.0
            assert abs(F.hd_swap(cache, a, F.hd_reduce(a, x)) - float(M.logL(x))) < 1e-8


def test_runtime_verification_detects_changed_files(tmp_path):
    from ptagwb.binding import StaleEvidenceError, runtime_file_hashes, verify_runtime

    rt = tmp_path / "rt"
    for d in ("clock", "earth", "observatory", "ephemeris"):
        (rt / d).mkdir(parents=True)
    (rt / "clock" / "a.clk").write_text("1 0\n")
    (rt / "earth" / "eop").write_text("x\n")
    (rt / "observatory" / "obs.dat").write_text("y\n")
    (rt / "ephemeris" / "DE440.1950.2050").write_text("z\n")
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"files": runtime_file_hashes(rt)}))
    assert verify_runtime(rt, pin)
    (rt / "clock" / "a.clk").write_text("1 1e-9\n")  # a changed clock file
    with pytest.raises(StaleEvidenceError):
        verify_runtime(rt, pin)
    (rt / "clock" / "a.clk").write_text("1 0\n")
    (rt / "clock" / "b.clk").write_text("extra\n")  # an added file
    with pytest.raises(StaleEvidenceError):
        verify_runtime(rt, pin)
    with pytest.raises(StaleEvidenceError):
        verify_runtime(rt, tmp_path / "absent.json")


def test_installed_runtime_matches_pin():
    from ptagwb.binding import RUNTIME_DIR, verify_runtime

    if not RUNTIME_DIR.exists():
        pytest.skip("runtime not prepared")
    assert verify_runtime()
    assert (REPO_ROOT / "configs" / "m3b" / "t2runtime_epta.json").exists()
