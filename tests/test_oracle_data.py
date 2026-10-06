"""Oracle checks of the data layer: our PINT 1.1.7 ingestion vs the released discovery /
enterprise feathers (written by the collaboration's enterprise+PINT front end), enterprise's
own PintPulsar, and the white-noise dictionary vs the production chains."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest
from oracle_helpers import (
    HAVE_DISCOVERY,
    HAVE_ENTERPRISE,
    compare_frontend,
    discovery_data_dir,
    to_discovery_pulsar,
)

from ptagwb.noise import build_white_noise, quantize, selection_masks

pytestmark = pytest.mark.oracle
B_NAMES = ("B1855+09", "B1937+21", "B1953+29")
# Known PINT-version differences (docs/M1_VALIDATION.md): J1713+0747 (DDK KIN/KOM partials and
# two TOAs on DMX-bin edges); ELL1 EPS1/EPS2 partials differ at the 1e-5..1e-3 level.
KNOWN_FRONTEND = ("J1713+0747",)


@pytest.mark.parametrize("model", ["m2a", "m3a"])
def test_wn_dict_matches_production_chains(noisedict, model):
    """All 645 fixed WN values of the released CURN (m2a) / HD (m3a) production runs equal
    v1p1_wn_dict.json bit for bit (parsed from the enterprise runtime info in the chain)."""
    import pyarrow.feather as pf

    try:
        meta = json.loads(pf.read_table(discovery_data_dir() / f"NG15yr-{model}-chain.feather").schema.metadata[b"json"])
    except (StopIteration, FileNotFoundError):
        pytest.skip("discovery chains not fetched")
    consts = {}
    for line in meta["runtime_info"][0]:
        m = re.match(r"(\S+):Constant=(\S+)", line.strip())
        if m:
            consts[m.group(1)] = float(m.group(2))
    assert len(consts) == 645
    assert all(noisedict[k] == v for k, v in consts.items())
    assert {k for k in noisedict if not k.endswith(("_red_noise_log10_A", "_red_noise_gamma"))
            and not k.startswith("J0614-3329_")} == set(consts)


def test_frontend_vs_feathers(ours, feathers, noisedict):
    refs = {p.name: p for p in feathers}
    assert [p.name for p in ours] == sorted(refs)
    rows = [compare_frontend(p, refs[p.name], noisedict) for p in ours]
    # the optional enterprise-convention positions reproduce the released ones
    for p in ours:
        assert np.linalg.norm(p.pos_enterprise - refs[p.name].pos) < 1e-12, p.name
    for r in rows:
        assert r["ntoa"] == r["ntoa_ref"], r["name"]
        assert r["max_dtoaerr_s"] == 0.0, r["name"]
        assert r["backend_flags_equal"], r["name"]
        assert r["ecorr_buckets_equal"], r["name"]
        assert r["max_dtoa_s"] <= 1e-6, r["name"]  # float64 ulp of ~4.6e9 s is 9.5e-7 s
        assert r["max_dfreq_mhz"] <= 1e-9, r["name"]
        assert r["ncol"] == r["ncol_ref"], r["name"]
        assert r["rms_dres_perp_s"] <= 3e-9, r
        if r["name"] in KNOWN_FRONTEND:
            continue
        assert r["max_dres_s"] <= 1e-8, r
        assert r["M_span_sin"] <= 1e-3, r
        if r["name"] not in B_NAMES:
            assert r["pos_angle_rad"] <= 1e-6, r
        else:  # enterprise converts B-name ecliptic coords with the B1950 equinox (pyephem)
            assert 5e-3 < r["pos_angle_rad"] < 2e-2, r


def test_ecorr_buckets_vs_enterprise_quantization(feathers, noisedict):
    if not HAVE_ENTERPRISE:
        pytest.skip("enterprise not installed")
    from enterprise.signals.utils import create_quantization_matrix

    for p in feathers:
        wn = build_white_noise(p, noisedict)
        ours = sorted(tuple(np.flatnonzero(wn.epoch == e)) for e in range(wn.n_epoch))
        theirs = []
        for mask in selection_masks(p.backend_flags, "backend").values():
            idx = np.flatnonzero(mask)
            U, _ = create_quantization_matrix(p.toas[idx], dt=1, nmin=2)
            theirs += [tuple(idx[np.flatnonzero(U[:, j])]) for j in range(U.shape[1])]
        assert ours == sorted(theirs), p.name
        # and our quantize() reproduces enterprise on the raw TOAs, backend by backend
        for mask in selection_masks(p.backend_flags, "backend").values():
            t = p.toas[mask]
            U, _ = create_quantization_matrix(t, dt=1, nmin=2)
            assert sorted(tuple(b) for b in map(sorted, quantize(t))) == sorted(
                tuple(np.flatnonzero(U[:, j])) for j in range(U.shape[1])
            )


def test_ecorr_buckets_vs_discovery(feathers, noisedict):
    if not HAVE_DISCOVERY:
        pytest.skip("discovery not installed")
    import discovery as ds

    n_single = 0
    for p in feathers:
        dp = to_discovery_pulsar(p, noisedict)
        wn = build_white_noise(p, noisedict)
        ours = sorted(tuple(np.flatnonzero(wn.epoch == e)) for e in range(wn.n_epoch))
        F = ds.makegp_ecorr(dp, dp.noisedict, enterprise=True).F
        theirs = sorted(tuple(np.flatnonzero(F[:, j])) for j in range(F.shape[1]))
        assert ours == theirs, p.name
        # discovery's default (enterprise=False) also keeps singleton epochs
        F1 = ds.makegp_ecorr(dp, dp.noisedict).F
        n_single += F1.shape[1] - F.shape[1]
    assert n_single > 0  # the two conventions really differ on NG15 (logL shift ~0.02)


@pytest.mark.slow
def test_vs_enterprise_pintpulsar(ours):
    """Our export == enterprise's PintPulsar built from the same PINT objects (exact)."""
    if not HAVE_ENTERPRISE:
        pytest.skip("enterprise not installed")
    import warnings

    from enterprise.pulsar import PintPulsar
    from pint.models import get_model_and_toas

    from ptagwb.data import clock_dir, find_par_tim

    warnings.filterwarnings("ignore")
    os.environ["PINT_CLOCK_OVERRIDE"] = str(clock_dir("release"))
    names = [p.name for p in ours] if os.environ.get("PTAGWB_ORACLE_FULL") else ["J0023+0923", "J1012+5307", "B1855+09"]
    omap = {p.name: p for p in ours}
    pairs = find_par_tim()
    for name in names:
        model, toas = get_model_and_toas(str(pairs[name][0]), str(pairs[name][1]), planets=True)
        ep = PintPulsar(toas, model, planets=True)
        o = omap[name]
        np.testing.assert_array_equal(o.toas, ep.toas)
        np.testing.assert_array_equal(o.stoas, ep.stoas)
        np.testing.assert_array_equal(o.residuals, ep.residuals)
        np.testing.assert_array_equal(o.toaerrs, ep.toaerrs)
        np.testing.assert_array_equal(o.freqs, ep.freqs)
        # same columns; PINT's free-parameter order (JUMPs) can differ between model instances
        assert sorted(o.fitpars) == sorted(ep.fitpars)
        col = {p: i for i, p in enumerate(ep.fitpars)}
        np.testing.assert_array_equal(o.Mmat, ep.Mmat[:, [col[p] for p in o.fitpars]])
        np.testing.assert_array_equal(o.backend_flags, ep.backend_flags.astype("U"))
        ang = np.arccos(np.clip(o.pos @ ep.pos, -1, 1))
        assert ang < (2e-2 if name.startswith("B") else 1e-6), (name, ang)
        np.testing.assert_allclose(o.pos_enterprise, ep.pos, rtol=0, atol=1e-15)
