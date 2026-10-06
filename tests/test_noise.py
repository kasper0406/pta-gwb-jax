"""White noise: ECORR quantisation, EFAC/EQUAD convention, block operations."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import make_pta

from ptagwb.noise import build_white_noise, load_noise_dict, quantize, selection_masks


def test_quantize_enterprise_rule():
    # bucket opens at its first TOA; join while t - t_start < 1 s; singletons dropped (nmin=2)
    t = np.array([10.2, 0.0, 3.0, 0.5, 1.5, 0.99, 1.0, 10.0])
    got = sorted(sorted(t[b].tolist()) for b in quantize(t))
    assert got == [[0.0, 0.5, 0.99], [1.0, 1.5], [10.0, 10.2]]
    got1 = sorted(sorted(t[b].tolist()) for b in quantize(t, nmin=1))
    assert got1 == [[0.0, 0.5, 0.99], [1.0, 1.5], [3.0], [10.0, 10.2]]
    # not a sliding adjacent-gap rule: a chain of 0.6 s gaps is split at the 1 s horizon
    chain = np.array([0.0, 0.6, 1.2, 1.8])
    assert sorted(sorted(chain[b].tolist()) for b in quantize(chain)) == [[0.0, 0.6], [1.2, 1.8]]


class _P:
    def __init__(self, name, toas, flags, errs):
        self.name, self.toas, self.backend_flags, self.toaerrs = name, toas, flags, errs


def test_ecorr_buckets_per_backend_and_values():
    # two backends observing at the same times -> separate epochs, separate ECORR values
    toas = np.array([100.0, 100.1, 100.0, 100.2, 200.0, 500.0, 500.3])
    flags = np.array(["A_PUPPI", "A_PUPPI", "B_GUPPI", "B_GUPPI", "A_PUPPI", "B_GUPPI", "B_GUPPI"])
    errs = np.full(7, 1e-6)
    nd = {}
    for be, (ef, eq, ec) in {"A_PUPPI": (1.1, -7.0, -6.5), "B_GUPPI": (0.9, -6.5, -6.0)}.items():
        nd[f"X_{be}_efac"], nd[f"X_{be}_log10_t2equad"], nd[f"X_{be}_log10_ecorr"] = ef, eq, ec
    wn = build_white_noise(_P("X", toas, flags, errs), nd)
    assert wn.n_epoch == 3
    assert wn.epoch[4] == -1  # singleton A epoch at 200 s
    assert wn.epoch[0] == wn.epoch[1] != wn.epoch[2]
    assert wn.epoch[2] == wn.epoch[3] and wn.epoch[5] == wn.epoch[6]
    np.testing.assert_allclose(sorted(wn.ecorr_var), sorted([10**-13, 10**-12, 10**-12]))
    # T2 EQUAD convention: EFAC^2 (sigma^2 + EQUAD^2); ECORR outside EFAC
    np.testing.assert_allclose(wn.ndiag[0], 1.1**2 * (1e-12 + 1e-14))
    np.testing.assert_allclose(wn.ndiag[2], 0.9**2 * (1e-12 + 1e-13))


def test_block_ops_match_dense():
    psrs, nd = make_pta(3, seed=3)
    rng = np.random.default_rng(1)
    for p in psrs:
        wn = build_white_noise(p, nd)
        N = wn.dense()
        X = rng.normal(size=(p.ntoa, 4))
        np.testing.assert_allclose(wn.solve(X), np.linalg.solve(N, X), rtol=1e-10, atol=0)
        W = wn.whiten(np.eye(p.ntoa))
        np.testing.assert_allclose(W.T @ W, np.linalg.inv(N), rtol=1e-9, atol=1e-6 * np.abs(np.linalg.inv(N)).max())
        assert abs(wn.logdet() - np.linalg.slogdet(N)[1]) < 1e-9 * abs(wn.logdet())


def test_missing_parameter_fails_loudly():
    psrs, nd = make_pta(1)
    p = psrs[0]
    for suffix in ("efac", "log10_t2equad", "log10_ecorr"):
        bad = {k: v for k, v in nd.items() if k != f"{p.name}_430_PUPPI_{suffix}"}
        with pytest.raises(KeyError, match=suffix):
            build_white_noise(p, bad)
    extra = dict(nd, **{f"{p.name}_S-wide_PUPPI_efac": 1.0})
    with pytest.raises(KeyError, match="not used"):
        build_white_noise(p, extra)


def test_selection_rules():
    flags = np.array(["430_ASP", "Rcvr1_2_GUPPI", "L-wide_PUPPI", "1.5GHz_YUPPI", "foo_bar"])
    assert set(selection_masks(flags, "backend")) == set(flags)
    assert set(selection_masks(flags, "nanograv")) == set(flags[:4])


def test_wn_dict_has_all_ng15_params():
    """The released dict covers EFAC/EQUAD/ECORR for every system it mentions."""
    try:
        nd = load_noise_dict()
    except FileNotFoundError:
        pytest.skip("v1p1_wn_dict.json not fetched")
    assert len(nd) == 697
    systems = {k.rsplit("_efac", 1)[0] for k in nd if k.endswith("_efac")}
    # 217 systems incl. the 2 of J0614-3329 (excluded from the GWB set: 215 systems, 645 params)
    assert len(systems) == 217
    assert len([s for s in systems if not s.startswith("J0614-3329_")]) == 215
    for s in systems:
        assert f"{s}_log10_t2equad" in nd and f"{s}_log10_ecorr" in nd
