"""Optimal statistic vs enterprise_extensions ``OptimalStatistic`` (oracle) and the released
Fig. 1(c) maximum-likelihood OS, on the released feathers (enterprise sky positions).

The enterprise PTA is the one of the NG15 Fig. 1(c) notebook: ``model_2a(psrs, noisedict,
tm_marg=False, psd='powerlaw', n_rnfreqs=30, n_gwbfreqs=14, gamma_common=13/3)``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from oracle_helpers import HAVE_ENTERPRISE, to_enterprise_pulsar

from ptagwb.config import RAW_DIR
from ptagwb.data import get_tspan
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.optstat import OptimalStatistic

pytestmark = [pytest.mark.oracle, pytest.mark.slow]
ML_JSON = RAW_DIR / "ng15_gwb_fig1_data" / "extracted" / "figure1_data" / "optstat_ml_gamma4p33.json"
COV_NPY = RAW_DIR / "ng15_gwb_fig1_data" / "extracted" / "figure1_data" / "os_covariance_matix_between_rhos.npy"
# printed by data_release/figure_1/upper_right.ipynb (released notebook output)
RELEASED_ML_OS = {"A2": 6.703650273305144e-30, "sigma": 1.2324322072882867e-30, "snr": 5.439366347018102}


@pytest.fixture(scope="module")
def ml_params():
    if not ML_JSON.exists():
        pytest.skip("figure1 data bundle not fetched")
    return json.loads(ML_JSON.read_text())


@pytest.fixture(scope="module")
def ours_os(feathers, noisedict):
    T = get_tspan(feathers)
    like = PTALikelihood(precompute(feathers, noisedict, T), T, orf="curn")  # feathers carry enterprise pos
    return OptimalStatistic(like)


def _ours_params(os_, named, gamma=13 / 3):
    p = os_.like.params_from_named({**named, "gw_gamma": gamma})
    return p


def test_ml_os_vs_released_notebook(ours_os, ml_params):
    """sigma (depends only on Z_a = F^T C^-1 F) matches the printed release value to 1e-7; A^2
    is 0.36% (0.02 sigma) higher. Current enterprise_extensions on the same (byte-identical)
    feathers agrees with us to 1e-4 sigma (test below), so the residual offset is in the 2023
    notebook environment, not in our OS. Recorded in docs/M2_RESULTS.md."""
    res = ours_os.os(_ours_params(ours_os, ml_params))
    print({k: (res[k], RELEASED_ML_OS[k]) for k in RELEASED_ML_OS})
    assert abs(res["sigma"] / RELEASED_ML_OS["sigma"] - 1) < 1e-6
    assert abs(res["A2"] - RELEASED_ML_OS["A2"]) < 0.05 * RELEASED_ML_OS["sigma"]
    assert abs(res["snr"] - RELEASED_ML_OS["snr"]) < 0.05


def test_pair_covariance_vs_released(ours_os, ml_params):
    if not COV_NPY.exists():
        pytest.skip("released OS covariance not fetched")
    ref = np.load(COV_NPY)
    ours = ours_os.pair_covariance(_ours_params(ours_os, ml_params))
    scale = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    err = np.abs(ours - ref) / scale
    print(f"pair covariance vs released: max {err.max():.2e}, median {np.median(err):.2e} (correlation units)")
    assert np.max(err) < 1e-4


def test_os_vs_enterprise_extensions(feathers, noisedict, ours_os, ml_params):
    if not HAVE_ENTERPRISE:
        pytest.skip("enterprise not installed")
    from enterprise_extensions.frequentist.optimal_statistic import OptimalStatistic as EOS
    from enterprise_extensions.models import model_2a

    epsrs = [to_enterprise_pulsar(p) for p in feathers]
    for p, f in zip(epsrs, feathers):
        p.pos = f.pos_enterprise
    rng = np.random.default_rng(0)
    for gamma in (13 / 3, None):
        pta = model_2a(epsrs, noisedict=noisedict, tm_marg=False, tm_svd=True, psd="powerlaw",
                       n_rnfreqs=30, n_gwbfreqs=14, gamma_common=gamma)
        eos = EOS(epsrs, pta=pta, orf="hd", gamma_common=gamma)
        for trial in range(2):
            named = dict(ml_params)
            if trial:  # perturb the noise point
                named = {k: v + rng.normal(0, 0.05) for k, v in named.items()}
            if gamma is None:
                named["gw_gamma"] = 3.2 + 0.3 * trial
            # white-noise constants first: the WN dictionary also carries 46 IRN entries, which must not
            # override the test point
            xi, rho, sig, A2, s = eos.compute_os(params={**noisedict, **named})
            res = ours_os.os(_ours_params(ours_os, named, gamma if gamma is not None else named["gw_gamma"]))
            assert np.allclose(np.sort(xi), np.sort(ours_os.xi), atol=1e-12)
            # pair order is the same (i < j row-major) in both
            assert np.max(np.abs(res["rho"] - rho) / sig) < 1e-4
            assert np.allclose(res["sig"], sig, rtol=1e-5)
            assert abs(res["A2"] - A2) / s < 1e-4 and abs(res["sigma"] / s - 1) < 1e-5
