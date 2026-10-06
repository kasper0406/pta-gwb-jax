"""Bayes-factor estimators on an analytic toy: two Gaussian likelihoods with a common uniform
box prior, exact posterior draws (truncated normals) and closed-form evidences."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ptagwb.evidence import bridge, reverse_reweight, reweight

LO, HI = np.array([-4.0, -3.0, 0.0]), np.array([4.0, 5.0, 2.0])


def _model(mu, sd):
    def logL(x):
        return np.sum(stats.norm.logpdf(x, mu, sd), axis=-1)

    def lnZ():
        return float(np.sum(np.log(stats.norm.cdf(HI, mu, sd) - stats.norm.cdf(LO, mu, sd)) - np.log(HI - LO)))

    def draws(n, rng, chains=4, rho=0.0):
        """(chains, n, D) exact posterior draws; rho > 0 adds AR(1) autocorrelation via a Gaussian copula."""
        u = rng.normal(size=(chains, n, len(mu)))
        if rho:
            for t in range(1, n):
                u[:, t] = rho * u[:, t - 1] + np.sqrt(1 - rho**2) * u[:, t]
        a, b = (LO - mu) / sd, (HI - mu) / sd
        return stats.truncnorm.ppf(stats.norm.cdf(u), a, b, loc=mu, scale=sd)

    return logL, lnZ, draws


M1 = _model(np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.8, 0.4]))
M2 = _model(np.array([0.3, 0.8, 1.1]), np.array([0.8, 0.7, 0.3]))


def test_estimators_recover_analytic_bf():
    rng = np.random.default_rng(0)
    truth = M2[1]() - M1[1]()
    x1 = M1[2](3000, rng, rho=0.7)
    x2 = M2[2](3000, rng, rho=0.7)
    l1 = M2[0](x1) - M1[0](x1)  # (chains, n)
    l2 = M2[0](x2) - M1[0](x2)
    for res in (reweight(l1, n_boot=400), reverse_reweight(l2, n_boot=400), bridge(l1, l2, n_boot=100)):
        assert abs(res["ln_bf"] - truth) < 4 * res["ln_bf_sd"] + 1e-3, (res, truth)
        assert res["ln_bf_sd"] < 0.2
    # autocorrelation-aware: block length > 1 for AR(0.7) draws
    assert reweight(l1, n_boot=50)["block"] > 1


def test_bootstrap_error_is_calibrated():
    """Spread of the reweighting estimate over independent replicate chains ~ bootstrap sd."""
    rng = np.random.default_rng(1)
    truth = M2[1]() - M1[1]()
    est, sds = [], []
    for _ in range(40):
        x1 = M1[2](800, rng, chains=2, rho=0.5)
        r = reweight(M2[0](x1) - M1[0](x1), n_boot=200)
        est.append(r["ln_bf"])
        sds.append(r["ln_bf_sd"])
    assert 0.6 < np.std(est) / np.mean(sds) < 1.6
    assert abs(np.mean(est) - truth) < 3 * np.std(est) / np.sqrt(len(est)) + 0.01


def test_self_bf_is_one():
    rng = np.random.default_rng(2)
    x = M1[2](1000, rng)
    l = M1[0](x) - M1[0](x)
    for res in (reweight(l, n_boot=50), reverse_reweight(l, n_boot=50), bridge(l, l, n_boot=20)):
        assert res["ln_bf"] == 0.0 and res["ln_bf_sd"] == 0.0
