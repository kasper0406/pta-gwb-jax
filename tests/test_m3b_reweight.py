"""M3b N7 reweighting estimators and diagnostics on synthetic targets with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from scipy.signal import lfilter

from ptagwb.reweight import (
    accept_reweighting,
    chain_stability,
    integrated_autocorr_time,
    kish_ess,
    mcse_lnbf_block_bootstrap,
    mcse_lnbf_obm,
    psis_khat,
    raw_bf,
    weighted_quantile,
    weighted_quantile_mcse,
)


def gauss_logw(x, mu, s, ln_c=0.0):
    """ln w for proposal N(0, 1) and unnormalised target c N(mu, s^2): ln BF = ln c."""
    return ln_c + stats.norm.logpdf(x, mu, s) - stats.norm.logpdf(x)


def ar1(rng, n_chains, n, rho):
    """Stationary AR(1) chains with N(0, 1) marginals, shape (n_chains, n)."""
    e = rng.normal(size=(n_chains, n)) * np.sqrt(1 - rho**2)
    e[:, 0] = rng.normal(size=n_chains)
    return lfilter([1.0], [1.0, -rho], e, axis=1)


def test_raw_bf_analytic_and_stable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5000))
    lw = gauss_logw(x, 0.3, 0.8, ln_c=np.log(60.0))
    ob = mcse_lnbf_obm(lw)
    assert abs(ob["lnbf"] - np.log(60.0)) < 4 * ob["mcse"]
    assert raw_bf(lw + 1000.0) == pytest.approx(raw_bf(lw) + 1000.0, abs=1e-9)
    assert raw_bf(np.zeros(10)) == 0.0
    assert kish_ess(np.zeros((2, 50))) == pytest.approx(100.0)


def test_weighted_quantiles_analytic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 10000))
    lw = gauss_logw(x, 0.3, 0.8)
    for p in (0.05, 0.5, 0.95):
        r = weighted_quantile_mcse(x, lw, p)
        truth = 0.3 + 0.8 * stats.norm.ppf(p)
        assert abs(r["q"] - truth) < 4 * r["mcse"], (p, r)
        assert r["bandwidth"] > 0
    # unweighted type-1 inverse CDF
    assert weighted_quantile(np.arange(1, 11), 0.5) == 5.0
    assert weighted_quantile(np.arange(1, 11), 0.51) == 6.0


def test_integrated_autocorr_time_ar1():
    rng = np.random.default_rng(2)
    rho = 0.9
    x = ar1(rng, 4, 20000, rho)
    tau_true = (1 + rho) / (1 - rho)
    assert integrated_autocorr_time(x) == pytest.approx(tau_true, rel=0.15)
    uneven = [x[0], x[1, :15000], x[2, :12000]]
    assert integrated_autocorr_time(uneven) == pytest.approx(tau_true, rel=0.15)
    assert integrated_autocorr_time(rng.normal(size=20000)) == pytest.approx(1.0, abs=0.1)


def test_obm_mcse_covers_ar1_while_iid_undercovers():
    rng = np.random.default_rng(3)
    n_rep, n, rho = 300, 4000, 0.9
    x = ar1(rng, n_rep, n, rho)
    lw = gauss_logw(x, 0.3, 0.8)  # normalised target: ln BF = 0
    est = np.empty(n_rep)
    se_obm = np.empty(n_rep)
    se_iid = np.empty(n_rep)
    for i in range(n_rep):
        ob = mcse_lnbf_obm(lw[i])
        est[i], se_obm[i] = ob["lnbf"], ob["mcse"]
        w = np.exp(lw[i])
        se_iid[i] = w.std(ddof=1) / w.mean() / np.sqrt(n)
    spread = est.std(ddof=1)
    assert np.median(se_obm) / spread == pytest.approx(1.0, abs=0.2)
    cov_obm = np.mean(np.abs(est) <= 1.645 * se_obm)
    cov_iid = np.mean(np.abs(est) <= 1.645 * se_iid)
    assert 0.80 <= cov_obm <= 0.96, cov_obm  # OBM with b ~ 5 tau is slightly anti-conservative
    assert cov_iid < 0.6, cov_iid


def test_quantile_mcse_coverage_ar1():
    rng = np.random.default_rng(4)
    n_rep, n, rho = 200, 4000, 0.8
    x = ar1(rng, n_rep, n, rho)
    lw = gauss_logw(x, 0.3, 0.8)
    hits = []
    for i in range(n_rep):
        r = weighted_quantile_mcse(x[i], lw[i], 0.5)
        hits.append(abs(r["q"] - 0.3) <= 1.645 * r["mcse"])
    assert 0.80 <= np.mean(hits) <= 0.97, np.mean(hits)


def test_block_bootstrap_agrees_with_obm():
    rng = np.random.default_rng(5)
    x = ar1(rng, 4, 8000, 0.9)
    lw = gauss_logw(x, 0.3, 0.8)
    ob = mcse_lnbf_obm(lw)
    boot = mcse_lnbf_block_bootstrap(lw, n_boot=500, seed=1, block_lengths=ob["batch_length"])
    assert boot / ob["mcse"] == pytest.approx(1.0, abs=0.35)
    assert mcse_lnbf_block_bootstrap(lw, n_boot=200, seed=7) == mcse_lnbf_block_bootstrap(
        lw, n_boot=200, seed=7)


def test_khat_heavy_and_light_tails():
    rng = np.random.default_rng(6)
    x = rng.normal(size=20000)
    # target N(0, 2^2) against N(0, 1): weight tail index k = 1 - 1/4 = 0.75
    assert psis_khat(gauss_logw(x, 0.0, 2.0)) >= 0.5
    assert psis_khat(gauss_logw(x, 0.3, 0.8)) < 0.5
    # exact Pareto weights with known shape
    for k in (0.3, 0.8):
        ks = [psis_khat(np.log(rng.pareto(1 / k, size=5000) + 1)) for _ in range(40)]
        assert np.mean(ks) == pytest.approx(k, abs=0.08)
    assert psis_khat(np.zeros(10)) == np.inf  # tail too short: fails closed


def test_stability_detects_inconsistent_chain():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 5000))
    lw = gauss_logw(x, 0.3, 0.8)
    good = chain_stability(lw, {"a": x})
    assert good["pass"] and good["lnbf"]["p"] > 0.01 and good["max_chain_weight_share"] < 0.5
    bad = lw.copy()
    bad[3] += 0.3  # chain 3 sees a different normalisation
    res = chain_stability(bad)
    assert not res["pass"] and res["lnbf"]["p"] < 0.01
    # one chain carrying most of the weight
    dom = lw.copy()
    dom[0] += 3.0
    assert chain_stability(dom)["max_chain_weight_share"] > 0.5
    # inconsistent weighted median in one chain
    xs = x.copy()
    xs[2] += 0.3
    assert chain_stability(lw, {"a": xs})["medians"]["a"]["p"] < 0.01
    # a single chain cannot demonstrate stability
    assert not chain_stability(lw[:1])["pass"]


def test_accept_reweighting_pass_and_fail_closed():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(4, 5000))
    lw = gauss_logw(x, 0.3, 0.8, ln_c=np.log(60.0))
    probs = (0.05, 0.5, 0.95)
    lim = {("a", p): 0.05 for p in probs}
    res = accept_reweighting(lw, {"a": x}, probs, lim, n_boot=300)
    assert res["accepted"], res["reasons"]
    assert res["khat"]["pass"] and res["lnbf"]["mcse_gate"] <= 0.10
    assert abs(res["lnbf"]["value"] - np.log(60.0)) < 4 * res["lnbf"]["mcse_gate"]
    # a reported-only quantity (None) is not gated
    lim2 = dict(lim)
    lim2[("a", 0.05)] = None
    assert accept_reweighting(lw, {"a": x}, probs, lim2, n_boot=100)["accepted"]
    # precision gate failure
    tight = {k: 1e-5 for k in lim}
    r2 = accept_reweighting(lw, {"a": x}, probs, tight, n_boot=100)
    assert not r2["accepted"] and any("MCSE(a" in s for s in r2["reasons"])
    # missing max_our_MCSE entry -> raises
    with pytest.raises(KeyError):
        accept_reweighting(lw, {"a": x}, probs, {("a", 0.5): 0.05}, n_boot=100)
    # heavy tails -> not accepted
    lwh = gauss_logw(x, 0.0, 2.0)
    r3 = accept_reweighting(lwh, {"a": x}, (0.5,), {("a", 0.5): 1.0}, n_boot=100)
    assert not r3["accepted"] and not r3["khat"]["pass"]
    # too few draws per chain to estimate MCSE -> not accepted
    r4 = accept_reweighting(lw[:, :20], {"a": x[:, :20]}, (0.5,), {("a", 0.5): 10.0}, n_boot=50)
    assert not r4["accepted"]


def _ar1(rng, n, rho):
    e = rng.normal(size=n) * np.sqrt(1 - rho**2)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_conditional_lnbf_on_domain_known_answer_and_coverage():
    """ln B_D = ln mean(w I) - ln mean(I) with the paired OBM MCSE on correlated chains: proposal
    N(0,1), target N(0.5,1) (w = exp(0.5 x - 1/8), BF = 1), D = {x > -1}:
    B_D = Phi(1.5) / Phi(1)."""
    from scipy import stats as st

    from ptagwb.reweight import mcse_lnbf_block_bootstrap

    truth = np.log(st.norm.cdf(1.5) / st.norm.cdf(1.0))
    rng = np.random.default_rng(21)
    hits, reps = 0, 120
    for _ in range(reps):
        xs = [_ar1(rng, 4000, 0.8) for _ in range(3)]
        lw = [0.5 * x - 0.125 for x in xs]
        m = [x > -1.0 for x in xs]
        ob = mcse_lnbf_obm(lw, m)
        assert ob["lnbf"] == raw_bf(lw, m)
        hits += abs(ob["lnbf"] - truth) <= 1.96 * ob["mcse"]
    assert 0.85 <= hits / reps <= 0.99
    boot = mcse_lnbf_block_bootstrap(lw, n_boot=400, mask=m)
    assert 0.5 < boot / ob["mcse"] < 2.0
    # no mask == all-True mask, exactly
    assert raw_bf(lw) == raw_bf(lw, [np.ones(4000, bool)] * 3)
    assert mcse_lnbf_obm(lw)["mcse"] == pytest.approx(mcse_lnbf_obm(lw, [np.ones(4000, bool)] * 3)["mcse"], rel=1e-12)
    with pytest.raises(ValueError):
        raw_bf(lw, [np.zeros(4000, bool)] * 3)
    with pytest.raises(ValueError):
        raw_bf(lw, [np.ones(10, bool)] * 3)


def test_accept_reweighting_with_domain():
    rng = np.random.default_rng(22)
    xs = [_ar1(rng, 6000, 0.5) for _ in range(4)]
    lw = [0.3 * x - 0.045 for x in xs]
    m = [x > -1.5 for x in xs]
    r = accept_reweighting(lw, {"a": xs}, (0.5,), {("a", 0.5): 0.05}, mask=m, n_boot=200)
    assert r["n_draws_in_domain"] == sum(int(v.sum()) for v in m)
    assert r["lnbf"]["value"] == raw_bf(lw, m)
    assert r["quantiles"]["a"][0.5]["q"] > np.quantile(np.concatenate(xs), 0.5)  # shelf excluded, tilted
    assert r["accepted"], r["reasons"]


def test_paired_quantile_shift_known_answer_and_coverage():
    """Shift of the median under a tilt w = exp(t x - t^2/2) on N(0,1) draws (target N(t,1)): the
    true shift is t; the paired MCSE covers it on AR(1) chains and is smaller than the endpoint
    MCSE (positive covariance between the endpoints)."""
    from ptagwb.reweight import paired_quantile_shift

    rng = np.random.default_rng(31)
    t, hits, reps, ratio = 0.05, 0, 100, []
    for _ in range(reps):
        xs = [_ar1(rng, 5000, 0.7) for _ in range(2)]
        r = paired_quantile_shift(xs, [t * x - t * t / 2 for x in xs], 0.5)
        hits += abs(r["shift"] - t) <= 1.96 * r["mcse_shift_paired"]
        ratio.append(r["mcse_shift_paired"] / r["endpoint_mcse_weighted"])
    assert 0.85 <= hits / reps <= 0.995
    assert np.median(ratio) < 0.7
