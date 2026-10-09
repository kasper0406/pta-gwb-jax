"""M3b acceptance logic: Sec. 6.1 classifier, Sec. 5.3 occupancy / transport gate, Sec. 6.6 D9."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import stats

from ptagwb.acceptance import (
    D9_CLASS,
    D9_TEMPLATE,
    FAIL,
    INCONCLUSIVE,
    PASS,
    RESOLVED,
    UNRESOLVED,
    Occupancy,
    aggregate,
    classify,
    classify_e6,
    classify_from_frozen,
    d9_eligibility,
    bf_domain_correction,
    conditional_quantile,
    d9_exclusions,
    descriptive_envelope,
    domain_indicator,
    decidable,
    envelope_quantile_bounds,
    event_interval_gate,
    event_interval_indicators,
    event_intervals,
    max_our_mcse,
    occupancy,
    shelf_peak_indicators,
    shelf_peak_regions,
    support_class,
    transport_gate,
)
from ptagwb.diagnostics import GateInputError
from ptagwb.reweight import Z90


def markov_chains(rng, n_chains, n, p_in, p_out, start=None):
    """Two-state Markov chains (True = in region R): P(enter) = p_in, P(exit) = p_out per draw.
    Stationary start unless ``start`` is given."""
    pi = p_in / (p_in + p_out)
    s = rng.random(n_chains) < pi if start is None else np.full(n_chains, bool(start))
    u = rng.random((n, n_chains))
    out = np.empty((n, n_chains), bool)
    for t in range(n):
        out[t] = s
        s = np.where(s, u[t] >= p_out, u[t] < p_in)
    return list(out.T)


def occ(p_lo, p_hi, *, entries=50, exits=50, visiting=4, sojourn=0.1, mcse=0.005,
        status=RESOLVED, case="estimable"):
    """Hand-built occupancy for decision-table tests."""
    return Occupancy(p_hat=0.5 * (p_lo + p_hi), p_lo=p_lo, p_hi=p_hi, status=status, case=case,
                     n_entries=entries, n_exits=exits, n_draws=40000, n_chains=4,
                     n_chains_visiting=visiting, tau=10.0, n_eff=4000.0, mcse=mcse,
                     longest_sojourn_frac=sojourn)


UNRES = occ(0.0, 1.0, entries=0, exits=0, visiting=0, sojourn=None, mcse=None,
            status=UNRESOLVED, case="zero-visit")


# ---------------------------------------------------------------- Sec. 6.1 / 6.2


def test_classify_boundaries():
    m = Z90 * 0.1
    assert classify(0.0, 0.1, 0.0, m).status == "EQUIVALENT"  # I_D = [-m, m] exactly
    assert classify(0.0, 0.1 + 1e-9, 0.0, m).status == INCONCLUSIVE
    assert classify(2 * m + 1e-9, 0.1, 0.0, m).status == "INCOMPATIBLE"  # lower end just > m
    assert classify(2 * m, 0.1, 0.0, m).status == INCONCLUSIVE  # touches m: not entirely outside
    assert classify(-2 * m - 1e-9, 0.1, 0.0, m).status == "INCOMPATIBLE"
    c = classify(0.0, 0.03, 0.04, 0.1)
    assert c.se_d == pytest.approx(0.05)
    # reference-limited iff 1.645 MCSE_ref >= m
    r = classify(0.0, 0.0, 0.1, m)
    assert r.status == "EQUIVALENT"  # exactly at the edge of I_D, still within
    r = classify(0.0, 0.01, 0.1, m)
    assert r.status == INCONCLUSIVE and r.limited == "reference-limited"
    o = classify(0.0, 0.1, 0.09, m)
    assert o.status == INCONCLUSIVE and o.limited == "ours-limited"
    assert classify(0.0, np.inf, 0.0, 0.1).status == INCONCLUSIVE


def test_max_our_mcse_and_decidable():
    # E-1 q50: sqrt((0.027/1.645)^2/4 - 0.0046^2)
    assert max_our_mcse(0.027, 0.0046) == pytest.approx(
        np.sqrt((0.027 / Z90) ** 2 / 4 - 0.0046**2))
    # E-1 q05: argument negative -> floor 0.2 m / 1.645
    assert max_our_mcse(0.046, 0.0368) == pytest.approx(0.2 * 0.046 / Z90)
    # positive argument below the floor -> floor
    m = 0.1
    ref = np.sqrt((m / Z90) ** 2 / 4 - (0.1 * m / Z90) ** 2)
    assert max_our_mcse(m, ref) == pytest.approx(0.2 * m / Z90)
    assert decidable(0.027, 0.0046) and not decidable(0.046, 0.0368)
    assert not decidable(Z90 * 0.01, 0.01)


def test_e6():
    t = np.log(60.0)
    assert classify_e6(t, 0.05).status == "EQUIVALENT"
    assert classify_e6(t, 0.11).status == INCONCLUSIVE
    assert classify_e6(t + 1.0, 0.05).status == "INCOMPATIBLE"
    assert classify_e6(t + 0.3, 0.05).status == INCONCLUSIVE
    assert classify_e6(t + 0.3 - Z90 * 0.1, 0.1).status == "EQUIVALENT"


def test_frozen_entry_fails_closed():
    assert classify_from_frozen({"q_ref": 1.0, "mcse_ref": 0.0, "m": 0.1}, 1.0, 0.01).status == \
        "EQUIVALENT"
    with pytest.raises(GateInputError):
        classify_from_frozen({"q_ref": 1.0, "m": 0.1}, 1.0, 0.01)
    good = {"name": "S_J0437", "type": "shelf", "boundaries": [-18, -17],
            "params": {"crn_pl": "J0437_red_noise_log10_A", "hd_pl": "J0437_red_noise_log10_A"},
            "reference_cases": {"crn_pl": "zero-visit", "hd_pl": "few-event"}}
    acc_file = {"d9": {"exclusions": [good], "models": ["crn_pl", "hd_pl"], "reference": {}}}
    assert d9_exclusions(acc_file) == [good]
    bad = dict(good, params={"crn_pl": "x"})  # a model missing from the parameter mapping
    with pytest.raises(GateInputError):
        d9_exclusions({"d9": {**acc_file["d9"], "exclusions": [bad]}})
    with pytest.raises(GateInputError):
        d9_exclusions({"d9": {**acc_file["d9"], "exclusions": [dict(good, type="box")]}})
    with pytest.raises(GateInputError):
        d9_exclusions({})


# ---------------------------------------------------------------- Sec. 5.3 occupancy


def test_occupancy_estimable_wilson_covers_truth():
    rng = np.random.default_rng(0)
    p_in, p_out = 0.02, 0.05
    truth = p_in / (p_in + p_out)
    tau_true = (2 - p_in - p_out) / (p_in + p_out)
    hits, taus = [], []
    for _ in range(60):
        o = occupancy(markov_chains(rng, 4, 5000, p_in, p_out))
        assert o.case == "estimable" and o.status == RESOLVED
        hits.append(o.p_lo <= truth <= o.p_hi)
        taus.append(o.tau)
    assert np.mean(hits) >= 0.88
    assert np.median(taus) == pytest.approx(tau_true, rel=0.2)
    assert o.n_chains_visiting == 4 and 0 < o.longest_sojourn_frac < 0.5
    assert o.mcse == pytest.approx(np.sqrt(o.p_hat * (1 - o.p_hat) / o.n_eff))
    assert support_class(o) == "material"


def test_occupancy_few_zero_all_visit():
    few = [np.r_[np.zeros(500, bool), np.ones(20, bool), np.zeros(480, bool)]] * 3
    o = occupancy(few)
    assert o.case == "few-event" and o.status == UNRESOLVED and o.n_entries == 3
    assert support_class(o) == UNRESOLVED
    z = occupancy([np.zeros(1000, bool)] * 4)
    assert z.case == "zero-visit" and z.status == UNRESOLVED and z.p_hat == 0.0
    a = occupancy([np.ones(1000, bool)] * 4)
    assert a.case == "all-visit" and a.status == UNRESOLVED and a.p_hi == 1.0
    assert occupancy([np.zeros(1000, bool)] * 4).status == a.status  # follows the complement


def test_review_counterexample_never_passes():
    """Entry 1e-5 / exit 9e-5 per draw (10 % shelf mass), 1000 draws: UNRESOLVED -> INCONCLUSIVE."""
    rng = np.random.default_rng(1)
    statuses = set()
    n_zero = 0
    for _ in range(100):
        ref_s = markov_chains(rng, 1, 1000, 1e-5, 9e-5)
        our_s = markov_chains(rng, 4, 1000, 1e-5, 9e-5)
        r_s, o_s = occupancy(ref_s), occupancy(our_s)
        r_p, o_p = occupancy([~c for c in ref_s]), occupancy([~c for c in our_s])
        assert r_s.status == UNRESOLVED and o_s.status == UNRESOLVED
        n_zero += r_s.case == "zero-visit"
        statuses.add(transport_gate((r_s, r_p), (o_s, o_p)).status)
    assert statuses == {INCONCLUSIVE}
    assert n_zero > 70
    # the zero-visit realisation explicitly
    zero = [np.zeros(1000, bool)]
    it = transport_gate((occupancy(zero), occupancy([~zero[0]])),
                        (occupancy(zero * 4), occupancy([~zero[0]] * 4)))
    assert it.status == INCONCLUSIVE and it.unresolved


# ---------------------------------------------------------------- Sec. 5.3 decision table


def test_precedence_unresolved_beats_fail_worthy_numbers():
    ref = (occ(0.3, 0.4), occ(0.6, 0.7))
    # ours: one region unresolved, other resolved but disjoint with no transport (would be FAIL)
    ours = (UNRES, occ(0.01, 0.02, entries=0, exits=0, visiting=1, sojourn=1.0, mcse=0.5))
    assert transport_gate(ref, ours).status == INCONCLUSIVE
    assert transport_gate((UNRES, occ(0.6, 0.7)), (occ(0.3, 0.4), occ(0.6, 0.7))).status == \
        INCONCLUSIVE


def test_table_both_material():
    ref = (occ(0.3, 0.4), occ(0.6, 0.7))
    assert transport_gate(ref, (occ(0.32, 0.38), occ(0.62, 0.68))).status == PASS
    assert transport_gate(ref, (occ(0.32, 0.38, visiting=1), occ(0.62, 0.68))).status == FAIL
    assert transport_gate(ref, (occ(0.32, 0.38, entries=9), occ(0.62, 0.68))).status == FAIL
    assert transport_gate(ref, (occ(0.32, 0.38, sojourn=0.6), occ(0.62, 0.68))).status == FAIL
    assert transport_gate(ref, (occ(0.32, 0.38, mcse=0.011), occ(0.62, 0.68))).status == FAIL
    assert transport_gate(ref, (occ(0.5, 0.55), occ(0.45, 0.5))).status == FAIL  # disjoint


def test_table_one_region_absent():
    ref = (occ(0.0, 0.005), occ(0.99, 1.0))
    assert support_class(ref[0]) == "absent"
    assert transport_gate(ref, (occ(0.0, 0.015, entries=0, exits=0), occ(0.98, 1.0))).status \
        == PASS
    assert transport_gate(ref, (occ(0.006, 0.019), occ(0.98, 0.99))).status == FAIL  # disjoint
    assert transport_gate(ref, (occ(0.02, 0.03), occ(0.97, 0.98))).status == FAIL  # p_lo >= 0.02
    # overlap but p_hi,ours >= 0.02 > p_lo: neither row -> INCONCLUSIVE (fail-safe choice)
    assert transport_gate(ref, (occ(0.001, 0.03), occ(0.97, 0.999))).status == INCONCLUSIVE


def test_table_ambiguous():
    ref = (occ(0.004, 0.03), occ(0.97, 0.996))
    assert support_class(ref[0]) == "ambiguous"
    assert transport_gate(ref, (occ(0.005, 0.02), occ(0.98, 0.995))).status == PASS
    assert transport_gate(ref, (occ(0.005, 0.02, visiting=1), occ(0.98, 0.995))).status == \
        INCONCLUSIVE
    assert transport_gate(ref, (occ(0.05, 0.06), occ(0.94, 0.95))).status == INCONCLUSIVE


def test_aggregate():
    assert aggregate([PASS, PASS]) == PASS
    assert aggregate([PASS, INCONCLUSIVE]) == INCONCLUSIVE
    assert aggregate([INCONCLUSIVE, FAIL, PASS]) == FAIL
    with pytest.raises(GateInputError):
        aggregate([])
    with pytest.raises(GateInputError):
        aggregate(["MAYBE"])


def test_shelf_peak_and_event_intervals():
    assert shelf_peak_regions(-20.0, -11.0) == {"S": (-20.0, -19.0), "P": (-19.0, -11.0)}
    ind = shelf_peak_indicators(np.array([-19.5, -19.0, -18.9]), -20.0)
    assert ind["S"][0].tolist() == [True, True, False]
    toas = [100.0, 110.0, 120.0, 130.0]
    rng = np.random.default_rng(2)
    ref = np.r_[rng.uniform(110, 120, 990), rng.uniform(125, 130, 5), [99.0] * 5]
    iv = event_intervals(toas, (100.0, 140.0), ref)
    assert iv == [(110.0, 120.0)]
    ours = [rng.uniform(100, 140, 2000)]
    ii = event_interval_indicators(ours, iv, (100.0, 140.0))
    assert set(ii) == {"I0", "rest"}
    assert np.all(ii["I0"][0] ^ ii["rest"][0])
    assert np.all(ii["I0"][0] == ((ours[0] >= 110) & (ours[0] < 120)))


def test_event_interval_gate():
    rng = np.random.default_rng(3)
    # t0 hopping between two inter-TOA gaps, well mixed in reference and ours
    ref_c = markov_chains(rng, 4, 4000, 0.2, 0.2)
    our_c = markov_chains(rng, 4, 4000, 0.2, 0.2)
    ref_ind = {"I0": ref_c, "rest": [~c for c in ref_c]}
    our_ind = {"I0": our_c, "rest": [~c for c in our_c]}
    assert event_interval_gate(ref_ind, our_ind).status == PASS
    # slower mixing: occupancy MCSE > 0.01 with both intervals material -> FAIL
    slow = markov_chains(rng, 4, 4000, 0.02, 0.02)
    assert event_interval_gate(ref_ind, {"I0": slow, "rest": [~c for c in slow]}).status == FAIL
    stuck = [np.ones(4000, bool)] * 4
    st = event_interval_gate(ref_ind, {"I0": stuck, "rest": [~c for c in stuck]})
    assert st.status == INCONCLUSIVE
    with pytest.raises(GateInputError):
        event_interval_gate(ref_ind, {"I0": our_c})


# ---------------------------------------------------------------- Sec. 6.6 D9


def _normal_sample(n=200001, loc=0.0):
    return loc + stats.norm.ppf((np.arange(n) + 0.5) / n)


def test_envelope_bounds_normal():
    x = _normal_sample()
    for alpha in (0.05, 0.5, 0.95):
        for p in (0.001, 0.01, 0.05, 0.10):
            lo, hi = envelope_quantile_bounds(x, alpha, p, -10.0, 10.0)
            exp_lo = -10.0 if alpha <= p else stats.norm.ppf((alpha - p) / (1 - p))
            exp_hi = 10.0 if alpha / (1 - p) >= 1 else stats.norm.ppf(alpha / (1 - p))
            assert lo == pytest.approx(exp_lo, abs=2e-3)
            assert hi == pytest.approx(exp_hi, abs=2e-3)
    assert envelope_quantile_bounds(x, 0.05, 0.05, -10.0, 10.0)[0] == -10.0
    assert envelope_quantile_bounds(x, 0.95, 0.06, -10.0, 10.0)[1] == 10.0
    # weighted (support weights for HD-reweighted samples): proposal N(0,1), target N(0.5,1)
    rng = np.random.default_rng(4)
    z = rng.normal(size=400000)
    w = np.exp(0.5 * z - 0.125)
    lo, hi = envelope_quantile_bounds(z, 0.5, 0.05, -10.0, 10.0, weights=w)
    assert lo == pytest.approx(0.5 + stats.norm.ppf(0.45 / 0.95), abs=0.02)
    assert hi == pytest.approx(0.5 + stats.norm.ppf(0.5 / 0.95), abs=0.02)


def test_domain_indicator_and_conditional_quantile():
    rng = np.random.default_rng(11)
    a = [rng.uniform(-18, -10, 3000) for _ in range(3)]
    g = [rng.normal(57510, 3, 3000) for _ in range(3)]
    ex = [{"name": "A:S", "type": "shelf", "boundaries": [-18, -17], "params": {"crn": "a", "hd": "a_hd"}},
          {"name": "t0:rest", "type": "outside", "boundaries": [57507.0, 57514.0], "params": {"crn": "t", "hd": "t_hd"}}]
    cols = {"a": a, "t": g, "a_hd": a, "t_hd": g}
    for model in ("crn", "hd"):  # one common domain under the parameter mapping
        ind = domain_indicator(lambda n: cols[n], ex, model)
        for i in range(3):
            exp = ~((a[i] >= -18) & (a[i] <= -17)) & (g[i] >= 57507.0) & (g[i] < 57514.0)
            assert np.array_equal(ind[i], exp)
    ind = domain_indicator(lambda n: cols[n], ex, "crn")
    r = conditional_quantile(a, ind, 0.5)
    sub = np.concatenate([x[m] for x, m in zip(a, ind)])
    assert r["q"] == pytest.approx(np.quantile(sub, 0.5), abs=0.01) and np.isfinite(r["mcse"])
    assert r["q"] == pytest.approx(-13.5, abs=0.1)  # shelf excluded: median of U(-17, -10) (t0 indep.)
    with pytest.raises(ValueError):
        conditional_quantile(a, [np.zeros(3000, bool)] * 3, 0.5)


def test_conditional_quantile_mcse_coverage_with_correlated_domain():
    """Ordered-chain MCSE of a conditional quantile: AR(1) draws, a domain indicator correlated with
    the draws (an excluded shelf of the chain's own values); coverage of the true conditional
    median over replicate chains."""
    rng = np.random.default_rng(12)
    rho, n, reps = 0.9, 4000, 150
    hits = 0
    truth = stats.norm.ppf((stats.norm.cdf(-1.5) + 1) / 2)  # median of N(0,1) restricted to x > -1.5
    for _ in range(reps):
        chains = []
        for _c in range(2):
            e = rng.normal(size=n) * np.sqrt(1 - rho**2)
            x = np.empty(n)
            x[0] = rng.normal()
            for i in range(1, n):
                x[i] = rho * x[i - 1] + e[i]
            chains.append(x)
        ind = [c > -1.5 for c in chains]
        r = conditional_quantile(chains, ind, 0.5)
        hits += abs(r["q"] - truth) <= 1.96 * r["mcse"]
    assert 0.85 <= hits / reps <= 0.99


def test_descriptive_envelope_and_bf_correction():
    x = _normal_sample()
    env = descriptive_envelope(x, x, 0.5, -10.0, 10.0)
    assert env["label"].startswith("descriptive")
    lo, hi = env["intervals"][0.05]
    assert lo < 0 < hi and hi - lo == pytest.approx(2 * (stats.norm.ppf(0.5 / 0.95) - stats.norm.ppf(0.45 / 0.95)), abs=0.01)
    assert bf_domain_correction(0.0, 0.0) == 0.0
    assert bf_domain_correction(0.01, 0.0) == pytest.approx(np.log(0.99))
    for bad in (np.nan, -0.1, 1.0):
        with pytest.raises(GateInputError):
            bf_domain_correction(bad, 0.0)


def test_d9_eligibility():
    gate = {"model_identity": PASS, "convergence_D": PASS}
    head = {"E-1 q50": "EQUIVALENT", "E-3 q50": "EQUIVALENT"}
    inv = {"required_checks": ("model_identity", "convergence_D"), "required_headline": ("E-1 q50", "E-3 q50"),
           "required_models": ("crn_pl", "hd_pl")}
    ex = {"crn_pl": {"reference": 38, "ours": 5}, "hd_pl": {"reference": 78, "ours": 0}}
    ok = d9_eligibility(gate, head, excluded_draws=ex, **inv)
    assert ok.eligible and ok.claim == D9_CLASS and ok.excluded_draws == ex
    assert ok.unconditional_verdict == INCONCLUSIVE and ok.template == D9_TEMPLATE
    assert "REPRODUCED" not in (ok.claim or "")
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(ok, unconditional_verdict=PASS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ok.unconditional_verdict = PASS
    bad = d9_eligibility({**gate, "convergence_D": INCONCLUSIVE}, head, excluded_draws=ex, **inv)
    assert not bad.eligible and bad.claim is None
    bad = d9_eligibility(gate, {**head, "E-3 q50": INCONCLUSIVE}, excluded_draws=ex, **inv)
    assert not bad.eligible


def test_d9_eligibility_fails_closed_on_bad_inventories():
    gate = {"model_identity": PASS, "convergence_D": PASS}
    head = {"E-1 q50": "EQUIVALENT"}
    inv = {"required_checks": ("model_identity", "convergence_D"), "required_headline": ("E-1 q50",),
           "required_models": ("crn_pl", "hd_pl")}
    ex = {"crn_pl": {"reference": 38, "ours": 5}, "hd_pl": {"reference": 78, "ours": 0}}
    with pytest.raises(GateInputError):  # an omitted check
        d9_eligibility({"model_identity": PASS}, head, excluded_draws=ex, **inv)
    with pytest.raises(GateInputError):  # an omitted headline quantity
        d9_eligibility(gate, {}, excluded_draws=ex, **inv)
    with pytest.raises(GateInputError):  # a model missing
        d9_eligibility(gate, head, excluded_draws={"crn_pl": ex["crn_pl"]}, **inv)
    for v in (np.nan, -1, 2.5, None, True):  # NaN / invalid excluded-draw counts
        with pytest.raises(GateInputError):
            d9_eligibility(gate, head, excluded_draws={**ex, "hd_pl": {"reference": v, "ours": 0}}, **inv)
    with pytest.raises(GateInputError):  # no inventories
        d9_eligibility(gate, head, excluded_draws=ex, required_checks=(), required_headline=(), required_models=())
