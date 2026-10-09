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
    D9Sample,
    Occupancy,
    aggregate,
    bf_sensitivity,
    classify,
    classify_e6,
    classify_from_frozen,
    d9_eligibility,
    d9_excluded_regions,
    d9_quantity_survives,
    decidable,
    envelope_quantile_bounds,
    epsilon_m,
    event_interval_gate,
    event_interval_indicators,
    event_intervals,
    max_our_mcse,
    occupancy,
    p_star,
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
    good = {"name": "S_J0437", "model": "CURN", "boundaries": [-20, -19],
            "reference_file": "chain_1.txt", "sha256": "ab", "burn_in": 0.25}
    assert d9_excluded_regions({"d9": {"excluded_regions": [good]}}) == [good]
    bad = dict(good)
    del bad["sha256"]
    with pytest.raises(GateInputError):
        d9_excluded_regions({"d9": {"excluded_regions": [bad]}})
    with pytest.raises(GateInputError):
        d9_excluded_regions({})


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


def test_survives_and_p_star():
    x = _normal_sample(20001)
    m = 0.3
    s = D9Sample(x, -10.0, 10.0, mcse=0.0)
    r0 = d9_quantity_survives(s.bounds(0.5, 0.0), s.bounds(0.5, 0.0), 0.0, m)
    assert r0["survives"] and r0["interval"][1] == pytest.approx(0.0, abs=1e-3)
    ps = p_star(s, s, 0.5, 0.0, m)
    # analytic: 2 Phi^-1(0.5 / (1 - p)) = m  ->  p* = 1 - 0.5 / Phi(m / 2)
    assert ps["p_star"] == pytest.approx(1 - 0.5 / stats.norm.cdf(m / 2), abs=1e-3)
    assert ps["p_star_lo90"] == pytest.approx(ps["p_star"], abs=1e-6)  # zero MCSE
    # monotone: larger margin -> larger p*; MCSE -> lower bound below p*
    prev = 0.0
    for mm in (0.1, 0.2, 0.3, 0.5):
        cur = p_star(s, s, 0.5, 0.0, mm)["p_star"]
        assert cur > prev
        prev = cur
    sm = D9Sample(x, -10.0, 10.0, mcse=0.01)
    r = p_star(sm, sm, 0.5, 0.02, m)
    assert r["p_star_lo90"] < r["p_star"] < ps["p_star"]
    grid = r["survival_on_grid"]
    vals = [grid[p] for p in sorted(grid)]
    assert vals == sorted(vals, reverse=True)  # survival non-increasing in p
    # not equivalent at p = 0 -> p* = 0; survives at p_max -> capped
    assert p_star(s, D9Sample(x + 1.0, -10, 10), 0.5, 0.0, m)["p_star"] == 0.0
    assert p_star(s, s, 0.5, 0.0, 100.0)["capped"]


def test_epsilon_m():
    assert epsilon_m([0.07, 0.2]) == 0.05
    assert epsilon_m([0.6]) == 0.10
    assert epsilon_m([0.001]) == 0.001
    assert epsilon_m([0.0009, 0.5]) is None
    assert epsilon_m([]) is None


def test_bf_sensitivity():
    t = np.log(60.0)
    r = bf_sensitivity(t, 0.05, 0.30, t, 0.01, 0.01)
    assert r["table"][(0.1, 0.0)] == pytest.approx(np.log(0.9))
    assert r["table"][(0.0, 0.1)] == pytest.approx(-np.log(0.9))
    assert r["table"][(0.05, 0.05)] == 0.0
    assert len(r["table"]) == 25
    lo, hi = r["interval"]
    assert lo == pytest.approx(t - Z90 * 0.05 + np.log(0.99))
    assert hi == pytest.approx(t + Z90 * 0.05 - np.log(0.99))
    assert r["qualifies"]
    assert not bf_sensitivity(t + 0.1, 0.08, 0.30, t, 0.10, 0.10)["qualifies"]
    assert bf_sensitivity(t + 0.1, 0.08, 0.30, t, 0.001, 0.001)["qualifies"]
    assert not bf_sensitivity(t, 0.11, 0.30, t, 0.001, 0.001)["qualifies"]  # E-6 SE cap
    assert not bf_sensitivity(t, 0.05, 0.30, t, None, 0.01)["qualifies"]


def test_d9_eligibility():
    zero = occupancy([np.zeros(1000, bool)] * 4)
    few = occupancy([np.r_[np.zeros(990, bool), np.ones(10, bool)]] * 2)
    visited = occupancy(markov_chains(np.random.default_rng(5), 4, 4000, 0.05, 0.05))
    gate = {"rhat_ess": PASS, "k_hat": PASS}
    head = {"E-1 q50": ("EQUIVALENT", True), "E-1 q05": (INCONCLUSIVE, False)}
    eps = {"CURN": 0.01, "HD": 0.001}
    inv = {"required_models": ("CURN", "HD"), "required_checks": ("rhat_ess", "k_hat")}
    ok = d9_eligibility({"S1": (zero, zero), "S2": (zero, zero)}, gate, head, eps, **inv)
    assert ok.eligible and ok.available and ok.claim == D9_CLASS
    assert ok.unconditional_verdict == INCONCLUSIVE and ok.template == D9_TEMPLATE
    assert "REPRODUCED" not in (ok.claim or "")
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(ok, unconditional_verdict=PASS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ok.unconditional_verdict = PASS
    asym = d9_eligibility({"S1": (zero, visited)}, gate, head, eps, **inv)
    assert not asym.eligible and asym.claim is None and asym.separate_review == ("S1",)
    asym2 = d9_eligibility({"S1": (visited, zero)}, gate, head, eps, **inv)
    assert not asym2.eligible and "S1" in asym2.separate_review
    fe = d9_eligibility({"S1": (zero, few)}, gate, head, eps, **inv)
    assert not fe.eligible and "few-event" in fe.reasons[0]
    both = d9_eligibility({"S1": (visited, visited)}, gate, head, eps, **inv)
    assert not both.eligible and not both.separate_review
    g = d9_eligibility({"S1": (zero, zero)}, {**gate, "transport_gamma": FAIL}, head, eps, **inv)
    assert not g.eligible
    h = d9_eligibility({"S1": (zero, zero)}, gate, {"E-1 q50": (INCONCLUSIVE, True)}, eps, **inv)
    assert not h.eligible
    e = d9_eligibility({"S1": (zero, zero)}, gate, head, {"CURN": 0.01, "HD": None}, **inv)
    assert e.eligible and not e.available and e.claim is None
    assert e.unconditional_verdict == INCONCLUSIVE
    with pytest.raises(GateInputError):
        d9_eligibility({}, gate, head, eps, **inv)


def test_d9_eligibility_fails_closed_on_bad_inventories():
    zero = occupancy([np.zeros(1000, bool)] * 4)
    gate = {"rhat_ess": PASS, "k_hat": PASS}
    head = {"E-1 q50": ("EQUIVALENT", True)}
    inv = {"required_models": ("CURN", "HD"), "required_checks": ("rhat_ess", "k_hat")}
    regs = {"S1": (zero, zero)}
    for bad in (np.nan, float("inf"), -np.inf):  # NaN / non-finite epsilon (review of 2ee1bb7)
        with pytest.raises(GateInputError):
            d9_eligibility(regs, gate, head, {"CURN": 0.01, "HD": bad}, **inv)
    with pytest.raises(GateInputError):  # a model missing from the inventory
        d9_eligibility(regs, gate, head, {"HD": 0.01}, **inv)
    with pytest.raises(GateInputError):  # an omitted check
        d9_eligibility(regs, {"rhat_ess": PASS}, head, {"CURN": 0.01, "HD": 0.01}, **inv)
    with pytest.raises(GateInputError):  # no inventories
        d9_eligibility(regs, gate, head, {"CURN": 0.01, "HD": 0.01}, required_models=(), required_checks=())
    with pytest.raises(GateInputError):
        epsilon_m([0.05, np.nan])
