"""M3a review fixes: order-invariant duplicate finder (G7), rank-revealing projector, configuration-
specific quarantine."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from ptagwb.multileg import find_duplicates
from ptagwb.timfile import TimRecord


def _rec(name, mjd, tobs, freq=1400.0, obs="eff"):
    return TimRecord("f.tim", 1, name, freq, mjd, 1.0, obs, [("-tobs", str(tobs))], 0.0)


def test_duplicates_symmetric_and_order_invariant():
    # a long observation A (3720 s) and a short one B (150 s) 1796 s apart: the intervals overlap,
    # but a search window built from the first observation's duration alone (150 s) misses it
    A = [_rec("long", "58810.256447794198788", 3720, obs="leap")]
    B = [_rec("short", "58810.277230804531478", 150, freq=1400.4)]
    r1 = find_duplicates({"a": B, "b": A})
    r2 = find_duplicates({"x": A, "y": B})
    assert r1 and r2 and r1[0]["n_pairs"] == r2[0]["n_pairs"] == 1
    # no overlap when the gap exceeds (d_a + d_b) / 2
    C = [_rec("far", "58810.30", 150)]
    assert not find_duplicates({"a": A, "b": C})
    # different channel: not a same-channel duplicate, but a band overlap with -bw
    D = [TimRecord("f.tim", 2, "d", 1430.0, "58810.256447794198788", 1.0, "eff", [("-tobs", "3720"), ("-bw", "200")], 0.0)]
    E = [TimRecord("f.tim", 3, "e", 1400.0, "58810.26", 1.0, "eff", [("-tobs", "600"), ("-bw", "100")], 0.0)]
    assert not find_duplicates({"a": D, "b": E})
    assert find_duplicates({"a": D, "b": E}, freq_mode="band")


@pytest.mark.slow
def test_released_j1022_overlap_found_in_both_orders():
    from ptagwb.m3data import leg_files
    from ptagwb.timfile import read_tim

    try:
        _, tim = leg_files("epta_dr2new")["J1022+1001"]
    except (FileNotFoundError, KeyError):
        pytest.skip("M3 data not fetched")
    by = defaultdict(list)
    for r in read_tim(tim)[0]:
        by[(r.flag_values("-group") or r.flag_values("-sys"))[0]].append(r)
    sub = {k: v for k, v in by.items() if k in ("LEAP.1396", "EFF.P217.1380")}
    for order in (list(sub), list(sub)[::-1]):
        res = find_duplicates({k: sub[k] for k in order})
        pairs = [(p["a"][:2], p["b"][:2]) for d in res for p in d["pairs"]]
        assert any({tuple(a), tuple(b)} == {("EFF.P217.1380.tim", 180), ("LEAP.1396.tim", 80)} for a, b in pairs), pairs


def test_projector_rank_revealing():
    from m3a_oracles import weighted_projector_complement

    A = np.ones((4, 2))  # two identical columns: rank 1
    P = weighted_projector_complement(A)
    assert P.rank == 1 and P.ncols == 2
    np.testing.assert_allclose(P(np.array([1.0, -1.0, 0.0, 0.0])), [1.0, -1.0, 0.0, 0.0], atol=1e-15)
    rng = np.random.default_rng(0)
    B = rng.normal(size=(50, 4))
    M = np.column_stack([B, B @ rng.normal(size=(4, 3))])  # 7 columns, rank 4
    P = weighted_projector_complement(M)
    assert P.rank == 4
    x = rng.normal(size=50)
    np.testing.assert_allclose(B.T @ P(x), 0, atol=1e-10)


def test_quarantine_is_configuration_specific():
    from ptagwb.m3data import quarantine

    q4 = quarantine({"ell1h_nharms": "tempo2"})
    q7 = quarantine({"ell1h_nharms": "pint7"})
    assert ("mpta", "J2145-0750") not in q4 and ("mpta", "J2145-0750") in q7
    assert ("mpta", "J1825-0319") in q4 and ("mpta", "J1825-0319") in q7
    assert len(q7) - len(q4) == 10
