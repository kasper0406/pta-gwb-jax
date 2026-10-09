"""M3b pre-registered acceptance logic (plan Sec. 5.3, 6.1, 6.2, 6.6). Pure numpy/scipy, fail closed.

Contents
--------
* Sec. 6.1 decision rule: ``classify``, ``max_our_mcse``, ``decidable``; Sec. 6.2 ``classify_e6``.
* Sec. 5.3 transport gate: region occupancy intervals (``occupancy``), support classes,
  the per-(parameter, region pair) ``transport_gate`` with the UNRESOLVED precedence rule, shelf /
  peak regions, event-epoch intervals, and ``aggregate``.
* Sec. 6.6 D9 class "CONDITIONALLY EQUIVALENT TO RELEASED RESULTS": mixture-envelope quantile
  bounds, survival, p*, epsilon_m, the BF sensitivity table and eligibility. The unconditional
  verdict stays INCONCLUSIVE, always.

Statuses are the strings "PASS" / "FAIL" / "INCONCLUSIVE" (gate) and "EQUIVALENT" /
"INCOMPATIBLE" / "INCONCLUSIVE" (quantities). Functions consuming frozen acceptance-file entries
raise ``ptagwb.diagnostics.GateInputError`` when a required field is missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from ptagwb.diagnostics import GateInputError
from ptagwb.reweight import Z90, integrated_autocorr_time, weighted_quantile

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
EQUIVALENT, INCOMPATIBLE = "EQUIVALENT", "INCOMPATIBLE"
UNRESOLVED, RESOLVED = "UNRESOLVED", "RESOLVED"
MATERIAL, ABSENT, AMBIGUOUS = "material", "absent", "ambiguous"
Z95_TWO_SIDED = 1.959963984540054

MIN_EVENTS = 10            # entries and exits, pooled, for an estimable occupancy
SUPPORT_THRESHOLD = 0.01   # material p_lo >= 0.01; absent p_hi < 0.01
ABSENT_PASS_MAX = 0.02     # one-region-absent row: p_hi,ours < 0.02 (PASS), p_lo,ours >= 0.02 (FAIL)
SOJOURN_MAX = 0.5
OCC_MCSE_MAX = 0.01        # amplitude pairs
EVENT_OCC_MCSE_MAX = 0.02  # extra Sec. 5.3(b) rule for material t0 intervals
E6_MAX_SE = 0.10
D9_P_GRID: Final = (0.001, 0.01, 0.05, 0.10)
D9_BF_GRID: Final = (0.0, 0.001, 0.01, 0.05, 0.10)
D9_CLASS: Final = "CONDITIONALLY EQUIVALENT TO RELEASED RESULTS"

D9_TEMPLATE: Final = (
    "**CONDITIONALLY EQUIVALENT TO RELEASED RESULTS.** All predeclared decidable headline "
    "quantities meet the equivalence criteria conditional on excluding the explicitly listed "
    "regions U. Both retained sample sets contain zero visits to U; its posterior mass remains "
    "unresolved. Extension to the unrestricted posterior assumes P_m(U) <= epsilon_m for each "
    "relevant model, with numerical thresholds and sensitivity results reported below. These mass "
    "assumptions have not been established by the chains. This claim is weaker than REPRODUCED; "
    "the unconditional verdict remains INCONCLUSIVE."
)


def require_fields(entry: Mapping, fields: Iterable[str], where: str = "acceptance entry") -> None:
    """Raise GateInputError unless every field is present (and not None) in ``entry``."""
    if not isinstance(entry, Mapping):
        raise GateInputError(f"{where}: expected a mapping, got {type(entry).__name__}")
    missing = [f for f in fields if f not in entry or entry[f] is None]
    if missing:
        raise GateInputError(f"{where}: missing required field(s) {missing}")


# ----------------------------------------------------------------------------------------------
# Sec. 6.1 / 6.2 decision rule
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """Result of the Sec. 6.1 rule. ``limited`` is "reference-limited" / "ours-limited" for
    INCONCLUSIVE (None otherwise); ``interval`` = D +- 1.645 SE_D."""

    status: str
    d: float
    se_d: float
    interval: tuple[float, float]
    m: float
    limited: str | None = None


def classify(d: float, se_ours: float, se_ref: float, m: float) -> Classification:
    """Sec. 6.1: SE_D = sqrt(MCSE_ours^2 + MCSE_ref^2), I_D = D +- 1.645 SE_D. EQUIVALENT if I_D is
    within [-m, m] (closed), INCOMPATIBLE if I_D lies entirely outside it (strictly beyond +-m),
    else INCONCLUSIVE, reference-limited iff 1.645 MCSE_ref >= m. Non-finite inputs give
    INCONCLUSIVE (ours-limited unless the reference is limiting)."""
    se_d = float(np.hypot(se_ours, se_ref))
    lo, hi = d - Z90 * se_d, d + Z90 * se_d
    if np.isfinite(lo) and np.isfinite(hi):
        if lo >= -m and hi <= m:
            return Classification(EQUIVALENT, d, se_d, (lo, hi), m)
        if lo > m or hi < -m:
            return Classification(INCOMPATIBLE, d, se_d, (lo, hi), m)
    lim = "reference-limited" if Z90 * se_ref >= m else "ours-limited"
    return Classification(INCONCLUSIVE, d, se_d, (lo, hi), m, lim)


def max_our_mcse(m: float, mcse_ref: float) -> float:
    """max_our_MCSE = sqrt((m / 1.645)^2 / 4 - MCSE_ref^2), floored at 0.2 m / 1.645 (the floor is
    also used when the sqrt argument is negative)."""
    floor = 0.2 * m / Z90
    arg = (m / Z90) ** 2 / 4.0 - mcse_ref**2
    return float(max(np.sqrt(arg), floor)) if arg > 0 else float(floor)


def decidable(m: float, mcse_ref: float) -> bool:
    """Sec. 6.2: decidable iff 1.645 MCSE_ref < m."""
    return bool(Z90 * mcse_ref < m)


def classify_e6(lnbf_ours: float, se: float, target: float = float(np.log(60.0)),
                m: float = 0.30, max_se: float = E6_MAX_SE) -> Classification:
    """E-6 (Sec. 6.2): point target without SE. SE > max_se (or non-finite) -> INCONCLUSIVE
    (ours-limited); else EQUIVALENT if ln BF +- 1.645 SE within [target - m, target + m],
    INCOMPATIBLE if entirely outside, else INCONCLUSIVE."""
    if not np.isfinite(se) or se > max_se:
        d = lnbf_ours - target
        return Classification(INCONCLUSIVE, d, float(se), (d - Z90 * se, d + Z90 * se), m,
                              "ours-limited")
    return classify(lnbf_ours - target, se, 0.0, m)


def classify_from_frozen(entry: Mapping, q_ours: float, mcse_ours: float) -> Classification:
    """Classify one quantity against a frozen acceptance entry. Required fields: q_ref, mcse_ref,
    m (GateInputError otherwise)."""
    require_fields(entry, ("q_ref", "mcse_ref", "m"))
    return classify(q_ours - float(entry["q_ref"]), mcse_ours, float(entry["mcse_ref"]),
                    float(entry["m"]))


# ----------------------------------------------------------------------------------------------
# Sec. 5.3 occupancy
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Occupancy:
    """Occupancy of a region R from retained draws (Sec. 5.3).

    case: "estimable" | "few-event" | "zero-visit" | "all-visit"; status RESOLVED only for
    "estimable". UNRESOLVED intervals are reported as the trivial [0, 1] (so p_hi = 1 for the
    all-visit case, as the plan requires). An entry is a 0 -> 1 transition of the indicator within
    a chain, an exit a 1 -> 0 transition (chain boundaries are not transitions). tau is the
    integrated autocorrelation time of the indicator itself (pooled across chains, floored at 1, see
    ``ptagwb.reweight.integrated_autocorr_time``), n_eff = N / tau, [p_lo, p_hi] the 95 % Wilson
    score interval on n_eff and mcse = sqrt(p_hat (1 - p_hat) / n_eff) (all None unless
    estimable). longest_sojourn_frac = longest run of consecutive draws in R / total draws in R
    (convention of ``diagnostics.region_events``; None with no visits).
    """

    p_hat: float
    p_lo: float
    p_hi: float
    status: str
    case: str
    n_entries: int
    n_exits: int
    n_draws: int
    n_chains: int
    n_chains_visiting: int
    tau: float | None
    n_eff: float | None
    mcse: float | None
    longest_sojourn_frac: float | None

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED


def wilson_interval(p: float, n: float, z: float = Z95_TWO_SIDED) -> tuple[float, float]:
    """Wilson score interval for a proportion p with (effective) sample size n."""
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))


def _as_indicator_chains(indicator) -> list[np.ndarray]:
    if isinstance(indicator, np.ndarray) and indicator.ndim == 1:
        indicator = [indicator]
    chains = [np.asarray(c).astype(bool).ravel() for c in indicator]
    if not chains or any(c.size == 0 for c in chains):
        raise ValueError("occupancy needs >= 1 non-empty chain")
    return chains


def _events(chains: list[np.ndarray]) -> tuple[int, int, int, int]:
    """(entries, exits, longest sojourn, chains visiting)."""
    ent = ex = longest = visiting = 0
    for c in chains:
        ci = c.astype(np.int8)
        d = np.diff(ci)
        ent += int(np.sum(d == 1))
        ex += int(np.sum(d == -1))
        if c.any():
            visiting += 1
            e = np.diff(np.concatenate([[0], ci, [0]]))
            longest = max(longest, int(np.max(np.flatnonzero(e == -1) - np.flatnonzero(e == 1))))
    return ent, ex, longest, visiting


def occupancy(indicator, min_events: int = MIN_EVENTS) -> Occupancy:
    """Occupancy interval of a region from per-chain boolean indicator arrays of retained draws.

    Cases (Sec. 5.3a): zero visits -> UNRESOLVED; every draw in R -> "all-visit", the zero-visit
    case of the complement, UNRESOLVED with p_hi = 1; some visits but < min_events entries or
    exits pooled -> "few-event", UNRESOLVED; else "estimable" with the Wilson interval on
    n_eff = N / tau_R. No certified-bound path exists (none is planned in M3b).
    """
    chains = _as_indicator_chains(indicator)
    n = int(sum(c.size for c in chains))
    k = int(sum(int(c.sum()) for c in chains))
    ent, ex, longest, visiting = _events(chains)
    p_hat = k / n
    sojourn = (longest / k) if k else None
    common = {"p_hat": p_hat, "n_entries": ent, "n_exits": ex, "n_draws": n,
              "n_chains": len(chains), "n_chains_visiting": visiting,
              "longest_sojourn_frac": sojourn}
    if k == 0:
        return Occupancy(p_lo=0.0, p_hi=1.0, status=UNRESOLVED, case="zero-visit", tau=None,
                         n_eff=None, mcse=None, **common)
    if k == n:
        comp = occupancy([~c for c in chains], min_events)
        return Occupancy(p_lo=0.0, p_hi=1.0, status=comp.status, case="all-visit", tau=None,
                         n_eff=None, mcse=None, **common)
    if ent < min_events or ex < min_events:
        return Occupancy(p_lo=0.0, p_hi=1.0, status=UNRESOLVED, case="few-event", tau=None,
                         n_eff=None, mcse=None, **common)
    tau = max(integrated_autocorr_time([c.astype(np.float64) for c in chains]), 1.0)
    n_eff = n / tau
    lo, hi = wilson_interval(p_hat, n_eff)
    return Occupancy(p_lo=lo, p_hi=hi, status=RESOLVED, case="estimable", tau=float(tau),
                     n_eff=float(n_eff), mcse=float(np.sqrt(p_hat * (1 - p_hat) / n_eff)),
                     **common)


def support_class(occ: Occupancy) -> str:
    """material if p_lo >= 0.01, absent if p_hi < 0.01, ambiguous otherwise; UNRESOLVED passes
    through."""
    if not occ.resolved:
        return UNRESOLVED
    if occ.p_lo >= SUPPORT_THRESHOLD:
        return MATERIAL
    if occ.p_hi < SUPPORT_THRESHOLD:
        return ABSENT
    return AMBIGUOUS


def _overlap(a: Occupancy, b: Occupancy) -> bool:
    return a.p_lo <= b.p_hi and b.p_lo <= a.p_hi


def bidirectional_transport(ours: Sequence[Occupancy], mcse_max: float = OCC_MCSE_MAX,
                            sojourn_max: float = SOJOURN_MAX,
                            min_events: int = MIN_EVENTS) -> tuple[bool, list[str]]:
    """Sec. 5.3 table: for every region of the pair, >= min_events entries and exits pooled,
    >= 2 chains visiting, longest sojourn <= sojourn_max, occupancy MCSE <= mcse_max."""
    why = []
    for i, o in enumerate(ours):
        if o.n_entries < min_events or o.n_exits < min_events:
            why.append(f"region {i}: {o.n_entries} entries / {o.n_exits} exits < {min_events}")
        if o.n_chains_visiting < 2:
            why.append(f"region {i}: {o.n_chains_visiting} chain(s) visiting < 2")
        if o.longest_sojourn_frac is None or o.longest_sojourn_frac > sojourn_max:
            why.append(f"region {i}: longest sojourn {o.longest_sojourn_frac} > {sojourn_max}")
        if o.mcse is None or not o.mcse <= mcse_max:
            why.append(f"region {i}: occupancy MCSE {o.mcse} > {mcse_max}")
    return not why, why


@dataclass(frozen=True)
class GateItem:
    """One gate item: status with the reasons and the support classes used."""

    status: str
    reasons: tuple[str, ...] = ()
    ref_support: tuple[str, ...] = ()
    our_support: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


def transport_gate(ref: Sequence[Occupancy], ours: Sequence[Occupancy],
                   labels: Sequence[str] = ("S", "P"), mcse_max: float = OCC_MCSE_MAX,
                   sojourn_max: float = SOJOURN_MAX) -> GateItem:
    """Transport decision for one (parameter, region pair), Sec. 5.3, evaluated in this order:

    1. Precedence: any UNRESOLVED interval (reference or ours, either region) -> INCONCLUSIVE.
    2. Table, by the reference support of the pair:
       * both material: PASS iff bidirectional transport and our interval overlaps the
         reference's for both regions; else FAIL.
       * one region absent: for that region R, PASS iff overlap and p_hi,ours(R) < 0.02; FAIL iff
         disjoint or p_lo,ours(R) >= 0.02. The remaining case (overlap, p_lo < 0.02 <= p_hi) is
         neither row and is returned INCONCLUSIVE (plan leaves it open; chosen fail-safe).
       * otherwise (a region ambiguous): PASS iff bidirectional transport and overlap; else
         INCONCLUSIVE.
    """
    if len(ref) != 2 or len(ours) != 2:
        raise ValueError("transport_gate takes a region pair (2 occupancies each)")
    rs = tuple(support_class(o) for o in ref)
    os_ = tuple(support_class(o) for o in ours)
    unresolved = tuple(f"{who}:{lab}" for who, occs in (("reference", ref), ("ours", ours))
                       for lab, o in zip(labels, occs) if not o.resolved)
    if unresolved:
        return GateItem(INCONCLUSIVE, (f"unresolved regions {list(unresolved)}",), rs, os_,
                        unresolved)
    overlap = [_overlap(r, o) for r, o in zip(ref, ours)]
    ok_tr, why_tr = bidirectional_transport(ours, mcse_max, sojourn_max)
    why_ov = [f"{lab}: intervals disjoint" for lab, ov in zip(labels, overlap) if not ov]
    if rs == (MATERIAL, MATERIAL):
        if ok_tr and all(overlap):
            return GateItem(PASS, (), rs, os_)
        return GateItem(FAIL, tuple(why_tr + why_ov), rs, os_)
    if ABSENT in rs:
        i = rs.index(ABSENT)
        o = ours[i]
        lab = labels[i]
        if overlap[i] and o.p_hi < ABSENT_PASS_MAX:
            return GateItem(PASS, (), rs, os_)
        if not overlap[i] or o.p_lo >= ABSENT_PASS_MAX:
            msg = (f"{lab}: absent in reference; ours [{o.p_lo:.4g}, {o.p_hi:.4g}], "
                   f"overlap={overlap[i]}")
            return GateItem(FAIL, (msg,), rs, os_)
        msg = (f"{lab}: absent in reference; ours p_hi {o.p_hi:.4g} >= {ABSENT_PASS_MAX} > "
               f"p_lo {o.p_lo:.4g}")
        return GateItem(INCONCLUSIVE, (msg,), rs, os_)
    if ok_tr and all(overlap):
        return GateItem(PASS, (), rs, os_)
    return GateItem(INCONCLUSIVE, tuple(why_tr + why_ov), rs, os_)


def event_interval_gate(ref_ind: Mapping[str, Sequence[np.ndarray]],
                        our_ind: Mapping[str, Sequence[np.ndarray]]) -> GateItem:
    """Sec. 5.3(b) t0 gate from per-chain interval indicators (``event_interval_indicators``,
    same keys for reference and ours, including "rest"). Each interval I is gated as the pair
    (I, not I) with ``transport_gate`` (occupancies from the indicators, all-visit via the
    complement); in addition every interval material in the reference must be visited by >= 2 of
    our chains with occupancy MCSE <= 0.02 (FAIL otherwise, unless precedence already made the
    item INCONCLUSIVE). Items are combined with ``aggregate``."""
    if set(ref_ind) != set(our_ind) or not ref_ind:
        raise GateInputError("event_interval_gate: reference and our interval keys differ")
    items = []
    for lab in ref_ind:
        rc = _as_indicator_chains(ref_ind[lab])
        oc = _as_indicator_chains(our_ind[lab])
        r, o = occupancy(rc), occupancy(oc)
        it = transport_gate((r, occupancy([~c for c in rc])), (o, occupancy([~c for c in oc])),
                            labels=(lab, f"not {lab}"))
        if it.status != INCONCLUSIVE and support_class(r) == MATERIAL and (
                o.n_chains_visiting < 2 or o.mcse is None or o.mcse > EVENT_OCC_MCSE_MAX):
            msg = f"{lab}: material interval needs >= 2 chains and MCSE <= {EVENT_OCC_MCSE_MAX}"
            it = GateItem(FAIL, (*it.reasons, msg), it.ref_support, it.our_support)
        items.append(it)
    status = aggregate([i.status for i in items])
    return GateItem(status, tuple(r for i in items for r in i.reasons),
                    unresolved=tuple(u for i in items for u in i.unresolved))


def aggregate(items: Iterable) -> str:
    """FAIL if any FAIL; INCONCLUSIVE if any INCONCLUSIVE (and none FAIL); PASS only if all pass.
    Accepts status strings or objects with a ``status`` attribute. An empty list or an unknown
    status raises (fail closed)."""
    st = [getattr(i, "status", i) for i in items]
    if not st:
        raise GateInputError("aggregate: no gate items")
    bad = [s for s in st if s not in (PASS, FAIL, INCONCLUSIVE)]
    if bad:
        raise GateInputError(f"aggregate: unknown statuses {bad}")
    if FAIL in st:
        return FAIL
    if INCONCLUSIVE in st:
        return INCONCLUSIVE
    return PASS


# ----------------------------------------------------------------------------------------------
# regions
# ----------------------------------------------------------------------------------------------


def _float_chains(x) -> list[np.ndarray]:
    """A 1-D array -> [array]; else a sequence (or 2-D array) of per-chain 1-D arrays."""
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = [x]
    return [np.asarray(c, np.float64).ravel() for c in x]


def shelf_peak_regions(lower: float, upper: float) -> dict[str, tuple[float, float]]:
    """Shelf S = [lower, lower + 1 dex] and peak P = (lower + 1, upper] within the prior box."""
    if not upper > lower + 1.0:
        raise ValueError("prior box narrower than the 1-dex shelf")
    return {"S": (lower, lower + 1.0), "P": (lower + 1.0, upper)}


def shelf_peak_indicators(chains, lower: float) -> dict[str, list[np.ndarray]]:
    """Per-chain indicators {"S": x <= lower + 1, "P": x > lower + 1}."""
    s = [c <= lower + 1.0 for c in _float_chains(chains)]
    return {"S": s, "P": [~v for v in s]}


def event_intervals(toas_mjd, window: tuple[float, float], t0_ref, min_mass: float = 0.01
                    ) -> list[tuple[float, float]]:
    """Inter-TOA gaps holding >= min_mass of the reference t0 draws (Sec. 5.3b).

    Gap boundaries are the window ends a, b plus the distinct TOA epochs strictly inside (a, b),
    sorted; gaps are half-open [lo, hi) except the last, [lo, b]. Reference mass = fraction of all
    reference draws (pooled, inside or outside the window) falling in the gap. The complementary
    "rest of window" bin is implicit (see ``event_interval_indicators``)."""
    a, b = map(float, window)
    t = np.unique(np.asarray(toas_mjd, np.float64))
    edges = np.concatenate([[a], t[(t > a) & (t < b)], [b]])
    ref = np.concatenate(_float_chains(t0_ref))
    out = []
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == edges.size - 2
        inside = (ref >= lo) & ((ref <= hi) if last else (ref < hi))
        if inside.mean() >= min_mass:
            out.append((float(lo), float(hi)))
    return out


def event_interval_indicators(t0_chains, intervals: Sequence[tuple[float, float]],
                              window: tuple[float, float]) -> dict[str, list[np.ndarray]]:
    """Per-chain indicators for each selected interval (keys "I0", "I1", ...) and the "rest" bin
    (everything not in a selected interval, including draws outside the window). Interval
    membership uses the same half-open convention as ``event_intervals`` (closed at b)."""
    b = float(window[1])
    cs = _float_chains(t0_chains)
    out: dict[str, list[np.ndarray]] = {}
    for j, (lo, hi) in enumerate(intervals):
        out[f"I{j}"] = [(c >= lo) & ((c <= hi) if hi == b else (c < hi)) for c in cs]
    any_in = [np.zeros(c.size, bool) for c in cs]
    for v in out.values():
        any_in = [x | y for x, y in zip(any_in, v)]
    out["rest"] = [~x for x in any_in]
    return out


# ----------------------------------------------------------------------------------------------
# Sec. 6.6 D9
# ----------------------------------------------------------------------------------------------


def envelope_quantile_bounds(x, alpha: float, p: float, lower_bound: float, upper_bound: float,
                             weights=None) -> tuple[float, float]:
    """Worst-case alpha-quantile bounds under F_p = (1 - p) F_0 + p G (Sec. 6.6 item 3):
    q_lo = F_0^-1((alpha - p) / (1 - p)) (prior lower bound if alpha <= p),
    q_hi = F_0^-1(alpha / (1 - p)) (prior upper bound if alpha / (1 - p) >= 1).
    F_0^-1 is the weighted empirical inverse CDF (``reweight.weighted_quantile``; pass support
    weights for HD-reweighted samples)."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p={p} outside [0, 1)")
    q_lo = lower_bound if alpha <= p else weighted_quantile(x, (alpha - p) / (1 - p), weights)
    a_hi = alpha / (1.0 - p)
    q_hi = upper_bound if a_hi >= 1.0 else weighted_quantile(x, a_hi, weights)
    return float(q_lo), float(q_hi)


def d9_quantity_survives(bounds_ours: tuple[float, float], bounds_ref: tuple[float, float],
                         se_d: float, m: float) -> dict:
    """Worst-case difference interval [q_lo,ours - q_hi,ref, q_hi,ours - q_lo,ref] widened by
    +- 1.645 SE_D; the conclusion survives iff it lies within [-m, m]."""
    lo = bounds_ours[0] - bounds_ref[1] - Z90 * se_d
    hi = bounds_ours[1] - bounds_ref[0] + Z90 * se_d
    return {"interval": (float(lo), float(hi)), "survives": bool(lo >= -m and hi <= m)}


@dataclass(frozen=True)
class D9Sample:
    """One sample set for a D9 quantity: draws x (pooled), optional weights, the quantile MCSE at
    alpha, and the prior box [lower_bound, upper_bound]."""

    x: np.ndarray
    lower_bound: float
    upper_bound: float
    mcse: float = 0.0
    weights: np.ndarray | None = None

    def bounds(self, alpha: float, p: float, shift: float = 0.0) -> tuple[float, float]:
        lo, hi = envelope_quantile_bounds(self.x, alpha, p, self.lower_bound, self.upper_bound,
                                          self.weights)
        return lo + shift, hi + shift


def _survives_at(ours: D9Sample, ref: D9Sample, alpha: float, p: float, se_d: float, m: float,
                 s_ours: float = 0.0, s_ref: float = 0.0) -> bool:
    return d9_quantity_survives(ours.bounds(alpha, p, s_ours), ref.bounds(alpha, p, s_ref),
                                se_d, m)["survives"]


def _bisect_pstar(fn, p_max: float, tol: float) -> float:
    if not fn(0.0):
        return 0.0
    if fn(p_max):
        return p_max
    lo, hi = 0.0, p_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return hi


def p_star(ours: D9Sample, ref: D9Sample, alpha: float, se_d: float, m: float,
           p_max: float = 0.5, tol: float = 1e-6) -> dict:
    """p* = smallest missing mass at which the D9 equivalence conclusion can change (Sec. 6.6).

    Survival is monotone non-increasing in p (q_lo decreases and q_hi increases with p), so
    bisection on [0, p_max] applies; p* = 0 if it fails already at p = 0, and p* = p_max (flagged
    ``capped``) if it survives at p_max. Monte Carlo uncertainty (chosen method): p* is recomputed
    with both quantile functions shifted by 1.645 x their own quantile MCSE in the adverse
    direction, i.e. ours by +d_o and ref by -d_r (pushing D up) and ours by -d_o and ref by +d_r
    (pushing D down); ``p_star_lo90`` = the smaller of the two (a conservative lower 90 % bound;
    the shift is constant in alpha, using the MCSE at alpha).
    """
    d_o, d_r = Z90 * ours.mcse, Z90 * ref.mcse

    def run(so: float, sr: float) -> float:
        return _bisect_pstar(lambda p: _survives_at(ours, ref, alpha, p, se_d, m, so, sr),
                             p_max, tol)

    ps = run(0.0, 0.0)
    lo90 = min(run(d_o, -d_r), run(-d_o, d_r))
    return {"p_star": ps, "p_star_lo90": lo90, "capped": ps >= p_max, "p_max": p_max,
            "survival_on_grid": {p: _survives_at(ours, ref, alpha, p, se_d, m) for p in D9_P_GRID}}


def epsilon_m(p_star_lo90: Iterable[float], grid: Sequence[float] = D9_P_GRID) -> float | None:
    """Largest grid value <= min over the model's headline quantities of the lower 90 % bound of
    p*; None (class unavailable) if that minimum is below the smallest grid value (0.001) or no
    quantities are given."""
    vals = [float(v) for v in p_star_lo90]
    if not all(np.isfinite(v) for v in vals):
        raise GateInputError(f"epsilon_m: non-finite p* bound in {vals}")
    if not vals:
        return None
    lim = min(vals)
    ok = [g for g in grid if g <= lim]
    return max(ok) if ok else None


def bf_sensitivity(lnb0: float, se: float, m: float, target: float, eps_curn: float | None,
                   eps_hd: float | None, grid: Sequence[float] = D9_BF_GRID,
                   max_se: float = E6_MAX_SE) -> dict:
    """Sec. 6.6 item 4. Table of ln B_full - ln B_0 = ln(1 - p_CURN) - ln(1 - p_HD) over the grid,
    keyed (p_CURN, p_HD), and the E-6 qualification: [ln B_0 - 1.645 SE + ln(1 - eps_CURN),
    ln B_0 + 1.645 SE - ln(1 - eps_HD)] within [target - m, target + m]. Also requires the E-6
    SE cap (SE <= 0.10, Sec. 6.2) and both epsilons available (not None)."""
    table = {(pc, ph): float(np.log1p(-pc) - np.log1p(-ph)) for pc in grid for ph in grid}
    if eps_curn is None or eps_hd is None:
        return {"table": table, "interval": None, "qualifies": False,
                "reason": "epsilon unavailable"}
    lo = lnb0 - Z90 * se + float(np.log1p(-eps_curn))
    hi = lnb0 + Z90 * se - float(np.log1p(-eps_hd))
    se_ok = bool(np.isfinite(se) and se <= max_se)
    q = bool(se_ok and lo >= target - m and hi <= target + m)
    return {"table": table, "interval": (lo, hi), "qualifies": q,
            "reason": None if q else ("SE above E-6 cap" if not se_ok else "outside E-6 margin")}


def d9_excluded_regions(acceptance: Mapping) -> list[Mapping]:
    """The frozen ``d9.excluded_regions`` list of an acceptance file, validated: each entry needs
    name, model, boundaries, reference_file, sha256, burn_in (GateInputError otherwise)."""
    require_fields(acceptance, ("d9",), "acceptance file")
    require_fields(acceptance["d9"], ("excluded_regions",), "acceptance file d9")
    regs = acceptance["d9"]["excluded_regions"]
    if not isinstance(regs, list) or not regs:
        raise GateInputError("d9.excluded_regions must be a non-empty list")
    for i, r in enumerate(regs):
        require_fields(r, ("name", "model", "boundaries", "reference_file", "sha256", "burn_in"),
                       f"d9.excluded_regions[{i}]")
    return regs


@dataclass(frozen=True)
class D9Result:
    """D9 outcome. ``unconditional_verdict`` is always INCONCLUSIVE; ``claim`` is the class name
    only when ``available`` (eligible and every epsilon_m >= 0.001), else None. Never REPRODUCED.
    """

    eligible: bool
    available: bool
    reasons: tuple[str, ...]
    separate_review: tuple[str, ...]
    epsilon: Mapping[str, float | None]
    claim: str | None
    unconditional_verdict: str = field(default=INCONCLUSIVE, init=False)
    template: str = field(default=D9_TEMPLATE, init=False)


def d9_eligibility(regions: Mapping[str, tuple[Occupancy, Occupancy]],
                   other_gate_items: Mapping[str, str],
                   headline: Mapping[str, tuple[str, bool]],
                   epsilon_by_model: Mapping[str, float | None], *,
                   required_models: Sequence[str],
                   required_checks: Sequence[str]) -> D9Result:
    """D9 eligibility (Sec. 6.6 items 2-3).

    regions: every listed region of U -> (reference occupancy, our occupancy) of retained draws.
      Eligible only if both are "zero-visit"; few-event or asymmetric visitation (visited by one
      set, not the other) -> not eligible, listed for separate review; visited by both -> not
      eligible (not a zero-visit region).
    other_gate_items: every other applicable check -> status; all must be PASS.
    headline: quantity -> (Sec. 6.1 status, decidable); every decidable one must be EQUIVALENT.
    epsilon_by_model: model -> epsilon_m (None = unavailable); availability needs all >= 0.001.
    required_models / required_checks: the complete inventories; a missing model or check, or a
    non-finite epsilon, raises GateInputError (fail closed).
    """
    if not regions:
        raise GateInputError("d9_eligibility: U lists no regions")
    if not required_models or not required_checks:
        raise GateInputError("d9_eligibility: the required model and check inventories must be given")
    if set(epsilon_by_model) != set(required_models):
        raise GateInputError(f"d9_eligibility: epsilon_m given for {sorted(epsilon_by_model)}, "
                             f"required {sorted(required_models)}")
    for mdl, e in epsilon_by_model.items():
        if e is not None and not (isinstance(e, (int, float, np.floating)) and np.isfinite(e)):
            raise GateInputError(f"d9_eligibility: epsilon_m[{mdl}] = {e!r} is not finite")
    missing = sorted(set(required_checks) - set(other_gate_items))
    if missing:
        raise GateInputError(f"d9_eligibility: required checks not reported: {missing}")
    reasons, review = [], []
    for name, (r, o) in regions.items():
        rz, oz = r.case == "zero-visit", o.case == "zero-visit"
        if rz and oz:
            continue
        if "few-event" in (r.case, o.case):
            review.append(name)
            reasons.append(f"{name}: few-event case (separate review)")
        elif rz != oz:
            review.append(name)
            reasons.append(f"{name}: asymmetric visitation (reference {r.case}, ours {o.case}; "
                           "separate review)")
        else:
            reasons.append(f"{name}: visited in both sample sets ({r.case}/{o.case}); not a "
                           "zero-visit region")
    for name, st in other_gate_items.items():
        if st != PASS:
            reasons.append(f"gate item {name}: {st}")
    n_dec = 0
    for name, (st, dec) in headline.items():
        if dec:
            n_dec += 1
            if st != EQUIVALENT:
                reasons.append(f"headline {name}: {st}")
    if n_dec == 0:
        reasons.append("no decidable headline quantities")
    eligible = not reasons
    unavailable = [mdl for mdl, e in epsilon_by_model.items() if e is None or e < D9_P_GRID[0]]
    if unavailable:
        reasons.append(f"epsilon_m < {D9_P_GRID[0]} for {unavailable}")
    available = eligible and not unavailable
    return D9Result(eligible, available, tuple(reasons), tuple(review), dict(epsilon_by_model),
                    D9_CLASS if available else None)
