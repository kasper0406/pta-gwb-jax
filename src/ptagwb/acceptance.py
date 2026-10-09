"""M3b pre-registered acceptance logic (plan Sec. 5.3, 6.1, 6.2, 6.6). Pure numpy/scipy, fail closed.

Contents
--------
* Sec. 6.1 decision rule: ``classify``, ``max_our_mcse``, ``decidable``; Sec. 6.2 ``classify_e6``.
* Sec. 5.3 transport gate: region occupancy intervals (``occupancy``), support classes,
  the per-(parameter, region pair) ``transport_gate`` with the UNRESOLVED precedence rule, shelf /
  peak regions, event-epoch intervals, and ``aggregate``.
* Sec. 6.6 D9, as revised by the user on 2026-10-09: the common-domain conditional comparison
  (frozen exclusions, domain indicators, conditional quantiles with ordered-chain MCSE,
  eligibility). The earlier p* / epsilon_m machinery is retired (no uncalibrated coverage claim);
  the mixture envelope and the BF domain correction remain as descriptive reports only. The
  unconditional verdict stays INCONCLUSIVE, always.

Statuses are the strings "PASS" / "FAIL" / "INCONCLUSIVE" (gate) and "EQUIVALENT" /
"INCOMPATIBLE" / "INCONCLUSIVE" (quantities). Functions consuming frozen acceptance-file entries
raise ``ptagwb.diagnostics.GateInputError`` when a required field is missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence  # noqa: F401
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
D9_ENVELOPE_GRID: Final = (0.001, 0.01, 0.05, 0.10)  # descriptive envelope only
D9_CLASS: Final = "CONDITIONALLY EQUIVALENT ON THE COMMON DOMAIN D"

# Revised D9 (user decision 2026-10-09, review of 2ee1bb7); frozen with the acceptance file.
D9_TEMPLATE: Final = (
    "**CONDITIONALLY EQUIVALENT ON THE COMMON DOMAIN D.** Every predeclared headline quantity, "
    "computed for the CURN and HD posteriors conditional on the frozen common domain D (the "
    "complement of the union U of the predeclared zero- and few-event exclusions listed below, "
    "identical for both models under the CURN<->HD parameter mapping), meets the equivalence "
    "criteria against the released chains conditioned on the same D. The retained draws in U are "
    "kept and reported for both sample sets (reference: {n_ref_curn} CURN, {n_ref_hd} HD; ours: "
    "{n_ours_curn} CURN, {n_ours_hd} HD). The posterior mass of U is not established by either "
    "sample set, and nothing is claimed about the unconditional posterior; the missing-mass "
    "envelope reported below is descriptive. This is not the zero-visit claim of the original "
    "Sec. 6.6, and it is weaker than REPRODUCED; the unconditional verdict remains INCONCLUSIVE."
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
    """Classify one quantity against a frozen acceptance entry. Acceptance files v2+ hold the
    conditional reference on the common domain D (``q_ref_D``, ``mcse_ref_D``, ``m``); v1 rows
    (``q_ref``, ``mcse_ref``, ``m``) are still read. A row carrying both, neither, or a
    non-finite value raises GateInputError (fail closed)."""
    v2 = "q_ref_D" in entry or "mcse_ref_D" in entry
    keys = ("q_ref_D", "mcse_ref_D", "m") if v2 else ("q_ref", "mcse_ref", "m")
    require_fields(entry, keys)
    if v2 and ("q_ref" in entry or "mcse_ref" in entry):
        raise GateInputError("acceptance row mixes conditional (v2) and unconditional (v1) references")
    vals = [float(entry[k]) for k in keys]
    if not all(np.isfinite(vals)) or not np.isfinite(q_ours) or not np.isfinite(mcse_ours):
        raise GateInputError(f"non-finite input in {keys} / ours")
    return classify(q_ours - vals[0], mcse_ours, vals[1], vals[2])


def classify_e6_from_frozen(entry: Mapping, lnb_d_ours: float, se_ours: float) -> Classification:
    """E-6 against the frozen same-domain reference (acceptance v3): ln B_D of the released chains
    with its MCSE. SE_ours > max_se (or non-finite) -> INCONCLUSIVE (ours-limited); otherwise the
    Sec. 6.1 rule with SE_D = sqrt(SE_ours^2 + MCSE_ref^2)."""
    require_fields(entry, ("reference_lnB_D", "mcse_ref", "m", "max_se"), "E6 entry")
    ref, mref, m, mx = (float(entry[k]) for k in ("reference_lnB_D", "mcse_ref", "m", "max_se"))
    if not all(np.isfinite([ref, mref, m, mx, lnb_d_ours])):
        raise GateInputError("non-finite E-6 input")
    d = lnb_d_ours - ref
    if not np.isfinite(se_ours) or se_ours > mx:
        return Classification(INCONCLUSIVE, d, float(se_ours), (d - Z90 * se_ours, d + Z90 * se_ours), m,
                              "ours-limited")
    return classify(d, se_ours, mref, m)


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
# Sec. 6.6 D9, revised by the user on 2026-10-09: common-domain conditional comparison
# ----------------------------------------------------------------------------------------------


def _region_excludes(x: np.ndarray, ex: Mapping) -> np.ndarray:
    if not (np.all(np.isfinite(ex["boundaries"])) and ex["boundaries"][0] < ex["boundaries"][1]):
        raise GateInputError(f"exclusion {ex.get('name')}: non-finite or unordered boundaries")
    """True where x lies in the excluded region of one exclusion entry:
    ``type="shelf"``: lo <= x <= hi (closed, the 1-dex shelf); ``type="outside"``: x not in the
    kept half-open interval [lo, hi) (the dip-epoch "rest" region)."""
    lo, hi = (float(v) for v in ex["boundaries"])
    if ex["type"] == "shelf":
        return (x >= lo) & (x <= hi)
    if ex["type"] == "outside":
        return ~((x >= lo) & (x < hi))
    raise GateInputError(f"unknown exclusion type {ex['type']!r}")


def d9_exclusions(acceptance: Mapping) -> list[Mapping]:
    """The frozen common-domain exclusion list of an acceptance file, validated (GateInputError on a
    missing field): each entry has name, type, boundaries, params (model -> parameter name, for
    every model of the comparison), reference_cases (model -> case) and the reference identity
    (file, sha256, burn-in)."""
    require_fields(acceptance, ("d9",), "acceptance file")
    require_fields(acceptance["d9"], ("exclusions", "models", "reference"), "acceptance file d9")
    regs = acceptance["d9"]["exclusions"]
    models = list(acceptance["d9"]["models"])
    if not isinstance(regs, list) or not regs or not models:
        raise GateInputError("d9.exclusions and d9.models must be non-empty")
    for i, r in enumerate(regs):
        require_fields(r, ("name", "type", "boundaries", "params", "reference_cases"), f"d9.exclusions[{i}]")
        require_fields(r["params"], models, f"d9.exclusions[{i}].params")
        b = r["boundaries"]
        if r["type"] not in ("shelf", "outside") or len(b) != 2:
            raise GateInputError(f"d9.exclusions[{i}]: bad type/boundaries")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in b) or not b[0] < b[1]:
            raise GateInputError(f"d9.exclusions[{i}]: boundaries {b!r} must be finite and ordered (lo < hi)")
    return regs


def domain_indicator(column, exclusions: Sequence[Mapping], model: str) -> list[np.ndarray]:
    """Per-chain boolean indicator of the common domain D = complement of the union of the
    exclusions. ``column(name)`` returns the per-chain draws (list of 1-D arrays, chain order
    preserved) of parameter ``name`` of ``model``; each exclusion maps to its parameter through
    ``params[model]`` (the CURN<->HD mapping)."""
    ind = None
    for ex in exclusions:
        chains = _float_chains(column(ex["params"][model]))
        out = [~_region_excludes(c, ex) for c in chains]
        ind = out if ind is None else [a & b for a, b in zip(ind, out, strict=True)]
    if ind is None:
        raise GateInputError("no exclusions")
    return ind


def conditional_quantile(x, in_domain, p: float, log_w=None) -> dict:
    """Quantile of the posterior conditional on D and its ratio-estimator MCSE: weights
    w_i 1[x_i in D] (w = 1, or raw importance weights exp(log_w) for HD from CURN), on the ordered
    chains, so the MCSE keeps the serial dependence of both the draws and the domain indicator
    (``reweight.weighted_quantile_mcse`` with ``mask``)."""
    from ptagwb.reweight import weighted_quantile_mcse

    xs = _float_chains(x)
    lw = [np.zeros_like(c) for c in xs] if log_w is None else _float_chains(log_w)
    return weighted_quantile_mcse(xs, lw, p, mask=[np.asarray(m, bool) for m in _as_list(in_domain)])


def _as_list(x):
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return [x]
    return list(x)


def envelope_quantile_bounds(x, alpha: float, p: float, lower_bound: float, upper_bound: float,
                             weights=None) -> tuple[float, float]:
    """DESCRIPTIVE missing-mass envelope (no coverage claim): with F_p = (1 - p) F_0 + p G,
    q_lo = F_0^-1((alpha - p) / (1 - p)) (prior lower bound if alpha <= p) and
    q_hi = F_0^-1(alpha / (1 - p)) (prior upper bound if alpha / (1 - p) >= 1), F_0^-1 the
    weighted empirical inverse CDF. Reported only; never used to qualify a claim."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p={p} outside [0, 1)")
    q_lo = lower_bound if alpha <= p else weighted_quantile(x, (alpha - p) / (1 - p), weights)
    a_hi = alpha / (1.0 - p)
    q_hi = upper_bound if a_hi >= 1.0 else weighted_quantile(x, a_hi, weights)
    return float(q_lo), float(q_hi)


def descriptive_envelope(x_ours, x_ref, alpha: float, lower_bound: float, upper_bound: float,
                         w_ours=None, w_ref=None, grid: Sequence[float] = D9_ENVELOPE_GRID) -> dict:
    """Descriptive table: worst-case difference interval [q_lo,ours - q_hi,ref, q_hi,ours - q_lo,ref]
    of the alpha-quantile if a mass p outside D (each sample set separately) were placed adversely,
    for p in the grid. No Monte Carlo uncertainty is attached and no conclusion is drawn from it."""
    out = {}
    for p in grid:
        bo = envelope_quantile_bounds(x_ours, alpha, p, lower_bound, upper_bound, w_ours)
        br = envelope_quantile_bounds(x_ref, alpha, p, lower_bound, upper_bound, w_ref)
        out[p] = (bo[0] - br[1], bo[1] - br[0])
    return {"label": "descriptive (no coverage claim)", "alpha": alpha, "intervals": out}


def bf_domain_correction(p_curn: float, p_hd: float) -> float:
    """Descriptive: ln B_full - ln B_D = ln(1 - p_CURN) - ln(1 - p_HD) for posterior masses p_m of
    U (identity for a common domain with shared priors; review r5 item 4). No coverage claim."""
    for v in (p_curn, p_hd):
        if not (np.isfinite(v) and 0.0 <= v < 1.0):
            raise GateInputError(f"bf_domain_correction: mass {v!r} outside [0, 1)")
    return float(np.log1p(-p_curn) - np.log1p(-p_hd))


@dataclass(frozen=True)
class D9Result:
    """Outcome of the revised D9 class. ``unconditional_verdict`` is always INCONCLUSIVE; ``claim``
    is the class name only when ``eligible``. Never REPRODUCED."""

    eligible: bool
    reasons: tuple[str, ...]
    excluded_draws: Mapping[str, Mapping[str, int]]
    claim: str | None
    unconditional_verdict: str = field(default=INCONCLUSIVE, init=False)
    template: str = field(default=D9_TEMPLATE, init=False)


def d9_eligibility(other_gate_items: Mapping[str, str], headline: Mapping[str, str], *,
                   required_checks: Sequence[str], required_headline: Sequence[str],
                   excluded_draws: Mapping[str, Mapping[str, int]], required_models: Sequence[str]) -> D9Result:
    """Revised D9 (common domain): eligible iff every required check is reported and PASS (model
    identity, numerical validation, convergence within D, reweighting diagnostics on D), every
    frozen headline quantity is reported and EQUIVALENT (conditional on D), and the number of
    retained draws in U is reported for the reference and our run of every required model (kept
    and reported, any value). Missing inventories, a missing check or headline quantity, or a
    non-integer / negative count raise GateInputError (fail closed)."""
    if not required_checks or not required_headline or not required_models:
        raise GateInputError("d9_eligibility: required check, headline and model inventories must be given")
    miss = sorted(set(required_checks) - set(other_gate_items))
    if miss:
        raise GateInputError(f"d9_eligibility: required checks not reported: {miss}")
    miss = sorted(set(required_headline) - set(headline))
    if miss:
        raise GateInputError(f"d9_eligibility: headline quantities not reported: {miss}")
    if set(excluded_draws) != set(required_models):
        raise GateInputError(f"d9_eligibility: excluded-draw counts for {sorted(excluded_draws)}, "
                             f"required {sorted(required_models)}")
    for mdl, d in excluded_draws.items():
        for who in ("reference", "ours"):
            v = d.get(who) if isinstance(d, Mapping) else None
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0:
                raise GateInputError(f"d9_eligibility: excluded_draws[{mdl}][{who}] = {v!r}")
    reasons = [f"gate item {k}: {other_gate_items[k]}" for k in required_checks if other_gate_items[k] != PASS]
    reasons += [f"headline {k}: {headline[k]}" for k in required_headline if headline[k] != EQUIVALENT]
    ok = not reasons
    return D9Result(ok, tuple(reasons), {k: dict(v) for k, v in excluded_draws.items()}, D9_CLASS if ok else None)
