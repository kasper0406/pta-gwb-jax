"""MCMC convergence diagnostics and Monte-Carlo-error-aware comparisons (numpy only).

Implements the rank-normalised split-R-hat, bulk-ESS and tail-ESS of Vehtari, Gelman, Simpson,
Carpenter & Buerkner (2021, Bayesian Analysis 16, 667), with the autocorrelation-based ESS
(Geyer initial monotone sequence) of Stan/ArviZ, plus the quantile MCSE of the same paper.
Inputs are draws shaped (chains, draws).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _autocov(x: np.ndarray) -> np.ndarray:
    """Autocovariance along the last axis (biased, FFT)."""
    n = x.shape[-1]
    x = x - x.mean(axis=-1, keepdims=True)
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m, axis=-1)
    ac = np.fft.irfft(f * np.conj(f), n=m, axis=-1)[..., :n] / n
    return ac


def ess(draws: np.ndarray) -> float:
    """Effective sample size of (chains, draws) via Geyer's initial monotone sequence."""
    x = np.atleast_2d(np.asarray(draws, np.float64))
    m, n = x.shape
    if n < 4:
        return float("nan")
    if np.ptp(x) == 0:
        return float(m * n)
    acov = _autocov(x)
    chain_mean = x.mean(axis=1)
    chain_var = acov[:, 0] * n / (n - 1)
    mean_var = chain_var.mean()
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += chain_mean.var(ddof=1)
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sum pairs while positive, enforce monotone decrease
    t = 0
    pair_sums = []
    while t + 1 < n:
        s = rho[t] + rho[t + 1]
        if s < 0:
            break
        pair_sums.append(s)
        t += 2
    pair_sums = np.minimum.accumulate(np.array(pair_sums)) if pair_sums else np.array([1.0])
    tau = -1.0 + 2.0 * np.sum(pair_sums)
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def _split(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, -n:]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return stats.norm.ppf((r - 3.0 / 8.0) / (x.size + 0.25))


def _rhat_basic(x: np.ndarray) -> float:
    _m, n = x.shape
    w = x.var(axis=1, ddof=1).mean()
    b = n * x.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * w + b / n
    return float(np.sqrt(var_plus / w)) if w > 0 else float("nan")


def rhat(draws: np.ndarray) -> float:
    """Rank-normalised split-R-hat (max of bulk and folded)."""
    x = _split(np.asarray(draws, np.float64))
    bulk = _rhat_basic(_rank_normalize(x))
    folded = np.abs(x - np.median(x))
    tail = _rhat_basic(_rank_normalize(folded))
    return max(bulk, tail)


def ess_bulk(draws: np.ndarray) -> float:
    return ess(_rank_normalize(_split(np.asarray(draws, np.float64))))


def ess_tail(draws: np.ndarray, prob: float = 0.05) -> float:
    """min over the 5% and 95% quantile indicators (split chains)."""
    x = _split(np.asarray(draws, np.float64))
    lo, hi = np.quantile(x, [prob, 1.0 - prob])
    return min(ess((x <= lo).astype(float)), ess((x <= hi).astype(float)))


def mcse_quantile(draws: np.ndarray, p: float) -> float:
    """Monte-Carlo standard error of the p-quantile (Vehtari et al. 2021, Sec. 4.3)."""
    x = np.asarray(draws, np.float64)
    x2 = np.atleast_2d(x)
    q = np.quantile(x2, p)
    S = ess(_split((x2 <= q).astype(float)))
    if not np.isfinite(S) or S <= 0:
        return float("nan")
    a, b = stats.beta.ppf([0.1586553, 0.8413447], S * p + 1, S * (1 - p) + 1)
    xs = np.sort(x2.ravel())
    n = xs.size
    th1 = xs[int(np.clip(np.ceil(a * n) - 1, 0, n - 1))]
    th2 = xs[int(np.clip(np.ceil(b * n) - 1, 0, n - 1))]
    return float((th2 - th1) / 2.0)


def summarize(draws: np.ndarray, probs=(0.05, 0.5, 0.95)) -> dict:
    """Quantiles with MCSE, R-hat, bulk/tail ESS for one parameter, draws (chains, n)."""
    x = np.atleast_2d(np.asarray(draws, np.float64))
    out = {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "rhat": rhat(x) if x.shape[0] > 1 else float("nan"),
        "ess_bulk": ess_bulk(x),
        "ess_tail": ess_tail(x),
    }
    for p in probs:
        out[f"q{round(100 * p):02d}"] = float(np.quantile(x, p))
        out[f"q{round(100 * p):02d}_mcse"] = mcse_quantile(x, p)
    return out


def ks_ess(a: np.ndarray, b: np.ndarray, ess_a: float, ess_b: float) -> dict:
    """Two-sample KS statistic with an ESS-based p-value (autocorrelated chains).

    The KS distance D is computed on all draws; its null distribution is evaluated with the
    effective sample sizes instead of the raw counts (n_eff = ess_a ess_b / (ess_a + ess_b)).
    This is an approximation: ESS is defined for means, so treat p as indicative.
    """
    d = stats.ks_2samp(np.ravel(a), np.ravel(b)).statistic
    ne = ess_a * ess_b / (ess_a + ess_b)
    p = float(stats.kstwobign.sf(d * np.sqrt(ne)))
    return {"D": float(d), "n_eff": float(ne), "p_ess": p}


def energy_distance(a: np.ndarray, b: np.ndarray, max_n: int = 4000, seed: int = 0) -> float:
    """Szekely-Rizzo energy distance between two (n, d) samples (thinned to max_n draws)."""
    rng = np.random.default_rng(seed)
    a = np.atleast_2d(np.asarray(a, np.float64).T).T
    b = np.atleast_2d(np.asarray(b, np.float64).T).T
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    a = a[rng.choice(len(a), min(max_n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(max_n, len(b)), replace=False)]

    def md(u, v):
        return np.mean(np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1))

    return float(2 * md(a, b) - md(a, a) - md(b, b))


# ---------------------------------------------------------------------- acceptance gate

GATE_DEFAULTS = {
    "rhat_max": 1.01,  # GWB App. B criterion, applied to every parameter (rank-normalised split R-hat)
    # Vehtari et al. (2021) recommend bulk and tail ESS > 400 before trusting R-hat and quantile
    # MCSEs; required for every parameter and for every available occupancy indicator.
    "ess_bulk_min": 400.0,
    "ess_tail_min": 400.0,
    "indicator_ess_min": 400.0,
    # an occupancy indicator with fewer than this many draws in its minority class is treated as
    # near-constant: its R-hat / ESS / SE are reported as unavailable, never as precise.
    "min_minority_count": 10,
    # Two regions per bin with hysteresis: "low" (log10_rho < region_low) and "high"
    # (log10_rho > region_high); a chain's region state is kept while it is in between.
    "region_low": -10.0,
    "region_high": -8.0,
    # a region is relevant when the reference puts at least this fraction of its draws in it
    # (used only when relevance is not declared explicitly)
    "region_min_mass": 0.005,
    # per relevant region (between-region transport): pooled entries and exits, chains with at
    # least one entry or exit, and the largest share of the region's draws held by one sojourn
    "region_min_events": 10,
    "region_min_chains": 2,
    "region_max_sojourn_frac": 0.5,
    # occupancy precision tied to the objective (reproduce the Fig. 1a probability masses to about
    # +-2 percentage points at 95%): conservative MCSE of every occupancy (the < threshold occupancy
    # of every bin and the occupancy of every relevant region) must not exceed this.
    "occupancy_mcse_max": 0.01,
    # reproduction agreement: |difference in occupancy| / SE (chain-as-unit batch means for our run,
    # contiguous batch means for the single-sequence reference) above this tolerance disagrees.
    # A conventional threshold, not a calibrated test.
    "agreement_z_tol": 3.5,
    # a comparison whose combined SE exceeds this cannot establish agreement (INCONCLUSIVE)
    "agreement_se_max": 0.01,
    "reference_batches": 20,
    # heuristic tail-stability warnings only (not part of any verdict; not calibrated)
    "heuristic_z_tol": 4.5,
}

REGIONS = ("low", "high")


class GateInputError(ValueError):
    """Missing or invalid input for the acceptance gate (CLI exit status 2)."""


def _indicator_stats(ind: np.ndarray, min_minority: int) -> dict:
    """R-hat / ESS of a 0/1 indicator (chains, n), or ``available: False`` when (near-)constant."""
    k = int(ind.sum())
    if min(k, ind.size - k) < min_minority:
        return {"available": False, "rhat": None, "ess": None}
    return {"available": True, "rhat": rhat(ind), "ess": ess(_split(ind))}


def _batch_means_se(ind: np.ndarray, n_batches: int) -> float:
    """SE of the mean of a 0/1 indicator from batch means. ``ind``: (chains, n). With >= 2 chains
    each chain is one batch (the chain as the unit); a single sequence is cut into ``n_batches``
    contiguous batches."""
    if ind.shape[0] >= 2:
        m = ind.mean(axis=1)
    else:
        m = np.array([b.mean() for b in np.array_split(ind[0], n_batches)])
    return float(m.std(ddof=1) / np.sqrt(len(m)))


def _conservative_se(ind: np.ndarray, st: dict, n_batches: int) -> float:
    """Never-zero SE of the occupancy (mean of a 0/1 indicator, (chains, n)).

    * available indicator: max(batch-means SE, binomial SE with the indicator ESS);
    * unavailable (near-constant, < min_minority_count draws in the rarer class): sparse-count
      treatment. The ESS is taken to be at most the number of independent units u (chains for a
      multi-chain run, ``n_batches`` contiguous batches for a single reference sequence) and the
      proportion is Jeffreys-smoothed, p~ = (k + 1/2) / (n + 1), giving
      SE = max(batch-means SE, sqrt(p~ (1 - p~) / u)). Deliberately conservative: a run whose
      chains barely visit a region cannot establish its probability.
    """
    se = _batch_means_se(ind, n_batches)
    f = float(ind.mean())
    if st["available"]:
        return max(se, float(np.sqrt(f * (1 - f) / st["ess"])))
    units = ind.shape[0] if ind.shape[0] >= 2 else n_batches
    pj = (float(ind.sum()) + 0.5) / (ind.size + 1.0)
    return max(se, float(np.sqrt(pj * (1 - pj) / units)))


def region_states(v: np.ndarray, region_low: float, region_high: float) -> np.ndarray:
    """Hysteresis region state per draw, (chains, n) int8: 0 = low (entered when x < region_low),
    1 = high (entered when x > region_high), kept while region_low <= x <= region_high; -1 before
    the chain first enters either region."""
    v = np.atleast_2d(np.asarray(v, np.float64))
    raw = np.full(v.shape, -1, np.int8)
    raw[v < region_low] = 0
    raw[v > region_high] = 1
    pos = np.maximum.accumulate(np.where(raw >= 0, np.arange(v.shape[1]), -1), axis=1)
    st = np.take_along_axis(raw, np.maximum(pos, 0), axis=1)
    st[pos < 0] = -1
    return st


def region_events(v: np.ndarray, region_low: float, region_high: float) -> dict:
    """Entries, exits and sojourns of the hysteresis region state of ``v`` (chains, n).

    Per region R in ("low", "high"): an entry is a completed transition of the state from the other
    region into R (the first entry from the undefined initial state is not counted), an exit a
    transition from R into the other region; a sojourn is a maximal run of consecutive draws with
    state R (including runs cut by the start or end of a chain). Returns, per region,
    ``entries_per_chain``, ``exits_per_chain``, ``entries``, ``exits``, ``chains_with_events``
    (chains with >= 1 entry or exit), ``sojourn_lengths`` (pooled, in chain order),
    ``draws_in_state`` and ``sojourns`` = {count, mean, max, max_frac} with max_frac = longest
    sojourn / draws_in_state (None if the region was never in state)."""
    st = region_states(v, region_low, region_high)
    prev, cur = st[:, :-1], st[:, 1:]
    out = {}
    for r, name in enumerate(REGIONS):
        o = 1 - r
        ent = ((prev == o) & (cur == r)).sum(axis=1)
        ex = ((prev == r) & (cur == o)).sum(axis=1)
        lengths: list[int] = []
        for row in st == r:
            d = np.diff(np.concatenate([[0], row.astype(np.int64), [0]]))
            lengths.extend((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist())
        tot = int(sum(lengths))
        out[name] = {
            "entries_per_chain": ent.tolist(), "exits_per_chain": ex.tolist(),
            "entries": int(ent.sum()), "exits": int(ex.sum()), "chains_with_events": int(np.sum(ent + ex > 0)),
            "sojourn_lengths": lengths, "draws_in_state": tot,
            "sojourns": {"count": len(lengths), "mean": float(np.mean(lengths)) if lengths else 0.0,
                         "max": int(max(lengths, default=0)), "max_frac": (max(lengths) / tot) if tot else None},
        }
    return out


def _region_indicator(v: np.ndarray, region: str, region_low: float, region_high: float) -> np.ndarray:
    return (v < region_low if region == "low" else v > region_high).astype(float)


def derive_relevant_regions(reference: dict, n_bins: int = 30, prefix: str = "gw_log10_rho_",
                            region_low: float = GATE_DEFAULTS["region_low"],
                            region_high: float = GATE_DEFAULTS["region_high"],
                            region_min_mass: float = GATE_DEFAULTS["region_min_mass"]) -> dict[str, list[str]]:
    """{bin: regions} with region R relevant when the reference puts >= region_min_mass of its draws
    in R (x < region_low resp. x > region_high). Raises GateInputError for a missing, non-numeric,
    non-finite or too short (< 100 draws) reference bin."""
    if not isinstance(reference, dict):
        raise GateInputError(f"relevance reference must be a dict, got {type(reference).__name__}")
    out = {}
    for k in range(n_bins):
        n = f"{prefix}{k}"
        if n not in reference:
            raise GateInputError(f"relevance reference lacks {n}")
        try:
            r = np.asarray(reference[n], np.float64).ravel()
        except (TypeError, ValueError) as e:
            raise GateInputError(f"relevance reference {n}: not numeric ({e})") from e
        if r.size < 100 or not np.all(np.isfinite(r)):
            raise GateInputError(f"relevance reference {n}: {r.size} draws, finite={bool(np.all(np.isfinite(r)))}")
        out[n] = [R for R in REGIONS if _region_indicator(r, R, region_low, region_high).mean() >= region_min_mass]
    return out


_INTEGER_CRITERIA = {"min_minority_count": 1, "reference_batches": 2,  # name -> minimum value
                     "region_min_events": 1, "region_min_chains": 1}
_UNIT_INTERVAL_CRITERIA = ("region_min_mass", "region_max_sojourn_frac")  # in (0, 1]
_SIGNED_CRITERIA = ("threshold", "region_low", "region_high")  # any finite real


def _is_real(v) -> bool:
    """Real numeric scalar (Python or NumPy int/float, not bool)."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def validate_gate_criteria(overrides: dict, threshold, n_bins, probs) -> list[str]:
    """Errors for unknown criterion keys or invalid values. Every criterion (defaults and
    overrides) must be a finite real scalar (Python or NumPy, not bool); integer criteria must be
    integral and >= their minimum; region_min_mass and region_max_sojourn_frac in (0, 1];
    region_low < threshold < region_high; other tolerances / ESS minimums / SE maximums must be
    positive; rhat_max > 1."""
    errs = [f"unknown criterion {k!r}" for k in overrides if k not in GATE_DEFAULTS]
    cfg = {**GATE_DEFAULTS, **{k: v for k, v in overrides.items() if k in GATE_DEFAULTS}}
    ok = {}
    for k, v in (*cfg.items(), ("threshold", threshold), ("n_bins", n_bins)):
        if not _is_real(v) or not np.isfinite(float(v)):
            errs.append(f"criterion {k} must be a finite real number, got {v!r} ({type(v).__name__})")
            continue
        v = float(v)
        ok[k] = v
        if k in _INTEGER_CRITERIA or k == "n_bins":
            lo = _INTEGER_CRITERIA.get(k, 1)
            if v != int(v) or v < lo:
                errs.append(f"criterion {k} must be an integer >= {lo}, got {v!r}")
        elif k == "rhat_max":
            if v <= 1.0:
                errs.append(f"rhat_max must be > 1, got {v!r}")
        elif k in _UNIT_INTERVAL_CRITERIA:
            if not 0.0 < v <= 1.0:
                errs.append(f"criterion {k} must be in (0, 1], got {v!r}")
        elif k not in _SIGNED_CRITERIA and v <= 0:
            errs.append(f"criterion {k} must be > 0, got {v!r}")
    if all(k in ok for k in _SIGNED_CRITERIA) and not ok["region_low"] < ok["threshold"] < ok["region_high"]:
        errs.append(f"need region_low < threshold < region_high, got {ok['region_low']!r}, "
                    f"{ok['threshold']!r}, {ok['region_high']!r}")
    try:
        pv = [float(p) for p in probs] if all(_is_real(p) for p in probs) else None
    except TypeError:
        pv = None
    if not pv or not all(0.0 < p < 1.0 for p in pv):
        errs.append(f"probs must be real numbers in (0, 1), got {probs!r}")
    return errs


def validate_relevant_regions(relevant_regions, n_bins: int = 30, prefix: str = "gw_log10_rho_") -> list[str]:
    """Errors for a relevance declaration: a dict naming every bin ``prefix0 .. prefix{n_bins-1}``
    exactly once (an explicit empty list declares no relevant region), each value a list / tuple of
    distinct strings from {"low", "high"}."""
    if relevant_regions is None:
        return []
    if not isinstance(relevant_regions, dict):
        return [f"relevant_regions must be a dict {{bin: [regions]}}, got {type(relevant_regions).__name__}"]
    bins = [f"{prefix}{k}" for k in range(n_bins)]
    errs = [f"relevant_regions: unknown bin {k!r}" for k in relevant_regions if k not in bins]
    errs += [f"relevant_regions: {b} not declared (use [] for no relevant region)" for b in bins
             if b not in relevant_regions]
    for k, v in relevant_regions.items():
        if not isinstance(v, (list, tuple)) or not all(isinstance(r, str) and r in REGIONS for r in v) \
                or len(set(v)) != len(v):
            errs.append(f"relevant_regions[{k!r}] must be distinct names from {list(REGIONS)}, got {v!r}")
    return errs


def validate_gate_inputs(x, names, expected_names=None, reference=None, compare: bool = False,
                         prefix: str = "gw_log10_rho_", n_bins: int = 30) -> list[str]:
    """Input errors (empty list if valid): shape/name consistency, unique names, the exact
    ``expected_names`` schema if given, all expected bins present, finite draws, and, if
    ``compare``, a complete, finite, non-trivial reference for every bin."""
    errs = []
    try:
        x = np.asarray(x, np.float64)
    except (TypeError, ValueError) as e:
        return [f"draws are not numeric: {e}"]
    if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
        return ["parameter names must be a list of strings"]
    if x.ndim != 3:
        return [f"draws must be (chains, draws, D), got shape {x.shape}"]
    C, N, D = x.shape
    if len(names) != D:
        errs.append(f"{len(names)} names for D = {D} parameters")
    if len(set(names)) != len(names):
        errs.append("parameter names are not unique")
    if expected_names is not None and list(names) != list(expected_names):
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        errs.append(f"parameter schema mismatch: {len(names)} given, {len(expected_names)} expected; "
                    f"missing {missing[:5]}{'...' if len(missing) > 5 else ''}, extra {extra[:5]}"
                    f"{' (order differs)' if not missing and not extra else ''}")
    for k in range(n_bins):
        if f"{prefix}{k}" not in names:
            errs.append(f"{prefix}{k}: missing")
    if C < 2 or N < 4:
        errs.append(f"need >= 2 chains and >= 4 draws, got {C} x {N}")
    if not np.all(np.isfinite(x)):
        errs.append(f"{int(np.sum(~np.isfinite(x)))} non-finite draws")
    if compare:
        if not reference:
            errs.append("reference requested but empty")
        else:
            for k in range(n_bins):
                r = reference.get(f"{prefix}{k}") if isinstance(reference, dict) else None
                if r is None:
                    errs.append(f"reference lacks {prefix}{k}")
                    continue
                try:
                    r = np.asarray(r, np.float64).ravel()
                except (TypeError, ValueError):
                    errs.append(f"reference {prefix}{k}: not numeric")
                    continue
                if r.size < 100 or not np.all(np.isfinite(r)):
                    errs.append(f"reference {prefix}{k}: {r.size} draws, finite={bool(np.all(np.isfinite(r)))}")
    return errs


def _compare(ind: np.ndarray, st: dict, rind: np.ndarray, rst: dict, cfg: dict) -> dict:
    """Occupancy comparison run vs single-sequence reference with conservative SEs."""
    se_o = _conservative_se(ind, st, cfg["reference_batches"])
    se_r = _conservative_se(rind, rst, cfg["reference_batches"])
    d = float(ind.mean()) - float(rind.mean())
    se = float(np.hypot(se_o, se_r))
    z = 0.0 if d == 0 else (d / se if se > 0 else float("inf"))
    # precision requirement on OUR SE only; the reference's SE enters z and is flagged (informational)
    return {"reference_occupancy": float(rind.mean()), "diff": d, "se_ours": se_o, "se_ref": se_r, "se": se,
            "z": float(z), "pass": bool(abs(z) <= cfg["agreement_z_tol"]), "precise": bool(se_o <= cfg["agreement_se_max"]),
            "reference_imprecise": bool(se_r > cfg["agreement_se_max"])}


def freespec_gate(
    x: np.ndarray,
    names: list[str],
    n_bins: int = 30,
    prefix: str = "gw_log10_rho_",
    threshold: float = -9.0,
    reference: dict | None = None,
    expected_names: list[str] | None = None,
    probs: tuple[float, ...] = (0.05, 0.5, 0.95),
    relevant_regions: dict[str, list[str]] | None = None,
    relevance_source: str | None = None,
    **overrides,
) -> dict:
    """Acceptance check for a free-spectrum run, with separate verdicts. Fails closed.

    ``x``: (chains, draws, D); ``names``: parameter names; ``expected_names``: the required
    schema (exact list); ``reference``: optional {bin name: 1-D reference draws} (e.g. the released
    core) for the reproduction comparison; ``relevant_regions``: predeclared {bin: subset of
    ["low", "high"]} naming every bin (``validate_relevant_regions``). If None, relevance is derived
    from ``reference`` (``derive_relevant_regions``: reference mass >= region_min_mass); with neither,
    relevance is undeclared and convergence cannot PASS. ``relevance_source`` labels the source in
    the output (default "declared" / "derived from reference").

    Regions (per bin): "low" = log10_rho < region_low, "high" = log10_rho > region_high, with a
    hysteresis state per chain (``region_states``) from which entries, exits and sojourns are
    counted (``region_events``). The region indicator used for occupancy, R-hat, ESS, MCSE and
    agreement is 1[x in R] (the scientific quantity); transport between the regions is assessed
    separately by the state-based entries / exits / sojourns.

    Returns ``input_errors`` (non-empty -> nothing else is evaluated; CLI exit 2), and

    * ``convergence`` PASS/FAIL/INCONCLUSIVE:
      - every parameter has rank-normalised split R-hat < rhat_max and bulk/tail ESS >= minimums;
      - every bin's occupancy indicator 1[log10_rho < threshold]: conservative MCSE <=
        occupancy_mcse_max; when available (not near-constant) R-hat < rhat_max and ESS >=
        indicator_ess_min; when unavailable, FAIL in a bin with a transport-assessed relevant
        region, a warning otherwise;
      - every relevant region R: if transport is assessed (the bin's other region is relevant too,
        or our run puts >= region_min_mass in it): pooled entries and exits >= region_min_events,
        chains with an entry or exit >= region_min_chains, longest sojourn <=
        region_max_sojourn_frac of the draws in state R, the region indicator available with R-hat
        < rhat_max and ESS >= indicator_ess_min, occupancy MCSE <= occupancy_mcse_max. If R is the
        bin's only relevant region (nothing to transport to): chains with draws in R >=
        region_min_chains, indicator R-hat / ESS when available, MCSE <= occupancy_mcse_max.
        Zero draws in R (or zero entries and exits when transport is assessed) is a FAIL with
        "no exploration evidence".
      - INCONCLUSIVE (never PASS) when nothing failed but relevance is undeclared.
    * ``reproduction_agreement`` PASS/FAIL/INCONCLUSIVE/UNAVAILABLE (UNAVAILABLE only without a
      reference): per bin the < threshold occupancy and the occupancy of every relevant region vs
      the reference, z = diff / sqrt(SE_ours^2 + SE_ref^2) with the conservative never-zero SEs of
      ``_conservative_se`` (batch means: our chains / contiguous batches of the reference; binomial
      SE from the indicator ESS; sparse-count SE for near-constant indicators). FAIL if any |z| >
      agreement_z_tol; else INCONCLUSIVE if our SE of any compared occupancy > agreement_se_max;
      else PASS. The reference's SE enters z only; comparisons where it exceeds agreement_se_max
      are flagged ``reference_imprecise`` (informational) and ``max_reference_se`` is reported: the
      reference's own precision (released core: up to ~0.010) limits how precisely agreement can
      be established. The
      reference's own split-R-hat, ESS, indicator diagnostics and region events (informational:
      does the single sequence itself meet the event criteria?) are recorded. A disagreement is a
      finding about the model / reference, not a sampler failure.
    * ``heuristic_warnings``: tail-stability contrasts (fraction of draws <= the pooled 5/50/95%
      quantile in each chain vs the rest and first vs second half, scaled by binomial SEs from the
      subsets' indicator ESS). Not calibrated under autocorrelation; never part of a verdict;
      unavailable for near-constant indicators.
    """
    cfg = {**GATE_DEFAULTS, **overrides}
    out = {"criteria": {k: repr(v) for k, v in cfg.items()} | {"threshold": repr(threshold), "n_bins": repr(n_bins),
                                                              "probs": repr(probs)}}
    errs = validate_gate_criteria(overrides, threshold, n_bins, probs)
    if not errs:
        errs = validate_gate_inputs(x, names, expected_names, reference, reference is not None, prefix, n_bins)
        errs += validate_relevant_regions(relevant_regions, int(n_bins), prefix)
        if relevance_source is not None and not isinstance(relevance_source, str):
            errs.append(f"relevance_source must be a string, got {type(relevance_source).__name__}")
    if errs:
        return out | {"input_errors": errs, "convergence": "UNAVAILABLE", "reproduction_agreement": "UNAVAILABLE",
                      "heuristic_warnings": []}
    cfg = {k: (int(v) if k in _INTEGER_CRITERIA else float(v)) for k, v in cfg.items()}
    threshold, n_bins, probs = float(threshold), int(n_bins), tuple(float(p) for p in probs)
    out["criteria"] = cfg | {"threshold": threshold, "n_bins": n_bins, "probs": list(probs)}
    lo, hi, nb, mm = cfg["region_low"], cfg["region_high"], cfg["reference_batches"], cfg["min_minority_count"]
    if relevant_regions is not None:
        rel = {k: list(v) for k, v in relevant_regions.items()}
        source = relevance_source or "declared"
    elif reference is not None:
        rel = derive_relevant_regions(reference, n_bins, prefix, lo, hi, cfg["region_min_mass"])
        source = relevance_source or f"derived from reference (mass >= {cfg['region_min_mass']})"
    else:
        rel, source = None, "undeclared"
    x = np.asarray(x, np.float64)
    C, N, _ = x.shape
    idx = {n: j for j, n in enumerate(names)}
    conv_fail, agree_fail, agree_imprecise, warnings, no_evidence, ref_se = [], [], [], [], [], []
    inconclusive = [] if rel is not None else ["region relevance not declared (no relevant_regions and no reference)"]
    params = {}
    for j, n in enumerate(names):
        s = {"rhat": rhat(x[..., j]), "ess_bulk": ess_bulk(x[..., j]), "ess_tail": ess_tail(x[..., j])}
        s["pass"] = bool(np.isfinite(list(s.values())).all() and s["rhat"] < cfg["rhat_max"]
                         and s["ess_bulk"] >= cfg["ess_bulk_min"] and s["ess_tail"] >= cfg["ess_tail_min"])
        params[n] = s
        if not s["pass"]:
            conv_fail.append(f"{n}: R-hat {s['rhat']:.4f}, bulk ESS {s['ess_bulk']:.0f}, tail ESS {s['ess_tail']:.0f}")
    bins, regions, ref_diag = {}, {}, {}
    h = N // 2
    for k in range(n_bins):
        n = f"{prefix}{k}"
        v = x[..., idx[n]]
        r = np.asarray(reference[n], np.float64).ravel() if reference is not None else None
        bin_fail = not params[n]["pass"]
        # ---- regions
        ev = region_events(v, lo, hi)
        occ_run = {R: float(_region_indicator(v, R, lo, hi).mean()) for R in REGIONS}
        regions[n] = {}
        for R in REGIONS:
            other = "high" if R == "low" else "low"
            ind = _region_indicator(v, R, lo, hi)
            st = _indicator_stats(ind, mm)
            mcse = _conservative_se(ind, st, nb)
            e = ev[R]
            relevant = None if rel is None else (R in rel[n])
            transport = bool(relevant and (other in rel[n] or occ_run[other] >= cfg["region_min_mass"]))
            rep = {"relevant": relevant, "transport_assessed": transport, "occupancy": occ_run[R],
                   "occupancy_per_chain": ind.mean(axis=1).tolist(), "draws": int(ind.sum()),
                   "chains_visiting": int(np.sum(ind.any(axis=1))),
                   "entries_per_chain": e["entries_per_chain"], "exits_per_chain": e["exits_per_chain"],
                   "entries": e["entries"], "exits": e["exits"], "chains_with_events": e["chains_with_events"],
                   "draws_in_state": e["draws_in_state"], "sojourns": e["sojourns"], "indicator": st, "mcse": mcse,
                   "failures": [], "pass": None}
            if relevant:
                f = rep["failures"]
                bounds = f"x < {lo:g}" if R == "low" else f"x > {hi:g}"
                if rep["draws"] == 0 or (transport and e["entries"] + e["exits"] == 0):
                    f.append(f"no exploration evidence ({rep['draws']} draws, {e['entries']} entries / {e['exits']} exits)")
                    no_evidence.append(f"{n}/{R}")
                else:
                    if transport:
                        if e["entries"] < cfg["region_min_events"]:
                            f.append(f"entries {e['entries']} < {cfg['region_min_events']}")
                        if e["exits"] < cfg["region_min_events"]:
                            f.append(f"exits {e['exits']} < {cfg['region_min_events']}")
                        if e["chains_with_events"] < cfg["region_min_chains"]:
                            f.append(f"chains with an entry/exit {e['chains_with_events']} < {cfg['region_min_chains']}")
                        mf = e["sojourns"]["max_frac"]
                        if mf is None or mf > cfg["region_max_sojourn_frac"]:
                            f.append(f"longest sojourn holds {mf:.2f} > {cfg['region_max_sojourn_frac']} of the region's draws"
                                     if mf is not None else "no sojourn")
                        if not st["available"]:
                            f.append(f"region indicator unavailable (minority count < {mm})")
                    elif rep["chains_visiting"] < cfg["region_min_chains"]:
                        f.append(f"chains visiting {rep['chains_visiting']} < {cfg['region_min_chains']}")
                    if st["available"] and not (st["rhat"] < cfg["rhat_max"] and st["ess"] >= cfg["indicator_ess_min"]):
                        f.append(f"region indicator R-hat {st['rhat']:.3f}, ESS {st['ess']:.0f}")
                    if not mcse <= cfg["occupancy_mcse_max"]:
                        f.append(f"occupancy MCSE {mcse:.4f} > {cfg['occupancy_mcse_max']}")
                rep["pass"] = not f
                if f:
                    bin_fail = True
                    conv_fail.append(f"{n} region {R} ({bounds}): " + "; ".join(f))
            if r is not None and relevant:
                rind = _region_indicator(r, R, lo, hi)[None]
                rep["agreement"] = _compare(ind, st, rind, _indicator_stats(rind, mm), cfg)
                a = rep["agreement"]
                if not a["pass"]:
                    agree_fail.append(f"{n} region {R}: occupancy {rep['occupancy']:.4f} vs reference "
                                      f"{a['reference_occupancy']:.4f} (z {a['z']:.1f})")
                if not a["precise"]:
                    agree_imprecise.append(f"{n} region {R}: our SE {a['se_ours']:.4f} > {cfg['agreement_se_max']}")
                ref_se.append((a["se_ref"], f"{n}/{R}"))
            regions[n][R] = rep
        bin_transport = any(regions[n][R]["transport_assessed"] for R in REGIONS)
        # ---- the < threshold occupancy indicator
        ind = (v < threshold).astype(float)
        st = _indicator_stats(ind, mm)
        b = {"occupancy_per_chain": ind.mean(axis=1).tolist(), "occupancy": float(ind.mean()), "indicator": st,
             "occupancy_mcse": _conservative_se(ind, st, nb), "parameter_pass": params[n]["pass"],
             "relevant_regions": None if rel is None else rel[n]}
        if st["available"]:
            b["indicator_pass"] = bool(st["rhat"] < cfg["rhat_max"] and st["ess"] >= cfg["indicator_ess_min"])
            if not b["indicator_pass"]:
                conv_fail.append(f"{n}: occupancy indicator R-hat {st['rhat']:.3f}, ESS {st['ess']:.0f}")
        elif bin_transport:
            b["indicator_pass"] = False
            conv_fail.append(f"{n}: occupancy indicator 1[x < {threshold:g}] unavailable ({int(ind.sum())} draws below) "
                             f"in a bin with relevant regions {rel[n]}")
        else:
            b["indicator_pass"] = True
            warnings.append(f"{n}: occupancy indicator near-constant (occupancy {b['occupancy']:.4f}); its diagnostics are unavailable")
        b["mcse_pass"] = bool(b["occupancy_mcse"] <= cfg["occupancy_mcse_max"])
        if not b["mcse_pass"]:
            conv_fail.append(f"{n}: occupancy (x < {threshold:g}) MCSE {b['occupancy_mcse']:.4f} > {cfg['occupancy_mcse_max']}")
        b["convergence_pass"] = not (bin_fail or not b["indicator_pass"] or not b["mcse_pass"])
        if r is not None:
            rind = (r < threshold).astype(float)[None]
            rst = _indicator_stats(rind, mm)
            r2 = r[None]
            rev = region_events(r2, lo, hi)
            ref_diag[n] = {"n": int(r.size), "split_rhat": rhat(r2), "ess_bulk": ess_bulk(r2), "ess_tail": ess_tail(r2),
                           "occupancy": float(rind.mean()), "indicator": rst,
                           "region_mass": {R: float(_region_indicator(r, R, lo, hi).mean()) for R in REGIONS},
                           "regions": {}}
            for R in REGIONS:
                ee = rev[R]
                mf = ee["sojourns"]["max_frac"]
                # informational; None unless the region is transport-assessed in our run's evaluation
                meets = None
                if regions[n][R]["transport_assessed"]:
                    meets = bool(ee["entries"] >= cfg["region_min_events"] and ee["exits"] >= cfg["region_min_events"]
                                 and mf is not None and mf <= cfg["region_max_sojourn_frac"])
                ref_diag[n]["regions"][R] = {"entries": ee["entries"], "exits": ee["exits"], "sojourns": ee["sojourns"],
                                             "meets_event_criteria": meets}
            b["agreement"] = _compare(ind, st, rind, rst, cfg)
            a = b["agreement"]
            if not a["pass"]:
                agree_fail.append(f"{n}: occupancy {b['occupancy']:.3f} vs reference {a['reference_occupancy']:.3f} (z {a['z']:.1f})")
            if not a["precise"]:
                agree_imprecise.append(f"{n}: occupancy (x < {threshold:g}) our SE {a['se_ours']:.4f} > {cfg['agreement_se_max']}")
            ref_se.append((a["se_ref"], f"{n}/x<{threshold:g}"))
        tails = []
        pooled_ok = st["available"]
        for p in probs:
            q = float(np.quantile(v, p))
            pairs = [(f"chain{c}_vs_rest", v[c : c + 1], np.delete(v, c, axis=0)) for c in range(C)]
            pairs.append(("first_vs_second_half", v[:, :h], v[:, h:]))
            for label, a_, bb in pairs:
                ia, ib = (a_ <= q).astype(float), (bb <= q).astype(float)
                sa = _indicator_stats(ia, mm)
                sb = _indicator_stats(ib, mm)
                t = {"test": label, "p": p, "F_a": float(ia.mean()), "F_b": float(ib.mean())}
                if not (pooled_ok and sa["available"] and sb["available"]):
                    t["z"] = None  # unavailable
                else:
                    var = [max(f * (1 - f), 1e-12) / e for f, e in ((t["F_a"], sa["ess"]), (t["F_b"], sb["ess"]))]
                    t["z"] = float((t["F_a"] - t["F_b"]) / np.sqrt(sum(var)))
                    if abs(t["z"]) > cfg["heuristic_z_tol"]:
                        warnings.append(f"{n}: tail-stability heuristic {label} at p={p}: z {t['z']:.1f}")
                tails.append(t)
        b["tail_heuristics"] = tails
        bins[n] = b
    convergence = "FAIL" if conv_fail else ("INCONCLUSIVE" if inconclusive else "PASS")
    if reference is None:
        agreement = "UNAVAILABLE"
    else:
        agreement = "FAIL" if agree_fail else ("INCONCLUSIVE" if agree_imprecise else "PASS")
    return out | {
        "input_errors": [],
        "convergence": convergence,
        "reproduction_agreement": agreement,
        "reproduction_acceptance": convergence == "PASS" and agreement == "PASS",
        "relevant_regions": rel,
        "relevance_source": source,
        "convergence_failures": conv_fail,
        "convergence_inconclusive": inconclusive,
        "no_exploration_evidence": no_evidence,
        "agreement_failures": agree_fail,
        "agreement_imprecise": agree_imprecise,
        # the reference's own occupancy SE limits how precisely agreement can be established
        "max_reference_se": None if not ref_se else {"se": max(ref_se)[0], "quantity": max(ref_se)[1]},
        "n_reference_imprecise": int(sum(se > cfg["agreement_se_max"] for se, _ in ref_se)),
        "heuristic_warnings": warnings,
        "parameters": params,
        "bins": bins,
        "regions": regions,
        "reference_diagnostics": ref_diag,
        "n_parameters_failing": int(sum(not s["pass"] for s in params.values())),
        "n_bins_failing_convergence": int(sum(not bb["convergence_pass"] for bb in bins.values())),
        "n_regions_failing": int(sum(rr["pass"] is False for rb in regions.values() for rr in rb.values())),
        "n_bins_disagreeing": len({s.split(":")[0].split(" ")[0] for s in agree_fail}),
    }


def gate_exit_code(result: dict, require_reproduction: bool = True) -> int:
    """0: convergence PASS and (when required) reproduction agreement PASS; 1: a verdict is FAIL or
    INCONCLUSIVE; 2: missing/invalid input, or reproduction UNAVAILABLE when required."""
    if result.get("input_errors"):
        return 2
    if result["convergence"] != "PASS":
        return 1
    if require_reproduction:
        if result["reproduction_agreement"] == "UNAVAILABLE":
            return 2
        if result["reproduction_agreement"] != "PASS":
            return 1
    return 0
