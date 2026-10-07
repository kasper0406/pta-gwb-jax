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
    # reproduction agreement: |difference in occupancy| / SE (chain-as-unit batch means for our run,
    # contiguous batch means for the single-sequence reference) above this tolerance disagrees.
    # A conventional threshold, not a calibrated test.
    "agreement_z_tol": 3.5,
    "reference_batches": 20,
    # heuristic tail-stability warnings only (not part of any verdict; not calibrated)
    "heuristic_z_tol": 4.5,
}


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


def freespec_gate(
    x: np.ndarray,
    names: list[str],
    n_bins: int = 30,
    prefix: str = "gw_log10_rho_",
    threshold: float = -9.0,
    reference: dict | None = None,
    expected_names: list[str] | None = None,
    probs: tuple[float, ...] = (0.05, 0.5, 0.95),
    **overrides,
) -> dict:
    """Acceptance check for a free-spectrum run, with separate verdicts.

    ``x``: (chains, draws, D); ``names``: parameter names; ``expected_names``: the required
    schema (exact list); ``reference``: optional {bin name: 1-D reference draws} (e.g. the released
    core) for the reproduction comparison.

    Returns ``input_errors`` (non-empty -> nothing else is evaluated; CLI exit 2), and

    * ``convergence`` PASS/FAIL: every parameter has rank-normalised split R-hat < rhat_max and
      bulk/tail ESS >= the minimums; every bin's occupancy indicator 1[log10_rho < threshold], when
      available (not near-constant), has R-hat < rhat_max and ESS >= indicator_ess_min.
    * ``reproduction_agreement`` PASS/FAIL/UNAVAILABLE (UNAVAILABLE only without a reference;
      sparse bins use the never-zero sparse-count SE of ``_conservative_se``): per bin, |occupancy - reference
      occupancy| / sqrt(SE_ours^2 + SE_ref^2) <= agreement_z_tol; each SE is the larger of a
      batch-means SE (our chains as units; contiguous batches of the single reference sequence)
      and the binomial SE from the indicator ESS (when available). The reference's own split-R-hat,
      ESS and occupancy-indicator diagnostics are recorded. A disagreement is a finding about the
      model / reference, not a sampler failure.
    * ``heuristic_warnings``: tail-stability contrasts (fraction of draws <= the pooled 5/50/95%
      quantile in each chain vs the rest and first vs second half, scaled by binomial SEs from the
      subsets' indicator ESS). Not calibrated under autocorrelation; never part of a verdict;
      unavailable for near-constant indicators.
    """
    cfg = {**GATE_DEFAULTS, **overrides}
    out = {"criteria": cfg | {"threshold": threshold, "n_bins": n_bins, "probs": list(probs)}}
    errs = validate_gate_inputs(x, names, expected_names, reference, reference is not None, prefix, n_bins)
    for key, val in (("threshold", threshold), *((k, v) for k, v in cfg.items() if isinstance(v, (int, float)))):
        if not (isinstance(val, (int, float, np.floating, np.integer)) and np.isfinite(val)):
            errs.append(f"{key} must be finite, got {val!r}")
    if errs:
        return out | {"input_errors": errs, "convergence": "UNAVAILABLE", "reproduction_agreement": "UNAVAILABLE",
                      "heuristic_warnings": []}
    x = np.asarray(x, np.float64)
    C, N, _ = x.shape
    idx = {n: j for j, n in enumerate(names)}
    conv_fail, agree_fail, warnings = [], [], []
    params = {}
    for j, n in enumerate(names):
        s = {"rhat": rhat(x[..., j]), "ess_bulk": ess_bulk(x[..., j]), "ess_tail": ess_tail(x[..., j])}
        s["pass"] = bool(np.isfinite(list(s.values())).all() and s["rhat"] < cfg["rhat_max"]
                         and s["ess_bulk"] >= cfg["ess_bulk_min"] and s["ess_tail"] >= cfg["ess_tail_min"])
        params[n] = s
        if not s["pass"]:
            conv_fail.append(f"{n}: R-hat {s['rhat']:.4f}, bulk ESS {s['ess_bulk']:.0f}, tail ESS {s['ess_tail']:.0f}")
    bins, ref_diag = {}, {}
    h = N // 2
    for k in range(n_bins):
        n = f"{prefix}{k}"
        v = x[..., idx[n]]
        ind = (v < threshold).astype(float)
        st = _indicator_stats(ind, cfg["min_minority_count"])
        b = {"occupancy_per_chain": ind.mean(axis=1).tolist(), "occupancy": float(ind.mean()), "indicator": st,
             "parameter_pass": params[n]["pass"]}
        b["indicator_pass"] = (not st["available"]) or bool(st["rhat"] < cfg["rhat_max"] and st["ess"] >= cfg["indicator_ess_min"])
        if not st["available"]:
            warnings.append(f"{n}: occupancy indicator near-constant (occupancy {b['occupancy']:.4f}); its diagnostics are unavailable")
        if not b["indicator_pass"]:
            conv_fail.append(f"{n}: occupancy indicator R-hat {st['rhat']:.3f}, ESS {st['ess']:.0f}")
        if reference is not None:
            r = np.asarray(reference[n], np.float64).ravel()
            rind = (r < threshold).astype(float)[None]
            rst = _indicator_stats(rind, cfg["min_minority_count"])
            r2 = r[None]
            ref_diag[n] = {"n": int(r.size), "split_rhat": rhat(r2), "ess_bulk": ess_bulk(r2), "ess_tail": ess_tail(r2),
                           "occupancy": float(rind.mean()), "indicator": rst}
            # conservative: the larger of the batch-means SE and the binomial SE from the indicator
            # ESS (the batch-means estimate from 8 chains is itself noisy)
            se_o = _conservative_se(ind, st, cfg["reference_batches"])
            se_r = _conservative_se(rind, rst, cfg["reference_batches"])
            d = b["occupancy"] - float(rind.mean())
            se = float(np.hypot(se_o, se_r))
            z = 0.0 if d == 0 else (d / se if se > 0 else float("inf"))
            b["agreement"] = {"reference_occupancy": float(rind.mean()), "diff": d, "se_ours": se_o, "se_ref": se_r,
                              "z": float(z), "pass": bool(abs(z) <= cfg["agreement_z_tol"])}
            if not b["agreement"]["pass"]:
                agree_fail.append(f"{n}: occupancy {b['occupancy']:.3f} vs reference {rind.mean():.3f} (z {z:.1f})")
        tails = []
        pooled_ok = st["available"]
        for p in probs:
            q = float(np.quantile(v, p))
            pairs = [(f"chain{c}_vs_rest", v[c : c + 1], np.delete(v, c, axis=0)) for c in range(C)]
            pairs.append(("first_vs_second_half", v[:, :h], v[:, h:]))
            for label, a, bb in pairs:
                ia, ib = (a <= q).astype(float), (bb <= q).astype(float)
                sa = _indicator_stats(ia, cfg["min_minority_count"])
                sb = _indicator_stats(ib, cfg["min_minority_count"])
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
    convergence = "PASS" if not conv_fail else "FAIL"
    agreement = "UNAVAILABLE" if reference is None else ("PASS" if not agree_fail else "FAIL")
    return out | {
        "input_errors": [],
        "convergence": convergence,
        "reproduction_agreement": agreement,
        "reproduction_acceptance": convergence == "PASS" and agreement == "PASS",
        "convergence_failures": conv_fail,
        "agreement_failures": agree_fail,
        "heuristic_warnings": warnings,
        "parameters": params,
        "bins": bins,
        "reference_diagnostics": ref_diag,
        "n_parameters_failing": int(sum(not s["pass"] for s in params.values())),
        "n_bins_failing_convergence": int(sum(not (bb["parameter_pass"] and bb["indicator_pass"]) for bb in bins.values())),
        "n_bins_disagreeing": len(agree_fail),
    }


def gate_exit_code(result: dict, require_reproduction: bool = True) -> int:
    """0: pass; 1: diagnostic failure (convergence FAIL, or reproduction FAIL when required);
    2: missing/invalid input, or reproduction UNAVAILABLE when required."""
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
