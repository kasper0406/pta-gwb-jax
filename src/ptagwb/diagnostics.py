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
    # Vehtari et al. (2021) recommend bulk and tail ESS > 400 (100 per chain for 4 chains) before
    # trusting R-hat and quantile MCSEs; we require it for every parameter and for the occupancy
    # indicators, independent of the number of chains.
    "ess_bulk_min": 400.0,
    "ess_tail_min": 400.0,
    "indicator_ess_min": 400.0,
    # tail-stability tolerance in units of the standard error of the difference. With 8 chains,
    # 30 bins, 3 quantiles (~810 tests incl. half-run tests), Gaussian |z| > 4.5 gives a
    # family-wise false-alarm rate of ~0.5% (|z| > 4 tripped on 1 of 10 synthetic well-mixed sets).
    "z_tol": 4.5,
    # 30 bins: Gaussian |z| > 3.5 has a ~1.4% family-wise false-alarm rate
    "occupancy_z_tol": 3.5,
}


def _indicator_stats(ind: np.ndarray) -> tuple[float, float]:
    """(R-hat, ESS) of a 0/1 indicator (chains, n); degenerate (constant) indicators get (1, N)."""
    if np.ptp(ind) == 0:
        return 1.0, float(ind.size)
    return rhat(ind), ess(_split(ind))


def _cdf_diff_test(a: np.ndarray, b: np.ndarray, q: float, z_tol: float) -> dict:
    """Tail-stability test at a reference quantile value q: the fractions of draws <= q in two
    disjoint subsets a, b ((chains, n) arrays) and z = (F_a - F_b) / sqrt(var_a + var_b), with
    var = F (1 - F) / ESS of the indicator computed within each subset (floored at 1/(n ESS)).

    This is a quantile-stability test expressed on the probability scale: q_p is stable across
    subsets iff the mass below the pooled q_p is. Unlike a difference of quantile values divided
    by quantile MCSEs, it stays well defined when q_p falls in a low-density gap between the two
    modes of a free-spectrum marginal, where quantile values jump and their MCSEs are unreliable
    (a synthetic, perfectly mixed bimodal set produced |z| up to 8 with the quantile-value form).
    """
    out = {}
    var = []
    for key, v in (("a", a), ("b", b)):
        ind = (v <= q).astype(float)
        f = float(ind.mean())
        _, e = _indicator_stats(ind)
        out[f"F_{key}"] = f
        var.append(max(f * (1 - f), 1.0 / ind.size) / max(e, 1.0))
    d = out["F_a"] - out["F_b"]
    se = float(np.sqrt(sum(var)))
    z = d / se if se > 0 else (0.0 if d == 0 else float("inf"))
    return {**out, "d": d, "se": se, "z": float(z), "pass": bool(np.isfinite(z) and abs(z) <= z_tol)}


def freespec_gate(
    x: np.ndarray,
    names: list[str],
    n_bins: int = 30,
    prefix: str = "gw_log10_rho_",
    threshold: float = -9.0,
    released: dict | None = None,
    probs: tuple[float, ...] = (0.05, 0.5, 0.95),
    **overrides,
) -> dict:
    """Aggregate acceptance check for a free-spectrum run. ``x``: (chains, draws, D) draws with
    parameter ``names``; ``released``: optional {name: 1-D reference draws}.

    Fails (``pass`` False) on any missing bin, missing/non-finite diagnostic, or:

    1. any parameter (all D, incl. IRN) with rank-normalised split R-hat >= rhat_max, bulk ESS <
       ess_bulk_min or tail ESS < ess_tail_min;
    2. any bin whose occupancy indicator 1[log10_rho < threshold] has R-hat >= rhat_max or ESS <
       indicator_ess_min, or (with ``released``) pooled occupancy differing from the released one
       by more than occupancy_z_tol binomial-ESS standard errors;
    3. tail instability: for each bin and p in ``probs``, with q = pooled p-quantile, compare the
       fraction of draws <= q in chain c vs the other chains (every c) and in the first vs second
       half of all chains; z = difference / sqrt(var_a + var_b) with binomial variances from each
       subset's own indicator ESS (``_cdf_diff_test``); any |z| > z_tol fails. Differences of the
       quantile values themselves are reported descriptively.

    The between-chain chi^2 occupancy test is reported as supplementary only (its ESS
    adjustment makes it weak).
    """
    from scipy import stats as _st

    cfg = {**GATE_DEFAULTS, **overrides}
    x = np.asarray(x, np.float64)
    C, N, D = x.shape
    failures: list[str] = []
    if len(names) != D:
        failures.append(f"names ({len(names)}) do not match D = {D}")
    if C < 2:
        failures.append("need >= 2 chains")
    params = {}
    for j, n in enumerate(names[:D]):
        s = {"rhat": rhat(x[..., j]), "ess_bulk": ess_bulk(x[..., j]), "ess_tail": ess_tail(x[..., j])}
        ok = (np.isfinite(list(s.values())).all() and s["rhat"] < cfg["rhat_max"]
              and s["ess_bulk"] >= cfg["ess_bulk_min"] and s["ess_tail"] >= cfg["ess_tail_min"])
        s["pass"] = bool(ok)
        params[n] = s
        if not ok:
            failures.append(f"{n}: R-hat {s['rhat']:.4f}, bulk ESS {s['ess_bulk']:.0f}, tail ESS {s['ess_tail']:.0f}")
    bins = {}
    idx = {n: j for j, n in enumerate(names)}
    h = N // 2
    for k in range(n_bins):
        n = f"{prefix}{k}"
        if n not in idx:
            failures.append(f"{n}: missing")
            bins[n] = {"pass": False, "missing": True}
            continue
        v = x[..., idx[n]]
        ind = (v < threshold).astype(float)
        ir, ie = _indicator_stats(ind)
        pooled = float(ind.mean())
        b = {"occupancy_per_chain": ind.mean(axis=1).tolist(), "occupancy": pooled,
             "indicator_rhat": ir, "indicator_ess": ie, "checks": {}}
        b["checks"]["indicator"] = bool(np.isfinite(ir) and ir < cfg["rhat_max"] and ie >= cfg["indicator_ess_min"])
        # supplementary between-chain chi^2 on ESS-scaled counts
        if 0 < pooled < 1:
            neff = max(ie / C, 1.0)
            chi2 = float(np.sum((ind.mean(axis=1) - pooled) ** 2 * neff / (pooled * (1 - pooled))))
            b["supplementary_chi2_p"] = float(_st.chi2(C - 1).sf(chi2))
        if released is not None and n in released:
            r = np.asarray(released[n], np.float64)
            ro = float(np.mean(r < threshold))
            _, re_ = _indicator_stats((r < threshold).astype(float)[None])
            se_o = np.sqrt(max(pooled * (1 - pooled), 1.0 / ind.size) / max(ie, 1.0))
            se_r = np.sqrt(max(ro * (1 - ro), 1.0 / r.size) / max(re_, 1.0))
            z = (pooled - ro) / np.hypot(se_o, se_r)
            b["released_occupancy"], b["occupancy_z"] = ro, float(z)
            b["checks"]["occupancy_vs_released"] = bool(abs(z) <= cfg["occupancy_z_tol"])
        tails = []
        for p in probs:
            q = float(np.quantile(v, p))
            for c in range(C):
                t = _cdf_diff_test(v[c : c + 1], np.delete(v, c, axis=0), q, cfg["z_tol"])
                t["q_chain_minus_rest"] = float(np.quantile(v[c], p) - np.quantile(np.delete(v, c, axis=0), p))
                tails.append({"test": f"chain{c}_vs_rest", "p": p, "q_pooled": q, **t})
            t = _cdf_diff_test(v[:, :h], v[:, h:], q, cfg["z_tol"])
            t["q_first_minus_second"] = float(np.quantile(v[:, :h], p) - np.quantile(v[:, h:], p))
            tails.append({"test": "first_vs_second_half", "p": p, "q_pooled": q, **t})
        b["tail_tests"] = tails
        b["checks"]["tail_stability"] = all(t["pass"] for t in tails)
        b["checks"]["parameter"] = params.get(n, {}).get("pass", False)
        b["pass"] = all(b["checks"].values())
        if not b["pass"]:
            bad = [kk for kk, vv in b["checks"].items() if not vv]
            worst = max((abs(t["z"]) if np.isfinite(t["z"]) else np.inf) for t in tails)
            failures.append(f"{n}: failed {bad} (indicator R-hat {ir:.3f}, ESS {ie:.0f}, max |tail z| {worst:.1f})")
        bins[n] = b
    return {
        "pass": not failures,
        "failures": failures,
        "n_failures": len(failures),
        "criteria": cfg | {"threshold": threshold, "n_bins": n_bins, "probs": list(probs)},
        "parameters": params,
        "bins": bins,
        "n_parameters_failing": int(sum(not s["pass"] for s in params.values())),
        "n_bins_failing": int(sum(not b["pass"] for b in bins.values())),
    }
