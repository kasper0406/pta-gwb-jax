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
