"""Bayes-factor estimators from posterior samples of two models on the same parameter space.

For models 1 (base) and 2 (target) with identical parameters and priors (e.g. CURN and HD with
the same IRN + common power-law priors), write l(x) = log L2(x) - log L1(x). Then

* **Reweighting** (Hourihane & Meyers 2022; importance sampling with the base posterior as the
  proposal):  BF_21 = E_{p1}[exp l],   ln BF = logsumexp(l_i) - ln n.
* **Reverse reweighting**:  1 / BF_21 = E_{p2}[exp(-l)].
* **Bridge sampling** with the Meng & Wong (1996) optimal bridge, which uses both sample sets:

      r = [ (1/n1) sum_{x in p1} e^l / (s1 r + s2 e^l) ] / [ (1/n2) sum_{x in p2} 1 / (s1 r + s2 e^l) ]

  iterated to a fixed point, with s_k = n_k / (n1 + n2) (we use effective sample sizes).

All sums are done in log space. Uncertainties come from a moving-block bootstrap over each
chain (block length from the integrated autocorrelation time of the weights), so they account
for autocorrelation in the MCMC draws. The Kish effective sample size of the importance weights
is reported for the reweighting estimators.

The bootstrap error is *conditional* on the draws: it cannot account for parts of the weight
distribution the chains never visited: it may underestimate the uncertainty and cannot account
for unvisited regions.
Report it together with block-length sensitivity and the spread between estimators.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .diagnostics import ess


def _as_chains(a) -> np.ndarray:
    a = np.asarray(a, np.float64)
    return a[None] if a.ndim == 1 else a


def log_mean_exp(a) -> float:
    a = np.ravel(a)
    return float(logsumexp(a) - np.log(a.size))


def kish_ess(logw) -> float:
    logw = np.ravel(logw)
    return float(np.exp(2 * logsumexp(logw) - logsumexp(2 * logw)))


def block_length(x: np.ndarray) -> int:
    """~2 x integrated autocorrelation time of x (chains, n), at least 1."""
    x = _as_chains(x)
    e = ess(x)
    tau = x.size / e if np.isfinite(e) and e > 0 else 1.0
    return int(max(1, np.ceil(2 * tau)))


def _block_resample_idx(n: int, b: int, rng) -> np.ndarray:
    """Moving-block bootstrap indices for one chain of length n with block length b."""
    b = min(b, n)
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=nb)
    return (starts[:, None] + np.arange(b)[None, :]).ravel()[:n]


def _resample(x: np.ndarray, b: int, rng) -> np.ndarray:
    return np.stack([row[_block_resample_idx(len(row), b, rng)] for row in x])


def reweight(l_base, n_boot: int = 2000, block: int | None = None, seed: int = 0) -> dict:
    """ln BF_21 from l = lnL2 - lnL1 evaluated at base-model (1) posterior draws (chains, n)."""
    l = _as_chains(l_base)
    lnbf = log_mean_exp(l)
    b = block or block_length(np.exp(l - l.max()))
    rng = np.random.default_rng(seed)
    boots = np.array([log_mean_exp(_resample(l, b, rng)) for _ in range(n_boot)])
    return _pack(lnbf, boots, b, kish_ess(l), l.size)


def reverse_reweight(l_target, n_boot: int = 2000, block: int | None = None, seed: int = 0) -> dict:
    """ln BF_21 from l = lnL2 - lnL1 evaluated at target-model (2) posterior draws."""
    l = _as_chains(l_target)
    lnbf = -log_mean_exp(-l)
    b = block or block_length(np.exp(-l - (-l).max()))
    rng = np.random.default_rng(seed)
    boots = np.array([-log_mean_exp(-_resample(l, b, rng)) for _ in range(n_boot)])
    return _pack(lnbf, boots, b, kish_ess(-l), l.size)


def _bridge(l1: np.ndarray, l2: np.ndarray, n1: float, n2: float, tol=1e-12, maxit=1000) -> float:
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    lr = log_mean_exp(l1)  # start from the reweighting estimate
    for _ in range(maxit):
        # log of (s1 r + s2 e^l) for both sample sets
        d1 = np.logaddexp(np.log(s1) + lr, np.log(s2) + l1)
        d2 = np.logaddexp(np.log(s1) + lr, np.log(s2) + l2)
        new = log_mean_exp(l1 - d1) - log_mean_exp(-d2)
        if abs(new - lr) < tol:
            return float(new)
        lr = new
    return float(lr)


def bridge_integrands(l1, l2, lnbf, n1, n2):
    """The two averaged quantities of the optimal bridge at the solution r = e^lnbf:
    numerator terms e^l / (s1 r + s2 e^l) at base draws (bounded by 1/s2) and denominator terms
    1 / (s1 r + s2 e^l) at target draws (bounded by 1/(s1 r)). Same shapes as l1, l2."""
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    d1 = np.logaddexp(np.log(s1) + lnbf, np.log(s2) + l1)
    d2 = np.logaddexp(np.log(s1) + lnbf, np.log(s2) + l2)
    return np.exp(l1 - d1), np.exp(-d2)


def bridge(l_base, l_target, n_boot: int = 1000, seed: int = 0, block: tuple[int, int] | None = None) -> dict:
    """Optimal-bridge ln BF_21 from l at base draws (chains, n1) and at target draws (chains, n2).

    The bridge weights s_k use the autocorrelation ESS of l in each sample set (they affect only
    the efficiency, not the consistency). Bootstrap block lengths, like those of the reweighting
    estimators, are 2x the integrated autocorrelation time of the *averaged quantities*, here the
    two bridge integrands at the solution (``bridge_integrands``); their R-hat and bulk ESS are
    returned as convergence diagnostics.
    """
    from .diagnostics import ess_bulk, rhat

    l1, l2 = _as_chains(l_base), _as_chains(l_target)
    n1, n2 = ess(l1), ess(l2)
    lnbf = _bridge(l1.ravel(), l2.ravel(), n1, n2)
    f1, f2 = bridge_integrands(l1, l2, lnbf, n1, n2)
    b1, b2 = block or (block_length(f1), block_length(f2))
    rng = np.random.default_rng(seed)
    boots = np.array(
        [_bridge(_resample(l1, b1, rng).ravel(), _resample(l2, b2, rng).ravel(), n1, n2) for _ in range(n_boot)]
    )
    out = _pack(lnbf, boots, (b1, b2), float("nan"), l1.size + l2.size)
    out["ess_base"], out["ess_target"] = float(n1), float(n2)
    out["integrand_diagnostics"] = {
        "base": {"rhat": rhat(f1), "ess_bulk": ess_bulk(f1), "max_over_mean": float(f1.max() / f1.mean())},
        "target": {"rhat": rhat(f2), "ess_bulk": ess_bulk(f2), "max_over_mean": float(f2.max() / f2.mean())},
    }
    return out


def _pack(lnbf, boots, b, kish, n) -> dict:
    sd = float(np.std(boots, ddof=1))
    bf_boot = np.exp(boots)
    return {
        "ln_bf": float(lnbf),
        "ln_bf_sd": sd,
        "bf": float(np.exp(lnbf)),
        "bf_sd": float(np.std(bf_boot, ddof=1)),
        "bf_q16_q84": [float(np.exp(np.quantile(boots, 0.16))), float(np.exp(np.quantile(boots, 0.84)))],
        "block": b,
        "kish_ess": kish,
        "n": int(n),
    }
