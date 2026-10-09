"""HD-from-CURN importance reweighting: estimators, diagnostics and acceptance (plan N7, Sec. 5.4).

Conventions
-----------
* Inputs are per-chain, *ordered* (retained, post-burn-in) series. ``log_w`` is a sequence of
  1-D arrays (one per chain) of log importance weights ln w_i = lnL_HD(x_i) - lnL_CURN(x_i); a
  single 1-D array is treated as one chain. Chains may have different lengths.
* Weights are only ever exponentiated after subtracting the global maximum log-weight, so all
  estimators are numerically stable for log-weights of any offset.
* Accepted estimator: raw only. ln BF = ln mean(w) (pooled over chains); weighted quantiles use the
  raw normalised weights. PSIS k-hat is a diagnostic (acceptance needs k-hat < 0.5 pooled and in
  every chain).
* MCSEs are autocorrelation-aware: overlapping batch means (OBM, Flegal & Jones 2010) per chain
  with batch length b = max(ceil(5 tau), floor(sqrt(n))) where tau is the integrated
  autocorrelation time of the relevant per-chain series (the plan's ">= 5 tau" lower bound, raised
  to the conventional sqrt(n) when that is larger), combined across independent chains. A chain
  with b > n / 4 (fewer than 4 non-overlapping batches) has an *unestimable* MCSE, returned as
  +inf (fails closed). The ln BF MCSE is cross-checked by a moving-block bootstrap with the same
  block length; the precision gate uses the larger of the two.
* Kish ESS is reported, never gated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from ptagwb.diagnostics import ess

Z90 = 1.6448536269514722  # one-sided 95 % / two-sided 90 % normal quantile
MAX_MCSE_LNBF = 0.10
KHAT_THRESHOLD = 0.5
STABILITY_P_MIN = 0.01
MAX_CHAIN_WEIGHT_SHARE = 0.5


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------


def as_chains(x) -> list[np.ndarray]:
    """A 1-D array -> [array]; a sequence of 1-D arrays (or a 2-D array, rows = chains) -> list of
    float64 1-D arrays. Raises ValueError on empty input, empty chains or non-finite values."""
    if isinstance(x, np.ndarray) and x.ndim == 1:
        chains = [x]
    elif isinstance(x, np.ndarray) and x.ndim == 2:
        chains = list(x)
    else:
        chains = list(x)
    if not chains:
        raise ValueError("no chains")
    out = [np.asarray(c, np.float64).ravel() for c in chains]
    if any(c.size == 0 for c in out):
        raise ValueError("empty chain")
    if not all(np.all(np.isfinite(c)) for c in out):
        raise ValueError("non-finite values in chain")
    return out


def _normalised_weights(log_w: list[np.ndarray]) -> list[np.ndarray]:
    """exp(log_w - global max) per chain (largest weight = 1)."""
    mx = max(float(c.max()) for c in log_w)
    return [np.exp(c - mx) for c in log_w]


def integrated_autocorr_time(x) -> float:
    """Integrated autocorrelation time tau = 1 + 2 sum_k rho_k of one or several chains.

    * All chains of equal length (or one chain): tau = (C n) / ESS with the multi-chain ESS of
      ``ptagwb.diagnostics.ess`` (Geyer initial monotone sequence, Stan's between-chain-aware
      autocorrelation, lower floor 1 / log10(C n)).
    * Unequal lengths: the same Geyer initial monotone sequence on the length-weighted average of
      per-chain autocovariances taken about the *pooled* mean (so between-chain differences of the
      mean inflate tau), normalised by the pooled variance.
    A constant series returns 1.0; fewer than 4 draws per chain returns nan.
    """
    chains = as_chains(x)
    lengths = {c.size for c in chains}
    if len(lengths) == 1:
        n = chains[0].size
        e = ess(np.stack(chains))
        return float(len(chains) * n / e) if np.isfinite(e) else float("nan")
    if min(lengths) < 4:
        return float("nan")
    pooled = np.concatenate(chains)
    mu, var = pooled.mean(), pooled.var()
    if var == 0:
        return 1.0
    nmax = max(lengths)
    acc = np.zeros(nmax)
    cnt = np.zeros(nmax)
    for c in chains:
        n = c.size
        y = c - mu
        m = 1 << (2 * n - 1).bit_length()
        f = np.fft.rfft(y, n=m)
        ac = np.fft.irfft(f * np.conj(f), n=m)[:n]  # sum_{i} y_i y_{i+k}
        acc[:n] += ac
        cnt[:n] += n  # biased normalisation, as in diagnostics._autocov
    rho = acc / np.maximum(cnt, 1) / var
    return _geyer_tau(rho, pooled.size, len(chains))


def _geyer_tau(rho: np.ndarray, n_total: int, n_chains: int) -> float:
    t, pairs = 0, []
    while t + 1 < rho.size:
        s = rho[t] + rho[t + 1]
        if s < 0:
            break
        pairs.append(s)
        t += 2
    pairs_a = np.minimum.accumulate(np.array(pairs)) if pairs else np.array([1.0])
    tau = -1.0 + 2.0 * float(np.sum(pairs_a))
    return max(tau, 1.0 / np.log10(max(n_total, 11)))


def batch_length(tau: float, n: int) -> int:
    """OBM / block-bootstrap length b = max(ceil(5 tau), floor(sqrt(n)), 1) (module docstring)."""
    t = tau if np.isfinite(tau) else float(n)
    return int(max(np.ceil(5.0 * t), np.floor(np.sqrt(n)), 1))


def obm_variance_of_mean(y: np.ndarray, b: int) -> float:
    """Variance of mean(y) by overlapping batch means with batch length b (Flegal & Jones 2010):
    sigma^2 = n b / ((n - b)(n - b + 1)) sum_j (Ybar_j(b) - Ybar)^2, Var(mean) = sigma^2 / n.
    Returns +inf when b > n / 4 (too few batches to estimate)."""
    y = np.asarray(y, np.float64)
    n = y.size
    if b < 1 or 4 * b > n:
        return float("inf")
    cs = np.concatenate([[0.0], np.cumsum(y - y.mean())])
    bm = (cs[b:] - cs[:-b]) / b  # n - b + 1 overlapping batch means, centred
    sigma2 = n * b / ((n - b) * (n - b + 1)) * float(np.sum(bm**2))
    return sigma2 / n


def kish_ess(log_w) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2 of all pooled weights (reported only)."""
    lw = np.concatenate(as_chains(log_w))
    return float(np.exp(2.0 * logsumexp(lw) - logsumexp(2.0 * lw)))


# ----------------------------------------------------------------------------------------------
# PSIS k-hat
# ----------------------------------------------------------------------------------------------


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Zhang & Stephens (2009) GPD fit of ascending exceedances x > 0 with the weakly informative
    shape prior of the PSIS paper / loo: k <- (M k + 10 * 0.5) / (M + 10). Returns (k, sigma)."""
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(np.sqrt(n))
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1.0)
    wts = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = wts >= 10 * np.finfo(float).eps
    b, wts = b[keep], wts[keep] / wts[keep].sum()
    b_post = float(np.sum(b * wts))
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def psis_khat(log_w) -> float:
    """Pareto k-hat of the PSIS tail fit (Vehtari, Simpson, Gelman, Yao & Gabry, arXiv:1507.02646).

    Pools all chains given. Tail size M = ceil(min(N / 5, 3 sqrt(N))) (r_eff = 1); the M largest
    weights minus the (M+1)-th largest are fitted with the Zhang-Stephens GPD estimate plus the
    weakly informative prior adjustment (as in loo / ArviZ). Diagnostic only.
    Returns +inf when M < 5 (too few draws to fit; fails closed) and -inf when the tail is flat
    (all M largest weights equal the cutoff, e.g. constant weights).
    """
    lw = np.sort(np.concatenate(as_chains(log_w)))
    n = lw.size
    m = int(np.ceil(min(0.2 * n, 3.0 * np.sqrt(n))))
    if m < 5 or n <= m:
        return float("inf")
    lw = lw - lw[-1]
    cutoff = lw[-m - 1]
    tail = np.exp(lw[-m:]) - np.exp(cutoff)
    if tail[-1] <= 0:
        return float("-inf")
    tail = np.maximum(tail, 0.0)
    return float(_gpdfit(tail)[0])


# ----------------------------------------------------------------------------------------------
# ln BF and its MCSE
# ----------------------------------------------------------------------------------------------


def _masks(mask, chains: list[np.ndarray]) -> list[np.ndarray]:
    """Per-chain domain indicators (None -> all True), shape-checked against the chains."""
    if mask is None:
        return [np.ones(c.size, bool) for c in chains]
    ms = [np.asarray(m, bool).ravel() for m in ([mask] if isinstance(mask, np.ndarray) and mask.ndim == 1 else mask)]
    if len(ms) != len(chains) or any(a.size != b.size for a, b in zip(ms, chains)):
        raise ValueError("mask chains differ from the weights in number or length")
    if not any(m.any() for m in ms):
        raise ValueError("domain mask selects no draw")
    return ms


def raw_bf(log_w, mask=None) -> float:
    """ln BF_raw = ln mean(w) over all pooled draws; with a domain indicator I (``mask``),
    ln B_D = ln mean(w I) - ln mean(I) (the restricted-domain evidence ratio for a common domain
    D; draws outside D stay in the chains with weight 0, so chain order is preserved)."""
    chains = as_chains(log_w)
    ms = _masks(mask, chains)
    lw = np.concatenate(chains)
    m = np.concatenate(ms)
    return float(logsumexp(lw[m]) - np.log(m.sum()))


def _ratio_parts(chains, ms):
    """(a, b) per chain: a = w I with w normalised by the global maximum weight, b = I."""
    w = _normalised_weights(chains)
    return [c * m for c, m in zip(w, ms)], [m.astype(np.float64) for m in ms]


def mcse_lnbf_obm(log_w, mask=None) -> dict:
    """OBM MCSE of ln B (ln B_D with a domain mask) by the delta method on the ratio
    R = mean(a) / mean(b), a = w I, b = I (paired numerator/denominator):
    Z_i = a_i / abar - b_i / bbar, Var(ln R) = Var(mean Z).

    Per chain c: tau_c = integrated autocorrelation time of Z_c, b_c = batch_length(tau_c, n_c),
    V_c = OBM variance of mean(Z_c) (chain order kept). Pooled: Var = sum_c (n_c / N)^2 V_c with
    the pooled means in Z. Without a mask (b = 1) this is the OBM variance of mean(w) / wbar^2.
    Returns ``mcse`` (pooled), per-chain ln B and MCSE (each chain's own means), ``tau``,
    ``batch_length``.
    """
    chains = as_chains(log_w)
    ms = _masks(mask, chains)
    a, b = _ratio_parts(chains, ms)
    n = np.array([c.size for c in a], float)
    big_n = n.sum()
    abar = float(np.sum([x.sum() for x in a]) / big_n)
    bbar = float(np.sum([x.sum() for x in b]) / big_n)
    var, taus, bs, lnb_c, mc_c = 0.0, [], [], [], []
    for ac, bc, lc, mc, nc in zip(a, b, chains, ms, n):
        z = ac / abar - bc / bbar
        tau = integrated_autocorr_time(z)
        bl = batch_length(tau, int(nc))
        taus.append(float(tau))
        bs.append(bl)
        var += (nc / big_n) ** 2 * obm_variance_of_mean(z, bl)
        if mc.any() and ac.sum() > 0:
            zc = ac / ac.mean() - bc / bc.mean()
            lnb_c.append(raw_bf(lc, mc))
            mc_c.append(float(np.sqrt(obm_variance_of_mean(zc, batch_length(integrated_autocorr_time(zc), int(nc))))))
        else:
            lnb_c.append(float("nan"))
            mc_c.append(float("inf"))
    return {"lnbf": raw_bf(chains, ms), "mcse": float(np.sqrt(var)), "per_chain_lnbf": lnb_c,
            "per_chain_mcse": mc_c, "tau": taus, "batch_length": bs}


def mcse_lnbf_block_bootstrap(log_w, n_boot: int = 1000, seed: int = 0,
                              block_lengths: Sequence[int] | None = None, mask=None) -> float:
    """Moving-block bootstrap SD of the pooled ln B (ln B_D with a mask): blocks of the paired
    series (a, b) = (w I, I) are resampled together (cross-check of the OBM MCSE).

    Each chain is resampled independently: floor(n_c / b_c) blocks of length b_c drawn with
    replacement from the n_c - b_c + 1 overlapping blocks (non-circular), b_c as in
    ``mcse_lnbf_obm`` unless given. Returns +inf if any chain has b_c > n_c / 4.
    """
    chains = as_chains(log_w)
    ms = _masks(mask, chains)
    a, b = _ratio_parts(chains, ms)
    if block_lengths is None:
        block_lengths = mcse_lnbf_obm(chains, ms)["batch_length"]
    rng = np.random.default_rng(seed)
    ta, tb = np.zeros(n_boot), np.zeros(n_boot)
    for ac, bc, bl in zip(a, b, block_lengths):
        n = ac.size
        if 4 * bl > n:
            return float("inf")
        k = n // bl
        csa = np.concatenate([[0.0], np.cumsum(ac)])
        csb = np.concatenate([[0.0], np.cumsum(bc)])
        starts = rng.integers(0, n - bl + 1, size=(n_boot, k))
        ta += (csa[bl:] - csa[:-bl])[starts].sum(axis=1)
        tb += (csb[bl:] - csb[:-bl])[starts].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(ta / tb)
    return float(np.std(lr, ddof=1)) if np.all(np.isfinite(lr)) else float("inf")


# ----------------------------------------------------------------------------------------------
# weighted quantiles and their MCSE
# ----------------------------------------------------------------------------------------------


def weighted_quantile(x, p: float, weights=None) -> float:
    """Inverse of the weighted empirical CDF: the smallest x_(i) with cumulative normalised weight
    >= p (type-1 / "inverted CDF"; weights=None -> equal weights). Requires 0 <= p <= 1."""
    x = np.asarray(x, np.float64).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, np.float64).ravel()
    if x.size == 0 or x.size != w.size:
        raise ValueError("x and weights must be non-empty and of equal size")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p={p} outside [0, 1]")
    o = np.argsort(x, kind="stable")
    xs, cw = x[o], np.cumsum(w[o])
    cw = cw / cw[-1]
    i = int(np.searchsorted(cw, p - 1e-12, side="left"))
    return float(xs[min(i, xs.size - 1)])


def weighted_kde_density(x: np.ndarray, w: np.ndarray, at: float) -> tuple[float, float]:
    """Gaussian-kernel weighted density at ``at`` with the Silverman bandwidth
    h = 0.9 min(sd_w, IQR_w / 1.34) n_kish^(-1/5) (weighted sd / IQR, Kish ESS as n; falls back
    to sd_w when the IQR is 0). Returns (density, h)."""
    w = w / w.sum()
    mu = float(np.sum(w * x))
    sd = float(np.sqrt(np.sum(w * (x - mu) ** 2)))
    iqr = weighted_quantile(x, 0.75, w) - weighted_quantile(x, 0.25, w)
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    n_kish = 1.0 / float(np.sum(w**2))
    h = 0.9 * spread * n_kish ** (-0.2)
    if h <= 0:
        return float("inf"), 0.0
    return float(np.sum(w * stats.norm.pdf((at - x) / h)) / h), h


def weighted_quantile_mcse(x, log_w, p: float, mask=None) -> dict:
    """Weighted p-quantile with raw normalised weights and its ratio-estimator MCSE (Sec. 5.4).

    R(q) = sum w 1[x <= q] / sum w. Linearisation (delta method on the pair (w 1[x <= q_p], w)):
    Z_i = w_i (1[x_i <= q_p] - p) / wbar, so Var(R) = Var(mean Z). Per chain, Var(mean Z_c) by OBM
    with b_c = batch_length(max(tau(Z_c), tau(w_c)), n_c); pooled
    Var = sum_c (n_c / N)^2 Var(mean Z_c). MCSE in quantile units = sqrt(Var(R)) / f_w(q_p) with
    f_w the weighted Gaussian KDE (Silverman bandwidth, returned). Chains of x and log_w must
    match in number and length. ``mask`` (optional, per-chain booleans): domain indicator; the
    weights become w_i 1[x_i in D] (a conditional quantile; draws outside D stay in the ordered
    chains, so serial dependence of the indicator enters the MCSE).
    """
    xs, lws = as_chains(x), as_chains(log_w)
    if len(xs) != len(lws) or any(a.size != b.size for a, b in zip(xs, lws)):
        raise ValueError("x and log_w chains differ in number or length")
    w = _normalised_weights(lws)
    if mask is not None:
        ms = [np.asarray(m, bool).ravel() for m in (mask if not (isinstance(mask, np.ndarray) and mask.ndim == 1) else [mask])]
        if len(ms) != len(w) or any(a.size != b.size for a, b in zip(ms, w)):
            raise ValueError("mask chains differ from x in number or length")
        w = [c * m for c, m in zip(w, ms)]
        if not any(c.any() for c in ms):
            raise ValueError("domain mask selects no draw")
    xa, wa = np.concatenate(xs), np.concatenate(w)
    q = weighted_quantile(xa, p, wa)
    wbar = float(wa.mean())
    n = np.array([c.size for c in w], float)
    big_n = n.sum()
    var = 0.0
    for xc, wc, nc in zip(xs, w, n):
        z = wc * ((xc <= q).astype(float) - p) / wbar
        tau = max(integrated_autocorr_time(z), integrated_autocorr_time(wc))
        var += (nc / big_n) ** 2 * obm_variance_of_mean(z, batch_length(tau, int(nc)))
    dens, h = weighted_kde_density(xa, wa, q)
    mcse_r = float(np.sqrt(var))
    return {"q": q, "p": p, "mcse": mcse_r / dens if dens > 0 else float("inf"),
            "mcse_prob": mcse_r, "density": dens, "bandwidth": h}


def paired_quantile_shift(x, log_w, p: float, mask=None) -> dict:
    """Displacement of the p-quantile under reweighting, delta = q_w - q_1 (weights w I vs I), with
    the **paired** MCSE: both endpoints come from the same ordered draws, so the linearisations are
    combined before the variance is taken,
        Z_i = -[w_i I_i (1[x_i <= q_w] - p) / mean(w I)] / f_w(q_w) + [I_i (1[x_i <= q_1] - p) / mean(I)] / f_1(q_1),
    Var(delta) = Var(mean Z) by OBM per chain (batch length from tau(Z)), pooled over chains. f are
    the weighted / unweighted Gaussian KDEs (``weighted_kde_density``)."""
    xs, lws = as_chains(x), as_chains(log_w)
    ms = _masks(mask, lws)
    w = [c * m for c, m in zip(_normalised_weights(lws), ms)]
    one = [m.astype(np.float64) for m in ms]
    xa, wa, oa = np.concatenate(xs), np.concatenate(w), np.concatenate(one)
    qw, q1 = weighted_quantile(xa, p, wa), weighted_quantile(xa, p, oa)
    fw, _ = weighted_kde_density(xa, wa, qw)
    f1, _ = weighted_kde_density(xa, oa, q1)
    mw, m1 = float(wa.mean()), float(oa.mean())
    n = np.array([c.size for c in xs], float)
    big_n = n.sum()
    var = 0.0
    for xc, wc, oc, nc in zip(xs, w, one, n):
        z = -wc * ((xc <= qw) - p) / mw / fw + oc * ((xc <= q1) - p) / m1 / f1
        var += (nc / big_n) ** 2 * obm_variance_of_mean(z, batch_length(integrated_autocorr_time(z), int(nc)))
    return {"q_unweighted": q1, "q_weighted": qw, "shift": qw - q1, "mcse_shift_paired": float(np.sqrt(var)),
            "endpoint_mcse_weighted": weighted_quantile_mcse(xs, lws, p, mask=ms)["mcse"]}


# ----------------------------------------------------------------------------------------------
# per-chain stability
# ----------------------------------------------------------------------------------------------


def chi2_consistency(values: Sequence[float], mcses: Sequence[float]) -> dict:
    """chi^2 = sum_c ((v_c - vbar) / s_c)^2 about the inverse-variance mean vbar, dof = C - 1,
    p = chi2.sf. Fewer than 2 chains or any non-finite / non-positive MCSE gives p = 0 (fail
    closed)."""
    v, s = np.asarray(values, float), np.asarray(mcses, float)
    if v.size < 2 or not np.all(np.isfinite(s)) or np.any(s <= 0) or not np.all(np.isfinite(v)):
        return {"chi2": float("nan"), "dof": int(max(v.size - 1, 0)), "p": 0.0, "mean": float("nan")}
    iv = 1.0 / s**2
    mean = float(np.sum(iv * v) / np.sum(iv))
    chi2 = float(np.sum(((v - mean) / s) ** 2))
    return {"chi2": chi2, "dof": int(v.size - 1), "p": float(stats.chi2.sf(chi2, v.size - 1)),
            "mean": mean}


def chain_stability(log_w, x_by_name: Mapping[str, Sequence[np.ndarray]] | None = None, mask=None) -> dict:
    """Independent-chain stability (Sec. 5.4): chi^2 consistency of per-chain ln B (ln B_D with a
    mask; OBM MCSEs) and of per-chain weighted medians of each target (ratio-estimator MCSEs, with
    the mask), plus the maximum single-chain share of the total weight w I. Passes iff every
    p > 0.01 and share <= 0.5."""
    chains = as_chains(log_w)
    ms = _masks(mask, chains)
    a, _ = _ratio_parts(chains, ms)
    tot = np.array([c.sum() for c in a])
    share = float(tot.max() / tot.sum())
    ob = mcse_lnbf_obm(chains, ms)
    lnbf = chi2_consistency(ob["per_chain_lnbf"], ob["per_chain_mcse"])
    medians = {}
    for name, xc in (x_by_name or {}).items():
        xc = as_chains(xc)
        per = [weighted_quantile_mcse(xa, lc, 0.5, mask=[m]) for xa, lc, m in zip(xc, chains, ms)]
        res = chi2_consistency([r["q"] for r in per], [r["mcse"] for r in per])
        res["per_chain"] = [r["q"] for r in per]
        medians[name] = res
    ok = (lnbf["p"] > STABILITY_P_MIN and share <= MAX_CHAIN_WEIGHT_SHARE
          and all(r["p"] > STABILITY_P_MIN for r in medians.values()))
    return {"lnbf": lnbf, "medians": medians, "max_chain_weight_share": share, "pass": bool(ok)}


# ----------------------------------------------------------------------------------------------
# acceptance
# ----------------------------------------------------------------------------------------------


def accept_reweighting(
    log_w,
    x_by_name: Mapping[str, Sequence[np.ndarray]],
    probs: Sequence[float],
    max_our_mcse: Mapping[tuple[str, float], float | None],
    *,
    max_mcse_lnbf: float = MAX_MCSE_LNBF,
    khat_threshold: float = KHAT_THRESHOLD,
    n_boot: int = 1000,
    seed: int = 0,
    mask=None,
) -> dict:
    """Reweighting acceptance (Sec. 5.4), fail closed. With ``mask`` (per-chain indicators of the
    common domain D, revised D9) every quantity is the conditional one: k-hat and Kish ESS on the
    weights of draws in D, ln B_D with the paired OBM / block-bootstrap MCSE, weighted quantiles
    with weights w I, and the per-chain stability on D; chain order is preserved throughout.

    ``max_our_mcse`` must contain an entry for every (name, p) in x_by_name x probs (KeyError
    otherwise); a value of None marks a reported-only (non-decidable) quantity whose MCSE is not
    gated. ``accepted`` requires: finite k-hat < khat_threshold pooled and in every chain; chain
    stability pass; max(OBM, block-bootstrap) MCSE(ln BF) <= max_mcse_lnbf; every gated quantile
    MCSE finite and <= its max_our_MCSE. Returns a dict with all diagnostics and ``reasons``.
    """
    chains = as_chains(log_w)
    ms = _masks(mask, chains)
    for name in x_by_name:
        for p in probs:
            if (name, p) not in max_our_mcse:
                raise KeyError(f"max_our_mcse missing for ({name!r}, {p})")
    reasons: list[str] = []
    in_d = [c[m] for c, m in zip(chains, ms)]
    k_pooled = psis_khat(np.concatenate(in_d))
    k_chain = [psis_khat(c) if c.size else float("inf") for c in in_d]
    khat_ok = all(k < khat_threshold for k in [k_pooled, *k_chain])  # nan / +inf fail
    if not khat_ok:
        reasons.append(f"k-hat >= {khat_threshold} (pooled {k_pooled:.3f}, chains {k_chain})")
    ob = mcse_lnbf_obm(chains, ms)
    boot = mcse_lnbf_block_bootstrap(chains, n_boot=n_boot, seed=seed,
                                     block_lengths=ob["batch_length"], mask=ms)
    mcse_gate = max(ob["mcse"], boot)
    lnbf_ok = bool(np.isfinite(mcse_gate) and mcse_gate <= max_mcse_lnbf)
    if not lnbf_ok:
        reasons.append(f"MCSE(ln BF) {mcse_gate:.4f} > {max_mcse_lnbf}")
    quant: dict[str, dict] = {}
    quant_ok = True
    for name, xc in x_by_name.items():
        quant[name] = {}
        for p in probs:
            r = weighted_quantile_mcse(xc, chains, p, mask=ms)
            lim = max_our_mcse[(name, p)]
            r["max_our_mcse"] = lim
            r["pass"] = True if lim is None else bool(np.isfinite(r["mcse"]) and r["mcse"] <= lim)
            if not r["pass"]:
                quant_ok = False
                reasons.append(f"MCSE({name} q{p}) {r['mcse']:.4g} > {lim}")
            quant[name][p] = r
    stab = chain_stability(chains, x_by_name, ms)
    if not stab["pass"]:
        reasons.append("per-chain stability failed")
    return {
        "accepted": bool(khat_ok and lnbf_ok and quant_ok and stab["pass"]),
        "reasons": reasons,
        "khat": {"pooled": k_pooled, "per_chain": k_chain, "threshold": khat_threshold,
                 "pass": khat_ok},
        "lnbf": {"value": ob["lnbf"], "mcse_obm": ob["mcse"], "mcse_bootstrap": boot,
                 "mcse_gate": mcse_gate, "max": max_mcse_lnbf, "pass": lnbf_ok,
                 "per_chain": ob["per_chain_lnbf"], "per_chain_mcse": ob["per_chain_mcse"],
                 "tau_w": ob["tau"], "batch_length": ob["batch_length"]},
        "quantiles": quant,
        "stability": stab,
        "kish_ess": kish_ess(np.concatenate(in_d)),
        "n_draws": int(sum(c.size for c in chains)),
        "n_draws_in_domain": int(sum(m.sum() for m in ms)),
        "n_chains": len(chains),
    }


__all__ = [
    "accept_reweighting", "as_chains", "batch_length", "chain_stability", "chi2_consistency",
    "integrated_autocorr_time", "kish_ess", "mcse_lnbf_block_bootstrap", "mcse_lnbf_obm",
    "obm_variance_of_mean", "paired_quantile_shift", "psis_khat", "raw_bf", "weighted_kde_density", "weighted_quantile",
    "weighted_quantile_mcse",
]
