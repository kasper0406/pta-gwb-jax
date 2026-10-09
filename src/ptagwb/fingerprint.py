"""Chain fingerprint gate (docs/M3B_PLAN.md Sec. 4.5 item 3; supporting, fail-closed).

At >= 50 released draws x_i spread over the retained chain, Delta_i = lnL_ours(x_i) -
lnlike_ref(x_i) in float64 (reference values parsed from text, never via float32). Frozen error
budget per draw:

    sigma_i^2 = sigma_store^2 + sigma_par,i^2 + sigma_eval^2

* sigma_store: six-decimal storage rounding of lnlike, uniform +-5e-7 -> 5e-7 / sqrt(3);
* sigma_par,i = sum_j |d lnL / d x_j| delta_x_j, delta_x_j = half a unit in the last stored digit
  of x_ij (``stored_precision``); a coordinate at which lnL is discontinuous (the dip epoch t0) is
  handled by ``discontinuity_margin``: the gate fails if any TOA lies within delta_t0 of t0;
* sigma_eval = 1e-6 nats, the arbiter-validated bound on our evaluator error (G5-PTA).

PASS iff chi2 = sum((Delta_i - mean)^2 / sigma_i^2) / (n - 1) <= 2. Anything above is FAIL,
whatever its suspected cause. Grid discrimination: a hypothesis is accepted only if it passes and
every alternative fails with chi2 >= 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMA_STORE = 5e-7 / np.sqrt(3.0)
SIGMA_EVAL = 1e-6
CHI2_PASS = 2.0
CHI2_ALTERNATIVE_FAIL = 100.0
MIN_DRAWS = 50


def stored_precision(text_values: list[str]) -> np.ndarray:
    """Half a unit in the last digit of each decimal string (e.g. '1.25' -> 0.005)."""
    out = []
    for s in text_values:
        s = s.strip().lower()
        mant, _, exp = s.partition("e")
        e = int(exp) if exp else 0
        dec = len(mant.split(".")[1]) if "." in mant else 0
        out.append(0.5 * 10.0 ** (e - dec))
    return np.asarray(out)


def spread_rows(n_rows: int, burn: int, n: int = 60) -> np.ndarray:
    """Draw indices spread evenly over the retained chain [burn, n_rows)."""
    if n < MIN_DRAWS:
        raise ValueError(f"fingerprint needs >= {MIN_DRAWS} draws")
    return np.unique(np.linspace(burn, n_rows - 1, n).round().astype(int))


@dataclass
class FingerprintResult:
    n: int
    mean: float
    sd: float
    chi2: float
    passed: bool
    sigma_median: float
    sigma_par_max: float
    deltas: np.ndarray

    def as_dict(self) -> dict:
        return {"n": self.n, "c_m": self.mean, "sd": self.sd, "chi2_per_dof": self.chi2, "pass": self.passed,
                "sigma_median": self.sigma_median, "sigma_par_max": self.sigma_par_max,
                "tolerance": f"chi2 <= {CHI2_PASS} with sigma_store {SIGMA_STORE:.3e}, sigma_eval {SIGMA_EVAL:.0e}"}


def evaluate(lnl_ours: np.ndarray, lnl_ref: np.ndarray, sigma_par: np.ndarray) -> FingerprintResult:
    d = np.asarray(lnl_ours, dtype=np.float64) - np.asarray(lnl_ref, dtype=np.float64)
    if d.size < MIN_DRAWS:
        raise ValueError(f"fingerprint needs >= {MIN_DRAWS} draws, got {d.size}")
    sig = np.sqrt(SIGMA_STORE**2 + np.asarray(sigma_par) ** 2 + SIGMA_EVAL**2)
    chi2 = float(np.sum((d - d.mean()) ** 2 / sig**2) / (d.size - 1))
    if not np.all(np.isfinite(d)):
        chi2 = float("inf")
    return FingerprintResult(n=int(d.size), mean=float(d.mean()), sd=float(d.std(ddof=1)), chi2=chi2,
                             passed=bool(chi2 <= CHI2_PASS), sigma_median=float(np.median(sig)),
                             sigma_par_max=float(np.max(sigma_par)), deltas=d)


def discontinuity_margin(t0_days: np.ndarray, toas_s: np.ndarray, delta_days: np.ndarray) -> float:
    """Smallest |t_TOA - t0| relative to the stored precision of t0 (> 1 required): a TOA within
    the rounding of t0 would make the stored draw ambiguous."""
    t = np.asarray(toas_s) / 86400.0
    gaps = [np.min(np.abs(t - x)) / dx for x, dx in zip(np.atleast_1d(t0_days), np.atleast_1d(delta_days), strict=True)]
    return float(np.min(gaps))


def discriminate(primary: FingerprintResult, alternatives: dict) -> dict:
    alt = {k: {"chi2": v.chi2, "fails_decisively": bool(v.chi2 >= CHI2_ALTERNATIVE_FAIL)} for k, v in alternatives.items()}
    return {"primary_pass": primary.passed, "alternatives": alt,
            "resolved": bool(primary.passed and all(a["fails_decisively"] for a in alt.values()))}
