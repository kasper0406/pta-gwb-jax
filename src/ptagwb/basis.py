"""Fourier GP bases on the whole-array span and their coefficient priors.

Conventions follow enterprise ``createfourierdesignmatrix_red`` / discovery ``fourierbasis``:

* f_k = k / T, k = 1..n_modes, with T the span of the *whole* selected PTA
  (``data.get_tspan``; 505861299.1401644 s for NG15, as in the collaboration's figure code).
* Columns are interleaved ``[sin(2 pi f_1 t), cos(2 pi f_1 t), sin(2 pi f_2 t), ...]`` with t the
  absolute barycentric TOA in seconds (common time origin for all pulsars; no column
  normalisation, no zero-frequency mode).
* Coefficient variances (enterprise ``powerlaw``, discovery ``powerlaw``)::

      phi_k = A^2 / (12 pi^2) f_yr^(gamma - 3) f_k^(-gamma) Delta f_k,   Delta f_k = 1 / T

  for *each* of the sine and cosine coefficient, f_yr = 1 / (365.25 d).
* Free spectrum (enterprise ``free_spectrum``): phi_k = 10^(2 log10_rho_k), no Delta f.

The common process uses the first ``n_common`` frequencies of the same basis, so its columns
are exactly the first ``2 n_common`` columns of the intrinsic-red-noise basis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

YEAR_S = 365.25 * 86400.0
FYR = 1.0 / YEAR_S


def fourier_frequencies(n_modes: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    """(f, df), each repeated for sine/cosine: shape (2 n_modes,)."""
    f = np.arange(1, n_modes + 1, dtype=np.float64) / T
    df = np.diff(np.concatenate(([0.0], f)))
    return np.repeat(f, 2), np.repeat(df, 2)


def fourier_basis(toas: np.ndarray, n_modes: int, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, df, F) with F of shape (n_toa, 2 n_modes), sin/cos interleaved."""
    f, df = fourier_frequencies(n_modes, T)
    toas = np.asarray(toas, dtype=np.float64)
    F = np.empty((toas.size, 2 * n_modes), dtype=np.float64)
    arg = 2.0 * np.pi * toas[:, None] * f[None, ::2]
    F[:, 0::2] = np.sin(arg)
    F[:, 1::2] = np.cos(arg)
    return f, df, F


def powerlaw(f, df, log10_A, gamma):
    """Coefficient variance [s^2] of a power-law process (works on numpy or JAX arrays)."""
    return 10.0 ** (2.0 * log10_A) / (12.0 * np.pi**2) * FYR ** (gamma - 3.0) * f ** (-gamma) * df


def free_spectrum(log10_rho):
    """Coefficient variances (2 n,) from per-frequency log10 RMS ``log10_rho`` (n,)."""
    xp = jnp if isinstance(log10_rho, jax.Array) else np
    return xp.repeat(10.0 ** (2.0 * log10_rho), 2, axis=-1)
