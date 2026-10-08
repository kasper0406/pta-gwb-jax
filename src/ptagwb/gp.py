"""Fourier Gaussian-process blocks with their own grid, span, scaling, selection and PSD
normalisation (docs/M3_PLAN.md Sec. 4.3, component L5).

A block is a fixed basis: columns ``[sin(2 pi f_1 (t - t0)), cos(...), sin(2 pi f_2 ...), ...]``
with f_k = k / T (k = 1..n_modes), multiplied row-wise by a per-TOA scale and a row mask:

* chromatic scale ``(fref / nu)^chrom_idx`` with nu the (barycentric) radio frequency in MHz
  (``chrom_idx = 2``: DM, enterprise ``createfourierdesignmatrix_dm``; 0: achromatic);
* ``norm="temponest_dm"`` additionally multiplies by sqrt(12) pi / (1400^2 * 2.41e-4), the
  TempoNest DM normalisation of enterprise ``createfourierdesignmatrix_dm_tn`` (EPTA DR2
  ``TNDMAmp`` values are used unchanged as ``dm_gp_log10_A`` with this basis);
* ``selection``: rows outside the selection are zero (band / system noise). Supported:
  ``("flag", value)`` on the (namespaced) system label, ``("freq", lo, hi)`` on the radio
  frequency [MHz] (lo < nu < hi), ``None`` = all rows.

Two blocks with equal ``basis_key`` produce identical columns for equal mode numbers; the
combined likelihood merges such columns (``combined.ColumnLayout``) so that e.g. a common process
on the first 14 frequencies of the 30-frequency IRN grid re-uses those columns, exactly as M1
does (the "prefix identity"). Blocks with different spans keep separate columns.

Coefficient prior (``powerlaw``): enterprise / discovery normalisation for every block,
phi_k = A^2 / (12 pi^2) f_yr^(gamma - 3) f_k^(-gamma) / T (see ``basis.powerlaw``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import fourier_frequencies

TN_DM_NORM = np.sqrt(12.0) * np.pi / 1400.0 / 1400.0 / 2.41e-4
NORMS = ("enterprise", "temponest_dm")


@dataclass(frozen=True)
class FourierBlock:
    """Basis definition of one Fourier GP block (no spectrum; see module docstring)."""

    name: str
    n_modes: int
    T: float
    chrom_idx: float = 0.0
    fref: float = 1400.0
    norm: str = "enterprise"
    t0: float = 0.0
    selection: tuple | None = None

    def __post_init__(self):
        if self.n_modes < 1 or not np.isfinite(self.T) or self.T <= 0:
            raise ValueError(f"block {self.name!r}: need n_modes >= 1 and T > 0")
        if self.norm not in NORMS:
            raise ValueError(f"block {self.name!r}: unknown normalisation {self.norm!r}")
        if self.selection is not None and self.selection[0] not in ("flag", "freq"):
            raise ValueError(f"block {self.name!r}: unknown selection {self.selection!r}")

    @property
    def basis_key(self) -> tuple:
        """Everything that determines a column apart from its mode number."""
        return (float(self.T), float(self.t0), float(self.chrom_idx), float(self.fref), self.norm, self.selection)

    def frequencies(self) -> tuple[np.ndarray, np.ndarray]:
        """(f, df), each of shape (2 n_modes,), sin/cos repeated."""
        return fourier_frequencies(self.n_modes, self.T)

    def row_scale(self, freqs: np.ndarray, systems: np.ndarray | None = None) -> np.ndarray:
        freqs = np.asarray(freqs, dtype=np.float64)
        s = np.ones_like(freqs)
        if self.chrom_idx != 0.0:
            s = s * (self.fref / freqs) ** self.chrom_idx
        if self.norm == "temponest_dm":
            s = s * TN_DM_NORM
        if self.selection is not None:
            if self.selection[0] == "flag":
                if systems is None:
                    raise ValueError(f"block {self.name!r}: flag selection needs system labels")
                s = s * (np.asarray(systems) == self.selection[1])
            else:
                lo, hi = self.selection[1], self.selection[2]
                s = s * ((freqs > lo) & (freqs < hi))
        return s

    def basis(self, toas: np.ndarray, freqs: np.ndarray, systems: np.ndarray | None = None) -> np.ndarray:
        """F of shape (n_toa, 2 n_modes)."""
        f, _ = self.frequencies()
        t = np.asarray(toas, dtype=np.float64) - self.t0
        arg = 2.0 * np.pi * t[:, None] * f[None, ::2]
        F = np.empty((t.size, 2 * self.n_modes), dtype=np.float64)
        F[:, 0::2] = np.sin(arg)
        F[:, 1::2] = np.cos(arg)
        return F * self.row_scale(freqs, systems)[:, None]
