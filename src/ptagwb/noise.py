"""Fixed white noise: EFAC, T2 EQUAD and ECORR, with enterprise's epoch quantisation.

Model (enterprise ``MeasurementNoise(efac, log10_t2equad)`` + ``EcorrKernelNoise``; DETCHAR
Eq. 2), for TOA i in receiver/backend system s(i)::

    N_ij = delta_ij EFAC_s^2 (sigma_i^2 + 10^(2 log10_t2equad_s))
           + sum_e 10^(2 log10_ecorr_s(e)) U_ie U_je

ECORR sits outside the EFAC scaling. The quantisation matrix U is built per system
(backend flag) from the barycentric TOAs in seconds exactly like enterprise's
``create_quantization_matrix(toas[mask], dt=1, nmin=2)``: sort, open a bucket at a TOA, add
the following TOAs while ``t - t_bucket_start < 1 s``, otherwise open a new bucket; buckets
with fewer than ``nmin = 2`` TOAs are discarded (no ECORR for those TOAs).

discovery's default ``makegp_ecorr`` (``enterprise=False``) keeps singleton buckets
(``nmin=1``) and starts a bucket when ``t - t_start > dt``; both are available here for
oracle comparisons via ``nmin`` (the ``>``/``<`` difference only matters at exactly 1 s).

The white-noise dictionary for the GWB analysis is ``v1p1_wn_dict.json`` (GWB Fig. 1 data
bundle). Its 645 EFAC/EQUAD/ECORR values are identical to the ``Constant=`` values in the
enterprise runtime info of the released NG15 CURN/HD production chains (checked in
``tests/test_oracle_data.py``); its 52 red-noise entries are ignored (IRN is sampled).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import RAW_DIR

WN_DICT_PATH = RAW_DIR / "ng15_gwb_fig1_data" / "extracted" / "figure1_data" / "v1p1_wn_dict.json"
NANOGRAV_BACKENDS = ("ASP", "GASP", "GUPPI", "PUPPI", "YUPPI", "CHIME", "VEGAS")


def load_noise_dict(path: Path | str | None = None) -> dict[str, float]:
    with open(path or WN_DICT_PATH) as f:
        return {k: float(v) for k, v in json.load(f).items()}


# ---------------------------------------------------------------------- quantisation


def quantize(toas: np.ndarray, dt: float = 1.0, nmin: int = 2) -> list[np.ndarray]:
    """Epoch buckets (lists of indices into ``toas``), enterprise ``create_quantization_matrix``."""
    toas = np.asarray(toas, dtype=np.float64)
    if toas.size == 0:
        return []
    isort = np.argsort(toas)
    buckets: list[list[int]] = [[int(isort[0])]]
    ref = toas[isort[0]]
    for i in isort[1:]:
        if toas[i] - ref < dt:
            buckets[-1].append(int(i))
        else:
            ref = toas[i]
            buckets.append([int(i)])
    return [np.array(b, dtype=np.int64) for b in buckets if len(b) >= nmin]


def selection_masks(backend_flags: np.ndarray, rule: str = "backend") -> dict[str, np.ndarray]:
    """Per-system boolean masks.

    ``backend``: enterprise ``by_backend`` (every distinct flag value).
    ``nanograv``: enterprise ``nanograv_backends`` (flags containing a NANOGrav backend name;
    the selection ``model_2a``/``white_noise_block`` uses for ECORR).
    """
    vals = np.unique(backend_flags)
    if rule == "nanograv":
        vals = [v for v in vals if any(b in v for b in NANOGRAV_BACKENDS)]
    elif rule != "backend":
        raise ValueError(f"unknown selection rule {rule!r}")
    return {str(v): backend_flags == v for v in vals}


# ---------------------------------------------------------------------- white noise


@dataclass
class WhiteNoise:
    """Fixed white-noise covariance of one pulsar: diagonal + disjoint ECORR epoch blocks."""

    ndiag: np.ndarray  # (n,) EFAC^2 (sigma^2 + EQUAD^2)  [s^2]
    epoch: np.ndarray  # (n,) epoch index per TOA, -1 if the TOA is in no ECORR bucket
    ecorr_var: np.ndarray  # (n_epoch,) 10^(2 log10_ecorr) of the bucket's system [s^2]
    epoch_system: np.ndarray  # (n_epoch,) system label of each bucket
    systems: list[str]

    @property
    def n_epoch(self) -> int:
        return len(self.ecorr_var)

    def U(self) -> np.ndarray:
        """Dense quantisation matrix (n, n_epoch); for tests only."""
        U = np.zeros((len(self.ndiag), self.n_epoch))
        m = self.epoch >= 0
        U[np.flatnonzero(m), self.epoch[m]] = 1.0
        return U

    def dense(self) -> np.ndarray:
        """Dense N (n, n); for tests only."""
        U = self.U()
        return np.diag(self.ndiag) + (U * self.ecorr_var) @ U.T

    # -- operations exploiting the block structure (all float64 numpy) --

    def _epoch_sums(self, x: np.ndarray) -> np.ndarray:
        """sum_{i in e} x_i for each epoch e (x has TOAs on axis 0)."""
        m = self.epoch >= 0
        out = np.zeros((self.n_epoch,) + x.shape[1:], dtype=x.dtype)
        np.add.at(out, self.epoch[m], x[m])
        return out

    def logdet(self) -> float:
        """log|N| = sum log D_i + sum_e log(1 + J_e sum_{i in e} 1/D_i)."""
        s = self._epoch_sums(1.0 / self.ndiag)
        return float(np.sum(np.log(self.ndiag)) + np.sum(np.log1p(self.ecorr_var * s)))

    def whiten(self, x: np.ndarray) -> np.ndarray:
        """W x with W^T W = N^-1, applied blockwise.

        For an epoch block N_e = D^1/2 (I + J w w^T) D^1/2 with w = D^-1/2 1, take
        W_e = (I + J w w^T)^-1/2 D^-1/2 = [I + (1/sqrt(1 + J|w|^2) - 1) w w^T / |w|^2] D^-1/2.
        """
        x = np.asarray(x, dtype=np.float64)
        dm12 = 1.0 / np.sqrt(self.ndiag)
        shape = (-1,) + (1,) * (x.ndim - 1)
        y = x * dm12.reshape(shape)
        if self.n_epoch == 0:
            return y
        w2 = self._epoch_sums(1.0 / self.ndiag)  # |w|^2 per epoch
        alpha = (1.0 / np.sqrt(1.0 + self.ecorr_var * w2) - 1.0) / w2
        wy = self._epoch_sums(y * dm12.reshape(shape))  # w^T y per epoch
        m = self.epoch >= 0
        corr = np.zeros_like(y)
        corr[m] = (alpha.reshape(shape)[self.epoch[m]] * wy[self.epoch[m]]) * dm12[m].reshape(shape)
        return y + corr

    def solve(self, x: np.ndarray) -> np.ndarray:
        """N^-1 x by Sherman-Morrison per epoch."""
        x = np.asarray(x, dtype=np.float64)
        shape = (-1,) + (1,) * (x.ndim - 1)
        dinv = (1.0 / self.ndiag).reshape(shape)
        y = x * dinv
        if self.n_epoch == 0:
            return y
        s = self._epoch_sums(1.0 / self.ndiag)
        coef = self.ecorr_var / (1.0 + self.ecorr_var * s)
        uy = self._epoch_sums(y)
        m = self.epoch >= 0
        corr = np.zeros_like(y)
        corr[m] = coef.reshape(shape)[self.epoch[m]] * uy[self.epoch[m]] * dinv[m]
        return y - corr


def build_white_noise(
    psr,
    noisedict: dict[str, float],
    *,
    efeq_selection: str = "backend",
    ecorr_selection: str = "nanograv",
    nmin: int = 2,
    dt: float = 1.0,
    strict: bool = True,
) -> WhiteNoise:
    """White noise of one pulsar from the fixed noise dictionary.

    Raises ``KeyError`` if any EFAC / log10_t2equad / log10_ecorr parameter of a selected
    system is missing. With ``strict``, also raises if the dictionary has white-noise
    entries for this pulsar that the selection does not use (catches flag-resolution bugs).
    """
    name = psr.name
    flags = np.asarray(psr.backend_flags)
    toaerrs = np.asarray(psr.toaerrs, dtype=np.float64)
    toas = np.asarray(psr.toas, dtype=np.float64)
    n = len(toas)

    used, missing = set(), []
    ndiag = np.full(n, np.nan)
    for sysname, mask in selection_masks(flags, efeq_selection).items():
        ke, kq = f"{name}_{sysname}_efac", f"{name}_{sysname}_log10_t2equad"
        for k in (ke, kq):
            if k not in noisedict:
                missing.append(k)
        if ke in noisedict and kq in noisedict:
            efac, equad2 = noisedict[ke], 10.0 ** (2.0 * noisedict[kq])
            ndiag[mask] = efac**2 * (toaerrs[mask] ** 2 + equad2)
            used.update((ke, kq))

    epoch = np.full(n, -1, dtype=np.int64)
    ecorr_var, epoch_system = [], []
    ecorr_masks = selection_masks(flags, ecorr_selection)
    for sysname in sorted(ecorr_masks):
        mask = ecorr_masks[sysname]
        kc = f"{name}_{sysname}_log10_ecorr"
        if kc not in noisedict:
            missing.append(kc)
            continue
        used.add(kc)
        idx = np.flatnonzero(mask)
        for b in quantize(toas[idx], dt=dt, nmin=nmin):
            epoch[idx[b]] = len(ecorr_var)
            ecorr_var.append(10.0 ** (2.0 * noisedict[kc]))
            epoch_system.append(sysname)

    if missing:
        raise KeyError(f"{name}: white-noise parameters missing from the noise dictionary: {missing}")
    if np.any(~np.isfinite(ndiag)):
        raise ValueError(f"{name}: {np.sum(~np.isfinite(ndiag))} TOAs not covered by any EFAC/EQUAD system")
    if strict:
        wn_suffixes = ("_efac", "_log10_t2equad", "_log10_tnequad", "_equad", "_log10_ecorr")
        unused = [k for k in noisedict if k.startswith(name + "_") and k.endswith(wn_suffixes) and k not in used]
        if unused:
            raise KeyError(f"{name}: noise-dictionary entries not used by the selection: {unused}")

    return WhiteNoise(
        ndiag=ndiag,
        epoch=epoch,
        ecorr_var=np.asarray(ecorr_var, dtype=np.float64),
        epoch_system=np.asarray(epoch_system, dtype="U"),
        systems=sorted(selection_masks(flags, efeq_selection)),
    )
