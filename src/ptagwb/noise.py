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
from dataclasses import dataclass, field
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


# ====================================================================== M3: general white noise
#
# Multi-PTA white noise (docs/M3_PLAN.md Sec. 4.3-4.4, L1-L3):
#
# * EQUAD convention per term (L1):  "t2": EFAC^2 (sigma^2 + Q^2)   (NG15, InPTA, YA refit)
#                                    "tn": EFAC^2 sigma^2 + Q^2     (TempoNest: EPTA, PPTA, MPTA)
#   Equal covariances require Q_TN = EFAC * Q_T2.
# * System labels are namespaced per PTA, ``"<pta>:<flag value>"`` (L2), so equal backend labels in
#   different PTAs never share a parameter (``namespace``).
# * ECORR terms may OVERLAP (L3, PPTA: backend, group and global ECORRs on the same TOA). Every
#   term is an additive covariance contribution sum_e J_e u_e u_e^T with its own selection and
#   epoch quantisation, i.e. a fixed "basis GP" with one column per (term, selection, epoch)
#   (enterprise ``EcorrBasisModel``). The total N is block diagonal over the connected components
#   of the TOA-epoch incidence graph (in practice: one observation), so it is whitened exactly by
#   a dense Cholesky factor per component. Disjoint ECORR (``WhiteNoise``) is the special case of
#   one epoch per component.

NAMESPACE_SEP = ":"


def namespace(pta: str, label: str) -> str:
    """``"<pta>:<label>"``: the system label of a TOA in a multi-PTA pulsar."""
    if NAMESPACE_SEP in pta:
        raise ValueError(f"PTA name {pta!r} must not contain {NAMESPACE_SEP!r}")
    return f"{pta}{NAMESPACE_SEP}{label}"


EQUAD_CONVENTIONS = ("t2", "tn")


def white_variance(sigma: np.ndarray, efac: float, log10_equad: float, convention: str) -> np.ndarray:
    """Diagonal white-noise variance [s^2] for TOA uncertainties ``sigma`` [s]."""
    q2 = 10.0 ** (2.0 * log10_equad)
    if convention == "t2":
        return efac**2 * (sigma**2 + q2)
    if convention == "tn":
        return efac**2 * sigma**2 + q2
    raise ValueError(f"unknown EQUAD convention {convention!r} (expected one of {EQUAD_CONVENTIONS})")


@dataclass(frozen=True)
class EcorrTerm:
    """One additive ECORR covariance term: per selection label a log10_ecorr value [log10 s].

    ``masks`` maps selection label -> boolean TOA mask (selections of one term may overlap other
    terms' selections but should be disjoint among themselves, as enterprise selections are).
    Quantisation of each selection: ``quantize(toas[mask], dt, nmin)`` (enterprise
    ``create_quantization_matrix``: dt = 1 s, nmin = 2).
    """

    name: str
    masks: dict
    log10_ecorr: dict
    dt: float = 1.0
    nmin: int = 2


@dataclass
class GeneralWhiteNoise:
    """N = diag(ndiag) + sum_e J_e u_e u_e^T with possibly overlapping epochs (see above).

    Epoch e covers TOAs ``ep_toa[ep_ptr[e]:ep_ptr[e+1]]``; ``ep_var[e]`` = J_e [s^2].
    """

    ndiag: np.ndarray
    ep_toa: np.ndarray
    ep_ptr: np.ndarray
    ep_var: np.ndarray
    ep_label: np.ndarray
    systems: list = field(default_factory=list)

    def __post_init__(self):
        self.ndiag = np.asarray(self.ndiag, dtype=np.float64)
        n = len(self.ndiag)
        if np.any(~np.isfinite(self.ndiag)) or np.any(self.ndiag <= 0):
            raise ValueError("white-noise diagonal must be finite and positive")
        if np.any(self.ep_var <= 0) or np.any(~np.isfinite(self.ep_var)):
            raise ValueError("ECORR variances must be finite and positive")
        # connected components of the TOA-epoch graph (union-find over TOAs)
        parent = np.arange(n)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for e in range(self.n_epoch):
            idx = self.ep_toa[self.ep_ptr[e] : self.ep_ptr[e + 1]]
            r0 = find(int(idx[0]))
            for i in idx[1:]:
                ri = find(int(i))
                if ri != r0:
                    parent[ri] = r0
        roots = np.array([find(i) for i in range(n)])
        comp_of_root: dict[int, list[int]] = {}
        for i, r in enumerate(roots):
            comp_of_root.setdefault(int(r), []).append(i)
        self._components = [np.array(v, dtype=np.int64) for v in comp_of_root.values() if len(v) > 1]
        self._singletons = np.array(sorted(v[0] for v in comp_of_root.values() if len(v) == 1), dtype=np.int64)
        # a singleton component may still carry ECORR epochs (nmin = 1): their variance adds to
        # the diagonal of that TOA
        self._sdiag = self.ndiag.copy()
        for e in range(self.n_epoch):
            idx = self.ep_toa[self.ep_ptr[e] : self.ep_ptr[e + 1]]
            if len(idx) == 1:
                self._sdiag[idx[0]] += self.ep_var[e]
        self._sdiag = self._sdiag[self._singletons]
        # epochs per component
        comp_id = np.full(n, -1, dtype=np.int64)
        for k, idx in enumerate(self._components):
            comp_id[idx] = k
        self._comp_epochs: list[list[int]] = [[] for _ in self._components]
        for e in range(self.n_epoch):
            i0 = int(self.ep_toa[self.ep_ptr[e]])
            if comp_id[i0] >= 0:
                self._comp_epochs[comp_id[i0]].append(e)
        self._chol = [self._component_chol(k) for k in range(len(self._components))]

    @property
    def n(self) -> int:
        return len(self.ndiag)

    @property
    def n_epoch(self) -> int:
        return len(self.ep_var)

    @property
    def max_component(self) -> int:
        return max((len(c) for c in self._components), default=1)

    def _component_chol(self, k: int) -> np.ndarray:
        idx = self._components[k]
        pos = {int(i): j for j, i in enumerate(idx)}
        C = np.diag(self.ndiag[idx])
        for e in self._comp_epochs[k]:
            loc = np.array([pos[int(i)] for i in self.ep_toa[self.ep_ptr[e] : self.ep_ptr[e + 1]]])
            C[np.ix_(loc, loc)] += self.ep_var[e]
        return np.linalg.cholesky(C)

    def U(self) -> np.ndarray:
        """Dense epoch incidence matrix (n, n_epoch); tests only."""
        U = np.zeros((self.n, self.n_epoch))
        for e in range(self.n_epoch):
            U[self.ep_toa[self.ep_ptr[e] : self.ep_ptr[e + 1]], e] = 1.0
        return U

    def dense(self) -> np.ndarray:
        """Dense N (n, n); tests only."""
        U = self.U()
        return np.diag(self.ndiag) + (U * self.ep_var) @ U.T

    def logdet(self) -> float:
        ld = float(np.sum(np.log(self._sdiag)))
        for L in self._chol:
            ld += 2.0 * float(np.sum(np.log(np.diag(L))))
        return ld

    def whiten(self, x: np.ndarray) -> np.ndarray:
        """W x with W = blockdiag(L_c^-1) (L_c L_c^T = N_c), so W^T W = N^-1."""
        from scipy.linalg import solve_triangular

        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        s = self._singletons
        shape = (-1,) + (1,) * (x.ndim - 1)
        y[s] = x[s] / np.sqrt(self._sdiag).reshape(shape)
        for idx, L in zip(self._components, self._chol, strict=True):
            y[idx] = solve_triangular(L, x[idx], lower=True)
        return y

    def colour(self, z: np.ndarray) -> np.ndarray:
        """L z with L L^T = N (blockwise Cholesky): turns unit normal draws into noise draws."""
        z = np.asarray(z, dtype=np.float64)
        y = np.empty_like(z)
        s = self._singletons
        shape = (-1,) + (1,) * (z.ndim - 1)
        y[s] = z[s] * np.sqrt(self._sdiag).reshape(shape)
        for idx, L in zip(self._components, self._chol, strict=True):
            y[idx] = L @ z[idx]
        return y

    def solve(self, x: np.ndarray) -> np.ndarray:
        from scipy.linalg import cho_solve

        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        s = self._singletons
        shape = (-1,) + (1,) * (x.ndim - 1)
        y[s] = x[s] / self._sdiag.reshape(shape)
        for idx, L in zip(self._components, self._chol, strict=True):
            y[idx] = cho_solve((L, True), x[idx])
        return y


def build_general_white_noise(
    toas: np.ndarray,
    toaerrs: np.ndarray,
    systems: np.ndarray,
    efeq: dict,
    ecorr_terms: list[EcorrTerm] = (),
    *,
    convention: str | dict = "t2",
) -> GeneralWhiteNoise:
    """General white noise of one (possibly multi-leg) pulsar.

    ``systems``: per-TOA (namespaced) system labels. ``efeq``: system -> (efac, log10_equad).
    ``convention``: "t2" / "tn", or a dict system -> convention (legs from different PTAs may use
    different conventions in one pulsar). Every system present must have an entry (KeyError
    otherwise); entries for absent systems are an error too (catches selection bugs).
    """
    toas = np.asarray(toas, dtype=np.float64)
    sig = np.asarray(toaerrs, dtype=np.float64)
    systems = np.asarray(systems).astype("U")
    present = sorted(set(systems.tolist()))
    missing = [s for s in present if s not in efeq]
    unused = [s for s in efeq if s not in present]
    if missing:
        raise KeyError(f"no EFAC/EQUAD for systems {missing}")
    if unused:
        raise KeyError(f"EFAC/EQUAD given for systems not present in the data: {unused}")
    ndiag = np.full(len(toas), np.nan)
    for s in present:
        m = systems == s
        conv = convention[s] if isinstance(convention, dict) else convention
        efac, leq = efeq[s]
        ndiag[m] = white_variance(sig[m], efac, leq, conv)
    ep_toa, ep_ptr, ep_var, ep_label = [], [0], [], []
    for term in ecorr_terms:
        for label in sorted(term.masks):
            mask = np.asarray(term.masks[label], dtype=bool)
            if not mask.any():
                continue
            if label not in term.log10_ecorr:
                raise KeyError(f"ECORR term {term.name!r}: no value for selection {label!r}")
            idx = np.flatnonzero(mask)
            for b in quantize(toas[idx], dt=term.dt, nmin=term.nmin):
                ep_toa.extend(idx[b].tolist())
                ep_ptr.append(len(ep_toa))
                ep_var.append(10.0 ** (2.0 * term.log10_ecorr[label]))
                ep_label.append(f"{term.name}/{label}")
    return GeneralWhiteNoise(
        ndiag=ndiag,
        ep_toa=np.asarray(ep_toa, dtype=np.int64),
        ep_ptr=np.asarray(ep_ptr, dtype=np.int64),
        ep_var=np.asarray(ep_var, dtype=np.float64),
        ep_label=np.asarray(ep_label, dtype="U"),
        systems=present,
    )


def general_from_white_noise(wn: WhiteNoise) -> GeneralWhiteNoise:
    """The disjoint-ECORR ``WhiteNoise`` as a ``GeneralWhiteNoise`` (same N)."""
    order = np.argsort(wn.epoch, kind="stable")
    order = order[wn.epoch[order] >= 0]
    counts = np.bincount(wn.epoch[order], minlength=wn.n_epoch)
    return GeneralWhiteNoise(
        ndiag=wn.ndiag.copy(),
        ep_toa=order.astype(np.int64),
        ep_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        ep_var=wn.ecorr_var.copy(),
        ep_label=wn.epoch_system.copy(),
        systems=list(wn.systems),
    )
