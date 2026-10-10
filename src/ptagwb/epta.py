"""EPTA DR2new published-analysis model (docs/M3B_PLAN.md Sec. 2.1, 4.5; components N1, N3, N9).

Builds, from the version-pinned manifest ``configs/m3b/manifest_epta.json`` and our tempo2 export
(``scripts/m3b_t2_export.py``, gate T1), the likelihood that EPTA's ``model_single.py`` built with
the EPTA forks of enterprise (EPTADR2-v1.1) and enterprise_extensions (051173f), following
enterprise conventions exactly (fork audit, docs/M3B0_VALIDATION.md Sec. 2):

* timing model: ``MarginalizingTimingModel(use_svd=True)``: flat prior on the thin-SVD basis
  U of the design matrix, constant -1/2 n_tm ln(1e40) included (``GeneralTerms.const("chain")``);
* white noise per ``-group`` backend: EFAC^2 sigma^2 + 10^(2 log10_tnequad) (``tnequad=True``),
  no ECORR, values fixed from the release's noise files;
* sampled power-law GPs on each pulsar's own span T_a (``Tspan=None``): ``red_noise``
  (achromatic), ``dm_gp`` (nu^-2) and ``chrom_gp`` (nu^-4), both chromatic ones with the TempoNest
  normalisation sqrt(12) pi / (1400^2 2.41e-4) (``createfourierdesignmatrix_dm_tn``), nu = SSB radio
  frequency; mode counts from the release's ``red_dict`` / ``dm_dict`` / ``chrom_dict``;
* common process: 9 modes on the array span (``model_utils.get_tspan``), power law; CURN = per-
  pulsar ``BasisGP`` with shared hyperparameters, HD = ``BasisCommonGP`` with ``hd_orf`` (diag 1);
* the J1713+0747 exponential dip (``chrom.dm_exponential_dip``, chromatic index 1, sign -1):
  d(t) = -10^log10_Amp H(t - t0) exp(-(t - t0) / tau) (1400 / nu)^1, tau = 10^log10_tau days,
  t0 in days (MJD); subtracted from the residuals.

The dip enters through the residual-dependent stage-1 terms (N3, "affine update without a dense
complement"): with W the white-noise whitening (diagonal here) and Q, Q_F the thin orthonormal
factors of stage 1, c(theta) = Q_F^T P_Q W (r - d(theta)) and s_perp(theta) = |(I - Q_F Q_F^T)
P_Q W (r - d(theta))|^2, P_Q = I - Q Q^T, both evaluated with the thin factors in O(n (m + K)) per
call (the same projection steps as ``combined.precompute_general``). log10_Amp and log10_tau are
smooth NUTS coordinates; t0 makes the likelihood discontinuous whenever it crosses a TOA and is
updated by exact Metropolis-Hastings outside NUTS (``mh_t0_step``; plan Sec. 5.1).

Parameter vectors are in the released chains' column order (``manifest["parameters"]``, i.e. the
enterprise ``pta.param_names`` order of the chain-producing run).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import combined as _C
from .config import REPO_ROOT, enable_x64
from .data import Pulsar
from .gp import FourierBlock
from .noise import GeneralWhiteNoise

enable_x64()

MANIFEST_PATH = REPO_ROOT / "configs" / "m3b" / "manifest_epta.json"
EXPORT_DIR = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "export" / "ours_canonical"
DAY = 86400.0
PROCESSES = ("red_noise", "dm_gp", "chrom_gp")
BUCKETS = (64, 128, 192, 256, 320, 416)  # N8 stage-2 size buckets for the DR2new K_a (38-416)


def load_manifest(path: Path | str = MANIFEST_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"EPTA manifest {p} missing (scripts/m3b_manifest_epta.py); the model fails closed")
    man = json.loads(p.read_text())
    for k in ("pulsars", "parameters", "blocks", "common", "dip", "white_noise"):
        if k not in man:
            raise KeyError(f"manifest lacks {k!r}")
    return man


def load_export(path: Path | str) -> Pulsar:
    """One pulsar of our tempo2 export as a ``data.Pulsar`` (arrays sorted by BAT; ``freqs`` are
    SSB frequencies, ``pos`` the enterprise-convention ORF position)."""
    with np.load(path, allow_pickle=False) as z:
        fl = json.loads(str(z["flags_json"]))
        return Pulsar(
            name=str(z["name"]), toas=np.asarray(z["toas"]), stoas=np.asarray(z["stoas"]),
            residuals=np.asarray(z["residuals"]), toaerrs=np.asarray(z["toaerrs"]), freqs=np.asarray(z["freqs"]),
            freqs_topo=np.asarray(z["site_freqs"]), backend_flags=np.asarray(z["backend_flags"]).astype(str),
            telescope=np.asarray(z["telescope"]).astype(str), Mmat=np.asarray(z["Mmat"]),
            fitpars=[str(x) for x in z["fitpars"]], pos=np.asarray(z["pos"]), pos_enterprise=np.asarray(z["pos"]),
            flags={k: np.asarray(v) for k, v in fl.items()}, meta=json.loads(str(z["meta"])))


def load_pulsars(directory: Path | str = EXPORT_DIR, names: list | None = None) -> list[Pulsar]:
    d = Path(directory)
    names = names if names is not None else sorted(p.stem for p in d.glob("J*.npz"))
    if not names:
        raise FileNotFoundError(f"no exports in {d} (scripts/m3b_t1.py)")
    return [load_export(d / f"{n}.npz") for n in names]


def array_span(psrs) -> float:
    """enterprise_extensions ``model_utils.get_tspan``: max TOA - min TOA over all pulsars [s]."""
    return float(max(p.toas.max() for p in psrs) - min(p.toas.min() for p in psrs))


def white_noise(psr: Pulsar, values: dict) -> GeneralWhiteNoise:
    """EFAC + TN EQUAD per backend (``white_noise_block(tnequad=True, select='backend')``).
    Fails if a backend lacks a value; values for unused keys are ignored."""
    flags = np.asarray(psr.backend_flags)
    sig = np.asarray(psr.toaerrs, dtype=np.float64)
    nd = np.full(len(sig), np.nan)
    for b in np.unique(flags):
        ke, kq = f"{psr.name}_{b}_efac", f"{psr.name}_{b}_log10_tnequad"
        if ke not in values or kq not in values:
            raise KeyError(f"{psr.name}: no white-noise value for backend {b!r}")
        m = flags == b
        nd[m] = values[ke] ** 2 * sig[m] ** 2 + 10.0 ** (2.0 * values[kq])
    return GeneralWhiteNoise(ndiag=nd, ep_toa=np.zeros(0, dtype=np.int64), ep_ptr=np.zeros(1, dtype=np.int64),
                             ep_var=np.zeros(0), ep_label=np.zeros(0, dtype=object), systems=sorted(np.unique(flags)))


def pulsar_model(psr: Pulsar, man: dict, T_common: float) -> _C.PulsarGPModel:
    """Sampled blocks on the pulsar's own span + the common block on the array span."""
    T_a = float(psr.toas.max() - psr.toas.min())
    sampled = {}
    for b in man["blocks"].get(psr.name, []):
        if b["span"] != "pulsar":
            raise ValueError(f"{psr.name}/{b['name']}: unsupported span rule {b['span']!r}")
        sampled[b["name"]] = FourierBlock(name=b["name"], n_modes=int(b["modes"]), T=T_a,
                                          chrom_idx=float(b["chrom_idx"]), fref=float(b.get("fref", 1400.0)),
                                          norm=b["norm"])
    c = man["common"]
    common = FourierBlock(name="common", n_modes=int(c["modes"]), T=T_common)
    return _C.PulsarGPModel(sampled=sampled, common=common)


@dataclass
class DipData:
    """Stage-1 operators of the dip pulsar (diagonal white noise)."""

    index: int
    toas: jnp.ndarray
    chrom: jnp.ndarray
    winv: jnp.ndarray  # 1 / sqrt(N_ii)
    wr0: jnp.ndarray  # W r
    Q: jnp.ndarray
    QF: jnp.ndarray
    K: int
    sign: float


class EPTAModel:
    """Marginalised EPTA DR2new likelihood (CURN or HD) on the chain's parameter vector."""

    def __init__(self, psrs: list[Pulsar], man: dict, orf: str = "crn", *, reduce: str = "prod",
                 tri_inv: str = "recursive", buckets: tuple | None = None, gamma_common: float | None = None):
        if [p.name for p in psrs] != list(man["pulsars"]):
            raise ValueError("pulsars must be the manifest roster in its order")
        if orf not in ("crn", "hd"):
            raise ValueError(orf)
        self.man, self.orf, self.psrs = man, orf, psrs
        self.T_common = array_span(psrs)
        wn_values = man["white_noise"]["values"]
        dip = man["dip"]
        self.terms = []
        for p in psrs:
            wn = white_noise(p, wn_values)
            self.terms.append(_C.precompute_general(p, wn, pulsar_model(p, man, self.T_common), timing="svd",
                                                    projector=p.name == dip["pulsar"]))
        self.like = _C.GeneralPTALikelihood(self.terms, orf="curn" if orf == "crn" else "hd", common="powerlaw",
                                            convention="chain", reduce=reduce, tri_inv=tri_inv, buckets=buckets)
        # ---- parameter layout (chain column order)
        # gamma_common: fixed common spectral index (the E-5 runs, gamma = 13/3); the parameter is
        # then not sampled (enterprise gamma_common -> parameter.Constant)
        self.gamma_common = gamma_common
        keep = [q for q in man["parameters"] if orf in q["models"]
                and not (gamma_common is not None and q["name"] == f"gw_{orf}_gamma")]
        self.param_names = [q["name"] for q in keep]
        self.lo = np.array([q["bounds"][0] for q in keep])
        self.hi = np.array([q["bounds"][1] for q in keep])
        ix = {n: i for i, n in enumerate(self.param_names)}
        self._proc_idx = {}
        for nm in PROCESSES:
            if nm not in self.like.processes:
                continue
            ps = self.like.process_pulsars(nm)
            self._proc_idx[nm] = (np.array([ix[f"{q}_{nm}_log10_A"] for q in ps]),
                                  np.array([ix[f"{q}_{nm}_gamma"] for q in ps]))
        gw = f"gw_{orf}"
        self._gw = (ix[f"{gw}_log10_A"], ix[f"{gw}_gamma"] if gamma_common is None else None)
        pre = dip["param_prefix"]
        self._dip_idx = (ix[f"{pre}_log10_Amp"], ix[f"{pre}_log10_tau"], ix[f"{pre}_t0"])
        self.t0_index = self._dip_idx[2]
        j = [p.name for p in psrs].index(dip["pulsar"])
        t, wn, psr = self.terms[j], white_noise(psrs[j], wn_values), psrs[j]
        if t.projector is None or t.projector.kaug:
            raise ValueError("dip pulsar needs a plain projector (no fixed blocks)")
        self.dip = DipData(index=j, toas=jnp.asarray(psr.toas), chrom=jnp.asarray((1400.0 / psr.freqs) ** float(dip["chrom_idx"])),
                           winv=jnp.asarray(1.0 / np.sqrt(wn.ndiag)), wr0=jnp.asarray(wn.whiten(psr.residuals)),
                           Q=jnp.asarray(t.projector.Q), QF=jnp.asarray(t.projector.QF), K=t.K,
                           sign=float(dip["sign"]))
        self.logL = jax.jit(self._logL)
        self.value_and_grad = jax.jit(jax.value_and_grad(self._logL))

    # -------------------------------------------------------------- pieces
    def params(self, x) -> dict:
        x = jnp.asarray(x, dtype=jnp.float64)
        out = {}
        for nm, (ia, ig) in self._proc_idx.items():
            out[f"{nm}_log10_A"] = x[ia]
            out[f"{nm}_gamma"] = x[ig]
        out["log10_A"] = x[self._gw[0]]
        out["gamma"] = x[self._gw[1]] if self._gw[1] is not None else jnp.asarray(float(self.gamma_common))
        return out

    def dip_delay(self, x):
        """d(t) [s] of the dip pulsar (enterprise ``chrom_exp_decay`` with sign and index)."""
        ia, it, i0 = self._dip_idx
        A, tau, t0 = 10.0 ** x[ia], 10.0 ** x[it] * DAY, x[i0] * DAY
        dt = self.dip.toas - t0
        wf = jnp.where(dt >= 0.0, jnp.exp(-jnp.maximum(dt, 0.0) / tau), 0.0)
        return self.dip.sign * A * wf * self.dip.chrom

    def dip_terms(self, x):
        """(c, s_perp) of the dip pulsar for the dip-subtracted residuals (module docstring)."""
        D = self.dip
        wr = D.wr0 - D.winv * self.dip_delay(x)
        rp = wr - D.Q @ (D.Q.T @ wr)
        rp = rp - D.Q @ (D.Q.T @ rp)
        c = D.QF.T @ rp
        rperp = rp - D.QF @ c
        rperp = rperp - D.QF @ (D.QF.T @ rperp)
        return c, jnp.sum(rperp * rperp)

    def _logL(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        c_d, s_d = self.dip_terms(x)
        j = self.dip.index
        c = self.like.c.at[j, : self.dip.K].set(c_d)
        s = self.like.s_perp.at[j].set(s_d)
        return self.like._logL_cs(self.params(x), c, s)

    # -------------------------------------------------------------- prior
    @property
    def log_prior_volume(self) -> float:
        """Physical log prior inside the box: -sum ln(width) (all priors uniform)."""
        return float(-np.sum(np.log(self.hi - self.lo)))

    def log_prior(self, x) -> float:
        x = np.asarray(x, dtype=np.float64)
        inside = np.all((x >= self.lo) & (x <= self.hi), axis=-1)
        return np.where(inside, self.log_prior_volume, -np.inf)

    def in_box(self, x) -> bool:
        x = np.asarray(x, dtype=np.float64)
        return bool(np.all((x >= self.lo) & (x <= self.hi)))


# ------------------------------------------------------------------ exact MH for t0 (N13)


def mh_t0_step(model: EPTAModel, x: np.ndarray, logL_x: float, rng: np.random.Generator, *, rw_scale_days: float = 2.0,
               p_independence: float = 0.2) -> tuple[np.ndarray, float, bool]:
    """One exact Metropolis-Hastings update of the dip epoch t0 given all other coordinates
    (plan Sec. 5.1): with probability ``p_independence`` an independence draw from the uniform
    prior window, otherwise a Gaussian random walk (symmetric). Both components are symmetric with
    respect to the uniform prior on the window (the independence proposal's density is the prior
    density, constant inside the window), so the acceptance ratio is the likelihood ratio times the
    prior indicator. Returns (x_new, logL_new, accepted)."""
    i = model.t0_index
    lo, hi = model.lo[i], model.hi[i]
    y = np.array(x, dtype=np.float64, copy=True)
    if rng.random() < p_independence:
        y[i] = rng.uniform(lo, hi)
    else:
        y[i] = x[i] + rw_scale_days * rng.standard_normal()
    if not (lo <= y[i] <= hi):
        return np.asarray(x), logL_x, False
    ly = float(model.logL(y))
    if np.log(rng.random()) < ly - logL_x:
        return y, ly, True
    return np.asarray(x), logL_x, False


# ------------------------------------------------------------------ reference chains (N11)

REFERENCE_DIR = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "reference"
CHAIN_COLUMNS_EXTRA = ("lnpost", "lnlike", "acceptance", "pt_acceptance")  # PTMCMCSampler chain_1.txt


def reference_tar(model: str) -> Path:
    from .config import RAW_DIR

    return RAW_DIR / "epta_dr2_gwb_chains" / "extracted" / "chains" / "DR2new" / f"{model}.tar.gz"


def extract_reference(model: str) -> Path:
    """Extract ``<model>.tar.gz`` (crn_pl / hd_pl) of the released DR2new chains (Zenodo 8091568)."""
    import tarfile

    out = REFERENCE_DIR / model
    if not (out / "chain_1.txt").exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(reference_tar(model)) as tf:
            tf.extractall(REFERENCE_DIR, filter="data")
    return out


def load_reference(model: str, man: dict | None = None, *, verify: bool = True) -> tuple[list[str], np.ndarray, int]:
    """(parameter names, full chain array (rows x 71) in float64 as stored, burn-in rows).

    Values are parsed directly to float64 from the text (never via float32). With ``verify`` the
    sha256 of the chain file must equal the manifest's (fail closed)."""
    import hashlib

    d = extract_reference(model)
    if verify:
        man = man if man is not None else load_manifest()
        want = man["reference_chains"][model]["chain_sha256"]
        got = hashlib.sha256((d / "chain_1.txt").read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(f"{model}: chain_1.txt sha256 {got} != manifest {want}")
    names = (d / "pars.txt").read_text().split()
    X = np.loadtxt(d / "chain_1.txt", dtype=np.float64)
    if X.shape[1] != len(names) + len(CHAIN_COLUMNS_EXTRA):
        raise ValueError(f"{model}: {X.shape[1]} columns for {len(names)} parameters")
    burn = int(X.shape[0] * 0.25) if man is None else int(man["reference_chains"][model]["burn_in_rows"])
    return names, X, burn
