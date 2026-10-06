"""PINT front end and on-disk pulsar cache.

PINT turns each NG15 narrowband ``.par``/``.tim`` pair into the per-pulsar arrays that the
rest of the pipeline needs. Everything is exported in the conventions of enterprise's
``PintPulsar`` (see ``docs/M1_VALIDATION.md`` for the checks against enterprise and the
released discovery feathers):

==================  ===========================================================================
``toas``            barycentric arrival times, ``model.get_barycentric_toas(toas)`` [s] (TDB)
``stoas``           observatory arrival times, ``toas.get_mjds()`` [s] (UTC MJD * 86400)
``residuals``       pre-fit residuals of the released (already GLS-fitted) model [s]
``toaerrs``         raw TOA uncertainties from the .tim file, *before* EFAC/EQUAD [s]
``freqs``           barycentric radio frequency, ``model.barycentric_radio_freq`` [MHz]
                    (this is what enterprise calls ``psr.freqs``)
``freqs_topo``      observatory radio frequency [MHz]
``backend_flags``   enterprise flag resolution: ``group`` > ``g`` > ``sys`` > ``i`` > ``f`` >
                    ``fe``+"_"+``be`` (first non-empty wins)
``Mmat``            ``model.designmatrix(toas)`` (n_toa, n_par), including the phase offset
``fitpars``         names of the design-matrix columns
``pos``             unit vector to the pulsar (ICRS) at the timing-model reference position
``pos_enterprise``  the unit vector enterprise / the released NG15 products use (pyephem; for
                    B-name pulsars B1950-equinox coordinates, 0.5 deg off) -- comparisons only
==================  ===========================================================================

All per-TOA arrays are sorted by barycentric TOA with a stable (merge) sort, as in
enterprise. TOAs from the same observation that coincide to float64 resolution (~1 us at
4.6e9 s) may therefore be ordered differently from another tool; the likelihood is
invariant under such permutations, and the oracle tests match TOAs by tim-file key.

Pulsar selection (GWB paper, Sec. 2 and App. A): the 68 combined narrowband pulsars minus
those with a span < 3 yr (only J0614-3329), i.e. 67 pulsars. The ``*ao`` / ``*gbt`` split
par/tim files are *not* part of the baseline: they exist for the Arecibo-only / GBT-only
cross-validation (GWB Sec. 5.4, Fig. 10) and are loaded only on request.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR, RAW_DIR, REPO_ROOT

CACHE_DIR = DATA_DIR / "cache" / "pulsars"
SCHEMA_VERSION = 3  # bump when the exported arrays or their conventions change
SPLIT_RE = re.compile(r"(ao|gbt)$")
MIN_SPAN_YR = 3.0
JULIAN_YEAR_S = 365.25 * 86400.0
# flags kept in the cache besides the resolved backend flag
KEEP_FLAGS = ("f", "fe", "be", "group", "g", "sys", "i", "pta", "name", "chan", "subint")
EXPECTED_EXCLUDED = ("J0614-3329",)

# Explicit, verified exceptions to "fit every parameter the par file flags as free".
# J1024-0719_PINT_20220302.nb.par has the line "F3 0 1": value exactly 0, fit flag set, no
# uncertainty column, i.e. F3 was never actually fitted. Evidence: (i) the tempo2 version of the
# same par file (narrowband/alternate/tempo2) has no F3; (ii) the design matrix of the released
# GWB-analysis pulsar file (discovery/tutorial feather v1p1_de440_pint_bipm2019-J1024-0719) has
# 149 columns and no F3 column, whereas PINT 1.1.7 would build 150. Freezing F3 changes no
# residual (value 0) and restores the analysis' timing subspace. Each entry is checked: the
# parameter must be free with value 0 and no uncertainty, and the resulting column count must
# equal the released design matrix's.
FROZEN_PARAMS = {"J1024-0719": ("F3",)}
EXPECTED_NCOL_AFTER_FREEZE = {"J1024-0719": 149}


@dataclass
class Pulsar:
    """Per-pulsar arrays in enterprise ``PintPulsar`` conventions (sorted by ``toas``)."""

    name: str
    toas: np.ndarray
    stoas: np.ndarray
    residuals: np.ndarray
    toaerrs: np.ndarray
    freqs: np.ndarray
    freqs_topo: np.ndarray
    backend_flags: np.ndarray
    telescope: np.ndarray
    Mmat: np.ndarray
    fitpars: list[str]
    pos: np.ndarray
    pos_enterprise: np.ndarray | None = None  # enterprise/feather convention (see enterprise_position)
    flags: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def ntoa(self) -> int:
        return len(self.toas)

    @property
    def span_s(self) -> float:
        return float(self.toas.max() - self.toas.min())

    def __repr__(self) -> str:
        return f"<Pulsar {self.name}: {self.ntoa} TOAs, {self.Mmat.shape[1]} timing params>"

    # ------------------------------------------------------------------ npz I/O
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "toas": self.toas,
            "stoas": self.stoas,
            "residuals": self.residuals,
            "toaerrs": self.toaerrs,
            "freqs": self.freqs,
            "freqs_topo": self.freqs_topo,
            "backend_flags": self.backend_flags.astype("U"),
            "telescope": self.telescope.astype("U"),
            "Mmat": self.Mmat,
            "pos": self.pos,
        }
        if self.pos_enterprise is not None:
            arrays["pos_enterprise"] = self.pos_enterprise
        arrays.update({f"flag_{k}": v.astype("U") for k, v in self.flags.items()})
        header = {"name": self.name, "fitpars": list(self.fitpars), "meta": self.meta}
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, _header=np.array(json.dumps(header)), **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Pulsar:
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["_header"]))
            flags = {k[5:]: z[k] for k in z.files if k.startswith("flag_")}
            return cls(
                name=header["name"],
                toas=z["toas"],
                stoas=z["stoas"],
                residuals=z["residuals"],
                toaerrs=z["toaerrs"],
                freqs=z["freqs"],
                freqs_topo=z["freqs_topo"],
                backend_flags=z["backend_flags"],
                telescope=z["telescope"],
                Mmat=z["Mmat"],
                fitpars=header["fitpars"],
                pos=z["pos"],
                pos_enterprise=z["pos_enterprise"] if "pos_enterprise" in z.files else None,
                flags=flags,
                meta=header["meta"],
            )


# ---------------------------------------------------------------------- file discovery


def release_dir(release: str = "ng15_v2.1.0") -> Path:
    base = RAW_DIR / release / "extracted"
    cands = sorted(base.glob("NANOGrav*"))
    if not cands:
        raise FileNotFoundError(f"no extracted NG15 release under {base}; run scripts/fetch_data.py")
    return cands[0]


def find_par_tim(
    release: str = "ng15_v2.1.0", include_split: bool = False, split_only: bool = False
) -> dict[str, tuple[Path, Path]]:
    """Map pulsar name -> (par, tim) for the narrowband release.

    ``include_split`` adds the ``*ao`` / ``*gbt`` split-telescope files (names keep their
    suffix, e.g. ``B1937+21ao``); ``split_only`` returns only those.
    """
    rel = release_dir(release)
    out = {}
    for par in sorted((rel / "narrowband" / "par").glob("*.nb.par")):
        name = par.name.split("_PINT_")[0]
        is_split = bool(SPLIT_RE.search(name))
        if (is_split and not (include_split or split_only)) or (split_only and not is_split):
            continue
        tims = sorted((rel / "narrowband" / "tim").glob(f"{name}_PINT_*.nb.tim"))
        if len(tims) != 1:
            raise FileNotFoundError(f"{name}: expected exactly one tim file, found {tims}")
        out[name] = (par, tims[0])
    return out


def _relpath(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clock_dir(clock: str, release: str = "ng15_v2.1.0") -> Path | None:
    if clock == "release":
        return release_dir(release) / "clock"
    if clock == "pint":
        return None
    raise ValueError(f"clock must be 'release' or 'pint', got {clock!r}")


def input_hash(par: Path, tim: Path, clock: str = "release", release: str = "ng15_v2.1.0") -> str:
    """Hash of everything that determines the exported arrays."""
    import pint

    h = hashlib.sha256()
    h.update(f"schema={SCHEMA_VERSION};pint={pint.__version__};clock={clock}".encode())
    h.update(_sha256(par).encode())
    h.update(_sha256(tim).encode())
    cdir = clock_dir(clock, release)
    if cdir is not None:
        for f in sorted(cdir.iterdir()):
            if f.is_file():
                h.update(f"{f.name}:{_sha256(f)}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------- PINT ingestion


def resolve_backend_flags(flags: dict[str, np.ndarray], n: int) -> np.ndarray:
    """enterprise ``BasePulsar.backend_flags``: group > g > sys > i > f > fe_be."""
    ret = np.array([""] * n, dtype=object)
    if "fe" in flags and "be" in flags:
        ret[:] = [(a + "_" + b if (a and b) else "") for a, b in zip(flags["fe"], flags["be"])]
    for flag in ["f", "i", "sys", "g", "group"]:
        if flag in flags:
            ret[:] = np.where(flags[flag] == "", ret, flags[flag])
    return ret.astype("U")


def load_pulsar_pint(
    par: Path,
    tim: Path,
    clock: str = "release",
    release: str = "ng15_v2.1.0",
    freeze_placeholders: bool = True,
) -> Pulsar:
    """Load one pulsar with PINT and export it (no caching).

    ``clock='release'`` uses the clock files shipped with the NG15 release (frozen; what the
    release README says was used) through ``PINT_CLOCK_OVERRIDE``; ``clock='pint'`` uses
    PINT's global clock-correction repository.
    """
    import warnings

    cdir = clock_dir(clock, release)
    if cdir is not None:
        os.environ["PINT_CLOCK_OVERRIDE"] = str(cdir)
    else:
        os.environ.pop("PINT_CLOCK_OVERRIDE", None)

    import astropy.units as u
    import pint
    import pint.logging
    from pint.models import get_model_and_toas
    from pint.residuals import Residuals

    pint.logging.setup(level="ERROR")
    warnings.filterwarnings("ignore")

    model, toas = get_model_and_toas(str(par), str(tim), planets=True)
    frozen_placeholders = []
    if freeze_placeholders and model.PSR.value in FROZEN_PARAMS:
        for pname in FROZEN_PARAMS[model.PSR.value]:
            prm = getattr(model, pname)
            if prm.frozen or prm.value != 0 or prm.uncertainty_value:
                raise ValueError(
                    f"{model.PSR.value}: expected {pname} to be a free, zero-valued placeholder without "
                    f"uncertainty (got frozen={prm.frozen}, value={prm.value}, unc={prm.uncertainty_value}); "
                    "input differs from the verified NG15 release"
                )
            prm.frozen = True
            frozen_placeholders.append(pname)
    if model.EPHEM.value != "DE440" or model.CLOCK.value != "TT(BIPM2019)":
        raise ValueError(f"{par}: unexpected EPHEM/CLOCK {model.EPHEM.value}/{model.CLOCK.value}")
    name = model.PSR.value

    btoas = np.asarray(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
    stoas = np.asarray(toas.get_mjds().value, dtype=np.float64) * 86400.0
    resid = np.asarray(Residuals(toas, model).time_resids.to_value(u.s), dtype=np.float64)
    errs = np.asarray(toas.get_errors().to_value(u.s), dtype=np.float64)
    freqs = np.asarray(model.barycentric_radio_freq(toas).to_value(u.MHz), dtype=np.float64)
    freqs_topo = np.asarray(toas.get_freqs().to_value(u.MHz), dtype=np.float64)
    M, fitpars, _units = model.designmatrix(toas)
    M = np.asarray(M, dtype=np.float64)
    if frozen_placeholders and M.shape[1] != EXPECTED_NCOL_AFTER_FREEZE[name]:
        raise ValueError(f"{name}: {M.shape[1]} design-matrix columns after freezing {frozen_placeholders}, "
                         f"expected {EXPECTED_NCOL_AFTER_FREEZE[name]}")
    telescope = np.asarray(toas.get_obss()).astype("U")

    # all flags as strings, "" where absent (enterprise PintPulsar convention)
    allflags: dict[str, list[str]] = {}
    for i, fl in enumerate(toas.get_flags()):
        for k, v in fl.items():
            allflags.setdefault(k, [""] * toas.ntoas)[i] = str(v)
    allflags_np = {k: np.array(v, dtype="U") for k, v in allflags.items()}
    backend = resolve_backend_flags(allflags_np, toas.ntoas)

    # sky position at the reference epoch (no proper-motion propagation), ICRS
    icrs = model.coords_as_ICRS(epoch=None)
    ra, dec = icrs.ra.to_value(u.rad), icrs.dec.to_value(u.rad)
    pos = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    pos_ent = enterprise_position(model, name)

    isort = np.argsort(btoas, kind="mergesort")
    flags = {k: allflags_np[k][isort] for k in KEEP_FLAGS if k in allflags_np}

    meta = {
        "par": _relpath(par),
        "tim": _relpath(tim),
        "par_sha256": _sha256(par),
        "tim_sha256": _sha256(tim),
        "pint_version": pint.__version__,
        "clock": clock,
        "ephem": model.EPHEM.value,
        "clock_standard": model.CLOCK.value,
        "schema": SCHEMA_VERSION,
        "input_hash": input_hash(par, tim, clock, release),
        "frozen_placeholders": frozen_placeholders,
        "raj_rad": float(ra),
        "decj_rad": float(dec),
    }
    return Pulsar(
        name=name,
        toas=btoas[isort],
        stoas=stoas[isort],
        residuals=resid[isort],
        toaerrs=errs[isort],
        freqs=freqs[isort],
        freqs_topo=freqs_topo[isort],
        backend_flags=backend[isort],
        telescope=telescope[isort],
        Mmat=M[isort],
        fitpars=list(fitpars),
        pos=pos,
        pos_enterprise=pos_ent,
        flags=flags,
        meta=meta,
    )


def enterprise_position(model, name: str) -> np.ndarray:
    """Unit vector exactly as enterprise ``PintPulsar`` computes ``psr.pos`` (and hence as in the
    released NG15 feathers and production chains), reimplemented here with pyephem.

    RAJ/DECJ are used directly if present. Otherwise ELONG/ELAT go through pyephem
    ``Equatorial(Ecliptic(elong, elat), epoch=...)`` with epoch "1950" whenever "B" occurs in the
    pulsar name and "2000" otherwise. For the B-name pulsars this yields the B1950-equinox
    direction, 0.46-0.59 deg from ICRS. Only for apples-to-apples comparisons with the
    released products; ``pos`` (ICRS) is the physically correct default.
    """
    import astropy.units as u
    from ephem import Ecliptic, Equatorial

    if hasattr(model, "RAJ") and hasattr(model, "DECJ"):
        ra, dec = model.RAJ.quantity.to_value(u.rad), model.DECJ.quantity.to_value(u.rad)
    else:
        elong, elat = model.ELONG.quantity.to_value(u.rad), model.ELAT.quantity.to_value(u.rad)
        eq = Equatorial(Ecliptic(elong, elat), epoch="1950" if "B" in name else "2000")
        ra, dec = float(eq.ra), float(eq.dec)
    return np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])


def _ingest_one(args) -> tuple[str, str]:
    name, par, tim, clock, release, cache_dir = args
    psr = load_pulsar_pint(par, tim, clock=clock, release=release)
    path = Path(cache_dir) / clock / f"{name}.npz"
    psr.save(path)
    return name, str(path)


def cache_path(name: str, clock: str = "release", cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / clock / f"{name}.npz"


def load_pulsars(
    names: list[str] | None = None,
    *,
    clock: str = "release",
    release: str = "ng15_v2.1.0",
    include_split: bool = False,
    cache_dir: Path = CACHE_DIR,
    jobs: int | None = None,
    select_gwb: bool = True,
    verbose: bool = True,
) -> list[Pulsar]:
    """Load (ingesting with PINT where the cache is stale or missing) and return pulsars.

    With ``names=None`` and ``select_gwb=True`` this returns the 67 pulsars of the NG15 GWB
    analysis (span >= 3 yr), sorted by name. The cache is keyed by an input hash (par, tim,
    clock files, PINT version, schema), so editing any input triggers re-ingestion.
    """
    pairs = find_par_tim(release, include_split=include_split or bool(names and any(SPLIT_RE.search(n) for n in names)))
    if names is not None:
        missing = [n for n in names if n not in pairs]
        if missing:
            raise KeyError(f"unknown pulsars: {missing}")
        pairs = {n: pairs[n] for n in names}

    todo = []
    for name, (par, tim) in pairs.items():
        p = cache_path(name, clock, cache_dir)
        if p.exists():
            try:
                with np.load(p, allow_pickle=False) as z:
                    h = json.loads(str(z["_header"]))["meta"]["input_hash"]
                if h == input_hash(par, tim, clock, release):
                    continue
            except Exception:  # noqa: BLE001, S110 - corrupt cache file -> re-ingest
                pass
        todo.append((name, par, tim, clock, release, str(cache_dir)))

    if todo:
        if verbose:
            print(f"ingesting {len(todo)} pulsar(s) with PINT (clock={clock}) ...", flush=True)
        # warm-up in-process so ephemeris/clock/IERS downloads happen once
        first = min(todo, key=lambda t: Path(t[2]).stat().st_size)
        _ingest_one(first)
        rest = [t for t in todo if t is not first]
        if rest:
            jobs = jobs or min(16, os.cpu_count() or 1)
            # spawn, not fork: the parent may already hold JAX/CUDA state
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
                for name, _ in ex.map(_ingest_one, rest):
                    if verbose:
                        print(f"  cached {name}", flush=True)

    psrs = [Pulsar.load(cache_path(n, clock, cache_dir)) for n in sorted(pairs)]
    if names is None and select_gwb:
        psrs = select_gwb_pulsars(psrs)
    return psrs


def select_gwb_pulsars(psrs: list[Pulsar], min_span_yr: float = MIN_SPAN_YR) -> list[Pulsar]:
    """Keep combined (non-split) pulsars with span >= 3 yr; check against the paper (67)."""
    keep = [p for p in psrs if not SPLIT_RE.search(p.name) and p.span_s / JULIAN_YEAR_S >= min_span_yr]
    dropped = sorted(p.name for p in psrs if p not in keep and not SPLIT_RE.search(p.name))
    if tuple(dropped) != EXPECTED_EXCLUDED or len(keep) != 67:
        raise RuntimeError(f"GWB selection gave {len(keep)} pulsars, dropped {dropped}; expected 67 / {EXPECTED_EXCLUDED}")
    return keep


def get_tspan(psrs: list[Pulsar]) -> float:
    """Whole-array span max(t) - min(t) over all selected pulsars [s] (enterprise get_tspan)."""
    return float(max(p.toas.max() for p in psrs) - min(p.toas.min() for p in psrs))
