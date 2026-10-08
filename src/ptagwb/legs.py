"""Physics-preserving ingestion of one PTA leg with PINT (M3a; gates G1, G2).

Pipeline for one leg (dataset, pulsar):

1. **tim**: ``timfile.read_tim`` (tempo2 semantics: per-file TIME/SKIP/END, INCLUDE, comment and
   flag rules), then ``timfile.write_flat_tim`` -> one flat file PINT reads with the same meaning.
   *Multi-valued mask flags*: when a TOA carries one flag several times with different values and
   the par selects on that flag (PPTA ``-j MEDUSA_59200 -j MEDUSA_58925`` with two ``JUMP -j``
   lines), tempo2 applies every matching JUMP; PINT keeps one value per flag. Such flags become
   indicator flags ``-<flag>__<value> 1`` (one per value on the TOA) and the par's mask lines are
   rewritten to select on the indicator (``JUMP -j__MEDUSA_59200 1 ...``): exact for any number of
   values.
2. **par**: the survey's canonicalisation rules (``canonical_par``; each rewrite is listed in the
   leg's provenance), then the clock/ephemeris profile (``profiles.ClockProfile.apply_to_par``).
3. **PINT load** with the leg's evaluator profile and the pinned clock directory, all warnings
   captured (loguru + Python warnings, every category) and classified (``classify_warning``).
   Free mask parameters (JUMPs ...) that select no TOA are frozen (tempo2 refuses to fit them).
4. **G1 TOA identity** (``toa_identity``): PINT's TOAs against the tempo2-semantics records, TOA by
   TOA (the flat file preserves order and names): count, arrival time incl. offsets (< 2 ns),
   uncertainty, frequency, observatory, every flag, -padd phase, pulse numbers.
5. Export in enterprise ``PintPulsar`` conventions (like ``data.load_pulsar_pint``), sorted by
   barycentric TOA, with the original record index of every TOA.

Run each leg in a fresh process (``load_legs`` uses a spawn pool): PINT caches observatory
clock objects per process, so a process must see one clock profile only.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR
from .data import FROZEN_PARAMS, Pulsar, enterprise_position, resolve_backend_flags
from .noise import namespace
from .profiles import (
    ClockProfile,
    EvaluatorProfile,
    check_used_clock_files,
    evaluator_for_par,
    published_profile,
)
from .timfile import TimRecord, canonical_flags, read_tim, write_flat_tim

WORK_DIR = DATA_DIR / "processed" / "m3a" / "legs"
INGEST_VERSION = 1
MASK_KEYS = ("JUMP", "FDJUMP", "DMJUMP", "EFAC", "EQUAD", "ECORR", "T2EFAC", "T2EQUAD", "TNEF", "TNEQ", "TNECORR",
             "DMEFAC", "DMEQUAD")


def _is_float(x: str) -> bool:
    try:
        float(x.replace("D", "E"))
        return True
    except ValueError:
        return False


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------- par canonicalisation


def canonical_par(text: str, fixes: Counter) -> str:
    """Par rewrites that let PINT read tempo2 par files with tempo2's meaning (survey Sec. 5):

    * a tempo2 par (EPHVER present) without UNITS gets ``UNITS TCB`` (tempo2's default);
    * DMXR1_/DMXR2_ carrying a fit flag -> frozen (range bounds are not parameters);
    * DMXR ranges without a DMX_ value -> ``DMX_xxxx 0`` frozen (tempo2 ignores them);
    * PB together with FB1.. and no FB0 -> FB0 = 1/(PB*86400) (tempo2 allows it, PINT needs FB0);
      the leg is in the PB+FB evaluator class (validated by G3/G4, quarantined otherwise);
    * a negative M2/H3 is NOT rewritten (signed-H3 class; PINT rejects it: quarantined);
    * if DMX ranges exist but DMX_0001 does not, an empty frozen DMX_0001 range is added (PINT's
      DMX component always holds a DMX_0001 template);
    * TRACK -2 is removed (no pulse numbers in the tim files; PINT then tracks the nearest pulse;
      phase connection is checked by G3 against tempo2).
    """
    lines = text.splitlines()
    keys = {ln.split()[0] for ln in lines if ln.split()}
    dmx_ids = {ln.split()[0][4:] for ln in lines if re.match(r"DMX_\d+\s", ln)}
    has_fb = any(re.match(r"FB[1-9]\d*\s", ln) for ln in lines)
    out = []
    if "UNITS" not in keys and "EPHVER" in keys:
        out.append("UNITS TCB")
        fixes["par:explicit-UNITS-TCB"] += 1
    for ln in lines:
        tok = ln.split()
        if not tok:
            out.append(ln)
            continue
        k = tok[0]
        if re.fullmatch(r"DMXR[12]_\d+", k) and len(tok) >= 3 and tok[2] == "1":
            out.append(f"{k} {tok[1]}")
            fixes["par:freeze-DMXR"] += 1
            continue
        if re.fullmatch(r"DMXR1_\d+", k) and k[6:] not in dmx_ids:
            out.append(ln)
            out.append(f"DMX_{k[6:]} 0")
            fixes["par:missing-DMX-value"] += 1
            continue
        if k in ("M2", "H3") and _is_float(tok[1]) and float(tok[1].replace("D", "E")) < 0:
            fixes[f"par:signed-{k}(kept)"] += 1
        if k == "PB" and has_fb and "FB0" not in keys:
            fb0 = 1.0 / (float(tok[1].replace("D", "E")) * 86400.0)
            out.append(f"FB0 {fb0:.25e} " + " ".join(tok[2:3]))
            fixes["par:PB->FB0"] += 1
            continue
        if k == "TRACK":
            fixes["par:drop-TRACK"] += 1
            continue
        out.append(ln)
    if dmx_ids and "0001" not in dmx_ids and "DMXR1_0001" not in keys:
        start = min(float(ln.split()[1]) for ln in lines if re.match(r"DMXR1_\d+\s", ln))
        out += ["DMX_0001 0", f"DMXR1_0001 {start - 10:.4f}", f"DMXR2_0001 {start - 9.99:.4f}"]
        fixes["par:empty-DMX_0001-template"] += 1
    return "\n".join(out) + "\n"


def multivalued_mask_flags(recs: list[TimRecord], par_text: str) -> set[str]:
    """Flags that some TOA carries with several different values AND that a par mask line uses."""
    multi = set()
    for r in recs:
        seen: dict[str, str] = {}
        for k, v in r.flags:
            if k in seen and seen[k] != v:
                multi.add(k)
            seen.setdefault(k, v)
    used = set()
    for ln in par_text.splitlines():
        tok = ln.split()
        if len(tok) >= 3 and (tok[0] in MASK_KEYS or re.fullmatch(r"FDJUMP\d+", tok[0])) and tok[1].startswith("-"):
            used.add(tok[1])
    return multi & used


def indicator_name(flag: str, value: str) -> str:
    return f"{flag}__{value}"


def apply_indicator_flags(recs: list[TimRecord], par_text: str, flags: set[str], fixes: Counter):
    """Rewrite multi-valued mask flags into indicator flags in the records and the par."""
    if not flags:
        return recs, par_text
    new = []
    for r in recs:
        fl = []
        for k, v in r.flags:
            if k in flags:
                fl.append((indicator_name(k, v), "1"))
            else:
                fl.append((k, v))
        new.append(TimRecord(r.file, r.lineno, r.name, r.freq, r.sat, r.err, r.obs, fl, r.time_offset))
    out = []
    for ln in par_text.splitlines():
        tok = ln.split()
        if len(tok) >= 3 and tok[1] in flags and (tok[0] in MASK_KEYS or re.fullmatch(r"FDJUMP\d+", tok[0])):
            out.append(" ".join([tok[0], indicator_name(tok[1], tok[2]), "1"] + tok[3:]))
            fixes[f"par:mask-{tok[0]}-on-multivalued-{tok[1]}->indicator"] += 1
        else:
            out.append(ln)
    fixes["tim:multivalued-mask-flags->indicators"] += sum(1 for r in recs if any(k in flags for k, _ in r.flags))
    return new, "\n".join(out) + "\n"


# ---------------------------------------------------------------------- warnings (G2)

# (regex, class, explanation). Classes ending in "!" are not acceptable without a leg-specific
# resolution (they make G2 fail unless the leg is quarantined for that reason).
WARNING_CLASSES = [
    (r"PINT does not support 'DILATEFREQ", "unsupported-tempo2-setting", "DILATEFREQ Y (tempo2 frequency dilation); effect quantified by G3 vs tempo2"),
    (r"PINT only supports 'TIMEEPH", "unsupported-tempo2-setting", "TIMEEPH IF99 (tempo2 time ephemeris); effect quantified by G3"),
    (r"T2CMETHOD", "unsupported-tempo2-setting", "T2CMETHOD (tempo2 celestial-to-terrestrial method); effect quantified by G3"),
    (r"does not support 'UNITS TCB' internally|Converting this timing model from TCB to TDB", "tcb-to-tdb", "TCB par converted to TDB by PINT (YA convert too); validated by G3"),
    (r"Unrecognized parfile line '(EPHVER|DM_SERIES|FDJUMP_SCALE|MODE|NITS|NTOA|TRES|CHI2R|START|FINISH|INFO|NE_SW_IFUNC|CORRECT_TROPOSPHERE|PLANET_SHAPIRO)", "ignored-par-metadata", "tempo2 bookkeeping/metadata line without effect on the delays"),
    (r"Unrecognized parfile line", "unrecognized-par-line!", "par line PINT ignores; must be shown to be inert"),
    (r"Data points out of range in clock file", "clock-coverage!", "clock file does not cover all TOAs; must be explained per leg"),
    (r"Found T2 binary model\. Gracefully converting", "t2-binary-resolution", "T2 binary resolved to a PINT model (evaluator class; validated by G3/G4)"),
    (r"DDK model uses KIN as inclination angle", "ddk-kin", "DDK uses KIN (SINI unused), as tempo2"),
    (r"A1DOT with a DDK model", "ddk-a1dot", "A1DOT in a DDK model (as released)"),
    (r"overflow encountered in conversion from string", "pint-parse-overflow", "PINT parses an over-long numeric literal; values checked by G3"),
    (r"(Start|End) of DMX_\d+ .* overlaps with DMX_\d+", "dmx-overlap", "DMX ranges overlap as released; no TOA in the overlap is double counted (PINT assigns per range)"),
    (r"Invalid altitude calculated", "altitude", "PINT altitude for a few TOAs (troposphere off for non-PINT pars)"),
    (r"Some TOAs are missing pulse numbers", "pulse-numbers-partial", "pulse numbers on some TOAs only; tracking mode nearest (TRACK dropped)"),
    (r"maskParameter.*has no TOAs|has no TOAs", "mask-no-toas", "mask parameter without TOAs (tempo2 refuses to fit it; frozen)"),
    (r"TZRMJD is not set", "tzr-default", "absolute phase reference defaulted; residuals are mean-subtracted and an offset column is fitted"),
    (r"divide by zero encountered", "numeric-divide-by-zero!", "numerical warning; must be located"),
    (r"MODE command is not supported", "tim-mode", "MODE 1 (weighted fit) is the only mode used"),
    (r"PINT_CLOCK_OVERRIDE|clock correction file .* from specified location", "clock-override", "pinned clock directory in use"),
    (r"very high covariance|covariance", "fit-covariance", "fit diagnostics (no fit is performed here)"),
    (r"DeprecationWarning|FutureWarning|PendingDeprecationWarning", "library-deprecation", "library API deprecation, no effect on values"),
    (r"ResourceWarning", "resource", "unclosed file handle in a library"),
    (r"ERFA function", "erfa-dubious-year", "ERFA 'dubious year' for TOAs past the leap-second table horizon"),
]


def classify_warning(msg: str) -> tuple[str, str]:
    for rx, cls, expl in WARNING_CLASSES:
        if re.search(rx, msg):
            return cls, expl
    return "UNCLASSIFIED!", ""


def _normalise(msg: str) -> str:
    return re.sub(r"\d+(\.\d+)?", "#", msg)[:200]


# ---------------------------------------------------------------------- G1


def toa_identity(toas, recs: list[TimRecord]) -> dict:
    """PINT TOAs vs tempo2-semantics records, TOA by TOA (flat-file order; names toaNNNNNNN)."""
    from pint.observatory import get_observatory
    from pint.pulsar_mjd import time_to_longdouble

    n = toas.ntoas
    out = {"g1_n_pint": int(n), "g1_n_records": len(recs)}
    if n != len(recs):
        out["g1_ok"] = False
        return out
    names = [fl.get("name", "") for fl in toas.table["flags"]] if "name" not in toas.table.colnames else list(toas.table["name"])
    idx = np.array([int(str(x)[3:]) - 1 for x in names])
    col = toas.table["mjd"]
    t = np.array([time_to_longdouble(x) for x in col], dtype=np.longdouble) if col.dtype == object else \
        np.asarray(time_to_longdouble(col), dtype=np.longdouble)
    flags = list(toas.table["flags"])
    clk = np.array([float(fl.get("clkcorr", 0.0)) - float(fl.get("to", 0.0)) for fl in flags], dtype=np.longdouble)
    t = t - clk / 86400
    freq = toas.get_freqs().value
    err = toas.get_errors().value
    obs = list(toas.table["obs"])
    dpn = np.asarray(toas.table["delta_pulse_number"]) if "delta_pulse_number" in toas.table.colnames else np.zeros(n)
    dt, dfreq, derr = [], [], []
    n_obs_bad = n_flag_bad = n_padd_bad = 0
    obs_cache: dict[str, str] = {}
    flag_bad_examples = []
    ignore = {"format", "clkcorr", "to", "name", "padd", "pn"}
    for i in range(n):
        r = recs[idx[i]]
        day, sec = r.mjd_parts()
        dt.append(float((t[i] - day) * 86400 - sec))
        dfreq.append(abs(freq[i] - r.freq))
        derr.append(abs(err[i] - r.err))
        if r.obs not in obs_cache:
            obs_cache[r.obs] = get_observatory(r.obs.upper()).name
        n_obs_bad += obs_cache[r.obs] != obs[i]
        exp = {k.lstrip("-"): v for k, v in canonical_flags(r) if k not in ("-to",)}
        got = {k: str(v) for k, v in flags[i].items() if k not in ignore}
        padd = float(exp.pop("padd", 0.0) or 0.0)
        exp.pop("pn", None)
        if got != exp:
            n_flag_bad += 1
            if len(flag_bad_examples) < 3:
                flag_bad_examples.append({"toa": r.name, "pint": got, "expected": exp})
        n_padd_bad += abs(float(dpn[i]) - padd) > 1e-12
    dt = np.abs(np.asarray(dt))
    out.update(
        g1_max_dt_ns=float(dt.max() * 1e9) if n else 0.0,
        g1_max_dfreq_mhz=float(max(dfreq, default=0.0)),
        g1_max_derr_us=float(max(derr, default=0.0)),
        g1_n_obs_mismatch=int(n_obs_bad),
        g1_n_flag_mismatch=int(n_flag_bad),
        g1_n_padd_mismatch=int(n_padd_bad),
        g1_flag_examples=flag_bad_examples,
        g1_order_is_identity=bool(np.array_equal(idx, np.arange(n))),
    )
    out["g1_ok"] = bool(out["g1_max_dt_ns"] < 2.0 and out["g1_max_dfreq_mhz"] < 1e-6 and out["g1_max_derr_us"] < 1e-9
                        and n_obs_bad == 0 and n_flag_bad == 0 and n_padd_bad == 0)
    return out


# ---------------------------------------------------------------------- leg loading


@dataclass
class LegResult:
    dataset: str
    pta: str
    label: str
    ok: bool
    psr: Pulsar | None = None
    meta: dict = field(default_factory=dict)
    error: str = ""


def prepare_leg(dataset: str, label: str, par: Path, tim: Path, clock: ClockProfile, outdir: Path,
                par_text: str | None = None) -> dict:
    """Canonical par + flat tim on disk; returns provenance (no PINT involved)."""
    fixes: Counter = Counter()
    recs, rep = read_tim(tim)
    src_par = par.read_text(errors="replace") if par_text is None else par_text
    ptxt = canonical_par(src_par, fixes)
    multi = multivalued_mask_flags(recs, ptxt)
    recs, ptxt = apply_indicator_flags(recs, ptxt, multi, fixes)
    ptxt, clock_changes = clock.apply_to_par(ptxt)
    outdir.mkdir(parents=True, exist_ok=True)
    cpar = outdir / f"{label}.par"
    cpar.write_text(ptxt)
    ctim = write_flat_tim(recs, outdir / f"{label}.flat.tim")
    return {
        "dataset": dataset, "label": label, "par": str(par), "tim": str(tim),
        "par_sha256": _sha256_bytes(par.read_bytes()), "canon_par": str(cpar), "canon_tim": str(ctim),
        "canon_par_sha256": _sha256_bytes(cpar.read_bytes()), "canon_tim_sha256": _sha256_bytes(ctim.read_bytes()),
        "tim_files": rep.files, "tim_counts": dict(rep.counts), "tim_duplicate_flags": dict(rep.duplicate_flags),
        "n_records": len(recs), "par_fixes": dict(fixes), "clock_changes": clock_changes,
        "multivalued_mask_flags": sorted(multi), "_recs": recs,
    }


def load_leg(dataset: str, label: str, *, clock: ClockProfile | None = None, evaluator: EvaluatorProfile | None = None,
             par_text: str | None = None, pin: bool = False, outdir: Path | None = None, identity: bool = True,
             tag: str = "published") -> LegResult:
    """Ingest one leg (see module docstring). Call in a fresh process per clock profile."""
    import warnings

    from .m3data import leg_files, pta_of

    pta = pta_of(dataset)
    par, tim = leg_files(dataset)[label]
    clock = clock or published_profile(pta)
    outdir = outdir or (WORK_DIR / tag / dataset / label)
    t0 = time.time()
    prov = prepare_leg(dataset, label, par, tim, clock, outdir, par_text)
    recs = prov.pop("_recs")
    Path(prov["canon_par"]).read_text()
    evaluator = evaluator or evaluator_for_par(par.read_text(errors="replace") if par_text is None else par_text)
    clock.activate()

    import astropy.units as u
    import pint
    import pint.logging
    from loguru import logger
    from pint.models import get_model_and_toas
    from pint.residuals import Residuals

    msgs: list[str] = []
    pint.logging.setup(level="WARNING")
    logger.remove()
    logger.add(lambda m: msgs.append(f"{m.record['level'].name}: {m.record['message']}"), level="WARNING")
    meta = {**prov, "profile": {"clock_profile": clock.name, "clock": clock.clock, "ephem": clock.ephem,
                                "evaluator": evaluator.name, "ell1h_shapiro": evaluator.ell1h_shapiro},
            "pint_version": pint.__version__, "ingest_version": INGEST_VERSION, "tag": tag}
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            model, toas = get_model_and_toas(prov["canon_par"], prov["canon_tim"], planets=True, **evaluator.pint_kwargs())
            frozen = []
            if pta == "NG15" and model.PSR.value in FROZEN_PARAMS:
                for pname in FROZEN_PARAMS[model.PSR.value]:
                    getattr(model, pname).frozen = True
                    frozen.append(pname)
            # free mask parameters that select no TOA: tempo2 refuses to fit them -> frozen
            for pname in list(model.free_params):
                prm = getattr(model, pname)
                if hasattr(prm, "select_toa_mask") and len(prm.select_toa_mask(toas)) == 0:
                    prm.frozen = True
                    frozen.append(pname)
            if identity:
                meta.update(toa_identity(toas, recs))
            res = Residuals(toas, model)
            resid = np.asarray(res.time_resids.to_value(u.s), dtype=np.float64)
            M, fitpars, _units = model.designmatrix(toas)
            M = np.asarray(M, dtype=np.float64)
            btoas = np.asarray(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
            stoas = np.asarray(toas.get_mjds().value, dtype=np.float64) * 86400.0
            errs = np.asarray(toas.get_errors().to_value(u.s), dtype=np.float64)
            freqs = np.asarray(model.barycentric_radio_freq(toas).to_value(u.MHz), dtype=np.float64)
            freqs_topo = np.asarray(toas.get_freqs().to_value(u.MHz), dtype=np.float64)
            telescope = np.asarray(toas.get_obss()).astype("U")
            names = [str(fl.get("name", "")) for fl in toas.get_flags()]
            rec_index = np.array([int(x[3:]) - 1 for x in names], dtype=np.int64)
            allflags: dict[str, list[str]] = {}
            for i, fl in enumerate(toas.get_flags()):
                for k, v in fl.items():
                    allflags.setdefault(k, [""] * toas.ntoas)[i] = str(v)
            allflags_np = {k: np.array(v, dtype="U") for k, v in allflags.items()}
            backend = resolve_backend_flags(allflags_np, toas.ntoas)
            icrs = model.coords_as_ICRS(epoch=None)
            ra, dec = icrs.ra.to_value(u.rad), icrs.dec.to_value(u.rad)
            pos = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
            try:
                pos_ent = enterprise_position(model, model.PSR.value)
            except Exception:  # noqa: BLE001
                pos_ent = None
            meta["clock_files"] = check_used_clock_files(clock, pin=pin)
            meta.update(
                psr_name=str(model.PSR.value), units_in=str(model.UNITS.value), binary=str(model.BINARY.value or ""),
                clock_used=str(model.CLOCK.value), ephem_used=str(model.EPHEM.value), ntoa=int(toas.ntoas),
                ncol=int(M.shape[1]), frozen_after_load=frozen, track_mode=res.track_mode,
                components=sorted(model.components), wrms_us=float(np.sqrt(np.average((resid - np.average(resid, weights=errs**-2))**2, weights=errs**-2)) * 1e6),
            )
            meta["par_as_loaded"] = model.as_parfile()
        wmsgs = [f"{w.category.__name__}: {w.message}" for w in wl]
    except Exception as ex:  # noqa: BLE001
        meta.update(error=f"{type(ex).__name__}: {str(ex)[:500]}", traceback=traceback.format_exc(limit=6)[-2000:])
        meta["warnings"] = _classified(msgs)
        return LegResult(dataset, pta, label, False, None, meta, meta["error"])
    meta["warnings"] = _classified(msgs + wmsgs)
    meta["g2_unexplained"] = sorted({w["class"] for w in meta["warnings"] if w["class"].endswith("!")})
    meta["seconds"] = round(time.time() - t0, 1)
    isort = np.argsort(btoas, kind="mergesort")
    flags = {k: v[isort] for k, v in allflags_np.items()}
    psr = Pulsar(
        name=meta["psr_name"], toas=btoas[isort], stoas=stoas[isort], residuals=resid[isort], toaerrs=errs[isort],
        freqs=freqs[isort], freqs_topo=freqs_topo[isort], backend_flags=backend[isort], telescope=telescope[isort],
        Mmat=M[isort], fitpars=list(fitpars), pos=pos, pos_enterprise=pos_ent, flags=flags,
        meta={"dataset": dataset, "pta": pta, "label": label, "record_index": rec_index[isort].tolist()},
    )
    return LegResult(dataset, pta, label, True, psr, meta)


def _classified(msgs: list[str]) -> list[dict]:
    c = Counter(_normalise(m) for m in msgs)
    out = []
    for m, n in sorted(c.items()):
        cls, expl = classify_warning(m)
        out.append({"message": m, "count": n, "class": cls, "explanation": expl})
    return out


def leg_cache_path(tag: str, dataset: str, label: str) -> Path:
    return WORK_DIR / tag / dataset / label / "leg.npz"


def save_leg(res: LegResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if res.psr is not None:
        res.psr.meta = dict(res.psr.meta, leg_meta=json.dumps(res.meta, default=str))
        res.psr.save(path)
    (path.parent / "leg_meta.json").write_text(json.dumps(res.meta, indent=1, default=str) + "\n")


def _worker(args):
    dataset, label, kw = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    try:
        res = load_leg(dataset, label, **kw)
    except Exception as ex:  # noqa: BLE001
        res = LegResult(dataset, "", label, False, None, {"error": repr(ex), "traceback": traceback.format_exc()}, repr(ex))
    if kw.get("tag"):
        save_leg(res, leg_cache_path(kw["tag"], dataset, label))
    return res


def load_legs(items: list[tuple[str, str]], *, jobs: int = 8, **kw) -> list[LegResult]:
    """Ingest legs in fresh spawned processes (one clock-profile state per process)."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, max_tasks_per_child=1) as ex:
        return list(ex.map(_worker, [(d, lab, kw) for d, lab in items]))


def namespaced(psr: Pulsar, pta: str) -> Pulsar:
    """Copy of a leg with ``<pta>:<system>`` labels (L2)."""
    import copy

    p = copy.copy(psr)
    p.backend_flags = np.array([namespace(pta, s) for s in psr.backend_flags], dtype="U")
    return p
