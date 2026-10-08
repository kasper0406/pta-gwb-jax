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
    SITE_PROFILE_OF_PTA,
    ClockProfile,
    EvaluatorProfile,
    apply_site_profile,
    check_used_clock_files,
    evaluator_for_par,
    published_profile,
)
from .timfile import TimRecord, canonical_flags, read_tim, write_flat_tim

WORK_DIR = DATA_DIR / "processed" / "m3a" / "legs"
INGEST_VERSION = 2
MASK_KEYS = ("JUMP", "FDJUMP", "DMJUMP", "EFAC", "EQUAD", "ECORR", "T2EFAC", "T2EQUAD", "TNEF", "TNEQ", "TNECORR",
             "DMEFAC", "DMEQUAD")


# par keywords that are bookkeeping, not model parameters (never given fit flags)
PAR_METADATA = {"CHI2R", "CHI2", "NTOA", "TRES", "START", "FINISH", "NITS", "EPHVER", "DMDATA", "MODE", "INFO", "IBOOT",
                "TZRMJD", "TZRFRQ", "TZRSITE", "CLK", "CLOCK", "UNITS", "EPHEM", "TIMEEPH", "T2CMETHOD", "DILATEFREQ",
                "PLANET_SHAPIRO", "CORRECT_TROPOSPHERE", "NE_SW_IFUNC", "DM_SERIES", "BINARY", "PSR", "PSRJ", "PSRB"}


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
    tempo2_par = "EPHVER" in keys
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
        if (tempo2_par and len(tok) == 3 and tok[2] not in ("0", "1", "2") and _is_float(tok[2])
                and k not in PAR_METADATA and not k.startswith(("JUMP", "T2EFAC", "T2EQUAD", "TNEF", "TNEQ", "ECORR",
                                                                 "EFAC", "EQUAD"))):
            # "NAME value X" with X not 0/1/2: tempo2 (readParfile.C readValue, nread == 2) takes X
            # as the uncertainty and leaves the parameter FROZEN; PINT keeps the parameter's
            # default fit state instead (InPTA DR2 "DMX_0001 1.85e-4 2.57e-4": DMX_0001 would be
            # free). Write the tempo2 meaning explicitly: fit flag 0, uncertainty X.
            out.append(f"{k} {tok[1]} 0 {tok[2]}")
            fixes["par:explicit-frozen-value-uncertainty-line"] += 1
            continue
        out.append(ln)
    if dmx_ids and "0001" not in dmx_ids and "DMXR1_0001" not in keys:
        start = min(float(ln.split()[1]) for ln in lines if re.match(r"DMXR1_\d+\s", ln))
        out += ["DMX_0001 0 0", f"DMXR1_0001 {start - 10:.4f}", f"DMXR2_0001 {start - 9.99:.4f}"]
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
    (r"PINT_CLOCK_OVERRIDE|clock correction file .* from specified location|Clock file from .*/m3a/clocks/.* overrides global clock file", "clock-override", "pinned clock directory in use (profile pinning)"),
    (r"very high covariance|covariance", "fit-covariance", "fit diagnostics (no fit is performed here)"),
    (r"DeprecationWarning|FutureWarning|PendingDeprecationWarning", "library-deprecation", "library API deprecation, no effect on values"),
    (r"ResourceWarning", "resource", "unclosed file handle in a library"),
    (r"ERFA function", "erfa-dubious-year", "ERFA 'dubious year' for TOAs past the leap-second table horizon"),
]


# leg-specific resolutions of warnings whose message alone is not specific (located by hand)
LEG_WARNING_RESOLUTIONS = {
    ("mpta", "J0955-6150", "divide by zero encountered"): (
        "ddgr-default-init",
        "PINT DDGR model set-up evaluates a1/ar with default (zero) parameters before the par values are "
        "assigned (DDGR_model._updatePK; located with numpy seterr(divide='raise')); no effect on loaded values"),
}


def classify_warning(msg: str) -> tuple[str, str]:
    for rx, cls, expl in WARNING_CLASSES:
        if re.search(rx, msg):
            return cls, expl
    return "UNCLASSIFIED!", ""


def _normalise(msg: str) -> str:
    return re.sub(r"\d+(\.\d+)?", "#", msg)[:200]


# ---------------------------------------------------------------------- clock coverage (G2)

GAP_DAYS = 100.0  # interpolation across a gap wider than this is treated as uncovered


def clock_coverage(toas, tol_s: float = 1e-9) -> tuple[list[dict], np.ndarray]:
    """For every observatory of the leg and every clock file in its chain (as PINT loaded them):
    TOAs before the first / after the last entry, TOAs inside an interpolation gap wider than
    GAP_DAYS, and the size of the correction change across those regions (an error bound for
    PINT's clamping / interpolation). The file's own text is re-read to catch entries PINT's
    reader drops (e.g. an explicit "0 0" line at MJD 0)."""
    import pint.observatory as po

    out = []
    mjd = np.asarray(toas.get_mjds().value, dtype=np.float64)
    obs = np.asarray(toas.get_obss())
    bad = np.zeros(len(mjd), dtype=bool)  # TOAs without a clock correction good to tol_s
    for name in sorted(set(obs.tolist())):
        o = po.get_observatory(name)
        sel = np.flatnonzero(obs == name)
        m = mjd[sel]
        for c in getattr(o, "_clock", None) or []:
            inner = getattr(c, "clock_file", c)
            t = np.asarray(inner.time.mjd, dtype=np.float64)
            v = np.asarray(inner.clock.to_value("s"), dtype=np.float64)
            if len(t) == 0:
                continue
            fname = Path(getattr(c, "filename", "") or getattr(inner, "filename", "")).name
            raw = []  # (mjd, value) as tempo2 reads a tempo2-format file, sentinels included
            try:
                # tempo2-format text (pinned "*.clk" files, or global-repository files read from the
                # astropy cache under an opaque name: their GlobalClockFile.format says "tempo2")
                if str(inner.filename).endswith(".clk") or getattr(c, "format", None) == "tempo2":
                    for ln in Path(inner.filename).read_text().splitlines():
                        tok = ln.split()
                        if len(tok) >= 2 and not ln.lstrip().startswith("#") and _is_float(tok[0]) and _is_float(tok[1]):
                            raw.append((float(tok[0]), float(tok[1])))
            except Exception:  # noqa: BLE001
                raw = []
            rt = np.array([a for a, _ in raw]) if raw else t
            rv = np.array([b for _, b in raw]) if raw else v
            first, last = float(min(rt.min(), t[0])), float(max(rt.max(), t[-1]))
            # regions where PINT's parsed table and the file's own entries (tempo2) disagree: PINT
            # clamps beyond its first/last parsed entry, tempo2 interpolates to the dropped entries
            region_dv = []
            if rt.max() > t[-1]:
                region_dv.append((t[-1], rt.max(), float(np.max(np.abs(rv[rt > t[-1]] - v[-1])))))
            if rt.min() < t[0]:
                region_dv.append((rt.min(), t[0], float(np.max(np.abs(rv[rt < t[0]] - v[0])))))
            n_dropped_region = 0
            dv_dropped = 0.0
            for lo, hi, dv in region_dv:
                inr = (m > lo) & (m < hi)
                k = int(np.sum(inr))
                if k and dv > tol_s:
                    n_dropped_region += k
                    dv_dropped = max(dv_dropped, dv)
                    bad[sel[inr]] = True
            before, after = m < first, m > last
            gi = np.searchsorted(t, m)
            inside = (gi > 0) & (gi < len(t))
            wide = np.zeros_like(m, dtype=bool)
            dv_gap = 0.0
            if inside.any():
                ii = np.flatnonzero(inside)
                lo, hi = gi[ii] - 1, gi[ii]
                dv = np.abs(v[hi] - v[lo])
                w = ((t[hi] - t[lo]) > GAP_DAYS) & (dv > 1e-9)
                wide[ii[w]] = True
                bad[sel[wide]] = True
                if w.any():
                    dv_gap = float(np.max(dv[w]))
            slope = float(abs(v[-1] - v[-2]) / max(t[-1] - t[-2], 1e-9)) if len(t) > 1 else 0.0
            extrap_bound = float(slope * max(0.0, float(m.max() - last))) if after.any() else 0.0
            if after.any():
                bad[sel[after & (slope * (m - last) >= tol_s)]] = True
            bad[sel[before]] = True
            out.append({"observatory": name, "file": fname, "n_toa": len(m), "file_mjd": [first, last],
                        "pint_mjd": [float(t[0]), float(t[-1])], "n_before": int(before.sum()),
                        "n_after": int(after.sum()), "days_after": float(max(0.0, m.max() - last)),
                        "n_in_wide_gap": int(wide.sum()), "max_dclock_across_gap_s": dv_gap,
                        "extrapolation_bound_s": extrap_bound, "n_pint_vs_file_region": n_dropped_region,
                        "dclock_pint_vs_file_s": dv_dropped})
    return out, bad


def coverage_verdict(rows: list[dict], tol_s: float = 1e-9) -> tuple[bool, list[str]]:
    """Clock coverage is explained if no TOA lies in a wide gap, TOAs before the first entry are
    covered by an explicit zero entry (or there are none), and the extrapolation bound after the
    last entry is below ``tol_s``."""
    notes, ok = [], True
    for r in rows:
        if r["n_in_wide_gap"]:
            ok = False
            notes.append(f"{r['file']}: {r['n_in_wide_gap']} TOAs interpolated across a >{GAP_DAYS:g}-d gap "
                         f"(correction changes by {r['max_dclock_across_gap_s']:.3g} s)")
        if r.get("n_pint_vs_file_region"):
            ok = False
            notes.append(f"{r['file']}: {r['n_pint_vs_file_region']} TOAs where PINT clamps but the file's own "
                         f"entries differ by up to {r['dclock_pint_vs_file_s']:.3g} s")
        if r["n_before"]:
            ok = False
            notes.append(f"{r['file']}: {r['n_before']} TOAs before the first entry")
        if r["n_after"] and r["extrapolation_bound_s"] >= tol_s:
            ok = False
            notes.append(f"{r['file']}: {r['n_after']} TOAs up to {r['days_after']:.2f} d after the last entry "
                         f"(bound {r['extrapolation_bound_s']:.2g} s)")
    return ok, notes


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
        exp = {k.lstrip("-").lower(): v for k, v in canonical_flags(r) if k not in ("-to",)}
        got = {k.lower(): str(v) for k, v in flags[i].items() if k not in ignore}
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


def canonical_leg_texts(par: Path, tim: Path, fixes: Counter):
    """(tempo2-semantics records with indicator flags, canonical par text, multi-valued flags)."""
    recs, rep = read_tim(tim)
    ptxt = canonical_par(par.read_text(errors="replace"), fixes)
    multi = multivalued_mask_flags(recs, ptxt)
    recs, ptxt = apply_indicator_flags(recs, ptxt, multi, fixes)
    return recs, rep, ptxt, multi


def prepare_leg(dataset: str, label: str, par: Path, tim: Path, clock: ClockProfile, outdir: Path,
                par_text: str | None = None) -> dict:
    """Canonical par + flat tim on disk; returns provenance (no PINT involved). ``par_text``: a
    final par (e.g. the option-C rewrite of the canonical par) used instead of the canonical one;
    the tim side (records, indicator flags) is always derived from the leg's own files."""
    fixes: Counter = Counter()
    recs, rep, ptxt, multi = canonical_leg_texts(par, tim, fixes)
    if par_text is not None:
        ptxt = par_text
        fixes["par:replaced-by-final-par (option C rewrite)"] += 1
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
             tag: str = "published", clock_policy: str = "exclude-uncovered",
             site_profile: str | None = "default", nharms: int | None = None) -> LegResult:
    """Ingest one leg (see module docstring). Call in a fresh process per clock profile."""
    import warnings

    from .m3data import leg_files, pta_of

    pta = pta_of(dataset)
    par, tim = leg_files(dataset)[label]
    clock = clock or published_profile(pta, dataset)
    outdir = outdir or (WORK_DIR / tag / dataset / label)
    t0 = time.time()
    prov = prepare_leg(dataset, label, par, tim, clock, outdir, par_text)
    recs = prov.pop("_recs")
    Path(prov["canon_par"]).read_text()
    evaluator = evaluator or evaluator_for_par(par.read_text(errors="replace"))
    clock.activate()
    site_profile = SITE_PROFILE_OF_PTA[pta] if site_profile == "default" else site_profile
    site_changes = apply_site_profile(site_profile, [r.obs for r in recs])

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
            "pint_version": pint.__version__, "ingest_version": INGEST_VERSION, "tag": tag,
            "site_profile": site_profile, "site_changes": site_changes}
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            model, toas = get_model_and_toas(prov["canon_par"], prov["canon_tim"], planets=True, **evaluator.pint_kwargs())
            if nharms is None and evaluator.tempo2_nharms and "NHARMS" in model.params \
                    and getattr(model, "H4", None) is not None and model.H4.quantity is not None:
                # tempo2 convention for ELL1H H3+H4: NHARMS from the par, else 4 (PINT forces >= 7)
                given = [ln.split()[1] for ln in Path(prov["canon_par"]).read_text().splitlines()
                         if ln.split()[:1] in (["NHARMS"], ["NHARM"])]
                nharms = int(float(given[0])) if given else 4
            if nharms is not None:
                # ELL1H H3+H4 harmonic count. PINT's setup forces NHARMS >= 7 when H4 is given;
                # tempo2 uses 4 (harmonics 3..NHARMS). Setting the value after setup makes the
                # binary delay and its derivatives use the requested count (as nanograv/PINT#2046).
                if "NHARMS" not in model.params or model.H4.quantity is None:
                    raise ValueError(f"{label}: nharms override needs an ELL1H model with H3+H4")
                meta["nharms_setup"] = int(model.NHARMS.value)
                model.NHARMS.value = int(nharms)
                meta["nharms_used"] = int(nharms)
            frozen = []
            if pta == "NG15" and model.PSR.value in FROZEN_PARAMS:
                for pname in FROZEN_PARAMS[model.PSR.value]:
                    getattr(model, pname).frozen = True
                    frozen.append(pname)
            if identity:
                meta.update(toa_identity(toas, recs))
            # clock coverage audit (G2); TOAs without a clock correction good to 1 ns are removed
            # explicitly (policy "exclude-uncovered", recorded) or kept (policy "keep")
            meta["clock_files"] = check_used_clock_files(clock, pin=pin)
            meta["clock_coverage"], bad = clock_coverage(toas)
            meta["clock_coverage_ok"], meta["clock_coverage_notes"] = coverage_verdict(meta["clock_coverage"])
            meta["clock_policy"] = clock_policy
            meta["clock_excluded"] = []
            if bad.any() and clock_policy == "exclude-uncovered":
                nm = [str(fl.get("name", "")) for fl in toas.get_flags()]
                mj = toas.get_mjds().value
                meta["clock_excluded"] = [{"toa": nm[i], "mjd": float(mj[i]), "obs": str(toas.get_obss()[i])}
                                          for i in np.flatnonzero(bad)]
                toas = toas[~bad]
                meta["clock_coverage_ok"] = True
                meta["clock_coverage_notes"] = [f"{int(bad.sum())} uncovered TOAs excluded: "] + meta["clock_coverage_notes"]
            elif clock_policy not in ("keep", "exclude-uncovered"):
                raise ValueError(f"unknown clock policy {clock_policy!r}")
            # free mask parameters that select no TOA: tempo2 refuses to fit them -> frozen
            for pname in list(model.free_params):
                prm = getattr(model, pname)
                if hasattr(prm, "select_toa_mask") and len(prm.select_toa_mask(toas)) == 0:
                    prm.frozen = True
                    frozen.append(pname)
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
    for w in meta["warnings"]:
        for (ds, lab, rx), (cls, expl) in LEG_WARNING_RESOLUTIONS.items():
            if ds == dataset and lab == label and re.search(rx, w["message"]):
                w["class"], w["explanation"] = cls, expl
    for w in meta["warnings"]:  # clock-coverage warnings are explained iff the coverage audit passes
        if w["class"] == "clock-coverage!" and meta.get("clock_coverage_ok"):
            w["class"], w["explanation"] = "clock-coverage-audited", "coverage audit passed (no TOA without a correction good to 1 ns, after the explicit exclusions listed in clock_excluded)"
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
    c = Counter()
    cls_of = {}
    for m in msgs:
        key = _normalise(m)
        c[key] += 1
        cls_of.setdefault(key, classify_warning(m))
    return [{"message": m, "count": n, "class": cls_of[m][0], "explanation": cls_of[m][1]} for m, n in sorted(c.items())]


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
