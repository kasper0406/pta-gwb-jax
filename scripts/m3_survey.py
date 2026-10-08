"""M3 survey: inventory and PINT smoke load of the five public PTA data sets.

For every pulsar of every data set below this script
  * parses the .par file as text (timing-model and noise keywords, tempo2-only constructs,
    clock / ephemeris / units, JUMP / FD / DMX counts, binary model),
  * parses the .tim file as text (INCLUDEs resolved; TOA count, MJD span, radio-frequency
    range, backend/system flags, wideband markers),
  * loads par + tim with PINT (``get_model_and_toas(planets=True, allow_tcb=True,
    allow_T2=True)``, PINT's global clock repository) and records success, the PINT TOA
    count, the design-matrix shape, the pre-fit weighted RMS, the warnings PINT emitted and,
    on failure, the exception (categorised later in docs/M3_SURVEY.md),
  * matches pulsars across data sets by sky position (30 arcsec), giving J-names to B-name
    pulsars, and writes the PTA overlap matrix.

Usage:
    python scripts/m3_survey.py                     # all data sets, PINT load included
    python scripts/m3_survey.py --no-pint           # text inventory only (seconds)
    python scripts/m3_survey.py --only mpta inpta_dr2
    python scripts/m3_survey.py --jobs 16 --timeout 900
    python scripts/m3_survey.py --canonical         # + PINT load of canonicalised copies
                                                    #   (tempo2 pars with ell1h_shapiro="absorbed")

Outputs (git-ignored): data/processed/m3_survey/{<dataset>.csv, survey.json, overlap.md,
tables.md}. Nothing in src/ptagwb is used or changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "m3_survey"


def _one(pattern: str) -> Path:
    cands = sorted(RAW.glob(pattern))
    if len(cands) != 1:
        raise FileNotFoundError(f"{pattern}: expected one match, got {cands}")
    return cands[0]


# --------------------------------------------------------------------------------------
# Data-set definitions: name -> (PTA label, description, function returning
# {psr_label: (par, tim)})
# --------------------------------------------------------------------------------------


def ds_ng15():
    rel = _one("ng15_v2.1.0/extracted/NANOGrav*")
    out = {}
    for par in sorted((rel / "narrowband" / "par").glob("*.nb.par")):
        name = par.name.split("_PINT_")[0]
        if re.search(r"(ao|gbt)$", name):
            continue
        (tim,) = sorted((rel / "narrowband" / "tim").glob(f"{name}_PINT_*.nb.tim"))
        out[name] = (par, tim)
    return out


def _epta(variant):
    base = _one("epta_dr2_gitlab/extracted/epta-dr2-*/EPTA-DR2") / variant
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        out[d.name] = (d / f"{d.name}.par", d / f"{d.name}_all.tim")
    return out


def _ppta(base: Path):
    out = {}
    for par in sorted(base.glob("J*.par")):
        if "singlePsrNoise" in par.name:
            continue
        out[par.stem] = (par, base / f"{par.stem}.tim")
    return out


def ds_ppta():
    return _ppta(RAW / "ppta_dr3_timing" / "ppta_dr3" / "toas_and_parameters" / "all")


def ds_ppta_gh():
    return _ppta(_one("ppta_dr3_github/extracted/PPTA-DR3-*") / "analysis_codes" / "data" / "all")


def ds_inpta_dr2():
    base = _one("inpta_dr2/extracted/InPTA.DR2-*")
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("J")):
        out[d.name] = (d / f"{d.name}.DMX.par", d / f"{d.name}_all.tim")
    return out


def ds_inpta_dr1():
    base = _one("inpta_dr1/extracted/InPTA.DR1-*")
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("J")):
        out[d.name] = (d / f"{d.name}.NB.par", d / f"{d.name}.NB.tim")
    return out


def ds_mpta():
    base = RAW / "mpta_4p5yr" / "extracted" / "partim"
    return {p.stem: (p, base / f"{p.stem}.tim") for p in sorted(base.glob("J*.par"))}


DATASETS = {
    # name: (PTA, description, loader)
    "ng15": ("NG15", "NANOGrav 15-yr v2.1.0 narrowband (PINT)", ds_ng15),
    "epta_dr2new": ("EPTA", "EPTA DR2new (GitLab 2911d0e)", lambda: _epta("DR2new")),
    "epta_dr2full": ("EPTA", "EPTA DR2full (GitLab 2911d0e)", lambda: _epta("DR2full")),
    "epta_dr2new+": ("EPTA", "EPTA DR2new+ = DR2new + InPTA DR1 (GitLab 2911d0e)", lambda: _epta("DR2new+")),
    "ppta_dr3": ("PPTA", "PPTA DR3 toas_and_parameters/all (CSIRO DAP 59374v2)", ds_ppta),
    "ppta_dr3_gh": ("PPTA", "PPTA DR3 analysis_codes/data/all (GitHub fdbe6eb, used by Yu & Allen)", ds_ppta_gh),
    "inpta_dr2": ("InPTA", "InPTA DR2 (GitHub e2806fc), <psr>.DMX.par + _all.tim", ds_inpta_dr2),
    "inpta_dr1": ("InPTA", "InPTA DR1 narrowband (GitHub 2c400d5, used by Yu & Allen)", ds_inpta_dr1),
    "mpta": ("MPTA", "MeerKAT PTA 4.5-yr partim (Data Central)", ds_mpta),
}
# the five-PTA set of this project (overlap matrix) and Yu & Allen's set
MAIN_SET = ["ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr2", "mpta"]  # selected configuration
YU_ALLEN_SET = ["ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr1", "mpta"]


# --------------------------------------------------------------------------------------
# Text parsing
# --------------------------------------------------------------------------------------

NOISE_PREFIXES = ("TNEF", "TNEQ", "TNECORR", "TNRedAmp", "TNRedGam", "TNRedC", "TNDMAmp", "TNDMGam",
                  "TNDMC", "TNChromAmp", "TNChromGam", "TNChromC", "TNChromIdx", "TNBandNoise",
                  "TNGroupNoise", "TNSubtractRed", "TNSubtractDM", "T2EFAC", "T2EQUAD", "ECORR", "EFAC",
                  "EQUAD", "RNAMP", "RNIDX", "TNRED", "TNDM")
TEMPO2_ONLY = ("DMMODEL", "_DM", "_CM", "CONSTRAIN", "TNEF", "TNEQ", "TNECORR", "TNRed", "TNDM", "TNChrom",
               "TNBand", "TNGroup", "TNSubtract", "T2EFAC", "T2EQUAD", "SWM", "NE_SW_IFUNC", "IFUNC", "GLEP",
               "EXPEP", "EXPPH", "EXPINDEX", "EXPTAU", "GAUSEP", "TEMPO1", "NO_SS_SHAPIRO", "IPM", "DMOFF",
               "TIMEEPH", "T2CMETHOD", "DILATEFREQ", "SATJUMP", "WAVE")


def parse_par(path: Path) -> dict:
    keys = Counter()
    vals = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "C ")):
            continue
        tok = line.split()
        k = tok[0]
        keys[k] += 1
        if k not in vals and len(tok) > 1:
            vals[k] = tok[1]
    get = lambda *ks: next((vals[k] for k in ks if k in vals), "")
    noise = sorted({k for k in keys if k.startswith(NOISE_PREFIXES)})
    t2only = sorted({k for k in keys if k.startswith(TEMPO2_ONLY)})
    return {
        "psrj": get("PSRJ", "PSR", "PSRB"),
        "units": get("UNITS") or "TCB(default)",
        "clock": get("CLK", "CLOCK"),
        "ephem": get("EPHEM"),
        "ephver": get("EPHVER"),
        "binary": get("BINARY"),
        "ne_sw": get("NE_SW", "NE1AU", "SOLARN0"),
        "tres_us": get("TRES"),
        "ntoa_par": get("NTOA"),
        "track": get("TRACK"),
        "swm": get("SWM"),
        "dm_series": get("DM_SERIES"),
        "dmmodel": "DMMODEL" in keys,
        "n_dmx": sum(v for k, v in keys.items() if re.fullmatch(r"DMX_\d+", k)),
        "n_jump": keys.get("JUMP", 0),
        "n_fd": sum(v for k, v in keys.items() if re.fullmatch(r"FD\d+", k)),
        "n_fdjump": sum(v for k, v in keys.items() if k.startswith("FDJUMP")),
        "n_wave": sum(v for k, v in keys.items() if k.startswith("WAVE")),
        "n_glitch": sum(v for k, v in keys.items() if k.startswith("GLEP")),
        "has_cm": any(k.startswith(("CM", "TNChrom")) for k in keys),
        # binary parameterisation audit (evaluator conventions differ between PINT and tempo2)
        "bin_h3": get("H3"),
        "bin_h4": "H4" in keys,
        "bin_stig": any(k in keys for k in ("STIG", "STIGMA", "VARSIGMA")),
        "bin_nharms": get("NHARMS", "NHARM"),
        "bin_m2_sini": "M2" in keys and "SINI" in keys,
        "bin_kin_kom": "KIN" in keys or "KOM" in keys,
        "bin_pb_fb": "PB" in keys and any(re.fullmatch(r"FB[1-9]\d*", k) for k in keys) and "FB0" not in keys,
        "bin_n_fb": sum(1 for k in keys if re.fullmatch(r"FB\d+", k)),
        "noise_keys": ",".join(noise),
        "tempo2_constructs": ",".join(t2only),
        "pos": par_position(vals),
    }


def _sexa(s: str, hours: bool) -> float:
    sign = -1.0 if s.strip().startswith("-") else 1.0
    parts = [abs(float(x)) for x in s.replace("+", "").replace("-", "").split(":")]
    v = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0) + (parts[2] / 3600 if len(parts) > 2 else 0)
    return sign * v * (15.0 if hours else 1.0)


def par_position(vals: dict) -> tuple[float, float] | None:
    """(ra, dec) in degrees, ICRS-ish (ecliptic converted with the IERS2010 obliquity)."""
    try:
        if "RAJ" in vals and "DECJ" in vals:
            return _sexa(vals["RAJ"], True), _sexa(vals["DECJ"], False)
        lon = float(vals.get("ELONG", vals.get("LAMBDA", "nan")).replace("D", "E"))
        lat = float(vals.get("ELAT", vals.get("BETA", "nan")).replace("D", "E"))
        if math.isnan(lon):
            return None
        eps = math.radians(84381.406 / 3600.0)
        lo, la = math.radians(lon), math.radians(lat)
        x = math.cos(la) * math.cos(lo)
        y = math.cos(la) * math.sin(lo) * math.cos(eps) - math.sin(la) * math.sin(eps)
        z = math.cos(la) * math.sin(lo) * math.sin(eps) + math.sin(la) * math.cos(eps)
        return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))
    except Exception:  # noqa: BLE001
        return None


TIM_DIRECTIVES = {"FORMAT", "MODE", "INCLUDE", "TIME", "JUMP", "EFAC", "EQUAD", "PHASE", "SKIP", "NOSKIP",
                  "END", "INFO", "TRACK", "EMIN", "EMAX", "FMIN", "FMAX", "SIGMA", "PHA1", "PHA2"}


def parse_tim(path: Path, _seen=None) -> dict:
    """Text inventory of a tempo2 FORMAT 1 tim file (INCLUDEs resolved, SKIP honoured)."""
    _seen = _seen if _seen is not None else set()
    path = path.resolve()
    if path in _seen:
        return {}
    _seen.add(path)
    agg = {"ntoa": 0, "mjd": [], "freq": [], "flags": defaultdict(Counter), "obs": Counter(), "n_include": 0,
           "has_time": False, "has_jump_directive": False, "wideband": 0}
    skipping = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("C ", "#", "c ")) or line == "C":
            continue
        tok = line.split()
        key = tok[0].upper()
        if key == "SKIP":
            skipping = True
            continue
        if key == "NOSKIP":
            skipping = False
            continue
        if skipping:
            continue
        if key == "END":  # tempo2: stop reading this (possibly INCLUDEd) file
            break
        if key == "INCLUDE":
            sub = parse_tim(path.parent / tok[1], _seen)
            agg["n_include"] += 1 + sub.get("n_include", 0)
            for k in ("ntoa", "wideband"):
                agg[k] += sub.get(k, 0)
            agg["mjd"] += sub.get("mjd", [])
            agg["freq"] += sub.get("freq", [])
            agg["obs"].update(sub.get("obs", {}))
            for f, c in sub.get("flags", {}).items():
                agg["flags"][f].update(c)
            agg["has_time"] |= sub.get("has_time", False)
            agg["has_jump_directive"] |= sub.get("has_jump_directive", False)
            continue
        if key in TIM_DIRECTIVES:
            agg["has_time"] |= key == "TIME"
            agg["has_jump_directive"] |= key == "JUMP"
            continue
        if len(tok) < 5:
            continue
        try:
            freq, mjd = float(tok[1]), float(tok[2])
        except ValueError:
            continue
        agg["ntoa"] += 1
        agg["freq"].append(freq)
        agg["mjd"].append(mjd)
        agg["obs"][tok[4]] += 1
        fl = tok[5:]
        for i in range(0, len(fl) - 1, 2):
            if fl[i].startswith("-") and fl[i] in ("-f", "-sys", "-group", "-be", "-fe", "-pta", "-B", "-g", "-i"):
                agg["flags"][fl[i][1:]][fl[i + 1]] += 1
            if fl[i] == "-pp_dm":
                agg["wideband"] += 1
    return agg


def tim_records(path: Path, offset: float = 0.0, _seen=None) -> list[tuple]:
    """Per-TOA records of a tempo2 FORMAT 1 tim tree, as tempo2 reads it: INCLUDE recursion,
    SKIP/NOSKIP, END ending only the current file, ``TIME`` offsets (cumulative seconds; an
    INCLUDEd file starts from the parent's running offset and its changes do not propagate
    back) and the ``-to`` flag (seconds). Returns (day, sec, freq_MHz, err_us, system) with
    sec the seconds of day including offsets. Used as the reference for the TOA-identity check.
    """
    _seen = _seen if _seen is not None else set()
    path = path.resolve()
    if path in _seen:
        return []
    _seen.add(path)
    out: list[tuple] = []
    skipping = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("C ", "#", "c ", "CC ")) or line == "C":
            continue
        tok = line.split()
        key = tok[0].upper()
        if key == "SKIP":
            skipping = True
            continue
        if key == "NOSKIP":
            skipping = False
            continue
        if skipping:
            continue
        if key == "END":
            break
        if key == "INCLUDE":
            out += tim_records(path.parent / tok[1], offset, _seen)
            continue
        if key == "TIME":
            offset += float(tok[1])
            continue
        if key in TIM_DIRECTIVES or tok[0].startswith("-"):
            continue
        if len(tok) < 5:
            continue
        try:
            freq, err = float(tok[1]), float(tok[3])
            day_s, _, frac_s = tok[2].partition(".")
            day, frac = int(day_s), float("0." + (frac_s or "0"))
        except ValueError:
            continue
        fl = dict(zip(tok[5::2], tok[6::2], strict=False))
        to = float(fl.get("-to", 0.0)) if _is_float(fl.get("-to", "0")) else 0.0
        system = fl.get("-group") or fl.get("-sys") or fl.get("-f") or ""
        out.append((day, frac * 86400.0 + offset + to, freq, err, system))
    return out


def tim_summary(path: Path) -> dict:
    a = parse_tim(path)
    if not a or a["ntoa"] == 0:
        return {"tim_ntoa": 0}
    mjd = a["mjd"]
    fl = a["flags"]

    def nval(f):
        return len(fl[f]) if f in fl else 0

    return {
        "tim_ntoa": a["ntoa"],
        "mjd_min": min(mjd),
        "mjd_max": max(mjd),
        "span_yr": (max(mjd) - min(mjd)) / 365.25,
        "fmin_mhz": min(a["freq"]),
        "fmax_mhz": max(a["freq"]),
        "observatories": ",".join(sorted(a["obs"])),
        "n_f": nval("f"),
        "n_sys": nval("sys"),
        "n_group": nval("group"),
        "n_be": nval("be"),
        "backends_be": ",".join(sorted(fl["be"])) if "be" in fl else "",
        "systems": ",".join(sorted(fl["f"] or fl["sys"] or fl["group"])) if fl else "",
        "has_pta_flag": "pta" in fl,
        "n_include": a["n_include"],
        "tim_TIME": a["has_time"],
        "tim_JUMP": a["has_jump_directive"],
        "wideband_toas": a["wideband"],
    }


# --------------------------------------------------------------------------------------
# Canonicalisation: minimal, auditable rewrites that let PINT read the as-released files
# --------------------------------------------------------------------------------------

CANON = OUT / "canon"
TIM_CMDS = ("FORMAT", "MODE", "INCLUDE", "TIME", "JUMP", "EFAC", "EQUAD", "PHASE", "SKIP", "NOSKIP", "END",
            "INFO", "TRACK", "EMIN", "EMAX", "FMIN", "FMAX", "SIGMA", "PHA1", "PHA2", "DITHER", "EMAP",
            "SEARCH", "SIM", "ZAWGT")


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def canon_tim(src: Path, dst: Path, fixes: Counter, counter: list[int]) -> None:
    """Rewrite a tempo2 FORMAT 1 tim file (and its INCLUDEs, mirrored relative to dst):

    * the first token of every TOA line (the archive name) becomes ``toaNNNNNN`` and leading
      whitespace is removed: PINT classifies a line by its first characters (a leading blank
      with '.' in column 42 = Parkes format, '[0-9a-z@] ' = Princeton, a name starting with a
      tim command word = command), so free-form archive names break parsing;
    * a flag without a value (e.g. InPTA DR2 ``-cycle_post34``) gets the value ``1``;
    * indented comment lines (`` C ...``, EPTA) are written unindented (PINT would otherwise
      parse them as TOAs); a line holding only flags (a broken continuation line, EPTA DR2full
      J1738+0333) is commented out, as tempo2 cannot attach it to a TOA either; so is any other
      line that is neither a directive nor a parseable TOA (EPTA DR2full ``C<non-ASCII> ...``);
    * ``END`` and everything after it in that file is commented out (tempo2 ends only the
      current INCLUDEd file at END, PINT stops reading all TOAs: EPTA DR2full);
    * a file whose TIME statements do not sum to zero gets a compensating ``TIME`` at its end
      (tempo2 keeps TIME local to the file; PINT carries it into later INCLUDEd files);
      trailing tokens after the TIME value (``TIME -1 -group ...``) are dropped;
    * everything else (directives, comments, flags) is copied unchanged.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = []
    ended = False
    net_time = 0.0
    for raw in src.read_text(errors="replace").splitlines():
        line = raw.strip()
        tok = line.split()
        if ended or (tok and tok[0].upper() == "END"):
            # tempo2 stops reading the current file at END but continues with the parent;
            # PINT stops reading altogether. Comment out the rest of this file instead.
            if not ended:
                fixes["tim:END-in-file"] += 1
            ended = True
            out.append("# " + line if line else line)
            continue
        if not tok or line.startswith(("C ", "#", "c ", "CC ")) or line == "C":
            if raw != raw.lstrip() and tok:
                fixes["tim:indented-comment"] += 1
            out.append(line)
            continue
        if tok[0].startswith("-"):
            fixes["tim:orphan-flag-line(dropped)"] += 1
            out.append("C " + line)
            continue
        key = tok[0].upper()
        if key == "INCLUDE":
            canon_tim(src.parent / tok[1], dst.parent / tok[1], fixes, counter)
            out.append(line)
            continue
        if key == "TIME" and len(tok) >= 2 and _is_float(tok[1]):
            net_time += float(tok[1])
            out.append(f"TIME {tok[1]}")
            if len(tok) > 2:
                fixes["tim:TIME-trailing-tokens(dropped)"] += 1
            continue
        if key in TIM_CMDS:
            out.append(line)
            continue
        if not (len(tok) >= 5 and _is_float(tok[1]) and _is_float(tok[2])):
            # neither a directive nor a parseable TOA (e.g. EPTA DR2full 'C<non-ASCII> name ...'):
            # tempo2 skips it, PINT would choke on it; make it an explicit comment
            fixes["tim:unparseable-line(commented)"] += 1
            out.append("# " + line)
            continue
        if len(tok) >= 5 and _is_float(tok[1]) and _is_float(tok[2]):
            counter[0] += 1
            if raw != raw.lstrip() or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.+-]*", tok[0]):
                fixes["tim:archive-name"] += 1
            fl, new = tok[5:], []
            i = 0
            while i < len(fl):
                t = fl[i]
                if t.startswith("-") and not _is_float(t):
                    if i + 1 >= len(fl) or (fl[i + 1].startswith("-") and not _is_float(fl[i + 1])):
                        new += [t, "1"]
                        fixes["tim:valueless-flag"] += 1
                        i += 1
                        continue
                    new += [t, fl[i + 1]]
                    i += 2
                    continue
                new.append(t)
                i += 1
            out.append(" ".join([f"toa{counter[0]:07d}"] + tok[1:5] + new))
            continue
        out.append(line)
    if abs(net_time) > 0:
        # tempo2 keeps TIME offsets local to the file that sets them; PINT carries the running
        # offset into the files INCLUDEd after it (EPTA DR2full J1713+0747: WSRT.P1.2273.tim ends
        # at -2 ms, which PINT applied to every later file). Reset it explicitly.
        out.append(f"TIME {-net_time:+.12g}")
        fixes["tim:TIME-reset-at-end-of-file"] += 1
    dst.write_text("\n".join(out) + "\n")


def canon_par(src: Path, dst: Path, fixes: Counter) -> None:
    """Par-file rewrites that let PINT read tempo2 par files. Each one is intended to reproduce
    tempo2 semantics; none is validated against tempo2 yet (survey Sec. 5, M3 plan Stage M3a):

    * DMXR1_/DMXR2_ carrying a fit flag -> frozen (range bounds are not parameters);
    * DMXR ranges without a DMX_ value -> ``DMX_xxxx 0`` frozen (tempo2 ignores them);
    * PB together with FB1.. (tempo2 allows it, PINT needs FB0) -> FB0 = 1/(PB*86400);
    * a tempo2 par (EPHVER present) without UNITS gets ``UNITS TCB`` (tempo2's default; PINT
      would assume TDB, e.g. all PPTA DR3 par files);
    * a negative M2/H3 is NOT rewritten (it would remove the STIG/SINI design column); the
      leg is reported as unsupported by PINT 1.1.7;
    * if DMX ranges exist but DMX_0001 does not, an empty frozen DMX_0001 range is added (PINT's
      DMX component always holds a DMX_0001 template);
    * TRACK -2 is removed (pulse numbers are absent from the tim files); PINT then tracks
      the nearest pulse. Phase connection must be checked separately (pre-fit wrms vs TRES).
    """
    lines = src.read_text(errors="replace").splitlines()
    keys = {ln.split()[0] for ln in lines if ln.split()}
    dmx_ids = {ln.split()[0][4:] for ln in lines if re.match(r"DMX_\d+\s", ln)}
    has_fb = any(re.match(r"FB[1-9]\d*\s", ln) for ln in lines)
    out = []
    if "UNITS" not in keys and "EPHVER" in keys:  # tempo2 par: no UNITS line means TCB
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
        if k in ("M2", "H3") and _is_float(tok[1]) and float(tok[1]) < 0:
            # NOT rewritten: zeroing a signed H3 also zeroes the STIG design column (the
            # Shapiro delay is H3 * g(STIG, t)), so it changes the marginalised likelihood.
            # PINT 1.1.7 rejects the leg; it needs a signed-H3 evaluator (PINT PR #2023) or tempo2.
            fixes[f"par:signed-{k}(unsupported,kept)"] += 1
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
        # PINT's DMX component always carries a template DMX_0001; without a range it breaks
        # the delay calculation (InPTA DR2 J1600-3053, J1614-2230). Give it an empty range.
        start = min(float(ln.split()[1]) for ln in lines if re.match(r"DMXR1_\d+\s", ln))
        out += ["DMX_0001 0", f"DMXR1_0001 {start - 10:.4f}", f"DMXR2_0001 {start - 9.99:.4f}"]
        fixes["par:empty-DMX_0001-template"] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")


def canonicalise(r: dict) -> None:
    fixes: Counter = Counter()
    d = CANON / r["dataset"] / r["psr"]
    par, tim = ROOT / r["par"], ROOT / r["tim"]
    cpar, ctim = d / par.name, d / "tim" / tim.name
    canon_par(par, cpar, fixes)
    canon_tim(tim, ctim, fixes, [0])
    r["canon_par"], r["canon_tim"] = str(cpar.relative_to(ROOT)), str(ctim.relative_to(ROOT))
    r["canon_fixes"] = ",".join(f"{k}={v}" for k, v in sorted(fixes.items()))


# --------------------------------------------------------------------------------------
# PINT load (subprocess)
# --------------------------------------------------------------------------------------


def toa_identity(toas, ref_tim: str) -> dict:
    """Compare PINT's TOAs with the tempo2-semantics text records of the released tim tree:
    count, arrival time (ns), uncertainty, radio frequency and system flag, after sorting both
    by (time, frequency)."""
    import numpy as np

    ref = tim_records(Path(ref_tim))
    from pint.pulsar_mjd import time_to_longdouble

    col = toas.table["mjd"]
    if col.dtype == object:  # a Column of scalar Time objects (PINT 1.1.7)
        t = np.array([time_to_longdouble(x) for x in col], dtype=np.longdouble)
    else:
        t = np.asarray(time_to_longdouble(col), dtype=np.longdouble)
    # PINT's "mjd" column already includes the clock corrections (observatory chain + BIPM)
    # *and* the TIME / -to offsets; their sum is the "clkcorr" flag [s] and the offsets alone
    # are the "to" flag. Remove clock corrections only, keeping the offsets (as tempo2 does)
    clk = np.array([float(fl.get("clkcorr", 0.0)) - float(fl.get("to", 0.0)) for fl in toas.table["flags"]],
                   dtype=np.longdouble)
    t = t - clk / 86400
    day = np.floor(t)
    sec = np.asarray((t - day) * 86400, dtype=np.float64)
    freq = toas.get_freqs().value
    err = toas.get_errors().value
    sysv = [fl.get("group") or fl.get("sys") or fl.get("f") or "" for fl in toas.table["flags"]]
    def order(recs):
        # group by (frequency, uncertainty, system) and order each group by time: sub-band TOAs
        # of one observation can lie within ns of each other, so a pure time sort would pair
        # them differently on the two sides under sub-ns rounding differences
        return sorted(recs, key=lambda x: (round(x[2], 6), round(x[3], 6), x[4], x[0], x[1]))

    a = order(list(zip(day.astype(np.int64).tolist(), sec.tolist(), freq.tolist(), err.tolist(), sysv, strict=True)))
    b = order(ref)
    row = {"id_n_pint": len(a), "id_n_text": len(b)}
    if len(a) != len(b):
        row["id_ok"] = False
        return row
    dt = np.array([(x[0] - y[0]) * 86400.0 + (x[1] - y[1]) for x, y in zip(a, b, strict=True)])
    row.update(
        id_max_dt_ns=float(np.max(np.abs(dt)) * 1e9) if len(dt) else 0.0,
        id_max_dfreq_mhz=float(max((abs(x[2] - y[2]) for x, y in zip(a, b, strict=True)), default=0.0)),
        id_max_derr_us=float(max((abs(x[3] - y[3]) for x, y in zip(a, b, strict=True)), default=0.0)),
        id_n_sys_mismatch=int(sum(x[4] != y[4] for x, y in zip(a, b, strict=True))),
    )
    row["id_ok"] = (row["id_max_dt_ns"] < 2.0 and row["id_max_dfreq_mhz"] < 1e-6
                    and row["id_max_derr_us"] < 1e-9 and row["id_n_sys_mismatch"] == 0)
    return row


def pint_load(par: str, tim: str, ell1h_shapiro: str = "full", ref_tim: str | None = None) -> dict:
    import warnings

    import astropy.units as u
    import numpy as np
    import pint.logging
    from loguru import logger
    from pint.models import get_model_and_toas
    from pint.residuals import Residuals

    msgs: list[str] = []
    pint.logging.setup(level="WARNING")
    logger.remove()
    logger.add(lambda m: msgs.append(m.record["message"][:300]), level="WARNING")
    warnings.simplefilter("always")
    caught = warnings.catch_warnings(record=True)
    wlist = caught.__enter__()
    t0 = time.time()
    row: dict = {}
    try:
        # model-only pre-pass: PINT's TCB->TDB path re-parses the par with its own warning
        # filters, which hides e.g. "PINT does not support 'DILATEFREQ Y'"; collect those here
        from pint.models import get_model

        get_model(par, allow_tcb=True, allow_T2=True, ell1h_shapiro=ell1h_shapiro)
    except Exception:  # noqa: BLE001, S110 - the full load below reports the error
        pass
    try:
        model, toas = get_model_and_toas(par, tim, planets=True, allow_tcb=True, allow_T2=True,
                                         ell1h_shapiro=ell1h_shapiro)
        if ref_tim is not None:
            try:
                row.update(toa_identity(toas, ref_tim))
            except Exception as ex:  # noqa: BLE001
                row.update(id_ok=False, id_error=f"{type(ex).__name__}: {str(ex)[:200]}")
        res = Residuals(toas, model)
        r = res.time_resids.to_value(u.s)
        row["pint_track_mode"] = res.track_mode
        e = toas.get_errors().to_value(u.s)
        w = 1 / e**2
        wrms = float(np.sqrt(np.sum(w * (r - np.sum(w * r) / np.sum(w)) ** 2) / np.sum(w)))
        M, _params, _ = model.designmatrix(toas)
        comps = sorted(model.components)
        row.update(
            pint_ok=True,
            pint_ntoa=int(toas.ntoas),
            pint_M_cols=int(M.shape[1]),
            pint_wrms_us=wrms * 1e6,
            pint_chi2r=float(res.chi2_reduced),
            pint_units_in=str(model.UNITS.value),
            pint_binary=str(model.BINARY.value) if "BINARY" in model.params else "",
            pint_noise_components=",".join(c for c in comps if "Noise" in c or "Ecorr" in c or "ScaleToa" in c),
            pint_components=",".join(comps),
            pint_clock=str(model.CLOCK.value),
        )
    except Exception as ex:  # noqa: BLE001
        row.update(pint_ok=False, pint_error=f"{type(ex).__name__}: {str(ex)[:400]}",
                   pint_tb=traceback.format_exc(limit=4)[-1500:])
    caught.__exit__(None, None, None)
    # every warning: PINT's loguru messages and Python warnings (e.g. unsupported DILATEFREQ,
    # TIMEEPH, T2CMETHOD settings, which PINT reports through the warnings module)
    msgs += [f"{w.category.__name__}: {str(w.message)[:300]}" for w in wlist
             if not issubclass(w.category, (ResourceWarning, DeprecationWarning))]
    uniq = sorted(set(msgs))
    row.update(pint_seconds=round(time.time() - t0, 1), pint_n_warnings=len(msgs),
               pint_warnings=" || ".join(uniq))
    return row


# --------------------------------------------------------------------------------------
# Cross-matching
# --------------------------------------------------------------------------------------


def angsep_arcsec(a, b) -> float:
    ra1, d1, ra2, d2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 3600.0


def canonical_names(rows: list[dict], tol_arcsec: float = 30.0) -> dict[tuple[str, str], str]:
    """(dataset, psr label) -> canonical J name, by sky-position clustering."""
    clusters: list[dict] = []
    for r in rows:
        pos = r.get("pos")
        for c in clusters:
            if pos and c["pos"] and angsep_arcsec(pos, c["pos"]) < tol_arcsec:
                c["members"].append(r)
                break
        else:
            clusters.append({"pos": pos, "members": [r]})
    out = {}
    for c in clusters:
        jn = sorted({m["psr"] for m in c["members"] if m["psr"].startswith("J")} |
                    {m["psrj"] for m in c["members"] if m.get("psrj", "").startswith("J")})
        name = jn[0] if jn else c["members"][0]["psr"]
        for m in c["members"]:
            out[(m["dataset"], m["psr"])] = name
    return out


# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(DATASETS))
    ap.add_argument("--no-pint", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--timeout", type=float, default=1200.0, help="per-pulsar PINT timeout [s]")
    ap.add_argument("--canonical", action="store_true",
                    help="also load canonicalised copies (data/processed/m3_survey/canon) of every pair")
    args = ap.parse_args()
    names = args.only or list(DATASETS)
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ds in names:
        pta, desc, loader = DATASETS[ds]
        pairs = loader()
        print(f"[{ds}] {desc}: {len(pairs)} pulsars", flush=True)
        for psr, (par, tim) in pairs.items():
            r = {"dataset": ds, "pta": pta, "psr": psr, "par": str(par.relative_to(ROOT)),
                 "tim": str(tim.relative_to(ROOT)) if tim.exists() else "", "tim_exists": tim.exists()}
            r.update(parse_par(par))
            if tim.exists():
                r.update(tim_summary(tim))
            rows.append(r)

    def run_pint(todo: list[dict], parkey: str, timkey: str, prefix: str) -> None:
        # canonical pass: tempo2 par files (EPHVER present) are evaluated with PINT's
        # ell1h_shapiro="absorbed", i.e. tempo2's ELL1H Shapiro expression (Freire & Wex 2010
        # eq. 28); PINT's default "full" (eq. 29) leaves several-us residuals in ELL1H/DDH legs
        def ell1h(r):
            return "absorbed" if prefix and r.get("ephver") else "full"

        print(f"PINT ({prefix or 'as released'}): loading {len(todo)} par/tim pairs with {args.jobs} workers ...",
              flush=True)
        warm = min(todo, key=lambda r: (ROOT / r[timkey]).stat().st_size)
        w = pint_load(str(ROOT / warm[parkey]), str(ROOT / warm[timkey]))
        print(f"  warm-up {warm['dataset']}/{warm['psr']}: ok={w.get('pint_ok')} {w.get('pint_error', '')}", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [(r, ex.submit(pint_load, str(ROOT / r[parkey]), str(ROOT / r[timkey]), ell1h(r),
                                  str(ROOT / r["tim"])))
                    for r in todo]
            for i, (r, f) in enumerate(futs):
                try:
                    res = f.result(timeout=args.timeout)
                except FutTimeout:
                    res = {"pint_ok": False, "pint_error": f"Timeout: > {args.timeout} s"}
                except Exception as e:  # noqa: BLE001
                    res = {"pint_ok": False, "pint_error": f"worker crash: {e!r}"}
                r.update({prefix + k: v for k, v in res.items()})
                s = "ok " if res.get("pint_ok") else "FAIL"
                print(f"  [{i + 1:3d}/{len(todo)}] {s} {r['dataset']:13s} {r['psr']:13s} "
                      f"{res.get('pint_seconds', 0):6.1f}s {res.get('pint_error', '')[:90]}", flush=True)

    if not args.no_pint:
        todo = [r for r in rows if r["tim_exists"]]
        run_pint(todo, "par", "tim", "")
        if args.canonical:
            for r in todo:
                canonicalise(r)
            run_pint(todo, "canon_par", "canon_tim", "canon_")

    canon = canonical_names(rows)
    for r in rows:
        r["jname"] = canon[(r["dataset"], r["psr"])]

    # ---- per-dataset CSV + JSON
    for ds in names:
        sub = [r for r in rows if r["dataset"] == ds]
        keys = sorted({k for r in sub for k in r if k not in ("pint_tb", "pos")})
        with open(OUT / f"{ds}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(sub)
    (OUT / "survey.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")

    # ---- summaries
    lines = [("| data set | pulsars | with tim | TOAs (text) | MJD range | span max [yr] | units | clock | "
              "ephem | PINT ok | PINT fail | canonical ok |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in names:
        sub = [r for r in rows if r["dataset"] == ds]
        wt = [r for r in sub if r.get("tim_ntoa")]
        lines.append(
            f"| {ds} | {len(sub)} | {len(wt)} | {sum(r.get('tim_ntoa', 0) for r in sub):,} | "
            f"{min(r['mjd_min'] for r in wt):.0f}-{max(r['mjd_max'] for r in wt):.0f} | "
            f"{max(r['span_yr'] for r in wt):.2f} | {dict(Counter(r['units'] for r in sub))} | "
            f"{dict(Counter(r['clock'] for r in sub))} | {dict(Counter(r['ephem'] for r in sub))} | "
            f"{sum(1 for r in sub if r.get('pint_ok'))} | {sum(1 for r in sub if r.get('pint_ok') is False)} | "
            f"{sum(1 for r in sub if r.get('canon_pint_ok'))} |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for label, sets in (("main", MAIN_SET), ("yu_allen", YU_ALLEN_SET)):
        sets = [s for s in sets if s in names]
        membership = defaultdict(set)
        for r in rows:
            if r["dataset"] in sets and r.get("tim_ntoa"):
                membership[r["jname"]].add(r["dataset"])
        n = len(membership)
        mat = [[sum(1 for m in membership.values() if a in m and b in m) for b in sets] for a in sets]
        mult = Counter(len(m) for m in membership.values())
        md = [f"### Overlap ({label}: {', '.join(sets)})", "", "| | " + " | ".join(sets) + " |",
              "|---" * (len(sets) + 1) + "|"]
        md += [f"| {a} | " + " | ".join(str(x) for x in row) + " |" for a, row in zip(sets, mat)]
        md += ["", (f"Unique pulsars: **{n}**; by number of PTAs: {dict(sorted(mult.items()))}; "
                    f"mean PTAs per pulsar {sum(len(m) for m in membership.values()) / max(n, 1):.2f}"), ""]
        md += ["| pulsar | " + " | ".join(sets) + " | n |", "|---" * (len(sets) + 2) + "|"]
        for j in sorted(membership, key=lambda j: (-len(membership[j]), j)):
            md.append(f"| {j} | " + " | ".join("x" if s in membership[j] else "" for s in sets) +
                      f" | {len(membership[j])} |")
        legs = [r for r in rows if r["dataset"] in sets and r.get("tim_ntoa")]
        inv = {"pulsars": n, "legs": len(legs), "multi_pta_pulsars": sum(1 for m in membership.values() if len(m) > 1),
               "toas_text": sum(r["tim_ntoa"] for r in legs), "datasets": sets}
        (OUT / f"inventory_{label}.json").write_text(json.dumps(inv, indent=1) + "\n")
        md += ["", f"Inventory: {inv}"]
        (OUT / f"overlap_{label}.md").write_text("\n".join(md) + "\n")
        print(f"{label}: {n} unique pulsars; multiplicity {dict(sorted(mult.items()))}")

    for pre in ("", "canon_"):
        bad = [r for r in rows if r.get(pre + "pint_ok") is False]
        if bad:
            print(f"PINT failures ({pre or 'as released'}):")
            for r in bad:
                print(f"  {r['dataset']}/{r['psr']}: {r.get(pre + 'pint_error')}")
    print(f"wrote {OUT.relative_to(ROOT)}/")
    sys.exit(0)


if __name__ == "__main__":
    main()
