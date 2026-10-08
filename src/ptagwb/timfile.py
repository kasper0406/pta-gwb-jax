"""tempo2-semantics reader and canonical flattener for FORMAT 1 tim files (M3a, gates G1/G2).

``read_tim`` reproduces tempo2's ``readTim`` (readTimfile.C, tempo2 2026.04.1) for the constructs
that occur in the five released data sets, and refuses everything else:

* **Per-file state.** ``TIME`` offset, ``SKIP`` and ``END`` are local to the file being read: an
  INCLUDEd file starts with TIME = 0 and SKIP off, and the parent's offset and SKIP state are
  unchanged when the child returns (tempo2 keeps them in locals of the recursive ``readTim``).
  ``TIME x`` adds x seconds to the running offset; it applies to the TOAs that follow it.
* **SKIP / NOSKIP.** While skipping, TOAs are dropped and every command except NOSKIP is ignored,
  including TIME, INCLUDE and END.
* **END** stops reading the current file only.
* **INCLUDE** paths are relative to the including file's directory.
* **Comments.** A line whose *first character* is ``C`` or ``#`` is a comment (so ``CJ0437...``
  is a commented-out TOA, but `` C ...`` with leading blanks is not a comment: it is parsed as a
  TOA attempt and dropped because its fields do not parse).
* **TOA lines.** ``sscanf("%s %lf %s %lf %s")`` must yield five fields (name, freq [MHz], SAT
  [MJD string], uncertainty [us], observatory); numbers are parsed like C's strtod (longest valid
  prefix).
* **Flags.** tempo2 scans the *whole* line for ``-`` followed by a non-digit; that token is a
  flag ID and the *next* token is its value, whatever it is (``-projid -beconfig -group X`` gives
  projid = "-beconfig", group = "X"). Duplicates are kept in order.
* **Arrival time.** SAT + TIME + ``-to`` + ``-addsat`` (seconds); ``-padd`` adds pulse phase
  (turns) to the residual (applied by PINT and tempo2 alike, not a time offset).

Unsupported (raise ``TimSemanticsError``): files without a FORMAT 1 header (tempo2 would switch to
Parkes/Princeton fixed formats), ``I``-prefixed (deleted) TOA lines, and the commands EFAC,
EQUAD, EMIN, EMAX, EFLOOR, ESET, FMIN, FMAX, SIGMA, PHASE, JUMP, INFO, T2EFAC, T2EQUAD,
GLOBAL_EFAC, PROFILE_DIR (none occurs in the released data; supporting them would need their
own validated semantics). Duplicated physics flags (``-to``, ``-addsat``, ``-padd``, ``-radd``)
are refused because tempo2 sums them while PINT keeps one.

``write_flat_tim`` writes one flat file that PINT reads with exactly these semantics: no INCLUDE,
TIME, SKIP or END left; the effective offset TIME + to + addsat of every TOA is written as its
single ``-to`` flag (PINT's own TIME handling *replaces* an existing ``-to`` rather than adding to
it, and shares TIME/END state across INCLUDEs); archive names become ``toaNNNNNNN``; flags are
written as tempo2 parsed them (duplicates with equal values once; duplicates with differing
values keep the first occurrence, reported, and refused for selection flags).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNSUPPORTED_COMMANDS = ("EFAC", "EQUAD", "EMIN", "EMAX", "EFLOOR", "ESET", "FMIN", "FMAX", "SIGMA", "PHASE",
                        "JUMP", "INFO", "T2EFAC", "T2EQUAD", "GLOBAL_EFAC", "PROFILE_DIR")
PHYSICS_FLAGS = ("-to", "-addsat", "-padd", "-radd")
# flags that select noise / timing parameters: conflicting duplicates are an error
SELECTION_FLAGS = ("-f", "-fe", "-be", "-sys", "-group", "-g", "-B", "-i", "-pta", "-h", "-v", "-pn", "-chan",
                   "-subint", "-name", "-tobs")
_FLOAT_PREFIX = re.compile(r"^[ \t]*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class TimSemanticsError(ValueError):
    """The tim tree uses a construct whose tempo2 semantics we do not reproduce."""


def _strtod(tok: str):
    """C strtod on a token: the longest valid float prefix, or None."""
    m = _FLOAT_PREFIX.match(tok)
    return float(m.group(0)) if m else None


def _sat_prefix(tok: str) -> str | None:
    m = _FLOAT_PREFIX.match(tok)
    return m.group(0).strip() if m else None


def tempo2_flags(line: str) -> list[tuple[str, str]]:
    """tempo2's flag scan of a TOA line (readTimfile.C L274-327), duplicates kept in order."""
    out = []
    n = len(line)
    i = 0
    while i < n - 1:
        if line[i] == "-" and not ("0" <= line[i + 1] <= "9"):
            j = line.find(" ", i)
            if j < 0:  # flag ID at end of line without a value: tempo2 needs a blank after it
                break
            fid = line[i:j]
            k = j
            while k < n and line[k] == " ":
                k += 1
            e = line.find(" ", k)
            val = line[k:] if e < 0 else line[k:e]
            out.append((fid, val))
            i = (n if e < 0 else e)
            continue
        i += 1
    return out


@dataclass
class TimRecord:
    file: str
    lineno: int
    name: str
    freq: float  # MHz
    sat: str  # MJD string as in the file (validated numeric prefix)
    err: float  # us
    obs: str
    flags: list  # [(flag_id, value)] as tempo2 parses them
    time_offset: float  # TIME offset in effect [s]

    def flag_values(self, fid: str) -> list[str]:
        return [v for k, v in self.flags if k == fid]

    @property
    def offset_s(self) -> float:
        """Total arrival-time offset TIME + to + addsat [s]."""
        tot = self.time_offset
        for k, v in self.flags:
            if k in ("-to", "-addsat"):
                x = _strtod(v)
                tot += x if x is not None else 0.0
        return tot

    def mjd_parts(self) -> tuple[int, np.longdouble]:
        """(integer day, seconds of day incl. all offsets) in long double."""
        day_s, _, frac_s = self.sat.partition(".")
        day = int(day_s)
        sec = np.longdouble("0." + (frac_s or "0")) * np.longdouble(86400) + np.longdouble(self.offset_s)
        return day, sec


@dataclass
class TimReport:
    files: list = field(default_factory=list)
    n_toa: int = 0
    counts: Counter = field(default_factory=Counter)  # events (comment kinds, END, SKIP, ...)
    duplicate_flags: Counter = field(default_factory=Counter)  # flag -> TOAs with differing duplicates


def read_tim(path: Path | str, *, _report: TimReport | None = None, _seen: tuple = ()) -> tuple[list[TimRecord], TimReport]:
    """All TOA records of a tim tree in tempo2 order (see module docstring)."""
    path = Path(path).resolve()
    rep = _report if _report is not None else TimReport()
    if path in _seen:
        raise TimSemanticsError(f"recursive INCLUDE of {path}")
    rep.files.append(str(path))
    text = path.read_text(errors="replace")
    # tempo2 scans the whole file for the word FORMAT; without it the file is read in the
    # fixed-column (Parkes / Princeton / ITOA) mode, in which we accept only INCLUDE, commands
    # and comments (EPTA top-level files are lists of INCLUDEs without FORMAT)
    fixed = not any(w == "FORMAT" for w in text.split())
    if fixed:
        rep.counts["file without FORMAT (fixed-column mode)"] += 1
    recs: list[TimRecord] = []
    time_off = 0.0
    skip = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line:
            continue
        if fixed:
            toks = line.split()
            if toks[0] in ("C", "c"):
                rep.counts["comment"] += 1
                continue
            if toks[0].upper() != "INCLUDE" and (line[0] == " " or (len(line) > 1 and line[1] == " ")
                                                 or (len(line) > 14 and line[14] == ".")):
                raise TimSemanticsError(f"{path}:{lineno}: fixed-column TOA line (Parkes/Princeton/ITOA) not supported")
        elif line[0] in ("C", "#"):
            toks = line[1:].split() if line[0] == "C" else []
            kind = "comment"
            if line[0] == "C" and len(toks) >= 5 and _strtod(toks[1]) is not None:
                # "CJ0437..." (no blank): tempo2 comment, but PINT would read it as a TOA
                kind = "commented TOA (C, no blank)" if line[1] not in (" ", "\t") else "commented TOA (C )"
            rep.counts[kind] += 1
            continue
        toks = line.split()
        if not fixed and line[0] == "I" and toks[0].upper() not in ("INCLUDE", "INFO"):
            raise TimSemanticsError(f"{path}:{lineno}: 'I'-prefixed TOA line (tempo2 marks it deleted)")
        # tempo2: sscanf("%s %lf %s %lf %s") >= 5 fields <=> freq and uncertainty parse
        is_toa = not fixed and len(toks) >= 5 and _strtod(toks[1]) is not None and _strtod(toks[3]) is not None
        if is_toa:
            if skip:
                rep.counts["toa skipped (SKIP)"] += 1
                continue
            sat = _sat_prefix(toks[2])
            if sat is None or sat != toks[2]:
                raise TimSemanticsError(f"{path}:{lineno}: SAT {toks[2]!r} is not a plain decimal MJD")
            flags = tempo2_flags(line)
            for fid in PHYSICS_FLAGS:
                if sum(k == fid for k, _ in flags) > 1:
                    raise TimSemanticsError(f"{path}:{lineno}: duplicated physics flag {fid}")
            if raw != raw.lstrip():
                rep.counts["toa with leading blanks"] += 1
            recs.append(TimRecord(str(path), lineno, toks[0], _strtod(toks[1]), sat, _strtod(toks[3]), toks[4],
                                  flags, time_off))
            continue
        key = toks[0].upper()
        if skip:
            if key == "NOSKIP":
                skip = False
                rep.counts["NOSKIP"] += 1
            else:
                rep.counts["line ignored (SKIP)"] += 1
            continue
        if key == "END":
            rep.counts["END"] += 1
            nrest = sum(1 for x in text.splitlines()[lineno:] if x.strip())
            if nrest:
                rep.counts["lines after END (ignored)"] += nrest
            break
        if key == "TIME":
            v = _strtod(toks[1]) if len(toks) > 1 else None
            if v is None:
                raise TimSemanticsError(f"{path}:{lineno}: TIME without a value")
            time_off += v
            rep.counts["TIME"] += 1
            if len(toks) > 2:
                rep.counts["TIME trailing tokens (ignored by tempo2)"] += 1
            continue
        if key == "INCLUDE":
            if len(toks) < 2:
                raise TimSemanticsError(f"{path}:{lineno}: INCLUDE without a file")
            rep.counts["INCLUDE"] += 1
            sub, _ = read_tim(path.parent / toks[1], _report=rep, _seen=_seen + (path,))
            recs.extend(sub)
            continue
        if key == "SKIP":
            skip = True
            rep.counts["SKIP"] += 1
            continue
        if key == "NOSKIP":
            rep.counts["NOSKIP"] += 1
            continue
        if key in ("FORMAT", "MODE"):
            rep.counts[key] += 1
            continue
        if key in UNSUPPORTED_COMMANDS:
            raise TimSemanticsError(f"{path}:{lineno}: tim command {key} not supported (no validated semantics)")
        if line[0] in (" ", "\t") and toks[0] in ("C", "c"):
            rep.counts["indented comment (not a tempo2 comment; unparseable, dropped)"] += 1
        elif toks[0].startswith("-"):
            rep.counts["flag-only line (dropped)"] += 1
        else:
            rep.counts["unparseable line (dropped)"] += 1
    if _report is None:
        _finish(recs, rep)
    return recs, rep


def _finish(recs, rep):
    for r in recs:
        seen: dict[str, str] = {}
        for k, v in r.flags:
            if k in seen and seen[k] != v:
                rep.duplicate_flags[k] += 1
            seen.setdefault(k, v)
    rep.n_toa = len(recs)


def canonical_flags(rec: TimRecord) -> list[tuple[str, str]]:
    """Flags written to the flat file: physics offsets folded into one ``-to``, duplicates once
    (first value kept; conflicting duplicates of selection flags refused)."""
    out, seen = [], {}
    for k, v in rec.flags:
        if k in ("-to", "-addsat"):
            continue
        if k in seen:
            if seen[k] != v and k in SELECTION_FLAGS:
                raise TimSemanticsError(f"{rec.file}:{rec.lineno}: conflicting duplicate selection flag {k}")
            continue
        if k.lstrip("-") in ("error", "freq", "scale", "MJD", "flags", "obs", "name", "clkcorr", "format"):
            raise TimSemanticsError(f"{rec.file}:{rec.lineno}: flag {k} collides with a PINT TOA field")
        seen[k] = v
        out.append((k, v))
    off = rec.offset_s
    if off != 0.0:
        out.append(("-to", repr(float(off))))
    return out


def write_flat_tim(recs: list[TimRecord], dst: Path | str) -> Path:
    """One flat FORMAT 1 file with the tempo2 semantics made explicit (module docstring)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = ["FORMAT 1"]
    for i, r in enumerate(recs, 1):
        fl = " ".join(f"{k} {v}" for k, v in canonical_flags(r))
        lines.append(f"toa{i:07d} {r.freq!r} {r.sat} {r.err!r} {r.obs} {fl}".rstrip())
    tmp = dst.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(dst)
    return dst
