"""Explicit, versioned timing-evaluation, clock and ephemeris profiles (docs/M3_PLAN.md Sec. 4.1,
4.2; components L10, L4).

Two kinds of clock/ephemeris profile, never mixed:

* **published-analysis profiles** (per PTA, for reproductions): the clock realisation and
  ephemeris the published analysis used, with the release's own observatory clock files where it
  ships them;
* the **combined profile** (for B/C combinations and YA): one realisation and ephemeris for every
  leg.

A profile is applied to a leg by rewriting the par file's CLK/CLOCK and EPHEM lines (recorded in
the leg's provenance) and by pointing PINT at a **pinned clock directory**
(``PINT_CLOCK_OVERRIDE``): the profile's override files plus every other clock file PINT needs,
copied once from PINT's global repository cache. Every file is hashed into a manifest
(``configs/m3/clocks/<profile>.json``, committed). A verified load checks the manifest before
loading and, afterwards, that every clock file PINT actually used came from the pinned directory
with the pinned hash (``check_used_clock_files``); a file from anywhere else is an error. The
first load of a new profile (``pin=True``) adds the missing files and writes the manifest.

Evaluator profiles hold the PINT options that change delays: ``ell1h_shapiro`` ("full" =
Freire & Wex eq. 29, PINT's default; "absorbed" = eq. 28 as tempo2 evaluates ELL1H), ``allow_T2``
and ``allow_tcb`` (TCB par files are converted to TDB by PINT; YA also convert to TDB). In
option C every leg must use the *reference* leg's evaluator profile, chosen before the copy
(``multileg``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR, RAW_DIR, REPO_ROOT

CLOCK_ROOT = DATA_DIR / "processed" / "m3a" / "clocks"
MANIFEST_DIR = REPO_ROOT / "configs" / "m3" / "clocks"
PROFILE_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class EvaluatorProfile:
    name: str
    ell1h_shapiro: str = "full"
    allow_T2: bool = True
    allow_tcb: bool = True

    def pint_kwargs(self) -> dict:
        return {"allow_T2": self.allow_T2, "allow_tcb": self.allow_tcb, "ell1h_shapiro": self.ell1h_shapiro}


PINT_NATIVE = EvaluatorProfile("pint-native", ell1h_shapiro="full")
TEMPO2_PAR = EvaluatorProfile("tempo2-par", ell1h_shapiro="absorbed")


def evaluator_for_par(par_text: str) -> EvaluatorProfile:
    """tempo2 par files (EPHVER line present) are evaluated with tempo2's ELL1H expression."""
    return TEMPO2_PAR if any(ln.split()[:1] == ["EPHVER"] for ln in par_text.splitlines()) else PINT_NATIVE


@dataclass(frozen=True)
class ClockProfile:
    """``clock`` / ``ephem``: value written into the par (None = keep the par's own).
    ``overrides``: release clock files (paths relative to data/raw) pinned under their own names."""

    name: str
    clock: str | None
    ephem: str | None
    overrides: tuple = ()
    note: str = ""

    @property
    def directory(self) -> Path:
        return CLOCK_ROOT / self.name

    @property
    def manifest_path(self) -> Path:
        return MANIFEST_DIR / f"{self.name}.json"

    def apply_to_par(self, text: str) -> tuple[str, list[str]]:
        """Rewrite CLK/CLOCK and EPHEM lines; returns (text, list of changes)."""
        out, changes, seen = [], [], set()
        for ln in text.splitlines():
            tok = ln.split()
            if tok and tok[0] in ("CLK", "CLOCK") and self.clock is not None:
                seen.add("CLK")
                if tok[1:2] != [self.clock]:
                    changes.append(f"{tok[0]} {' '.join(tok[1:])} -> CLOCK {self.clock}")
                out.append(f"CLOCK {self.clock}")
                continue
            if tok and tok[0] == "EPHEM" and self.ephem is not None:
                seen.add("EPHEM")
                if tok[1:2] != [self.ephem]:
                    changes.append(f"EPHEM {' '.join(tok[1:])} -> {self.ephem}")
                out.append(f"EPHEM {self.ephem}")
                continue
            out.append(ln)
        if self.clock is not None and "CLK" not in seen:
            out.append(f"CLOCK {self.clock}")
            changes.append(f"CLOCK (absent) -> {self.clock}")
        if self.ephem is not None and "EPHEM" not in seen:
            out.append(f"EPHEM {self.ephem}")
            changes.append(f"EPHEM (absent) -> {self.ephem}")
        return "\n".join(out) + "\n", changes

    # ------------------------------------------------------------------ pinning
    def prepare(self) -> Path:
        """Create the pinned directory with the override files (idempotent; verifies hashes)."""
        d = self.directory
        d.mkdir(parents=True, exist_ok=True)
        for rel in self.overrides:
            src = RAW_DIR / rel
            dst = d / Path(rel).name
            if not dst.exists():
                shutil.copy2(src, dst)
            elif sha256(dst) != sha256(src):
                raise RuntimeError(f"{self.name}: pinned {dst.name} differs from the release file {src}")
        return d

    def manifest(self) -> dict:
        p = self.manifest_path
        return json.loads(p.read_text()) if p.exists() else {}

    def verify_directory(self) -> None:
        man = self.manifest()
        if not man:
            raise RuntimeError(f"clock profile {self.name}: no manifest {self.manifest_path}; pin it first")
        for name, h in man["files"].items():
            f = self.directory / name
            if not f.exists():
                raise RuntimeError(f"clock profile {self.name}: pinned file {name} missing")
            if sha256(f) != h:
                raise RuntimeError(f"clock profile {self.name}: {name} does not match the pinned hash")

    def activate(self) -> None:
        """Point PINT at the pinned directory (call before PINT reads any clock file)."""
        os.environ["PINT_CLOCK_OVERRIDE"] = str(self.prepare())


def used_clock_files() -> list[tuple[str, Path]]:
    """(name PINT looks up, file actually read) of every clock file the current PINT process has
    loaded (observatories, BIPM, GPS). Files from PINT's global repository are read from the
    astropy download cache under an opaque path; their lookup name is the repository basename."""
    import pint.observatory as po

    objs = []
    if getattr(po, "_gps_clock", None) is not None:
        objs.append(po._gps_clock)
    objs += list(getattr(po, "_bipm_clock_versions", {}).values())
    for name in po.Observatory.names():
        try:
            o = po.get_observatory(name)
        except Exception:  # noqa: BLE001, S112 - observatory not constructible: never used
            continue
        for c in getattr(o, "_clock", None) or []:
            objs.append(c)
    out = set()
    for c in objs:
        inner = getattr(c, "clock_file", None)
        if inner is not None:  # GlobalClockFile: repository name + cached copy
            if getattr(inner, "filename", None):
                out.add((Path(c.filename).name, Path(inner.filename)))
        elif getattr(c, "filename", None):
            out.add((Path(c.filename).name, Path(c.filename)))
    return sorted(out)


def check_used_clock_files(profile: ClockProfile, pin: bool = False) -> dict:
    """After a PINT load: every used clock file must be the pinned one. With ``pin``, files from
    elsewhere (PINT's global cache) are copied into the pinned directory and the manifest is
    (re)written; returns {name: sha256} of the files used."""
    import fcntl

    d = profile.prepare().resolve()
    if pin:  # concurrent pinning loads (spawn pool) serialise on a lock file
        lock = open(d / ".lock", "w")  # noqa: SIM115
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        return _check_used(profile, d, pin)
    finally:
        if pin:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


def _check_used(profile: ClockProfile, d: Path, pin: bool) -> dict:
    used = used_clock_files()
    man = profile.manifest().get("files", {})
    result, foreign = {}, []
    for name, f in used:
        f = f.resolve()
        if f.parent != d or f.name != name:
            foreign.append((name, f))
            continue
        h = sha256(f)
        if man and name in man and man[name] != h:
            raise RuntimeError(f"clock profile {profile.name}: {name} changed since pinning")
        result[name] = h
    if foreign:
        if not pin:
            raise RuntimeError(f"clock profile {profile.name}: clock files used from outside the pinned "
                               f"directory: {[(n, str(x)) for n, x in foreign]}")
        for name, f in foreign:
            dst = d / name
            if dst.exists() and sha256(dst) != sha256(f):
                raise RuntimeError(f"{profile.name}: {name} exists in the pinned dir with other content")
            shutil.copy2(f, dst)
            result[name] = sha256(dst)
    if pin:
        files = dict(man)
        files.update(result)
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        profile.manifest_path.write_text(json.dumps(
            {"profile": profile.name, "version": PROFILE_VERSION, "clock": profile.clock, "ephem": profile.ephem,
             "overrides": list(profile.overrides), "note": profile.note,
             "files": dict(sorted(files.items()))}, indent=1) + "\n")
    elif man:
        missing = [n for n in result if n not in man]
        if missing:
            raise RuntimeError(f"clock profile {profile.name}: files not in the manifest: {missing}")
    return result


def _ng15_clock_files() -> tuple:
    rel = sorted(RAW_DIR.glob("ng15_v2.1.0/extracted/NANOGrav*"))
    if not rel:
        return ()
    base = rel[0] / "clock"
    return tuple(str(p.relative_to(RAW_DIR)) for p in sorted(base.iterdir())
                 if p.is_file() and p.suffix in (".clk", ".dat") and p.name != "ut1.dat")


def _glob1(pattern: str) -> str:
    c = sorted(RAW_DIR.glob(pattern))
    return str(c[0].relative_to(RAW_DIR)) if c else pattern


EPTA_NCY = "epta_dr2_gitlab/extracted/epta-dr2-*/EPTA-DR2/clockfiles/ncyobs2obspm.clk"
PPTA_PKS = "ppta_dr3_github/extracted/PPTA-DR3-*/clockfiles/pks2gps.clk"


def published_profile(pta: str) -> ClockProfile:
    if pta == "NG15":
        return ClockProfile("ng15-published-v1", "TT(BIPM2019)", "DE440", _ng15_clock_files(),
                            "NG15 v2.1.0 release clock files (as M1/M2)")
    if pta == "EPTA":
        return ClockProfile("epta-dr2-published-v1", "TT(BIPM2021)", "DE440", (_glob1(EPTA_NCY),),
                            "release's corrected Nancay clock file")
    if pta == "PPTA":
        return ClockProfile("ppta-dr3-published-v1", "TT(BIPM2020)", "DE440", (_glob1(PPTA_PKS),),
                            "DE440 is the GW paper's override (the par files say DE436)")
    if pta == "InPTA":
        return ClockProfile("inpta-dr2-published-v1", "TT(BIPM2023)", "DE440", (), "")
    if pta == "MPTA":
        return ClockProfile("mpta-published-v1", None, "DE440", (),
                            "[UNRESOLVED] par CLK TT(BIPM2020) (81/83) vs paper BIPM2022: par value kept")
    raise KeyError(pta)


COMBINED = ClockProfile("combined-v1", "TT(BIPM2023)", "DE440", (_glob1(EPTA_NCY), _glob1(PPTA_PKS)),
                        "one realisation and ephemeris for all legs; release observatory files pinned")


@dataclass
class LegProfile:
    """Everything that fixes how one leg is evaluated (recorded in its provenance)."""

    clock: ClockProfile
    evaluator: EvaluatorProfile
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"clock_profile": self.clock.name, "clock": self.clock.clock, "ephem": self.clock.ephem,
                "evaluator": self.evaluator.name, "ell1h_shapiro": self.evaluator.ell1h_shapiro, **self.extra}
