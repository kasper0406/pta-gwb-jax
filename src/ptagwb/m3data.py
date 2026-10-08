"""M3 data sets: file discovery for the five released PTA data sets, the selected configuration,
the quarantine registry and the M3a validation set (docs/M3_PLAN.md Sec. 1, 5.1; M3A_VALIDATION.md).

Dataset discovery mirrors ``scripts/m3_survey.py`` (which produced the survey numbers) but lives in
the package so that ingestion, tests and the validation driver share one definition.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import RAW_DIR, REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs" / "m3"


def _one(pattern: str) -> Path:
    cands = sorted(RAW_DIR.glob(pattern))
    if len(cands) != 1:
        raise FileNotFoundError(f"{pattern}: expected one match under {RAW_DIR}, got {cands}")
    return cands[0]


def _ng15():
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
    return {d.name: (d / f"{d.name}.par", d / f"{d.name}_all.tim") for d in sorted(p for p in base.iterdir() if p.is_dir())}


def _ppta_gh():
    base = _one("ppta_dr3_github/extracted/PPTA-DR3-*") / "analysis_codes" / "data" / "all"
    return {p.stem: (p, base / f"{p.stem}.tim") for p in sorted(base.glob("J*.par")) if "singlePsrNoise" not in p.name}


def _inpta_dr2():
    base = _one("inpta_dr2/extracted/InPTA.DR2-*")
    return {d.name: (d / f"{d.name}.DMX.par", d / f"{d.name}_all.tim")
            for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("J"))}


def _inpta_dr1():
    base = _one("inpta_dr1/extracted/InPTA.DR1-*")
    return {d.name: (d / f"{d.name}.NB.par", d / f"{d.name}.NB.tim")
            for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("J"))}


def _mpta():
    base = RAW_DIR / "mpta_4p5yr" / "extracted" / "partim"
    return {p.stem: (p, base / f"{p.stem}.tim") for p in sorted(base.glob("J*.par"))}


# dataset -> (PTA label, loader). PTA labels are the namespaces of system labels and parameters.
DATASETS = {
    "ng15": ("NG15", _ng15),
    "epta_dr2new": ("EPTA", lambda: _epta("DR2new")),
    "ppta_dr3_gh": ("PPTA", _ppta_gh),
    "inpta_dr2": ("InPTA", _inpta_dr2),
    "inpta_dr1": ("InPTA", _inpta_dr1),
    "mpta": ("MPTA", _mpta),
}
SELECTED = ("ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr2", "mpta")  # selected configuration
YU_ALLEN = ("ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr1", "mpta")
# YA reference order (MetaPulsar v0.9.3 configuration; InPTA is never the reference)
REFERENCE_ORDER = ("NG15", "EPTA", "PPTA", "MPTA")


def leg_files(dataset: str) -> dict[str, tuple[Path, Path]]:
    """psr label -> (par, tim) of every leg of ``dataset`` that has a tim file."""
    _, loader = DATASETS[dataset]
    return {k: v for k, v in loader().items() if v[1].exists()}


def pta_of(dataset: str) -> str:
    return DATASETS[dataset][0]


def load_json_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text())


DEFAULT_CONFIG = {"ell1h_nharms": "tempo2"}


def quarantine(config: dict | None = None) -> dict:
    """(dataset, psr) -> entry, from configs/m3/quarantine.json, for an evaluation configuration.

    Entries in ``lifted`` are lifted only under the configuration recorded in their
    ``lifted_under`` (e.g. ELL1H H3+H4 legs under the tempo2 harmonic convention); under any other
    configuration (e.g. PINT's forced 7 harmonics, ``{"ell1h_nharms": "pint7"}``) they are
    quarantined."""
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    q = load_json_config("quarantine.json")
    out = {(e["dataset"], e["psr"]): e for e in q["legs"]}
    for e in q.get("lifted", []):
        cond = e.get("lifted_under", {})
        if any(cfg.get(k) != v for k, v in cond.items()):
            out[(e["dataset"], e["psr"])] = e
    return out


def validation_set() -> dict:
    return load_json_config("validation_set.json")
