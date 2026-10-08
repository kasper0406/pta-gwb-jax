"""Oracle glue for the M3a gates (tempo2/libstempo, MetaPulsar v0.9.3, enterprise/discovery).
Kept out of ``src/ptagwb`` (oracle packages are never imported by the package); tempo2 runs in its
own isolated env through ``scripts/t2py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T2PY = ROOT / "scripts" / "t2py"
T2_ENV = Path(os.environ.get("T2_ENV", Path.home() / ".local" / "opt" / "tempo2-env"))
WORK = ROOT / "data" / "processed" / "m3a" / "oracles"
METAPULSAR_SRC = next(iter(sorted((ROOT / "data" / "raw" / "metapulsar_v0.9.3" / "extracted").glob("metapulsar-*"))), None)


def have_tempo2() -> bool:
    return (T2_ENV / "bin" / "python").exists() and T2PY.exists()


def have_metapulsar() -> bool:
    return METAPULSAR_SRC is not None and (METAPULSAR_SRC / "src" / "metapulsar").exists()


# ---------------------------------------------------------------------- tempo2 runtime overlay


def t2_runtime(profile) -> Path:
    """A TEMPO2 runtime that is the env's T2runtime with the profile's pinned clock files laid
    over its clock directory (same observatory/BIPM/GPS clock files for PINT and tempo2), so G3
    compares evaluators rather than clock-file versions. Only tempo2-format (.clk) files are laid
    over (tempo-format time_*.dat files are PINT-only)."""
    with _T2_LOCK:
        if profile.name not in _T2_BUILT:
            _T2_BUILT[profile.name] = _build_t2_runtime(profile)
        return _T2_BUILT[profile.name]


_T2_LOCK = __import__("threading").Lock()
_T2_BUILT: dict = {}


def _build_t2_runtime(profile) -> Path:
    base = T2_ENV / "share" / "tempo2"
    out = WORK / "t2runtime" / profile.name
    if (out / "clock").exists():  # rebuilt once per process (pinned files may have been added)
        shutil.rmtree(out / "clock")
    if True:
        out.mkdir(parents=True, exist_ok=True)
        for sub in base.iterdir():
            if sub.name != "clock" and not (out / sub.name).exists():
                (out / sub.name).symlink_to(sub)
        shutil.copytree(base / "clock", out / "clock")
    # tempo2 chains clock files by their header line "# <from> <to>"; a pinned file replaces the
    # tempo2 file with the same name *and* any tempo2 file with the same (from, to) header (e.g.
    # PINT's mk2utc_observatory.clk is tempo2's mk2utc.clk, a newer version)
    def header(f):
        for ln in f.read_text(errors="replace").splitlines()[:5]:
            if ln.startswith("#") and len(ln[1:].split()) >= 2:
                return tuple(x.upper() for x in ln[1:].split()[:2])
        return None

    by_header = {}
    for f in (base / "clock").glob("*.clk"):
        by_header.setdefault(header(f), []).append(f.name)
    replaced = {}
    for f in sorted(profile.directory.glob("*.clk")):
        targets = {f.name} | set(by_header.get(header(f), []) if header(f) else [])
        for name in targets:
            shutil.copy2(f, out / "clock" / name)
            replaced[name] = f.name
    (out / "overlay.json").write_text(json.dumps(replaced, indent=1, sort_keys=True))
    return out


def t2_par_text(par: Path, profile) -> str:
    """Original (released) tempo2 par with the profile's CLK/EPHEM and without TRACK (both engines
    then track the nearest pulse)."""
    text, _ = profile.apply_to_par(par.read_text(errors="replace"))
    lines = [ln for ln in text.splitlines() if ln.split()[:1] != ["TRACK"]]
    out = []
    for ln in lines:  # tempo2 spells the clock keyword CLK
        tok = ln.split()
        out.append(f"CLK {' '.join(tok[1:])}" if tok[:1] == ["CLOCK"] else ln)
    return "\n".join(out) + "\n"


def run_tempo2(par: Path, tim: Path, profile, tag: str, design: bool = True, timeout: int = 3600) -> dict:
    """libstempo evaluation of (par with profile, released tim tree); returns the npz contents."""
    wd = WORK / "t2" / tag
    wd.mkdir(parents=True, exist_ok=True)
    tpar = wd / "t2.par"
    tpar.write_text(t2_par_text(par, profile))
    out = wd / "t2.npz"
    env = dict(os.environ, TEMPO2_OVERRIDE=str(t2_runtime(profile)))
    cmd = [str(T2PY), str(ROOT / "scripts" / "t2_dump.py"), str(tpar), str(tim), str(out)] + (["--design"] if design else [])
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=str(tim.parent), check=False)
    (wd / "t2.log").write_text(p.stdout[-20000:] + "\n---\n" + p.stderr[-20000:])
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"tempo2 failed for {tag}: {p.stderr[-2000:]}")
    with np.load(out, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------- G3 / G4 metrics


def weighted_projector_complement(WM: np.ndarray):
    """x -> (I - P) x with P the orthogonal projector onto span(WM) (columns normalised first:
    design-matrix columns span ~20 decades in natural units)."""
    WM = WM[:, np.linalg.norm(WM, axis=0) > 0]
    Q, _ = np.linalg.qr(WM / np.linalg.norm(WM, axis=0))

    def proj(x):
        y = x - Q @ (Q.T @ x)
        return y - Q @ (Q.T @ y)

    return proj


def principal_sines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sines of the principal angles between span(A) and span(B) (columns normalised first),
    for min(rank) angles, largest first."""
    def basis(X):
        X = X / np.linalg.norm(X, axis=0)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        return U[:, s > s[0] * 1e-13]

    Ua, Ub = basis(A), basis(B)
    c = np.clip(np.linalg.svd(Ua.T @ Ub, compute_uv=False), 0, 1)
    return np.sort(np.sqrt(np.maximum(0.0, 1 - c**2)))[::-1]


def g3_g4(psr, t2: dict) -> dict:
    """Compare our PINT leg (published profile) with tempo2's evaluation, TOA by TOA."""
    rec = np.asarray(psr.meta["record_index"])
    n = len(rec)
    out = {"n_pint": n, "n_tempo2": len(t2["freqs"]), "n_excluded_by_us": int(len(t2["freqs"]) - n)}
    if rec.max() >= out["n_tempo2"] or len(set(rec.tolist())) != n:
        out.update(g3_ok=False, g4_ok=False, note="record indices do not map onto tempo2's TOAs")
        return out
    # align tempo2 arrays (tim order = record order) with our barycentric-sorted arrays
    f_t2 = t2["freqs"][rec]
    out["max_dfreq_mhz"] = float(np.max(np.abs(f_t2 - psr.freqs_topo)))
    sig = np.asarray(psr.toaerrs)
    r_ours = np.asarray(psr.residuals)
    r_t2 = np.asarray(t2["residuals"])[rec]
    d = r_ours - r_t2
    P = weighted_projector_complement(psr.Mmat / sig[:, None])
    dp = P(d / sig)  # whitened (by raw sigma), projected difference
    out.update(
        rms_res_ours_us=float(np.sqrt(np.mean(r_ours**2)) * 1e6),
        rms_res_t2_us=float(np.sqrt(np.mean(r_t2**2)) * 1e6),
        rms_diff_raw_ns=float(np.sqrt(np.mean((d - np.average(d, weights=sig**-2)) ** 2)) * 1e9),
        rms_diff_proj_ns=float(np.sqrt(np.mean((dp * sig) ** 2)) * 1e9),
        max_diff_proj_ns=float(np.max(np.abs(dp * sig)) * 1e9),
        rms_diff_proj_sigma=float(np.sqrt(np.mean(dp**2))),
        max_diff_proj_sigma=float(np.max(np.abs(dp))),
    )
    out["g3_ok"] = bool(out["rms_diff_proj_ns"] < 1.0 and out["rms_diff_proj_sigma"] < 0.01)
    if "design" in t2:
        Mt2 = np.asarray(t2["design"])[rec]
        ok_cols = np.linalg.norm(Mt2, axis=0) > 0
        WMp, WMt = psr.Mmat / sig[:, None], Mt2[:, ok_cols] / sig[:, None]
        s_pt = principal_sines(WMp, WMt)
        out.update(ncol_pint=int(psr.Mmat.shape[1]), ncol_tempo2=int(ok_cols.sum()),
                   tempo2_zero_cols=[str(x) for x in np.asarray(t2["design_params"])[~ok_cols]],
                   g4_max_sin=float(s_pt[0]) if len(s_pt) else float("nan"),
                   g4_sines_top3=[float(x) for x in s_pt[:3]])
        out["g4_ok"] = bool(out["ncol_pint"] == out["ncol_tempo2"] and out["g4_max_sin"] < 1e-6)
    return out


def load_cached_leg(tag: str, dataset: str, label: str):
    from ptagwb.data import Pulsar
    from ptagwb.legs import leg_cache_path

    p = leg_cache_path(tag, dataset, label)
    psr = Pulsar.load(p)
    meta = json.loads((p.parent / "leg_meta.json").read_text())
    return psr, meta


# ---------------------------------------------------------------------- MetaPulsar v0.9.3 (option C / B)


def metapulsar_build(legs: list[tuple[str, str]], ptas: list[str], strategy: str, clock_profile, site_profiles: dict):
    """Build the MetaPulsar v0.9.3 object for a multi-leg pulsar from the same canonical inputs as
    ours (canonical par text incl. indicator-flag rewrites, flat tim): ``strategy`` "consistent"
    (ParameterManager.make_parfiles_consistent, v0.9.3 defaults, then PINT loads of the
    consistent pars) or "composite". Runs in the calling process (one clock profile per process).
    Returns (MetaPulsar, {pta: (model, toas)})."""
    import sys
    import tempfile
    from collections import Counter

    sys.path.insert(0, str(METAPULSAR_SRC / "src"))
    from metapulsar.metapulsar import MetaPulsar
    from metapulsar.parameter_manager import ParameterManager
    from pint.models import get_model_and_toas

    from ptagwb.legs import canonical_leg_texts, leg_cache_path
    from ptagwb.m3data import leg_files
    from ptagwb.profiles import apply_site_profile
    from ptagwb.timfile import read_tim

    clock_profile.activate()
    file_data, tims = {}, {}
    for (ds, lab), pta in zip(legs, ptas, strict=True):
        par, tim = leg_files(ds)[lab]
        _, _, text, _ = canonical_leg_texts(par, tim, Counter())
        text, _ = clock_profile.apply_to_par(text)
        # v0.9.3 copies the reference's CLOCK *or* CLK key verbatim, so a CLOCK reference and a
        # CLK target end up with both keys (PINT then refuses the par). Harness adaptation: spell
        # the keyword CLK everywhere (same meaning for PINT and tempo2).
        text = "\n".join(("CLK " + " ".join(ln.split()[1:])) if ln.split()[:1] == ["CLOCK"] else ln
                          for ln in text.splitlines()) + "\n"
        recs, _ = read_tim(tim)
        apply_site_profile(site_profiles[pta], [r.obs for r in recs])
        flat = leg_cache_path("published", ds, lab).parent / f"{lab}.flat.tim"
        file_data[pta] = {"par": None, "par_content": text, "timing_package": "pint", "tim": str(flat)}
        tims[pta] = str(flat)
    if strategy == "consistent":
        outdir = Path(tempfile.mkdtemp(prefix="mp093_"))
        pm = ParameterManager(file_data=file_data, output_dir=outdir, pulsar_name="psr")
        pars = {k: str(v) for k, v in pm.make_parfiles_consistent().items()}
    else:
        pars = {}
        for pta, fd in file_data.items():
            f = Path(tempfile.mkdtemp(prefix="mpcomp_")) / f"{pta}.par"
            f.write_text(fd["par_content"])
            pars[pta] = str(f)
    pulsars = {}
    for pta in ptas:
        m, t = get_model_and_toas(pars[pta], tims[pta], planets=True, allow_T2=True, allow_tcb=True)
        pulsars[pta] = (m, t)
    mp = MetaPulsar(pulsars=pulsars, combination_strategy=strategy)
    return mp, pulsars


def compare_with_metapulsar(ours, mp) -> dict:
    """TOA-matched comparison of our stacked multi-leg pulsar with a MetaPulsar object:
    residuals per leg and the (weighted) timing column space."""
    names = np.asarray(mp.flags["name"]).astype(str)
    ptas = np.asarray(mp.flags["pta_dataset"]).astype(str)
    key_mp = {(p, n): i for i, (p, n) in enumerate(zip(ptas, names, strict=True))}
    our_names = np.asarray(ours.flags["name"]).astype(str)
    our_ptas = np.asarray(ours.flags["pta"]).astype(str)
    idx = np.array([key_mp[(p, n)] for p, n in zip(our_ptas, our_names, strict=True)])
    out = {"n_ours": int(ours.ntoa), "n_metapulsar": len(names), "ncol_ours": int(ours.Mmat.shape[1]),
           "ncol_metapulsar": int(mp.Mmat.shape[1])}
    r_mp = np.asarray(mp.residuals)[idx]
    sig = np.asarray(ours.toaerrs)
    per_leg = {}
    for p in sorted(set(our_ptas.tolist())):
        m = our_ptas == p
        d = ours.residuals[m] - r_mp[m]
        per_leg[p] = {"max_abs_dres_ns": float(np.max(np.abs(d)) * 1e9), "rms_dres_ns": float(np.sqrt(np.mean(d**2)) * 1e9)}
    out["residuals"] = per_leg
    Mmp = np.asarray(mp.Mmat)[idx]
    s = principal_sines(ours.Mmat / sig[:, None], Mmp / sig[:, None])
    out["colspace_max_sin"] = float(s[0]) if len(s) else float("nan")
    d = ours.residuals - r_mp
    P = weighted_projector_complement(ours.Mmat / sig[:, None])
    out["proj_dres_rms_ns"] = float(np.sqrt(np.mean((P(d / sig) * sig) ** 2)) * 1e9)
    out["fitpars_metapulsar"] = list(mp.fitpars)
    return out
