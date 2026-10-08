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


import importlib.util

HAVE_DISCOVERY_ENTERPRISE = all(
    importlib.util.find_spec(m) is not None for m in ("discovery", "enterprise", "sksparse"))


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


from ptagwb.multileg import PROJ_RTOL, complement_projector


def weighted_projector_complement(WM: np.ndarray, rtol: float = PROJ_RTOL):
    """Rank-revealing complement projector (``ptagwb.multileg.complement_projector``; review M3a #5)."""
    return complement_projector(WM, rtol)


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


# ---------------------------------------------------------------------- G5 on real multi-leg systems


def g5_setup(mps, seed: int = 0, n_rn: int = 10, n_dm: int = 10, n_c: int = 5, global_ecorr_pta: str = "PPTA"):
    """Fixed white noise (random per namespaced system, T2 EQUAD), two ECORR terms (per system, and
    a global term over one PTA's leg that overlaps the first), IRN on each pulsar's own span, DM on
    1.2 x span (nu^-2), HD common process on the array span: distinct grids everywhere."""
    from ptagwb.combined import PulsarGPModel
    from ptagwb.gp import FourierBlock
    from ptagwb.noise import EcorrTerm, build_general_white_noise

    rng = np.random.default_rng(seed)
    Tarr = max(p.toas.max() for p in mps) - min(p.toas.min() for p in mps)
    common = FourierBlock("gw", n_c, Tarr)
    wns, models, nd = [], [], {}
    for p in mps:
        systems = sorted(set(p.backend_flags.tolist()))
        efeq = {s: (float(rng.uniform(0.8, 1.3)), float(rng.uniform(-7.5, -6.5))) for s in systems}
        ec = {s: float(rng.uniform(-7.5, -6.5)) for s in systems}
        terms = [EcorrTerm("ecorr", {s: p.backend_flags == s for s in systems}, ec)]
        gmask = np.char.startswith(p.backend_flags.astype(str), global_ecorr_pta + ":")
        glab = f"{global_ecorr_pta}:all"
        if gmask.any():
            terms.append(EcorrTerm("ecorr_all", {glab: gmask}, {glab: -7.2}))
            nd[f"{p.name}_{glab}_log10_ecorr"] = -7.2
        for s in systems:
            nd[f"{p.name}_{s}_efac"], nd[f"{p.name}_{s}_log10_t2equad"] = efeq[s]
            nd[f"{p.name}_{s}_log10_ecorr"] = ec[s]
        wns.append(build_general_white_noise(p.toas, p.toaerrs, p.backend_flags, efeq, terms, convention="t2"))
        span = p.toas.max() - p.toas.min()
        models.append(PulsarGPModel(sampled={"rn": FourierBlock("red_noise", n_rn, span),
                                             "dm": FourierBlock("dm_gp", n_dm, 1.2 * span, chrom_idx=2.0)},
                                    common=common))
    return wns, models, nd, Tarr


def g5_params(mps, rng, n: int):
    pts = []
    for _ in range(n):
        P = len(mps)
        pts.append({"rn_log10_A": rng.uniform(-15, -13, P), "rn_gamma": rng.uniform(1.5, 5, P),
                    "dm_log10_A": rng.uniform(-14.5, -12.5, P), "dm_gamma": rng.uniform(1, 4, P),
                    "log10_A": np.asarray(rng.uniform(-15, -13.8)), "gamma": np.asarray(rng.uniform(2.5, 5))})
    return pts


def _named(mps, p):
    out = {"gw_log10_A": float(p["log10_A"]), "gw_gamma": float(p["gamma"])}
    for i, m in enumerate(mps):
        out[f"{m.name}_rn_log10_A"], out[f"{m.name}_rn_gamma"] = float(p["rn_log10_A"][i]), float(p["rn_gamma"][i])
        out[f"{m.name}_dm_log10_A"], out[f"{m.name}_dm_gamma"] = float(p["dm_log10_A"][i]), float(p["dm_gamma"][i])
    return out


def discovery_g5(mps, nd, models, Tarr, n_c, global_ecorr_pta="PPTA"):
    """discovery likelihood of the same system (value + JAX gradient)."""
    import discovery as ds
    from oracle_helpers import to_discovery_pulsar

    psls, dpsrs = [], []
    for p, mdl in zip(mps, models, strict=True):
        dp = to_discovery_pulsar(p, nd)
        dpsrs.append(dp)
        glab = f"{global_ecorr_pta}:all"
        gsel = lambda psr, glab=glab: np.where(np.char.startswith(np.asarray(psr.backend_flags).astype(str),
                                                                   global_ecorr_pta + ":"), glab, "")
        comps = [dp.residuals, ds.makenoise_measurement(dp, dp.noisedict),
                 ds.makegp_ecorr(dp, dp.noisedict, enterprise=True, name="ecorrA")]
        if np.any(gsel(dp) != ""):
            comps.append(ds.makegp_ecorr(dp, dp.noisedict, enterprise=True, selection=gsel, name="ecorrB"))
        comps += [ds.makegp_timing(dp, svd=True),
                  ds.makegp_fourier(dp, ds.powerlaw, mdl.sampled["rn"].n_modes, T=mdl.sampled["rn"].T, name="rn"),
                  ds.makegp_fourier(dp, ds.powerlaw, mdl.sampled["dm"].n_modes, T=mdl.sampled["dm"].T,
                                    fourierbasis=ds.dmfourierbasis, name="dm"),
                  ds.makegp_fourier(dp, ds.powerlaw, n_c, T=Tarr, common=["gw_log10_A", "gw_gamma"], name="gw")]
        psls.append(ds.PulsarLikelihood(comps))
    # discovery's ArrayLikelihood cannot combine variable per-pulsar GPs with a commongp/globalgp,
    # so discovery checks CURN: the common process enters each pulsar as its own GP on the array
    # grid (distinct from the pulsar's IRN/DM grids) with shared hyperparameters; JAX gradients.
    # HD values are checked against enterprise.
    return ds.ArrayLikelihood(psls)


def enterprise_g5(mps, nd, models, Tarr, n_c, global_ecorr_pta="PPTA"):
    """enterprise PTA object of the same system (value only)."""
    from enterprise.signals import (
        gp_bases,
        gp_signals,
        parameter,
        selections,
        signal_base,
        utils,
        white_signals,
    )
    from oracle_helpers import to_enterprise_pulsar

    def pl(name=None):
        return utils.powerlaw(log10_A=parameter.Uniform(-20, -11), gamma=parameter.Uniform(0, 7))

    sel = selections.Selection(selections.by_backend)

    def global_sel(backend_flags):
        m = np.char.startswith(np.asarray(backend_flags).astype(str), global_ecorr_pta + ":")
        return {f"{global_ecorr_pta}:all": m} if m.any() else {}

    gw = gp_signals.FourierBasisCommonGP(
        utils.powerlaw(log10_A=parameter.Uniform(-18, -11)("gw_log10_A"), gamma=parameter.Uniform(0, 7)("gw_gamma")),
        orf=utils.hd_orf(), components=n_c, Tspan=Tarr, name="gw")
    models_e = []
    for p, mdl in zip(mps, models, strict=True):
        rn, dm = mdl.sampled["rn"], mdl.sampled["dm"]
        s = gp_signals.TimingModel(use_svd=True)
        s += white_signals.MeasurementNoise(efac=parameter.Constant(), log10_t2equad=parameter.Constant(), selection=sel)
        s += white_signals.EcorrKernelNoise(log10_ecorr=parameter.Constant(), selection=sel)
        s += gp_signals.EcorrBasisModel(log10_ecorr=parameter.Constant(), selection=selections.Selection(global_sel),
                                        name="ecorrB")
        s += gp_signals.FourierBasisGP(spectrum=pl(), components=rn.n_modes, Tspan=rn.T, name="rn")
        s += gp_signals.BasisGP(pl(), gp_bases.createfourierdesignmatrix_dm(nmodes=dm.n_modes, Tspan=dm.T), name="dm")
        s += gw
        models_e.append(s(to_enterprise_pulsar(p)))
    pta = signal_base.PTA(models_e)
    fixed = {}
    for k in pta.param_names:
        pass
    # constants: map our noise dict onto enterprise's constant names
    for name, v in nd.items():
        fixed[name] = v
    for sc in models_e:
        for sig in sc._signals:
            for prm in getattr(sig, "_params", {}).values() if isinstance(getattr(sig, "_params", None), dict) else []:
                pass
    pta.set_default_params(_enterprise_constants(pta, nd))
    return pta


def _enterprise_constants(pta, nd):
    """enterprise names constants '<psr>_<sel>_efac', '<psr>_<sel>_log10_t2equad', '<psr>_<sel>_log10_ecorr'
    and for the basis ECORR '<psr>_<sel>_ecorrB_log10_ecorr' (checked below)."""
    out = {}
    for sc in pta._signalcollections:
        for sig in sc._signals:
            for pname in getattr(sig, "param_names", []):
                pass
    # enterprise constant parameter names are only visible through the signals' _params dicts
    for sc in pta._signalcollections:
        for sig in sc._signals:
            for key, prm in getattr(sig, "_params", {}).items():
                nm = getattr(prm, "name", key)
                if nm in nd:
                    out[nm] = nd[nm]
                elif nm.replace("_ecorrB_", "_") in nd:
                    out[nm] = nd[nm.replace("_ecorrB_", "_")]
                elif nm.replace("_basis_ecorr_", "_") in nd:
                    out[nm] = nd[nm.replace("_basis_ecorr_", "_")]
    return out


def g3_likelihood_impact(psr, t2: dict, n_points: int = 12, seed: int = 0) -> dict:
    """Post-hoc consequence of the G3/G4 engine differences (not a pre-registered gate): the same
    single-pulsar likelihood (white noise = raw TOA errors with EFAC 1, IRN on the leg's span, DM GP)
    evaluated on PINT's (residuals, design matrix) and on tempo2's; max |difference of lnL shapes|
    over random hyperparameter points (nats)."""
    import jax.numpy as jnp

    from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
    from ptagwb.gp import FourierBlock
    from ptagwb.noise import build_general_white_noise

    rec = np.asarray(psr.meta["record_index"])
    span = float(psr.toas.max() - psr.toas.min())
    model = PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 30, span),
                                   "dm": FourierBlock("dm_gp", 30, span, chrom_idx=2.0)})
    systems = sorted(set(psr.backend_flags.tolist()))
    wn = build_general_white_noise(psr.toas, psr.toaerrs, psr.backend_flags, {s: (1.0, -9.0) for s in systems})
    Mt2 = np.asarray(t2["design"])[rec]
    Mt2 = Mt2[:, np.linalg.norm(Mt2, axis=0) > 0]
    other = type("V", (), {})()
    for k in ("toas", "freqs", "toaerrs", "backend_flags", "name", "pos"):
        setattr(other, k, getattr(psr, k))
    other.residuals, other.Mmat, other.pos_enterprise = np.asarray(t2["residuals"])[rec], Mt2, None
    la = GeneralPTALikelihood([precompute_general(psr, wn, model)], orf=None, common=None)
    lb = GeneralPTALikelihood([precompute_general(other, wn, model)], orf=None, common=None)
    rng = np.random.default_rng(seed)
    va, vb = [], []
    for _ in range(n_points):
        p = {"rn_log10_A": jnp.asarray([rng.uniform(-16, -12.5)]), "rn_gamma": jnp.asarray([rng.uniform(1, 6)]),
             "dm_log10_A": jnp.asarray([rng.uniform(-15, -12)]), "dm_gamma": jnp.asarray([rng.uniform(1, 5)])}
        va.append(float(la.logL(p)))
        vb.append(float(lb.logL(p)))
    va, vb = np.array(va), np.array(vb)
    return {"max_dshape_nats": float(np.max(np.abs((va - va[0]) - (vb - vb[0]))))}
