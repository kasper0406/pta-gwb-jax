"""EPTA DR2new oracle in the EPTA fork env (run with ``scripts/eptapy``; docs/M3B_PLAN.md Sec. 4.1,
4.5, 6.2). Never imported by ``src/ptagwb``.

Builds what EPTA's ``model_single.py`` (original version, epta-dr2 be91c6b) built:
``enterprise.pulsar.Pulsar(par, tim, ephem='DE440')`` (tempo2 via libstempo, the par's clock) and
``enterprise_extensions.models.model_general(...)`` with the script's keyword arguments, using the
EPTA fork (enterprise EPTADR2-v1.1, enterprise_extensions 051173f).

Subcommands
-----------
export   --variant original|canonical
         every consumed array of every pulsar -> data/processed/m3b/epta/export/enterprise_<variant>/
like     --variant V --orf crn|hd --points P.npz --out O.npz [--ncommon 9]
         log-likelihood and log-prior of the model at parameter vectors (rows of P['x'], columns
         in P['names'] order, mapped onto pta.param_names) + the model's structure (parameter
         names, per-signal basis frequencies and shapes, Tspan_common).

Environment: TEMPO2 must point at data/processed/m3b/epta/t2runtime (``TEMPO2_OVERRIDE``), which
the script checks.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed" / "m3b" / "epta"
RAW = sorted((ROOT / "data" / "raw" / "epta_dr2_gitlab" / "extracted").glob("epta-dr2-*"))[0] / "EPTA-DR2"
NOISEDIR = RAW / "noisefiles" / "DR2new"


def check_env():
    """TEMPO2 must be the profile runtime (data/processed/m3b/epta/t2runtime) or, for diagnostics
    only, a variant runtime next to it (t2runtime-<tag>)."""
    t2 = Path(os.environ.get("TEMPO2", "")).resolve()
    if t2.parent != BASE.resolve() or not t2.name.startswith("t2runtime"):
        sys.exit(f"TEMPO2={t2} is not an M3b EPTA runtime under {BASE} (set TEMPO2_OVERRIDE)")
    return t2.name


def files(variant):
    """``original`` (release), ``canonical`` (ours) or ``dir:<path>`` (a diagnostic data tree with
    <psr>/<psr>.par and <psr>/<psr>_all.tim, e.g. an older epta-dr2 revision)."""
    prep = json.loads((BASE / "canonical" / "prepare.json").read_text())
    if variant.startswith("dir:"):
        d = Path(variant[4:])
        return {psr: (str(d / psr / f"{psr}.par"), str(d / psr / f"{psr}_all.tim")) for psr in sorted(prep)}
    out = {}
    for psr, r in sorted(prep.items()):
        if variant == "original":
            out[psr] = (r["par"], r["tim"])
        elif variant == "canonical":
            out[psr] = (r["canon_par"], r["canon_tim"])
        else:
            raise ValueError(variant)
    return out


def load_psrs(variant, keep_t2=False):
    from enterprise.pulsar import Pulsar

    tag = variant if not variant.startswith("dir:") else "dir_" + Path(variant[4:]).name
    cache = BASE / "export" / f"psrs_{tag}_{Path(os.environ['TEMPO2']).name}.pkl"
    if cache.exists() and not keep_t2:
        with open(cache, "rb") as f:
            return pickle.load(f)
    psrs = []
    for psr, (par, tim) in files(variant).items():
        p = Pulsar(par, tim, ephem="DE440", drop_t2pulsar=not keep_t2)  # as model_single.py
        psrs.append(p)
    if not keep_t2:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(psrs, f)
    return psrs


def cmd_export(a):
    import enterprise
    import libstempo
    import libstempo.libstempo as L

    out = BASE / "export" / f"enterprise_{a.variant}" if a.variant in ("original", "canonical") else Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    psrs = load_psrs(a.variant, keep_t2=True)
    for p in psrs:
        t2 = p.t2pulsar
        fl = p.flags
        d = {
            "name": np.array(p.name), "toas": p.toas, "stoas": p.stoas, "residuals": p.residuals,
            "toaerrs": p.toaerrs, "freqs": p.freqs, "Mmat": p.Mmat, "fitpars": np.array(p.fitpars),
            "setpars": np.array(p.setpars), "backend_flags": p.backend_flags.astype(str),
            "flags_json": np.array(json.dumps({k: np.asarray(v).astype(str).tolist() for k, v in fl.items()})),
            "pos": p.pos, "pos_t": p.pos_t, "sunssb": p.sunssb, "planetssb": p.planetssb,
            "telescope": p.telescope.astype(str), "isort": np.asarray(p.isort), "raj": np.array(p._raj),
            "decj": np.array(p._decj), "pdist": np.array(p.pdist),
            "t2_deleted": np.asarray(t2.deleted, dtype=bool)[p.isort], "t2_nobs": np.array(int(t2.nobs)),
            "t2_site_freqs": np.asarray(t2.freqs, dtype=np.float64)[p.isort],
            "mmat_sv": np.linalg.svd(p.Mmat / np.linalg.norm(p.Mmat, axis=0), compute_uv=False),
            "meta": np.array(json.dumps({"enterprise": enterprise.__file__, "libstempo": libstempo.__version__,
                                         "tempo2": str(L.tempo2version()), "TEMPO2": os.environ["TEMPO2"],
                                         "variant": a.variant})),
        }
        np.savez(out / f"{p.name}.npz", **d)
        print(p.name, len(p.toas), p.Mmat.shape, "deleted", int(d["t2_deleted"].sum()), flush=True)


def build_pta(psrs, orf, ncommon):
    from enterprise_extensions import models

    params = {}
    for nf in sorted(NOISEDIR.glob("*_noise.json")):
        params.update(json.loads(nf.read_text()))
    red = json.loads((NOISEDIR / "red_dict.json").read_text())
    dm = json.loads((NOISEDIR / "dm_dict.json").read_text())
    chrom = json.loads((NOISEDIR / "chrom_dict.json").read_text())
    # keyword arguments of model_single.py (be91c6b / 0b35448, the versions matching this fork);
    # --num_dmdips 1 (the chains have one dip), --common_components ncommon, gamma_common None
    return models.model_general(psrs, noisedict=params, orf=orf, orf_bins=None, common_psd="powerlaw",
                                common_components=ncommon, gamma_common=None, bayesephem=False,
                                sat_orb_elements=True, tnequad=True, tm_svd=True, tm_marg=True,
                                red_var=True, red_components=red, dm_var=True, dm_components=dm,
                                dm_chrom=True, chrom_components=chrom, dmchrom_kernel="diag", tndm=True,
                                num_dmdips=1, dmpsr_list=["J1713+0747"], dm_expdip_idx=[1, 4],
                                dm_expdip_tmin=[57490, 54650], dm_expdip_tmax=[57530, 54850],
                                extra_sigs=None, dropout=False, dropout_psr=None)


def structure(pta):
    sig = {}
    for sc in pta._signalcollections:
        for s in sc._signals:
            ent = {"class": s.__class__.__name__, "params": [p.name for p in s.params]}
            try:
                B = s.get_basis()
                if B is not None:
                    ent["basis_shape"] = list(np.shape(B))
                    lab = getattr(s, "_labels", None)
                    if isinstance(lab, dict) and "" in lab:
                        ent["freqs"] = np.asarray(lab[""], dtype=float).tolist()
                    elif lab is not None and not isinstance(lab, dict):
                        ent["freqs"] = np.asarray(lab, dtype=float).tolist()
            except Exception as e:  # noqa: BLE001
                ent["basis_error"] = repr(e)
            sig[f"{sc.psrname}:{s.signal_name}:{s.name}"] = ent
    return sig


def cmd_like(a):
    from enterprise_extensions import model_utils

    psrs = load_psrs(a.variant)
    t0 = time.time()
    pta = build_pta(psrs, a.orf, a.ncommon)
    tb = time.time() - t0
    P = np.load(a.points, allow_pickle=False)
    names = [str(n) for n in P["names"]]
    X = np.asarray(P["x"], dtype=np.float64)
    if sorted(names) != sorted(pta.param_names):
        miss = sorted(set(pta.param_names) ^ set(names))
        sys.exit(f"parameter names differ from the model's: {miss}")
    idx = [names.index(n) for n in pta.param_names]
    lnl, lnp, secs = [], [], []
    for x in X:
        xx = x[idx]
        t1 = time.time()
        lnl.append(float(pta.get_lnlikelihood(xx)))
        secs.append(time.time() - t1)
        lnp.append(float(pta.get_lnprior(xx)))
    np.savez(a.out, lnlike=np.array(lnl), lnprior=np.array(lnp), seconds=np.array(secs),
             param_names=np.array(pta.param_names), same_order=np.array(names == list(pta.param_names)),
             tspan_common=np.array(model_utils.get_tspan(psrs)), build_seconds=np.array(tb),
             structure=np.array(json.dumps(structure(pta))),
             summary=np.array(pta.summary()))
    print(f"{a.orf} n={len(lnl)} build {tb:.1f}s eval {np.median(secs):.3f}s/pt same_order={names == list(pta.param_names)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--variant", default="original")
    e.add_argument("--outdir", default=None)
    l_ = sub.add_parser("like")
    l_.add_argument("--variant", default="original")
    l_.add_argument("--orf", required=True)
    l_.add_argument("--points", required=True)
    l_.add_argument("--out", required=True)
    l_.add_argument("--ncommon", type=int, default=9)
    a = ap.parse_args()
    check_env()
    {"export": cmd_export, "like": cmd_like}[a.cmd](a)


if __name__ == "__main__":
    main()
