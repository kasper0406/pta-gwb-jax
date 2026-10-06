"""Shared helpers for the M2 analysis scripts: released NG15 chains and run loading.

Released chains are treated as single (concatenated) chains after the la_forge burn-in stored in
the core metadata; their ESS is estimated by autocorrelation, so their quantile MCSEs are
approximate (concatenation of independent runs inflates the apparent autocorrelation slightly,
i.e. the MCSE is conservative).
"""

from __future__ import annotations

import glob
import json
from functools import cache
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIG1 = RAW / "ng15_gwb_fig1_data" / "extracted" / "figure1_data"
TUT = next((RAW / "ng15_tutorial_repo" / "extracted").glob("15yr*")) / "tutorials"


def _core(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        names = [p.decode() if isinstance(p, bytes) else p for p in h["params"][:]]
        burn = int(h["metadata"]["burn"][()]) if "burn" in h["metadata"] else 0
        ch = h["chain"][burn:]
    return {n: ch[:, i] for i, n in enumerate(names)}


def _feather(model: str) -> dict:
    import pyarrow.feather as pf

    p = glob.glob(str(RAW / "discovery_repo" / "extracted" / "*" / "data" / f"NG15yr-{model}-chain.feather"))[0]
    t = pf.read_table(p)
    return {c: t[c].to_numpy() for c in t.column_names}


@cache
def released(name: str) -> dict:
    """Released chain ``name`` as {parameter: 1-D array} with our parameter names."""
    if name == "hd_g433":  # Fig. 1(b) HD^13/3 (Lamb & Pol long chains)
        return {"gw_log10_A": np.load(FIG1 / "nano15_hd_chain_fg_long_050523.npy")}
    if name == "hd_vg":  # Fig. 1(b) HD^gamma
        a = np.load(FIG1 / "nano15_hd_chain_long_050523.npy")
        return {"gw_gamma": a[:, 0], "gw_log10_A": a[:, 1]}
    if name in ("curn_vg_m2a", "hd_vg_m3a"):  # discovery repo chains (full IRN)
        d = _feather(name.split("_")[-1])
        return {k: v for k, v in d.items()}
    if name in ("curn_vg_hm", "hd_vg_hm"):  # Fig. 1 product-space core 14f_PL_hd_crn (split by nmodel)
        d = _core(FIG1 / "14f_PL_hd_crn.core")
        sel = d["nmodel"] < 0.5 if name.startswith("curn") else d["nmodel"] > 0.5
        out = {k.replace("gw__", "gw_"): v[sel] for k, v in d.items()}
        return out
    if name in ("curn_g433_tut", "hd_g433_tut"):  # tutorial product-space core curn_hd (gamma fixed)
        d = _core(TUT / "presampled_cores" / "curn_hd.core")
        if name.startswith("curn"):
            sel = d["nmodel"] < 0.5
            key = "gw_crn_log10_A"
        else:
            sel = d["nmodel"] > 0.5
            key = "gw_hd_log10_A"
        out = {k: v[sel] for k, v in d.items() if "red_noise" in k}
        out["gw_log10_A"] = d[key][sel]
        return out
    if name == "hd_fs30":  # Fig. 1(a) HD free spectrum core
        d = _core(FIG1 / "30fCP_30fiRN_3A_freespec_chain.core")
        return {k.replace("gw_hd_log10_rho", "gw_log10_rho"): v for k, v in d.items()}
    raise KeyError(name)


def run_draws(run: dict, name: str) -> np.ndarray:
    """(chains, n) draws of parameter ``name`` from ``ptagwb.sampling.load_run`` output."""
    return run["x"][..., run["names"].index(name)]


def save_json(obj, path: Path):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=conv))
