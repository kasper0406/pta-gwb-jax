"""Optimal statistic from the M2 CURN chains: fixed-noise (at the highest-likelihood saved CURN
draw), noise-marginalised, binned pair-covariance-aware correlations (Fig. 1c), and the released
references.

    uv run --no-sync python scripts/m2_optstat.py

Writes outputs/m2/optstat.json and outputs/m2/optstat_draws.npz (distributions for figures).
"""

from __future__ import annotations

import json
import sys

import jax
import numpy as np
from scipy import stats

jax.config.update("jax_enable_x64", True)

from m2_common import FIG1, RAW, ROOT, save_json

from ptagwb.data import get_tspan, load_pulsars
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.noise import load_noise_dict
from ptagwb.optstat import OptimalStatistic, binned_correlations
from ptagwb.sampling import ModelSpec, load_run, unpack

# Fig. 1(c) notebook bin edges (degrees), 15 bins
XI_BINS_DEG = np.array([0.95374012, 19.17571344, 27.97758157, 36.88462878, 44.45930411, 49.2, 61.21951091,
                        71.13671451, 81.52651267, 91.76848602, 102.58676647, 113.15847004, 125.06124956,
                        139.03110153, 152.7987445, 178.7581949])
RELEASED_ML_OS = {"A2": 6.703650273305144e-30, "sigma": 1.2324322072882867e-30, "snr": 5.439366347018102}


def params_from_x(X, run):
    spec = ModelSpec(**run["meta"]["config"]["model"])
    P = sum(1 for n in run["names"] if n.endswith("_red_noise_log10_A"))
    return unpack(np.asarray(X), spec, P)


def released_os(path):
    z = np.load(path)
    rho, sig, hd = z["rho"], z["sig"], z["hd"]
    a2 = np.sum(rho * hd / sig**2, axis=1) / np.sum(hd**2 / sig**2, axis=1)
    s = 1 / np.sqrt(np.sum(hd**2 / sig**2, axis=1))
    return a2, s, a2 / s


def main():
    psrs = load_pulsars(verbose=False)
    T = get_tspan(psrs)
    nd = load_noise_dict()
    oss = {pos: OptimalStatistic(PTALikelihood(precompute(psrs, nd, T, position=pos), T, orf="curn"))
           for pos in ("enterprise", "icrs")}
    out, draws = {}, {}
    # released noise-marginalised distributions (Fig. 4 bundle)
    rel_dir = RAW / "ng15_gwb_fig4_correlations" / "extracted"
    for tag, f in (("g433", "correlations_gamma4p33_nfreq14.npz"), ("vg", "correlations_variedgamma_nfreq14.npz")):
        a2, _s, snr = released_os(rel_dir / f)
        out[f"released_nm_{tag}"] = {"snr_mean": snr.mean(), "snr_std": snr.std(), "A2_mean": a2.mean(),
                                     "A2_std": a2.std(), "n": len(snr)}
        draws[f"released_snr_{tag}"] = snr
    # noise-marginalised over our CURN posteriors
    for tag, rn in (("g433", "curn_g433_14f"), ("vg", "curn_vg_14f")):
        try:
            run = load_run(rn)
        except FileNotFoundError:
            continue
        X = run["x"].reshape(-1, run["x"].shape[-1])
        p = params_from_x(X, run)
        for pos, os_ in oss.items():
            nm = os_.noise_marginalized(p)
            out[f"ours_nm_{tag}_{pos}"] = {"snr_mean": nm["snr"].mean(), "snr_std": nm["snr"].std(),
                                           "A2_mean": nm["A2"].mean(), "A2_std": nm["A2"].std(), "n": len(X)}
            draws[f"ours_snr_{tag}_{pos}"] = nm["snr"]
            draws[f"ours_A2_{tag}_{pos}"] = nm["A2"]
        # fixed-noise OS at the highest-likelihood saved draw (not an optimised MAP)
        i = int(np.argmax(run["logL"].ravel()))
        pbest = {k: v[i] for k, v in p.items()}
        out[f"maxlike_draw_{tag}_params"] = {n: float(X[i, j]) for j, n in enumerate(run["names"]) if n.startswith("gw_")}
        out[f"maxlike_draw_{tag}_logL"] = float(run["logL"].ravel()[i])
        for pos, os_ in oss.items():
            r = os_.os(pbest)
            out[f"ours_maxlike_draw_{tag}_{pos}"] = {k: r[k] for k in ("A2", "sigma", "snr")}
            if tag == "g433":
                C = os_.pair_covariance(pbest)
                a2n = 10 ** (2 * float(pbest["log10_A"]))
                b = binned_correlations(os_.xi, r["rho"], os_.orf_pairs, C, np.deg2rad(XI_BINS_DEG), a2_norm=a2n)
                b["p_chi2_15dof"] = float(stats.chi2(15).sf(b["chi2"]))
                out[f"ours_binned_maxlike_draw_{pos}"] = b
                if pos == "enterprise":  # sensitivity of the binned chi2 to which high-likelihood draw is used
                    top = np.argsort(run["logL"].ravel())[::-1][:20]
                    chis = []
                    for t in top:
                        pt = {kk: vv[t] for kk, vv in p.items()}
                        rt = os_.os(pt)
                        Ct = os_.pair_covariance(pt)
                        bt = binned_correlations(os_.xi, rt["rho"], os_.orf_pairs, Ct, np.deg2rad(XI_BINS_DEG),
                                                 a2_norm=10 ** (2 * float(pt["log10_A"])))
                        chis.append(bt["chi2"])
                    out["binned_chi2_top20_logL_draws"] = {"min": min(chis), "median": float(np.median(chis)),
                                                           "max": max(chis), "values": chis}
    # at the released ML vector (our PINT arrays, enterprise positions): reproduces the release?
    ml = json.loads((FIG1 / "optstat_ml_gamma4p33.json").read_text())
    for pos, os_ in oss.items():
        pml = os_.like.params_from_named({**ml, "gw_gamma": 13 / 3})
        r = os_.os(pml)
        out[f"ours_at_released_ml_{pos}"] = {k: r[k] for k in ("A2", "sigma", "snr")}
        C = os_.pair_covariance(pml)
        b = binned_correlations(os_.xi, r["rho"], os_.orf_pairs, C, np.deg2rad(XI_BINS_DEG),
                                a2_norm=10 ** (2 * ml["gw_log10_A"]))
        b["p_chi2_15dof"] = float(stats.chi2(15).sf(b["chi2"]))
        out[f"ours_binned_at_released_ml_{pos}"] = b
        if pos == "enterprise":
            Cref = np.load(FIG1 / "os_covariance_matix_between_rhos.npy")
            out["pair_cov_vs_released_max_rel"] = float(
                np.max(np.abs(C - Cref) / np.sqrt(np.outer(np.diag(Cref), np.diag(Cref)))))
            # the release's own binned values: their rho from the feathers are not shipped; use ours + their C
            b2 = binned_correlations(os_.xi, r["rho"], os_.orf_pairs, Cref, np.deg2rad(XI_BINS_DEG),
                                     a2_norm=10 ** (2 * ml["gw_log10_A"]))
            out["binned_at_released_ml_with_released_cov_chi2"] = b2["chi2"]
    out["released_ml_os_notebook"] = RELEASED_ML_OS
    save_json(out, ROOT / "outputs" / "m2" / "optstat.json")
    np.savez(ROOT / "outputs" / "m2" / "optstat_draws.npz", **draws)
    for k, v in out.items():
        if isinstance(v, dict) and "snr_mean" in v:
            print(k, round(v["snr_mean"], 3), "+-", round(v["snr_std"], 3))
        elif isinstance(v, dict) and "snr" in v:
            print(k, v)
        elif isinstance(v, dict) and "chi2" in v:
            print(k, "chi2", round(v["chi2"], 2), "p", round(v["p_chi2_15dof"], 3))


if __name__ == "__main__":
    sys.exit(main())
