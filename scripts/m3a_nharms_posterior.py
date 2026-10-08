"""NHARMS addendum, posterior level (docs/M3A_VALIDATION.md Sec. 12b; review M3a #9).

Conditional (fixed-noise) posteriors of the common process with the affected ELL1H H3+H4 legs
evaluated with 4 harmonics (tempo2 convention) and with 7 (PINT's forced value); everything else
identical.

Arrays
  "affected": the 7 physical pulsars of the 10 legs (legs stacked per pulsar, option B timing);
  "affected+NG15": the same plus the NG15 legs of J1909-3744, J1713+0747, J1744-1134, J0030+0451
  (unaffected; identical in both variants).

Noise (fixed; absorbed in stage 1)
  white: released per-PTA values -- EPTA DR2new noisefiles (EFAC, TNEQUAD), InPTA DR2 par T2EFAC,
    MPTA example_noise.json (EFAC, TNEQUAD, ECORR), PPTA DR3 singlePsrNoise JSON (EFAC, TNEQUAD,
    band/global ECORR terms with the PPTA selections), NG15 v1p1_wn_dict;
  red / DM: released values -- EPTA J0751 (DM, TempoNest normalisation, 92 modes), J1012 (RN 92 modes,
    DM 16 modes, TN norm.) on the EPTA leg span; PPTA J0613 (RN 30, DM 100 modes, band-noise-low 30
    modes on the PPTA leg span); NG15 strong pulsars (IRN medians of the released CURN chain, 30
    modes, NG15 span). MPTA legs (J1327, J1545, J1804, J2145): no machine-readable RN/DM release,
    so IRN + DM power laws (30 modes each, MPTA span) are fitted by maximum likelihood at the
    released white noise on the 4-harmonic data (bounded L-BFGS-B within the prior box, convergence
    and the objective at the returned parameters recorded) and used in BOTH variants. This is a
    conditional approximation built from released inputs, not the released noise analyses.
    Not modelled: PPTA chromatic
    GP (log10 A = -16.7), annual DM sinusoid, solar-wind GP; MPTA chromatic terms.
Common process: power law, 14 modes on the array span; uniform priors log10_A in [-18, -11],
gamma in [0, 7]; grid 141 x 141 (CURN and HD) and a 1401-point log10_A grid at gamma = 13/3.

Outputs: data/processed/m3a/results/nharms_posterior.json (+ .npz with grids and surfaces).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from m3a_nharms import LEGS

NG15_STRONG = ["J1909-3744", "J1713+0747", "J1744-1134", "J0030+0451"]
RAW = ROOT / "data" / "raw"


def _one(pattern):
    c = sorted(RAW.glob(pattern))
    assert len(c) == 1, (pattern, c)
    return c[0]


def white_noise_for(mp, psr):
    """GeneralWhiteNoise of a stacked pulsar from the released per-PTA values."""
    import numpy as np

    from ptagwb.noise import EcorrTerm, build_general_white_noise, load_noise_dict

    efeq, conv, terms = {}, {}, []
    systems = sorted(set(mp.backend_flags.tolist()))
    pta_of = {s: s.split(":", 1)[0] for s in systems}
    raw = {s: s.split(":", 1)[1] for s in systems}
    for pta in sorted(set(pta_of.values())):
        mine = [s for s in systems if pta_of[s] == pta]
        if pta == "EPTA":
            nd = json.loads(_one(f"epta_dr2_gitlab/extracted/*/EPTA-DR2/noisefiles/DR2new/{psr}_noise.json").read_text())
            for s in mine:
                efeq[s] = (nd[f"{psr}_{raw[s]}_efac"], nd[f"{psr}_{raw[s]}_log10_tnequad"])
                conv[s] = "tn"
        elif pta == "InPTA":
            par = _one(f"inpta_dr2/extracted/InPTA.DR2-*/{psr}/{psr}.DMX.par").read_text()
            t2 = {ln.split()[2]: float(ln.split()[3]) for ln in par.splitlines() if ln.split()[:2] == ["T2EFAC", "-group"]}
            for s in mine:
                efeq[s] = (t2[raw[s]], -12.0)
                conv[s] = "t2"
        elif pta == "MPTA":
            nd = json.loads(_one("mpta_gw_scripts/extracted/*/example_noise.json").read_text())
            for s in mine:
                efeq[s] = (nd[f"{psr}_{raw[s]}_efac"], nd[f"{psr}_{raw[s]}_log10_tnequad"])
                conv[s] = "tn"
                k = f"{psr}_{raw[s]}_log10_ecorr"
                if k in nd:
                    terms.append(EcorrTerm(f"ecorr_{s}", {s: mp.backend_flags == s}, {s: nd[k]}))
        elif pta == "PPTA":
            nd = json.loads(_one(f"ppta_dr3_timing/ppta_dr3/toas_and_parameters/noisefiles/{psr}_singlePsrNoise_sw_nesw0_noise.json").read_text())
            for s in mine:
                efeq[s] = (nd[f"{psr}_{raw[s]}_efac"], nd[f"{psr}_{raw[s]}_log10_tnequad"])
                conv[s] = "tn"
            leg = np.char.startswith(mp.backend_flags.astype(str), "PPTA:")
            uwl = leg & np.array(["UWL" in s for s in mp.backend_flags])
            f = mp.freqs
            bands = {"40CM": f < 960, "20CM": (f > 960) & (f < 2048), "10CM": (f > 2048) & (f < 4032)}
            for b, m in bands.items():
                for suf, sel in (("", leg & ~uwl), ("_uwl", uwl)):
                    k = f"{psr}_basis_ecorr_{b}{suf}_log10_ecorr"
                    if k in nd and (m & sel).any():
                        terms.append(EcorrTerm(f"ppta_{b}{suf}", {f"{b}{suf}": m & sel}, {f"{b}{suf}": nd[k]}))
            for nm, sel in (("dr2", leg & ~uwl), ("uwl", uwl)):
                k = f"{psr}_basis_ecorr_all_{nm}_log10_ecorr"
                if k in nd and sel.any():
                    terms.append(EcorrTerm(f"ppta_all_{nm}", {nm: sel}, {nm: nd[k]}))
        elif pta == "NG15":
            nd = load_noise_dict()
            for s in mine:
                efeq[s] = (nd[f"{psr}_{raw[s]}_efac"], nd[f"{psr}_{raw[s]}_log10_t2equad"])
                conv[s] = "t2"
                k = f"{psr}_{raw[s]}_log10_ecorr"
                if k in nd:
                    terms.append(EcorrTerm(f"ecorr_{s}", {s: mp.backend_flags == s}, {s: nd[k]}))
    return build_general_white_noise(mp.toas, mp.toaerrs, mp.backend_flags, efeq, terms, convention=conv)


def red_dm_released(mp, psr, legs_by_pta):
    """Fixed (block, phi) list from released RN/DM values, or None (MPTA-only pulsars)."""
    from ptagwb.basis import powerlaw
    from ptagwb.gp import FourierBlock

    def span(p):
        return float(p.toas.max() - p.toas.min())

    out = []
    if "EPTA" in legs_by_pta:
        nd = json.loads(_one(f"epta_dr2_gitlab/extracted/*/EPTA-DR2/noisefiles/DR2new/{psr}_noise.json").read_text())
        dmd = json.loads(_one("epta_dr2_gitlab/extracted/*/EPTA-DR2/noisefiles/DR2new/dm_dict.json").read_text())
        rnd = json.loads(_one("epta_dr2_gitlab/extracted/*/EPTA-DR2/noisefiles/DR2new/red_dict.json").read_text())
        T = span(legs_by_pta["EPTA"])
        if f"{psr}_dm_gp_log10_A" in nd:
            b = FourierBlock("dm_gp", int(dmd.get(psr) or 30), T, chrom_idx=2.0, norm="temponest_dm")
            f, df = b.frequencies()
            out.append((b, powerlaw(f, df, nd[f"{psr}_dm_gp_log10_A"], nd[f"{psr}_dm_gp_gamma"])))
        if f"{psr}_rn_log10_A" in nd:
            b = FourierBlock("red_noise", int(rnd.get(psr) or 30), T)
            f, df = b.frequencies()
            out.append((b, powerlaw(f, df, nd[f"{psr}_rn_log10_A"], nd[f"{psr}_rn_gamma"])))
        return out
    if "PPTA" in legs_by_pta:
        nd = json.loads(_one(f"ppta_dr3_timing/ppta_dr3/toas_and_parameters/noisefiles/{psr}_singlePsrNoise_sw_nesw0_noise.json").read_text())
        T = span(legs_by_pta["PPTA"])
        for name, n, kw, key in (("red_noise", 30, {}, "red_noise"), ("dm_gp", 100, {"chrom_idx": 2.0}, "dm_gp"),
                                 ("band_low", 30, {"selection": ("freq", 0.0, 960.0)}, "band_noise_low")):
            if f"{psr}_{key}_log10_A" in nd:
                b = FourierBlock(name, n, T, **kw)
                f, df = b.frequencies()
                out.append((b, powerlaw(f, df, nd[f"{psr}_{key}_log10_A"], nd[f"{psr}_{key}_gamma"])))
        return out
    if "NG15" in legs_by_pta:
        from oracle_helpers import load_chain

        ch = load_chain("m2a")
        A, g = float(ch[f"{psr}_red_noise_log10_A"].median()), float(ch[f"{psr}_red_noise_gamma"].median())
        b = FourierBlock("red_noise", 30, 505861299.1401644)
        f, df = b.frequencies()
        return [(b, powerlaw(f, df, A, g))]
    return None


def fit_red_dm(mp, wn):
    """ML IRN + DM power laws (30 modes each, pulsar span) at fixed white noise (no common process)."""
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize

    from ptagwb.basis import powerlaw
    from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
    from ptagwb.gp import FourierBlock

    T = float(mp.toas.max() - mp.toas.min())
    rn, dm = FourierBlock("red_noise", 30, T), FourierBlock("dm_gp", 30, T, chrom_idx=2.0)
    like = GeneralPTALikelihood([precompute_general(mp, wn, PulsarGPModel(sampled={"rn": rn, "dm": dm}))],
                                orf=None, common=None)

    import jax

    def nll_and_grad(x):
        def f(v):
            p = {"rn_log10_A": v[0:1], "rn_gamma": v[1:2], "dm_log10_A": v[2:3], "dm_gamma": v[3:4]}
            return -like._logL(p)

        val, g = jax.value_and_grad(f)(jnp.asarray(x, dtype=jnp.float64))
        return float(val), np.asarray(g, dtype=np.float64)

    bounds = [(-20.0, -11.0), (0.0, 7.0), (-20.0, -11.0), (0.0, 7.0)]
    best = None
    for x0 in ([-14.5, 3.0, -13.5, 2.5], [-13.5, 1.5, -13.0, 1.0], [-16.0, 5.0, -14.0, 4.0], [-18.0, 3.0, -18.0, 3.0]):
        r = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    x = np.asarray(best.x)
    f_at_x, g_at_x = nll_and_grad(x)  # objective and gradient at the returned (bounded) parameters
    at_bound = [bool(abs(xi - lo) < 1e-6 or abs(xi - hi) < 1e-6) for xi, (lo, hi) in zip(x, bounds, strict=True)]
    proj_grad = [0.0 if ab else float(gi) for ab, gi in zip(at_bound, g_at_x, strict=True)]
    out = []
    for b, A, g in ((rn, x[0], x[1]), (dm, x[2], x[3])):
        f, df = b.frequencies()
        out.append((b, powerlaw(f, df, A, g)))
    return out, {"rn_log10_A": float(x[0]), "rn_gamma": float(x[1]), "dm_log10_A": float(x[2]), "dm_gamma": float(x[3]),
                 "nll_at_returned": f_at_x, "optimizer": "L-BFGS-B (bounded, analytic JAX gradient, 4 starts)",
                 "success": bool(best.success), "message": str(best.message), "nit": int(best.nit),
                 "at_bound": at_bound, "max_abs_free_gradient": float(np.max(np.abs(proj_grad)))}


def summarize(grid_A, grid_g, lnL):
    """Marginal posteriors (uniform priors) of log10_A and gamma: median, 5/16/84/95 % quantiles."""
    import numpy as np

    post = np.exp(lnL - lnL.max())
    out = {}
    # lnL has shape (len(grid_g), len(grid_A)): the log10_A marginal sums over gamma (axis 0), the
    # gamma marginal over log10_A (axis 1) (review M3a r2 #1: these were swapped)
    if lnL.shape != (len(grid_g), len(grid_A)):
        raise ValueError(f"lnL shape {lnL.shape} != (len(gamma), len(log10_A)) = {(len(grid_g), len(grid_A))}")
    for axis, x, name in ((0, grid_A, "log10_A"), (1, grid_g, "gamma")):
        m = post.sum(axis=axis)
        c = np.cumsum(m) / m.sum()
        q = {f"q{int(p * 100):02d}": float(np.interp(p, c, x)) for p in (0.05, 0.16, 0.5, 0.84, 0.95)}
        out[name] = q
    i, j = np.unravel_index(np.argmax(lnL), lnL.shape)
    out["max"] = {"gamma": float(grid_g[i]), "log10_A": float(grid_A[j]),
                  "interior": bool(0 < i < len(grid_g) - 1 and 0 < j < len(grid_A) - 1)}
    return out


def summarize_1d(grid_A, lnL):
    import numpy as np

    post = np.exp(lnL - lnL.max())
    c = np.cumsum(post) / post.sum()
    q = {f"q{int(p * 100):02d}": float(np.interp(p, c, grid_A)) for p in (0.05, 0.16, 0.5, 0.84, 0.95)}
    j = int(np.argmax(lnL))
    q["max"] = {"log10_A": float(grid_A[j]), "interior": bool(0 < j < len(grid_A) - 1)}
    return q


def shifts(a, b):
    """Shift of b (7 harmonics) relative to a (4) in units of a's widths."""
    out = {}
    for k in ("log10_A", "gamma"):
        if k not in a:
            continue
        x, y = a[k], b[k]
        sig = 0.5 * (x["q84"] - x["q16"])
        w90 = x["q95"] - x["q05"]
        out[k] = {"dmedian": y["q50"] - x["q50"], "dmedian_over_sigma68": (y["q50"] - x["q50"]) / sig,
                  "dq05_over_w90": (y["q05"] - x["q05"]) / w90, "dq95_over_w90": (y["q95"] - x["q95"]) / w90}
    return out


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
    from ptagwb.data import Pulsar
    from ptagwb.gp import FourierBlock
    from ptagwb.legs import leg_cache_path, load_legs
    from ptagwb.multileg import stack_legs

    r = {n: {(x.dataset, x.label): x for x in load_legs(LEGS, jobs=10, tag=f"nharms{n}", nharms=n)} for n in (4, 7)}
    for n in (4, 7):
        bad = [k for k, v in r[n].items() if not v.ok]
        assert not bad, bad
    ng = {p: Pulsar.load(leg_cache_path("published", "ng15", p)) for p in NG15_STRONG}
    by_psr = {}
    for ds, lab in LEGS:
        by_psr.setdefault(lab, []).append((ds, lab))
    stacked = {n: {} for n in (4, 7)}
    for psr, keys in sorted(by_psr.items()):
        for n in (4, 7):
            ptas = [r[n][k].pta for k in keys]
            stacked[n][psr] = stack_legs(psr, [r[n][k].psr for k in keys], ptas, timing="per_leg")
    for psr, p in ng.items():
        for n in (4, 7):
            stacked[n][psr] = stack_legs(psr, [p], ["NG15"], timing="per_leg")
    allp = list(stacked[4].values())
    Tarr = float(max(p.toas.max() for p in allp) - min(p.toas.min() for p in allp))
    common = FourierBlock("gw", 14, Tarr)

    noise_info, terms = {}, {4: {}, 7: {}}
    for psr in stacked[4]:
        mp4 = stacked[4][psr]
        legs_by_pta = {lg.pta: Pulsar.load(leg_cache_path("nharms4" if psr in by_psr else "published", lg.dataset, lg.label))
                       for lg in mp4.legs}
        wn = white_noise_for(mp4, psr)
        fixed = red_dm_released(mp4, psr, legs_by_pta)
        if fixed is None:
            fixed, fit = fit_red_dm(mp4, wn)
            noise_info[psr] = {"red_dm": "bounded ML fit at released WN (4-harmonic data), used in both variants", **fit}
        else:
            noise_info[psr] = {"red_dm": "released", "blocks": [(b.name, b.n_modes, round(b.T / 3.15576e7, 2)) for b, _ in fixed]}
        model = PulsarGPModel(common=common, fixed=fixed)
        for n in (4, 7):
            terms[n][psr] = precompute_general(stacked[n][psr], wn, model)
        print(psr, noise_info[psr], flush=True)

    gA = np.linspace(-18.0, -11.0, 141)
    gg = np.linspace(0.0, 7.0, 141)
    AA, GG = np.meshgrid(gA, gg)
    g1 = np.linspace(-18.0, -11.0, 1401)
    arrays = {"affected": sorted(by_psr), "affected+NG15": sorted(by_psr) + NG15_STRONG}
    results = {"noise": noise_info, "Tarr_yr": Tarr / 3.15576e7, "grid": {"log10_A": [-18, -11, 141], "gamma": [0, 7, 141]},
               "arrays": {}}
    surfaces = {"grid_log10_A": gA, "grid_gamma": gg, "grid_log10_A_1d": g1}
    for aname, names in arrays.items():
        results["arrays"][aname] = {"pulsars": names}
        for orf in ("curn", "hd"):
            lls = {}
            for n in (4, 7):
                like = GeneralPTALikelihood([terms[n][p] for p in names], orf=orf, common="powerlaw")
                f = jax.jit(jax.vmap(lambda a, g, like=like: like._logL({"log10_A": a, "gamma": g})))
                flat = np.concatenate([np.asarray(f(jnp.asarray(a), jnp.asarray(g)))
                                       for a, g in zip(np.array_split(AA.ravel(), 40), np.array_split(GG.ravel(), 40),
                                                       strict=True)])
                lls[n] = flat.reshape(AA.shape)
                one = np.asarray(f(jnp.asarray(g1), jnp.full_like(jnp.asarray(g1), 13.0 / 3.0)))
                lls[f"{n}_1d"] = one
                surfaces[f"{aname}_{orf}_lnL{n}"] = lls[n]
                surfaces[f"{aname}_{orf}_lnL{n}_1d"] = one
            s4, s7 = summarize(gA, gg, lls[4]), summarize(gA, gg, lls[7])
            o4, o7 = summarize_1d(g1, lls["4_1d"]), summarize_1d(g1, lls["7_1d"])
            D = lls[7] - lls[4]
            results["arrays"][aname][orf] = {
                "nharms4": s4, "nharms7": s7, "shift_7_vs_4": shifts(s4, s7),
                "gamma13_3": {"nharms4": o4, "nharms7": o7, "shift_7_vs_4": shifts({"log10_A": o4}, {"log10_A": o7})},
                "D_max_abs_minus_D_at_map4": float(np.max(np.abs(D - D[np.unravel_index(np.argmax(lls[4]), D.shape)]))),
                "D_peak_to_peak": float(D.max() - D.min())}
            print(aname, orf, json.dumps(results["arrays"][aname][orf]["shift_7_vs_4"]),
                  results["arrays"][aname][orf]["nharms4"]["max"], flush=True)
    out = ROOT / "data" / "processed" / "m3a" / "results"
    (out / "nharms_posterior.json").write_text(json.dumps(results, indent=1) + "\n")
    np.savez(out / "nharms_posterior_surfaces.npz", **surfaces)
    print("->", out / "nharms_posterior.json")


if __name__ == "__main__":
    main()
