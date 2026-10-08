"""M3a addendum: effect of the ELL1H H3+H4 harmonic count (PINT forces NHARMS >= 7, tempo2 uses 4)
on the likelihood, for the 10 legs of that class (docs/M3A_VALIDATION.md Sec. 12). The posterior-
level answer is scripts/m3a_nharms_posterior.py.

Each leg is loaded twice in fresh processes with identical inputs and profiles, NHARMS = 7 (PINT
today) and NHARMS = 4 (tempo2; set after PINT's set-up, as nanograv/PINT#2046 allows).

Conventions (review M3a #7). W = diag(1/sigma) with the raw TOA uncertainties. Projectors are the
rank-revealing complement projectors of ``multileg.complement_projector`` (thin SVD of the
column-normalised matrix, rtol 1e-10); their effective ranks are reported.
  shapiro_diff_rms_ns  rms of d = r7 - r4 (weighted mean removed): the delay difference at
                       identical parameter values;
  union_*              d projected out of span[W M7, W M4] (a diagnostic only: a small value
                       does NOT imply equal column spaces);
  own_*                e = P7 W r7 - P4 W r4: each analysis marginalises its OWN timing model;
                       this is what differs between the two likelihoods' data terms;
  own_gw{14,30}_M7/M4  norm of the component of e in span(P_M F_gw) with F_gw the first 14 / 30
                       Fourier bins of the combined span, P_M the complement projector of M7 or of
                       M4 (both reported; the value depends on the choice). A whitened norm is not
                       in general the square root of a chi^2 change (cross terms with the data);
  D(theta)             lnL7 - lnL4 on the 9 x 9 CURN grid (log10_A in [-15.5, -13.5],
                       gamma in [2, 6.5]) at fixed white noise (EFAC 1, EQUAD 1 ns) and fixed IRN
                       (log10_A = -14, gamma = 3); reported: max |D - D(theta0)| with theta0 the
                       grid point (-14.5, 4.25), and the peak-to-peak range of D. The grid
                       coordinates and both surfaces are saved (nharms_surfaces.npz).
  vs_tempo2            G3/G4 of each variant against tempo2 (oracle env; skipped if absent).

Usage: JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/m3a_nharms.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

LEGS = [("epta_dr2new", "J0751+1807"), ("epta_dr2new", "J1012+5307"), ("inpta_dr2", "J0751+1807"),
        ("inpta_dr2", "J1012+5307"), ("mpta", "J0613-0200"), ("mpta", "J1327-0755"), ("mpta", "J1545-4550"),
        ("mpta", "J1804-2717"), ("mpta", "J2145-0750"), ("ppta_dr3_gh", "J0613-0200")]
T_COMBINED_YR = 20.146212114645603  # selected-configuration array span (G8)
GRID_A = (-15.5, -13.5, 9)
GRID_G = (2.0, 6.5, 9)
THETA0 = (-14.5, 4.25)


def main():
    import jax.numpy as jnp
    import numpy as np

    from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
    from ptagwb.gp import FourierBlock
    from ptagwb.legs import load_legs
    from ptagwb.m3data import leg_files, pta_of
    from ptagwb.multileg import complement_projector, stack_legs
    from ptagwb.noise import build_general_white_noise
    from ptagwb.profiles import published_profile

    r7 = {(r.dataset, r.label): r for r in load_legs(LEGS, jobs=10, tag="nharms7", nharms=7)}
    r4 = {(r.dataset, r.label): r for r in load_legs(LEGS, jobs=10, tag="nharms4", nharms=4)}
    T = T_COMBINED_YR * 365.25 * 86400
    common = FourierBlock("gw", 14, T)
    gA, gg = np.linspace(*GRID_A), np.linspace(*GRID_G)
    grid = [(a, g) for g in gg for a in gA]  # gamma-major: surfaces reshape to (len(gg), len(gA))
    i0 = int(np.argmin([(a - THETA0[0]) ** 2 + (g - THETA0[1]) ** 2 for a, g in grid]))
    out = {"T_combined_yr": T_COMBINED_YR, "grid": {"log10_A": list(GRID_A), "gamma": list(GRID_G), "theta0": THETA0},
           "legs": {}}
    surfaces = {"grid_log10_A": gA, "grid_gamma": gg}

    def like_for(psr):
        span = float(psr.toas.max() - psr.toas.min())
        systems = sorted(set(psr.backend_flags.tolist()))
        wn = build_general_white_noise(psr.toas, psr.toaerrs, psr.backend_flags, {s: (1.0, -9.0) for s in systems})
        model = PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 30, span)}, common=common)
        return precompute_general(psr, wn, model)

    def surface(terms, orf):
        L = GeneralPTALikelihood(terms, orf=orf)
        P = len(terms)
        return np.array([float(L.logL({"rn_log10_A": jnp.full(P, -14.0), "rn_gamma": jnp.full(P, 3.0),
                                        "log10_A": jnp.asarray(a), "gamma": jnp.asarray(g)})) for a, g in grid])

    def dstats(s7, s4):
        D = s7 - s4
        return {"max_abs_D_minus_D0": float(np.max(np.abs(D - D[i0]))), "D_peak_to_peak": float(D.max() - D.min()),
                "argmax_lnL4": list(grid[int(np.argmax(s4))]), "argmax_lnL7": list(grid[int(np.argmax(s7))])}

    have_t2 = False
    try:
        import m3a_oracles as O

        have_t2 = O.have_tempo2()
    except Exception:  # noqa: BLE001
        pass

    for key in LEGS:
        a, b = r7[key], r4[key]
        name = f"{key[0]}/{key[1]}"
        if not (a.ok and b.ok):
            out["legs"][name] = {"error": a.error or b.error}
            continue
        p7, p4 = a.psr, b.psr
        assert np.array_equal(p7.toas, p4.toas)
        sig = p7.toaerrs
        W = 1.0 / sig
        d = p7.residuals - p4.residuals
        d = d - np.average(d, weights=sig**-2)
        Pu = complement_projector(np.hstack([p7.Mmat * W[:, None], p4.Mmat * W[:, None]]))
        P7, P4 = complement_projector(p7.Mmat * W[:, None]), complement_projector(p4.Mmat * W[:, None])
        u = Pu(d * W)
        e = P7(p7.residuals * W) - P4(p4.residuals * W)
        row = {"ntoa": len(d), "nharms_setup": a.meta.get("nharms_setup"), "nharms_used": [a.meta.get("nharms_used"), b.meta.get("nharms_used")],
               "ranks": {"M7": P7.rank, "M4": P4.rank, "union": Pu.rank, "union_ncols": Pu.ncols},
               "shapiro_diff_rms_ns": float(np.sqrt(np.mean(d**2)) * 1e9),
               "union_rms_ns": float(np.sqrt(np.mean((u * sig) ** 2)) * 1e9), "union_whitened_norm": float(np.linalg.norm(u)),
               "own_rms_ns": float(np.sqrt(np.mean((e * sig) ** 2)) * 1e9), "own_whitened_norm": float(np.linalg.norm(e))}
        for nb in (14, 30):
            F = FourierBlock("gw", nb, T).basis(p7.toas, p7.freqs) * W[:, None]
            for lab, P in (("M7", P7), ("M4", P4)):
                G = np.column_stack([P(F[:, j]) for j in range(F.shape[1])])
                U, sv, _ = np.linalg.svd(G, full_matrices=False)
                U = U[:, sv > 1e-10 * sv[0]]
                row[f"own_gw{nb}_{lab}_whitened_norm"] = float(np.linalg.norm(U.T @ e))
        s7, s4 = surface([like_for(p7)], "curn"), surface([like_for(p4)], "curn")
        surfaces[f"{name}_lnL7"], surfaces[f"{name}_lnL4"] = s7, s4
        row["curn_single_leg"] = dstats(s7, s4)
        if have_t2:
            par, tim = leg_files(key[0])[key[1]]
            t2 = O.run_tempo2(par, tim, published_profile(pta_of(key[0]), key[0]), f"nharms/{key[0]}/{key[1]}")
            for lab, p in (("nharms7", p7), ("nharms4", p4)):
                g = O.g3_g4(p, t2)
                row[f"vs_tempo2_{lab}"] = {k: g.get(k) for k in ("rms_diff_proj_ns", "rms_diff_proj_sigma", "g4_max_sin", "ncol_pint", "ncol_tempo2")}
        out["legs"][name] = row
        print(name, json.dumps(row), flush=True)

    by_psr: dict = {}
    for key in LEGS:
        if r7[key].ok and r4[key].ok:
            by_psr.setdefault(key[1], []).append(key)
    t7, t4, names = [], [], []
    for psr, keys in sorted(by_psr.items()):
        ptas = [r7[k].pta for k in keys]
        t7.append(like_for(stack_legs(psr, [r7[k].psr for k in keys], ptas, timing="per_leg")))
        t4.append(like_for(stack_legs(psr, [r4[k].psr for k in keys], ptas, timing="per_leg")))
        names.append(f"{psr} ({'+'.join(ptas)})")
    res = {"pulsars": names}
    for orf in ("curn", "hd"):
        s7, s4 = surface(t7, orf), surface(t4, orf)
        surfaces[f"combined_{orf}_lnL7"], surfaces[f"combined_{orf}_lnL4"] = s7, s4
        res[orf] = dstats(s7, s4)
    out["combined"] = res
    print("combined", json.dumps(res), flush=True)
    rd = ROOT / "data" / "processed" / "m3a" / "results"
    (rd / "nharms.json").write_text(json.dumps(out, indent=1) + "\n")
    np.savez(rd / "nharms_surfaces.npz", **surfaces)
    print("->", rd / "nharms.json")


if __name__ == "__main__":
    main()
