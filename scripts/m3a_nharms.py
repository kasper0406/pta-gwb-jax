"""M3a addendum: effect of the ELL1H H3+H4 harmonic count (PINT forces NHARMS >= 7, tempo2 uses 4)
on GW inference, for the 10 quarantined legs (docs/M3A_VALIDATION.md Sec. 12).

Per leg, the leg is loaded twice in fresh processes with identical inputs and profiles, with
NHARMS = 7 (PINT today) and NHARMS = 4 (tempo2; set after PINT's set-up, as nanograv/PINT#2046
allows). Reported:

(a) the Shapiro-delay difference (residual difference at identical parameters, mean removed) and
    the part that survives timing marginalisation: the difference projected out of the weighted
    (1/sigma) column space of both design matrices;
(b) the part of that surviving difference that lies in the GW band: least-squares projection onto
    the timing-projected Fourier columns of the first 14 and 30 bins of the combined span;
(c) log-likelihood changes: per leg (raw TOA errors, IRN on the leg's span, CURN common process
    on the combined grid) over a 9 x 9 grid of common (log10_A, gamma) at fixed noise; and for a
    small combined system (the 7 physical pulsars, legs stacked per pulsar as option B) for CURN
    and HD.

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


def main():
    import jax.numpy as jnp
    import m3a_oracles as O
    import numpy as np

    from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
    from ptagwb.gp import FourierBlock
    from ptagwb.legs import load_legs
    from ptagwb.multileg import stack_legs
    from ptagwb.noise import build_general_white_noise

    r7 = {(r.dataset, r.label): r for r in load_legs(LEGS, jobs=10, tag="nharms7", nharms=7)}
    r4 = {(r.dataset, r.label): r for r in load_legs(LEGS, jobs=10, tag="nharms4", nharms=4)}
    T = T_COMBINED_YR * 365.25 * 86400
    common = FourierBlock("gw", 14, T)
    grid = [(a, g) for a in np.linspace(-15.5, -13.5, 9) for g in np.linspace(2.0, 6.5, 9)]
    out = {"T_combined_yr": T_COMBINED_YR, "legs": {}}

    def like_for(psr, residuals):
        span = float(psr.toas.max() - psr.toas.min())
        systems = sorted(set(psr.backend_flags.tolist()))
        wn = build_general_white_noise(psr.toas, psr.toaerrs, psr.backend_flags, {s: (1.0, -9.0) for s in systems})
        v = type("V", (), {})()
        for k in ("toas", "freqs", "toaerrs", "backend_flags", "name", "pos", "Mmat"):
            setattr(v, k, getattr(psr, k))
        v.residuals, v.pos_enterprise = residuals, None
        model = PulsarGPModel(sampled={"rn": FourierBlock("red_noise", 30, span)}, common=common)
        return precompute_general(v, wn, model)

    for key in LEGS:
        a, b = r7[key], r4[key]
        if not (a.ok and b.ok):
            out["legs"][f"{key[0]}/{key[1]}"] = {"error": a.error or b.error}
            continue
        p7, p4 = a.psr, b.psr
        assert np.array_equal(p7.toas, p4.toas)
        sig = p7.toaerrs
        d = p7.residuals - p4.residuals
        d = d - np.average(d, weights=sig**-2)
        Wd = d / sig
        proj = O.weighted_projector_complement(np.hstack([p7.Mmat / sig[:, None], p4.Mmat / sig[:, None]]))
        e = proj(Wd)
        row = {"ntoa": len(d), "nharms_setup": a.meta.get("nharms_setup"), "nharms": [7, 4],
               "shapiro_diff_rms_ns": float(np.sqrt(np.mean(d**2)) * 1e9),
               "post_fit_diff_rms_ns": float(np.sqrt(np.mean((e * sig) ** 2)) * 1e9),
               "post_fit_diff_whitened_norm": float(np.linalg.norm(e)),
               "post_fit_diff_rms_sigma": float(np.sqrt(np.mean(e**2)))}
        for nb in (14, 30):
            F = FourierBlock("gw", nb, T).basis(p7.toas, p7.freqs)
            G = np.column_stack([proj(F[:, j] / sig) for j in range(F.shape[1])])
            coef, *_ = np.linalg.lstsq(G, e, rcond=None)
            ge = G @ coef
            row[f"gw_band_{nb}bins_whitened_norm"] = float(np.linalg.norm(ge))
            row[f"gw_band_{nb}bins_fraction_of_power"] = float(ge @ ge / max(e @ e, 1e-300))
        t7, t4 = like_for(p7, p7.residuals), like_for(p4, p4.residuals)
        L7, L4 = GeneralPTALikelihood([t7], orf="curn"), GeneralPTALikelihood([t4], orf="curn")
        dl = []
        for la, g in grid:
            prm = {"rn_log10_A": jnp.asarray([-14.0]), "rn_gamma": jnp.asarray([3.0]),
                   "log10_A": jnp.asarray(la), "gamma": jnp.asarray(g)}
            dl.append(float(L7.logL(prm)) - float(L4.logL(prm)))
        dl = np.array(dl)
        row["dlnL_at_grid_center"] = float(dl[len(dl) // 2])
        row["max_dlnL_shape_over_grid"] = float(np.max(np.abs(dl - dl[len(dl) // 2])))
        out["legs"][f"{key[0]}/{key[1]}"] = row
        print(key, json.dumps(row), flush=True)

    # combined system: the physical pulsars, legs stacked per pulsar (option B), CURN and HD
    by_psr: dict = {}
    for key in LEGS:
        if r7[key].ok and r4[key].ok:
            by_psr.setdefault(key[1], []).append(key)
    terms7, terms4, names = [], [], []
    for psr, keys in sorted(by_psr.items()):
        ptas = [r7[k].pta for k in keys]
        m7 = stack_legs(psr, [r7[k].psr for k in keys], ptas, timing="per_leg")
        m4 = stack_legs(psr, [r4[k].psr for k in keys], ptas, timing="per_leg")
        terms7.append(like_for(m7, m7.residuals))
        terms4.append(like_for(m4, m4.residuals))
        names.append(f"{psr} ({'+'.join(ptas)})")
    res = {"pulsars": names}
    for orf in ("curn", "hd"):
        L7, L4 = GeneralPTALikelihood(terms7, orf=orf), GeneralPTALikelihood(terms4, orf=orf)
        P = len(terms7)
        dl = []
        for la, g in grid:
            prm = {"rn_log10_A": jnp.full(P, -14.0), "rn_gamma": jnp.full(P, 3.0),
                   "log10_A": jnp.asarray(la), "gamma": jnp.asarray(g)}
            dl.append(float(L7.logL(prm)) - float(L4.logL(prm)))
        dl = np.array(dl)
        res[orf] = {"dlnL_at_grid_center": float(dl[len(dl) // 2]),
                    "max_dlnL_shape_over_grid": float(np.max(np.abs(dl - dl[len(dl) // 2])))}
    out["combined"] = res
    print("combined", json.dumps(res), flush=True)
    p = ROOT / "data" / "processed" / "m3a" / "results" / "nharms.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("->", p)


if __name__ == "__main__":
    main()
