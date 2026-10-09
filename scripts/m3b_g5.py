"""G5-PTA for EPTA DR2new (docs/M3B_PLAN.md Sec. 4.4, 6.2): our likelihood against the
independent long-double arbiter (``tests/m3b_arbiter.py``) and against the pinned fork
enterprise, for CURN and HD, including the cross-model difference lnL_HD - lnL_CURN, for both the
validated padded configuration and the bucketed production configuration (N8 buckets +
structured Householder).

Points (all retained; revision after the review of 2ee1bb7, item 1):
* ``chain``: 12 random retained draws of the released CURN chain (seed 20261009);
* ``prior``: 4 uniform draws from the prior box (same seed; the first run's stress points);
* ``corner``: 4 fixed shelf/corner cases: (a) every amplitude 0.05 dex above its lower bound
  (shelf), every gamma 0.05 (dip at the first chain point); (b) the same with every gamma 6.95;
  (c) every amplitude 0.05 dex below its upper bound, every gamma 0.05 (dip at the chain point);
  (d) the first chain point with only the common process moved to the shelf corner (-17.95, 6.95);
* ``onset``: 4 dip-onset-adjacent cases: the first chain point with t0 = 1e-5 d before / after
  each of the two J1713+0747 TOAs bounding the reference's t0 interval.

Mixed criterion (frozen in this commit before the rerun; recommended by the review):
* shape error e_i = (lnL_ours - lnL_arb)_i - c, c = mean over the ``chain`` points; CURN and HD;
* absolute vs enterprise d_i = lnL_ours - lnL_enterprise (common constant convention);
* tolerance for e_i: 1e-6 nats at ``chain`` and ``onset`` points, max(1e-6, 1e-12 S_i) at
  ``prior`` and ``corner`` points; for d_i: 1e-4 and max(1e-4, 1e-12 S_i); S_i = |lnL_arb|, the
  likelihood magnitude with every parameter-independent constant removed (the arbiter omits the
  white-noise and timing log-determinants);
* gradient: CURN analytic, every coordinate except t0, |g_ours - g_arb| <= 1e-8 max(1, |g_arb|)
  at every point; HD by long-double 4-point differences of the arbiter on 8 coordinates at 3
  chain points and the 2 onset points inside the window interval, same tolerance;
* cross-model (lnL_HD - lnL_CURN): vs arbiter <= 1e-6 nats, vs enterprise <= 1e-4, every point.

History: the first run (2ee1bb7) gated on all 16 random points and failed the absolute
tolerances at one prior point (dip log10_Amp = -2.23, a 6-ms dip, lnL = -2.96e9): shape error
1.4e-4 nats = 4.8e-14 relative, about 300 ulp of 2.96e9 (ulp 4.8e-7); the gate was then
restricted post hoc to the chain points. That restriction is withdrawn here.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_g5.py
Writes data/processed/m3b/epta/results/g5_pta.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ptagwb import epta
from ptagwb.binding import evidence_binding
from ptagwb.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import m3b_arbiter as A  # noqa: E402
from m3b_fingerprint import enterprise_lnl  # noqa: E402

BASE = REPO_ROOT / "data" / "processed" / "m3b" / "epta"
TOL = {"shape": 1e-6, "abs_enterprise": 1e-4, "grad": 1e-8, "cross_arbiter": 1e-6, "cross_enterprise": 1e-4,
       "rel_stress": 1e-12}
STRESS_KINDS = ("prior", "corner")
SEED = 20261009
ONSET_EPS_D = 1e-5


def points(man, psrs):
    names, X, burn = epta.load_reference("crn_pl", man)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(np.arange(burn, len(X)), 12, replace=False))
    q = [p for p in man["parameters"] if "crn" in p["models"]]
    lo = np.array([p["bounds"][0] for p in q])
    hi = np.array([p["bounds"][1] for p in q])
    Xp = lo + (hi - lo) * rng.random((4, len(lo)))
    base = X[rows[0], :67].copy()
    amp = np.array([p["role"].endswith("log10_A") or p["role"] == "dip_log10_Amp" for p in q])
    gam = np.array([p["role"].endswith("gamma") for p in q])
    dip = np.array([p["role"].startswith("dip_") for p in q])
    corners = []
    for a_val, g_val in (("lo", 0.05), ("lo", 6.95), ("hi", 0.05)):
        x = base.copy()
        sel = amp & ~dip
        x[sel] = (lo[sel] + 0.05) if a_val == "lo" else (hi[sel] - 0.05)
        x[gam] = g_val
        corners.append(x)
    x = base.copy()
    ia, ig = names.index("gw_crn_log10_A"), names.index("gw_crn_gamma")
    x[ia], x[ig] = -17.95, 6.95
    corners.append(x)
    i0 = names.index(f"{man['dip']['param_prefix']}_t0")
    dip_psr = psrs[[p.name for p in psrs].index(man["dip"]["pulsar"])]
    t = dip_psr.toas / 86400.0
    a_toa = float(t[t <= 57507.2].max())   # the TOAs bounding the reference's t0 interval
    b_toa = float(t[t >= 57514.0].min())   # [57507.113, 57514.106) (configs/m3b/relevance/epta.json)
    onsets = []
    for t0 in (a_toa - ONSET_EPS_D, a_toa + ONSET_EPS_D, b_toa - ONSET_EPS_D, b_toa + ONSET_EPS_D):
        x = base.copy()
        x[i0] = t0
        onsets.append(x)
    P = np.vstack([X[rows, :67], Xp, np.stack(corners), np.stack(onsets)])
    kinds = ["chain"] * 12 + ["prior"] * 4 + ["corner"] * 4 + ["onset"] * 4
    return names, P, kinds, rows, (a_toa, b_toa)


def fd4(f, x, i, h=1e-3):
    e = np.zeros_like(x)
    e[i] = h
    return (-f(x + 2 * e) + 8 * f(x + e) - 8 * f(x - e) + f(x - 2 * e)) / (12 * h)


def evaluate_config(label, Mc, Mh, X, kinds, arb, ent, names_h, man, psrs):
    oc, oh, gc, gh = [], [], [], []
    for x in X:
        lc, g1 = Mc.value_and_grad(x)
        lh, g2 = Mh.value_and_grad(x)
        oc.append(float(lc))
        oh.append(float(lh))
        gc.append(np.asarray(g1))
        gh.append(np.asarray(g2))
    oc, oh = np.array(oc), np.array(oh)
    ac, ah, ga = arb["crn"], arb["hd"], arb["grad_crn"]
    ec, eh = ent
    k = np.array(kinds)
    chain = k == "chain"
    stress = np.isin(k, STRESS_KINDS)
    rows = []
    fails = []
    cc, ch = np.mean((oc - ac)[chain]), np.mean((oh - ah)[chain])
    i0 = Mc.t0_index
    m = np.arange(X.shape[1]) != i0
    for i in range(len(X)):
        S_c, S_h = abs(ac[i]), abs(ah[i])
        tol_s = (max(TOL["shape"], TOL["rel_stress"] * S_c), max(TOL["shape"], TOL["rel_stress"] * S_h)) if stress[i] else (TOL["shape"],) * 2
        tol_a = (max(TOL["abs_enterprise"], TOL["rel_stress"] * S_c), max(TOL["abs_enterprise"], TOL["rel_stress"] * S_h)) if stress[i] else (TOL["abs_enterprise"],) * 2
        e = (oc[i] - ac[i] - cc, oh[i] - ah[i] - ch)
        d = (oc[i] - ec[i], oh[i] - eh[i])
        cross_a = (oh[i] - oc[i]) - (ah[i] - ac[i])
        cross_e = (oh[i] - oc[i]) - (eh[i] - ec[i])
        g_rel = float(np.max(np.abs(gc[i][m] - ga[i][m]) / np.maximum(1.0, np.abs(ga[i][m]))))
        r = {"i": i, "kind": kinds[i], "S_crn": S_c, "S_hd": S_h, "shape_err": e, "shape_tol": tol_s, "abs_ent": d,
             "abs_tol": tol_a, "cross_vs_arbiter": cross_a, "cross_vs_enterprise": cross_e, "grad_crn_rel": g_rel,
             "lnl_crn": oc[i], "lnl_hd": oh[i]}
        checks = {"shape": abs(e[0]) <= tol_s[0] and abs(e[1]) <= tol_s[1],
                  "abs_enterprise": abs(d[0]) <= tol_a[0] and abs(d[1]) <= tol_a[1],
                  "cross_arbiter": abs(cross_a) <= TOL["cross_arbiter"],
                  "cross_enterprise": abs(cross_e) <= TOL["cross_enterprise"],
                  "grad_crn": g_rel <= TOL["grad"]}
        r["checks"] = checks
        for c, ok in checks.items():
            if not ok:
                fails.append(f"{label} point {i} ({kinds[i]}): {c}")
        rows.append(r)
    # HD gradient by long-double differences of the arbiter
    sel = [names_h.index(n) for n in ("gw_hd_log10_A", "gw_hd_gamma", f"{man['dip']['param_prefix']}_log10_Amp",
                                       f"{man['dip']['param_prefix']}_log10_tau", "J0900-3144_red_noise_log10_A",
                                       "J1909-3744_dm_gp_gamma", "J1600-3053_chrom_gp_log10_A", "J1713+0747_dm_gp_log10_A")]
    hd_fd = []
    for i in arb["hd_fd_points"]:
        for j in sel:
            g_fd = arb["hd_fd"][(i, j)]
            rel = abs(gh[i][j] - g_fd) / max(1.0, abs(g_fd))
            hd_fd.append({"point": int(i), "kind": kinds[i], "param": names_h[j], "ours": float(gh[i][j]),
                          "arbiter_fd": float(g_fd), "rel": float(rel)})
            if rel > TOL["grad"]:
                fails.append(f"{label} point {i}: HD gradient {names_h[j]} rel {rel:.2e}")
    summ = {
        "shape_max_chain": float(max(max(abs(r["shape_err"][0]), abs(r["shape_err"][1])) for r in rows if r["kind"] == "chain")),
        "shape_max_onset": float(max(max(abs(r["shape_err"][0]), abs(r["shape_err"][1])) for r in rows if r["kind"] == "onset")),
        "shape_max_rel_stress": float(max(max(abs(r["shape_err"][0]) / r["S_crn"], abs(r["shape_err"][1]) / r["S_hd"])
                                          for r in rows if r["kind"] in STRESS_KINDS)),
        "abs_enterprise_max_posterior": float(max(max(abs(r["abs_ent"][0]), abs(r["abs_ent"][1])) for r in rows
                                                  if r["kind"] in ("chain", "onset"))),
        "abs_enterprise_max_rel_stress": float(max(max(abs(r["abs_ent"][0]) / r["S_crn"], abs(r["abs_ent"][1]) / r["S_hd"])
                                                   for r in rows if r["kind"] in STRESS_KINDS)),
        "cross_vs_arbiter_max": float(max(abs(r["cross_vs_arbiter"]) for r in rows)),
        "cross_vs_enterprise_max": float(max(abs(r["cross_vs_enterprise"]) for r in rows)),
        "grad_crn_max_rel": float(max(r["grad_crn_rel"] for r in rows)),
        "grad_hd_max_rel": float(max(e["rel"] for e in hd_fd)),
    }
    return {"rows": rows, "hd_fd": hd_fd, "summary": summ, "fails": fails, "pass": not fails}


def run() -> dict:
    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    names_c, X, kinds, rows, bounding = points(man, psrs)
    names_h = [n.replace("gw_crn", "gw_hd") for n in names_c]
    arbs = A.build(psrs, man)
    G = A.hd_gamma_ld(np.stack([p.pos for p in psrs]))
    arb = {"crn": [], "hd": [], "grad_crn": []}
    for x in X:
        la, ga = A.curn_logL_grad(arbs, man, x, names_c)
        arb["crn"].append(la)
        arb["grad_crn"].append(ga)
        arb["hd"].append(A.hd_logL(arbs, man, x, names_h, G))
    arb["crn"], arb["hd"] = np.array(arb["crn"]), np.array(arb["hd"])
    # HD FD points: 3 chain points + the two onset points inside the reference interval
    fd_points = [0, 5, 11] + [i for i, k in enumerate(kinds) if k == "onset"][1:3]
    sel_names = ("gw_hd_log10_A", "gw_hd_gamma", f"{man['dip']['param_prefix']}_log10_Amp",
                 f"{man['dip']['param_prefix']}_log10_tau", "J0900-3144_red_noise_log10_A", "J1909-3744_dm_gp_gamma",
                 "J1600-3053_chrom_gp_log10_A", "J1713+0747_dm_gp_log10_A")
    f = lambda y: A.hd_logL(arbs, man, y, names_h, G)  # noqa: E731
    arb["hd_fd"] = {(i, names_h.index(n)): fd4(f, X[i], names_h.index(n)) for i in fd_points for n in sel_names}
    arb["hd_fd_points"] = fd_points
    ent = (enterprise_lnl("crn", X, names_c), enterprise_lnl("hd", X, names_h))
    configs = {
        "padded_m1": (epta.EPTAModel(psrs, man, "crn"), epta.EPTAModel(psrs, man, "hd")),
        "bucketed_hh_production": (epta.EPTAModel(psrs, man, "crn", reduce="hh", buckets=epta.BUCKETS),
                                   epta.EPTAModel(psrs, man, "hd", reduce="hh", buckets=epta.BUCKETS)),
    }
    out = {"n_points": len(X), "kinds": kinds, "chain_rows": rows.tolist(), "onset_bounding_toas_mjd": bounding,
           "tolerances": TOL, "stress_kinds": STRESS_KINDS, "configs": {}}
    for label, (Mc, Mh) in configs.items():
        out["configs"][label] = evaluate_config(label, Mc, Mh, X, kinds, arb, ent, names_h, man, psrs)
    out["G5_PTA_pass"] = all(c["pass"] for c in out["configs"].values())
    out["binding"] = evidence_binding()
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "g5_pta.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    out = run()
    for label, c in out["configs"].items():
        print(label, json.dumps(c["summary"]))
        for f in c["fails"]:
            print("   FAIL", f)
    print("G5-PTA PASS" if out["G5_PTA_pass"] else "G5-PTA FAIL")


if __name__ == "__main__":
    main()
