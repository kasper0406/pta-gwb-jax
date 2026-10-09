"""Gate T2 (reported, not gating): timing-engine sensitivity of the EPTA DR2new common-process
posterior (docs/M3B_PLAN.md Sec. 4.1). CPU only, no sampling.

Conditional posteriors of the common process (log10_A, gamma), CURN and HD, on a grid, with every
other parameter fixed at the released CURN chain's posterior medians (noise and dip; "fixed
released noise"), evaluated with
* ``tempo2``: our tempo2 export (the D1 evaluator, runtime epta-dr2-chain-runtime-v1);
* ``pint``: the M3a PINT legs (PINT 1.1.7, profile epta-dr2-published-v1) for every pulsar;
* ``pint:<psr>``: tempo2 everywhere except the named E8 leg evaluated by PINT, one at a time
  (EPTA's E8-named legs: J1600-3053 and the DDH class J1022+1001, J1640+2224, J1918-0642).
Both engines use the same TOAs: the M3a PINT legs exclude 65 clock-uncovered and 2 duplicate TOAs
(M3a defaults), so every variant uses the common subset (matched per backend and site frequency
to the nearest site arrival time, |dt| < 1 ms). ORF positions are our export's for every variant
(the engine comparison is about the timing arrays). Grid: 61 x 61 over the tempo2 baseline's
mean +- 6 sd box (found on a 25 x 25 coarse grid). Exact likelihood (``ptagwb.fastcond``).

Reported: marginal medians, sigma68 = (q84 - q16)/2 and w90 = q95 - q05 of log10_A and gamma per
variant, and each variant's shifts relative to ``tempo2`` in sigma68 (medians) and w90 (q05, q95)
units. Plan rule: a shift <= 0.1 sigma68 means the engine choice is immaterial; larger shifts are
reported as an evaluator dependence; the tempo2 result stays the reproduction.

Usage: OMP_NUM_THREADS=1 PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_t2.py
Writes data/processed/m3b/epta/results/t2.json.
"""

from __future__ import annotations

import json

import numpy as np

from ptagwb import epta
from ptagwb.binding import evidence_binding
from ptagwb.config import REPO_ROOT
from ptagwb.data import Pulsar
from ptagwb.fastcond import FastEPTA

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
PINT_LEGS = REPO_ROOT / "data" / "processed" / "m3a" / "legs" / "published" / "epta_dr2new"
E8_LEGS = ("J1600-3053", "J1022+1001", "J1640+2224", "J1918-0642")
PROBS = (0.05, 0.16, 0.5, 0.84, 0.95)


def subset(p: Pulsar, keep: np.ndarray) -> Pulsar:
    return Pulsar(name=p.name, toas=p.toas[keep], stoas=p.stoas[keep], residuals=p.residuals[keep],
                  toaerrs=p.toaerrs[keep], freqs=p.freqs[keep], freqs_topo=p.freqs_topo[keep],
                  backend_flags=p.backend_flags[keep], telescope=p.telescope[keep], Mmat=p.Mmat[keep],
                  fitpars=p.fitpars, pos=p.pos, pos_enterprise=p.pos_enterprise,
                  flags={k: np.asarray(v)[keep] for k, v in p.flags.items()}, meta=p.meta)


def match(t2: Pulsar, pi: Pulsar) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i_t2, i_pint) of the common TOAs: same backend and site frequency, nearest site
    arrival time within 1 ms; one-to-one."""
    idx = {}
    for j, (b, f) in enumerate(zip(pi.backend_flags, np.round(pi.freqs_topo, 6))):
        idx.setdefault((str(b), float(f)), []).append(j)
    it, ip, used = [], [], set()
    for i, (b, f, s) in enumerate(zip(t2.backend_flags, np.round(t2.freqs_topo, 6), t2.stoas)):
        cand = [j for j in idx.get((str(b), float(f)), []) if j not in used]
        if not cand:
            continue
        j = min(cand, key=lambda k: abs(pi.stoas[k] - s))
        if abs(pi.stoas[j] - s) < 1e-3:
            it.append(i)
            ip.append(j)
            used.add(j)
    return np.array(it), np.array(ip)


def marginals(G, a_grid, g_grid):
    w = np.exp(G - G.max())
    out = {}
    for name, grid, m in (("log10_A", a_grid, w.sum(1)), ("gamma", g_grid, w.sum(0))):
        c = np.cumsum(m)
        c = c / c[-1]
        q = {str(p): float(np.interp(p, c, grid)) for p in PROBS}
        out[name] = {"q": q, "sigma68": (q["0.84"] - q["0.16"]) / 2, "w90": q["0.95"] - q["0.05"],
                     "mean": float(np.sum(m * grid) / m.sum()), "sd": float(np.sqrt(np.sum(m * grid**2) / m.sum() - (np.sum(m * grid) / m.sum()) ** 2))}
    return out


def grid_logL(F, x0, ia, ig, a_grid, g_grid, orf):
    G = np.empty((a_grid.size, g_grid.size))
    y = x0.copy()
    if orf == "crn":
        dc = F.dip_terms(x0)
        for i, a in enumerate(a_grid):
            for j, g in enumerate(g_grid):
                y[ia], y[ig] = a, g
                pc = F.phi_common(y)
                G[i, j] = sum(F.curn_term(k, y, pc, dc if k == F.dip_a else None) for k in range(F.P))
    else:
        parts = [F.hd_reduce(k, x0) for k in range(F.P)]
        for i, a in enumerate(a_grid):
            for j, g in enumerate(g_grid):
                y[ia], y[ig] = a, g
                G[i, j] = F.hd_joint(parts, y)
    return G


def main():
    man = epta.load_manifest()
    ours = epta.load_pulsars()
    names_c, X, burn = epta.load_reference("crn_pl", man)
    med = np.median(X[burn:, :67], axis=0)
    pint, sub_t2, n_drop = {}, {}, {}
    for p in ours:
        q = Pulsar.load(PINT_LEGS / p.name / "leg.npz")
        it, ip = match(p, q)
        n_drop[p.name] = {"tempo2": int(len(p.toas) - len(it)), "pint": int(len(q.toas) - len(ip))}
        sub_t2[p.name] = subset(p, it)
        pq = subset(q, ip)
        pq.pos = p.pos  # same ORF positions for every variant
        pint[p.name] = pq
    variants = {"tempo2": [sub_t2[p.name] for p in ours], "pint": [pint[p.name] for p in ours]}
    for leg in E8_LEGS:
        variants[f"pint:{leg}"] = [pint[p.name] if p.name == leg else sub_t2[p.name] for p in ours]
    out = {"label": "T2, reported (not gating)", "fixed_noise": "released CURN chain posterior medians",
           "dropped_toas": n_drop, "n_common_toas": int(sum(len(v.toas) for v in variants["tempo2"])), "models": {}}
    for orf in ("crn", "hd"):
        x0 = med.copy()
        names = [n if orf == "crn" else n.replace("gw_crn", "gw_hd") for n in names_c]
        ia, ig = names.index(f"gw_{orf}_log10_A"), names.index(f"gw_{orf}_gamma")
        res = {}
        base_F = FastEPTA(epta.EPTAModel(variants["tempo2"], man, orf))
        ca, cg = np.linspace(-15.5, -13.0, 25), np.linspace(0.25, 6.75, 25)
        mc = marginals(grid_logL(base_F, x0, ia, ig, ca, cg, orf), ca, cg)
        a_grid = np.linspace(mc["log10_A"]["mean"] - 6 * mc["log10_A"]["sd"], mc["log10_A"]["mean"] + 6 * mc["log10_A"]["sd"], 61)
        g_grid = np.linspace(max(0.0, mc["gamma"]["mean"] - 6 * mc["gamma"]["sd"]), min(7.0, mc["gamma"]["mean"] + 6 * mc["gamma"]["sd"]), 61)
        for v, psrs in variants.items():
            F = base_F if v == "tempo2" else FastEPTA(epta.EPTAModel(psrs, man, orf))
            res[v] = marginals(grid_logL(F, x0, ia, ig, a_grid, g_grid, orf), a_grid, g_grid)
        b = res["tempo2"]
        for v, r in res.items():
            r["shift"] = {k: {"median_in_sigma68": (r[k]["q"]["0.5"] - b[k]["q"]["0.5"]) / b[k]["sigma68"],
                              "q05_in_w90": (r[k]["q"]["0.05"] - b[k]["q"]["0.05"]) / b[k]["w90"],
                              "q95_in_w90": (r[k]["q"]["0.95"] - b[k]["q"]["0.95"]) / b[k]["w90"]} for k in ("log10_A", "gamma")}
            r["immaterial"] = bool(all(abs(r["shift"][k]["median_in_sigma68"]) <= 0.1 for k in ("log10_A", "gamma")))
            print(orf, v, {k: (round(r[k]["q"]["0.5"], 4), round(r["shift"][k]["median_in_sigma68"], 3)) for k in ("log10_A", "gamma")},
                  "immaterial" if r["immaterial"] else "EVALUATOR DEPENDENCE", flush=True)
        out["models"][orf] = {"grid": {"log10_A": [a_grid[0], a_grid[-1], a_grid.size], "gamma": [g_grid[0], g_grid[-1], g_grid.size]},
                              "variants": res}
    out["completed"] = True
    out["binding"] = evidence_binding()
    (RES / "t2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
