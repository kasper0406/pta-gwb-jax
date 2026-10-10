"""Rao-Blackwellised conditional-occupancy diagnostics on the released EPTA DR2new chains
(docs/M3B_PLAN.md Sec. 5.3; **supporting diagnostic only**: it never classifies support, never
resolves an UNRESOLVED occupancy and never enters a gate). CPU only, no sampling.

At 2,000 retained reference draws x_i (evenly spaced, chain order kept), for both released chains
(CURN and HD):
* every shelf-prone amplitude (each noise log10_A with its gamma, the common (log10_A, gamma), the
  dip log10_Amp with log10_tau): pi_i = P(log10_A in S | all other parameters at x_i),
  S = [lower, lower + 1 dex], by 2-D midpoint quadrature over the parameter pair's prior box
  (32 x 14 cells, the S boundary on a cell edge); the likelihood is the exact model
  (``ptagwb.fastcond``, equal to ``EPTAModel.logL``), only the terms that change are recomputed;
* the dip epoch: pi_i = P(t0 not in [a, b) | others), a, b the reference interval of the frozen
  common domain, by the trapezoid rule on 801 window points plus both sides of every TOA epoch
  inside the window (the likelihood is discontinuous at TOAs and smooth between them).

Reported per (model, region): mean(pi), its batch-means MCSE (OBM over the ordered draws, batch
length >= 5 tau), ESS = n / tau, the largest single-draw share max(pi)/sum(pi), split-half means,
the reference's own visit count and case, and whether the region is in the frozen U. A grid-
resolution check (64 x 28 cells, 1,601 t0 points) is run on 20 draws per model.

These estimate occupancy only over the conditioning states the chains visited (the plan's
analytic counterexample: a basin the chains never visit is invisible to them).

Usage: OMP_NUM_THREADS=1 PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_conditional_occupancy.py [--draws 2000] [--workers 28]
Writes data/processed/m3b/epta/results/conditional_occupancy.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ptagwb.config import REPO_ROOT

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
ACC = REPO_ROOT / "configs" / "m3b" / "acceptance_epta.json"
NA, NB, NT = 32, 14, 801

_W = {}


def _init(orf):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from ptagwb import epta
    from ptagwb.fastcond import FastEPTA

    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    M = epta.EPTAModel(psrs, man, orf)
    F = FastEPTA(M)
    acc = json.loads(ACC.read_text())
    t0ex = [e for e in acc["d9"]["exclusions"] if e["type"] == "outside"][0]
    dip_toas = np.asarray(F.dip["toas"]) / 86400.0
    q = {p["name"]: p for p in man["parameters"]}
    pairs = []
    for n in M.param_names:
        if not q[n]["shelf_prone"]:
            continue
        if n.endswith("_log10_Amp"):
            partner = n[: -len("_log10_Amp")] + "_log10_tau"
        else:
            partner = n[: -len("_log10_A")] + "_gamma"
        pairs.append((n, partner))
    _W.update(F=F, M=M, orf=orf, pairs=pairs, q=q, t0ex=t0ex, dip_toas=dip_toas, man=man)


def _cells(lo, hi, n):
    e = np.linspace(lo, hi, n + 1)
    return 0.5 * (e[1:] + e[:-1])


def _owner(name):
    F = _W["F"]
    for a, psr in enumerate(F.names):
        if name.startswith(psr + "_"):
            return a
    return None  # common


def _pair_pi(x, n, partner, na, nb):
    F, orf, q = _W["F"], _W["orf"], _W["q"]
    ia, ib = F.ix[n], F.ix[partner]
    lo, hi = q[n]["bounds"]
    blo, bhi = q[partner]["bounds"]
    ga, gb = _cells(lo, hi, na), _cells(blo, bhi, nb)
    a = _owner(n)
    is_dip = n.endswith("_log10_Amp")
    L = np.empty((na, nb))
    y = x.copy()
    dc0 = F.dip_terms(x)
    if orf == "crn":
        phic = F.phi_common(x)
        for i, va in enumerate(ga):
            for j, vb in enumerate(gb):
                y[ia], y[ib] = va, vb
                if a is None:
                    pc = F.phi_common(y)
                    L[i, j] = sum(F.curn_term(k, y, pc, dc0 if k == F.dip_a else None) for k in range(F.P))
                else:
                    L[i, j] = F.curn_term(a, y, phic, (F.dip_terms(y) if is_dip else dc0) if a == F.dip_a else None)
    else:
        parts = [F.hd_reduce(k, x, dc0 if k == F.dip_a else None) for k in range(F.P)]
        if a is None:
            for i, va in enumerate(ga):
                for j, vb in enumerate(gb):
                    y[ia], y[ib] = va, vb
                    L[i, j] = F.hd_joint(parts, y)
        else:
            cache = F.hd_prepare(parts, x)
            for i, va in enumerate(ga):
                for j, vb in enumerate(gb):
                    y[ia], y[ib] = va, vb
                    dc = (F.dip_terms(y) if is_dip else dc0) if a == F.dip_a else None
                    L[i, j] = F.hd_swap(cache, a, F.hd_reduce(a, y, dc))
    w = np.exp(L - L.max())
    in_s = ga < lo + 1.0
    return float(w[in_s].sum() / w.sum())


def _t0_pi(x, nt):
    F, orf, ex = _W["F"], _W["orf"], _W["t0ex"]
    i0 = F._dipix[2]
    lo, hi = _W["q"][_W["M"].param_names[i0]]["bounds"]
    toas = _W["dip_toas"]
    inside = toas[(toas > lo) & (toas < hi)]
    eps = 1e-7
    g = np.unique(np.concatenate([np.linspace(lo, hi, nt), inside - eps, inside + eps]))
    y = x.copy()
    L = np.empty(g.size)
    if orf == "crn":
        phic = F.phi_common(x)
        for k, t in enumerate(g):
            y[i0] = t
            L[k] = F.curn_term(F.dip_a, y, phic, F.dip_terms(y))
    else:
        cache = F.hd_prepare([F.hd_reduce(k, x) for k in range(F.P)], x)
        for k, t in enumerate(g):
            y[i0] = t
            L[k] = F.hd_swap(cache, F.dip_a, F.hd_reduce(F.dip_a, y, F.dip_terms(y)))
    w = np.exp(L - L.max())
    a, b = ex["boundaries"]
    seg = 0.5 * (w[1:] + w[:-1]) * np.diff(g)
    mid = 0.5 * (g[1:] + g[:-1])
    keep = (mid >= a) & (mid < b)
    return float(seg[~keep].sum() / seg.sum())


def _work(args):
    orf, rows_x, fine = args
    if _W.get("orf") != orf:
        _init(orf)
    na, nb, nt = (2 * NA, 2 * NB, 2 * NT - 1) if fine else (NA, NB, NT)
    out = []
    for x in rows_x:
        r = {n: _pair_pi(x, n, p, na, nb) for n, p in _W["pairs"]}
        r["t0:rest"] = _t0_pi(x, nt)
        out.append(r)
    return out


def summarise(pis: np.ndarray) -> dict:
    from ptagwb.reweight import batch_length, integrated_autocorr_time, obm_variance_of_mean

    n = pis.size
    tau = integrated_autocorr_time(pis) if pis.std() > 0 else 1.0
    var = obm_variance_of_mean(pis, batch_length(tau, n))
    h = n // 2
    return {"mean_pi": float(pis.mean()), "mcse": float(np.sqrt(var)), "tau": float(tau), "ess": float(n / tau),
            "max_single_draw_share": float(pis.max() / pis.sum()) if pis.sum() > 0 else 0.0,
            "split_half": [float(pis[:h].mean()), float(pis[h:].mean())], "n_draws": int(n)}


def main():
    import multiprocessing as mp

    from ptagwb import epta
    from ptagwb.binding import evidence_binding

    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    man = epta.load_manifest()
    accf = json.loads(ACC.read_text())
    rel = json.loads((REPO_ROOT / "configs" / "m3b" / "relevance" / "epta.json").read_text())
    excl = {e["params"][m]: e for e in accf["d9"]["exclusions"] for m in ("crn_pl", "hd_pl")}
    out = {"label": "supporting diagnostic only (plan Sec. 5.3): never classifies support or resolves UNRESOLVED",
           "grid": {"pair_cells": [NA, NB], "t0_points": NT}, "models": {}}
    ctx = mp.get_context("spawn")
    for key, orf in (("crn_pl", "crn"), ("hd_pl", "hd")):
        names, X, burn = epta.load_reference(key, man)
        rows = np.unique(np.linspace(burn, len(X) - 1, a.draws).round().astype(int))
        R = X[rows, :67]
        t = time.time()
        chunks = np.array_split(np.arange(len(R)), a.workers * 4)
        with ctx.Pool(a.workers) as pool:
            res = pool.map(_work, [(orf, R[c], False) for c in chunks if c.size])
            fine_rows = np.linspace(0, len(R) - 1, 20).astype(int)
            fine = pool.map(_work, [(orf, R[[i]], True) for i in fine_rows])
        res = [r for part in res for r in part]
        fine = [f[0] for f in fine]
        coarse_at_fine = [res[i] for i in fine_rows]
        regs = {}
        for reg in res[0]:
            pis = np.array([r[reg] for r in res])
            s = summarise(pis)
            s["grid_check_max_abs_diff"] = float(max(abs(f[reg] - c[reg]) for f, c in zip(fine, coarse_at_fine)))
            param = reg if reg != "t0:rest" else f"{man['dip']['param_prefix']}_t0"
            e = excl.get(param)
            s["in_U"] = bool(e is not None and (reg != "t0:rest" or e["type"] == "outside"))
            if reg != "t0:rest":
                ra = next((c for c in rel["class_a"] if c["model"] == key and c["param"] == reg), None)
                s["reference_shelf"] = ra["reference"]["S"] if ra else None
            else:
                s["reference_rest"] = rel["class_b"][key]["reference"]["rest"]
            regs[reg] = s
        out["models"][key] = {"rows": rows.tolist(), "seconds": time.time() - t, "regions": regs}
        print(key, f"{time.time() - t:.0f} s", flush=True)
        for reg, s in regs.items():
            print(f"   {reg}: mean {s['mean_pi']:.3g} mcse {s['mcse']:.2g} ess {s['ess']:.0f} max-share "
                  f"{s['max_single_draw_share']:.2g} grid {s['grid_check_max_abs_diff']:.1g} in_U {s['in_U']}", flush=True)
    out["completed"] = True
    out["binding"] = evidence_binding()
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "conditional_occupancy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
