"""Chain fingerprint gate on the released EPTA DR2new CURN and HD chains (docs/M3B_PLAN.md
Sec. 4.5 item 3; ``ptagwb.fingerprint``), with the common-grid discrimination (alternatives: 8 and
10 common modes) and the pinned fork enterprise as a diagnostic at the same draws.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_fingerprint.py [--n 60] [--no-enterprise]
Writes data/processed/m3b/epta/results/fingerprint.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ptagwb import epta
from ptagwb import fingerprint as fp
from ptagwb.binding import evidence_binding
from ptagwb.config import REPO_ROOT

BASE = REPO_ROOT / "data" / "processed" / "m3b" / "epta"


def chain_rows_text(model: str, rows: np.ndarray) -> list[list[str]]:
    want = set(int(r) for r in rows)
    out = {}
    with open(epta.extract_reference(model) / "chain_1.txt") as f:
        for i, ln in enumerate(f):
            if i in want:
                out[i] = ln.split()
    return [out[int(r)] for r in rows]


def enterprise_lnl(orf: str, X: np.ndarray, names: list[str]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        pts, out = Path(td) / "pts.npz", Path(td) / "out.npz"
        np.savez(pts, x=X, names=np.array(names))
        env = dict(os.environ, TEMPO2_OVERRIDE=str(BASE / "t2runtime"))
        p = subprocess.run([str(REPO_ROOT / "scripts" / "eptapy"), str(REPO_ROOT / "scripts" / "m3b_epta_oracle.py"),
                            "like", "--orf", orf, "--points", str(pts), "--out", str(out)], env=env,
                           capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-3000:])
        with np.load(out) as z:
            return np.asarray(z["lnlike"])


def run(n: int = 60, with_enterprise: bool = True) -> dict:
    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    res = {"manifest": man["manifest"], "n_requested": n, "models": {}}
    for orf, key in (("crn", "crn_pl"), ("hd", "hd_pl")):
        names, X, burn = epta.load_reference(key, man)
        rows = fp.spread_rows(len(X), burn, n)
        txt = chain_rows_text(key, rows)
        theta = X[rows, :67]
        lnl_ref = X[rows, 68]
        model = epta.EPTAModel(psrs, man, orf)
        if model.param_names != names:
            raise RuntimeError("parameter order differs from pars.txt")
        vals, grads = [], []
        for x in theta:
            v, g = model.value_and_grad(x)
            vals.append(float(v))
            grads.append(np.asarray(g))
        vals, grads = np.array(vals), np.array(grads)
        dx = np.array([fp.stored_precision(t[:67]) for t in txt])
        i0 = model.t0_index
        mask = np.ones(67, dtype=bool)
        mask[i0] = False
        sig_par = np.sum(np.abs(grads[:, mask]) * dx[:, mask], axis=1)
        dip = psrs[model.dip.index]
        margin = fp.discontinuity_margin(theta[:, i0], dip.toas, dx[:, i0])
        r = fp.evaluate(vals, lnl_ref, sig_par)
        ent = {}
        if with_enterprise:
            le = enterprise_lnl(orf, theta, names)
            re_ = fp.evaluate(le, lnl_ref, np.zeros(len(le)))
            dd = vals - le
            ent = {"enterprise_vs_ref": re_.as_dict(), "ours_vs_enterprise_max_abs": float(np.max(np.abs(dd))),
                   "ours_vs_enterprise_mean": float(dd.mean())}
        alts = {}
        for nc in (8, 10):
            m2 = copy.deepcopy(man)
            m2["common"]["modes"] = nc
            alt_model = epta.EPTAModel(psrs, m2, orf)
            la = np.array([float(alt_model.logL(x)) for x in theta])
            alts[f"common_modes_{nc}"] = fp.evaluate(la, lnl_ref, sig_par)
        disc = fp.discriminate(r, alts)
        res["models"][key] = {"rows": rows.tolist(), "result": r.as_dict(), "t0_margin_in_units_of_precision": margin,
                              "t0_margin_ok": bool(margin > 1.0), "deltas": r.deltas.tolist(), **ent,
                              "discrimination": disc, "alternatives": {k: v.as_dict() for k, v in alts.items()}}
        res["models"][key]["pass"] = bool(r.passed and margin > 1.0)
    # Model-dependent constants (plan Sec. 4.5, 6.2): both reference chains come from the same
    # fork code, whose absolute constant convention the oracle check reproduces for both models
    # (ours - enterprise ~1e-10 nats at every point), so the oracle-predicted c_HD - c_CURN is 0.
    # Budget: 3.5 combined standard errors of the two means plus 2 sigma_eval.
    c = {k: v["result"]["c_m"] for k, v in res["models"].items()}
    se = np.sqrt(sum((v["result"]["sd"] / np.sqrt(v["result"]["n"])) ** 2 for v in res["models"].values()))
    res["c_hd_minus_c_crn"] = c["hd_pl"] - c["crn_pl"]
    res["c_diff_predicted"] = 0.0
    res["c_diff_budget"] = float(3.5 * se + 2 * fp.SIGMA_EVAL)
    res["c_diff_consistent"] = bool(abs(res["c_hd_minus_c_crn"]) <= res["c_diff_budget"])
    res["pass"] = fp.overall_pass(res["models"], res["c_diff_consistent"])
    res["binding"] = evidence_binding()
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    (BASE / "results" / "fingerprint.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--no-enterprise", action="store_true")
    a = ap.parse_args()
    res = run(a.n, not a.no_enterprise)
    for k, v in res["models"].items():
        r = v["result"]
        print(f"{k}: n={r['n']} c_m={r['c_m']:.3e} sd={r['sd']:.3e} chi2={r['chi2_per_dof']:.3f} pass={v['pass']} "
              f"t0 margin {v['t0_margin_in_units_of_precision']:.2e}")
        if "enterprise_vs_ref" in v:
            print(f"   enterprise vs ref chi2 {v['enterprise_vs_ref']['chi2_per_dof']:.3f}; ours-enterprise max {v['ours_vs_enterprise_max_abs']:.2e}")
        for ak, av in v["alternatives"].items():
            print(f"   alternative {ak}: chi2 {av['chi2_per_dof']:.3e} sd {av['sd']:.3e}")
        print("   discrimination resolved:", v["discrimination"]["resolved"])
    print("c_HD - c_CURN", res["c_hd_minus_c_crn"], "consistent:", res.get("c_diff_consistent"))
    print("FINGERPRINT PASS" if res["pass"] else "FINGERPRINT FAIL")


if __name__ == "__main__":
    main()
