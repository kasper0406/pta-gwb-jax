"""Q4 analysis: aggregate sampler runs (ESS/s, ESS per 1000 gradients; mean +- sd over seeds) and
bias checks against the M2 production chains.

Bias checks (per run, against runs/curn_g433_14f or runs/hd_g433_14f):

* quantiles q05/q50/q95 of the common amplitude and 16 IRN parameters (8 pulsars x {log10_A,
  gamma}, including the funnel-shaped ones named in docs/M2_RESULTS.md):
  z = (q_run - q_ref) / sqrt(MCSE_run^2 + MCSE_ref^2) (Vehtari et al. 2021 quantile MCSE);
* all-parameter mean comparison: z_j = (mean_run - mean_ref) / sqrt(MCSE_run^2 + MCSE_ref^2)
  with the MCSE of the raw (untransformed) mean, var / ESS_mean, ESS_mean = autocorrelation ESS
  of the raw split chains (Vehtari et al. 2021); reported as max |z_j| and the *descriptive*
  sum_j z_j^2 / D (the z_j are correlated, so this is not a calibrated chi^2_D test), plus the
  median / extreme posterior-sd ratios (variance bias of unadjusted samplers).
"No discrepancy detected" means no |z| > 3; it is not a proof of agreement, in particular for
tails no run visited.

    uv run --no-sync python bench/analyze_samplers.py [glob ...]
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptagwb import diagnostics as dg
from ptagwb.sampling import load_run

BENCH = Path(__file__).resolve().parent
PSRS = ["B1855+09", "J0610-2100", "J1747-4036", "J1903+0327", "J2145-0750", "J0437-4715", "B1937+21", "J1909-3744"]
CHECK = ["gw_log10_A"] + [f"{p}_red_noise_{q}" for p in PSRS for q in ("log10_A", "gamma")]
REF = {"curn": "curn_g433_14f", "hd": "hd_g433_14f"}


def _ref(model, cache={}):  # noqa: B006
    if model not in cache:
        r = load_run(REF[model])
        cache[model] = (r["x"], list(r["names"]))
    return cache[model]


def bias(model: str, X: np.ndarray, names: list[str]) -> dict:
    Xr, nr = _ref(model)
    assert nr == names
    zq = []
    for p in CHECK:
        j = names.index(p)
        for q in (0.05, 0.5, 0.95):
            a, b = X[:, :, j], Xr[:, :, j]
            d = np.quantile(a, q) - np.quantile(b, q)
            se = np.hypot(dg.mcse_quantile(a, q), dg.mcse_quantile(b, q))
            zq.append((p, q, float(d / se), float(d)))
    zs = np.array([z for _, _, z, _ in zq])
    D = X.shape[-1]
    zm, sdr = np.empty(D), np.empty(D)
    for j in range(D):
        a, b = X[:, :, j], Xr[:, :, j]
        ea, eb = dg.ess(dg._split(a)), dg.ess(dg._split(b))  # raw-mean ESS (not rank-normalised)
        va, vb = a.var(), b.var()
        zm[j] = (a.mean() - b.mean()) / np.sqrt(va / ea + vb / eb)
        sdr[j] = np.sqrt(va / vb)
    worst = max(zq, key=lambda t: abs(t[2]))
    jc = names.index("gw_log10_A")
    return {
        "quantile_z_max_abs": float(np.max(np.abs(zs))),
        "quantile_z_worst": {"param": worst[0], "q": worst[1], "z": worst[2], "delta": worst[3]},
        "quantile_n_abs_z_gt3": int(np.sum(np.abs(zs) > 3)),
        "quantile_n_tests": len(zs),
        "common_q50_delta": float(np.quantile(X[:, :, jc], 0.5) - np.quantile(Xr[:, :, jc], 0.5)),
        "mean_sumz2": float(np.sum(zm**2)),  # descriptive only (correlated z_j)
        "mean_dof": D,
        "mean_z_max_abs": float(np.max(np.abs(zm))),
        "mean_z_worst_param": names[int(np.argmax(np.abs(zm)))],
        "sd_ratio_median": float(np.median(sdr)),
        "sd_ratio_min": float(sdr.min()),
        "sd_ratio_max": float(sdr.max()),
        "sd_ratio_common": float(sdr[jc]),
    }


def main():
    pats = sys.argv[1:] or [str(BENCH / "results" / "samplers" / "*.json")]
    files = sorted({f for p in pats for f in glob.glob(p)})
    rows, groups = [], defaultdict(list)
    for f in files:
        s = json.loads(Path(f).read_text())
        tag = Path(f).stem
        npz = BENCH / "runs" / f"{tag}.npz"
        if npz.exists() and (s.get("bias") or {}).get("version") != 2:
            with np.load(npz) as d:
                s["bias"] = bias(s["model"], d["x"].astype(np.float64), [str(n) for n in d["names"]])
                s["bias"]["version"] = 2
            Path(f).write_text(json.dumps(s, indent=1, default=float))
        key = (s["model"], s["sampler"], s.get("like"), s["chains"], json.dumps(s.get("opts", {}), sort_keys=True),
               s["warmup"], s["samples"], s.get("group", ""))
        groups[key].append(s)
        rows.append((tag, s))
    out = []
    mets = ["ess_per_s_bulk_common", "ess_per_s_tail_common", "ess_per_s_bulk_min", "ess_per_s_tail_min",
            "ess_per_kgrad_bulk_common", "ess_per_kgrad_tail_common", "ess_per_kgrad_bulk_min", "ess_per_kgrad_tail_min",
            "ess_bulk_common", "ess_bulk_min", "ess_tail_min", "rhat_max", "accept_mean", "grads_per_draw_mean", "t_samp", "t_warm"]
    for key, ss in sorted(groups.items()):
        g = {"model": key[0], "sampler": key[1], "like": key[2], "chains": key[3], "opts": json.loads(key[4]),
             "warmup": key[5], "samples": key[6], "n_seeds": len(ss), "seeds": [s["seed"] for s in ss]}
        for m in mets:
            v = np.array([s[m] for s in ss], dtype=float)
            g[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "values": v.tolist()}
        g["divergences"] = [s["divergences"] for s in ss]
        if all("bias" in s for s in ss):
            g["bias"] = [s["bias"] for s in ss]
        out.append(g)
    (BENCH / "results" / "samplers_summary.json").write_text(json.dumps(out, indent=1, default=float))

    def f(g, m, scale=1.0):
        d = g[m]
        return f"{d['mean'] * scale:.3g}" + (f" ± {d['sd'] * scale:.2g}" if d["sd"] is not None else "")

    print("| model | sampler | chains | opts | seeds | ESS/s common bulk | ESS/s min bulk | ESS/s min tail | ESS/kgrad common | ESS/kgrad min bulk | ESS/kgrad min tail | max R-hat | grads/draw | |z|max quantile (n>3/N) | mean z: sum z^2/D (max abs z), descriptive | sd ratio med [min,max] |")
    print("|" + "---|" * 16)
    for g in out:
        b = g.get("bias", [])
        bz = ", ".join(f"{x['quantile_z_max_abs']:.1f} ({x['quantile_n_abs_z_gt3']}/{x['quantile_n_tests']})" for x in b)
        bc = ", ".join(f"{x['mean_sumz2'] / x['mean_dof']:.2f} (max {x['mean_z_max_abs']:.1f})" for x in b)
        bs = ", ".join(f"{x['sd_ratio_median']:.2f} [{x['sd_ratio_min']:.2f},{x['sd_ratio_max']:.2f}]" for x in b)
        print(f"| {g['model']} | {g['sampler']} | {g['chains']} | {g['opts'] or ''} | {g['n_seeds']} | {f(g, 'ess_per_s_bulk_common')} | "
              f"{f(g, 'ess_per_s_bulk_min')} | {f(g, 'ess_per_s_tail_min')} | {f(g, 'ess_per_kgrad_bulk_common')} | "
              f"{f(g, 'ess_per_kgrad_bulk_min')} | {f(g, 'ess_per_kgrad_tail_min')} | {f(g, 'rhat_max')} | {f(g, 'grads_per_draw_mean')} | {bz} | {bc} | {bs} |")


if __name__ == "__main__":
    main()
