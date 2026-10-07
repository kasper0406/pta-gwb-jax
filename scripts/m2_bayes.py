"""HD vs CURN Bayes factors from the M2 chains (reweighting, reverse reweighting, bridge).

    uv run --no-sync python scripts/m2_bayes.py [--pairs vg14 vg5] [--thin 1]

For each pair (CURN run, HD run) on the same prior: evaluates log L_HD and log L_CURN at both
chains' draws, then ``ptagwb.evidence`` estimators. Self-check: the CURN chain reweighted by the
CURN likelihood computed through the correlated-ORF code path with Gamma = I must give BF = 1.
Writes outputs/m2/bayes_factors.json.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from m2_common import ROOT, save_json

from ptagwb import evidence
from ptagwb.data import get_tspan, load_pulsars
from ptagwb.diagnostics import ess_bulk, ess_tail, rhat
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.noise import load_noise_dict
from ptagwb.sampling import ModelSpec, Posterior, load_run

PAIRS = {
    "vg14": ("curn_vg_14f", "hd_vg_14f", 14),
    "g433_14": ("curn_g433_14f", "hd_g433_14f", 14),
    "vg5": ("curn_vg_5f", "hd_vg_5f", 5),
}


MODEL_KEYS = ("common", "gamma", "n_common", "n_modes", "position", "prior_overrides")


def validate_pair(runs: dict, n_common: int) -> ModelSpec:
    """Fail unless the CURN and HD runs sample the same parameter space under the same prior and
    differ only in the ORF, and match the likelihood this script evaluates (enterprise positions,
    ``n_common`` modes). Returns the CURN run's ModelSpec."""
    specs = {k: r["meta"]["config"]["model"] for k, r in runs.items()}
    full = {k: {**ModelSpec().__dict__, **v} for k, v in specs.items()}
    for k, m in full.items():
        if m["orf"] != k:
            raise ValueError(f"run for {k!r} has orf={m['orf']!r}")
        if m["position"] != "enterprise" or m["n_common"] != n_common or m["common"] != "powerlaw":
            raise ValueError(f"{k}: position/n_common/common = {m['position']}/{m['n_common']}/{m['common']}, "
                             f"expected enterprise/{n_common}/powerlaw")
    if "hd" in runs:
        for key in MODEL_KEYS:
            if full["curn"][key] != full["hd"][key]:
                raise ValueError(f"CURN and HD runs differ in {key!r}: {full['curn'][key]!r} vs {full['hd'][key]!r}")
        a, b = runs["curn"]["meta"], runs["hd"]["meta"]
        if a["names"] != b["names"] or a["lo"] != b["lo"] or a["hi"] != b["hi"]:
            raise ValueError("CURN and HD runs have different parameters or prior bounds")
        la, lb = a.get("likelihood", {}), b.get("likelihood", {})
        for key in ("common", "n_common", "n_modes", "convention"):
            if la.get(key) != lb.get(key):
                raise ValueError(f"stored likelihood metadata differ in {key!r}: {la.get(key)} vs {lb.get(key)}")
    return ModelSpec(**specs["curn"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["vg14", "g433_14", "vg5"])
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="draws per batched device call")
    ap.add_argument("--systematics", action="store_true", help="ICRS-position and released-feather BFs")
    ap.add_argument("--recompute", action="store_true", help="ignore cached log-likelihood arrays")
    ap.add_argument("--own-from-chain", action=argparse.BooleanOptionalAction, default=True,
                    help="use the stored chain logL for each run's own model (verified on 64 draws)")
    args = ap.parse_args()
    psrs = load_pulsars(verbose=False)
    T = get_tspan(psrs)
    nd = load_noise_dict()
    terms = precompute(psrs, nd, T, position="enterprise")
    terms_icrs = precompute(psrs, nd, T, position="icrs") if args.systematics else None
    feather_terms = None
    if args.systematics:
        try:
            sys.path.insert(0, str(ROOT / "tests"))
            from oracle_helpers import feather_pulsars

            fp = feather_pulsars()
            name_order = [p.name for p in psrs]
            fp = sorted(fp, key=lambda p: name_order.index(p.name))
            assert abs(get_tspan(fp) - T) < 1e-3
            ft = precompute(fp, nd, T)  # feathers carry enterprise positions
            feather_terms = (ft, ft)
        except (FileNotFoundError, StopIteration, ImportError) as e:  # feathers not fetched
            print("feather systematics skipped:", e)
    out = {}
    for key in args.pairs:
        rc, rh, nc = PAIRS[key]
        try:
            runs = {"curn": load_run(rc), "hd": load_run(rh)}
        except FileNotFoundError:
            try:
                runs = {"curn": load_run(rc)}
            except FileNotFoundError:
                print(f"{key}: runs missing, skipped")
                continue
        spec = validate_pair(runs, nc)
        posts = {
            o: Posterior(PTALikelihood(terms, T, n_common=nc, orf=o), ModelSpec(**{**spec.__dict__, "orf": o}))
            for o in ("curn", "hd")
        }
        res = {"curn_run": rc, "hd_run": rh if "hd" in runs else None}
        cache = ROOT / "outputs" / "m2" / f"bf_loglikes_{key}_thin{args.thin}.npz"
        ll = {}
        if cache.exists() and not args.recompute:
            with np.load(cache) as f:
                ll = {tuple(k.split("__")): f[k] for k in f.files}
            res["loglikes_from_cache"] = str(cache.relative_to(ROOT))
        for src, run in runs.items():
            assert run["names"] == posts["curn"].names
            X = run["x"][:, :: args.thin]
            C, N, D = X.shape
            if (src, "hd") not in ll:
                t0 = time.time()
                for o, post in posts.items():
                    if o == src and args.own_from_chain:
                        # the run's own model: the sampler stored log L for every draw (verified on
                        # a random subset below) instead of recomputing all (HD ~1 s/draw on CPU)
                        ll[(src, o)] = run["logL"][:, :: args.thin].copy()
                    else:
                        ll[(src, o)] = post.logL_samples(X.reshape(-1, D), batch=args.batch).reshape(C, N)
                res[f"eval_seconds_{src}"] = time.time() - t0
            # the evaluation model must reproduce the stored chain logL (random subset, always)
            idx = np.random.default_rng(0).choice(C * N, size=min(64, C * N), replace=False)
            chk = posts[src].logL_samples(X.reshape(-1, D)[idx], batch=args.batch)
            res[f"max_abs_logL_subset_check_{src}"] = float(np.abs(chk - run["logL"][:, :: args.thin].ravel()[idx]).max())
            if not res[f"max_abs_logL_subset_check_{src}"] < 1e-5:
                raise RuntimeError(f"{key}/{src}: stored chain logL does not match the evaluation model")
            # consistency of the (possibly cached) arrays with the sampler's own logL: fail on mismatch
            own = ll[(src, src)] - run["logL"][:, :: args.thin]
            res[f"max_abs_logL_recompute_diff_{src}"] = float(np.abs(own).max())
            if not np.abs(own).max() < 1e-5:
                raise RuntimeError(f"{key}/{src}: recomputed logL differs from the stored chain logL by "
                                   f"{np.abs(own).max():.3e}; the evaluation model is not the sampled model")
        np.savez(cache, **{"__".join(k): v for k, v in ll.items()})
        l_curn = ll[("curn", "hd")] - ll[("curn", "curn")]
        res["reweight"] = evidence.reweight(l_curn)
        if "hd" in runs:
            l_hd = ll[("hd", "hd")] - ll[("hd", "curn")]
            res["reverse_reweight"] = evidence.reverse_reweight(l_hd)
            res["bridge"] = evidence.bridge(l_curn, l_hd, n_boot=500)
            # block-length sensitivity of the (conditional) bootstrap errors
            res["block_sensitivity"] = {}
            for b in (1, 10, 30, 100, 300):
                res["block_sensitivity"][str(b)] = {
                    "reweight_bf_sd": evidence.reweight(l_curn, n_boot=500, block=b)["bf_sd"],
                    "reverse_reweight_bf_sd": evidence.reverse_reweight(l_hd, n_boot=500, block=b)["bf_sd"],
                    "bridge_bf_sd": evidence.bridge(l_curn, l_hd, n_boot=300, block=(b, b))["bf_sd"],
                }
            # convergence of the log likelihood ratio itself in each chain set
            res["log_lr_diagnostics"] = {
                "curn_draws": {"rhat": rhat(l_curn), "ess_bulk": ess_bulk(l_curn), "ess_tail": ess_tail(l_curn)},
                "hd_draws": {"rhat": rhat(l_hd), "ess_bulk": ess_bulk(l_hd), "ess_tail": ess_tail(l_hd)},
            }
        # self-check: CURN via the correlated path with Gamma = I
        eye = Posterior(PTALikelihood(terms, T, n_common=nc, orf=np.eye(len(psrs))), spec)
        Xc = runs["curn"]["x"][:, :: max(1, args.thin * 10)]
        C, N, D = Xc.shape
        Xf = Xc.reshape(-1, D)
        l_self = (eye.logL_samples(Xf, batch=args.batch) - posts["curn"].logL_samples(Xf, batch=args.batch)).reshape(C, N)
        res["self_check_identity_orf"] = evidence.reweight(l_self, n_boot=200)
        res["self_check_max_abs_dlogL"] = float(np.abs(l_self).max())
        # Systematics by importance-ratio estimation from our CURN posterior (thinned 5x):
        #   BF' = E[L'_HD / L_CURN] / E[L'_CURN / L_CURN]
        # with primed likelihoods on (a) ICRS positions (HD only) and (b) the released feathers.
        if args.systematics and key in ("vg14", "g433_14"):
            Xs = runs["curn"]["x"][:, ::5]
            C, N, D = Xs.shape
            Xf = Xs.reshape(-1, D)
            base = posts["curn"].logL_samples(Xf, batch=args.batch).reshape(C, N)
            lhb = posts["hd"].logL_samples(Xf, batch=args.batch).reshape(C, N) - base  # baseline, same draws
            variants = {"icrs_positions": (terms_icrs, terms_icrs)}
            if feather_terms is not None:
                variants["released_feathers"] = feather_terms
            for vname, (tc, th) in variants.items():
                pc = Posterior(PTALikelihood(tc, T, n_common=nc, orf="curn"), spec)
                ph = Posterior(PTALikelihood(th, T, n_common=nc, orf="hd"), spec)
                lc = pc.logL_samples(Xf, batch=args.batch).reshape(C, N) - base
                lh = ph.logL_samples(Xf, batch=args.batch).reshape(C, N) - base
                rh_, rc_ = evidence.reweight(lh, n_boot=300), evidence.reweight(lc, n_boot=300)
                lnbf = rh_["ln_bf"] - rc_["ln_bf"]
                # joint bootstrap of the ratio (same blocks for numerator and denominator)
                rng = np.random.default_rng(1)
                b = max(rh_["block"], rc_["block"])
                lnbf0 = evidence.log_mean_exp(lhb)
                boots, dboots = [], []
                for _ in range(300):
                    idx = [evidence._block_resample_idx(N, b, rng) for _ in range(C)]
                    rs = lambda a: np.stack([a[c, i] for c, i in enumerate(idx)])  # noqa: B023
                    v = evidence.log_mean_exp(rs(lh)) - evidence.log_mean_exp(rs(lc))
                    boots.append(v)
                    dboots.append(v - evidence.log_mean_exp(rs(lhb)))
                res[f"systematic_{vname}"] = {
                    "bf": float(np.exp(lnbf)), "ln_bf": float(lnbf), "ln_bf_sd": float(np.std(boots, ddof=1)),
                    "baseline_same_draws_bf": float(np.exp(lnbf0)),
                    "delta_ln_bf_vs_baseline": float(lnbf - lnbf0), "delta_ln_bf_sd": float(np.std(dboots, ddof=1)),
                    "kish_ess_hd": rh_["kish_ess"], "kish_ess_curn": rc_["kish_ess"], "n": int(C * N),
                    "std_dlogL_curn": float(np.std(lc)),
                }
        out[key] = res
        print(key, {k: (v["bf"], v.get("bf_sd", v.get("ln_bf_sd")), v.get("kish_ess"))
                    for k, v in res.items() if isinstance(v, dict) and "bf" in v})
    save_json(out, ROOT / "outputs" / "m2" / "bayes_factors.json")


if __name__ == "__main__":
    sys.exit(main())
