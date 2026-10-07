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
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.noise import load_noise_dict
from ptagwb.sampling import ModelSpec, Posterior, load_run

PAIRS = {
    "vg14": ("curn_vg_14f", "hd_vg_14f", 14),
    "g433_14": ("curn_g433_14f", "hd_g433_14f", 14),
    "vg5": ("curn_vg_5f", "hd_vg_5f", 5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["vg14", "g433_14", "vg5"])
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="draws per batched device call")
    ap.add_argument("--systematics", action="store_true", help="ICRS-position and released-feather BFs")
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
        spec = ModelSpec(**runs["curn"]["meta"]["config"]["model"])
        posts = {
            o: Posterior(PTALikelihood(terms, T, n_common=nc, orf=o), ModelSpec(**{**spec.__dict__, "orf": o}))
            for o in ("curn", "hd")
        }
        res = {"curn_run": rc, "hd_run": rh if "hd" in runs else None}
        ll = {}
        for src, run in runs.items():
            assert run["names"] == posts["curn"].names
            X = run["x"][:, :: args.thin]
            C, N, D = X.shape
            t0 = time.time()
            for o, post in posts.items():
                ll[(src, o)] = post.logL_samples(X.reshape(-1, D), batch=args.batch).reshape(C, N)
            res[f"eval_seconds_{src}"] = time.time() - t0
            # consistency with the sampler's own logL
            own = ll[(src, src)] - run["logL"][:, :: args.thin]
            res[f"max_abs_logL_recompute_diff_{src}"] = float(np.abs(own).max())
        l_curn = ll[("curn", "hd")] - ll[("curn", "curn")]
        res["reweight"] = evidence.reweight(l_curn)
        if "hd" in runs:
            l_hd = ll[("hd", "hd")] - ll[("hd", "curn")]
            res["reverse_reweight"] = evidence.reverse_reweight(l_hd)
            res["bridge"] = evidence.bridge(l_curn, l_hd, n_boot=500)
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
                boots = []
                for _ in range(300):
                    idx = [evidence._block_resample_idx(N, b, rng) for _ in range(C)]
                    boots.append(evidence.log_mean_exp(np.stack([lh[c, i] for c, i in enumerate(idx)]))
                                 - evidence.log_mean_exp(np.stack([lc[c, i] for c, i in enumerate(idx)])))
                res[f"systematic_{vname}"] = {
                    "bf": float(np.exp(lnbf)), "ln_bf": float(lnbf), "ln_bf_sd": float(np.std(boots, ddof=1)),
                    "kish_ess_hd": rh_["kish_ess"], "kish_ess_curn": rc_["kish_ess"], "n": int(C * N),
                    "std_dlogL_curn": float(np.std(lc)),
                }
        out[key] = res
        print(key, {k: (v["bf"], v.get("bf_sd", v.get("ln_bf_sd")), v.get("kish_ess")) for k, v in res.items() if isinstance(v, dict)})
    save_json(out, ROOT / "outputs" / "m2" / "bayes_factors.json")


if __name__ == "__main__":
    sys.exit(main())
