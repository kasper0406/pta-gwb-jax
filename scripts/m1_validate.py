"""M1 validation: front end vs released feathers, likelihood vs discovery / enterprise / chains.

Writes outputs/m1_validation.json (git-ignored) and prints the numbers quoted in
docs/M1_VALIDATION.md. Needs the oracle group (scripts/setup_oracle_env.sh) and a populated
pulsar cache (scripts/ingest.py).

    uv run --no-sync python scripts/m1_validate.py [--enterprise] [--n-chain 6 --n-prior 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
from oracle_helpers import (
    compare_frontend,
    discovery_model,
    feather_pulsars,
    load_chain,
    parameter_points,
    to_discovery_params,
    to_discovery_pulsar,
)

from ptagwb.data import get_tspan, load_pulsars
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.noise import load_noise_dict


def timeit(fn, *a, k=20):
    r = fn(*a)
    jax.block_until_ready(r)
    t = time.perf_counter()
    for _ in range(k):
        r = fn(*a)
        jax.block_until_ready(r)
    return (time.perf_counter() - t) / k * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chain", type=int, default=6)
    ap.add_argument("--n-prior", type=int, default=5)
    ap.add_argument("--enterprise", action="store_true", help="also evaluate enterprise (slow)")
    args = ap.parse_args()

    out: dict = {}
    nd = load_noise_dict()
    ours = load_pulsars(verbose=False)
    refs = feather_pulsars()
    rmap = {p.name: p for p in refs}
    assert [p.name for p in ours] == [p.name for p in refs]
    T, Tref = get_tspan(ours), get_tspan(refs)
    out["tspan"] = {"ours": T, "feathers": Tref}
    print(f"Tspan ours {T!r}  feathers {Tref!r}")

    # ---------------- front end
    fe = [compare_frontend(p, rmap[p.name], nd) for p in ours]
    out["frontend"] = fe
    mx = {k: max(r[k] for r in fe) for k in ("max_dres_s", "rms_dres_s", "max_dres_perp_s", "rms_dres_perp_s")}
    print(f"front end: max |dres| {mx['max_dres_s']:.3e} s, max rms dres {mx['rms_dres_s']:.3e} s, "
          f"max |dres_perp| {mx['max_dres_perp_s']:.3e} s, max rms dres_perp {mx['rms_dres_perp_s']:.3e} s")

    # ---------------- precompute (ours) on both inputs
    t0 = time.perf_counter()
    terms_ours = precompute(ours, nd, T)
    t_pre = time.perf_counter() - t0
    terms_ref = precompute(refs, nd, Tref)
    out["precompute_s"] = t_pre
    names = [p.name for p in ours]

    import discovery as ds  # noqa: F401

    d_ref = [to_discovery_pulsar(p, nd) for p in refs]
    d_ours = [to_discovery_pulsar(p, nd) for p in ours]

    res = {}
    for orf, chain_name in (("curn", "m2a"), ("hd", "m3a")):
        chain = load_chain(chain_name)
        seed = 1 if orf == "curn" else 2
        groups = (
            ["chain"] * args.n_chain + ["prior<=-13"] * args.n_prior + ["prior<=-11"] * args.n_prior
        )
        pts = parameter_points(names, chain, args.n_chain, args.n_prior, seed=seed, rn_log10_A_range=(-20, -13))
        pts += parameter_points(names, None, 0, args.n_prior, seed=seed + 10)
        L_ref = PTALikelihood(terms_ref, Tref, orf=orf)
        L_ours = PTALikelihood(terms_ours, T, orf=orf)
        # feather arrays with ICRS positions (isolates the enterprise B1950 position quirk)
        terms_ref_icrs = [type(t)(**{**t.__dict__, "pos": rmap_ours.pos}) for t, rmap_ours in zip(terms_ref, ours)]
        L_ref_icrs = PTALikelihood(terms_ref_icrs, Tref, orf=orf)
        D = {}
        for tag, dps, Tx in (("ref", d_ref, Tref), ("ours", d_ours, T)):
            for ent in (True, False):
                m = discovery_model(dps, orf, Tx, ecorr_enterprise=ent)
                D[(tag, ent)] = (jax.jit(m.logL), m.logL.params)
        rows = []
        for grp, pt in zip(groups, pts):
            row = {"chain_row": pt.get("_chain_row"), "group": grp}
            pd = to_discovery_params(pt, orf)
            for key, (fn, params) in D.items():
                row[f"disc_{key[0]}_{'nmin2' if key[1] else 'nmin1'}"] = float(fn({k: pd[k] for k in params}))
            row["ours_ref"] = float(L_ref.logL(L_ref.params_from_named(pt)))
            row["ours_ours"] = float(L_ours.logL(L_ours.params_from_named(pt)))
            row["ours_ref_icrs"] = float(L_ref_icrs.logL(L_ref_icrs.params_from_named(pt)))
            if pt.get("_chain_row") is not None:
                row["chain_logl"] = float(chain.iloc[pt["_chain_row"]]["logl"])
            rows.append(row)
        res[orf] = rows

        a = np.array([[r[k] for k in ("ours_ref", "disc_ref_nmin2", "ours_ours", "disc_ours_nmin2", "ours_ref_icrs")] for r in rows])
        print(f"\n[{orf}] {len(rows)} points")
        for grp in ("chain", "prior<=-13", "prior<=-11"):
            sel = np.array([g == grp for g in groups])
            d_ref_abs = a[sel, 0] - a[sel, 1]
            d_ours_abs = a[sel, 2] - a[sel, 3]
            print(f"  [{grp:10s}] ours - discovery: feathers max |d| {np.max(np.abs(d_ref_abs)):.2e} "
                  f"spread {np.ptp(d_ref_abs):.2e};  our arrays max |d| {np.max(np.abs(d_ours_abs)):.2e} "
                  f"spread {np.ptp(d_ours_abs):.2e}")
        a = a[np.array([g != "prior<=-11" for g in groups])]
        fe_d = a[:, 2] - a[:, 0]
        print(f"  front-end impact (our arrays - feathers, both ours): mean {fe_d.mean():.4f}, "
              f"spread {np.ptp(fe_d):.4f}, std {fe_d.std():.4f}")
        if orf == "hd":
            pq = a[:, 4] - a[:, 0]
            print(f"  B-name position quirk (ICRS - enterprise pos, feathers): mean {pq.mean():.3e} spread {np.ptp(pq):.3e}")
        ch = [(r["ours_ref"] - r["chain_logl"], r["ours_ours"] - r["chain_logl"]) for r in rows if "chain_logl" in r]
        if ch:
            ch = np.array(ch)
            print(f"  chain logl: ours(feathers) - logl max |.| {np.max(np.abs(ch[:, 0])):.3f}; "
                  f"ours(our arrays) - logl: {ch[:, 1].mean():.3f} +- {ch[:, 1].std():.3f}")
        n1 = np.array([r["disc_ref_nmin1"] - r["disc_ref_nmin2"] for r in rows])
        print(f"  discovery default ECORR (keeps singletons) - nmin2: mean {n1.mean():.4f} spread {np.ptp(n1):.2e}")

        # timings
        p = L_ours.params_from_named(pts[0])
        tm = {"value_ms": timeit(L_ours.logL, p), "value_and_grad_ms": timeit(L_ours.value_and_grad, p)}
        if orf == "hd":
            Lm = PTALikelihood(terms_ours, T, orf="hd", grad_precision="mixed")
            tm["value_and_grad_mixed_ms"] = timeit(Lm.value_and_grad, p)
            vg0, vg1 = L_ours.value_and_grad(p), Lm.value_and_grad(p)
            g0 = np.concatenate([np.ravel(vg0[1][k]) for k in sorted(vg0[1])])
            g1 = np.concatenate([np.ravel(vg1[1][k]) for k in sorted(vg1[1])])
            tm["mixed_grad_max_abs_err"] = float(np.max(np.abs(g1 - g0)))
            tm["mixed_grad_max_rel_err_vs_max"] = float(np.max(np.abs(g1 - g0)) / np.max(np.abs(g0)))
        fn, params = D[("ours", True)]
        pd = to_discovery_params(pts[0], orf)
        tm["discovery_value_ms"] = timeit(fn, {k: pd[k] for k in params})
        print("  timings:", {k: float(f"{v:.3g}") for k, v in tm.items()})
        res[orf + "_timing"] = tm

    out["likelihood"] = res

    # per-pulsar front-end impact on the (separable) CURN likelihood at chain samples
    chain = load_chain("m2a")
    pts = parameter_points(names, chain, 20, 0, seed=5)
    per = []
    for i, name in enumerate(names):
        Lo = PTALikelihood([terms_ours[i]], T, orf="curn")
        Lr = PTALikelihood([terms_ref[i]], Tref, orf="curn")
        d = np.array([float(Lo.logL(Lo.params_from_named(pt))) - float(Lr.logL(Lr.params_from_named(pt))) for pt in pts])
        per.append({"name": name, "mean": float(d.mean()), "std": float(d.std()), "ptp": float(np.ptp(d))})
    per.sort(key=lambda r: -r["std"])
    out["frontend_curn_per_pulsar"] = per
    print("\nper-pulsar CURN front-end impact (std over 20 chain samples), top 5:")
    for r in per[:5]:
        print(f"  {r['name']:12s} mean {r['mean']:+.4f} std {r['std']:.4f} ptp {r['ptp']:.4f}")

    if args.enterprise:
        out["enterprise"] = enterprise_check(refs, ours, nd, Tref, T, names, terms_ref, terms_ours)

    path = ROOT / "outputs" / "m1_validation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path.relative_to(ROOT)}")


def enterprise_check(refs, ours, nd, Tref, T, names, terms_ref, terms_ours, n_pts=10):
    from oracle_helpers import enterprise_pta

    res = {}
    for orf in ("curn", "hd"):
        t0 = time.time()
        pta = enterprise_pta(refs, nd, Tref, orf)
        print(f"\n[enterprise {orf}] built in {time.time() - t0:.0f} s")
        L = PTALikelihood(terms_ref, Tref, orf=orf, convention="enterprise")
        chain = load_chain("m2a" if orf == "curn" else "m3a")
        pts = parameter_points(names, chain, n_pts // 2, n_pts - n_pts // 2, seed=11, rn_log10_A_range=(-20, -13))
        rows = []
        for pt in pts:
            t0 = time.time()
            e = float(pta.get_lnlikelihood(enterprise_params(pt, orf)))
            te = time.time() - t0
            o = float(L.logL(L.params_from_named(pt)))
            rows.append({"enterprise": e, "ours": o, "t_enterprise_s": te})
            print(f"  ours {o:.6f}  enterprise {e:.6f}  diff {o - e:+.3e}  ({te:.1f} s)")
        d = np.array([r["ours"] - r["enterprise"] for r in rows])
        print(f"  max |diff| {np.max(np.abs(d)):.3e}, spread {np.ptp(d):.3e}")
        res[orf] = rows
    return res


def enterprise_params(pt: dict, orf: str) -> dict:
    out = {k: v for k, v in pt.items() if not k.startswith("_")}
    return out


if __name__ == "__main__":
    main()
