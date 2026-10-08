"""M3a validation driver (docs/M3A_VALIDATION.md). CPU only.

Subcommands (results in data/processed/m3a/results/*.json; the doc tables are generated from them):

  legs      ingest every leg of the selected configuration (+ InPTA DR1) with its published-analysis
            profile in fresh processes: G1 (TOA identity) and G2 (classified warnings) per leg
  tempo2    G3/G4 for the validation set: PINT (ours) vs tempo2 (libstempo, isolated env)
  ...       (see the other subcommands below)

Usage: JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/m3a_validate.py legs --jobs 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

RESULTS = ROOT / "data" / "processed" / "m3a" / "results"


def _dump(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    return p


def cmd_legs(args):
    from ptagwb.legs import load_legs
    from ptagwb.m3data import leg_files, quarantine

    datasets = args.datasets or ["mpta", "inpta_dr2", "epta_dr2new", "ppta_dr3_gh", "inpta_dr1", "ng15"]
    items = [(ds, lab) for ds in datasets for lab in leg_files(ds)]
    if args.only:
        items = [it for it in items if it[1] in args.only]
    t0 = time.time()
    print(f"ingesting {len(items)} legs with {args.jobs} workers (pin={args.pin}) ...", flush=True)
    res = load_legs(items, jobs=args.jobs, pin=args.pin, tag="published")
    q = quarantine()
    rows = []
    for r in res:
        m = r.meta
        rows.append({
            "dataset": r.dataset, "psr": r.label, "ok": r.ok, "error": r.error,
            "quarantined": (r.dataset, r.label) in q, "quarantine_class": q.get((r.dataset, r.label), {}).get("class"),
            **{k: m.get(k) for k in ("g1_ok", "g1_n_pint", "g1_n_records", "g1_max_dt_ns", "g1_max_dfreq_mhz",
                                     "g1_max_derr_us", "g1_n_obs_mismatch", "g1_n_flag_mismatch", "g1_n_padd_mismatch",
                                     "g1_flag_examples", "g2_unexplained", "ntoa", "ncol", "wrms_us", "binary",
                                     "clock_used", "ephem_used", "par_fixes", "clock_changes", "tim_counts",
                                     "multivalued_mask_flags", "frozen_after_load", "seconds", "profile")},
            "n_clock_excluded": len(m.get("clock_excluded", [])), "clock_coverage_notes": m.get("clock_coverage_notes"),
            "warnings": [(w["class"], w["count"], w["message"][:160]) for w in m.get("warnings", [])],
        })
    if args.only or args.datasets:  # partial rerun: merge into the full table
        prev = RESULTS / "legs_published.json"
        if prev.exists():
            old = json.loads(prev.read_text())["rows"]
            new_keys = {(r["dataset"], r["psr"]) for r in rows}
            rows = [r for r in old if (r["dataset"], r["psr"]) not in new_keys] + rows
    out = {"n_legs": len(rows), "seconds": round(time.time() - t0), "rows": rows}
    p = _dump("legs_published", out)
    ok = sum(1 for r in rows if r["ok"])
    g1 = sum(1 for r in rows if r.get("g1_ok"))
    g2 = sum(1 for r in rows if r["ok"] and not r.get("g2_unexplained"))
    print(f"loaded {ok}/{len(rows)}; G1 pass {g1}; G2 no-unexplained {g2}; -> {p}")
    for r in rows:
        if not r["ok"] or not r.get("g1_ok") or r.get("g2_unexplained"):
            print(f"  {r['dataset']:12s} {r['psr']:12s} ok={r['ok']} g1={r.get('g1_ok')} g2={r.get('g2_unexplained')} "
                  f"q={r['quarantine_class']} {r['error'][:120]}")
    c = Counter(w[0] for r in rows for w in r["warnings"])
    print("warning classes (legs):", dict(c))


def cmd_tempo2(args):
    """G3 (projected residuals) and G4 (weighted column space) vs tempo2 for the validation set."""
    from concurrent.futures import ThreadPoolExecutor

    import m3a_oracles as O

    from ptagwb.m3data import leg_files, pta_of, quarantine, validation_set
    from ptagwb.profiles import published_profile

    vs = validation_set()["legs"]
    if args.only:
        vs = [e for e in vs if e["psr"] in args.only]
    q = quarantine()

    def one(e):
        ds, psr = e["dataset"], e["psr"]
        row = {"dataset": ds, "psr": psr, "role": e["role"], "quarantined": (ds, psr) in q}
        try:
            par, tim = leg_files(ds)[psr]
            if ds == "ng15":  # tempo2 cannot read the PINT par; use the release's tempo2 version
                alt = sorted(par.parents[1].glob(f"alternate/tempo2/{psr}_tempo2_*.par"))
                par = alt[0]
                row["tempo2_par"] = str(par.name)
            prof = published_profile(pta_of(ds), ds)
            t0 = time.time()
            t2 = O.run_tempo2(par, tim, prof, f"{ds}/{psr}", design=True)
            row["t2_seconds"] = round(time.time() - t0, 1)
            try:
                ours, meta = O.load_cached_leg("published", ds, psr)
            except FileNotFoundError:
                row.update(note="no PINT leg (ingestion failed)", g3_ok=False, g4_ok=False)
                return row
            row.update(O.g3_g4(ours, t2))
            try:
                row["post_hoc_likelihood_impact"] = O.g3_likelihood_impact(ours, t2)
            except Exception as ex:  # noqa: BLE001
                row["post_hoc_likelihood_impact"] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        except Exception as ex:  # noqa: BLE001
            row.update(error=f"{type(ex).__name__}: {str(ex)[:400]}", g3_ok=False, g4_ok=False)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(one, vs))
    p = _dump("g3g4_tempo2" + ("_" + "_".join(args.only) if args.only else ""), {"rows": rows})
    for r in rows:
        print(f"{r['dataset']:12s} {r['psr']:12s} q={int(r['quarantined'])} G3={r.get('g3_ok')} "
              f"proj_rms={r.get('rms_diff_proj_ns', float('nan')):.3g}ns ({r.get('rms_diff_proj_sigma', float('nan')):.2g} sig) "
              f"G4={r.get('g4_ok')} sin={r.get('g4_max_sin', float('nan')):.2g} cols={r.get('ncol_pint')}/{r.get('ncol_tempo2')} "
              f"dlnL={r.get('post_hoc_likelihood_impact', {}).get('max_dshape_nats', float('nan')):.3g} "
              f"{r.get('error', r.get('note', ''))[:100]}")
    print("->", p)


def cmd_multileg(args):
    """Build option B (combined clock profile, per-leg models) and C (YA-v3 rewrite) for the
    validation multi-leg pulsars; saved under data/processed/m3a/multileg/."""
    from ptagwb.m3data import validation_set
    from ptagwb.multileg import build_multileg, save_multileg
    from ptagwb.profiles import COMBINED, YA_V3_CLOCKS

    out = ROOT / "data" / "processed" / "m3a" / "multileg"
    ml = validation_set()["multileg"]
    names = args.only or list(ml)
    summary = {}
    for name in names:
        legs = [tuple(x) for x in ml[name]]
        for timing in args.timing:
            refs = [None] if timing == "per_leg" else (args.refs or [None])
            for ref in refs:
                t0 = time.time()
                try:
                    clk = COMBINED if timing == "per_leg" else YA_V3_CLOCKS
                    force = {"clock": args.force_clock, "ephem": args.force_ephem} if args.force_clock else {}
                    if args.local_dm:
                        force["local_dm"] = True
                    suffix = ("-forced" if args.force_clock else "") + ("-localDM" if args.local_dm else "")
                    mp, res = build_multileg(name, legs, timing=timing, clock=clk, reference=ref, jobs=len(legs), pin=args.pin,
                                             force=force or None)
                    tag = "B" if timing == "per_leg" else f"C-ref{mp.reference}" + suffix
                    save_multileg(mp, out / f"{name}_{tag}.npz")
                    summary[f"{name}_{tag}"] = {"repr": repr(mp), "ntoa": int(mp.ntoa), "ncol": int(mp.Mmat.shape[1]),
                                                "shared": mp.meta.get("shared"), "seconds": round(time.time() - t0),
                                                "legs": {p: {"g1_ok": r.meta.get("g1_ok"), "g2": r.meta.get("g2_unexplained"),
                                                             "n_clock_excluded": len(r.meta.get("clock_excluded", [])),
                                                             "wrms_us": r.meta.get("wrms_us"),
                                                             "consistent": r.meta.get("consistent")} for p, r in res.items()}}
                    print(repr(mp), f"{time.time() - t0:.0f}s", flush=True)
                except Exception as ex:  # noqa: BLE001
                    summary[f"{name}_{timing}_{ref}"] = {"error": f"{type(ex).__name__}: {str(ex)[:1500]}"}
                    print("FAILED", name, timing, ref, str(ex)[:1500], flush=True)
    _dump("multileg" + ("_" + "_".join(names) if args.only else "") + ("_localDM" if args.local_dm else "")
          + ("_forced" if args.force_clock else ""), summary)


ML_DIR = ROOT / "data" / "processed" / "m3a" / "multileg"


def _load_ml(name, tag):
    from ptagwb.multileg import load_multileg

    return load_multileg(ML_DIR / f"{name}_{tag}.npz")


def cmd_g5(args):
    """G5 on real multi-leg systems (B and C): ours vs enterprise (HD + CURN values) and discovery
    (CURN values + JAX gradients), distinct grids and overlapping ECORR; pass: |d_ours - d_oracle|
    <= max(1e-6, 1e-9 |lnL|) for lnL differences between points, gradients <= 1e-8 max(|g|, 1)."""
    import jax
    import jax.numpy as jnp
    import m3a_oracles as O
    import numpy as np

    from ptagwb.combined import GeneralPTALikelihood, precompute_general

    out = {}
    for tag in ("C", "B"):
        # option C with the v0.9.3 shared-DM rule loses phase connection for J1022/J0437 (see G6/R5 in
        # docs/M3A_VALIDATION.md); the likelihood check uses the phase-connected local-DM C build
        mps = [_load_ml(n, tag if tag == "B" else f"C-ref{ref}-localDM") for n, ref in (("J1022+1001", "NG15"), ("J0437-4715", "PPTA"))]
        wns, models, nd, Tarr = O.g5_setup(mps, seed=7)
        terms = [precompute_general(p, w, m) for p, w, m in zip(mps, wns, models, strict=True)]
        hd, curn = GeneralPTALikelihood(terms, orf="hd"), GeneralPTALikelihood(terms, orf="curn")
        dl = O.discovery_g5(mps, nd, models, Tarr, 5)
        fd, gd = jax.jit(dl.logL), jax.jit(jax.grad(dl.logL))
        pta = O.enterprise_g5(mps, nd, models, Tarr, 5)
        pts = O.g5_params(mps, np.random.default_rng(11), 6)
        vals, gerr = [], 0.0
        for p in pts:
            named = O._named(mps, p)
            jp = {k: jnp.asarray(v) for k, v in p.items()}
            vals.append([float(curn.logL(jp)), float(fd({k: jnp.asarray(named[k]) for k in dl.logL.params})),
                         float(hd.logL(jp)), float(pta.get_lnlikelihood({k: named[k] for k in pta.param_names}))])
            go = jax.grad(curn._logL)(jp)
            gdd = gd({k: jnp.asarray(named[k]) for k in dl.logL.params})
            for i, m in enumerate(mps):
                for proc in ("rn", "dm"):
                    for par in ("log10_A", "gamma"):
                        a, b = float(np.asarray(go[f"{proc}_{par}"])[i]), float(gdd[f"{m.name}_{proc}_{par}"])
                        gerr = max(gerr, abs(a - b) / max(abs(b), 1.0))
            for par, key in (("log10_A", "gw_log10_A"), ("gamma", "gw_gamma")):
                a, b = float(go[par]), float(gdd[key])
                gerr = max(gerr, abs(a - b) / max(abs(b), 1.0))
        v = np.array(vals)
        d = v - v[0]
        tol = np.maximum(1e-6, 1e-9 * np.abs(v[:, 0]))
        out[tag] = {"systems": [repr(m) for m in mps], "ntoa": int(sum(m.ntoa for m in mps)),
                    "max_dshape_curn_vs_discovery": float(np.max(np.abs(d[:, 0] - d[:, 1]))),
                    "max_dshape_hd_vs_enterprise": float(np.max(np.abs(d[:, 2] - d[:, 3]))),
                    "max_grad_rel_err_curn_vs_discovery": gerr, "abs_lnL": float(np.abs(v[0, 0])),
                    "pass": bool(np.all(np.abs(d[:, 0] - d[:, 1]) <= tol) and np.all(np.abs(d[:, 2] - d[:, 3]) <= tol)
                                 and gerr <= 1e-8)}
        print(tag, json.dumps(out[tag], indent=1), flush=True)
    _dump("g5_multileg", out)


def cmd_g6(args):
    """G6 reference-model invariance for option C (J1909-3744): references NG15 vs PPTA vs EPTA with
    CLOCK/EPHEM held fixed (TT(BIPM2019)/DE440) and identical shared free-parameter sets."""
    import jax.numpy as jnp
    import m3a_oracles as O
    import numpy as np

    from ptagwb.combined import GeneralPTALikelihood, precompute_general

    base = _load_ml("J1909-3744", "C-refNG15")
    out = {"bound": {"g6_max_dshape_nats": 0.1, "g6_linearisation_whitened_norm": 0.1, "g6_colspace_sin": 1e-3}}
    for ref in args.refs:
        other = _load_ml("J1909-3744", f"C-ref{ref}-forced")
        # TOA alignment by (pta, name)
        key = {(a, b): i for i, (a, b) in enumerate(zip(other.flags["pta"], other.flags["name"], strict=True))}
        idx = np.array([key[(a, b)] for a, b in zip(base.flags["pta"], base.flags["name"], strict=True)])
        sig = base.toaerrs
        W = 1 / sig
        e = O.weighted_projector_complement(np.hstack([base.Mmat * W[:, None], other.Mmat[idx] * W[:, None]]))(
            (base.residuals - other.residuals[idx]) * W)
        lin = float(np.linalg.norm(e))
        sin = float(O.principal_sines(base.Mmat * W[:, None], other.Mmat[idx] * W[:, None])[0])
        wns, models, _, _ = O.g5_setup([base], seed=3)
        tb = precompute_general(base, wns[0], models[0])
        # same white noise / GP model on the other construction (same TOAs, same systems)
        from ptagwb.multileg import MultiLegPulsar  # noqa: F401
        oth = type("V", (), {})()
        for k in ("toas", "freqs", "toaerrs", "backend_flags", "residuals", "Mmat"):
            setattr(oth, k, np.asarray(getattr(other, k))[idx])
        oth.name, oth.pos, oth.pos_enterprise = base.name, base.pos, None
        to = precompute_general(oth, wns[0], models[0])
        la, lb = GeneralPTALikelihood([tb], orf="curn"), GeneralPTALikelihood([to], orf="curn")
        pts = O.g5_params([base], np.random.default_rng(5), 20)
        va = np.array([float(la.logL({k: jnp.asarray(v) for k, v in p.items()})) for p in pts])
        vb = np.array([float(lb.logL({k: jnp.asarray(v) for k, v in p.items()})) for p in pts])
        dshape = float(np.max(np.abs((va - va[0]) - (vb - vb[0]))))
        out[ref] = {"linearisation_whitened_norm": lin, "colspace_max_sin": sin, "max_dshape_nats": dshape,
                    "ncol": [int(base.Mmat.shape[1]), int(other.Mmat.shape[1])],
                    "pass": bool(dshape <= 0.1 and lin <= 0.1 and sin <= 1e-3)}
        print(ref, out[ref], flush=True)
    _dump("g6_reference_swap", out)


def cmd_g7(args):
    """G7 duplicate observations: every pulsar with >= 2 legs in the selected configuration (and the
    YA set), cross-leg same-site overlapping observations; plus LEAP vs single-telescope within EPTA."""
    from collections import defaultdict

    from ptagwb.m3data import SELECTED, YU_ALLEN, leg_files, pta_of
    from ptagwb.multileg import find_duplicates, site_of
    from ptagwb.timfile import read_tim

    sys.path.insert(0, str(ROOT / "scripts"))
    from m3_survey import canonical_names, parse_par

    out = {}
    for label, sets in (("selected", SELECTED), ("yu_allen", YU_ALLEN)):
        rows = []
        for ds in sets:
            for psr, (par, tim) in leg_files(ds).items():
                r = {"dataset": ds, "psr": psr, "par": par, "tim": tim}
                r.update(parse_par(par))
                rows.append(r)
        canon = canonical_names(rows)
        groups = defaultdict(list)
        for r in rows:
            groups[canon[(r["dataset"], r["psr"])]].append(r)
        cross, leap, n_multi = [], [], 0
        for j, legs in sorted(groups.items()):
            recs = {}
            for r in legs:
                rr, _ = read_tim(r["tim"])
                recs[f"{pta_of(r['dataset'])}/{r['dataset']}"] = rr
            if len(recs) >= 2:
                n_multi += 1
                for d in find_duplicates(recs):
                    cross.append({"pulsar": j, **d})
            for key, rr in recs.items():
                if key.startswith("EPTA"):
                    by = defaultdict(list)
                    for x in rr:
                        by["leap" if site_of(x.obs) == "leap" else site_of(x.obs)].append(x)
                    if "leap" in by:
                        for d in find_duplicates({k: v for k, v in by.items()}):
                            if "leap" in (d["a"], d["b"]):
                                leap.append({"pulsar": j, **d})
        out[label] = {"n_multi_leg_pulsars": n_multi, "cross_pta_same_site_pairs": cross,
                      "n_cross_pairs": sum(d["n_pairs"] for d in cross), "epta_leap_vs_single_telescope": leap,
                      "n_leap_pairs": sum(d["n_pairs"] for d in leap)}
        print(label, n_multi, "cross pairs", out[label]["n_cross_pairs"], "leap pairs", out[label]["n_leap_pairs"], flush=True)
    _dump("g7_duplicates", out)


def cmd_g8(args):
    import m3a_injection as I

    names = (("J1909-3744", "NG15"), ("J1022+1001", "NG15"), ("J0437-4715", "PPTA"))
    C = [_load_ml(n, f"C-ref{r}") for n, r in names]
    B = [_load_ml(n, "B") for n, _ in names]
    out = I.run(C, B, R=args.R)
    print(json.dumps(out, indent=1))
    _dump("g8_injections", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("legs")
    a.add_argument("--datasets", nargs="*")
    a.add_argument("--only", nargs="*")
    a.add_argument("--jobs", type=int, default=16)
    a.add_argument("--pin", action="store_true", help="first run of a profile: pin missing clock files")
    a.set_defaults(func=cmd_legs)
    a = sub.add_parser("tempo2")
    a.add_argument("--only", nargs="*")
    a.add_argument("--jobs", type=int, default=6)
    a.set_defaults(func=cmd_tempo2)
    a = sub.add_parser("multileg")
    a.add_argument("--only", nargs="*")
    a.add_argument("--timing", nargs="*", default=["shared", "per_leg"])
    a.add_argument("--refs", nargs="*", help="reference PTAs for option C (default: YA order)")
    a.add_argument("--force-clock", help="G6: hold CLOCK fixed (e.g. 'TT(BIPM2019)') while swapping the reference")
    a.add_argument("--force-ephem", default="DE440")
    a.add_argument("--local-dm", action="store_true", help="option C with per-leg DM (MetaPulsar-main configuration)")
    a.add_argument("--pin", action="store_true")
    a.set_defaults(func=cmd_multileg)
    for nm, fn in (("g5", cmd_g5), ("g7", cmd_g7)):
        a = sub.add_parser(nm)
        a.set_defaults(func=fn)
    a = sub.add_parser("g6")
    a.add_argument("--refs", nargs="*", default=["PPTA", "EPTA"])
    a.set_defaults(func=cmd_g6)
    a = sub.add_parser("g8")
    a.add_argument("--R", type=int, default=1000)
    a.set_defaults(func=cmd_g8)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
