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
                    mp, res = build_multileg(name, legs, timing=timing, clock=clk, reference=ref, jobs=len(legs), pin=args.pin)
                    tag = "B" if timing == "per_leg" else f"C-ref{mp.reference}"
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
    _dump("multileg" + ("_" + "_".join(names) if args.only else ""), summary)


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
    a.add_argument("--pin", action="store_true")
    a.set_defaults(func=cmd_multileg)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
