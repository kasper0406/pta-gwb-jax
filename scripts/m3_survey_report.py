"""Markdown tables for docs/M3_SURVEY.md from data/processed/m3_survey/survey.json.

Usage:
    python scripts/m3_survey_report.py > data/processed/m3_survey/report.md

Run scripts/m3_survey.py --canonical first. Prints:
  * per-data-set summary (pulsars, TOAs by text / PINT, spans, systems, conventions),
  * PINT outcome per data set (as released and canonicalised) with failure categories,
  * consistency flags: PINT TOA count vs text count vs par NTOA; pre-fit wrms vs TRES,
  * the per-pulsar overlap matrix (TOA counts per PTA) for the main and the Yu & Allen sets.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "data" / "processed" / "m3_survey" / "survey.json"
ORDER = ["ng15", "epta_dr2new", "epta_dr2full", "epta_dr2new+", "ppta_dr3", "ppta_dr3_gh", "inpta_dr2",
         "inpta_dr1", "mpta"]
MAIN = ["ng15", "epta_dr2new", "ppta_dr3", "inpta_dr2", "mpta"]
YA = ["ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr1", "mpta"]


def mjd2yr(m: float) -> float:
    return 2000.0 + (m - 51544.5) / 365.25


def category(err: str) -> str:
    rules = [
        (r"could not convert string to float|invalid literal for int|list index out of range",
         "tim line misparsed (archive name / indented comment / continuation line)"),
        (r"Pulse numbers missing", "TRACK -2 in par but no pulse numbers in tim"),
        (r"Flags and flag-values should be given in pairs", "valueless tim flag (-cycle_post34)"),
        (r"FBn parameters are set but FB0", "PB + FBn (tempo2) without FB0"),
        (r"M2 cannot be negative", "negative M2/H3 (DDH)"),
        (r"DMX_ parameters do not match DMXR1_", "DMXR ranges without DMX value"),
        (r"unfittable parameters", "fit flag on DMXR range bound"),
        (r"has no attribute 'mjd'", "DMX present but no DMX_0001 (PINT template)"),
        (r"Timeout", "timeout"),
    ]
    for pat, lab in rules:
        if re.search(pat, err or ""):
            return lab
    return (err or "")[:80]


def main() -> None:
    rows = json.loads(SURVEY.read_text())
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    out: list[str] = []
    w = out.append

    # ------------------------------------------------------------ summary
    w("### Per-data-set summary\n")
    w("| data set | pulsars | with tim | TOAs (text) | TOAs (PINT, canonical) | first-last TOA | max span [yr] | "
      "median span [yr] | systems/psr (median) | observatories | radio freq [MHz] | wideband TOAs |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        sub = by.get(ds, [])
        if not sub:
            continue
        wt = [r for r in sub if r.get("tim_ntoa")]
        spans = sorted(r["span_yr"] for r in wt)
        nsys = sorted(max(r.get("n_group", 0), r.get("n_sys", 0), r.get("n_f", 0)) for r in wt)
        obs = Counter(o for r in wt for o in r.get("observatories", "").split(",") if o)
        pint_n = sum(r.get("canon_pint_ntoa") or 0 for r in wt)
        w(f"| {ds} | {len(sub)} | {len(wt)} | {sum(r['tim_ntoa'] for r in wt):,} | {pint_n:,} | "
          f"{mjd2yr(min(r['mjd_min'] for r in wt)):.2f}-{mjd2yr(max(r['mjd_max'] for r in wt)):.2f} | "
          f"{spans[-1]:.2f} | {spans[len(spans) // 2]:.2f} | {nsys[len(nsys) // 2]} | "
          f"{', '.join(sorted(obs))} | {min(r['fmin_mhz'] for r in wt):.0f}-{max(r['fmax_mhz'] for r in wt):.0f} | "
          f"{sum(r.get('wideband_toas', 0) for r in wt)} |")
    w("")

    # ------------------------------------------------------------ conventions
    w("### Par-file conventions (counts of pulsars)\n")
    w("| data set | UNITS | CLK | EPHEM | BINARY | NE_SW | DM_SERIES | DMX psr | JUMP lines | FD psr | "
      "noise keys in par | tempo2-only keywords |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        sub = by.get(ds, [])
        if not sub:
            continue
        def c(k, sub=sub):
            return dict(Counter(r.get(k) or "-" for r in sub))

        nk = Counter(k for r in sub for k in r["noise_keys"].split(",") if k)
        tk = Counter(k for r in sub for k in r["tempo2_constructs"].split(",") if k)
        w(f"| {ds} | {c('units')} | {c('clock')} | {c('ephem')} | {c('binary')} | {c('ne_sw')} | "
          f"{c('dm_series')} | {sum(r['n_dmx'] > 0 for r in sub)} | {sum(r['n_jump'] for r in sub)} | "
          f"{sum(r['n_fd'] > 0 for r in sub)} | {dict(nk) or '-'} | {dict(tk) or '-'} |")
    w("")

    # ------------------------------------------------------------ PINT outcome
    w("### PINT load outcome\n")
    w("| data set | as released: ok / fail | canonicalised: ok / fail | failure categories (as released) | "
      "canonical fixes applied (pulsars) |")
    w("|---|---|---|---|---|")
    for ds in ORDER:
        sub = [r for r in by.get(ds, []) if r.get("tim_ntoa")]
        if not sub:
            continue
        ok = sum(1 for r in sub if r.get("pint_ok"))
        cok = sum(1 for r in sub if r.get("canon_pint_ok"))
        cats = Counter(category(r.get("pint_error", "")) for r in sub if r.get("pint_ok") is False)
        fx = Counter(f.split("=")[0] for r in sub for f in (r.get("canon_fixes") or "").split(",") if f)
        w(f"| {ds} | {ok} / {len(sub) - ok} | {cok} / {len(sub) - cok} | "
          f"{'; '.join(f'{k} ({v})' for k, v in cats.most_common()) or '-'} | "
          f"{'; '.join(f'{k} ({v})' for k, v in fx.most_common()) or '-'} |")
    w("")
    bad = [r for r in rows if r.get("tim_ntoa") and r.get("canon_pint_ok") is False]
    if bad:
        w("Still failing after canonicalisation: " + ", ".join(
            f"{r['dataset']}/{r['psr']} ({r.get('canon_pint_error', '')[:100]})" for r in bad) + "\n")
    tim_missing = [f"{r['dataset']}/{r['psr']}" for r in rows if not r.get("tim_exists")]
    if tim_missing:
        w("Par without tim: " + ", ".join(tim_missing) + "\n")

    # ------------------------------------------------------------ consistency flags
    w("### Consistency flags (canonicalised PINT load)\n")
    w("TOA counts: PINT vs text count of the tim tree (INCLUDE/SKIP/END honoured) vs the par's NTOA "
      "(written by tempo2 when the par was last fitted; it can predate the released tim). "
      "Pre-fit wrms: PINT pre-fit weighted RMS with raw TOA errors vs the par's TRES "
      "(tempo2 post-fit wrms); a ratio > 1.5 flags a possible phase-connection or model problem.\n")
    w("| data set | psr | text | PINT | par NTOA | wrms [us] | TRES [us] | ratio | note |")
    w("|---|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        for r in by.get(ds, []):
            if not r.get("canon_pint_ok"):
                continue
            n_t, n_p = r.get("tim_ntoa"), r.get("canon_pint_ntoa")
            try:
                n_par = int(float(r.get("ntoa_par") or "nan"))
            except ValueError:
                n_par = None
            try:
                tres = float(r.get("tres_us") or "nan")
            except ValueError:
                tres = float("nan")
            wr = r.get("canon_pint_wrms_us", float("nan"))
            ratio = wr / tres if tres and not math.isnan(tres) else float("nan")
            notes = []
            if n_t != n_p:
                notes.append("PINT != text")
            if n_par is not None and n_par != n_p:
                notes.append("NTOA != PINT")
            if not math.isnan(ratio) and (ratio > 1.5 or ratio < 0.67):
                notes.append("wrms/TRES")
            if notes:
                w(f"| {ds} | {r['psr']} | {n_t} | {n_p} | {n_par} | {wr:.3f} | {tres:.3f} | {ratio:.2f} | "
                  f"{', '.join(notes)} |")
    w("")

    # ------------------------------------------------------------ overlap with TOA counts
    for label, sets in (("main set", MAIN), ("Yu & Allen set", YA)):
        cell = defaultdict(dict)
        for r in rows:
            if r["dataset"] in sets and r.get("tim_ntoa"):
                cell[r["jname"]][r["dataset"]] = r
        n = len(cell)
        mult = Counter(len(v) for v in cell.values())
        w(f"### Per-pulsar overlap, {label} ({', '.join(sets)})\n")
        w(f"{n} unique pulsars; number of PTAs per pulsar: {dict(sorted(mult.items()))}. "
          f"Cells: TOAs / span in yr.\n")
        w("| pulsar | " + " | ".join(sets) + " | n | combined span [yr] |")
        w("|---" * (len(sets) + 3) + "|")
        for j in sorted(cell, key=lambda j: (-len(cell[j]), j)):
            v = cell[j]
            lo = min(x["mjd_min"] for x in v.values())
            hi = max(x["mjd_max"] for x in v.values())
            w(f"| {j} | " + " | ".join(
                f"{v[s]['tim_ntoa']} / {v[s]['span_yr']:.1f}" if s in v else "" for s in sets) +
              f" | {len(v)} | {(hi - lo) / 365.25:.1f} |")
        w("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
