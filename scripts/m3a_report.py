"""Fill the generated tables of docs/M3A_VALIDATION.md from data/processed/m3a/results/*.json.

Each table sits between ``<!-- NAME -->`` and ``<!-- /NAME -->`` (the closing marker is added on
the first run); rerunning replaces the block. Usage: python scripts/m3a_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "m3a" / "results"
DOC = ROOT / "docs" / "M3A_VALIDATION.md"


def _load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, fmt=".3g"):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return "-"


def g3g4():
    d = _load("g3g4_tempo2")
    if d is None:
        return "(g3g4_tempo2.json missing)"
    extra = _load("g3g4_tempo2_J1327-0755") or {"rows": []}
    rows = {(r["dataset"], r["psr"]): r for r in d["rows"]}
    for r in extra["rows"]:
        rows[(r["dataset"], r["psr"])] = dict(r, note="re-run with the tempo2 NHARMS convention (Sec. 12)")
    out = ["| leg | role | q | TOAs | proj. rms [ns] | proj. rms [sigma] | max proj. [ns] | G3 | cols PINT/tempo2 | G4 max sin | G4 | post-hoc dlnL shape [nats] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (ds, psr), r in rows.items():
        imp = (r.get("post_hoc_likelihood_impact") or {}).get("max_dshape_nats")
        out.append(f"| {ds}/{psr} | {r['role']} | {'Q' if r['quarantined'] else ''} | {r.get('n_pint', '-')} | "
                   f"{_f(r.get('rms_diff_proj_ns'))} | {_f(r.get('rms_diff_proj_sigma'), '.2g')} | {_f(r.get('max_diff_proj_ns'))} | "
                   f"{'pass' if r.get('g3_ok') else 'FAIL'} | {r.get('ncol_pint', '-')}/{r.get('ncol_tempo2', '-')} | "
                   f"{_f(r.get('g4_max_sin'), '.2g')} | {'pass' if r.get('g4_ok') else 'FAIL'} | {_f(imp, '.2g')} |"
                   + (f" {r.get('error', r.get('note', ''))[:80]}" if r.get("error") or r.get("note") else ""))
    nq = [r for r in rows.values() if not r["quarantined"]]
    out.append("")
    out.append(f"Non-quarantined validation legs: G3 pass {sum(1 for r in nq if r.get('g3_ok'))}/{len(nq)}, "
               f"G4 pass {sum(1 for r in nq if r.get('g4_ok'))}/{len(nq)}.")
    return "\n".join(out)


def g5():
    d = _load("g5_multileg")
    if d is None:
        return "(g5_multileg.json missing)"
    out = ["| system | TOAs | abs lnL | shape diff CURN vs discovery [nats] | shape diff HD vs enterprise [nats] | max grad rel. err vs discovery | components > 1e-8 | ours CURN vs identity-ORF path | strict | arbitrated |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in d.items():
        out.append(f"| {tag}: {'; '.join(s.split('[')[0].replace('<MultiLegPulsar ', '') for s in r['systems'])} | {r['ntoa']} | "
                   f"{_f(r['abs_lnL'], '.4g')} | {_f(r['max_dshape_curn_vs_discovery'], '.2g')} | {_f(r['max_dshape_hd_vs_enterprise'], '.2g')} | "
                   f"{_f(r['max_grad_rel_err_curn_vs_discovery'], '.2g')} | {r.get('n_grad_components_beyond_1e-8', '-')}/{r.get('n_grad_components', '-')} | "
                   f"{_f(r.get('max_grad_rel_diff_ours_curn_vs_identity_orf_path'), '.2g')} | "
                   f"{'pass' if r.get('pass_strict', r.get('pass')) else 'FAIL'} | {'pass' if r.get('pass_arbitrated', r.get('pass')) else 'FAIL'} |")
    return "\n".join(out)


def g6():
    d = _load("g6_reference_swap")
    if d is None:
        return "(g6_reference_swap.json missing)"
    out = ["| swap | linearisation (whitened norm) | column-space max sin | max shape diff [nats] | pass |", "|---|---|---|---|---|"]
    for ref, r in d.items():
        if ref == "bound":
            continue
        out.append(f"| NG15 -> {ref} | {_f(r['linearisation_whitened_norm'])} | {_f(r['colspace_max_sin'])} | {_f(r['max_dshape_nats'])} | "
                   f"{'pass' if r['pass'] else 'FAIL'} |")
    return "\n".join(out)


def g8():
    d = _load("g8_injections")
    if d is None:
        return "(g8_injections.json missing)"
    out = ["| case | z(mean score) | Var(s)/I | tolerance | I(A^2) | pass |", "|---|---|---|---|---|---|"]
    for k, r in d["results"].items():
        out.append(f"| {k} | {_f(r['z_mean_score'], '.2f')} | {_f(r['var_over_info'], '.3f')} | 1 +- {_f(r['ratio_tol'], '.2f')} | "
                   f"{_f(r['info_A2'], '.3g')} | {'pass' if r['pass'] else 'FAIL'} |")
    out.append("")
    out.append(f"Information ratios at the injected amplitude: I_C/I_B = {_f(d['info_ratio_C_over_B_signal'], '.3f')}, "
               f"I_C/I_A = {_f(d['info_ratio_C_over_A_signal'], '.3f')}; at the null: I_C/I_B = "
               f"{_f(d['info_ratio_C_over_B_null'], '.3f')}, I_C/I_A = {_f(d['info_ratio_C_over_A_null'], '.3f')}.")
    return "\n".join(out)


def nharms():
    d, g = _load("nharms"), _load("nharms_g3")
    if d is None or g is None:
        return "(nharms results missing)"
    out = ["| leg | TOAs | Shapiro diff rms [ns] | union proj. [ns] | own proj. [ns] (whitened) | GW14 / GW30 (whitened) | dlnL shape full grid / posterior region [nats] | vs tempo2, 7 harm.: ns / sin | 4 harm.: ns / sin |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, r in d["legs"].items():
        h = g.get(k, {})
        n7, n4 = h.get("nharms7", {}), h.get("nharms4", {})
        out.append(f"| {k} | {r['ntoa']} | {_f(r['shapiro_diff_rms_ns'])} | {_f(r['post_fit_diff_rms_ns'], '.2g')} | "
                   f"{_f(h.get('own_projection_diff_rms_ns'))} ({_f(h.get('own_projection_diff_whitened_norm'), '.2g')}) | "
                   f"{_f(h.get('own_projection_gw14_whitened_norm'), '.2g')} / {_f(h.get('own_projection_gw30_whitened_norm'), '.2g')} | "
                   f"{_f(r['max_dlnL_shape_over_grid'], '.2g')} / {_f(r.get('max_dlnL_shape_posterior_region'), '.2g')} | "
                   f"{_f(n7.get('rms_diff_proj_ns'))} / {_f(n7.get('g4_max_sin'), '.2g')} | {_f(n4.get('rms_diff_proj_ns'))} / {_f(n4.get('g4_max_sin'), '.2g')} |")
    return "\n".join(out)


def nharms_combined():
    d = _load("nharms")
    if d is None:
        return ""
    c = d["combined"]
    return (f"Combined system ({', '.join(c['pulsars'])}; legs stacked per pulsar as option B): "
            f"CURN shape change max {_f(c['curn']['max_dlnL_shape_over_grid'], '.3g')} nats over the full grid, "
            f"{_f(c['curn'].get('max_dlnL_shape_posterior_region'), '.3g')} in the posterior region; HD "
            f"{_f(c['hd']['max_dlnL_shape_over_grid'], '.3g')} / {_f(c['hd'].get('max_dlnL_shape_posterior_region'), '.3g')} nats "
            f"(7 vs 4 harmonics, each with its own design matrix, fixed noise).")


def main():
    text = DOC.read_text()
    for name, fn in (("G3G4_TABLE", g3g4), ("G5_TABLE", g5), ("G6_TABLE", g6), ("G8_TABLE", g8),
                     ("NHARMS_TABLE", nharms), ("NHARMS_COMBINED", nharms_combined)):
        block = f"<!-- {name} -->\n{fn()}\n<!-- /{name} -->"
        pat = re.compile(rf"<!-- {name} -->.*?<!-- /{name} -->", re.S)
        if pat.search(text):
            text = pat.sub(lambda _m, b=block: b, text)
        else:
            text = text.replace(f"<!-- {name} -->", block)
    DOC.write_text(text)
    print("updated", DOC)


if __name__ == "__main__":
    main()
