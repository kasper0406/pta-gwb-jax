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
    extra = {"rows": []}
    rows = {(r["dataset"], r["psr"]): r for r in d["rows"]}
    for r in extra["rows"]:
        rows[(r["dataset"], r["psr"])] = r
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
    out = ["| system | points | CURN ours-arbiter | HD ours-arbiter | grad ours-arbiter (rel.) | CURN ours-discovery | HD ours-enterprise | discovery-arbiter | enterprise-arbiter (HD) | grad ours-discovery (info) | pre-fixed criterion (all oracles) | vs arbiter |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ("C", "B"):
        r = d.get(tag)
        if r is None:
            continue
        for lab in ("original", "held_out"):
            x = r[lab]
            out.append(f"| {tag}: {'; '.join(s.split('[')[0].replace('<MultiLegPulsar ', '') for s in r['systems'])}"
                       f" ({r['ntoa']} TOAs) | {lab} | {_f(x['dshape_curn_vs_arbiter'], '.2g')} | {_f(x.get('dshape_hd_vs_arbiter'), '.2g')} | "
                       f"{_f(x['grad_rel_vs_arbiter'], '.2g')} | {_f(x['dshape_curn_vs_discovery'], '.2g')} | {_f(x['dshape_hd_vs_enterprise'], '.2g')} | "
                       f"{_f(x.get('dshape_discovery_vs_arbiter'), '.2g')} | {_f(x.get('dshape_enterprise_vs_arbiter_hd'), '.2g')} | "
                       f"{_f(x['grad_rel_vs_discovery'], '.2g')} | {'pass' if x['pass'] else 'FAIL'} | {'pass' if x.get('pass_vs_arbiter') else 'FAIL'} |")
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
    d = _load("nharms")
    if d is None:
        return "(nharms results missing)"
    out = ["| leg | TOAs | ranks M7/M4/union | Shapiro diff rms [ns] | union rms [ns] | own rms [ns] (whitened norm) | own GW14 M7 / M4 | own GW30 M7 / M4 | single-leg CURN max abs(D-D0) / peak-to-peak [nats] | vs tempo2, 7 harm.: ns / sin | 4 harm.: ns / sin |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in d["legs"].items():
        if "error" in r:
            out.append(f"| {k} | error: {r['error'][:80]} |")
            continue
        rk, c = r["ranks"], r["curn_single_leg"]
        n7, n4 = r.get("vs_tempo2_nharms7", {}), r.get("vs_tempo2_nharms4", {})
        out.append(f"| {k} | {r['ntoa']} | {rk['M7']}/{rk['M4']}/{rk['union']} | {_f(r['shapiro_diff_rms_ns'])} | {_f(r['union_rms_ns'], '.2g')} | "
                   f"{_f(r['own_rms_ns'])} ({_f(r['own_whitened_norm'], '.2g')}) | {_f(r['own_gw14_M7_whitened_norm'], '.2g')} / {_f(r['own_gw14_M4_whitened_norm'], '.2g')} | "
                   f"{_f(r['own_gw30_M7_whitened_norm'], '.2g')} / {_f(r['own_gw30_M4_whitened_norm'], '.2g')} | "
                   f"{_f(c['max_abs_D_minus_D0'], '.2g')} / {_f(c['D_peak_to_peak'], '.2g')} | "
                   f"{_f(n7.get('rms_diff_proj_ns'))} / {_f(n7.get('g4_max_sin'), '.2g')} | {_f(n4.get('rms_diff_proj_ns'))} / {_f(n4.get('g4_max_sin'), '.2g')} |")
    return "\n".join(out)


def nharms_combined():
    d = _load("nharms")
    if d is None:
        return ""
    c = d["combined"]
    return (f"Combined system ({', '.join(c['pulsars'])}; legs stacked per pulsar as option B), same fixed noise: "
            f"CURN max abs(D - D0) {_f(c['curn']['max_abs_D_minus_D0'], '.3g')} nats, peak-to-peak {_f(c['curn']['D_peak_to_peak'], '.3g')}; "
            f"HD {_f(c['hd']['max_abs_D_minus_D0'], '.3g')} / {_f(c['hd']['D_peak_to_peak'], '.3g')} nats. "
            f"Both likelihoods peak at the scan boundary {c['curn']['argmax_lnL4']} (this toy noise model has no "
            "red-noise freedom), so these numbers are likelihood-shape diagnostics, not posterior shifts (Sec. 12b).")


def nharms_post():
    d = _load("nharms_posterior")
    if d is None:
        return "(nharms_posterior.json missing)"
    out = ["| array | ORF | quantity | 4 harm.: median [5%, 95%] | 7 harm.: median [5%, 95%] | dmedian / sigma68 | d5% / w90 | d95% / w90 | lnL max interior (4 / 7) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for an, a in d["arrays"].items():
        for orf in ("curn", "hd"):
            x = a[orf]
            for q in ("log10_A", "gamma"):
                s4, s7, sh = x["nharms4"][q], x["nharms7"][q], x["shift_7_vs_4"][q]
                out.append(f"| {an} | {orf.upper()} | {q} | {_f(s4['q50'])} [{_f(s4['q05'])}, {_f(s4['q95'])}] | "
                           f"{_f(s7['q50'])} [{_f(s7['q05'])}, {_f(s7['q95'])}] | {_f(sh['dmedian_over_sigma68'], '.2g')} | "
                           f"{_f(sh['dq05_over_w90'], '.2g')} | {_f(sh['dq95_over_w90'], '.2g')} | "
                           f"{x['nharms4']['max']['interior']} / {x['nharms7']['max']['interior']} |")
            g = x["gamma13_3"]
            s4, s7, sh = g["nharms4"], g["nharms7"], g["shift_7_vs_4"]["log10_A"]
            out.append(f"| {an} | {orf.upper()} | log10_A (gamma = 13/3) | {_f(s4['q50'])} [{_f(s4['q05'])}, {_f(s4['q95'])}] | "
                       f"{_f(s7['q50'])} [{_f(s7['q05'])}, {_f(s7['q95'])}] | {_f(sh['dmedian_over_sigma68'], '.2g')} | "
                       f"{_f(sh['dq05_over_w90'], '.2g')} | {_f(sh['dq95_over_w90'], '.2g')} | {s4['max']['interior']} / {s7['max']['interior']} |")
    return "\n".join(out)


def admissibility():
    rows = []
    for fn in ("multileg", "multileg_localDM", "multileg_J1909-3744_forced"):
        d = _load(fn)
        if not d:
            continue
        for k, v in d.items():
            if "admissibility" not in v or not v.get("admissibility"):
                continue
            for pta, a in v["admissibility"].items():
                rows.append(f"| {k} | {v.get('config', '')[:40]} | {pta} | {a['n_matched']} | {a['union_rank']}/{a['union_ncols']} | "
                            f"{_f(a['linearisation_rms_whitened'], '.3g')} | {_f(a['linearisation_rms_ns'], '.3g')} | "
                            f"{'yes' if a['admissible'] else 'NO'} |")
    return "\n".join(["| build | configuration | leg | TOAs matched | union rank / cols | linearisation rms (whitened) | rms [ns] | admissible |",
                      "|---|---|---|---|---|---|---|---|"] + rows)


def main():
    text = DOC.read_text()
    for name, fn in (("G3G4_TABLE", g3g4), ("G5_TABLE", g5), ("G6_TABLE", g6), ("G8_TABLE", g8),
                     ("NHARMS_TABLE", nharms), ("NHARMS_COMBINED", nharms_combined), ("NHARMS_POSTERIOR", nharms_post),
                     ("ADMISSIBILITY_TABLE", admissibility)):
        block = f"<!-- {name} -->\n{fn()}\n<!-- /{name} -->"
        pat = re.compile(rf"<!-- {name} -->.*?<!-- /{name} -->", re.DOTALL)
        if pat.search(text):
            text = pat.sub(lambda _m, b=block: b, text)
        else:
            text = text.replace(f"<!-- {name} -->", block)
    DOC.write_text(text)
    print("updated", DOC)


if __name__ == "__main__":
    main()
