"""Q3 exactness on NG15: perf likelihood variants vs the production likelihood.

All 67 NG15 pulsars (enterprise positions), the full matrix of tests/test_perf_likelihood.py:
common spectrum {power law, free spectrum} x {14, 30} common modes x ORF {curn, hd, dipole,
monopole}, at interior draws, IRN / common prior corners and free-spectrum bound profiles,
single and vmapped (compiled). Reported per configuration:

    dv   = max |logL_variant - logL_prod|, both without the parameter-independent constant
           (single vs single and batched vs batched, reported separately)
    dg   = max_i |g_i - g_i^prod| / max(|g_i^prod|, 1)
    floor_dv, floor_dg = the same for production vs exact identities of production (pulsars
           permuted; split_fraction 0.45 / 0.55; vmapped vs single kernels): its reproducibility floor

Criterion (as the test): fixed per-ORF budgets (tests/test_perf_likelihood.py BUDGET), applied
separately to fast-single vs prod-single, fast-batched vs prod-batched, and production's own floor.
Run on the GPU and with JAX_PLATFORMS=cpu.

    uv run --no-sync python bench/check_exact.py [variant ...]      # default: hh+levels
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import jax
from common import env_info, get_terms, save_json
from test_perf_likelihood import BUDGET, _finite, run_matrix


def main():
    variants = [tuple(v.split("+")) for v in (sys.argv[1:] or ["hh+levels"])]
    terms, T = get_terms()
    res = run_matrix(terms, T, variants=variants)
    out = {"env": env_info(), "budget": BUDGET, "results": {}}
    ok = True
    for k, (dvs, dgs, dvb, dgb, fv, fg, orf) in res.items():
        tv, tg = BUDGET[orf]
        passed = bool(_finite(dvs, dgs, dvb, dgb, fv, fg) and max(dvs, dvb, fv) <= tv and max(dgs, dgb, fg) <= tg)
        ok &= passed
        out["results"][k] = {"dv_single": dvs, "dg_single": dgs, "dv_batched": dvb, "dg_batched": dgb,
                             "floor_dv": fv, "floor_dg": fg, "budget": [tv, tg], "within_budget": passed}
        print(f"{k:30s} single dv {dvs:.1e} dg {dgs:.1e} | batched dv {dvb:.1e} dg {dgb:.1e} | prod floor dv {fv:.1e} "
              f"dg {fg:.1e} | budget {tv:.0e}/{tg:.0e} {'ok' if passed else 'OUTSIDE'}", flush=True)
    out["all_within_tolerance"] = ok
    save_json(f"exact_ng15_{jax.default_backend()}.json", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
