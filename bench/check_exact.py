"""Q3 exactness on NG15: perf likelihood variants vs the production likelihood.

All 67 NG15 pulsars (enterprise positions), the full matrix of tests/test_perf_likelihood.py:
common spectrum {power law, free spectrum} x {14, 30} common modes x ORF {curn, hd, dipole,
monopole}, at interior draws, IRN / common prior corners and free-spectrum bound profiles,
single and vmapped (compiled). Reported per configuration:

    dv   = max |logL_variant - logL_prod|, both without the parameter-independent constant
    dg   = max_i |g_i - g_i^prod| / max(|g_i^prod|, 1)
    floor_dv, floor_dg = the same for production vs production with the pulsars permuted
           (a mathematically exact identity; production's own rounding floor)

Criterion (as the test): dv <= 1e-9 and dg <= 1e-8, or within 10x the production floor where
that floor already exceeds them (monopole / dipole). Run on the GPU and with JAX_PLATFORMS=cpu.

    uv run --no-sync python bench/check_exact.py [variant ...]      # default: hh+levels
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import jax
from common import env_info, get_terms, save_json
from test_perf_likelihood import run_matrix


def main():
    variants = [tuple(v.split("+")) for v in (sys.argv[1:] or ["hh+levels"])]
    terms, T = get_terms()
    res = run_matrix(terms, T, variants=variants)
    out = {"env": env_info(), "criteria": {"dv": 1e-9, "dg": 1e-8, "floor_factor": 10}, "results": {}}
    ok = True
    for k, (dv, dg, fv, fg) in res.items():
        passed = dv <= max(1e-9, 10 * fv) and dg <= max(1e-8, 10 * fg)
        ok &= passed
        out["results"][k] = {"dv": dv, "dg": dg, "floor_dv": fv, "floor_dg": fg, "within_tolerance": passed}
        print(f"{k:32s} dv {dv:.1e} dg {dg:.1e} | prod floor dv {fv:.1e} dg {fg:.1e}  {'ok' if passed else 'OUTSIDE'}", flush=True)
    out["all_within_tolerance"] = ok
    save_json(f"exact_ng15_{jax.default_backend()}.json", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
