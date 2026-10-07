"""Impact of the XLA:CPU YNNPACK miscompilation on the PRODUCTION likelihood (docs/PERF.md).

For production ``PTALikelihood`` on all 67 NG15 pulsars, compares the *compiled* value+gradient
with the *eager* (op-by-op, ``jax.disable_jit``) one -- and on request with central finite
differences of the compiled value -- for CURN / HD / dipole / monopole x {power law 14, free
spectrum 30}, at interior and corner points; also the reducer cotangent patterns of
tests/ynn_cases.py on 5 and 67 pulsars, and whether ``__ynn_fusion`` occurs in the HLO.

    JAX_PLATFORMS=cpu PTAGWB_KEEP_XLA_CPU_YNN_FUSION=1 python bench/ynn_impact.py --tag _cpu_ynn_on
    JAX_PLATFORMS=cpu python bench/ynn_impact.py --tag _cpu_workaround
    python bench/ynn_impact.py --fd --tag _gpu                      # GPU (default backend)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import jax
import jax.numpy as jnp
import numpy as np
import ynn_cases
from common import env_info, get_terms, save_json
from test_perf_likelihood import _flat, _orf, points

from ptagwb.config import xla_cpu_ynn_fusion_active
from ptagwb.likelihood import PTALikelihood


def fd_check(like, p, rng, k=6, h=1e-5):
    """Central differences of the compiled value for k random parameter components."""
    keys = sorted(p)
    flat = np.concatenate([np.atleast_1d(np.asarray(p[q], dtype=float)) for q in keys])
    sizes = [np.atleast_1d(np.asarray(p[q])).size for q in keys]

    def unflat(x):
        out, i = {}, 0
        for q, s in zip(keys, sizes, strict=True):
            out[q] = jnp.asarray(x[i : i + s] if np.ndim(p[q]) else x[i])
            i += s
        return out

    g = _flat(like.value_and_grad(p)[1])
    worst = 0.0
    for j in rng.choice(len(flat), k, replace=False):
        xp, xm = flat.copy(), flat.copy()
        xp[j] += h
        xm[j] -= h
        fd = (float(like.logL(unflat(xp))) - float(like.logL(unflat(xm)))) / (2 * h)
        worst = max(worst, abs(fd - g[j]) / max(abs(g[j]), 1.0))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    terms, T = get_terms()
    out = {"env": env_info(), "ynn_fusion_active": xla_cpu_ynn_fusion_active(), "likelihood": {}, "reducer_cases": {}}
    rng = np.random.default_rng(0)
    for (common, nc), orf in itertools.product([("powerlaw", 14), ("freespec", 30)], ["curn", "hd", "dipole", "monopole"]):
        like = PTALikelihood(terms, T, n_modes=30, n_common=nc, orf=_orf(orf, terms), common=common)
        pts = points(like.P, common, nc, n_interior=2)
        pts = pts[:2] + pts[2::7]  # 2 interior + a spread of corner / bound-profile points
        hlo = jax.jit(jax.value_and_grad(like._logL)).lower(pts[0]).compile().as_text()
        dv = dg = 0.0
        for p in pts:
            v1, g1 = like.value_and_grad(p)
            with jax.disable_jit():
                v0, g0 = jax.value_and_grad(like._logL)(p)
            a, b = _flat(g0), _flat(g1)
            assert np.isfinite(float(v0)) and np.all(np.isfinite(a)), "non-finite eager reference"
            dv = max(dv, abs(float(v1) - float(v0)))
            dg = max(dg, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))) if np.all(np.isfinite(b)) else np.inf)
        row = {"n_points": len(pts), "dv_compiled_vs_eager": dv, "dg_compiled_vs_eager": dg,
               "ynn_fusion_in_hlo": "__ynn_fusion" in hlo}
        if args.fd:
            row["dg_fd_vs_compiled"] = max(fd_check(like, p, rng) for p in pts[:2])
        out["likelihood"][f"{common}{nc}/{orf}"] = row
        print(f"{common}{nc}/{orf:8s} " + " ".join(f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    for npsr in (5, 67):
        like = PTALikelihood(terms[:npsr], T, n_modes=30, n_common=30, orf="curn", common="freespec")
        out["reducer_cases"][str(npsr)] = r = ynn_cases.run_cases(like)
        print(f"reducer cotangent cases, {npsr} pulsars: " + ", ".join(f"{k} {v:.1e}" for k, v in r.items()), flush=True)
    save_json(f"ynn_impact{args.tag}.json", out)


if __name__ == "__main__":
    main()
