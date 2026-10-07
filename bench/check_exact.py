"""Q3 exactness: perf likelihood variants vs the production likelihood.

For HD and CURN (14-mode power law, enterprise positions) and every variant, evaluates value and
gradient at 8 HD posterior draws, the 16 prior corners + 2 reviewer points of
tests/test_corners.py, and 8 random prior draws, and reports

    dv   = |logL_variant - logL_prod|, both without the parameter-independent constant
           (criterion <= 1e-9; with the constant the logL ulp is ~1e-9, so dv would be 0)
    dg   = max_i |g_i - g_i^prod| / max(|g_i^prod|, 1)      (criterion <= 1e-8)
    dgn  = max_i |g_i - g_i^prod| / max_i |g_i^prod|         (norm-wise, informational)

also through the vmapped (batched) value+grad. Writes results/exact_<backend>.json.

    uv run --no-sync python bench/check_exact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from common import build_like, env_info, get_terms, save_json, test_points

from ptagwb.perf_likelihood import FastPTALikelihood

VARIANTS = {
    "hh+recursive": {"reduce": "hh", "tri_inv": "recursive"},
    "prod+levels": {"reduce": "prod", "tri_inv": "levels"},
    "hh+levels": {"reduce": "hh", "tri_inv": "levels"},
}


def flat(g):
    return np.concatenate([np.atleast_1d(np.asarray(g[k])) for k in sorted(g)])


def strip_const(like):
    """Same object with const_total = 0 (re-jitted), so that dv measures the parameter-dependent
    part: the full logL ~ -8e6 has an ulp of 1.9e-9, which would hide differences < 1e-9."""
    like.const_total = 0.0
    like.logL = jax.jit(like._logL)
    like.value_and_grad = jax.jit(jax.value_and_grad(like._logL))
    if hasattr(like, "value_and_grad_batched"):
        like.value_and_grad_batched = jax.jit(jax.vmap(jax.value_and_grad(like._logL)))
    return like


def compare(prod, fast, pts, batched: bool):
    rows = []
    if batched:
        stacked = {k: jnp.stack([p[k] for _, p in pts]) for k in pts[0][1]}
        vb, gb = fast.value_and_grad_batched(stacked)
    for i, (lab, p) in enumerate(pts):
        v0, g0 = prod.value_and_grad(p)
        if batched:
            v1, g1 = vb[i], {k: gb[k][i] for k in gb}
        else:
            v1, g1 = fast.value_and_grad(p)
        a, b = flat(g0), flat(g1)
        rows.append(
            {
                "point": lab,
                "logL_var": float(v0),
                "dv": abs(float(v1) - float(v0)),
                "dg": float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))),
                "dgn": float(np.max(np.abs(a - b)) / np.max(np.abs(a))),
                "finite": bool(np.isfinite(float(v1)) and np.all(np.isfinite(b))),
            }
        )
    return rows


def main():
    terms, T = get_terms()
    out = {"env": env_info(), "criteria": {"dv": 1e-9, "dg": 1e-8}, "results": {}}
    ok = True
    for orf in ("hd", "curn"):
        prod = strip_const(build_like(orf))
        pts = test_points(prod.P, n_post=8, n_prior=8)
        for vname, kw in VARIANTS.items():
            if orf == "curn" and kw["tri_inv"] != "recursive":
                continue
            fast = strip_const(FastPTALikelihood(terms, T, n_modes=30, n_common=14, orf=orf, **kw))
            for batched in (False, True):
                rows = compare(prod, fast, pts, batched)
                key = f"{orf}/{vname}/{'batched' if batched else 'single'}"
                worst = {m: max(r[m] for r in rows) for m in ("dv", "dg", "dgn")}
                passed = worst["dv"] <= 1e-9 and worst["dg"] <= 1e-8 and all(r["finite"] for r in rows)
                ok &= passed
                out["results"][key] = {"worst": worst, "pass": passed, "rows": rows}
                wp = {m: max(rows, key=lambda r: r[m])["point"] for m in ("dv", "dg")}
                print(f"{key:35s} dv {worst['dv']:.2e} dg {worst['dg']:.2e} dgn {worst['dgn']:.2e}  {'PASS' if passed else 'FAIL'}  worst at {wp}")
    out["all_pass"] = ok
    save_json(f"exact_{jax.default_backend()}.json", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
