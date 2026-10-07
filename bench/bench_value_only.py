"""Value-only (no gradient) batched HD log-likelihood throughput -- the cost of reweighting
posterior draws (e.g. sky-scramble / phase-shift null Bayes factors from CURN draws).

    uv run --no-sync python bench/bench_value_only.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
from bench_backends import make, points
from common import env_info, like_options, save_json, timeit

out = {"env": env_info(), "rows": []}
for v in ("prod", "hh+levels"):
    like = make("hd", v)
    f = jax.jit(jax.vmap(like._logL))
    for B in (16, 64, 256):
        t = timeit(f, points(like.P, B), n=5)
        out["rows"].append({"variant": v, "options": like_options(like), "batch": B, "call_ms": t["median_ms"], "per_eval_ms": t["median_ms"] / B})
        print(v, B, f"{t['median_ms']:.1f} ms, {t['median_ms'] / B:.3f} ms/eval", flush=True)
save_json("value_only_hd_gpu.json", out)
