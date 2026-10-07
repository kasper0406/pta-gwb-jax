"""jax.profiler trace of the vmapped value+grad (B chains) of a FastPTALikelihood variant.

    uv run --no-sync python bench/trace_batched.py hd hh+levels 16
    uv run --no-sync python bench/parse_trace.py bench/traces/batched_hd_hh+levels_16 10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
from bench_backends import make, points

orf, variant, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
gp = sys.argv[4] if len(sys.argv) > 4 else "float64"
like = make(orf, variant, gp)
p = points(like.P, B)
f = like.value_and_grad_batched if B > 1 else like.value_and_grad
if B == 1:
    p = {k: v[0] for k, v in p.items()}
for _ in range(3):
    jax.block_until_ready(f(p))
tdir = Path(__file__).resolve().parent / "traces" / f"batched_{orf}_{variant}_{gp}_{B}"
with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
    for _ in range(10):
        jax.block_until_ready(f(p))
print(tdir)
