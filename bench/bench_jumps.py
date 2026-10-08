"""Forward-only cost of the hybrid kernel's block-MH sweep (docs/FS_PILOT.md), GPU, B = 8 chains.

    uv run --no-sync python bench/bench_jumps.py --model hd   # or curn

Times (median of n calls after JIT): value+grad of the potential, value-only log-likelihood, and
one full jump sweep (36 blocks: 30 bins + 6 IRN pairs) including the potential/gradient refresh.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from common import env_info, get_terms, save_json

from ptagwb.hybrid import BlockProposals, HybridNUTS
from ptagwb.sampling import ModelSpec, Posterior, load_run, make_likelihood


def med(f, *a, n=10):
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hd")
    ap.add_argument("--B", type=int, default=8)
    args = ap.parse_args()
    run, prop_file = {"hd": ("hd_fs30", "configs/m2/proposals/hd_fs30_v1.json"),
                      "curn": ("curn_fs30", "configs/m2/proposals/curn_fs30_v1.json")}[args.model]
    gp = "mixed" if args.model == "hd" else "float64"
    spec = ModelSpec(orf=args.model, common="freespec", n_common=30, position="enterprise", grad_precision=gp)
    terms, T = get_terms()
    post = Posterior(make_likelihood(terms, T, spec, "fast"), spec)
    prop = BlockProposals.from_json(prop_file, post.names)
    Z = load_run(run)["z"].reshape(-1, post.transform.dim)
    z = jnp.asarray(Z[np.random.default_rng(0).choice(len(Z), args.B)])
    vg = jax.jit(jax.vmap(jax.value_and_grad(post.potential_fn)))
    vo = jax.jit(jax.vmap(post.logL_x_raw))
    k = HybridNUTS(post, prop, sweeps=1)
    pe, g = vg(z)
    keys = jax.random.split(jax.random.PRNGKey(0), args.B)
    jump = jax.jit(jax.vmap(k._jump1))
    out = {"env": env_info(), "model": args.model, "B": args.B, "n_blocks": prop.n_blocks,
           "value_grad_ms": med(vg, z), "value_only_ms": med(vo, post.transform.to_constrained(z)),
           "sweep_plus_refresh_ms": med(jump, keys, z, pe, g, n=5)}
    _, _, _, acc = jump(keys, z, pe, g)
    out["accept_rate_at_posterior_draws"] = float(np.mean(np.asarray(acc)))
    print(out)
    save_json(f"jumps_{args.model}_B{args.B}.json", out)


if __name__ == "__main__":
    main()
