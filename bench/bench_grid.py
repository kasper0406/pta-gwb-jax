"""Cost of the complete hybrid update with conditional-grid moves (docs/FS_PILOT.md), GPU, B = 8.

    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python bench/bench_grid.py --config configs/m2/hd_fs30_grid_pilot.json

Times (median after JIT): value+grad of the potential (one leapfrog step), the block sweep alone, the
conditional-grid moves alone (cache rebuilds, grids, interpolation, MH incl. one production logL per
move), and the complete post-NUTS update (sweep + grid moves + potential/gradient refresh).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from common import env_info, get_terms, save_json

from ptagwb.hybrid import BlockProposals, HybridNUTS, make_grid_moves, make_sweep
from ptagwb.sampling import ModelSpec, Posterior, RunConfig, load_run, make_likelihood


def med(f, *a, n=5):
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = RunConfig(**json.loads(Path(args.config).read_text()))
    spec = ModelSpec(**cfg.model)
    terms, T = get_terms()
    post = Posterior(make_likelihood(terms, T, spec, cfg.likelihood_impl), spec)
    prop = BlockProposals.from_json(cfg.jumps, post.names) if cfg.jumps else None
    Z = load_run("hd_fs30_hybrid_pilot")["z"].reshape(-1, post.transform.dim)
    B = cfg.num_chains
    z = jnp.asarray(Z[np.random.default_rng(0).choice(len(Z), B)])
    tr = post.transform
    x = tr.to_constrained(z)
    keys = jax.random.split(jax.random.PRNGKey(0), B)
    vg = jax.jit(jax.vmap(jax.value_and_grad(post.potential_fn)))
    pe, g = vg(z)
    ll = jax.jit(jax.vmap(post.logL_x_raw))(x)
    out = {"env": env_info(), "config": args.config, "B": B, "grid_bins": cfg.grid_bins, "grid_kw": cfg.grid_kw,
           "value_grad_ms": med(vg, z)}
    if prop is not None:
        sw = jax.jit(jax.vmap(make_sweep(prop, tr, post.logL_x_raw, cfg.jump_sweeps)))
        out["sweep_ms"] = med(sw, keys, z, ll)
    gm = jax.jit(jax.vmap(make_grid_moves(post, list(cfg.grid_bins), proposals=prop, **cfg.grid_kw)))
    out["grid_moves_ms"] = med(gm, keys, x, ll)
    _, _, acc = gm(keys, x, ll)
    out["grid_accept_at_draws"] = np.asarray(acc).mean(axis=0).tolist()
    k = HybridNUTS(post, prop, sweeps=cfg.jump_sweeps, grid_bins=list(cfg.grid_bins), grid_kw=dict(cfg.grid_kw))
    jump = jax.jit(jax.vmap(k._jump1))
    out["complete_update_ms"] = med(jump, keys, z, pe, g)
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "env"}, indent=1))
    save_json(f"grid_update_{Path(args.config).stem}.json", out)


if __name__ == "__main__":
    main()
