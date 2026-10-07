"""Q2/Q3: per-chain value+grad throughput vs backend, batch size (vmapped chains) and variant.

Each configuration evaluates value+grad of the log-likelihood at B distinct HD posterior draws
(vmapped; B = 1 uses the un-vmapped function) and reports the median wall time per call and per
chain-gradient. Backend is chosen by the environment (JAX_PLATFORMS=cpu for XLA:CPU; thread
settings via XLA_FLAGS / OPENBLAS_NUM_THREADS, recorded in the output).

    uv run --no-sync python bench/bench_backends.py --orf hd --batch 1 4 16 64 --variant prod aug
    JAX_PLATFORMS=cpu uv run --no-sync python bench/bench_backends.py --orf hd --batch 1 4 16 --tag _cpu16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from common import env_info, get_terms, save_json, timeit

from ptagwb.perf_likelihood import FastPTALikelihood
from ptagwb.sampling import load_run


def make(orf, variant, grad_precision="float64"):
    terms, T = get_terms()
    parts = variant.split("+")
    kw = {"reduce": parts[0]}
    if len(parts) > 1:
        kw["tri_inv"] = parts[1]
    return FastPTALikelihood(terms, T, n_modes=30, n_common=14, orf=orf, grad_precision=grad_precision, **kw)


def points(P, B, seed=0):
    r = load_run("hd_g433_14f")
    X = r["x"].reshape(-1, r["x"].shape[-1])
    idx = np.random.default_rng(seed).choice(len(X), B, replace=len(X) < B)
    X = X[idx]
    return {"rn_log10_A": jnp.asarray(X[:, :P]), "rn_gamma": jnp.asarray(X[:, P : 2 * P]),
            "log10_A": jnp.asarray(X[:, -1]), "gamma": jnp.full(B, 13 / 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orf", default="hd")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--variant", nargs="+", default=["prod", "aug"])
    ap.add_argument("--grad-precision", default="float64")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pmap", action="store_true",
                    help="split the batch over all local devices with pmap (CPU: run with "
                         "XLA_FLAGS=--xla_force_host_platform_device_count=16 OPENBLAS_NUM_THREADS=1)")
    args = ap.parse_args()
    out = {"env": env_info(), "orf": args.orf, "grad_precision": args.grad_precision,
           "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""), "rows": []}
    for v in args.variant:
        like = make(args.orf, v, args.grad_precision)
        for B in args.batch:
            p = points(like.P, B)
            if args.pmap:
                nd = jax.local_device_count()
                assert B % nd == 0, (B, nd)
                f = jax.pmap(jax.vmap(jax.value_and_grad(like._logL)))
                pp = {k: x.reshape(nd, B // nd, *x.shape[1:]) for k, x in p.items()}
                t = timeit(f, pp, n=max(5, args.n // 2))
                tv = None
            elif B == 1:
                p1 = {k: x[0] for k, x in p.items()}
                t = timeit(like.value_and_grad, p1, n=args.n)
                tv = timeit(like.logL, p1, n=args.n)
            else:
                t = timeit(like.value_and_grad_batched, p, n=max(5, args.n // max(1, B // 8)))
                tv = None
            row = {"variant": v, "batch": B, "pmap_devices": jax.local_device_count() if args.pmap else 1, "call_ms": t["median_ms"], "per_chain_grad_ms": t["median_ms"] / B,
                   "min_ms": t["min_ms"], "value_only_ms": tv["median_ms"] if tv else None}
            out["rows"].append(row)
            print(f"{args.orf} {v:16s} B={B:3d}  call {t['median_ms']:9.2f} ms  per-chain-grad {t['median_ms'] / B:8.3f} ms"
                  + (f"  (value only {tv['median_ms']:.2f} ms)" if tv else ""), flush=True)
    save_json(f"backends_{args.orf}_{jax.default_backend()}{args.tag}.json", out)


if __name__ == "__main__":
    main()
