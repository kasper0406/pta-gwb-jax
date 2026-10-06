"""Run one M2 NUTS configuration: ``uv run --no-sync python scripts/m2_run.py configs/m2/<name>.json``.

Writes ``runs/<name>/{meta.json,samples.npz}`` (git-ignored) with the config, git SHA and
dirty-file list as a sidecar. ``--set key=value`` overrides config fields for pilots (the
override is recorded in the sidecar).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

from ptagwb.data import get_tspan, load_pulsars
from ptagwb.likelihood import PTALikelihood, precompute
from ptagwb.noise import load_noise_dict
from ptagwb.sampling import Posterior, RunConfig, run_nuts


def build_posterior(spec):
    psrs = load_pulsars(verbose=False)
    T = get_tspan(psrs)
    terms = precompute(psrs, load_noise_dict(), T, n_modes=spec.n_modes, position=spec.position)
    like = PTALikelihood(
        terms,
        T,
        n_modes=spec.n_modes,
        n_common=spec.n_common,
        orf=spec.orf,
        common=spec.common,
        grad_precision=spec.grad_precision,
    )
    return Posterior(like, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set num_samples=200")
    args = ap.parse_args()
    with open(args.config) as fh:
        d = json.load(fh)
    for kv in args.set:
        k, v = kv.split("=", 1)
        d[k] = json.loads(v) if v[:1] in "0123456789-[{tfn\"" else v
    if args.set:
        d["notes"] = (d.get("notes", "") + f" [overrides: {args.set}]").strip()
    cfg = RunConfig(**d)
    t0 = time.time()
    post = build_posterior(cfg.spec)
    print(f"[{cfg.name}] D = {post.transform.dim}, setup {time.time() - t0:.0f} s", flush=True)
    run_nuts(cfg, post, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main())
