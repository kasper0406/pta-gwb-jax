"""Shared helpers for the performance study (bench/*; docs/PERF.md).

* ``get_terms``: stage-1 precompute, cached on disk (bench/cache/, git-ignored) so that benchmark
  processes start in ~1 s instead of PINT load + precompute (~30-80 s).
* ``build_like`` / ``build_post``: the *production* likelihood/posterior objects, exactly as
  scripts/m2_run.py builds them (enterprise positions, 30 IRN modes, 14 common modes).
* ``test_points``: posterior draws (M2 runs), prior corners and the reviewers' points of
  tests/test_corners.py, and random prior draws -- for exactness checks of perf variants.
* ``timeit``: median wall time of a jitted call with ``block_until_ready``.
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
CACHE = BENCH / "cache"
RESULTS = BENCH / "results"
sys.path.insert(0, str(ROOT / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ptagwb.likelihood import PTALikelihood
from ptagwb.sampling import ModelSpec, Posterior, load_run, unpack

GAMMA = 13.0 / 3.0


def get_terms(position: str = "enterprise", n_modes: int = 30):
    """(terms, T) for the 67 GWB pulsars; cached pickle of the stage-1 contractions."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"terms_{position}_{n_modes}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    from ptagwb.data import get_tspan, load_pulsars
    from ptagwb.likelihood import precompute
    from ptagwb.noise import load_noise_dict

    psrs = load_pulsars(verbose=False)
    T = get_tspan(psrs)
    terms = precompute(psrs, load_noise_dict(), T, n_modes=n_modes, position=position)
    with open(p, "wb") as fh:
        pickle.dump((terms, T), fh)
    return terms, T


def build_like(orf: str = "hd", common: str = "powerlaw", n_common: int = 14, position: str = "enterprise", **kw):
    terms, T = get_terms(position)
    return PTALikelihood(terms, T, n_modes=30, n_common=n_common, orf=orf, common=common, **kw)


def build_post(orf: str = "hd", gamma: float | None = GAMMA, common: str = "powerlaw", n_common: int = 14, like=None, **kw):
    spec = ModelSpec(orf=orf, gamma=gamma, common=common, n_common=n_common, position="enterprise")
    like = like if like is not None else build_like(orf, common, n_common, **kw)
    return Posterior(like, spec)


CORNERS = list(itertools.product((-20.0, -11.0), (0.0, 7.0), (-18.0, -11.0), (0.0, 7.0)))
REVIEW_POINTS = [(-11.1, 6.9, -11.1, 6.9), (-11.0, 7.0, -11.0, 7.0)]


def _named_point(P, la, g, lac, gc):
    return {"rn_log10_A": np.full(P, la), "rn_gamma": np.full(P, g), "log10_A": np.asarray(lac), "gamma": np.asarray(gc)}


def test_points(P: int, n_post: int = 8, n_prior: int = 8, run: str = "hd_g433_14f", seed: int = 0):
    """List of (label, params-dict) for the power-law likelihood: posterior draws of ``run``,
    the 16 prior corners + 2 reviewer points of tests/test_corners.py, random prior draws."""
    rng = np.random.default_rng(seed)
    pts = []
    try:
        r = load_run(run)
        X = r["x"].reshape(-1, r["x"].shape[-1])
        names = r["names"]
        for i in rng.choice(len(X), n_post, replace=False):
            x = X[i]
            d = {"rn_log10_A": x[:P], "rn_gamma": x[P : 2 * P]}
            if "gw_gamma" in names:
                d["gamma"], d["log10_A"] = x[names.index("gw_gamma")], x[names.index("gw_log10_A")]
            else:
                d["gamma"], d["log10_A"] = np.asarray(GAMMA), x[names.index("gw_log10_A")]
            pts.append((f"post{i}", d))
    except FileNotFoundError:
        pass
    for c in CORNERS + REVIEW_POINTS:
        pts.append((f"corner{c}", _named_point(P, *c)))
    for j in range(n_prior):
        pts.append(
            (
                f"prior{j}",
                {
                    "rn_log10_A": rng.uniform(-20, -11, P),
                    "rn_gamma": rng.uniform(0, 7, P),
                    "log10_A": np.asarray(rng.uniform(-18, -11)),
                    "gamma": np.asarray(rng.uniform(0, 7)),
                },
            )
        )
    return [(lab, {k: jnp.asarray(v, dtype=jnp.float64) for k, v in d.items()}) for lab, d in pts]


def timeit(fn, *args, n: int = 30, warmup: int = 3) -> dict:
    """Median/min wall time [ms] of fn(*args) incl. device sync."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    return {"median_ms": float(np.median(ts)), "min_ms": float(ts.min()), "p90_ms": float(np.percentile(ts, 90)), "n": n}


def env_info() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=ROOT).stdout.strip()
        except Exception:
            return ""

    return {
        "git_sha": sh("git", "rev-parse", "HEAD"),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "host": os.uname().nodename,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, default=float))
    return p


__all__ = ["GAMMA", "build_like", "build_post", "env_info", "get_terms", "save_json", "test_points", "timeit", "unpack"]
