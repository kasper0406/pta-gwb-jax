"""Cotangent patterns for compiled reducer VJPs (XLA:CPU YNNPACK regression; docs/PERF.md).

``run_cases(terms, T)`` returns {case: max scaled error of the compiled gradient vs the eager
(op-by-op, ``jax.disable_jit``) gradient} for the production reducer ``likelihood._reduce`` and
the fast reducer ``perf_likelihood.REDUCERS["hh"]``, on the vmapped per-pulsar stage with a
30-mode free spectrum (all modes common, CURN-like). Patterns:

* ``scalar``      loss = -0.5 sum(q + ld)                (E, d cotangents symbolic zero / zero)
* ``const``       ... + 1e-16 sum(E) + 1e-8 sum(d)       (constant nonzero, broadcast cotangents)
* ``zeros``       ... + 0 sum(E) + 0 sum(d)             (explicit zero cotangents)
* ``partial``     ... + 1e-16 sum(E[:, :28, :28]) + 1e-8 sum(d[:, :28])   (HD-like sparse block)
* ``runtime``     jax.vjp with random cotangents passed as runtime arguments of the compiled fn
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ptagwb import likelihood as L
from ptagwb.perf_likelihood import REDUCERS


def _phi(like, rng):
    P = like.P
    p = {"rn_log10_A": jnp.full(P, -13.5), "rn_gamma": jnp.full(P, 3.0),
         "log10_rho": jnp.asarray(rng.uniform(-8.0, -6.0, like.n_common))}
    nc2 = 2 * like.n_common
    return like.phi_rn(p["rn_log10_A"], p["rn_gamma"]).at[:, :nc2].add(like.phi_common(p)[None, :])


def run_cases(like, seed=0):
    rng = np.random.default_rng(seed)
    phi = _phi(like, rng)
    RA, c, s = like.RA, like.c, like.s_perp
    P, n = RA.shape[0], RA.shape[-1]
    cts = (jnp.asarray(rng.standard_normal(P)), jnp.asarray(rng.standard_normal(P)),
           jnp.asarray(1e-14 * rng.standard_normal((P, n, n))), jnp.asarray(1e-6 * rng.standard_normal((P, n))))
    out = {}
    for rname, red in (("prod", L._reduce), ("hh", REDUCERS["hh"])):
        def outs(r, red=red):
            return jax.vmap(red)(RA, c, s, r)

        def make_loss(wE, wd, block=None, outs=outs):
            def _loss(r):
                q, ld, E, d = outs(r)
                if block is not None:
                    E, d = E[:, :block, :block], d[:, :block]
                return -0.5 * jnp.sum(q + ld) + wE * jnp.sum(E) + wd * jnp.sum(d)

            return _loss

        losses = {
            "scalar": lambda r: -0.5 * jnp.sum(outs(r)[0] + outs(r)[1]),
            "const": make_loss(1e-16, 1e-8),
            "zeros": make_loss(0.0, 0.0),
            "partial": make_loss(1e-16, 1e-8, block=28),
        }
        for cname, loss in losses.items():
            g = jax.jit(jax.grad(loss))(phi)
            with jax.disable_jit():
                g0 = jax.grad(loss)(phi)
            out[f"{rname}/{cname}"] = _err(g, g0)

        def vjp_rt(r, ct):
            return jax.vjp(outs, r)[1](ct)[0]

        g = jax.jit(vjp_rt)(phi, cts)
        with jax.disable_jit():
            g0 = vjp_rt(phi, cts)
        out[f"{rname}/runtime"] = _err(g, g0)
    return out


def _err(g, g0):
    g, g0 = np.asarray(g), np.asarray(g0)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(g0))):
        return float("inf")
    return float(np.max(np.abs(g - g0) / np.maximum(np.abs(g0), 1.0)))
