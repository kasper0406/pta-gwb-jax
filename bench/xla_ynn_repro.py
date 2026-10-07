"""Minimal standalone reproducer: XLA:CPU YNNPACK library fusion miscompiles a batched dot with a
broadcast operand that is fused into a multiply + reduce. No project imports; jax + numpy only.

    JAX_PLATFORMS=cpu python bench/xla_ynn_repro.py                       # shows wrong results
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type= python bench/xla_ynn_repro.py

Observed with jax/jaxlib 0.11.2 on x86-64 (AMD Zen5): the "bcast scalar" and "einsum bcast" rows
differ from the op-by-op (eager) result by O(1) relative error or ~1e102 (NaN for large inputs),
and the optimised HLO contains a ``__ynn_fusion``. Exit code 1 if any case mismatches.
Pattern in practice: the reverse-mode derivative of sum(f(E)) contains E @ broadcast(cotangent).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main():
    print(f"jax {jax.__version__}, backend {jax.default_backend()}, XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 60, 60))
    E = jnp.asarray(A @ A.transpose(0, 2, 1))  # batch of 5 SPD 60x60 matrices
    X = jnp.asarray(rng.standard_normal((5, 60, 60)))
    c = 0.37
    cases = {
        "bcast scalar": lambda E, X, c: jnp.sum((E @ jnp.full_like(E, c)) * E, axis=-1),
        "einsum bcast": lambda E, X, c: jnp.einsum("bij,bjk,bik->bi", E, jnp.full_like(E, c), E),
        "full matrix (control)": lambda E, X, c: jnp.sum((E @ X) * E, axis=-1),
        "bcast, unbatched (control)": lambda E, X, c: jnp.sum((E[0] @ jnp.full_like(E[0], c)) * E[0], axis=-1),
        "bcast + optimization_barrier": lambda E, X, c: jnp.sum(jax.lax.optimization_barrier(E @ jnp.full_like(E, c)) * E, axis=-1),
    }
    bad = False
    for name, f in cases.items():
        with jax.disable_jit():
            ref = np.asarray(f(E, X, c))
        comp = jax.jit(f).lower(E, X, c).compile()
        out = np.asarray(comp(E, X, c))
        err = float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))
        ok = bool(np.isfinite(err) and err < 1e-12)
        bad |= not ok
        print(f"{name:30s} __ynn_fusion in HLO: {'__ynn_fusion' in comp.as_text()!s:5s}  max rel err vs eager: {err:.2e}  {'ok' if ok else 'WRONG'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
