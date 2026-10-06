"""Verify that JAX sees the GPU and that float64 linear algebra runs on it.

Usage: uv run python scripts/check_gpu.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxlib


def main() -> None:
    print(f"jax {jax.__version__}  jaxlib {jaxlib.__version__}")
    print("default backend:", jax.default_backend())
    devs = jax.devices()
    print("devices:", devs)
    gpus = [d for d in devs if d.platform == "gpu"]
    if not gpus:
        raise SystemExit("FAIL: no GPU device visible to JAX")
    gpu = gpus[0]
    print("device kind:", getattr(gpu, "device_kind", "?"))

    n = 2048
    key = jax.random.PRNGKey(0)
    a = jax.device_put(jax.random.normal(key, (n, n), dtype=jnp.float64), gpu)

    @jax.jit
    def f(a):
        m = a @ a.T + n * jnp.eye(n, dtype=a.dtype)
        L = jnp.linalg.cholesky(m)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        resid = jnp.max(jnp.abs(L @ L.T - m)) / jnp.max(jnp.abs(m))
        return m, L, logdet, resid

    _m, L, logdet, resid = f(a)
    L.block_until_ready()
    assert L.dtype == jnp.float64, L.dtype
    placed = {d.platform for d in L.devices()}
    print(
        f"matmul+cholesky n={n}: dtype={L.dtype} on={placed} logdet={float(logdet):.6f} "
        f"rel.recon.err={float(resid):.2e}"
    )
    if placed != {"gpu"}:
        raise SystemExit("FAIL: result not on GPU")
    if not float(resid) < 1e-12:
        raise SystemExit("FAIL: float64 Cholesky reconstruction error too large")
    print("OK: float64 matmul + Cholesky ran on GPU")


if __name__ == "__main__":
    main()
