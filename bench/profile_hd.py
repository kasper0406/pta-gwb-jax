"""Q1: where does the HD (14-mode power law) value+grad time go?

Times the production ``PTALikelihood.value_and_grad`` and its stages as separately jitted
functions at an HD posterior draw, records XLA cost-analysis FLOPs, and (``--nsys-loop N``) runs
N value+grad calls for an Nsight Systems kernel-level trace (see docs/PERF.md for the command).

    uv run --no-sync python bench/profile_hd.py                    # stage timings -> results/profile_<backend>.json
    nsys profile -o bench/traces/hd --force-overwrite true \
        uv run --no-sync python bench/profile_hd.py --nsys-loop 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from common import build_like, env_info, save_json, test_points, timeit

from ptagwb import likelihood as L


def flops(fn, *args):
    try:
        ca = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(ca, list):
            ca = ca[0]
        return float(ca.get("flops", float("nan")))
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys-loop", type=int, default=0)
    ap.add_argument("--jax-trace", type=int, default=0, help="N value+grad calls under jax.profiler -> bench/traces/")
    ap.add_argument("--orf", default="hd")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    like = build_like(args.orf)
    pts = test_points(like.P, n_post=1, n_prior=0)
    p = pts[0][1]

    vg = like.value_and_grad
    if args.nsys_loop:
        jax.block_until_ready(vg(p))
        jax.block_until_ready(vg(p))
        for _ in range(args.nsys_loop):
            jax.block_until_ready(vg(p))
        return

    if args.jax_trace:
        jax.block_until_ready(vg(p))
        tdir = Path(__file__).resolve().parent / "traces" / f"jax_{args.orf}_{jax.default_backend()}{args.tag}"
        with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
            for _ in range(args.jax_trace):
                jax.block_until_ready(vg(p))
        print("trace in", tdir)
        return

    out = {"env": env_info(), "orf": args.orf, "point": pts[0][0], "stages": {}}
    S = out["stages"]
    S["value"] = timeit(like.logL, p)
    S["value_and_grad"] = timeit(vg, p)
    out["flops_value_and_grad_xla"] = flops(jax.value_and_grad(like._logL), p)
    out["flops_value_xla"] = flops(like._logL, p)

    # --- stage inputs at this point
    nc2 = 2 * like.n_common
    rn_A, rn_g = p["rn_log10_A"], p["rn_gamma"]
    phi = like.phi_rn(rn_A, rn_g)
    phic = like.phi_common(p)
    phi = phi.at[:, :nc2].add(like.lam0 * phic[None, :])
    RA, c, s_perp = like.RA, like.c, like.s_perp

    S["phi_spectra"] = timeit(jax.jit(lambda a, g: like.phi_rn(a, g)), rn_A, rn_g)

    red_fwd = jax.jit(jax.vmap(L._reduce_fwd_impl))
    S["reduce_fwd (67x: QR 120x60 + trsm + matmuls)"] = timeit(red_fwd, RA, c, s_perp, phi)
    out["flops_reduce_fwd_xla"] = flops(jax.vmap(L._reduce_fwd_impl), RA, c, s_perp, phi)

    n = RA.shape[-1]
    Mstack = jnp.concatenate([RA * jnp.sqrt(phi)[:, None, :], jnp.broadcast_to(jnp.eye(n), RA.shape)], axis=1)
    S["batched_qr_reduced (67 x 120x60, Q and R)"] = timeit(jax.jit(lambda M: jnp.linalg.qr(M)), Mstack)
    S["batched_qr_r_only (67 x 120x60)"] = timeit(jax.jit(lambda M: jnp.linalg.qr(M, mode="r")), Mstack)
    _Qs, Tm = jax.jit(jnp.linalg.qr)(Mstack)
    S["batched_trsm (67 x 60x60, 61 rhs)"] = timeit(
        jax.jit(lambda T, B: jsl.solve_triangular(T, B, lower=False)), Tm, jnp.concatenate([RA, c[:, :, None]], axis=2)
    )

    _q, _ld, E, d = red_fwd(RA, c, s_perp, phi)
    E, d = E[:, :nc2, :nc2], d[:, :nc2]
    sqc = jnp.sqrt(phic)
    Es = E * sqc[None, :, None] * sqc[None, None, :]
    dt = d * sqc[None, :]
    core = like._core_sigma
    S["core_sigma value (assemble + chol 1876 + solve)"] = timeit(jax.jit(core), Es, dt)
    S["core_sigma value+grad"] = timeit(jax.jit(jax.value_and_grad(core, argnums=(0, 1))), Es, dt)
    P, k = dt.shape
    N = P * k
    Sig = jnp.kron(jnp.eye(P), jnp.eye(k)) * 3.0 + jax.scipy.linalg.block_diag(*[Es[a] for a in range(P)])
    S["cholesky 1876 alone"] = timeit(jax.jit(jnp.linalg.cholesky), Sig)
    Lc = jax.jit(jnp.linalg.cholesky)(Sig)
    S["tri_inv_lower recursive (bwd, float64)"] = timeit(jax.jit(L._tri_inv_lower), Lc)
    S["tri_inv via trsm(L, I) (float64)"] = timeit(
        jax.jit(lambda Lm: jsl.solve_triangular(Lm, jnp.eye(N), lower=True)), Lc
    )
    S["tri_inv_lower recursive float32"] = timeit(jax.jit(L._tri_inv_lower), Lc.astype(jnp.float32))
    S["trsv 1876 (solve)"] = timeit(jax.jit(lambda Lm, b: jsl.solve_triangular(Lm, b, lower=True)), Lc, dt.reshape(-1))
    G = jax.jit(L._tri_inv_lower)(Lc).reshape(N, P, k)
    S["einsum diag blocks of inverse"] = timeit(jax.jit(lambda G: jnp.einsum("iaj,iak->ajk", G, G)), G)
    S["gemm 1876 float64"] = timeit(jax.jit(lambda a: a @ a), Sig)
    S["gemm 1876 float32"] = timeit(jax.jit(lambda a: a @ a), Sig.astype(jnp.float32))
    out["gemm_tflops_f64"] = 2 * N**3 / (S["gemm 1876 float64"]["median_ms"] * 1e-3) / 1e12
    out["gemm_tflops_f32"] = 2 * N**3 / (S["gemm 1876 float32"]["median_ms"] * 1e-3) / 1e12

    # reduce backward (custom VJP) alone
    res = (red_fwd(RA, c, s_perp, phi)[2], red_fwd(RA, c, s_perp, phi)[3], RA, c, s_perp)
    cts = (jnp.ones(P), jnp.ones(P), jnp.ones((P, n, n)), jnp.ones((P, n)))
    S["reduce_bwd (67x analytic VJP)"] = timeit(jax.jit(jax.vmap(L._reduce_bwd)), res, cts)

    # analytic FLOP model
    m2 = 2 * n
    qr = 2 * m2 * n**2 - 2 * n**3 / 3
    fl = {
        "qr_R_67": P * qr,
        "qr_formQ_67": P * qr,
        "reduce_rest_67": P * (2 * n**3 + n**2 * (n + 1)),
        "chol_1876": N**3 / 3,
        # production _tri_inv_lower: dense GEMMs at every level (the triangular zeros are multiplied):
        # 4 h^3 + 2 * 4 (h/2)^3 + ... = (2/3) n^3; the triangular minimum would be n^3 / 3
        "tri_inv_1876_production_dense": 2 * N**3 / 3,
        "diag_blocks_einsum": 2 * N * P * k * k,
        "reduce_bwd_67": P * 2 * n**3,
    }
    fl["total"] = sum(fl.values())
    out["flop_model"] = fl
    out["achieved_tflops_value_and_grad"] = fl["total"] / (S["value_and_grad"]["median_ms"] * 1e-3) / 1e12
    out["achieved_tflops_chol"] = fl["chol_1876"] / (S["cholesky 1876 alone"]["median_ms"] * 1e-3) / 1e12
    for kk, v in S.items():
        print(f"{kk:55s} {v['median_ms']:8.3f} ms (min {v['min_ms']:.3f})")
    for kk in ("flops_value_and_grad_xla", "flops_value_xla", "gemm_tflops_f64", "gemm_tflops_f32", "achieved_tflops_value_and_grad", "achieved_tflops_chol"):
        print(kk, out[kk])
    print({k_: f"{v / 1e9:.3f} GF" for k_, v in fl.items()})
    save_json(f"profile_{jax.default_backend()}{args.tag}.json", out)


if __name__ == "__main__":
    main()
