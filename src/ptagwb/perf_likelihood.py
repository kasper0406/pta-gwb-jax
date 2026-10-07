"""Performance variants of the marginalised PTA likelihood (performance study, docs/PERF.md).

NOT used by any production path. Every variant computes the *same* mathematical quantities as
``ptagwb.likelihood.PTALikelihood`` with the same square-root (Householder QR) numerics and the
same analytic VJPs; agreement with the production likelihood (value <= 1e-9 abs without the
constant, gradient <= 1e-8 rel) is checked by ``bench/check_exact.py`` (NG15, corners + posterior
+ prior draws) and ``tests/test_perf_likelihood.py``.

Knobs of ``FastPTALikelihood(..., reduce=..., tri_inv=...)``:

``reduce`` -- per-pulsar square-root stage. It needs T (R factor of [Y; I], Y = R_F Phi^1/2),
    Q1^T R_F and y = Q1^T c. Production forms Q with a batched geqrf + orgqr (cuSOLVER
    ``geqr2_batch_kernel``: 3.3 ms for 67 120x60 matrices, ~15 GFLOP/s) and then multiplies.

    * ``"hh"`` (default): R rows of the QR of the augmented matrix [[Y, R_F, c], [I, 0, 0]],
      using its structure (``_hh_bottom``): Householder step k only touches top row k and the
      n x (2n+1) bottom block, so 60 rank-1 updates of a 60 x 121 block replace the dense QR and
      the Q formation. R(M) = Q_full^T M gives T, Q1^T R_F and Q1^T c directly (row signs cancel
      in E, d, q and log|T|). Same backward stability (Householder reflections). 1.0 ms.
    * ``"aug"``: the same augmented matrix through ``jnp.linalg.qr(mode="r")`` (cuSOLVER):
      9.4 ms, slower than production (kept as a measured negative result).
    * ``"prod"``: the production code path.

``tri_inv`` -- how the backward pass of the 1876-dim core gets L^-1 (needed for the diagonal
    blocks of Sigma'^-1):

    * ``"levels"`` (default): ``_tri_inv_levels``, bottom-up block doubling with batched leaf
      solves and batched GEMMs that skip the zero upper blocks of the triangular factors three
      levels deep. 1.9 ms vs 5.2 ms (production recursion, 32 sequential small trsm calls).
    * ``"recursive"``: production ``likelihood._tri_inv_lower``; ``"trsm"``: one trsm against
      the identity (8.0 ms).

``value_and_grad_batched`` is the vmapped value+grad over a leading chain axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax.custom_derivatives import SymbolicZero

from . import likelihood as _L
from .config import enable_x64

enable_x64()


def _reduce_fwd_aug(RA, c, s_perp, r):
    """Same outputs as ``likelihood._reduce_fwd_impl`` via one R-only QR of [[Y, R_F, c], [I, 0, 0]]."""
    sq = jnp.sqrt(r)
    n = RA.shape[-1]
    top = jnp.concatenate([RA * sq[None, :], RA, c[:, None]], axis=1)
    bot = jnp.concatenate([jnp.eye(n, dtype=RA.dtype), jnp.zeros((n, n + 1), RA.dtype)], axis=1)
    R = jnp.linalg.qr(jnp.concatenate([top, bot], axis=0), mode="r")
    Tm = R[:n, :n]
    W = R[:n, n:]  # [Q1^T R_F, Q1^T c]
    y = W[:, n]
    q = s_perp + c @ c - y @ y
    ld = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(Tm))))
    Z = jsl.solve_triangular(Tm, W, lower=False) / sq[:, None]
    E = 0.5 * (Z[:, :n] + Z[:, :n].T)
    d = Z[:, n]
    return q, ld, E, d


def _hh_bottom(top):
    """R rows of the QR of [[top], [I_n, 0]] (top: (n, m) whose first n columns are upper triangular),
    restricted to the first n Householder steps: returns the (n, m) top block of R.

    Structure: reflection k only involves top row k and the bottom n x m block B (the top block
    stays upper triangular, top row k is untouched before step k), so the only state is B::

        for k: x = [top[k, k]; B[:, k]],  beta = -sign(x_0) |x|,  v = [1; B[:, k] / (x_0 - beta)]
               tau = (beta - x_0) / beta,  w = top[k] + v_B^T B
               R[k] = top[k] - tau w   (R[k, k] = beta),   B <- B - tau v_B w^T

    One rank-1 update of an n x m block per step instead of a dense 2n x m Householder sweep.
    """
    n, m = top.shape
    B0 = jnp.concatenate([jnp.eye(n, dtype=top.dtype), jnp.zeros((n, m - n), top.dtype)], axis=1)

    def step(k, carry):
        B, R = carry
        r = jax.lax.dynamic_index_in_dim(top, k, axis=0, keepdims=False)
        a = r[k]
        b = jax.lax.dynamic_index_in_dim(B, k, axis=1, keepdims=False)
        nrm = jnp.sqrt(a * a + b @ b)
        beta = -jnp.where(a >= 0, 1.0, -1.0) * nrm
        v = b / (a - beta)
        tau = (beta - a) / beta
        w = r + v @ B
        R = jax.lax.dynamic_update_index_in_dim(R, r - tau * w, k, axis=0)
        B = B - tau * v[:, None] * w[None, :]
        return B, R

    _, R = jax.lax.fori_loop(0, n, step, (B0, jnp.zeros_like(top)))
    return R


def _reduce_fwd_hh(RA, c, s_perp, r):
    """``_reduce_fwd_aug`` with the structured Householder ``_hh_bottom`` instead of geqrf."""
    sq = jnp.sqrt(r)
    n = RA.shape[-1]
    R = _hh_bottom(jnp.concatenate([RA * sq[None, :], RA, c[:, None]], axis=1))
    Tm = R[:, :n]
    W = R[:, n:]
    y = W[:, n]
    q = s_perp + c @ c - y @ y
    ld = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(Tm))))
    Z = jsl.solve_triangular(Tm, W, lower=False) / sq[:, None]
    E = 0.5 * (Z[:, :n] + Z[:, :n].T)
    d = Z[:, n]
    return q, ld, E, d


def _reduce_bwd_sz(res, cts):
    """``likelihood._reduce_bwd`` with symbolic-zero cotangents skipped (``symbolic_zeros=True``).

    Same terms, same order, minus the ones whose cotangent is a symbolic zero::

        rb = -qb d^2 + ldb diag(E) - rowsum((E Eb) * E) - d * (E db)

    Skipping symbolic zeros removes n^3 wasted work when E or d are unused (CURN). It is *not* the
    protection against the XLA:CPU YNNPACK bug (any broadcast cotangent -- explicit zeros or a
    nonzero constant -- triggers it): that is the XLA flag set by ``ptagwb/__init__`` plus the
    optimization barriers below (docs/PERF.md).
    """
    E, d, RA, c, s_perp = res
    qb, ldb, Eb, db = cts
    rb = jnp.zeros_like(d)
    if not isinstance(qb, SymbolicZero):
        rb = rb - qb * d * d
    if not isinstance(ldb, SymbolicZero):
        rb = rb + ldb * jnp.diagonal(E)
    # optimization_barrier: second defence against the XLA:CPU YNNPACK miscompilation of
    # reduce(dot(E, broadcast(c)) * E) (any constant/broadcast cotangent, not only zeros); it
    # keeps the product out of the fused reduction. The package-wide fix is the XLA flag set in
    # ptagwb/__init__ (config.disable_xla_cpu_ynn_fusion).
    if not isinstance(Eb, SymbolicZero):
        Eb = 0.5 * (Eb + Eb.T)
        rb = rb - jnp.sum(jax.lax.optimization_barrier(E @ Eb) * E, axis=1)
    if not isinstance(db, SymbolicZero):
        rb = rb - d * jax.lax.optimization_barrier(E @ db)
    return jnp.zeros_like(RA), jnp.zeros_like(c), jnp.zeros_like(s_perp), rb


def _make_reduce(impl):
    @jax.custom_vjp
    def red(RA, c, s_perp, r):
        return impl(RA, c, s_perp, r)

    def fwd(RA, c, s_perp, r):  # symbolic_zeros=True: primals arrive as CustomVJPPrimal
        RA, c, s_perp, r = RA.value, c.value, s_perp.value, r.value
        out = impl(RA, c, s_perp, r)
        _, _, E, d = out
        return out, (E, d, RA, c, s_perp)

    red.defvjp(fwd, _reduce_bwd_sz, symbolic_zeros=True)
    return red


REDUCERS = {"prod": _L._reduce, "aug": _make_reduce(_reduce_fwd_aug), "hh": _make_reduce(_reduce_fwd_hh)}


def _tri_inv_trsm(L):
    return jsl.solve_triangular(L, jnp.eye(L.shape[0], dtype=L.dtype), lower=True)


def _mm_lower_right(M, T, depth):
    """M @ T for lower-triangular T (batched), skipping T's zero upper block ``depth`` levels deep."""
    s = T.shape[-1]
    if depth == 0 or s % 2:
        return jnp.matmul(M, T)
    h = s // 2
    left = _mm_lower_right(M[..., :h], T[..., :h, :h], depth - 1) + jnp.matmul(M[..., h:], T[..., h:, :h])
    right = _mm_lower_right(M[..., h:], T[..., h:, h:], depth - 1)
    return jnp.concatenate([left, right], axis=-1)


def _mm_lower_left(T, M, depth):
    """T @ M for lower-triangular T (batched), skipping T's zero upper block ``depth`` levels deep."""
    s = T.shape[-1]
    if depth == 0 or s % 2:
        return jnp.matmul(T, M)
    h = s // 2
    top = _mm_lower_left(T[..., :h, :h], M[..., :h, :], depth - 1)
    bot = jnp.matmul(T[..., h:, :h], M[..., :h, :]) + _mm_lower_left(T[..., h:, h:], M[..., h:, :], depth - 1)
    return jnp.concatenate([top, bot], axis=-2)


def _tri_inv_levels(L, leaf: int = 64, depth: int = 3):
    """Inverse of a lower-triangular L by bottom-up block doubling with *batched* kernels.

    Pad L to N = b * 2^m >= n (b = ceil(n / 2^m) ~ ``leaf``; n = 1876 -> b = 59, N = 1888) with an
    identity block, invert all b x b diagonal blocks in one batched triangular solve, then at each
    level merge pairs [[A, 0], [L21, C]] -> [[A^-1, 0], [-C^-1 L21 A^-1, C^-1]] with batched GEMMs.
    The products with the (lower-triangular) A^-1 and C^-1 skip their zero upper blocks ``depth``
    levels deep (flops x 0.75 / 0.625 / 0.56 for depth 1 / 2 / 3). Same arithmetic as
    ``_tri_inv_lower`` (the production recursion), but log2(N/b) batched launches instead of a
    depth-first recursion with one small trsm per leaf.
    """
    n = L.shape[0]
    m = max(0, int(np.ceil(np.log2(n / leaf))))
    b = -(-n // 2**m)
    N = b * 2**m
    if N > n:
        L = jnp.block([[L, jnp.zeros((n, N - n), L.dtype)], [jnp.zeros((N - n, n), L.dtype), jnp.eye(N - n, dtype=L.dtype)]])
    nb = N // b
    Lb = L.reshape(nb, b, nb, b)
    idx = jnp.arange(nb)
    D = Lb[idx, :, idx, :]  # (nb, b, b) diagonal blocks
    inv = jsl.solve_triangular(D, jnp.broadcast_to(jnp.eye(b, dtype=L.dtype), D.shape), lower=True)
    s, lev = b, 0
    while s < N:
        c = N // s  # blocks of size s
        Ls = L.reshape(c, s, c, s)
        p = jnp.arange(c // 2)
        L21 = Ls[2 * p + 1, :, 2 * p, :]  # (c/2, s, s)
        Ai, Ci = inv[0::2], inv[1::2]
        d = min(depth, lev)  # block structure of A^-1, C^-1 is known lev levels deep
        X = -_mm_lower_left(Ci, _mm_lower_right(L21, Ai, d), d)
        top = jnp.concatenate([Ai, jnp.zeros_like(Ai)], axis=2)
        bot = jnp.concatenate([X, Ci], axis=2)
        inv = jnp.concatenate([top, bot], axis=1)
        s *= 2
        lev += 1
    return inv[0][:n, :n]


def _chol_inv_rec(S, leaf: int = 128, depth: int = 3):
    """(L, L^-1) of an SPD S by recursive 2x2 blocking with GEMMs (``tri_inv="fused"``):

        L11, L11^-1 = rec(S11);  L21 = S21 L11^-T;  L22, L22^-1 = rec(S22 - L21 L21^T)
        L^-1 = [[L11^-1, 0], [-L22^-1 L21 L11^-1, L22^-1]]

    Computes the Cholesky factor and its inverse together (the backward pass needs L^-1 anyway),
    replacing cuSOLVER potrf (0.45 TFLOP/s at n = 1876) by GEMMs. L21 uses the explicit L11^-1
    instead of a triangular solve (forward error ~ cond(L11) eps instead of ~eps); measured by
    bench/check_exact.py like every variant.
    """
    n = S.shape[0]
    if n <= leaf:
        L = jnp.linalg.cholesky(S)
        return L, jsl.solve_triangular(L, jnp.eye(n, dtype=S.dtype), lower=True)
    h = n // 2
    L11, L11i = _chol_inv_rec(S[:h, :h], leaf, depth)
    L21 = _mm_lower_left(L11i, S[h:, :h].T, depth).T
    L22, L22i = _chol_inv_rec(S[h:, h:] - L21 @ L21.T, leaf, depth)
    X = -_mm_lower_left(L22i, _mm_lower_right(L21, L11i, depth), depth)
    Z = jnp.zeros((h, n - h), S.dtype)
    return jnp.block([[L11, Z], [L21, L22]]), jnp.block([[L11i, Z], [X, L22i]])


def make_core_sigma(Ginv, k: int, tri_inv: str = "recursive", grad_precision: str = "float64"):
    """``likelihood._make_core_sigma`` with a selectable backward triangular inverse."""
    P = Ginv.shape[0]
    n = P * k
    eye_k = jnp.eye(k)
    inv = {"recursive": _L._tri_inv_lower, "trsm": _tri_inv_trsm, "levels": _tri_inv_levels, "fused": None}[tri_inv]

    def assemble(Es):
        Sig = jnp.kron(Ginv, eye_k)
        blocks = jnp.zeros((P, k, P, k), dtype=Es.dtype)
        idx = jnp.arange(P)
        blocks = blocks.at[idx, :, idx, :].set(Es)
        return Sig + blocks.reshape(n, n)

    fused = tri_inv == "fused"

    def fwd(Es, dt):
        if fused:
            L, Linv = _chol_inv_rec(assemble(Es))
        else:
            L, Linv = jnp.linalg.cholesky(assemble(Es)), None
        w = jsl.solve_triangular(L, dt.reshape(-1), lower=True)
        return _L._chol_logdet(L) - w @ w, (L, w, Linv)

    def bwd(res, g):
        L, w, Linv = res
        u = jsl.solve_triangular(L.T, w, lower=False)
        if fused:
            pass
        elif grad_precision == "mixed":
            dsc = 1.0 / jnp.sqrt(jnp.sum(L * L, axis=1))
            Leq = (dsc[:, None] * L).astype(jnp.float32)
            Linv = inv(Leq).astype(jnp.float64) * dsc[None, :]
        else:
            Linv = inv(L)
        G = Linv.reshape(n, P, k)
        S = jnp.einsum("iaj,iak->ajk", G, G)
        ua = u.reshape(P, k)
        return g * (S + ua[:, :, None] * ua[:, None, :]), g * (-2.0 * ua)

    @jax.custom_vjp
    def core(Es, dt):
        return fwd(Es, dt)[0]

    core.defvjp(fwd, bwd)
    return core


class FastPTALikelihood(_L.PTALikelihood):
    """``PTALikelihood`` with selectable per-pulsar reducer and backward triangular inverse.

    Constructor arguments as ``PTALikelihood`` plus ``reduce`` ("aug" | "prod") and ``tri_inv``
    ("recursive" | "trsm"). ``value_and_grad_batched`` is the vmapped value+grad over a leading
    chain axis of every parameter.
    """

    def __init__(self, terms, T, *, reduce: str = "hh", tri_inv: str = "levels", **kw):
        if reduce not in REDUCERS:
            raise ValueError(f"unknown reduce {reduce!r}")
        if kw.get("method", "sigma") != "sigma":
            raise ValueError("FastPTALikelihood implements only method='sigma' (use PTALikelihood for method='B')")
        self.reduce_name, self.tri_inv = reduce, tri_inv
        self._reducer = REDUCERS[reduce]
        super().__init__(terms, T, **kw)
        if self.Lgamma is not None:  # same Ginv as production (numpy float64)
            G = np.asarray(self.Gamma) - self.lam0 * np.eye(self.P)
            Lg = np.linalg.cholesky(G)
            Ginv = np.linalg.solve(Lg.T, np.linalg.solve(Lg, np.eye(self.P)))
            Ginv = jnp.asarray(0.5 * (Ginv + Ginv.T))
            self._core_sigma = make_core_sigma(Ginv, 2 * self.n_common, tri_inv, self.grad_precision)
        self.logL = jax.jit(self._logL)
        self.value_and_grad = jax.jit(jax.value_and_grad(self._logL))
        self.value_and_grad_batched = jax.jit(jax.vmap(jax.value_and_grad(self._logL)))

    def _logL(self, params):
        rn_A = jnp.asarray(params["rn_log10_A"], dtype=jnp.float64)
        rn_g = jnp.asarray(params["rn_gamma"], dtype=jnp.float64)
        phi = self.phi_rn(rn_A, rn_g)
        phic = self.phi_common(params)
        nc2 = 2 * self.n_common
        red = jax.vmap(self._reducer)
        if self.Lgamma is None:
            phi = phi.at[:, :nc2].add(phic[None, :])
            q, ld, _, _ = red(self.RA, self.c, self.s_perp, phi)
            return -0.5 * (jnp.sum(q + ld) + self.const_total)
        phi = phi.at[:, :nc2].add(self.lam0 * phic[None, :])
        q, ld, E, d = red(self.RA, self.c, self.s_perp, phi)
        E, d = E[:, :nc2, :nc2], d[:, :nc2]
        sqc = jnp.sqrt(phic)
        Es = E * sqc[None, :, None] * sqc[None, None, :]
        dt = d * sqc[None, :]
        core = self._core_sigma(Es, dt) + self._logdet_gamma_term
        return -0.5 * (jnp.sum(q + ld) + self.const_total + core)
