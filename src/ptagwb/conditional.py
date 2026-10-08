"""Cached conditional log-likelihood of one HD free-spectrum bin (JAX; used by the conditional-grid
move of ``ptagwb.hybrid``; prototype: ``scripts/fs_schur_prototype.py``).

Only the two Fourier coefficients of bin k in every pulsar (2P = 134 coefficients) carry rho_k. With
the coefficients ordered by (frequency j, pulsar a), Phi = blockdiag_j(Phi_j), Phi_j =
diag(phi_irn[:, j]) + phi_c[j] Gamma, L_j = chol(Phi_j), A_a = F_a^T P F_a, b_a = F_a^T P r_a:

    logL = -1/2 [ s - z^T B^-1 z + log|B| ] - const/2,   B = I + L^T A L,   z = L^T b.

Split the coordinates into U (frequencies 2k, 2k+1) and R (the rest). L is block diagonal over
frequency, so with B_RR = I + L_R^T A_RR L_R (independent of rho_k):

    log|B| = log|B_RR| + log|S|,            S = I + L_U^T M L_U,
    z^T B^-1 z = z_R^T B_RR^-1 z_R + t^T S^-1 t,   t = L_U^T (b_U - h),
    M = A_UU - A_UR L_R B_RR^-1 L_R^T A_RU,  h = A_UR L_R B_RR^-1 z_R.

``build(x)`` costs one (2 n_modes - 2) P dimensional Cholesky; ``cond_logL(cache, rho)`` one 2P
dimensional Cholesky. The cache depends on every parameter except rho_k (by construction; tested).
S >= I bounds its smallest eigenvalue below in exact arithmetic but not its condition number
(~1e7 observed at high common power). Accuracy vs the production likelihood
(``tests/test_conditional.py``, ``scripts/fs_conditional_analysis.py --validate``): <= 1e-9 relative when the other
common bins carry at most moderate power (log10_rho <= -6, the posterior regime); with several bins
near maximal power B is very ill conditioned and this dense form loses precision (errors up to ~6e-6
in logL vs a long-double reference, where production stays within 2e-9). It is therefore used only to construct proposals; the
Metropolis-Hastings acceptance always uses the production likelihood.
The returned value omits the constant ``const_total`` of ``PTALikelihood`` (it cancels in every
use: differences over rho_k at fixed other parameters).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from .config import enable_x64

enable_x64()


def make_bin_conditional(like, k: int):
    """(build, cond_logL) for bin k of an HD free-spectrum ``PTALikelihood`` (or subclass).

    ``build(params)`` takes the likelihood's parameter dict (``rn_log10_A``, ``rn_gamma``,
    ``log10_rho``); ``cond_logL(cache, rho)`` returns logL + const_total / 2 at log10_rho_k = rho.
    """
    if like.common != "freespec" or like.Gamma is None:
        raise ValueError("bin conditional needs a free-spectrum likelihood with a correlated ORF")
    P, n, nc2 = like.P, 2 * like.n_modes, 2 * like.n_common
    if not 0 <= k < like.n_common:
        raise ValueError(f"bin {k} outside 0..{like.n_common - 1}")
    RA, c = np.asarray(like.RA), np.asarray(like.c)
    Aa = jnp.asarray(np.einsum("aij,aik->ajk", RA, RA))  # (P, n, n)
    ba = jnp.asarray(np.einsum("aij,ai->aj", RA, c))  # (P, n)
    s_tot = float(np.sum(np.asarray(like.s_perp)) + np.sum(c * c))
    G = jnp.asarray(np.asarray(like.Gamma))
    U = np.array([2 * k, 2 * k + 1])
    R = np.setdiff1d(np.arange(n), U)
    nR = len(R)
    A_RR = Aa[:, R][:, :, R]  # (P, nR, nR)
    A_RU = Aa[:, R][:, :, U]  # (P, nR, 2)
    A_UU = Aa[:, U][:, :, U]  # (P, 2, 2)
    b_R, b_U = ba[:, R], ba[:, U]  # (P, nR), (P, 2)
    eyeP = jnp.eye(P)

    def chol_phi(phi_diag, phic_j):
        return jnp.linalg.cholesky(jnp.diag(phi_diag) + phic_j * G)

    def build(params):
        phi_irn = like.phi_rn(jnp.asarray(params["rn_log10_A"]), jnp.asarray(params["rn_gamma"]))  # (P, n)
        phic = jnp.zeros(n).at[:nc2].set(jnp.repeat(10.0 ** (2.0 * jnp.asarray(params["log10_rho"])), 2))
        LR = jax.vmap(chol_phi)(phi_irn[:, R].T, phic[R])  # (nR, P, P), LR[j][b, a]
        # B_RR[(j,a),(k,c)] = delta + sum_b LR[j,b,a] A_RR[b,j,k] LR[k,b,c]
        T = LR[:, :, :, None] * jnp.transpose(A_RR, (1, 0, 2))[:, :, None, :]  # (j, b, a, k)
        Bm = jnp.einsum("jbak,kbc->jakc", T, LR).reshape(nR * P, nR * P)
        LB = jnp.linalg.cholesky(Bm + jnp.eye(nR * P))
        zR = jnp.einsum("jba,bj->ja", LR, b_R).reshape(-1)
        X = jnp.einsum("jba,bju->jaub", LR, A_RU).reshape(nR * P, 2 * P)
        W = jsl.solve_triangular(LB, X, lower=True)
        wz = jsl.solve_triangular(LB, zR, lower=True)
        AUU = jnp.einsum("auv,ab->uavb", A_UU, eyeP).reshape(2 * P, 2 * P)
        M = AUU - W.T @ W
        M = 0.5 * (M + M.T)
        bU_h = b_U.T.reshape(-1) - W.T @ wz
        return {"M": M, "bU_h": bU_h, "ldBRR": 2.0 * jnp.sum(jnp.log(jnp.diagonal(LB))), "qR": wz @ wz,
                "phi_U": phi_irn[:, U]}

    def cond_logL(cache, rho):
        phic = 10.0 ** (2.0 * rho)
        L0 = chol_phi(cache["phi_U"][:, 0], phic)
        L1 = chol_phi(cache["phi_U"][:, 1], phic)
        LU = jsl.block_diag(L0, L1)
        S = jnp.eye(2 * P) + LU.T @ cache["M"] @ LU
        t = LU.T @ cache["bU_h"]
        cS = jnp.linalg.cholesky(0.5 * (S + S.T))
        w = jsl.solve_triangular(cS, t, lower=True)
        ld = cache["ldBRR"] + 2.0 * jnp.sum(jnp.log(jnp.diagonal(cS)))
        return -0.5 * (s_tot - cache["qR"] - w @ w + ld)

    return build, cond_logL
