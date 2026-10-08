"""PROTOTYPE (not used by any run): cached conditional log-likelihood of one HD free-spectrum bin.

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_schur_prototype.py [--bin 2] [--draw 0]

Groundwork for Metropolised conditional-grid moves (docs/FS_PILOT.md Sec. 8). Only the two Fourier
coefficients of bin k in every pulsar (134 coefficients) carry rho_k, so after one O(n^3) setup at
the current point the conditional logL(rho_k | rest) costs O(134^3) per value.

Formulation (coefficients ordered (frequency j, pulsar a); Phi = blockdiag_j(Phi_j), Phi_j =
diag(phi_irn[:, j]) + phi_c[j] Gamma, L_j = chol(Phi_j); A = F^T P F, b = F^T P r per pulsar):

    logL = -1/2 [ s - z^T B^-1 z + log|B| ] + const,  B = I + L^T A L,  z = L^T b.

Split the coordinates into U (frequencies 2k, 2k+1: all pulsars) and R (the rest). L is block
diagonal over frequency, so with B_RR = I + L_R^T A_RR L_R (independent of rho_k),

    log|B|    = log|B_RR| + log|S|,                 S = I + L_U^T M L_U,
    z^T B^-1 z = z_R^T B_RR^-1 z_R + t^T S^-1 t,     t = L_U^T (b_U - h),
    M = A_UU - A_UR L_R B_RR^-1 L_R^T A_RU,          h = A_UR L_R B_RR^-1 z_R,

and only L_U depends on rho_k. S >= I, so the per-value work is well conditioned. Checked here
against the production ``PTALikelihood._logL`` (differences between grid values, which cancel
the constant).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import get_terms

from ptagwb import basis as _basis
from ptagwb.sampling import ModelSpec, load_run, make_likelihood, unpack


class BinConditional:
    """Cached conditional logL(rho_k | rest) for one free-spectrum bin of the HD likelihood."""

    def __init__(self, like, x: np.ndarray, spec: ModelSpec, k: int):
        P, n = like.P, 2 * like.n_modes
        nc2 = 2 * like.n_common
        par = unpack(x, spec, P)
        phi_irn = np.asarray(like.phi_rn(np.asarray(par["rn_log10_A"]), np.asarray(par["rn_gamma"])))  # (P, n)
        self.phic = np.asarray(_basis.free_spectrum(np.asarray(par["log10_rho"])))  # (nc2,)
        G = np.asarray(like.Gamma)
        RA, c = np.asarray(like.RA), np.asarray(like.c)
        Aa = np.einsum("aij,aik->ajk", RA, RA)  # (P, n, n)
        ba = np.einsum("aij,ai->aj", RA, c)  # (P, n)
        self.s = float(np.sum(np.asarray(like.s_perp)) + np.sum(c * c))
        self.k, self.P, self.n, self.G, self.phi_irn = k, P, n, G, phi_irn
        # dense A, b in (j, a) ordering
        N = P * n
        A = np.zeros((n, P, n, P))
        for a in range(P):
            A[:, a, :, a] = Aa[a]
        A = A.reshape(N, N)
        b = ba.T.reshape(N)
        U = np.r_[np.arange(2 * k * P, (2 * k + 2) * P)]
        R = np.setdiff1d(np.arange(N), U)
        self.U, self.R = U, R

        def Lj(j, phic_j):
            return np.linalg.cholesky(np.diag(phi_irn[:, j]) + phic_j * G)

        jR = [j for j in range(n) if j not in (2 * k, 2 * k + 1)]
        LR = sl.block_diag(*[Lj(j, self.phic[j] if j < nc2 else 0.0) for j in jR])
        self._Lj = Lj
        BRR = np.eye(len(R)) + LR.T @ A[np.ix_(R, R)] @ LR
        cB = sl.cho_factor(BRR, lower=True)
        zR = LR.T @ b[R]
        X = LR.T @ A[np.ix_(R, U)]  # (|R|, 134)
        self.M = A[np.ix_(U, U)] - X.T @ sl.cho_solve(cB, X)
        self.M = 0.5 * (self.M + self.M.T)
        self.bU_h = b[U] - X.T @ sl.cho_solve(cB, zR)
        self.ldBRR = 2.0 * float(np.sum(np.log(np.diag(cB[0]))))
        self.qR = float(zR @ sl.cho_solve(cB, zR))

    def logL(self, rho: float) -> float:
        """Conditional log-likelihood at log10_rho_k = rho, up to the constant of the production
        likelihood (``const_total``)."""
        phi = float(np.asarray(_basis.free_spectrum(np.array([rho])))[0])
        LU = sl.block_diag(self._Lj(2 * self.k, phi), self._Lj(2 * self.k + 1, phi))
        S = np.eye(len(self.U)) + LU.T @ self.M @ LU
        t = LU.T @ self.bU_h
        cS = sl.cho_factor(0.5 * (S + S.T), lower=True)
        ld = self.ldBRR + 2.0 * float(np.sum(np.log(np.diag(cS[0]))))
        q = self.qR + float(t @ sl.cho_solve(cS, t))
        return -0.5 * (self.s - q + ld)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--n-grid", type=int, default=40)
    args = ap.parse_args()
    spec = ModelSpec(orf="hd", common="freespec", n_common=30, position="enterprise")
    terms, T = get_terms()
    like = make_likelihood(terms, T, spec, "production")
    r = load_run("hd_fs30")
    x0 = r["x"].reshape(-1, r["x"].shape[-1])[args.draw].copy()
    j = r["names"].index(f"gw_log10_rho_{args.bin}")
    t0 = time.time()
    bc = BinConditional(like, x0, spec, args.bin)
    t_setup = time.time() - t0
    grid = np.linspace(-15.4, -2.0, args.n_grid)
    t0 = time.time()
    cond = np.array([bc.logL(g) for g in grid])
    t_grid = time.time() - t0
    prod = []
    for g in grid:
        x = x0.copy()
        x[j] = g
        prod.append(float(like.logL(unpack(x, spec, like.P))))
    prod = np.array(prod)
    d = (cond - cond[0]) - (prod - prod[0])
    print(f"bin f_{args.bin + 1}, draw {args.draw}: setup {t_setup:.1f} s (CPU, numpy), {args.n_grid} values {t_grid:.2f} s")
    print(f"logL range over the grid {prod.max() - prod.min():.1f}; max |conditional - production| (relative to grid[0]) {np.abs(d).max():.2e}")
    print("rho, production logL - logL(grid0), conditional - production:")
    for g, p_, dd in zip(grid[::5], (prod - prod[0])[::5], d[::5]):
        print(f"  {g:7.2f} {p_:14.4f} {dd:+.2e}")
    return 0 if np.abs(d).max() < 1e-6 * max(1.0, prod.max() - prod.min()) else 1


if __name__ == "__main__":
    sys.exit(main())
