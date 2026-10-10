"""Fast numpy evaluation of the EPTA DR2new likelihood for many hyperparameter values at fixed
stage-1 data (used by the conditional-occupancy diagnostics and the T2 grids; CPU, single
thread per process). Validated against ``ptagwb.epta.EPTAModel`` (tests/test_m3b_fastcond.py).

Per pulsar a, with stage-1 factors (R_F, c, s_perp) of ``combined.precompute_general``
(A = R_F^T R_F, b = R_F^T c) and the diagonal prior Phi_a (noise processes, plus phi_CP on the
common columns for CURN):

    lnL_a = -1/2 [s_perp + c.c - |L^-1 S b|^2 + log|M|],  M = I + S A S = L L^T,  S = Phi^(1/2)

(the scaled form: M has eigenvalues >= 1). HD: every pulsar's noise columns are eliminated
exactly (Schur complement in the same scaled form), then the joint common system over all
pulsars, M_c = I + (L_G (x) D)^T blockdiag(A'_a) (L_G (x) D) with Gamma = L_G L_G^T and
D = diag(phi_CP)^(1/2). The total adds the model's parameter-independent constant, so values
equal ``EPTAModel.logL``.

The dip pulsar's (c, s_perp) are recomputed from the dip parameters with the stage-1 thin factors
(the operations of ``EPTAModel.dip_terms``), in numpy.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .basis import powerlaw


class FastEPTA:
    def __init__(self, model):
        self.m = model
        like = model.like
        self.orf = model.orf
        self.P = like.P
        self.const = float(like.const_total)
        self.K = [t.K for t in model.terms]
        self.RA = [np.asarray(like.RA[a, :k, :k]) for a, k in enumerate(self.K)]
        self.A = [r.T @ r for r in self.RA]
        self.c0 = [np.asarray(like.c[a, :k]) for a, k in enumerate(self.K)]
        self.s0 = np.asarray(like.s_perp).copy()
        self.G = np.asarray(like.G)  # (P, 2 n_common) common columns
        self.fc, self.dfc = np.asarray(like.fc), np.asarray(like.dfc)
        # per pulsar: list of (process, cols, f, df)
        self.proc = [[] for _ in range(self.P)]
        for nm, pr in like.processes.items():
            f, df, idx = np.asarray(pr["f"]), np.asarray(pr["df"]), np.asarray(pr["idx"])
            for i, a in enumerate(pr["psr_idx"]):
                k = self.K[a]
                ok = idx[i] < (a + 1) * like.K
                cols = idx[i][ok] - a * like.K
                if np.any(cols >= k):
                    raise ValueError("process column outside the pulsar's layout")
                self.proc[a].append((nm, cols, f[i][ok], df[i][ok]))
        self.ix = {n: i for i, n in enumerate(model.param_names)}
        self.names = [p.name for p in model.psrs]
        self.Gamma = None if model.orf == "crn" else np.asarray(like.Gamma)
        if self.Gamma is not None:
            self.LG = np.linalg.cholesky(self.Gamma)
            self.logdet_G = 2.0 * float(np.sum(np.log(np.diag(self.LG))))
        d = model.dip
        self.dip_a = d.index
        self.dip = {k: np.asarray(getattr(d, k)) for k in ("toas", "chrom", "winv", "wr0", "Q", "QF")}
        self.dip_sign = d.sign
        self._dipix = model._dip_idx
        self.noise_cols = []
        for a in range(self.P):
            cc = set(self.G[a].tolist())
            self.noise_cols.append(np.array([j for j in range(self.K[a]) if j not in cc], dtype=np.int64))

    # ------------------------------------------------------------- pieces
    def phi_noise(self, a: int, x: np.ndarray) -> np.ndarray:
        phi = np.zeros(self.K[a])
        name = self.names[a]
        for nm, cols, f, df in self.proc[a]:
            phi[cols] += powerlaw(f, df, x[self.ix[f"{name}_{nm}_log10_A"]], x[self.ix[f"{name}_{nm}_gamma"]])
        return phi

    def phi_common(self, x: np.ndarray) -> np.ndarray:
        gw = f"gw_{self.orf}"
        g = x[self.ix[f"{gw}_gamma"]] if f"{gw}_gamma" in self.ix else float(self.m.gamma_common)
        return powerlaw(self.fc, self.dfc, x[self.ix[f"{gw}_log10_A"]], g)

    def dip_terms(self, x: np.ndarray):
        ia, it, i0 = self._dipix
        A, tau, t0 = 10.0 ** x[ia], 10.0 ** x[it] * 86400.0, x[i0] * 86400.0
        D = self.dip
        dt = D["toas"] - t0
        wf = np.where(dt >= 0.0, np.exp(-np.maximum(dt, 0.0) / tau), 0.0)
        wr = D["wr0"] - D["winv"] * (self.dip_sign * A * wf * D["chrom"])
        Q, QF = D["Q"], D["QF"]
        rp = wr - Q @ (Q.T @ wr)
        rp = rp - Q @ (Q.T @ rp)
        c = QF.T @ rp
        rperp = rp - QF @ c
        rperp = rperp - QF @ (QF.T @ rperp)
        return c, float(rperp @ rperp)

    def data(self, a: int, x: np.ndarray, dip_cache=None):
        if a == self.dip_a:
            c, s = dip_cache if dip_cache is not None else self.dip_terms(x)
        else:
            c, s = self.c0[a], float(self.s0[a])
        return c, s, self.RA[a].T @ c

    @staticmethod
    def _scaled(A, b, phi):
        S = np.sqrt(phi)
        M = (S[:, None] * A) * S[None, :]
        M[np.diag_indices_from(M)] += 1.0
        L = np.linalg.cholesky(M)
        y = solve_triangular(L, S * b, lower=True)
        return L, y, 2.0 * float(np.sum(np.log(np.diag(L))))

    # ------------------------------------------------------------- CURN
    def curn_term(self, a: int, x: np.ndarray, phic=None, dip_cache=None) -> float:
        phic = self.phi_common(x) if phic is None else phic
        phi = self.phi_noise(a, x)
        phi[self.G[a]] += phic
        c, s, b = self.data(a, x, dip_cache)
        _, y, ld = self._scaled(self.A[a], b, phi)
        return -0.5 * (s + c @ c - y @ y + ld)

    def curn(self, x: np.ndarray) -> float:
        phic = self.phi_common(x)
        return float(sum(self.curn_term(a, x, phic) for a in range(self.P)) - 0.5 * self.const)

    # ------------------------------------------------------------- HD
    def hd_reduce(self, a: int, x: np.ndarray, dip_cache=None):
        """(scalar contribution, A'_cc, b'_c) of pulsar a after eliminating its noise columns."""
        c, s, b = self.data(a, x, dip_cache)
        A = self.A[a]
        n, g = self.noise_cols[a], self.G[a]
        phin = self.phi_noise(a, x)[n]
        if n.size == 0:
            return -0.5 * (s + c @ c), A[np.ix_(g, g)], b[g]
        L, y, ld = self._scaled(A[np.ix_(n, n)], b[n], phin)
        Sn = np.sqrt(phin)
        Z = solve_triangular(L, Sn[:, None] * A[np.ix_(n, g)], lower=True)
        Ared = A[np.ix_(g, g)] - Z.T @ Z
        bred = b[g] - Z.T @ y
        return -0.5 * (s + c @ c - y @ y + ld), Ared, bred

    def hd_joint(self, parts, x: np.ndarray) -> float:
        phic = self.phi_common(x)
        nc = phic.size
        P = self.P
        D = np.sqrt(phic)
        # T = blockdiag(A'_a) (L_G (x) D); M = I + (L_G (x) D)^T T
        Ab = [p[1] for p in parts]
        bb = np.concatenate([p[2] for p in parts])
        Lfull = np.kron(self.LG, np.diag(D))
        T = np.zeros((P * nc, P * nc))
        for a in range(P):
            T[a * nc:(a + 1) * nc] = Ab[a] @ Lfull[a * nc:(a + 1) * nc]
        M = Lfull.T @ T
        M = 0.5 * (M + M.T)
        M[np.diag_indices_from(M)] += 1.0
        cf = cho_factor(M, lower=True)
        u = Lfull.T @ bb
        ld = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
        return float(sum(p[0] for p in parts) - 0.5 * (-(u @ cho_solve(cf, u)) + ld))

    def hd(self, x: np.ndarray) -> float:
        parts = [self.hd_reduce(a, x) for a in range(self.P)]
        return self.hd_joint(parts, x) - 0.5 * self.const

    def logL(self, x: np.ndarray) -> float:
        return self.curn(x) if self.orf == "crn" else self.hd(x)

    # ------------------------------------------------------------- HD with one pulsar replaced
    def hd_prepare(self, parts, x):
        """Cache of the joint common system for fixed common hyperparameters, so that one
        pulsar's reduced block can be swapped at O((P nc)^2 nc) cost (``hd_swap``)."""
        phic = self.phi_common(x)
        nc = phic.size
        Lfull = np.kron(self.LG, np.diag(np.sqrt(phic)))
        Mb = np.zeros((self.P * nc, self.P * nc))
        for a in range(self.P):
            La = Lfull[a * nc:(a + 1) * nc]
            Mb += La.T @ parts[a][1] @ La
        u = Lfull.T @ np.concatenate([p[2] for p in parts])
        return {"Lfull": Lfull, "Mb": Mb, "u": u, "parts": parts, "nc": nc, "s": sum(p[0] for p in parts)}

    def hd_swap(self, cache, a: int, part) -> float:
        nc, Lfull = cache["nc"], cache["Lfull"]
        La = Lfull[a * nc:(a + 1) * nc]
        old = cache["parts"][a]
        M = cache["Mb"] + La.T @ (part[1] - old[1]) @ La
        M = 0.5 * (M + M.T)
        M[np.diag_indices_from(M)] += 1.0
        u = cache["u"] + La.T @ (part[2] - old[2])
        cf = cho_factor(M, lower=True)
        ld = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
        return float(cache["s"] - old[0] + part[0] - 0.5 * (-(u @ cho_solve(cf, u)) + ld) - 0.5 * self.const)
