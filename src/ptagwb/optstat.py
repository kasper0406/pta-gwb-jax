"""Optimal statistic (OS), noise marginalisation, pair covariance and binned correlations.

Cross-correlation OS (Anholm et al. 2009; Demorest et al. 2013; Chamberlin et al. 2015) as in
enterprise_extensions ``OptimalStatistic.compute_os`` and the NG15 release
(``data_release/figure_1/optimal_statistic_covariances.py``). For pulsar a with the CURN noise
model (fixed white noise, timing model, intrinsic RN, and the common uncorrelated process at the
given amplitude), C_a = N_a + M_a inf M_a^T + F_a Phi_a F_a^T, and with F the *common-process*
columns (first 2 n_common Fourier columns)::

    X_a = F^T C_a^-1 r_a,    Z_a = F^T C_a^-1 F
    rho_ab = X_a^T phi X_b / tr(Z_a phi Z_b phi),   sigma_ab = tr(Z_a phi Z_b phi)^-1/2
    A^2_OS = sum rho Gamma / sigma^2 / sum Gamma^2 / sigma^2,   sigma_OS = (sum Gamma^2/sigma^2)^-1/2

with phi = diag of the common power-law PSD at log10_A = 0 (so that A^2_OS estimates A^2).

In our coefficient-space likelihood these are exactly the projected quantities of the
square-root reduction ``likelihood._reduce`` with the diagonal prior r = Phi_a (IRN + common):
``X_a = d_a = E A^-1 b`` and ``Z_a = E_a = (A^-1 + Phi)^-1`` restricted to the common block
(Woodbury: F^T C^-1 F = A - A (Phi^-1 + A)^-1 A = (A^-1 + Phi)^-1). So the OS costs one CURN
likelihood evaluation plus O(P^2 k^2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import basis as _basis
from . import orf as _orf
from .config import enable_x64
from .likelihood import _reduce

enable_x64()


def hd_curve(xi):
    """HD correlation (cross-pulsar, no auto term) at angular separation xi [rad]."""
    x = (1.0 - np.cos(xi)) / 2.0
    return 1.5 * x * np.log(x) - 0.25 * x + 0.5


class OptimalStatistic:
    """OS machinery for a ``PTALikelihood`` (uses its precomputed terms and sky positions).

    ``orf``: correlation pattern searched for ("hd", "monopole", "dipole"); only the
    pair values (a < b) are used, so the diagonal regulariser of monopole/dipole is irrelevant.
    """

    def __init__(self, like, orf: str = "hd"):
        self.like = like
        self.P, self.nc2 = like.P, 2 * like.n_common
        ia, ib = np.triu_indices(self.P, k=1)
        self.ia, self.ib = ia, ib
        pos = like.pos
        self.xi = np.arccos(np.clip(np.sum(pos[ia] * pos[ib], axis=1), -1.0, 1.0))
        G = _orf.ORFS[orf](pos) if orf == "hd" else _orf.ORFS[orf](pos, diag_eps=0.0)
        self.Gamma_full = G
        self.orf_pairs = G[ia, ib]
        self._components = jax.jit(self._components_impl)
        self._rho_sig = jax.jit(self._rho_sig_impl)

    # -- per-pulsar projections --
    def _components_impl(self, params):
        like = self.like
        phi = like.phi_rn(jnp.asarray(params["rn_log10_A"]), jnp.asarray(params["rn_gamma"]))
        nc2 = self.nc2
        phic = _basis.powerlaw(like.f[:nc2], like.df[:nc2], params["log10_A"], params["gamma"])
        phi = phi.at[:, :nc2].add(phic[None, :])
        _, _, E, d = jax.vmap(_reduce)(like.RA, like.c, like.s_perp, phi)
        return d[:, :nc2], E[:, :nc2, :nc2]

    def components(self, params):
        """(X (P, k), Z (P, k, k)) with k = 2 n_common."""
        return self._components(params)

    def phihat(self, gamma):
        nc2 = self.nc2
        return _basis.powerlaw(self.like.f[:nc2], self.like.df[:nc2], 0.0, gamma)

    def _rho_sig_impl(self, params):
        X, Z = self._components_impl(params)
        ph = self.phihat(params["gamma"])
        num = (X * ph[None, :]) @ X.T  # (P, P)
        W = (ph[None, :, None] * Z * ph[None, None, :]).reshape(self.P, -1)
        den = W @ Z.reshape(self.P, -1).T  # tr(Z_a phi Z_b phi), Z symmetric
        num, den = num[self.ia, self.ib], den[self.ia, self.ib]
        return num / den, 1.0 / jnp.sqrt(den)

    def rho_sigma(self, params):
        """Pair cross-correlations rho_ab and their (null) uncertainties sigma_ab, pairs a < b."""
        rho, sig = self._rho_sig(params)
        return np.asarray(rho), np.asarray(sig)

    def os_from_pairs(self, rho, sig, orf_pairs=None):
        """(A^2, sigma_A2, S/N) from pair values; works on (..., n_pairs) arrays."""
        u = self.orf_pairs if orf_pairs is None else orf_pairs
        w = u**2 / sig**2
        a2 = np.sum(rho * u / sig**2, axis=-1) / np.sum(w, axis=-1)
        s = 1.0 / np.sqrt(np.sum(w, axis=-1))
        return a2, s, a2 / s

    def os(self, params) -> dict:
        rho, sig = self.rho_sigma(params)
        a2, s, snr = self.os_from_pairs(rho, sig)
        return {"A2": float(a2), "sigma": float(s), "snr": float(snr), "rho": rho, "sig": sig}

    def noise_marginalized(self, param_batches, batch: int = 64) -> dict:
        """OS at many parameter dicts. ``param_batches``: dict of arrays with leading dim N."""
        n = len(np.asarray(param_batches["log10_A"]))
        u = jnp.asarray(self.orf_pairs)

        def one(p):
            rho, sig = self._rho_sig_impl(p)
            w = u**2 / sig**2
            a2 = jnp.sum(rho * u / sig**2) / jnp.sum(w)
            s = 1.0 / jnp.sqrt(jnp.sum(w))
            return a2, s

        f = jax.jit(jax.vmap(one))
        a2s, ss = [], []
        for i in range(0, n, batch):
            pb = {k: jnp.asarray(np.asarray(v)[i : i + batch]) for k, v in param_batches.items()}
            m = len(pb["log10_A"])
            if m < batch:
                pb = {k: jnp.concatenate([v, jnp.repeat(v[-1:], batch - m, axis=0)]) for k, v in pb.items()}
            a2, s = f(pb)
            a2s.append(np.asarray(a2)[:m])
            ss.append(np.asarray(s)[:m])
        a2, s = np.concatenate(a2s), np.concatenate(ss)
        return {"A2": a2, "sigma": s, "snr": a2 / s}

    # -- pair covariance (Allen & Romano 2023 as implemented in the NG15 release) --
    def pair_covariance(self, params) -> np.ndarray:
        """Covariance of rho_ab including GW self-noise, reproducing ``OS.gw_corr`` of the NG15
        release (``optimal_statistic_covariances.py``) term by term, vectorised.

        With M_a = Z_a phi, T2 = tr(M_a M_b), T3 = tr(M_a M_b M_c), T4 = tr(M_a M_b M_c M_d) and
        A = 10^log10_A of ``params``. As in the release, the ORF factors use the searched ORF
        for the pairs and the traces carry no ORF.
        """
        _X, Z = (np.asarray(a) for a in self.components(params))
        ph = np.asarray(self.phihat(params["gamma"]))
        M = Z * ph[None, None, :]
        P, k = self.P, self.nc2
        Q = np.einsum("aij,bjk->abik", M, M)  # M_a M_b
        T2 = np.einsum("abii->ab", Q)
        T3 = np.einsum("abij,cji->abc", Q, M)
        Qf = Q.reshape(P * P, k * k)
        QTf = np.transpose(Q, (0, 1, 3, 2)).reshape(P * P, k * k)
        T4 = (Qf @ QTf.T).reshape(P, P, P, P)  # tr(Q_ab Q_cd)
        A2 = 10.0 ** (2.0 * float(params["log10_A"]))
        A4 = A2 * A2
        G = self.Gamma_full
        ia, ib = self.ia, self.ib
        npair = len(ia)
        I, J = ia[:, None], ib[:, None]
        K, L = ia[None, :], ib[None, :]
        I, J, K, L = (np.broadcast_to(a, (npair, npair)) for a in (I, J, K, L))
        S = np.zeros((npair, npair))
        same = (I == K) & (J == L)
        c_ik = (I == K) & (J != L)
        c_jl = (I != K) & (J == L)
        distinct = (I != K) & (J != L) & (I != L) & (J != K)
        c_jk = (J == K) & ~same & ~c_ik & ~c_jl & ~distinct
        c_il = (I == L) & ~same & ~c_ik & ~c_jl & ~distinct & ~c_jk
        g = lambda a, b: G[a, b]
        S = np.where(same, T2[I, J] + A4 * T4[I, J, I, J] * g(I, J) ** 2, S)
        S = np.where(c_ik, A2 * T3[I, J, L] * g(J, L) + A4 * T4[I, J, I, L] * g(I, L) * g(I, J), S)
        S = np.where(c_jl, A2 * T3[J, K, I] * g(I, K) + A4 * T4[I, J, K, J] * g(I, J) * g(K, J), S)
        S = np.where(
            distinct, A4 * g(I, L) * g(K, J) * T4[I, J, K, L] + A4 * g(I, K) * g(J, L) * T4[I, J, L, K], S
        )
        S = np.where(c_jk, A4 * g(I, J) * g(J, L) * T4[J, I, J, L] + A2 * g(I, L) * T3[J, I, L], S)
        # i == l with k < l = i < j is the mirror of the j == k case for the transposed pair
        S = np.where(c_il, A4 * g(K, L) * g(L, J) * T4[L, K, L, J] + A2 * g(K, J) * T3[L, K, J], S)
        # The release computes the upper triangle (kl >= ij) and mirrors it.
        iu = np.triu(np.ones((npair, npair), bool))
        S = np.where(iu, S, S.T)
        sig2 = 1.0 / T2[ia, ib]
        return S * sig2[:, None] * sig2[None, :]


def binned_correlations(xi, rho, orf_pairs, Crho, bin_edges, a2_norm=None):
    """Pair-covariance-aware binned estimator (Allen & Romano 2023), as the NG15 Fig. 1(c) notebook.

    Returns dict with the mean angle per bin, the binned correlation estimates rho_bin (in units
    of A^2), their covariance B (n_bins x n_bins), and, if ``a2_norm`` is given, the chi^2 of the
    binned values against A^2 HD(xi_mean) and the values normalised by ``a2_norm``.
    """
    bin_inds = np.digitize(xi, bin_edges) - 1
    nb = len(bin_edges) - 1
    masks = [bin_inds == i for i in range(nb)]
    xi_mean = np.array([np.mean(xi[m]) for m in masks])
    hdf = hd_curve(xi_mean)
    vals, invs, us = [], [], []
    for i, m in enumerate(masks):
        u = orf_pairs[m]
        Ci = np.linalg.inv(Crho[np.ix_(m, m)])
        X = 1.0 / (u @ Ci @ u)
        vals.append(hdf[i] * X * (u @ Ci @ rho[m]))
        invs.append(Ci)
        us.append(u)
    vals = np.array(vals)
    B = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            Cij = Crho[np.ix_(masks[i], masks[j])]
            top = us[i] @ invs[i] @ Cij @ invs[j] @ us[j]
            bot = (us[i] @ invs[i] @ us[i]) * (us[j] @ invs[j] @ us[j])
            B[i, j] = hdf[i] * hdf[j] * top / bot
    out = {
        "xi_mean": xi_mean,
        "n_pairs": np.array([int(m.sum()) for m in masks]),
        "rho_bin": vals,
        "sig_bin": np.sqrt(np.diag(B)),
        "cov": B,
    }
    if a2_norm is not None:
        r = vals - a2_norm * hdf
        out["chi2"] = float(r @ np.linalg.solve(B, r))
        out["rho_bin_norm"] = vals / a2_norm
        out["sig_bin_norm"] = np.sqrt(np.diag(B)) / a2_norm
    return out
