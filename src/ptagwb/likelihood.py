"""Marginalised PTA likelihood (CURN / correlated common process) in JAX, float64.

Model for pulsar a (GWB Eqs. 1-3)::

    r_a = M_a eps_a + F_a c_a + n_a,    n_a ~ N(0, N_a)   (fixed EFAC/EQUAD/ECORR)

with a flat improper prior on the timing offsets eps_a and Gaussian Fourier coefficients
c_a (30 sin/cos pairs). The first 14 pairs also carry the common process with
Cov(c_ak, c_bk) = delta_ab phi^RN_ak + Gamma_ab phi^CP_k.

Two stages
----------
1. **Parameter-independent precompute (host, numpy float64, once per pulsar).** Whiten with
   N_a^-1/2 (exact blockwise ECORR whitening, ``noise.WhiteNoise.whiten``), QR-factor the
   whitened timing basis, and project the whitened Fourier basis F_p and residuals r_p onto
   its orthogonal complement. A second QR, F_p = Q_F R_F, gives the *square-root*
   representation of the timing-marginalised contractions::

       R_F (60, 60) upper triangular,  c = Q_F^T r_p (60,),  s_perp = |r_p - Q_F c|^2
       (A = F^T P F = R_F^T R_F,  b = F^T P r = R_F^T c,  r^T P r = s_perp + |c|^2)

   plus log|N_a| and log|M^T N^-1 M|. No TOA-sized array is touched afterwards, and every
   pulsar's terms have the same shape, so the hot path is a dense stacked batch (no padding).
   A = R_F^T R_F is never formed in the hot path (forming it squares its condition number).

2. **Per evaluation (JAX, jit/grad).** With Phi_a the diagonal coefficient prior, the
   Woodbury/determinant lemmas on the projected operator (valid because the flat timing prior
   is the infinite-variance limit of a GP and GPs can be absorbed in any order) become, with
   Y = R_F Phi^1/2 and the reduced QR  [Y; I] = [Q1; Q2] T  (T^T T = I + Phi^1/2 A Phi^1/2)::

       y = Q1^T c,   -2 log L_a = s_perp + |c|^2 - |y|^2 + log|T^T T| + const_a

   Gradients of this stage use an analytic VJP (``_reduce``) in terms of E = (A^-1 + R)^-1 and
   d = E A^-1 b rather than JAX's generic QR derivative, which is inaccurate when R spans
   many decades. No normal-equation matrix is formed: QR of the stacked matrix works with the square root,
   so the ~1e20 condition number of I + Phi^1/2 A Phi^1/2 in the IRN prior corner
   (log10_A -> -11, gamma -> 7) is never squared.

   * CURN: Phi_a = phi^RN_a + phi^CP on the first 28 entries; separable.
   * Correlated common process (HD, monopole, dipole, any SPD Gamma): split
     Gamma = lam0 I + Gamma' with lam0 = 0.5 lambda_min(Gamma), and absorb per pulsar
     R = Phi^RN + lam0 phi^CP (on the common modes) in the square-root stage. This keeps
     phi^CP E <= 1/lam0 bounded when the common process dominates; without the split, Sigma'
     inherits cond(A) ~ 1e12 in that corner. The projected common-mode precision and score
     are::

         E_a = [(A^-1 + R)^-1]_GG = [R^-1/2 (I + R^1/2 A R^1/2)^-1 R^1/2 A]_GG
             = [R^-1/2 T^-1 Q1^T R_F]_G,G ,          d_a = [R^-1/2 T^-1 y]_G

     G = the 28 common columns, R = Phi^RN. No subtraction takes place. The earlier form
     E = A_GG - K^T K, d = b_G - K^T y cancelled catastrophically when the intrinsic RN
     dominates (relative error ~ eps |A| / |R^-1| ~ 1e-1 at the prior corner). Then with
     Q = Gamma' (x) diag(phi^CP)::

         Sigma' = Gamma'^-1 (x) I_28 + blockdiag(phi^1/2 E_a phi^1/2)    (67*28 = 1876 square)
         -2 log L = sum_a [...] + log|Sigma'| + 28 log|Gamma'| - dt^T Sigma'^-1 dt,  dt = phi^1/2 d

     using log|Q| + log|Q^-1 + E| = log|Sigma'| + 28 log|Gamma'|. ``method="B"`` instead
     factorises the congruent B = I + Z^T E Z (Z = chol(Gamma') (x) phi^1/2) with plain
     autodiff, as a reference. Accuracy against a 50-digit joint reference: tests/test_corners.py.

Constants (``convention``): ``"chain"`` reproduces the absolute value of discovery and of the
enterprise v3.3.1 likelihood that wrote the NG15 production chains' ``logl`` column
(includes the m log(1e40) timing-prior term, no 2 pi); ``"enterprise"`` additionally
subtracts n_toa/2 log(2 pi) as current enterprise (>= 3.4) does.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from . import basis as _basis
from . import orf as _orf
from .config import enable_x64
from .noise import WhiteNoise, build_white_noise

enable_x64()  # the likelihood is float64 throughout

TIMING_PRIOR_VARIANCE = 1e40  # enterprise / discovery improper timing-model prior


# ---------------------------------------------------------------------- precompute


TIMING_RCOND_MIN = 1e-10  # NG15 worst case: 4.9e-7 (J1853+1303)


class TimingRankError(ValueError):
    """The (column-normalised) timing design matrix is numerically rank deficient."""


def check_timing_rank(Mmat: np.ndarray, name: str = "", rcond_min: float = TIMING_RCOND_MIN) -> np.ndarray:
    """Singular values of the column-normalised design matrix; raise if s_min/s_max < rcond_min.

    A rank-deficient M would make the SVD basis contain arbitrary directions (and the
    unit-norm basis a singular M^T N^-1 M). We neither truncate nor regularise silently: fix the
    timing model (e.g. drop the duplicated/unconstrained parameter) instead.
    """
    Mmat = np.asarray(Mmat, dtype=np.float64)
    n, m = Mmat.shape
    if not np.all(np.isfinite(Mmat)):
        raise TimingRankError(f"{name}: non-finite entries in the timing design matrix")
    if m > n:
        raise TimingRankError(f"{name}: {m} timing parameters but only {n} TOAs (cannot have full column rank)")
    norms = np.linalg.norm(Mmat, axis=0)
    if np.any(norms == 0):
        raise TimingRankError(f"{name}: all-zero design-matrix column(s) {np.flatnonzero(norms == 0).tolist()}")
    sv = np.linalg.svd(Mmat / norms, compute_uv=False)
    if len(sv) != m or sv[-1] / sv[0] < rcond_min:
        raise TimingRankError(
            f"{name}: timing design matrix numerically rank deficient: s_min/s_max = {sv[-1] / sv[0]:.2e} "
            f"< {rcond_min:.0e} (column-normalised, {m} columns)"
        )
    return sv


def timing_basis(Mmat: np.ndarray, method: str = "svd", name: str = "") -> np.ndarray:
    """Timing-model basis: ``svd`` = thin-SVD left singular vectors (discovery
    ``makegp_timing(svd=True)``, enterprise ``use_svd=True``); ``normed`` = unit-norm
    columns (enterprise default). Fails on numerical rank deficiency (``check_timing_rank``)."""
    Mmat = np.asarray(Mmat, dtype=np.float64)
    check_timing_rank(Mmat, name)
    if method == "svd":
        U, _, _ = np.linalg.svd(Mmat, full_matrices=False)
        return U
    if method == "normed":
        return Mmat / np.sqrt(np.sum(Mmat**2, axis=0))
    raise ValueError(f"unknown timing basis method {method!r}")


@dataclass
class PulsarTerms:
    """Timing-marginalised, white-noise-whitened contractions of one pulsar."""

    name: str
    pos: np.ndarray  # (3,)
    ntoa: int
    m: int  # number of timing-model columns
    RA: np.ndarray  # (2 n_modes, 2 n_modes) upper-triangular R_F, F^T P F = R_F^T R_F
    c: np.ndarray  # (2 n_modes,) Q_F^T r_p
    s_perp: float  # |r_p - Q_F c|^2
    logdet_N: float
    logdet_MNM: float
    timing_singular_values: np.ndarray  # of the column-normalised design matrix

    @property
    def A(self) -> np.ndarray:
        """F^T P F (diagnostics / references only; never used in the likelihood)."""
        return self.RA.T @ self.RA

    @property
    def b(self) -> np.ndarray:
        return self.RA.T @ self.c

    @property
    def s(self) -> float:
        return float(self.s_perp + self.c @ self.c)

    def const(self, convention: str = "chain") -> float:
        c = self.logdet_N + self.logdet_MNM + self.m * np.log(TIMING_PRIOR_VARIANCE)
        if convention == "enterprise":
            c += self.ntoa * np.log(2.0 * np.pi)
        elif convention != "chain":
            raise ValueError(f"unknown convention {convention!r}")
        return float(c)


def precompute_pulsar(
    psr, wn: WhiteNoise, T: float, n_modes: int = 30, timing: str = "svd", position: str = "icrs"
) -> PulsarTerms:
    """Stage-1 contractions for one pulsar (numpy float64, host).

    ``position``: ``"icrs"`` (default, PINT ICRS unit vector) or ``"enterprise"`` (the
    vector enterprise / the released NG15 feathers use; differs for B-name pulsars, see
    ``data.enterprise_position``). Only the HD/monopole/dipole ORFs depend on it.
    """
    sv = check_timing_rank(psr.Mmat, psr.name)
    Mt = timing_basis(psr.Mmat, timing, psr.name)
    _, _, F = _basis.fourier_basis(psr.toas, n_modes, T)
    r = np.asarray(psr.residuals, dtype=np.float64)

    Wm = wn.whiten(Mt)
    Wf = wn.whiten(F)
    wr = wn.whiten(r)

    Q, R = np.linalg.qr(Wm, mode="reduced")
    logdet_MNM = 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))

    def project(X):
        X = X - Q @ (Q.T @ X)
        return X - Q @ (Q.T @ X)  # one re-orthogonalisation pass

    Fp, rp = project(Wf), project(wr)
    QF, RF = np.linalg.qr(Fp, mode="reduced")
    c = QF.T @ rp
    rperp = rp - QF @ c
    rperp = rperp - QF @ (QF.T @ rperp)
    if position == "icrs":
        pos = psr.pos
    elif position == "enterprise":
        pos = getattr(psr, "pos_enterprise", None)
        if pos is None:
            raise ValueError(f"{psr.name}: no enterprise position stored (re-ingest with the current schema)")
    else:
        raise ValueError(f"unknown position convention {position!r}")
    return PulsarTerms(
        name=psr.name,
        pos=np.asarray(pos, dtype=np.float64),
        ntoa=len(r),
        m=Mt.shape[1],
        RA=RF,
        c=c,
        s_perp=float(rperp @ rperp),
        logdet_N=wn.logdet(),
        logdet_MNM=logdet_MNM,
        timing_singular_values=sv,
    )


def precompute(
    psrs,
    noisedict: dict[str, float],
    T: float,
    n_modes: int = 30,
    timing: str = "svd",
    position: str = "icrs",
    **wn_kwargs,
) -> list[PulsarTerms]:
    return [
        precompute_pulsar(p, build_white_noise(p, noisedict, **wn_kwargs), T, n_modes, timing, position)
        for p in psrs
    ]


# ---------------------------------------------------------------------- JAX likelihood


def _chol_logdet(L):
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)


def _reduce_fwd_impl(RA, c, s_perp, r):
    """Square-root absorption of a diagonal coefficient prior r (>0) for one pulsar.

    QR of [R_F R^1/2; I] = [Q1; Q2] T gives T^T T = I + R^1/2 A R^1/2 without forming it. Returns

        q  = r^T P r - b^T (R^-1 + A)^-1 b          (= s_perp + |c|^2 - |Q1^T c|^2)
        ld = log|I + R^1/2 A R^1/2|
        E  = (A^-1 + R)^-1 = R^-1/2 T^-1 Q1^T R_F    (full n x n, symmetrised; no subtraction)
        d  = (A^-1 + R)^-1 A^-1 b = R^-1/2 T^-1 Q1^T c
    """
    sq = jnp.sqrt(r)
    n = RA.shape[-1]
    Qs, Tm = jnp.linalg.qr(jnp.concatenate([RA * sq[None, :], jnp.eye(n, dtype=RA.dtype)], axis=0))
    Q1 = Qs[:n]
    y = Q1.T @ c
    q = s_perp + c @ c - y @ y
    ld = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(Tm))))
    Z = jsl.solve_triangular(Tm, jnp.concatenate([Q1.T @ RA, y[:, None]], axis=1), lower=False) / sq[:, None]
    E = 0.5 * (Z[:, :n] + Z[:, :n].T)
    d = Z[:, n]
    return q, ld, E, d


@jax.custom_vjp
def _reduce(RA, c, s_perp, r):
    return _reduce_fwd_impl(RA, c, s_perp, r)


def _reduce_fwd(RA, c, s_perp, r):
    out = _reduce_fwd_impl(RA, c, s_perp, r)
    _, _, E, d = out
    return out, (E, d, RA, c, s_perp)


def _reduce_bwd(res, cts):
    """Analytic derivatives w.r.t. the diagonal prior r, in terms of the (accurately computed)
    E = (A^-1 + R)^-1 and d = E A^-1 b only:

        dq/dr_k = -d_k^2,   dld/dr_k = E_kk,   dE/dr_k = -E e_k e_k^T E,   dd/dr_k = -E e_k d_k

    Differentiating through the QR (JAX's generic rule) instead loses up to ~1e-4 relative
    accuracy when R spans many decades (e.g. free spectrum at log10_rho -> -1 with IRN at the
    prior corner). The data (R_F, c, s_perp) are constants: zero cotangents.
    """
    E, d, RA, c, s_perp = res
    qb, ldb, Eb, db = cts
    Eb = 0.5 * (Eb + Eb.T)
    rb = -qb * d * d + ldb * jnp.diagonal(E) - jnp.sum((E @ Eb) * E, axis=1) - d * (E @ db)
    return jnp.zeros_like(RA), jnp.zeros_like(c), jnp.zeros_like(s_perp), rb


_reduce.defvjp(_reduce_fwd, _reduce_bwd)


def _tri_inv_lower(L, base: int = 128):
    """Inverse of a lower-triangular matrix by recursive blocking (GEMM-rich; ~n^3/3 flops)."""
    n = L.shape[0]
    if n <= base:
        return jsl.solve_triangular(L, jnp.eye(n, dtype=L.dtype), lower=True)
    h = n // 2
    A = _tri_inv_lower(L[:h, :h], base)
    C = _tri_inv_lower(L[h:, h:], base)
    X = -(C @ (L[h:, :h] @ A))
    return jnp.block([[A, jnp.zeros((h, n - h), L.dtype)], [X, C]])


def _core_B(Es, dt, Lg):
    """log|B| - z^T B^-1 z with B = I + Z'^T blockdiag(Es) Z', z = Z'^T dt, Z' = chol(Gamma) (x) I.

    Reference formulation (plain autodiff). Es: (P, k, k), dt: (P, k).
    """
    P, k = dt.shape
    B = jnp.einsum("ab,ajk,ac->bjck", Lg, Es, Lg).reshape(P * k, P * k) + jnp.eye(P * k)
    z = jnp.einsum("ab,aj->bj", Lg, dt).reshape(-1)
    LB = jnp.linalg.cholesky(B)
    w = jsl.solve_triangular(LB, z, lower=True)
    return _chol_logdet(LB) - w @ w


def _make_core_sigma(Ginv, k: int, grad_precision: str = "float64"):
    """log|Sigma'| - dt^T Sigma'^-1 dt with Sigma' = Gamma^-1 (x) I_k + blockdiag(Es).

    Equal to ``_core_B`` minus k log|Gamma| (B = Z'^T Sigma' Z'). Sigma' is assembled without any
    P^3 contraction, and the custom VJP needs only the diagonal k x k blocks of Sigma'^-1::

        d/dEs_a = [Sigma'^-1]_aa + u_a u_a^T,   d/ddt = -2 u,   u = Sigma'^-1 dt

    obtained from the triangular inverse of the forward Cholesky factor (n^3/3 flops instead
    of the ~3 n^3 of differentiating through ``cholesky``). With ``grad_precision="mixed"``
    that triangular inverse is computed in float32 on the symmetrically equilibrated factor
    (value still float64; see docs/M1_VALIDATION.md for the measured gradient error).
    """
    P = Ginv.shape[0]
    n = P * k
    eye_k = jnp.eye(k)

    def assemble(Es):
        Sig = jnp.kron(Ginv, eye_k)
        blocks = jnp.zeros((P, k, P, k), dtype=Es.dtype)
        idx = jnp.arange(P)
        blocks = blocks.at[idx, :, idx, :].set(Es)
        return Sig + blocks.reshape(n, n)

    def fwd(Es, dt):
        L = jnp.linalg.cholesky(assemble(Es))
        w = jsl.solve_triangular(L, dt.reshape(-1), lower=True)
        return _chol_logdet(L) - w @ w, (L, w)

    def bwd(res, g):
        L, w = res
        u = jsl.solve_triangular(L.T, w, lower=False)
        if grad_precision == "mixed":
            dsc = 1.0 / jnp.sqrt(jnp.sum(L * L, axis=1))  # diag(Sigma')^-1/2
            Leq = (dsc[:, None] * L).astype(jnp.float32)
            Linv = _tri_inv_lower(Leq).astype(jnp.float64) * dsc[None, :]
        else:
            Linv = _tri_inv_lower(L)
        G = Linv.reshape(n, P, k)
        S = jnp.einsum("iaj,iak->ajk", G, G)
        ua = u.reshape(P, k)
        dEs = g * (S + ua[:, :, None] * ua[:, None, :])
        ddt = g * (-2.0 * ua)
        return dEs, ddt

    @jax.custom_vjp
    def core(Es, dt):
        return fwd(Es, dt)[0]

    core.defvjp(fwd, bwd)
    return core


class PTALikelihood:
    """Marginalised likelihood of a PTA with intrinsic RN and one common process.

    Parameters (``logL(params)`` takes a dict of JAX/numpy arrays):

    * ``rn_log10_A``, ``rn_gamma``: shape (P,), intrinsic red noise of every pulsar.
    * common power law: ``log10_A``, ``gamma`` (scalars), or
      common free spectrum (``common="freespec"``): ``log10_rho`` shape (n_common,).

    ``orf``: ``"curn"`` (separable), ``"hd"``, ``"monopole"``, ``"dipole"`` or an explicit
    (P, P) SPD matrix. ``"curn"`` uses the separable per-pulsar path; any other ORF goes
    through the reduced 1876-dimensional system (``orf=np.eye(P)`` also works and must
    reproduce CURN; see tests).

    ``method``: ``"sigma"`` (default) factorises Sigma' = Gamma^-1 (x) I + blockdiag(E_a phi)
    with a hand-written VJP; ``"B"`` factorises B = I + Z^T E Z with plain autodiff
    (reference). Both give the same value (tests). ``grad_precision="mixed"`` computes only
    the backward-pass triangular inverse in float32 (value stays float64).
    """

    def __init__(
        self,
        terms: list[PulsarTerms],
        T: float,
        *,
        n_modes: int = 30,
        n_common: int = 14,
        orf: str | np.ndarray = "curn",
        common: str = "powerlaw",
        convention: str = "chain",
        orf_diag_eps: float = 1e-5,
        method: str = "sigma",
        grad_precision: str = "float64",
        split_fraction: float = 0.5,
    ):
        self.names = [t.name for t in terms]
        self.P = len(terms)
        self.n_modes, self.n_common, self.T = n_modes, n_common, float(T)
        self.common, self.convention = common, convention
        if common not in ("powerlaw", "freespec"):
            raise ValueError(f"unknown common spectrum {common!r}")
        if any(t.RA.shape != (2 * n_modes, 2 * n_modes) for t in terms):
            raise ValueError("precomputed terms have a different number of Fourier modes")

        f, df = _basis.fourier_frequencies(n_modes, T)
        self.f, self.df = jnp.asarray(f), jnp.asarray(df)
        self.RA = jnp.asarray(np.stack([t.RA for t in terms]))
        self.c = jnp.asarray(np.stack([t.c for t in terms]))
        self.s_perp = jnp.asarray(np.array([t.s_perp for t in terms]))
        self.const_total = float(sum(t.const(convention) for t in terms))
        self.pos = np.stack([t.pos for t in terms])

        if isinstance(orf, str):
            self.orf_name = orf
            if orf == "curn":
                G = None
            elif orf in ("monopole", "dipole"):
                G = _orf.ORFS[orf](self.pos, diag_eps=orf_diag_eps)
            else:
                G = _orf.ORFS[orf](self.pos)
        else:
            self.orf_name = "custom"
            G = np.asarray(orf, dtype=np.float64)
        self.Gamma = G
        if method not in ("sigma", "B"):
            raise ValueError(f"unknown method {method!r}")
        if grad_precision not in ("float64", "mixed"):
            raise ValueError(f"unknown grad_precision {grad_precision!r}")
        self.method, self.grad_precision = method, grad_precision
        if G is not None:
            if G.shape != (self.P, self.P) or not np.all(np.isfinite(G)) or not np.allclose(G, G.T, rtol=0, atol=1e-14):
                raise ValueError("ORF matrix must be a finite, symmetric (P, P) array")
            if not (np.isfinite(split_fraction) and 0.0 < split_fraction < 1.0):
                raise ValueError(f"split_fraction must be finite and in (0, 1), got {split_fraction!r}")
            # Diagonal splitting Gamma = lam0 I + Gamma', lam0 = split_fraction * lambda_min(Gamma).
            # The lam0 phi^CP part is absorbed per pulsar together with the intrinsic RN (square-root
            # QR stage), so phi^1/2 E' phi^1/2 <= 1/lam0 stays bounded even when the common
            # process dominates; without this, Sigma' inherits cond(A) ~ 1e12 in the corner where
            # the common power is huge and the IRN negligible.
            lam_min = float(np.linalg.eigvalsh(G)[0])
            if lam_min <= 0:
                raise ValueError(f"ORF matrix not positive definite (lambda_min = {lam_min:.3e})")
            self.lam0 = split_fraction * lam_min
            G = G - self.lam0 * np.eye(self.P)
            Lg = np.linalg.cholesky(G)
            Ginv = np.linalg.solve(Lg.T, np.linalg.solve(Lg, np.eye(self.P)))
            Ginv = 0.5 * (Ginv + Ginv.T)
            self.Lgamma = jnp.asarray(Lg)
            # log|B| = log|Sigma'| + 2 n_common log|Gamma|
            self._logdet_gamma_term = 2 * n_common * 2.0 * float(np.sum(np.log(np.diag(Lg))))
            self._core_sigma = _make_core_sigma(jnp.asarray(Ginv), 2 * n_common, grad_precision)
        else:
            self.Lgamma = None
            self.lam0 = 0.0

        self.logL = jax.jit(self._logL)
        self.value_and_grad = jax.jit(jax.value_and_grad(self._logL))

    # -- spectra --
    def phi_rn(self, rn_log10_A, rn_gamma):
        return _basis.powerlaw(self.f[None, :], self.df[None, :], rn_log10_A[:, None], rn_gamma[:, None])

    def phi_common(self, params):
        nc2 = 2 * self.n_common
        if self.common == "powerlaw":
            return _basis.powerlaw(self.f[:nc2], self.df[:nc2], params["log10_A"], params["gamma"])
        return _basis.free_spectrum(jnp.asarray(params["log10_rho"]))

    # -- likelihood --
    def _logL(self, params):
        rn_A = jnp.asarray(params["rn_log10_A"], dtype=jnp.float64)
        rn_g = jnp.asarray(params["rn_gamma"], dtype=jnp.float64)
        phi = self.phi_rn(rn_A, rn_g)  # (P, 2 n_modes)
        phic = self.phi_common(params)  # (2 n_common,)
        nc2 = 2 * self.n_common

        if self.Lgamma is None:  # CURN: common power on the diagonal, separable
            phi = phi.at[:, :nc2].add(phic[None, :])
            q, ld, _, _ = jax.vmap(_reduce)(self.RA, self.c, self.s_perp, phi)
            return -0.5 * (jnp.sum(q + ld) + self.const_total)

        phi = phi.at[:, :nc2].add(self.lam0 * phic[None, :])  # IRN + lam0 * common (diagonal part)
        q, ld, E, d = jax.vmap(_reduce)(self.RA, self.c, self.s_perp, phi)
        E, d = E[:, :nc2, :nc2], d[:, :nc2]

        sqc = jnp.sqrt(phic)
        Es = E * sqc[None, :, None] * sqc[None, None, :]
        dt = d * sqc[None, :]
        if self.method == "B":
            core = _core_B(Es, dt, self.Lgamma)
        else:
            core = self._core_sigma(Es, dt) + self._logdet_gamma_term
        return -0.5 * (jnp.sum(q + ld) + self.const_total + core)

    # -- parameter helpers --
    def params_from_named(
        self,
        named: dict[str, float],
        rn_fmt: str = "{psr}_red_noise_{par}",
        common_names: dict[str, str] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Build ``logL`` arguments from enterprise/discovery-style parameter names.

        ``common_names`` maps our common parameter (``log10_A``, ``gamma``, ``log10_rho``)
        to the external name; defaults to the NG15 chain names ``gw_log10_A``/``gw_gamma``.
        """
        cn = common_names or {"log10_A": "gw_log10_A", "gamma": "gw_gamma", "log10_rho": "gw_log10_rho"}
        out = {
            "rn_log10_A": jnp.array([named[rn_fmt.format(psr=n, par="log10_A")] for n in self.names]),
            "rn_gamma": jnp.array([named[rn_fmt.format(psr=n, par="gamma")] for n in self.names]),
        }
        if self.common == "powerlaw":
            out["log10_A"] = jnp.asarray(named[cn["log10_A"]], dtype=jnp.float64)
            out["gamma"] = jnp.asarray(named[cn["gamma"]], dtype=jnp.float64)
        else:
            key = cn["log10_rho"]
            out["log10_rho"] = jnp.array(
                named[key] if key in named else [named[f"{key}_{i}"] for i in range(self.n_common)]
            )
        return out
