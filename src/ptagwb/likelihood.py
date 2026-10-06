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
   whitened timing basis, and project the whitened Fourier basis and residuals onto its
   orthogonal complement. This yields the timing-marginalised contractions::

       s_a = r^T P r,   b_a = F^T P r  (60,),   A_a = F^T P F  (60, 60),
       P   = N^-1 - N^-1 M (M^T N^-1 M)^-1 M^T N^-1

   plus log|N_a| and log|M^T N^-1 M|. No TOA-sized array is touched afterwards, and every
   pulsar's terms have the same shape, so the hot path is a dense stacked batch (no padding).

2. **Per evaluation (JAX, jit/grad).** With Phi_a the diagonal prior and the Woodbury and
   determinant lemmas applied to the *projected* operator P (valid because the flat timing
   prior is the infinite-variance limit of a GP and the GPs can be absorbed in any order)::

       Shat_a = I + Phi_a^1/2 A_a Phi_a^1/2 = L_a L_a^T,   y_a = L_a^-1 Phi_a^1/2 b_a
       -2 log L_a = s_a - |y_a|^2 + log|Shat_a| + log|N_a| + log|M^T N^-1 M| + m_a log(1e40)

   * CURN (``orf=None``): Phi_a = phi^RN_a + phi^CP on the first 28 entries; separable.
   * Correlated common process (HD, monopole, dipole, any SPD Gamma): absorb only the
     intrinsic RN per pulsar, which gives the RN-projected common-mode terms::

         K_a = L_a^-1 Phi_a^1/2 A_a[:, :28],  E_a = A_a[:28, :28] - K_a^T K_a,  d_a = b_a[:28] - K_a^T y_a

     then with Q = Gamma (x) diag(phi^CP) = Z Z^T, Z = chol(Gamma) (x) diag(sqrt phi^CP)::

         B = I + Z^T blockdiag(E_a) Z   (67*28 = 1876 square),   z = Z^T d
         -2 log L = sum_a [s_a - |y_a|^2 + log|Shat_a| + const_a] + log|B| - z^T B^-1 z

     using log|Q| + log|Q^-1 + E| = log|B| and d^T (Q^-1 + E)^-1 d = z^T B^-1 z. B >= I, so
     its Cholesky is well conditioned, and no explicit inverse of Q or Gamma is formed.

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


def timing_basis(Mmat: np.ndarray, method: str = "svd") -> np.ndarray:
    """Timing-model basis: ``svd`` = thin-SVD left singular vectors (discovery
    ``makegp_timing(svd=True)``, enterprise ``use_svd=True``); ``normed`` = unit-norm
    columns (enterprise default)."""
    Mmat = np.asarray(Mmat, dtype=np.float64)
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
    s: float  # r^T P r
    b: np.ndarray  # (2 n_modes,) F^T P r
    A: np.ndarray  # (2 n_modes, 2 n_modes) F^T P F
    logdet_N: float
    logdet_MNM: float
    timing_singular_values: np.ndarray  # of the raw design matrix (rank diagnostics)

    def const(self, convention: str = "chain") -> float:
        c = self.logdet_N + self.logdet_MNM + self.m * np.log(TIMING_PRIOR_VARIANCE)
        if convention == "enterprise":
            c += self.ntoa * np.log(2.0 * np.pi)
        elif convention != "chain":
            raise ValueError(f"unknown convention {convention!r}")
        return float(c)


def precompute_pulsar(
    psr, wn: WhiteNoise, T: float, n_modes: int = 30, timing: str = "svd"
) -> PulsarTerms:
    """Stage-1 contractions for one pulsar (numpy float64, host)."""
    Mt = timing_basis(psr.Mmat, timing)
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
    sv = np.linalg.svd(np.asarray(psr.Mmat, dtype=np.float64) / np.linalg.norm(psr.Mmat, axis=0), compute_uv=False)
    return PulsarTerms(
        name=psr.name,
        pos=np.asarray(psr.pos, dtype=np.float64),
        ntoa=len(r),
        m=Mt.shape[1],
        s=float(rp @ rp),
        b=Fp.T @ rp,
        A=Fp.T @ Fp,
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
    **wn_kwargs,
) -> list[PulsarTerms]:
    return [
        precompute_pulsar(p, build_white_noise(p, noisedict, **wn_kwargs), T, n_modes, timing)
        for p in psrs
    ]


# ---------------------------------------------------------------------- JAX likelihood


def _chol_logdet(L):
    return 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)


def _rn_absorb(A, b, s, phi):
    """Per pulsar: Shat = I + phi^1/2 A phi^1/2 = L L^T; returns (q, logdet Shat, L, y, sqrt phi)."""
    sq = jnp.sqrt(phi)
    n = A.shape[-1]
    S = jnp.eye(n) + sq[:, None] * A * sq[None, :]
    L = jnp.linalg.cholesky(S)
    y = jsl.solve_triangular(L, sq * b, lower=True)
    return s - y @ y, _chol_logdet(L), L, y, sq


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
    ):
        self.names = [t.name for t in terms]
        self.P = len(terms)
        self.n_modes, self.n_common, self.T = n_modes, n_common, float(T)
        self.common, self.convention = common, convention
        if common not in ("powerlaw", "freespec"):
            raise ValueError(f"unknown common spectrum {common!r}")
        if any(t.A.shape != (2 * n_modes, 2 * n_modes) for t in terms):
            raise ValueError("precomputed terms have a different number of Fourier modes")

        f, df = _basis.fourier_frequencies(n_modes, T)
        self.f, self.df = jnp.asarray(f), jnp.asarray(df)
        self.A = jnp.asarray(np.stack([t.A for t in terms]))
        self.b = jnp.asarray(np.stack([t.b for t in terms]))
        self.s = jnp.asarray(np.array([t.s for t in terms]))
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
            Lg = np.linalg.cholesky(G)
            Ginv = np.linalg.solve(Lg.T, np.linalg.solve(Lg, np.eye(self.P)))
            Ginv = 0.5 * (Ginv + Ginv.T)
            self.Lgamma = jnp.asarray(Lg)
            # log|B| = log|Sigma'| + 2 n_common log|Gamma|
            self._logdet_gamma_term = 2 * n_common * 2.0 * float(np.sum(np.log(np.diag(Lg))))
            self._core_sigma = _make_core_sigma(jnp.asarray(Ginv), 2 * n_common, grad_precision)
        else:
            self.Lgamma = None

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
            q, ld, _, _, _ = jax.vmap(_rn_absorb)(self.A, self.b, self.s, phi)
            return -0.5 * (jnp.sum(q + ld) + self.const_total)

        q, ld, L, y, sq = jax.vmap(_rn_absorb)(self.A, self.b, self.s, phi)
        # RN-projected common-mode terms E_a, d_a
        X = sq[:, :, None] * self.A[:, :, :nc2]  # (P, 2n, nc2)
        K = jax.vmap(lambda L_, X_: jsl.solve_triangular(L_, X_, lower=True))(L, X)
        E = self.A[:, :nc2, :nc2] - jnp.einsum("pij,pik->pjk", K, K)
        d = self.b[:, :nc2] - jnp.einsum("pij,pi->pj", K, y)

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
