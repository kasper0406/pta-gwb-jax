"""General multi-block, multi-leg marginalised likelihood (docs/M3_PLAN.md Sec. 4.3; M3a).

Generalises ``likelihood.PTALikelihood`` (M1) in three ways, keeping its two-stage square-root
numerics and analytic VJPs unchanged:

1. **Distinct Fourier grids.** Each pulsar carries sampled GP blocks (intrinsic red noise,
   DM with nu^-2 scaling, extra chromatic / band / system blocks), each with its own span T,
   mode count, scaling and normalisation (``gp.FourierBlock``), plus the common-process block on
   the array grid. ``ColumnLayout`` takes the union of their columns: columns of blocks with the
   same ``basis_key`` and mode number are *the same column* (variances add); all other columns are
   kept separately. M1's prefix identity (common columns = first 2 n_common IRN columns) is the
   special case in which the common block has the IRN block's key. With distinct spans the
   common and IRN columns are distinct, nearly collinear columns; stage 1 keeps all of them and
   their cross-contractions (R_F is the QR factor of the full projected basis).
2. **Fixed blocks and general white noise in stage 1.** Fixed-hyperparameter GP blocks B (prior
   Phi_B) are absorbed into the effective noise C0 = N + B Phi_B B^T in square-root form: with
   W^T W = N^-1, QR of the augmented matrix [[W B Phi_B^1/2, W M], [I, 0]] gives
   log|C0| = log|N| + 2 sum log|R_BB|, log|M^T C0^-1 M| = 2 sum log|R_MM| and an orthonormal
   basis whose complement carries exactly the timing-marginalised C0^-1 inner products
   (Woodbury: x^T C0^-1 y = [Wx; 0]^T (I - P_[WB Phi^1/2; I]) [Wy; 0]). F and r are projected in
   that augmented space. N may be ``noise.WhiteNoise`` (disjoint ECORR, M1) or
   ``noise.GeneralWhiteNoise`` (overlapping ECORR, TN/T2 EQUAD, namespaced systems).
3. **Variable K_a.** Pulsars with fewer columns are padded to K_max with zero columns of R_F and
   unit prior variance. In the square-root reduction a zero column of R_F contributes an exact
   identity block to T, zero rows/columns to E and zero entries to d, so the value and the
   gradient are unchanged (exactly, not approximately; see tests).

Stage 2 is M1's: per pulsar the diagonal prior Phi (sum of the sampled processes' variances on
their columns; plus phi_CP on the common columns for CURN, or lam0 phi_CP for a correlated ORF) is
absorbed by the square-root reducer (``likelihood._reduce`` or the structured-Householder
``perf_likelihood`` reducer), and correlated ORFs go through the same Sigma' core with the same
custom VJP. No new backward contraction is introduced (the only new operations are gathers and
scatter-adds, differentiated by JAX), so the XLA:CPU YNNPACK defence of the reducers' backward
rules (optimization barriers) covers this path as well.

NG15 regression (gate G9): with the M1 white noise, the IRN block on the array span and the
common block on the same grid, ``precompute_general`` performs M1's stage-1 operations in M1's
order and ``GeneralPTALikelihood`` M1's stage-2 operations, so values agree bit-for-bit or to
rounding (tests/test_m3a_ng15_regression.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import basis as _basis
from . import likelihood as _L
from . import orf as _orf
from .config import enable_x64
from .gp import FourierBlock

enable_x64()


# ---------------------------------------------------------------------- column layout


@dataclass
class PulsarGPModel:
    """GP content of one (multi-leg) pulsar.

    ``sampled``: process name -> block (power-law hyperparameters sampled in stage 2).
    ``common``: the common-process block (array grid) or None.
    ``fixed``: (block, phi) pairs with fixed coefficient variances phi (2 n_modes,), absorbed in
    stage 1.
    """

    sampled: dict = field(default_factory=dict)
    common: FourierBlock | None = None
    fixed: list = field(default_factory=list)


class ColumnLayout:
    """Union of the columns of the sampled and common blocks (see module docstring)."""

    def __init__(self, model: PulsarGPModel):
        self.model = model
        procs = list(model.sampled.items()) + ([("__common__", model.common)] if model.common is not None else [])
        self.columns: list[tuple] = []  # (basis_key, j)
        index: dict[tuple, int] = {}
        self.blocks: dict[tuple, FourierBlock] = {}  # basis_key -> block with the most modes
        self.proc_cols: dict[str, np.ndarray] = {}
        for name, blk in procs:
            key = blk.basis_key
            if key not in self.blocks or self.blocks[key].n_modes < blk.n_modes:
                self.blocks[key] = blk
            cols = []
            for j in range(2 * blk.n_modes):
                cid = (key, j)
                if cid not in index:
                    index[cid] = len(self.columns)
                    self.columns.append(cid)
                cols.append(index[cid])
            self.proc_cols[name] = np.asarray(cols, dtype=np.int64)
        self.K = len(self.columns)

    @property
    def common_cols(self) -> np.ndarray | None:
        return self.proc_cols.get("__common__")

    def basis(self, toas, freqs, systems=None) -> np.ndarray:
        F = np.empty((len(toas), self.K), dtype=np.float64)
        cache = {key: blk.basis(toas, freqs, systems) for key, blk in self.blocks.items()}
        for i, (key, j) in enumerate(self.columns):
            F[:, i] = cache[key][:, j]
        return F


# ---------------------------------------------------------------------- stage 1


@dataclass
class GeneralTerms:
    """Stage-1 contractions of one pulsar (generalised ``likelihood.PulsarTerms``)."""

    name: str
    pos: np.ndarray
    ntoa: int
    m: int
    RA: np.ndarray  # (K, K)
    c: np.ndarray  # (K,)
    s_perp: float
    logdet_N: float  # log|C0|, C0 = N + fixed GP blocks
    logdet_MNM: float  # log|M^T C0^-1 M|
    timing_singular_values: np.ndarray
    layout: ColumnLayout
    n_fixed: int = 0
    projector: object = None  # residual -> (c, s_perp), if requested

    @property
    def K(self) -> int:
        return self.RA.shape[0]

    @property
    def A(self) -> np.ndarray:
        return self.RA.T @ self.RA

    def const(self, convention: str = "chain") -> float:
        c = self.logdet_N + self.logdet_MNM + self.m * np.log(_L.TIMING_PRIOR_VARIANCE)
        if convention == "enterprise":
            c += self.ntoa * np.log(2.0 * np.pi)
        elif convention != "chain":
            raise ValueError(f"unknown convention {convention!r}")
        return float(c)


class _Projector:
    """Maps residual vectors (n,) or (n, R) to (c, s_perp) with stage 1's operators."""

    def __init__(self, wn, Q, QF, kaug):
        self.wn, self.Q, self.QF, self.kaug = wn, Q, QF, kaug

    def __call__(self, r):
        r = np.asarray(r, dtype=np.float64)
        wr = self.wn.whiten(r)
        if self.kaug:
            wr = np.concatenate([wr, np.zeros((self.kaug,) + wr.shape[1:])], axis=0)
        rp = wr - self.Q @ (self.Q.T @ wr)
        rp = rp - self.Q @ (self.Q.T @ rp)
        c = self.QF.T @ rp
        rperp = rp - self.QF @ c
        rperp = rperp - self.QF @ (self.QF.T @ rperp)
        return c, np.sum(rperp * rperp, axis=0)


def precompute_general(
    psr,
    wn,
    model: PulsarGPModel,
    *,
    timing: str = "svd",
    position: str = "icrs",
    systems=None,
    projector: bool = False,
    allow_inadmissible: bool = False,
) -> GeneralTerms:
    """Stage-1 contractions for one (multi-leg) pulsar with any block layout and fixed blocks.

    ``psr``: anything with ``toas, residuals, freqs, Mmat, name, pos`` (``data.Pulsar`` or a
    stacked ``multileg.MultiLegPulsar``). ``systems``: per-TOA labels for flag-selected blocks
    (default ``psr.backend_flags``). ``wn``: ``WhiteNoise`` or ``GeneralWhiteNoise``.
    """
    meta = getattr(psr, "meta", None) or {}
    if meta.get("admissible") is False and not allow_inadmissible:
        raise ValueError(f"{psr.name}: inadmissible multi-leg build ({meta.get('config')}; "
                         f"{meta.get('admissibility')}); allow_inadmissible=True only for diagnostics")
    layout = ColumnLayout(model)
    sv = _L.check_timing_rank(psr.Mmat, psr.name)
    Mt = _L.timing_basis(psr.Mmat, timing, psr.name)
    systems = psr.backend_flags if systems is None else systems
    F = layout.basis(psr.toas, psr.freqs, systems)
    r = np.asarray(psr.residuals, dtype=np.float64)

    Wm = wn.whiten(Mt)
    Wf = wn.whiten(F)
    wr = wn.whiten(r)
    logdet_N = wn.logdet()
    kaug = 0
    if not model.fixed:
        # M1 (likelihood.precompute_pulsar), operation for operation
        Q, R = np.linalg.qr(Wm, mode="reduced")
        logdet_MNM = 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))
    else:
        Bs = []
        for blk, phi in model.fixed:
            phi = np.asarray(phi, dtype=np.float64)
            if phi.shape != (2 * blk.n_modes,) or np.any(phi < 0) or not np.all(np.isfinite(phi)):
                raise ValueError(f"{psr.name}: fixed block {blk.name!r} needs finite phi >= 0 of shape (2 n_modes,)")
            Bs.append(blk.basis(psr.toas, psr.freqs, systems) * np.sqrt(phi)[None, :])
        WB = wn.whiten(np.concatenate(Bs, axis=1))
        kaug = WB.shape[1]
        n, m = Wm.shape
        X = np.zeros((n + kaug, kaug + m))
        X[:n, :kaug] = WB
        X[n:, :kaug] = np.eye(kaug)
        X[:n, kaug:] = Wm
        Q, R = np.linalg.qr(X, mode="reduced")
        d = np.log(np.abs(np.diag(R)))
        logdet_N += 2.0 * float(np.sum(d[:kaug]))
        logdet_MNM = 2.0 * float(np.sum(d[kaug:]))
        Wf = np.concatenate([Wf, np.zeros((kaug, Wf.shape[1]))], axis=0)
        wr = np.concatenate([wr, np.zeros(kaug)])

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
            raise ValueError(f"{psr.name}: no enterprise position stored")
    else:
        raise ValueError(f"unknown position convention {position!r}")
    return GeneralTerms(
        name=psr.name,
        pos=np.asarray(pos, dtype=np.float64),
        ntoa=len(r),
        m=Mt.shape[1],
        RA=RF,
        c=c,
        s_perp=float(rperp @ rperp),
        logdet_N=logdet_N,
        logdet_MNM=logdet_MNM,
        timing_singular_values=sv,
        layout=layout,
        n_fixed=kaug,
        projector=_Projector(wn, Q, QF, kaug) if projector else None,
    )


# ---------------------------------------------------------------------- stage 2


class GeneralPTALikelihood:
    """Marginalised likelihood over pulsars with general block layouts (module docstring).

    ``terms``: list of ``GeneralTerms``. Every pulsar's sampled processes must be power laws with
    parameters ``<process>_log10_A`` and ``<process>_gamma``, arrays over the pulsars that carry
    the process, in ``terms`` order (``process_pulsars(name)``); with process "rn" on every pulsar
    these are M1's ``rn_log10_A``/``rn_gamma``. Common process: ``log10_A``/``gamma``
    (``common="powerlaw"``) or ``log10_rho`` (``"freespec"``), as in M1; every pulsar must have the
    same common block (array grid), or none of them (``common=None``: noise-only likelihood).

    ``orf``/``method``/``convention``/``split_fraction``/``grad_precision`` as in M1. ``reduce``:
    "prod" (M1's reducer) or "hh" (``perf_likelihood`` structured Householder); ``tri_inv``:
    "recursive" (M1) or "levels" (``perf_likelihood``). Defaults reproduce ``PTALikelihood``.
    """

    def __init__(
        self,
        terms: list[GeneralTerms],
        *,
        orf: str | np.ndarray | None = "curn",
        common: str | None = "powerlaw",
        convention: str = "chain",
        orf_diag_eps: float = 1e-5,
        split_fraction: float = 0.5,
        reduce: str = "prod",
        tri_inv: str = "recursive",
        grad_precision: str = "float64",
        buckets: tuple | None = None,
    ):
        from . import perf_likelihood as _perf

        self.terms = terms
        self.names = [t.name for t in terms]
        self.P = len(terms)
        self.common, self.convention = common, convention
        self.K = max(t.K for t in terms)
        P, K = self.P, self.K
        sink = P * K

        # ---- stacked stage-1 terms, zero-padded to K
        RA = np.zeros((P, K, K))
        c = np.zeros((P, K))
        pad_flat = []
        for a, t in enumerate(terms):
            RA[a, : t.K, : t.K] = t.RA
            c[a, : t.K] = t.c
            pad_flat += [a * K + j for j in range(t.K, K)]
        self.RA, self.c = jnp.asarray(RA), jnp.asarray(c)
        self.s_perp = jnp.asarray(np.array([t.s_perp for t in terms]))
        self.const_total = float(sum(t.const(convention) for t in terms))
        self.pos = np.stack([t.pos for t in terms])
        self._pad_flat = jnp.asarray(np.asarray(pad_flat, dtype=np.int64))
        self._sink = sink

        # ---- sampled processes: frequencies and scatter indices (padded entries -> sink)
        self.processes: dict[str, dict] = {}
        names = []
        for t in terms:
            for nm in t.layout.model.sampled:
                if nm not in names:
                    names.append(nm)
        for nm in names:
            psr_idx = [a for a, t in enumerate(terms) if nm in t.layout.model.sampled]
            L = max(2 * terms[a].layout.model.sampled[nm].n_modes for a in psr_idx)
            f = np.ones((len(psr_idx), L))
            df = np.ones((len(psr_idx), L))
            idx = np.full((len(psr_idx), L), sink, dtype=np.int64)
            for i, a in enumerate(psr_idx):
                blk = terms[a].layout.model.sampled[nm]
                fa, dfa = blk.frequencies()
                f[i, : len(fa)], df[i, : len(fa)] = fa, dfa
                idx[i, : len(fa)] = a * K + terms[a].layout.proc_cols[nm]
            self.processes[nm] = {"psr_idx": psr_idx, "f": jnp.asarray(f), "df": jnp.asarray(df), "idx": jnp.asarray(idx)}

        # ---- common process
        self.G = None
        if common is not None:
            blks = [t.layout.model.common for t in terms]
            if any(b is None for b in blks) or len({(b.basis_key, b.n_modes) for b in blks}) != 1:
                raise ValueError("every pulsar needs the same common block (array grid)")
            cb = blks[0]
            self.common_block = cb
            self.n_common = cb.n_modes
            fc, dfc = cb.frequencies()
            self.fc, self.dfc = jnp.asarray(fc), jnp.asarray(dfc)
            G = np.stack([t.layout.common_cols for t in terms])
            self.G = jnp.asarray(G)
            self._G_flat = jnp.asarray(G + (np.arange(P) * K)[:, None])
            if common not in ("powerlaw", "freespec"):
                raise ValueError(f"unknown common spectrum {common!r}")
        elif orf not in (None, "curn"):
            raise ValueError("a correlated ORF needs a common process")

        # ---- ORF (as M1)
        if common is None or orf is None or (isinstance(orf, str) and orf == "curn"):
            Gm = None
            self.orf_name = "curn" if common is not None else "none"
        elif isinstance(orf, str):
            self.orf_name = orf
            Gm = _orf.ORFS[orf](self.pos, diag_eps=orf_diag_eps) if orf in ("monopole", "dipole") else _orf.ORFS[orf](self.pos)
        else:
            self.orf_name = "custom"
            Gm = np.asarray(orf, dtype=np.float64)
        self.Gamma = Gm
        if reduce not in _perf.REDUCERS:
            raise ValueError(f"unknown reduce {reduce!r}")
        self._reducer = _perf.REDUCERS[reduce]
        # N8 stage-2 size buckets: pulsars grouped by K_a; each group is reduced at its own size
        # (bucket edge), padded only to that edge (the zero-column / unit-prior padding is exact)
        self.buckets = None
        if buckets is not None:
            Ks = np.array([t.K for t in terms])
            edges = sorted(int(e) for e in buckets)
            if edges[-1] < K:
                edges.append(K)
            groups, lo = [], 0
            for e in edges:
                idx = np.flatnonzero((Ks > lo) & (Ks <= e))
                if idx.size:
                    groups.append((jnp.asarray(idx), int(min(e, K))))
                lo = e
            self.buckets = groups
            self._bucket_order = jnp.asarray(np.argsort(np.concatenate([np.asarray(g[0]) for g in groups])))
        self.lam0 = 0.0
        self.Lgamma = None
        if Gm is not None:
            if Gm.shape != (P, P) or not np.all(np.isfinite(Gm)) or not np.allclose(Gm, Gm.T, rtol=0, atol=1e-14):
                raise ValueError("ORF matrix must be a finite, symmetric (P, P) array")
            if not (np.isfinite(split_fraction) and 0.0 < split_fraction < 1.0):
                raise ValueError(f"split_fraction must be finite and in (0, 1), got {split_fraction!r}")
            lam_min = float(np.linalg.eigvalsh(Gm)[0])
            if lam_min <= 0:
                raise ValueError(f"ORF matrix not positive definite (lambda_min = {lam_min:.3e})")
            self.lam0 = split_fraction * lam_min
            Gp = Gm - self.lam0 * np.eye(P)
            Lg = np.linalg.cholesky(Gp)
            Ginv = np.linalg.solve(Lg.T, np.linalg.solve(Lg, np.eye(P)))
            Ginv = 0.5 * (Ginv + Ginv.T)
            self.Lgamma = jnp.asarray(Lg)
            nc2 = 2 * self.n_common
            self._logdet_gamma_term = 2 * self.n_common * 2.0 * float(np.sum(np.log(np.diag(Lg))))
            if tri_inv == "recursive" and grad_precision in ("float64", "mixed"):
                self._core_sigma = _L._make_core_sigma(jnp.asarray(Ginv), nc2, grad_precision)
            else:
                self._core_sigma = _perf.make_core_sigma(jnp.asarray(Ginv), nc2, tri_inv, grad_precision)

        self.logL = jax.jit(self._logL)
        self.value_and_grad = jax.jit(jax.value_and_grad(self._logL))

    # -- helpers --
    def process_pulsars(self, name: str) -> list[str]:
        return [self.names[a] for a in self.processes[name]["psr_idx"]]

    def phi_common(self, params):
        if self.common == "powerlaw":
            return _basis.powerlaw(self.fc, self.dfc, params["log10_A"], params["gamma"])
        return _basis.free_spectrum(jnp.asarray(params["log10_rho"]))

    def _phi_flat(self, params, phic):
        flat = jnp.zeros(self.P * self.K + 1, dtype=jnp.float64)
        for nm, pr in self.processes.items():
            A = jnp.asarray(params[f"{nm}_log10_A"], dtype=jnp.float64)
            g = jnp.asarray(params[f"{nm}_gamma"], dtype=jnp.float64)
            phi = _basis.powerlaw(pr["f"], pr["df"], A[:, None], g[:, None])
            flat = flat.at[pr["idx"]].add(phi)
        if phic is not None:
            add = phic if self.Lgamma is None else self.lam0 * phic
            flat = flat.at[self._G_flat].add(jnp.broadcast_to(add[None, :], self._G_flat.shape))
        if self._pad_flat.size:
            flat = flat.at[self._pad_flat].add(1.0)
        return flat[: self.P * self.K].reshape(self.P, self.K)

    def _logL(self, params):
        return self._logL_cs(params, self.c, self.s_perp)

    def _logL_cs(self, params, c, s_perp):
        """logL with the residual-dependent stage-1 terms (c, s_perp) as arguments (injections:
        ``jax.vmap(like._logL_cs, in_axes=(None, 0, 0))`` over residual realisations)."""
        phic = self.phi_common(params) if self.common is not None else None
        phi = self._phi_flat(params, phic)
        red = jax.vmap(self._reducer)
        if self.buckets is not None:
            return self._logL_bucketed(red, phic, phi, c, s_perp)
        if self.Lgamma is None:
            q, ld, _, _ = red(self.RA, c, s_perp, phi)
            return -0.5 * (jnp.sum(q + ld) + self.const_total)
        q, ld, E, d = red(self.RA, c, s_perp, phi)
        ar = jnp.arange(self.P)[:, None, None]
        E = E[ar, self.G[:, :, None], self.G[:, None, :]]
        d = jnp.take_along_axis(d, self.G, axis=1)
        return self._core(q, ld, E, d, phic)

    def _logL_bucketed(self, red, phic, phi, c, s_perp):
        qs, lds, Es, ds = [], [], [], []
        for idx, kb in self.buckets:
            q, ld, E, d = red(self.RA[idx, :kb, :kb], c[idx, :kb], s_perp[idx], phi[idx, :kb])
            qs.append(q)
            lds.append(ld)
            if self.Lgamma is not None:
                G = self.G[idx]
                ar = jnp.arange(G.shape[0])[:, None, None]
                Es.append(E[ar, G[:, :, None], G[:, None, :]])
                ds.append(jnp.take_along_axis(d, G, axis=1))
        q, ld = jnp.concatenate(qs), jnp.concatenate(lds)
        if self.Lgamma is None:
            return -0.5 * (jnp.sum(q + ld) + self.const_total)
        o = self._bucket_order
        return self._core(q, ld, jnp.concatenate(Es)[o], jnp.concatenate(ds)[o], phic)

    def _core(self, q, ld, E, d, phic):
        sqc = jnp.sqrt(phic)
        Es = E * sqc[None, :, None] * sqc[None, None, :]
        dt = d * sqc[None, :]
        core = self._core_sigma(Es, dt) + self._logdet_gamma_term
        return -0.5 * (jnp.sum(q + ld) + self.const_total + core)

    def residual_terms(self, residuals: list) -> tuple[np.ndarray, np.ndarray]:
        """Stacked (c, s_perp) for new residual vectors, one per pulsar, each (n_a,) or (n_a, R)
        (requires ``precompute_general(..., projector=True)``). Returns c (P, K[, R]) zero-padded
        and s_perp (P[, R])."""
        cs, ss = [], []
        for t, r in zip(self.terms, residuals, strict=True):
            if t.projector is None:
                raise ValueError(f"{t.name}: precompute_general(..., projector=True) needed")
            c, s = t.projector(r)
            pad = np.zeros((self.K - t.K,) + c.shape[1:])
            cs.append(np.concatenate([c, pad], axis=0))
            ss.append(s)
        return np.stack(cs), np.stack(ss)
