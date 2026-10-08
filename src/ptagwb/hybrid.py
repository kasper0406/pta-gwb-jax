"""Exact hybrid kernel: NUTS composed with block Metropolis-Hastings independence jumps.

Why
---
In the HD/CURN free-spectrum posteriors several bins (and some IRN amplitudes) have a narrow
likelihood peak and a broad prior-dominated shelf (docs/FS_PILOT.md). NUTS moves between the two
only rarely. An independence proposal for one block (one bin, or one pulsar's IRN (log10_A,
gamma) pair), drawn from a frozen continuous density with full prior support, can jump straight
between the regions.

Kernel
------
One iteration = one NUTS transition (numpyro, unchanged, including its warmup adaptation),
followed by ``sweeps`` sweeps over the blocks; for block b with current physical value x_b

    x_b' ~ q_b,    accept with  min(1, pi(x') q_b(x_b) / (pi(x) q_b(x_b'))),

where pi is the posterior density of the *physical* (constrained) parameters x. With uniform box
priors pi(x) is proportional to L(x) inside the box. This is the MH ratio in the sampler's
coordinate z as well: the proposal density in z is q_b(x_b(z_b)) |dx_b/dz_b| and the target is
pi(x(z)) |dx/dz|, so the Jacobians cancel. Each block move leaves pi invariant (detailed balance);
so does NUTS; their composition is therefore pi-invariant (``tests/test_hybrid.py``). After the
sweep, NUTS's cached potential energy and gradient are recomputed at the final point.

Proposal q_b (frozen before any measurement, fitted from pilot draws by
``scripts/fs_fit_proposals.py``): mixture of w_prior x uniform on the block's prior box and
(1 - w_prior) x an equal-mass histogram of the pilot draws (1-D: K quantile bins; 2-D: K_a
quantile bins of the first coordinate, each split into K_b conditional quantile bins of the
second, uniform within cells). Continuous, full support, exact density.

Bookkeeping: the number of accepted block moves of the iteration is stored in the
``trajectory_length`` field of numpyro's HMCState (unused by NUTS, passed through unchanged), as a
(n_blocks,) vector, so that it can be collected as an extra field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import NUTS

from .config import REPO_ROOT, enable_x64

enable_x64()

_EPS_U = 1e-12  # relative trimming of every uniform component (see _trim)


# ---------------------------------------------------------------------- proposals


@dataclass(frozen=True)
class BlockProposals:
    """Frozen block proposals for a parameter vector of dimension D (physical coordinates).

    1-D blocks: ``idx1`` (n1,), ``edges1`` (n1, K1 + 1) quantile edges, prior bounds ``lo1``/``hi1``.
    2-D blocks: ``idx2`` (n2, 2), ``edges2a`` (n2, Ka + 1), ``edges2b`` (n2, Ka, Kb + 1), bounds
    ``lo2``/``hi2`` (n2, 2). ``w_prior``: weight of the uniform prior component.
    """

    w_prior: float
    idx1: np.ndarray
    edges1: np.ndarray
    lo1: np.ndarray
    hi1: np.ndarray
    idx2: np.ndarray
    edges2a: np.ndarray
    edges2b: np.ndarray
    lo2: np.ndarray
    hi2: np.ndarray
    names: tuple = ()

    @property
    def n_blocks(self) -> int:
        return len(self.idx1) + len(self.idx2)

    # ---- serialisation
    def to_json(self, path, meta: dict | None = None):
        d = {k: (np.asarray(getattr(self, k)).tolist()) for k in
             ("idx1", "edges1", "lo1", "hi1", "idx2", "edges2a", "edges2b", "lo2", "hi2")}
        d.update(w_prior=self.w_prior, names=list(self.names), meta=meta or {})
        Path(path).write_text(json.dumps(d))

    @classmethod
    def from_json(cls, path, names: list[str] | None = None) -> BlockProposals:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        d = json.loads(p.read_text())
        if names is not None and list(d["names"]) != list(names):
            raise ValueError(f"proposal file {path} was fitted for a different parameter vector")

        def arr(k, shape_tail, dtype=float):
            a = np.asarray(d[k], dtype)
            return a.reshape((-1,) + shape_tail) if a.size == 0 else a

        return cls(
            w_prior=float(d["w_prior"]),
            idx1=arr("idx1", (), int), edges1=arr("edges1", (2,)), lo1=arr("lo1", ()), hi1=arr("hi1", ()),
            idx2=arr("idx2", (2,), int), edges2a=arr("edges2a", (2,)), edges2b=arr("edges2b", (1, 2)),
            lo2=arr("lo2", (2,)), hi2=arr("hi2", (2,)), names=tuple(d["names"]),
        )


def _quantile_edges(v: np.ndarray, k: int, lo: float, hi: float) -> np.ndarray:
    """k equal-mass bins of the sample v (strictly increasing edges inside [lo, hi])."""
    e = np.quantile(v, np.linspace(0.0, 1.0, k + 1))
    e = np.clip(e, lo, hi)
    span = max(hi - lo, 1e-300)
    min_w = 1e-6 * span
    for i in range(1, len(e)):  # enforce a minimum width (ties)
        e[i] = max(e[i], e[i - 1] + min_w)
    if e[-1] > hi:  # shift back inside if the minimum widths pushed past hi
        e = lo + (e - e[0]) * (hi - lo) / (e[-1] - e[0]) if e[-1] - e[0] > hi - lo else e - (e[-1] - hi)
    return e


def fit_proposals(X: np.ndarray, names: list[str], lo: np.ndarray, hi: np.ndarray, blocks1: list[str],
                  blocks2: list[tuple[str, str]], w_prior: float = 0.2, k1: int = 40, k2a: int = 12,
                  k2b: int = 8) -> BlockProposals:
    """Fit equal-mass histogram proposals from pilot draws ``X`` (N, D) in physical coordinates."""
    idx = {n: j for j, n in enumerate(names)}
    i1 = np.array([idx[n] for n in blocks1], int)
    e1 = np.array([_quantile_edges(X[:, j], k1, lo[j], hi[j]) for j in i1]).reshape(len(i1), k1 + 1)
    i2 = np.array([[idx[a], idx[b]] for a, b in blocks2], int).reshape(-1, 2)
    e2a, e2b = [], []
    for ja, jb in i2:
        ea = _quantile_edges(X[:, ja], k2a, lo[ja], hi[ja])
        eb = []
        for i in range(k2a):
            sel = (X[:, ja] >= ea[i]) & (X[:, ja] <= ea[i + 1])
            vb = X[sel, jb] if sel.sum() >= 2 else X[:, jb]
            eb.append(_quantile_edges(vb, k2b, lo[jb], hi[jb]))
        e2a.append(ea)
        e2b.append(eb)
    return BlockProposals(
        w_prior=float(w_prior), idx1=i1, edges1=e1, lo1=lo[i1], hi1=hi[i1],
        idx2=i2, edges2a=np.array(e2a).reshape(len(i2), k2a + 1), edges2b=np.array(e2b).reshape(len(i2), k2a, k2b + 1),
        lo2=lo[i2].reshape(-1, 2), hi2=hi[i2].reshape(-1, 2), names=tuple(names),
    )


# ---- densities and samplers. Every component is uniform on a TRIMMED interval [a + d, b - d],
# d = _EPS_U (b - a), and the density is evaluated on exactly the same trimmed support, so sampling
# and density agree (the trimming keeps proposals strictly inside the open prior box).


def _trim(a, b):
    d = _EPS_U * (b - a)
    return a + d, b - d


def _unif_logpdf(x, a, b):
    a2, b2 = _trim(a, b)
    return jnp.where((x >= a2) & (x <= b2), -jnp.log(b2 - a2), -jnp.inf)


def _unif_sample(key, a, b, shape=()):
    a2, b2 = _trim(a, b)
    return a2 + (b2 - a2) * jax.random.uniform(key, shape)


def _hist_logpdf(x, edges):
    """log density of the equal-mass histogram with ``edges`` (cells trimmed; -inf outside)."""
    k = edges.shape[-1] - 1
    i = jnp.clip(jnp.searchsorted(edges, x, side="right") - 1, 0, k - 1)
    return jnp.log(1.0 / k) + _unif_logpdf(x, edges[i], edges[i + 1])


def _hist_sample(key, edges):
    k = edges.shape[-1] - 1
    k1, k2 = jax.random.split(key)
    i = jax.random.randint(k1, (), 0, k)
    return _unif_sample(k2, edges[i], edges[i + 1])


def _mix(lp, lh, w_prior):
    """log(w p + (1 - w) h) with w in (0, 1]; -inf only if both components vanish."""
    lw = jnp.log(w_prior)
    l1w = jnp.log1p(-w_prior) if w_prior < 1 else -jnp.inf
    return jnp.logaddexp(lw + lp, l1w + lh)


def log_q1(x, edges, lo, hi, w_prior):
    return _mix(_unif_logpdf(x, lo, hi), _hist_logpdf(x, edges), w_prior)


def sample_q1(key, edges, lo, hi, w_prior):
    k1, k2, k3 = jax.random.split(key, 3)
    return jnp.where(jax.random.uniform(k1) < w_prior, _unif_sample(k2, lo, hi), _hist_sample(k3, edges))


def log_q2(x, ea, eb, lo, hi, w_prior):
    lp = _unif_logpdf(x[0], lo[0], hi[0]) + _unif_logpdf(x[1], lo[1], hi[1])
    ka = ea.shape[-1] - 1
    i = jnp.clip(jnp.searchsorted(ea, x[0], side="right") - 1, 0, ka - 1)
    lh = jnp.log(1.0 / ka) + _unif_logpdf(x[0], ea[i], ea[i + 1]) + _hist_logpdf(x[1], eb[i])
    return _mix(lp, lh, w_prior)


def sample_q2(key, ea, eb, lo, hi, w_prior):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    xp = jnp.stack([_unif_sample(k2, lo[0], hi[0]), _unif_sample(k3, lo[1], hi[1])])
    ka = ea.shape[-1] - 1
    i = jax.random.randint(k4, (), 0, ka)
    xh = jnp.stack([_unif_sample(k5, ea[i], ea[i + 1]), _hist_sample(k6, eb[i])])
    return jnp.where(jax.random.uniform(k1) < w_prior, xp, xh)


def validate_proposals(prop: BlockProposals, transform) -> list[str]:
    """Errors (empty if valid) for using ``prop`` with the sampler's ``transform``: names, block
    indices (in range, distinct, no parameter in two blocks), the proposal's prior bounds equal to the
    target's (an independence proposal with different bounds has the wrong density or support),
    w_prior in (0, 1], and finite, strictly increasing edges inside the bounds."""
    errs = []
    D = transform.dim
    if prop.names and list(prop.names) != list(transform.names):
        errs.append("proposal names differ from the target's parameter names")
    if not (np.isfinite(prop.w_prior) and 0.0 < prop.w_prior <= 1.0):
        errs.append(f"w_prior must be in (0, 1], got {prop.w_prior!r}")
    i1 = np.asarray(prop.idx1).reshape(-1)
    i2 = np.asarray(prop.idx2).reshape(-1, 2) if np.size(prop.idx2) else np.zeros((0, 2), int)
    allidx = np.concatenate([i1, i2.ravel()]).astype(int)
    if np.any((allidx < 0) | (allidx >= D)):
        errs.append(f"block indices outside [0, {D})")
        return errs
    if len(set(allidx.tolist())) != len(allidx):
        errs.append("a parameter appears in more than one block (or twice in a pair)")
    lo, hi = np.asarray(transform.lo), np.asarray(transform.hi)

    def same(a, b):
        if np.size(b) == 0:
            return np.size(a) == 0
        return np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=0, atol=1e-12 * max(1.0, float(np.max(np.abs(b)))))

    if not (same(np.asarray(prop.lo1), lo[i1]) and same(np.asarray(prop.hi1), hi[i1])):
        errs.append("1-D proposal bounds differ from the target's prior bounds")
    if len(i2) and not (same(np.asarray(prop.lo2).reshape(-1, 2), lo[i2]) and same(np.asarray(prop.hi2).reshape(-1, 2), hi[i2])):
        errs.append("2-D proposal bounds differ from the target's prior bounds")

    def check_edges(e, a, b, label):
        e = np.asarray(e, float)
        if e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)):
            errs.append(f"{label}: edges must be a finite 1-D array of >= 2 values")
        elif not np.all(np.diff(e) > 0):
            errs.append(f"{label}: edges not strictly increasing")
        elif e[0] < a or e[-1] > b:
            errs.append(f"{label}: edges outside the prior bounds [{a}, {b}]")

    e1 = np.asarray(prop.edges1)
    if len(i1) and (e1.ndim != 2 or e1.shape[0] != len(i1)):
        errs.append("edges1 must have one row per 1-D block")
    else:
        for b_, j in enumerate(i1):
            check_edges(e1[b_], lo[j], hi[j], f"block {b_} ({transform.names[j]})")
    e2a, e2b = np.asarray(prop.edges2a), np.asarray(prop.edges2b)
    if len(i2) and (e2a.ndim != 2 or e2b.ndim != 3 or e2a.shape[0] != len(i2) or e2b.shape[:2] != (len(i2), e2a.shape[1] - 1)):
        errs.append("edges2a / edges2b shapes inconsistent with the 2-D blocks")
    else:
        for b_, (ja, jb) in enumerate(i2):
            check_edges(e2a[b_], lo[ja], hi[ja], f"pair {b_} first coordinate")
            for r in range(e2b.shape[1]):
                check_edges(e2b[b_, r], lo[jb], hi[jb], f"pair {b_} cell {r} second coordinate")
    return errs


# ---------------------------------------------------------------------- MH sweep


def make_sweep(prop: BlockProposals, transform, logL_x, sweeps: int = 1):
    """Single-chain sweep ``(key, z, logL) -> (z, logL, n_accept (n_blocks,))`` over all blocks.

    ``logL_x``: log-likelihood of the physical vector (uniform priors: log pi = logL + const inside
    the box). Blocks are visited in a fixed order (1-D blocks, then 2-D blocks); each move is an
    exact MH independence step for that block.
    """
    errs = validate_proposals(prop, transform)
    if errs:
        raise ValueError("invalid block proposals: " + "; ".join(errs[:8]))
    lo, hi = jnp.asarray(transform.lo), jnp.asarray(transform.hi)
    w = prop.w_prior
    i1, e1, l1, h1 = (jnp.asarray(a) for a in (prop.idx1, prop.edges1, prop.lo1, prop.hi1))
    i2, e2a, e2b, l2, h2 = (jnp.asarray(a) for a in (prop.idx2, prop.edges2a, prop.edges2b, prop.lo2, prop.hi2))
    n1, n2 = len(prop.idx1), len(prop.idx2)

    def to_z(x):
        u = (x - lo) / (hi - lo)
        return jnp.log(u) - jnp.log1p(-u)

    def step1(b, carry):
        key, x, ll, acc = carry
        key, kp, ka = jax.random.split(key, 3)
        j = i1[b]
        xn = sample_q1(kp, e1[b], l1[b], h1[b], w)
        xo = x[j]
        x2 = x.at[j].set(xn)
        ll2 = logL_x(x2)
        la = ll2 - ll + log_q1(xo, e1[b], l1[b], h1[b], w) - log_q1(xn, e1[b], l1[b], h1[b], w)
        ok = jnp.log(jax.random.uniform(ka, minval=1e-300)) < jnp.where(jnp.isfinite(la), la, -jnp.inf)
        return key, jnp.where(ok, x2, x), jnp.where(ok, ll2, ll), acc.at[b].add(ok.astype(acc.dtype))

    def step2(b, carry):
        key, x, ll, acc = carry
        key, kp, ka = jax.random.split(key, 3)
        j = i2[b]
        xn = sample_q2(kp, e2a[b], e2b[b], l2[b], h2[b], w)
        xo = x[j]
        x2 = x.at[j].set(xn)
        ll2 = logL_x(x2)
        la = (ll2 - ll + log_q2(xo, e2a[b], e2b[b], l2[b], h2[b], w)
              - log_q2(xn, e2a[b], e2b[b], l2[b], h2[b], w))
        ok = jnp.log(jax.random.uniform(ka, minval=1e-300)) < jnp.where(jnp.isfinite(la), la, -jnp.inf)
        return key, jnp.where(ok, x2, x), jnp.where(ok, ll2, ll), acc.at[n1 + b].add(ok.astype(acc.dtype))

    def sweep(key, z, ll):
        x = lo + (hi - lo) * jax.nn.sigmoid(z)
        acc = jnp.zeros(n1 + n2, jnp.float64)
        carry = (key, x, ll, acc)
        for _ in range(sweeps):
            if n1:
                carry = jax.lax.fori_loop(0, n1, step1, carry)
            if n2:
                carry = jax.lax.fori_loop(0, n2, step2, carry)
        _, x, ll, acc = carry
        # keep z unchanged where x was not moved (avoids a z -> x -> z round trip)
        moved = jnp.zeros(z.shape, bool)
        if n1:
            moved = moved.at[i1].set(True)
        if n2:
            moved = moved.at[i2.ravel()].set(True)
        x0 = lo + (hi - lo) * jax.nn.sigmoid(z)
        z_new = jnp.where(moved & (x != x0), to_z(x), z)
        return z_new, ll, acc

    return sweep


# ---------------------------------------------------------------------- conditional-grid move


def _logphi(r):
    """log((e^r - 1) / r), stable for all r (-> r / 2 near 0)."""
    small = jnp.abs(r) < 1e-6
    rs = jnp.where(small, 1.0, r)
    pos = rs + jnp.log(-jnp.expm1(-rs) / rs)
    neg = jnp.log(jnp.expm1(rs) / rs)
    return jnp.where(small, 0.5 * r, jnp.where(r > 0, pos, neg))


def grid_density(g, l):
    """Piecewise log-linear interpolant of log-density values ``l`` at sorted nodes ``g``.

    Returns (log_mass per cell, normalising log Z, l - max) defining the continuous density
    q(x) = exp(l_i + (l_{i+1} - l_i) (x - g_i) / (g_{i+1} - g_i) - max - logZ) on [g_0, g_-1].
    Non-finite node values are replaced by (min finite - 50); zero-width cells get zero mass."""
    fin = jnp.isfinite(l)
    lmin = jnp.min(jnp.where(fin, l, jnp.inf))
    l = jnp.where(fin, l, jnp.where(jnp.isfinite(lmin), lmin - 50.0, 0.0))
    l = l - jnp.max(l)
    d = jnp.diff(g)
    la, lb = l[:-1], l[1:]
    logm = jnp.where(d > 0, jnp.log(jnp.where(d > 0, d, 1.0)) + la + _logphi(lb - la), -jnp.inf)
    return logm, jax.scipy.special.logsumexp(logm), l


def grid_logpdf(x, g, l, logZ):
    i = jnp.clip(jnp.searchsorted(g, x, side="right") - 1, 0, g.shape[0] - 2)
    d = g[i + 1] - g[i]
    t = jnp.where(d > 0, (x - g[i]) / jnp.where(d > 0, d, 1.0), 0.0)
    v = l[i] + (l[i + 1] - l[i]) * t - logZ
    return jnp.where((x >= g[0]) & (x <= g[-1]), v, -jnp.inf)


def grid_sample(key, g, l, logm):
    k1, k2 = jax.random.split(key)
    i = jax.random.categorical(k1, logm)
    u = jax.random.uniform(k2)
    r = l[i + 1] - l[i]
    small = jnp.abs(r) < 1e-6
    rs = jnp.where(small, 1.0, r)
    t_pos = 1.0 + jnp.log(jnp.exp(-jnp.abs(rs)) + u * (-jnp.expm1(-jnp.abs(rs)))) / jnp.abs(rs)
    t_neg = jnp.log1p(u * jnp.expm1(rs)) / rs
    t = jnp.clip(jnp.where(small, u, jnp.where(r > 0, t_pos, t_neg)), 0.0, 1.0)
    return g[i] + (g[i + 1] - g[i]) * t


def make_grid_moves(posterior, bins, n_coarse: int = 48, n_fine: int = 32, half_width: float = 0.75,
                    w_uniform: float = 0.05, companions: dict | None = None, proposals: BlockProposals | None = None):
    """Metropolised conditional-grid moves for free-spectrum bins ``bins`` (single chain):
    ``(key, x, ll) -> (x, ll, n_accept (len(bins),))``.

    For bin k, at the current values of all OTHER parameters: the exact conditional log-likelihood
    (``ptagwb.conditional``; cache rebuilt from the state, independent of rho_k) is evaluated on a grid
    = ``n_coarse`` uniform nodes on the trimmed prior range [a, b] = [lo + d, hi - d] (d = 1e-9 (hi -
    lo)) plus ``n_fine`` uniform nodes on [x* - half_width, x* + half_width] (clipped to [a, b]), x* the
    coarse argmax. The proposal density for rho_k is

        q(rho | rest) = w_uniform / (b - a) + (1 - w_uniform) q_grid(rho | rest)   on [a, b],

    q_grid the normalised piecewise log-linear interpolant of the conditional through the nodes (exact
    inverse-CDF sampling within cells). It depends only on the other parameters, so it is an
    independence proposal for rho_k given the rest; acceptance min(1, L(x') q(rho | rest) / (L(x)
    q(rho' | rest))) uses the PRODUCTION likelihood (the conditional only shapes the proposal).

    ``companions`` {k: [pulsar, ...]}: joint move of rho_k and those pulsars' IRN (log10_A, gamma)
    pairs. The pairs are drawn from their frozen 2-D block proposals q2 (``proposals``), then rho_k from
    q(. | rest with the NEW pairs); the reverse density uses q2 at the current pairs and q(. | current
    rest). Acceptance min(1, L(x') prod q2(c) q(rho | rest) / (L(x) prod q2(c') q(rho' | rest'))):
    an independence proposal for the joint block (two cache builds per move)."""
    from .conditional import make_bin_conditional
    from .sampling import unpack

    like, spec, tr = posterior.like, posterior.spec, posterior.transform
    if like is None or spec is None or spec.common != "freespec":
        raise ValueError("conditional-grid moves need a free-spectrum Posterior with a likelihood")
    if not (0.0 < w_uniform < 1.0 and n_coarse >= 2 and n_fine >= 2 and half_width > 0):
        raise ValueError("invalid conditional-grid settings")
    companions = companions or {}
    if companions:
        if proposals is None:
            raise ValueError("companion moves need the block proposals (2-D IRN blocks)")
        errs = validate_proposals(proposals, tr)
        if errs:
            raise ValueError("invalid block proposals: " + "; ".join(errs[:8]))
    P = like.P
    logL_x = posterior.logL_x_raw
    parts = []
    for k in bins:
        j = tr.names.index(f"gw_log10_rho_{k}")
        build, cond = make_bin_conditional(like, k)
        lo, hi = float(tr.lo[j]), float(tr.hi[j])
        dl = 1e-9 * (hi - lo)
        comp = []
        for psr in companions.get(k, companions.get(str(k), [])):
            ja, jb = tr.names.index(f"{psr}_red_noise_log10_A"), tr.names.index(f"{psr}_red_noise_gamma")
            rows = [r for r, (u, v) in enumerate(np.asarray(proposals.idx2).reshape(-1, 2)) if (u, v) == (ja, jb)]
            if not rows:
                raise ValueError(f"no 2-D block proposal for {psr} (needed as companion of bin {k})")
            r = rows[0]
            comp.append((jnp.asarray([ja, jb]), jnp.asarray(proposals.edges2a[r]), jnp.asarray(proposals.edges2b[r]),
                         jnp.asarray(proposals.lo2[r]), jnp.asarray(proposals.hi2[r])))
        parts.append((j, build, jax.vmap(cond, in_axes=(None, 0)), lo + dl, hi - dl, comp))
    wq2 = proposals.w_prior if proposals is not None else 0.2

    def grid_for(x, build, condv, a, b):
        cache = build(unpack(x, spec, P))
        g1 = jnp.linspace(a, b, n_coarse)
        l1 = condv(cache, g1)
        xs = g1[jnp.argmax(jnp.where(jnp.isfinite(l1), l1, -jnp.inf))]
        g2 = jnp.linspace(jnp.clip(xs - half_width, a, b), jnp.clip(xs + half_width, a, b), n_fine)
        l2 = condv(cache, g2)
        g = jnp.concatenate([g1, g2])
        perm = jnp.argsort(g)
        g, lv = g[perm], jnp.concatenate([l1, l2])[perm]
        logm, logZ, lv = grid_density(g, lv)
        return g, lv, logm, logZ

    def log_q(v, grid, a, b):
        g, lv, _, logZ = grid
        return jnp.logaddexp(jnp.log(w_uniform) + jnp.where((v >= a) & (v <= b), -jnp.log(b - a), -jnp.inf),
                             jnp.log1p(-w_uniform) + grid_logpdf(v, g, lv, logZ))

    def sample_rho(key, grid, a, b):
        ku, km, kp = jax.random.split(key, 3)
        g, lv, logm, _ = grid
        return jnp.where(jax.random.uniform(ku) < w_uniform, a + (b - a) * jax.random.uniform(km), grid_sample(kp, g, lv, logm))

    def moves(key, x, ll):
        acc = []
        for j, build, condv, a, b, comp in parts:
            key, kc, kr, ka = jax.random.split(key, 4)
            lq_c = 0.0  # log q2(current pairs) - log q2(proposed pairs)
            xprop = x
            for i, (jj, ea, eb, l2, h2) in enumerate(comp):
                cn = sample_q2(jax.random.fold_in(kc, i), ea, eb, l2, h2, wq2)
                lq_c = lq_c + log_q2(x[jj], ea, eb, l2, h2, wq2) - log_q2(cn, ea, eb, l2, h2, wq2)
                xprop = xprop.at[jj].set(cn)
            grid_new = grid_for(xprop, build, condv, a, b)
            grid_old = grid_for(x, build, condv, a, b) if comp else grid_new
            prop = sample_rho(kr, grid_new, a, b)
            x2 = xprop.at[j].set(prop)
            ll2 = logL_x(x2)
            la = ll2 - ll + lq_c + log_q(x[j], grid_old, a, b) - log_q(prop, grid_new, a, b)
            ok = jnp.log(jax.random.uniform(ka, minval=1e-300)) < jnp.where(jnp.isfinite(la), la, -jnp.inf)
            x = jnp.where(ok, x2, x)
            ll = jnp.where(ok, ll2, ll)
            acc.append(ok.astype(jnp.float64))
        return x, ll, jnp.stack(acc)

    return moves


class HybridNUTS(NUTS):
    """numpyro NUTS followed, each iteration, by block-MH sweeps (see the module docstring).

    ``posterior``: a ``sampling.Posterior`` (its ``potential_fn`` must be the potential passed to
    NUTS; ``logL_x`` and ``transform`` define the MH target in physical coordinates).
    """

    def __init__(self, posterior, proposals: BlockProposals | None = None, sweeps: int = 1,
                 grid_bins=(), grid_kw: dict | None = None, **nuts_kw):
        super().__init__(potential_fn=posterior.potential_fn, **nuts_kw)
        if proposals is None and not grid_bins:
            raise ValueError("HybridNUTS needs block proposals and/or conditional-grid bins")
        self._post = posterior
        self._prop = proposals
        tr = posterior.transform
        lo, hi = jnp.asarray(tr.lo), jnp.asarray(tr.hi)

        def logL_from_pe(z, pe):  # potential = -(logL + logJ - logV)
            return -pe - tr.log_jacobian(z) + tr.log_volume

        sweep = make_sweep(proposals, tr, posterior.logL_x_raw, sweeps) if proposals is not None else None
        grid = make_grid_moves(posterior, list(grid_bins), proposals=proposals, **(grid_kw or {})) if grid_bins else None
        self._n_acc = (proposals.n_blocks if proposals is not None else 0) + len(grid_bins)
        vg = jax.value_and_grad(posterior.potential_fn)

        def jump(key, z, pe, grad):
            k1, k2 = jax.random.split(key)
            ll = logL_from_pe(z, pe)
            accs = []
            if sweep is not None:
                z, ll, acc = sweep(k1, z, ll)
                accs.append(acc)
            if grid is not None:
                x = lo + (hi - lo) * jax.nn.sigmoid(z)
                x2, ll, acc = grid(k2, x, ll)
                u = (x2 - lo) / (hi - lo)
                z = jnp.where(x2 != x, jnp.log(u) - jnp.log1p(-u), z)
                accs.append(acc)
            pe2, g2 = vg(z)
            return z, pe2, g2, jnp.concatenate(accs)

        self._jump1 = jump

    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):  # noqa: B006
        st = super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        nb = self._n_acc
        shape = (st.z.shape[0], nb) if jnp.ndim(st.z) == 2 else (nb,)
        return st._replace(trajectory_length=jnp.zeros(shape, jnp.float64))

    def sample(self, state, model_args, model_kwargs):
        st = super().sample(state, model_args, model_kwargs)
        if jnp.ndim(st.z) == 2:  # vectorised chains
            keys = jax.vmap(lambda k: jax.random.split(k))(st.rng_key)
            z, pe, g, acc = jax.vmap(self._jump1)(keys[:, 1], st.z, st.potential_energy, st.z_grad)
            new_key = keys[:, 0]
        else:
            k0, k1 = jax.random.split(st.rng_key)
            z, pe, g, acc = self._jump1(k1, st.z, st.potential_energy, st.z_grad)
            new_key = k0
        return st._replace(z=z, potential_energy=pe, z_grad=g, trajectory_length=acc, rng_key=new_key)
