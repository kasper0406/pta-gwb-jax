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

_EPS_U = 1e-12  # uniform draws in [eps, 1 - eps]: keeps proposals strictly inside the open box


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


# ---- 1-D histogram + prior mixture


def _hist_logpdf(x, edges):
    """log density of the equal-mass histogram with ``edges`` (-inf outside)."""
    k = edges.shape[-1] - 1
    i = jnp.clip(jnp.searchsorted(edges, x, side="right") - 1, 0, k - 1)
    w = edges[i + 1] - edges[i]
    inside = (x >= edges[0]) & (x <= edges[-1])
    return jnp.where(inside, -jnp.log(k * w), -jnp.inf)


def _hist_sample(key, edges):
    k = edges.shape[-1] - 1
    k1, k2 = jax.random.split(key)
    i = jax.random.randint(k1, (), 0, k)
    u = jax.random.uniform(k2, (), minval=_EPS_U, maxval=1 - _EPS_U)
    return edges[i] + u * (edges[i + 1] - edges[i])


def log_q1(x, edges, lo, hi, w_prior):
    lp = jnp.log(w_prior) - jnp.log(hi - lo)
    lh = jnp.log1p(-w_prior) + _hist_logpdf(x, edges)
    return jnp.logaddexp(lp, lh)


def sample_q1(key, edges, lo, hi, w_prior):
    k1, k2, k3 = jax.random.split(key, 3)
    xp = lo + (hi - lo) * jax.random.uniform(k2, (), minval=_EPS_U, maxval=1 - _EPS_U)
    xh = _hist_sample(k3, edges)
    return jnp.where(jax.random.uniform(k1) < w_prior, xp, xh)


# ---- 2-D conditional histogram + prior mixture


def log_q2(x, ea, eb, lo, hi, w_prior):
    lp = jnp.log(w_prior) - jnp.sum(jnp.log(hi - lo))
    ka = ea.shape[-1] - 1
    i = jnp.clip(jnp.searchsorted(ea, x[0], side="right") - 1, 0, ka - 1)
    inside_a = (x[0] >= ea[0]) & (x[0] <= ea[-1])
    la = jnp.where(inside_a, -jnp.log(ka * (ea[i + 1] - ea[i])), -jnp.inf)
    lb = _hist_logpdf(x[1], eb[i])
    lh = jnp.log1p(-w_prior) + jnp.where(inside_a, la + lb, -jnp.inf)
    return jnp.logaddexp(lp, lh)


def sample_q2(key, ea, eb, lo, hi, w_prior):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    xp = lo + (hi - lo) * jax.random.uniform(k2, (2,), minval=_EPS_U, maxval=1 - _EPS_U)
    ka = ea.shape[-1] - 1
    i = jax.random.randint(k3, (), 0, ka)
    k5, k6 = jax.random.split(k4)
    xa = ea[i] + jax.random.uniform(k5, (), minval=_EPS_U, maxval=1 - _EPS_U) * (ea[i + 1] - ea[i])
    xb = _hist_sample(k6, eb[i])
    return jnp.where(jax.random.uniform(k1) < w_prior, xp, jnp.stack([xa, xb]))


# ---------------------------------------------------------------------- MH sweep


def make_sweep(prop: BlockProposals, transform, logL_x, sweeps: int = 1):
    """Single-chain sweep ``(key, z, logL) -> (z, logL, n_accept (n_blocks,))`` over all blocks.

    ``logL_x``: log-likelihood of the physical vector (uniform priors: log pi = logL + const inside
    the box). Blocks are visited in a fixed order (1-D blocks, then 2-D blocks); each move is an
    exact MH independence step for that block.
    """
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


class HybridNUTS(NUTS):
    """numpyro NUTS followed, each iteration, by block-MH sweeps (see the module docstring).

    ``posterior``: a ``sampling.Posterior`` (its ``potential_fn`` must be the potential passed to
    NUTS; ``logL_x`` and ``transform`` define the MH target in physical coordinates).
    """

    def __init__(self, posterior, proposals: BlockProposals, sweeps: int = 1, **nuts_kw):
        super().__init__(potential_fn=posterior.potential_fn, **nuts_kw)
        self._post = posterior
        self._prop = proposals
        tr = posterior.transform

        def logL_from_pe(z, pe):  # potential = -(logL + logJ - logV)
            return -pe - tr.log_jacobian(z) + tr.log_volume

        sweep = make_sweep(proposals, tr, posterior.logL_x_raw, sweeps)
        vg = jax.value_and_grad(posterior.potential_fn)

        def jump(key, z, pe, grad):
            z2, _, acc = sweep(key, z, logL_from_pe(z, pe))
            pe2, g2 = vg(z2)
            return z2, pe2, g2, acc

        self._jump1 = jump

    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):  # noqa: B006
        st = super().init(rng_key, num_warmup, init_params, model_args, model_kwargs)
        nb = self._prop.n_blocks
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
