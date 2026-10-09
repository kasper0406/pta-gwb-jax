"""N13: NUTS on the continuous coordinates composed with exact Metropolis-Hastings for event
epochs (docs/M3B_PLAN.md Sec. 5.1).

Why: the released exponential-dip waveform switches on with a Heaviside step at t0, so the
likelihood is discontinuous in t0 wherever t0 crosses a TOA. NUTS dynamics and derivative checks
are invalid in t0; t0 is never a NUTS coordinate.

Kernel (Metropolis-within-Gibbs), one iteration:

1. one NUTS transition (numpyro's functional ``hmc(potential_fn_gen=...)``, warmup adaptation
   included) on the unconstrained continuous coordinates z (logistic box transform of every
   parameter except t0), with the potential U(z; t0) = -[logL(x(z), t0) + log|dx/dz|] at the
   current t0 (passed as a model argument, so no recompilation);
2. ``n_mh`` exact MH updates of t0 | x with the proposal of ``T0Proposal``;
3. NUTS's cached potential energy and gradient are recomputed at (z, t0_new) (as in
   ``ptagwb.hybrid``).

Each step leaves the joint posterior invariant (NUTS: the conditional of z given t0; MH: the
conditional of t0 given z), hence so does the composition.

``T0Proposal`` is the mixture q(y | x) = w_u U(y; window) + w_g g(y) + w_rw N(y; x, s^2) with g an
equal-mass histogram density on the window (fitted from pilot draws, frozen) or absent. The full
Hastings ratio q(x | y) / q(y | x) is applied; ``hastings=False`` drops it (the negative control
of the validation: with a non-uniform g the chain then targets the wrong distribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.hmc import hmc

from .config import enable_x64

enable_x64()


@dataclass(frozen=True)
class T0Proposal:
    lo: float
    hi: float
    w_uniform: float = 0.2
    w_hist: float = 0.0
    edges: np.ndarray | None = None  # equal-mass histogram edges on [lo, hi]
    rw_scale: float = 1.0  # days

    @property
    def w_rw(self) -> float:
        return 1.0 - self.w_uniform - self.w_hist

    def __post_init__(self):
        if not (self.hi > self.lo and 0 <= self.w_uniform and 0 <= self.w_hist and self.w_rw >= 0):
            raise ValueError("invalid T0Proposal weights / window")
        if self.w_hist > 0 and (self.edges is None or self.edges[0] != self.lo or self.edges[-1] != self.hi):
            raise ValueError("histogram component needs edges spanning the window")


def fit_t0_hist(draws: np.ndarray, lo: float, hi: float, k: int = 20) -> np.ndarray:
    """Equal-mass histogram edges of pilot t0 draws on [lo, hi] (frozen before production)."""
    q = np.quantile(np.clip(draws, lo, hi), np.linspace(0, 1, k + 1))
    q[0] = lo
    q = np.maximum.accumulate(q + np.arange(k + 1) * 1e-9)  # strictly increasing
    q[-1] = hi
    if not np.all(np.diff(q) > 0):
        raise ValueError("degenerate t0 histogram")
    return q


def _hist_logpdf(y, edges):
    e = jnp.asarray(edges)
    k = e.shape[0] - 1
    j = jnp.clip(jnp.searchsorted(e, y, side="right") - 1, 0, k - 1)
    inside = (y >= e[0]) & (y <= e[-1])
    return jnp.where(inside, -jnp.log(k * (e[j + 1] - e[j])), -jnp.inf)


def _hist_sample(key, edges):
    e = jnp.asarray(edges)
    k = e.shape[0] - 1
    k1, k2 = jax.random.split(key)
    j = jax.random.randint(k1, (), 0, k)
    return e[j] + (e[j + 1] - e[j]) * jax.random.uniform(k2)


def log_q(prop: T0Proposal, y, x):
    """log q(y | x) of the mixture (y, x scalars)."""
    terms = []
    inside = (y >= prop.lo) & (y <= prop.hi)
    if prop.w_uniform > 0:
        terms.append(jnp.where(inside, jnp.log(prop.w_uniform) - jnp.log(prop.hi - prop.lo), -jnp.inf))
    if prop.w_hist > 0:
        terms.append(jnp.log(prop.w_hist) + _hist_logpdf(y, prop.edges))
    if prop.w_rw > 0:
        s = prop.rw_scale
        terms.append(jnp.log(prop.w_rw) - 0.5 * ((y - x) / s) ** 2 - jnp.log(s * jnp.sqrt(2 * jnp.pi)))
    return jax.scipy.special.logsumexp(jnp.stack(terms))


def sample_q(prop: T0Proposal, key, x):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    u = jax.random.uniform(k1)
    y_u = prop.lo + (prop.hi - prop.lo) * jax.random.uniform(k2)
    y_h = _hist_sample(k3, prop.edges) if prop.w_hist > 0 else y_u
    y_r = x + prop.rw_scale * jax.random.normal(k4)
    return jnp.where(u < prop.w_uniform, y_u, jnp.where(u < prop.w_uniform + prop.w_hist, y_h, y_r))


def mh_t0(key, x, ll, logL_x, i0: int, prop: T0Proposal, hastings: bool = True):
    """One exact MH update of x[i0] (uniform prior on [lo, hi]); JAX-traceable.
    Returns (x, logL, accepted)."""
    k1, k2 = jax.random.split(key)
    y = sample_q(prop, k1, x[i0])
    x2 = x.at[i0].set(y)
    inside = (y >= prop.lo) & (y <= prop.hi)
    ll2 = jnp.where(inside, logL_x(x2), -jnp.inf)
    la = ll2 - ll
    if hastings:
        la = la + log_q(prop, x[i0], y) - log_q(prop, y, x[i0])
    ok = inside & (jnp.log(jax.random.uniform(k2, minval=1e-300)) < jnp.where(jnp.isfinite(la), la, -jnp.inf))
    return jnp.where(ok, x2, x), jnp.where(ok, ll2, ll), ok


class EventMHNUTS:
    """NUTS on all coordinates but ``i0`` composed with MH on ``i0`` (module docstring).

    ``logL_x``: JAX log-likelihood of the full physical vector; ``lo``/``hi``: uniform prior box of
    the full vector; ``i0``: index of the event epoch."""

    def __init__(self, logL_x, lo, hi, i0: int, prop: T0Proposal, *, n_mh: int = 1, hastings: bool = True,
                 max_tree_depth: int = 10, dense_mass: bool = True, target_accept_prob: float = 0.8):
        self.logL_x, self.i0, self.prop, self.n_mh, self.hastings = logL_x, i0, prop, n_mh, hastings
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        D = len(self.lo)
        self.cont = np.array([i for i in range(D) if i != i0])
        lo_c, hi_c = jnp.asarray(self.lo[self.cont]), jnp.asarray(self.hi[self.cont])
        cont = jnp.asarray(self.cont)

        def full_x(z, t0):
            xc = lo_c + (hi_c - lo_c) * jax.nn.sigmoid(z)
            return jnp.zeros(D, jnp.float64).at[cont].set(xc).at[i0].set(t0)

        def potential_gen(t0):
            def pot(z):
                logj = jnp.sum(jnp.log(hi_c - lo_c) - jax.nn.softplus(-z) - jax.nn.softplus(z))
                return -(logL_x(full_x(z, t0)) + logj)
            return pot

        self._full_x = full_x
        self._pot_gen = potential_gen
        self._init_k, self._sample_k = hmc(potential_fn_gen=potential_gen, algo="NUTS")
        self.kw = {"max_tree_depth": max_tree_depth, "dense_mass": dense_mass, "target_accept_prob": target_accept_prob}

    def to_z(self, x):
        xc = jnp.asarray(x, jnp.float64)[jnp.asarray(self.cont)]
        lo, hi = jnp.asarray(self.lo[self.cont]), jnp.asarray(self.hi[self.cont])
        u = (xc - lo) / (hi - lo)
        return jnp.log(u) - jnp.log1p(-u)

    def init(self, key, x0, num_warmup, *, step_size=0.1, inverse_mass_matrix=None, adapt_step_size=True,
             adapt_mass_matrix=True):
        t0 = jnp.asarray(x0, jnp.float64)[self.i0]
        st = self._init_k(self.to_z(x0), num_warmup, step_size=step_size,
                          inverse_mass_matrix=inverse_mass_matrix, adapt_step_size=adapt_step_size,
                          adapt_mass_matrix=adapt_mass_matrix, model_args=(t0,), rng_key=key, **self.kw)
        return st, t0

    def step(self, key, st, t0):
        """One iteration: (state, t0) -> (state, t0, info) with info = (n_accept_mh, NUTS diverging,
        accept_prob, num_steps)."""
        st = self._sample_k(st, model_args=(t0,))
        x = self._full_x(st.z, t0)
        ll = self.logL_x(x)
        acc = jnp.zeros((), jnp.int32)
        keys = jax.random.split(key, self.n_mh)
        for k in range(self.n_mh):
            x, ll, ok = mh_t0(keys[k], x, ll, self.logL_x, self.i0, self.prop, self.hastings)
            acc = acc + ok.astype(jnp.int32)
        t0n = x[self.i0]
        pe, g = jax.value_and_grad(self._pot_gen(t0n))(st.z)
        st = st._replace(potential_energy=pe, z_grad=g)
        return st, t0n, (acc, st.diverging, st.accept_prob, st.num_steps)

    def physical(self, st, t0):
        return self._full_x(st.z, t0)
