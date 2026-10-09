"""N13 validation (docs/M3B_PLAN.md Sec. 5.1): NUTS on the continuous dip parameters composed
with exact MH for the dip epoch t0, against a numerical reference, with a negative control.

Toy: one pulsar, one exponential dip (EPTA waveform: sign -1, index 1, Heaviside onset at t0),
white noise, realistic TOA gaps (an observing gap spanning several days inside the t0 window), so
that t0's posterior spreads over several inter-TOA intervals. Parameters (log10_Amp, log10_tau,
t0) with uniform priors. The reference is a 3-D grid: exact in the amplitude direction given
(t0, tau) through the sufficient statistics S1 = sum r w / s^2, S2 = sum w^2 / s^2, with fine
t0 / log10_tau spacing.

Criteria (fixed before running): t0 interval occupancies (inter-TOA intervals with >= 1 %
reference mass, plus "rest") and the marginal quantiles (5/50/95 %) of t0 and log10_Amp agree with
the reference within 3.5 Monte Carlo standard errors; the same kernel without the Hastings term,
with a non-uniform independence component, must fail at least one criterion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ptagwb import acceptance as acc
from ptagwb.diagnostics import mcse_quantile
from ptagwb.eventmh import EventMHNUTS, T0Proposal, fit_t0_hist

DAY = 86400.0
WIN = (57490.0, 57530.0)
LO = np.array([-8.0, 0.0, WIN[0]])
HI = np.array([-4.0, 2.5, WIN[1]])


def toy_data(seed=1):
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.arange(57400, 57500, 4.0), [57503.2, 57506.1, 57509.7, 57514.4],
                        np.arange(57521.0, 57640, 4.0)])
    t = np.sort(t + rng.uniform(0, 0.3, t.size))
    nu = np.where(np.arange(t.size) % 2 == 0, 1400.0, 2500.0)
    sig = np.full(t.size, 0.4e-6)
    a, lt, t0 = -5.9, 1.3, 57507.5
    w = np.where(t >= t0, np.exp(-(t - t0) / 10**lt), 0.0) * (1400.0 / nu)
    r = -(10**a) * w + sig * rng.standard_normal(t.size)
    return t, nu, sig, r


def make_logL(t, nu, sig, r):
    tj, nj, sj, rj = (jnp.asarray(v) for v in (t * DAY, nu, sig, r))

    def logL(x):
        A, tau, t0 = 10.0 ** x[0], 10.0 ** x[1] * DAY, x[2] * DAY
        dt = tj - t0
        w = jnp.where(dt >= 0, jnp.exp(-jnp.maximum(dt, 0) / tau), 0.0) * (1400.0 / nj)
        res = rj + A * w
        return -0.5 * jnp.sum((res / sj) ** 2)

    return logL


def reference(t, nu, sig, r, n_t0=8001, n_lt=121, n_a=801):
    t0g = np.linspace(WIN[0], WIN[1], n_t0)
    ltg = np.linspace(LO[1], HI[1], n_lt)
    ag = np.linspace(LO[0], HI[0], n_a)
    A = 10**ag
    logp = np.empty((n_t0, n_lt, n_a))
    for j, lt in enumerate(ltg):
        dt = t[None, :] - t0g[:, None]
        w = np.where(dt >= 0, np.exp(-np.maximum(dt, 0) / 10**lt), 0.0) * (1400.0 / nu)[None, :]
        # residual model: r = -A w + n  ->  lnL = A S1' - A^2 S2 / 2 with S1' = -sum r w / s^2
        S1 = -(w * (r / sig**2)[None, :]).sum(1)
        S2 = (w**2 / sig[None, :] ** 2).sum(1)
        logp[:, j, :] = A[None, :] * S1[:, None] - 0.5 * A[None, :] ** 2 * S2[:, None]
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return t0g, ltg, ag, p


def grid_quantile(g, pm, q):
    c = np.cumsum(pm)
    return float(np.interp(q, c / c[-1], g))


def run_kernel(logL, prop, hastings, n_chains=4, warmup=600, samples=3000, seed=0, blocks=()):
    k = EventMHNUTS(logL, LO, HI, 2, prop, n_mh=2, hastings=hastings, max_tree_depth=6, blocks=blocks)
    rng = np.random.default_rng(seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_chains)
    x0s = [np.array([-5.9 + 0.2 * rng.standard_normal(), 1.3 + 0.2 * rng.standard_normal(),
                     rng.uniform(*WIN)]) for _ in range(n_chains)]
    st, t0 = jax.vmap(lambda kk, x: k.init(kk, x, warmup, step_size=0.1))(keys, jnp.asarray(np.stack(x0s)))

    def body(carry, kk):
        st, t0 = carry
        st, t0, info = jax.vmap(k.step)(jax.random.split(kk, n_chains), st, t0)
        return (st, t0), (jax.vmap(k.physical)(st, t0), info[0], info[1])

    (st, t0), (xs, accs, divs) = jax.lax.scan(body, (st, t0), jax.random.split(jax.random.PRNGKey(seed + 1), warmup + samples))
    xs = np.asarray(xs)[warmup:]  # (samples, chains, 3)
    return np.transpose(xs, (1, 0, 2)), np.asarray(accs)[warmup:], np.asarray(divs)[warmup:]


def compare(X, t, ref):
    t0g, ltg, ag, p = ref
    pt0, pa = p.sum((1, 2)), p.sum((0, 1))
    # reference t0 draws by inverse-CDF sampling of the grid marginal, for the interval definition
    rng = np.random.default_rng(5)
    c = np.cumsum(pt0)
    tref = np.interp(rng.random(200000), c / c[-1], t0g)
    ints = acc.event_intervals(t, WIN, tref)
    rows = []
    inside_ref = np.zeros_like(t0g, dtype=bool)
    inside_ours = [np.zeros(ch.shape[0], bool) for ch in X]
    for lo, hi in ints:
        mref_mask = (t0g >= lo) & ((t0g < hi) if hi != WIN[1] else (t0g <= hi))
        inside_ref |= mref_mask
        ind = [(ch[:, 2] >= lo) & ((ch[:, 2] < hi) if hi != WIN[1] else (ch[:, 2] <= hi)) for ch in X]
        inside_ours = [a | b for a, b in zip(inside_ours, ind)]
        o = acc.occupancy(ind)
        rows.append(("interval", (lo, hi), o.p_hat, float(pt0[mref_mask].sum()), o.mcse if o.mcse else np.inf))
    o = acc.occupancy([~v for v in inside_ours])
    if o.mcse:  # "rest" bin, when estimable
        rows.append(("interval", "rest", o.p_hat, float(pt0[~inside_ref].sum()), o.mcse))
    for name, col, g, pm in (("t0", 2, t0g, pt0), ("log10_Amp", 0, ag, pa)):
        for q in (0.05, 0.5, 0.95):
            draws = np.stack([ch[:, col] for ch in X])
            rows.append((name, q, float(np.quantile(draws, q)), grid_quantile(g, pm, q), float(mcse_quantile(draws, q))))
    z = [abs(o - r) / se for _, _, o, r, se in rows]
    return rows, max(z), len(ints)


@pytest.fixture(scope="module")
def toy():
    t, nu, sig, r = toy_data()
    return t, make_logL(t, nu, sig, r), reference(t, nu, sig, r)


@pytest.mark.slow
def test_event_mh_kernel_matches_quadrature(toy):
    t, logL, ref = toy
    prop = T0Proposal(WIN[0], WIN[1], w_uniform=0.2, rw_scale=2.0)
    X, accs, divs = run_kernel(logL, prop, hastings=True)
    rows, zmax, n_int = compare(X, t, ref)
    assert n_int >= 3, "the toy must spread t0 over several inter-TOA intervals"
    assert accs.mean() > 0.05
    assert zmax < 3.5, rows


@pytest.mark.slow
def test_event_mh_negative_control_fails(toy):
    """Same kernel, non-uniform independence component, Hastings term dropped: must fail."""
    t, logL, ref = toy
    edges = fit_t0_hist(np.random.default_rng(3).uniform(WIN[0], 57500.0, 5000), *WIN, k=10)
    good = T0Proposal(WIN[0], WIN[1], w_uniform=0.1, w_hist=0.6, edges=edges, rw_scale=2.0)
    X, _, _ = run_kernel(logL, good, hastings=True, seed=7)
    _, z_ok, _ = compare(X, t, ref)
    Xb, _, _ = run_kernel(logL, good, hastings=False, seed=7)
    _, z_bad, _ = compare(Xb, t, ref)
    assert z_ok < 3.5
    assert z_bad > 3.5


@pytest.mark.slow
def test_event_mh_kernel_with_block_moves_matches_quadrature(toy):
    """The complete kernel: NUTS + t0 MH + independence block MH on (log10_Amp, log10_tau) and the
    joint (t0, log10_tau, log10_Amp) block (prior proposals, as in the pilot config) still targets
    the reference posterior."""
    from ptagwb.eventmh import BlockProposal

    t, logL, ref = toy
    prop = T0Proposal(WIN[0], WIN[1], w_uniform=0.2, rw_scale=2.0)
    blocks = (BlockProposal(idx=(0, 1), lo=tuple(LO[:2]), hi=tuple(HI[:2])),
              BlockProposal(idx=(2, 1, 0), lo=(LO[2], LO[1], LO[0]), hi=(HI[2], HI[1], HI[0])))
    X, accs, divs = run_kernel(logL, prop, hastings=True, blocks=blocks, seed=3)
    rows, zmax, _ = compare(X, t, ref)
    assert zmax < 3.5, rows
