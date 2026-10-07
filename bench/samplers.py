"""Q4: sampler comparison on the CURN^13/3 and HD^13/3 posteriors (docs/PERF.md).

All samplers target the production posterior ``Posterior.logpost_z`` (logistic box transform,
float64 energies). Fair set-up shared by every sampler:

* **Same preconditioning.** Production HD NUTS uses the pooled covariance C of the M2 CURN^13/3
  run's unconstrained draws as a fixed dense metric. Here every sampler works in whitened
  coordinates w = L_C^-1 (z - mu) (L_C = chol C, mu = CURN mean) with an identity metric, which
  is the same thing for HMC/NUTS (a dense metric == a linear reparameterisation) and gives
  MCLMC/MAMS/ChEES/MEADS (diagonal-metric methods) the same dense preconditioner. NumPyro runs
  through the production driver ``ptagwb.sampling.run_nuts`` with ``metric=run:curn_g433_14f``
  (identical C).
* **Same initialisation**: random draws of the M2 CURN^13/3 run (production HD practice), so
  warmup is tuning only (step size; trajectory length where the method has one).
* Gradient accounting: one leapfrog / velocity-Verlet step = 1 gradient; the isokinetic
  McLachlan (2-stage minimal-norm) integrator of MCLMC/MAMS = 2 gradients per step.

    uv run --no-sync python bench/samplers.py --model curn --sampler bj_nuts --seed 0
    JAX_PLATFORMS=cpu uv run --no-sync python bench/samplers.py ...   # CPU backend

Writes bench/runs/<tag>.npz (draws, git-ignored) and bench/results/samplers/<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
import jax.numpy as jnp
import numpy as np
from common import BENCH, build_post, env_info, get_terms

from ptagwb import diagnostics as dg
from ptagwb.sampling import load_run, metric_from_runs

RUNS = BENCH / "runs"
OUT = BENCH / "results" / "samplers"
CURN_RUN = "curn_g433_14f"


# ---------------------------------------------------------------------- target


class Target:
    def __init__(self, model: str, like_variant: str = "aug"):
        terms, T = get_terms()
        if like_variant == "prod":
            post = build_post(model)
        else:
            from ptagwb.perf_likelihood import FastPTALikelihood

            like = FastPTALikelihood(terms, T, n_modes=30, n_common=14, orf=model, reduce=like_variant.split("+")[0],
                                     **({"tri_inv": like_variant.split("+")[1]} if "+" in like_variant else {}))
            post = build_post(model, like=like)
        self.post, self.model, self.like_variant = post, model, like_variant
        self.names = post.names
        self.D = len(self.names)
        C = metric_from_runs([CURN_RUN], self.names)
        src = load_run(CURN_RUN)
        assert list(src["names"]) == self.names
        Z = src["z"].reshape(-1, self.D)
        self.Zsrc = Z
        self.mu = jnp.asarray(Z.mean(axis=0))
        self.Lc = jnp.asarray(np.linalg.cholesky(C))
        mu, Lc = self.mu, self.Lc
        self.logdensity = lambda w: post.logpost_z(mu + Lc @ w)

    def w_to_z(self, W):
        return np.asarray(self.mu) + np.asarray(W) @ np.asarray(self.Lc).T

    def init_w(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        z = self.Zsrc[rng.choice(len(self.Zsrc), n, replace=False)]
        return jnp.asarray(np.linalg.solve(np.asarray(self.Lc), (z - np.asarray(self.mu)).T).T)


def _aot(fn, *args):
    t0 = time.perf_counter()
    c = jax.jit(fn).lower(*args).compile()
    return c, time.perf_counter() - t0


def _timed(c, *args):
    t0 = time.perf_counter()
    out = jax.block_until_ready(c(*args))
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------------- samplers (BlackJAX)


def run_bj_nuts(tg: Target, C, W, N, seed, block=100, target_accept=0.8, max_doublings=10, init_step=0.25):
    import blackjax
    from blackjax.adaptation.step_size import dual_averaging_adaptation

    logd = tg.logdensity
    imm = jnp.ones(tg.D)
    key = jax.random.PRNGKey(seed)
    w0 = tg.init_w(C, seed)
    states = jax.vmap(lambda w: blackjax.nuts.init(w, logd))(w0)
    da_init, da_update, da_final = dual_averaging_adaptation(target_accept)

    def warm_step(carry, k):
        st, da = carry

        def one(k_, s_, d_):
            kern = blackjax.nuts(logd, jnp.exp(d_.log_step_size), imm, max_num_doublings=max_doublings)
            s2, info = kern.step(k_, s_)
            return s2, da_update(d_, info.acceptance_rate), info.num_integration_steps

        ks = jax.random.split(k, C)
        st, da, ns = jax.vmap(one)(ks, st, da)
        return (st, da), ns

    def warm(st, da, keys):
        return jax.lax.scan(warm_step, (st, da), keys)

    # dual averaging from init_step (0.25 default; the whitened CURN/HD step sizes end at ~0.05)
    da0 = jax.vmap(da_init)(jnp.full(C, init_step))
    kw, ks = jax.random.split(key)
    wkeys = jax.random.split(kw, W)
    cw, t_cw = _aot(warm, states, da0, wkeys)
    ((states, da), ns_w), t_warm = _timed(cw, states, da0, wkeys)
    step = jax.vmap(da_final)(da)

    def samp_step(st, k):
        def one(k_, s_, h_):
            kern = blackjax.nuts(logd, h_, imm, max_num_doublings=max_doublings)
            s2, info = kern.step(k_, s_)
            return s2, (s2.position, info.num_integration_steps, info.acceptance_rate, info.is_divergent)

        return jax.vmap(one)(jax.random.split(k, C), st, step)

    def samp_block(st, keys):
        return jax.lax.scan(samp_step, st, keys)

    return _sample_blocks(tg, samp_block, states, ks, N, block, t_cw, t_warm, int(np.sum(ns_w)),
                          extra={"step_size": np.asarray(step).tolist(), "target_accept": target_accept, "init_step": init_step})


def _sample_blocks(tg, samp_block, states, key, N, block, t_compile_warm, t_warm, warm_grads, extra, grads_per_step=1):
    keys = jax.random.split(key, N)
    block = min(block, N)
    cs, t_cs = _aot(samp_block, states, keys[:block])
    if N % block:  # separate executable for the last partial block
        ctail, t_tail = _aot(samp_block, states, keys[: N % block])
        t_cs += t_tail
    Ws, steps, accs, divs = [], [], [], []
    t_samp = 0.0
    for i in range(0, N, block):
        c_ = cs if i + block <= N else ctail
        (states, (w, ns, acc, dv)), dt = _timed(c_, states, keys[i : i + block])
        t_samp += dt
        Ws.append(np.asarray(w))
        steps.append(np.asarray(ns))
        accs.append(np.asarray(acc))
        divs.append(np.asarray(dv))
    Wd = np.concatenate(Ws, 0).swapaxes(0, 1)  # (C, N, D)
    steps = np.concatenate(steps, 0).swapaxes(0, 1) * grads_per_step
    return {
        "w": Wd,
        "grads_per_draw": steps,
        "accept": np.concatenate(accs, 0).swapaxes(0, 1),
        "divergent": np.concatenate(divs, 0).swapaxes(0, 1),
        "t_compile": t_compile_warm + t_cs,
        "t_warm": t_warm,
        "t_samp": t_samp,
        "warm_grads": warm_grads,
        "extra": extra,
    }


# ---------------------------------------------------------------------- matched NUTS drivers
#
# Both drivers: same whitened target, identity metric, same init draws, same warmup length and
# initial step size; compile (AOT) timed separately; every execution timer stops after
# block_until_ready. Sampling proceeds in blocks of ``block`` draws until ``until`` is met
# (common bulk ESS >= ess_target and max R-hat < rhat_target) or N draws; per-block cumulative
# times and diagnostics are recorded so the time to any target can be read off.


def _common_diag(tg, Wd):
    C, N, D = Wd.shape
    X = np.asarray(tg.post.transform.to_constrained(tg.w_to_z(Wd.reshape(-1, D)).reshape(C, N, D)))
    jc = tg.names.index("gw_log10_A")
    rh = np.array([dg.rhat(X[:, :, j]) for j in range(D)])
    return {"ess_bulk_common": float(dg.ess_bulk(X[:, :, jc])), "rhat_common": float(rh[jc]),
            "rhat_max": float(np.nanmax(rh)), "ess_bulk_min": float(min(dg.ess_bulk(X[:, :, j]) for j in range(D)))}


def _adaptive_blocks(tg, samp_block, states, key, N, block, until, t_compile, t_warm, warm_grads, extra):
    ess_target, rhat_target = until if until else (np.inf, 0.0)
    keys = jax.random.split(key, N)
    cs, t_cs = _aot(samp_block, states, keys[:block])
    Ws, steps, accs, divs, trace = [], [], [], [], []
    t_samp, n = 0.0, 0
    while n + block <= N:
        (states, (w, ns, acc, dv)), dt = _timed(cs, states, keys[n : n + block])
        t_samp += dt
        n += block
        Ws.append(np.asarray(w)); steps.append(np.asarray(ns)); accs.append(np.asarray(acc)); divs.append(np.asarray(dv))
        Wd = np.concatenate(Ws, 0).swapaxes(0, 1)
        d = _common_diag(tg, Wd)
        st = np.concatenate(steps, 0)
        d.update(draws=n, t_samp=t_samp, t_total=t_compile + t_cs + t_warm + t_samp,
                 grads=float(st.sum()), lockstep_grads=float(st.max(1).sum() * st.shape[1]))
        trace.append(d)
        print(f"  block: {n} draws, {t_samp:.0f} s, ESS common {d['ess_bulk_common']:.0f}, R-hat common "
              f"{d['rhat_common']:.4f} max {d['rhat_max']:.4f}", flush=True)
        if d["ess_bulk_common"] >= ess_target and d["rhat_max"] < rhat_target:
            break
    Wd = np.concatenate(Ws, 0).swapaxes(0, 1)
    st = np.concatenate(steps, 0).swapaxes(0, 1)
    hit = next((t for t in trace if t["ess_bulk_common"] >= ess_target and t["rhat_common"] < rhat_target), None)
    hit_all = next((t for t in trace if t["ess_bulk_common"] >= ess_target and t["rhat_max"] < rhat_target), None)
    extra = dict(extra, trace=trace, until=list(until) if until else None,
                 target_common=hit, target_common_and_max_rhat=hit_all, t_compile_sampling=t_cs)
    return {"w": Wd, "grads_per_draw": st, "accept": np.concatenate(accs, 0).swapaxes(0, 1),
            "divergent": np.concatenate(divs, 0).swapaxes(0, 1), "t_compile": t_compile + t_cs,
            "t_warm": t_warm, "t_samp": t_samp, "warm_grads": warm_grads, "extra": extra}


def run_np_nuts(tg: Target, C, W, N, seed, block=50, init_step=1.0, adapt=True, until=None, target_accept=0.8):
    """NumPyro NUTS kernel (production algorithm: windowed step-size dual averaging, fixed metric)
    driven by our own scan loops (AOT compile, synchronised timers) instead of ``MCMC``."""
    from numpyro.infer import NUTS

    D = tg.D
    kernel = NUTS(potential_fn=lambda w: -tg.logdensity(w), dense_mass=True, inverse_mass_matrix=jnp.eye(D),
                  adapt_mass_matrix=False, adapt_step_size=adapt, step_size=init_step,
                  target_accept_prob=target_accept, max_tree_depth=10)
    w0 = tg.init_w(C, seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), C)

    def init(keys, w0):
        return jax.vmap(lambda k, w: kernel.init(k, W, init_params=w, model_args=(), model_kwargs={}))(keys, w0)

    ci, t_ci = _aot(init, keys, w0)
    states, t_init = _timed(ci, keys, w0)

    def step(st, _):
        st = jax.vmap(lambda s: kernel.sample(s, (), {}))(st)
        return st, (st.z, st.num_steps, st.accept_prob, st.diverging)

    def warm(st):
        return jax.lax.scan(step, st, None, length=W)

    if W > 0:
        cw, t_cw = _aot(warm, states)
        (states, (_, ns_w, _, _)), t_warm = _timed(cw, states)
        warm_grads = int(np.sum(np.asarray(ns_w)))
    else:
        t_cw, t_warm, warm_grads = 0.0, 0.0, 0
    step_size = np.asarray(states.adapt_state.step_size)

    def samp_block(st, keys):  # keys unused (state carries its own rng); kept for a common signature
        return jax.lax.scan(lambda s, _k: step(s, None), st, keys)

    return _adaptive_blocks(tg, samp_block, states, jax.random.PRNGKey(seed + 1000), N, block, until,
                            t_ci + t_cw, t_init + t_warm, warm_grads,
                            {"step_size": step_size.tolist(), "init_step": init_step, "adapt": adapt, "driver": "numpyro-kernel"})


def run_bj_nuts2(tg: Target, C, W, N, seed, block=50, init_step=1.0, adapt=True, until=None, target_accept=0.8):
    """BlackJAX NUTS, continuous dual averaging of the step size (no windows), same conventions."""
    import blackjax
    from blackjax.adaptation.step_size import dual_averaging_adaptation

    logd = tg.logdensity
    imm = jnp.ones(tg.D)
    w0 = tg.init_w(C, seed)
    da_init, da_update, da_final = dual_averaging_adaptation(target_accept)
    ci, t_ci = _aot(lambda w: jax.vmap(lambda x: blackjax.nuts.init(x, logd))(w), w0)
    states, t_init = _timed(ci, w0)
    kw, ks = jax.random.split(jax.random.PRNGKey(seed))

    def warm_step(carry, k):
        st, da = carry

        def one(k_, s_, d_):
            s2, info = blackjax.nuts(logd, jnp.exp(d_.log_step_size), imm).step(k_, s_)
            return s2, da_update(d_, info.acceptance_rate), info.num_integration_steps

        st, da, ns = jax.vmap(one)(jax.random.split(k, C), st, da)
        return (st, da), ns

    da0 = jax.vmap(da_init)(jnp.full(C, init_step))
    if adapt and W > 0:
        wkeys = jax.random.split(kw, W)
        cw, t_cw = _aot(lambda st, da, keys: jax.lax.scan(warm_step, (st, da), keys), states, da0, wkeys)
        ((states, da), ns_w), t_warm = _timed(cw, states, da0, wkeys)
        step = jax.vmap(da_final)(da)
        warm_grads = int(np.sum(np.asarray(ns_w)))
    else:
        t_cw, t_warm, warm_grads, step = 0.0, 0.0, 0, jnp.full(C, init_step)

    def samp_step(st, k):
        def one(k_, s_, h_):
            s2, info = blackjax.nuts(logd, h_, imm).step(k_, s_)
            return s2, (s2.position, info.num_integration_steps, info.acceptance_rate, info.is_divergent)

        return jax.vmap(one)(jax.random.split(k, C), st, step)

    return _adaptive_blocks(tg, lambda st, keys: jax.lax.scan(samp_step, st, keys), states, ks, N, block, until,
                            t_ci + t_cw, t_init + t_warm, warm_grads,
                            {"step_size": np.asarray(step).tolist(), "init_step": init_step, "adapt": adapt, "driver": "blackjax"})


def run_mclmc(tg: Target, C, W, N, seed, block=500, thin=1):
    """Unadjusted MCLMC (biased unless shown otherwise); BlackJAX tuner per chain."""
    import blackjax

    logd = tg.logdensity
    key = jax.random.PRNGKey(seed)
    kinit, ktune, ks = jax.random.split(key, 3)
    w0 = tg.init_w(C, seed)
    states = jax.vmap(lambda w, k: blackjax.mcmc.mclmc.init(w, logd, k))(w0, jax.random.split(kinit, C))
    kernel = blackjax.mcmc.mclmc.build_kernel()

    def tune(st, k):
        s2, params, nsteps = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, num_steps=int(W / 0.3), state=st, rng_key=k, logdensity_fn=logd,
            frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1, diagonal_preconditioning=False,
        )
        return s2, params, nsteps

    def tune_all(st, keys):
        return jax.vmap(tune)(st, keys)

    tkeys = jax.random.split(ktune, C)
    ct, t_ct = _aot(tune_all, states, tkeys)
    (states, params, nsteps), t_warm = _timed(ct, states, tkeys)
    warm_steps = int(np.max(np.asarray(nsteps)))  # the tuner's own integrator-step count (per chain)
    L, eps = params.L, params.step_size

    def samp_step(st, k):
        def one(k_, s_, L_, e_):
            s2, info = kernel(k_, s_, logd, 1.0, L_, e_)
            return s2, (s2.position, jnp.int32(1), jnp.float64(1.0), ~info.nonans)

        return jax.vmap(one)(jax.random.split(k, C), st, L, eps)

    def samp_block(st, keys):
        return jax.lax.scan(samp_step, st, keys)

    return _sample_blocks(tg, samp_block, states, ks, N, block, t_ct, t_warm, 2 * warm_steps * C,
                          extra={"L": np.asarray(L).tolist(), "step_size": np.asarray(eps).tolist(), "unadjusted": True},
                          grads_per_step=2)


def run_mams(tg: Target, C, W, N, seed, block=200, target_accept=0.9, avg_steps=None):
    """Adjusted MCLMC with random trajectory length (MAMS; exact MH correction). BlackJAX tuner.

    ``avg_steps``: target average number of integration steps per proposal (BlackJAX default 2);
    the tuner pins L = avg_steps * step_size."""
    import blackjax
    from blackjax.mcmc import adjusted_mclmc_dynamic as amd

    logd = tg.logdensity
    key = jax.random.PRNGKey(seed)
    kinit, ktune, ks = jax.random.split(key, 3)
    w0 = tg.init_w(C, seed)
    # shared random_generator_arg -> all chains draw the same number of steps (no lockstep loss)
    states = jax.vmap(lambda w: amd.init(w, logd, kinit))(w0)
    steps_fn = amd.make_random_trajectory_length_fn(True)
    kernel = amd.build_kernel(integration_steps_fn=steps_fn)
    avg = 2.0 if avg_steps is None else float(avg_steps)

    def tune(st, k):
        s2, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, logdensity_fn=logd, num_steps=int(W / 0.2), state=st, rng_key=k, target=target_accept,
            frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, diagonal_preconditioning=False,
            target_num_integration_steps=avg,
        )
        return s2, params

    tkeys = jax.random.split(ktune, C)
    ct, t_ct = _aot(lambda st, keys: jax.vmap(tune)(st, keys), states, tkeys)
    (states, params), t_warm = _timed(ct, states, tkeys)
    eps = params.step_size
    avgn = params.L / params.step_size
    warm_props = int(0.1 * int(W / 0.2)) * 2

    def samp_step(st, k):
        def one(k_, s_, e_, a_):
            s2, info = kernel(k_, s_, logd, e_, jnp.inf, 1.0, (a_,))
            return s2, (s2.position, info.num_integration_steps, info.acceptance_rate, info.is_divergent)

        return jax.vmap(one)(jax.random.split(k, C), st, eps, avgn)

    def samp_block(st, keys):
        return jax.lax.scan(samp_step, st, keys)

    return _sample_blocks(tg, samp_block, states, ks, N, block, t_ct, t_warm,
                          int(2 * warm_props * float(np.mean(np.asarray(avgn))) * C),
                          extra={"step_size": np.asarray(eps).tolist(), "avg_steps": np.asarray(avgn).tolist(),
                                 "target_accept": target_accept, "warm_grads_approx": True},
                          grads_per_step=2)


def run_chees(tg: Target, C, W, N, seed, block=100, lr=0.025, step0=0.1):
    """ChEES-HMC (jittered trajectory length, cross-chain adaptation). Exact (MH-corrected HMC)."""
    import blackjax
    import optax
    from blackjax.mcmc import dynamic_hmc

    logd = tg.logdensity
    key = jax.random.PRNGKey(seed)
    kw, ks = jax.random.split(key)
    w0 = tg.init_w(C, seed)
    warmup = blackjax.chees_adaptation(logd, C)
    optim = optax.adam(lr)

    def warm(k, pos):
        (st, params), _info = warmup.run(k, pos, step0, optim, W, max_sampling_steps=N)
        return st, params["step_size"], params["integration_steps_params"][0], params["inverse_mass_matrix"]

    cw, t_cw = _aot(warm, kw, w0)
    (states, eps, nleap, imm), t_warm = _timed(cw, kw, w0)
    # rebuild the (non-array) jitter function exactly as chees_adaptation does (Halton, jitter 1)
    max_bits = np.ceil(np.log2(W + N))

    def integration_steps_fn(i, n_leap):
        return jnp.asarray(jnp.ceil(dynamic_hmc.halton_sequence(i, max_bits) * n_leap), dtype=int)

    kern = dynamic_hmc.build_kernel(next_random_arg_fn=lambda i: i + 1, integration_steps_fn=integration_steps_fn)
    def samp_step(st, k):
        def one(k_, s_):
            s2, info = kern(k_, s_, logd, eps, imm, integration_steps_params=(nleap,))
            return s2, (s2.position, info.num_integration_steps, info.acceptance_rate, info.is_divergent)

        return jax.vmap(one)(jax.random.split(k, C), st)

    def samp_block(st, keys):
        return jax.lax.scan(samp_step, st, keys)

    wg = -1  # ChEES warmup cost: W iterations x (jittered) leapfrog steps; not reported by BlackJAX
    return _sample_blocks(tg, samp_block, states, ks, N, block, t_cw, t_warm, wg,
                          extra={"step_size": float(eps), "mean_leapfrog": float(nleap), "lr": lr})


def run_meads(tg: Target, C, W, N, seed, block=500, thin=1):
    """MEADS (generalised HMC, 1 leapfrog step per iteration, cross-chain adaptation). Exact."""
    import blackjax

    logd = tg.logdensity
    key = jax.random.PRNGKey(seed)
    kw, ks = jax.random.split(key)
    w0 = tg.init_w(C, seed)
    warmup = blackjax.meads_adaptation(logd, C)

    def warm(k, pos):
        (st, params), _ = warmup.run(k, pos, W)
        return st, params

    cw, t_cw = _aot(warm, kw, w0)
    (states, params), t_warm = _timed(cw, kw, w0)
    alg = blackjax.ghmc(logd, **params)

    def samp_step(st, k):
        def one(k_, s_):
            s2, info = alg.step(k_, s_)
            return s2, (s2.position, jnp.int32(1), info.acceptance_rate, info.is_divergent)

        return jax.vmap(one)(jax.random.split(k, C), st)

    def samp_block(st, keys):
        return jax.lax.scan(samp_step, st, keys)

    return _sample_blocks(tg, samp_block, states, ks, N, block, t_cw, t_warm, W * C,
                          extra={k: (float(v) if np.ndim(v) == 0 else "array") for k, v in params.items()})


# ---------------------------------------------------------------------- NumPyro (production driver)


def run_numpyro(tg: Target, C, W, N, seed, block=250, **_):
    from ptagwb.sampling import RunConfig, run_nuts

    name = f"perf_numpyro_{tg.model}_{tg.like_variant}_s{seed}_c{C}"
    cfg = RunConfig(
        name=name,
        model={"orf": tg.model, "gamma": 13 / 3, "n_common": 14, "position": "enterprise"},
        num_chains=C, num_warmup=W, num_samples=N, block=min(block, N), seed=seed,
        target_accept=0.8, max_tree_depth=10, dense_mass=True, chain_method="vectorized",
        adapt_mass_matrix=False, init=f"run:{CURN_RUN}", metric=f"run:{CURN_RUN}",
        notes="perf study: production NUTS driver, fixed CURN metric, step-size-only warmup",
    )
    d = run_nuts(cfg, tg.post, log=lambda s: print(s, flush=True))
    r = load_run(name)
    meta = r["meta"]
    z = r["z"]
    w = np.linalg.solve(np.asarray(tg.Lc), (z.reshape(-1, tg.D) - np.asarray(tg.mu)).T).T.reshape(z.shape)
    return {
        "w": w,
        "grads_per_draw": r["num_steps"],
        "accept": r["accept_prob"],
        "divergent": r["diverging"],
        "t_compile": float("nan"),  # included in warmup by the production driver
        "t_warm": meta["warmup_seconds"],
        "t_samp": meta["sampling_seconds"],
        "warm_grads": meta["warmup_grad_evals"],
        "extra": {"step_size": meta["step_size"], "run_dir": str(d)},
    }


SAMPLERS = {"numpyro": run_numpyro, "bj_nuts": run_bj_nuts, "np_nuts": run_np_nuts, "bj_nuts2": run_bj_nuts2, "mclmc": run_mclmc, "mams": run_mams,
            "chees": run_chees, "meads": run_meads}


# ---------------------------------------------------------------------- summary


def summarize(tg: Target, res: dict, thin: int = 1) -> dict:
    W = res["w"][:, ::thin]
    C, N, D = W.shape
    Z = tg.w_to_z(W.reshape(-1, D)).reshape(C, N, D)
    X = np.asarray(tg.post.transform.to_constrained(Z))
    bulk = np.array([dg.ess_bulk(X[:, :, j]) for j in range(D)])
    tail = np.array([dg.ess_tail(X[:, :, j]) for j in range(D)])
    rh = np.array([dg.rhat(X[:, :, j]) for j in range(D)])
    jc = tg.names.index("gw_log10_A")
    grads = float(np.sum(res["grads_per_draw"]))
    t = res["t_samp"]
    s = {
        "chains": C, "draws_per_chain": N, "thin": thin,
        "grads_sampling": grads, "grads_warmup": res["warm_grads"],
        "grads_per_draw_mean": float(np.mean(res["grads_per_draw"])),
        "t_compile": res["t_compile"], "t_warm": res["t_warm"], "t_samp": t,
        "accept_mean": float(np.mean(res["accept"])), "divergences": int(np.sum(res["divergent"])),
        # vectorised chains wait for the longest trajectory: useful / executed leapfrog steps
        "lockstep_efficiency": float(np.mean(res["grads_per_draw"]) / np.mean(np.max(res["grads_per_draw"], axis=0))),
        "ms_per_lockstep_step": 1e3 * t / float(np.sum(np.max(res["grads_per_draw"], axis=0))),
        "t_total": float(np.nansum([res["t_compile"], res["t_warm"], t])),
        "ess_bulk_common": float(bulk[jc]), "ess_tail_common": float(tail[jc]),
        "ess_bulk_min": float(bulk.min()), "ess_tail_min": float(tail.min()),
        "ess_bulk_min_param": tg.names[int(bulk.argmin())], "ess_tail_min_param": tg.names[int(tail.argmin())],
        "ess_bulk_median": float(np.median(bulk)),
        "rhat_max": float(np.nanmax(rh)), "rhat_max_param": tg.names[int(np.nanargmax(rh))],
        "rhat_common": float(rh[jc]),
    }
    for k in ("ess_bulk_common", "ess_tail_common", "ess_bulk_min", "ess_tail_min"):
        s[k.replace("ess", "ess_per_s")] = s[k] / t
        s[k.replace("ess", "ess_per_kgrad")] = 1e3 * s[k] / grads
    s["extra"] = res["extra"]
    return s, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="curn", choices=["curn", "hd"])
    ap.add_argument("--sampler", required=True, choices=list(SAMPLERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--like", default="aug", help="likelihood variant: prod | aug | aug+trsm ...")
    ap.add_argument("--thin", type=int, default=1, help="thinning for the stored draws / ESS")
    ap.add_argument("--opt", action="append", default=[], help="sampler kwarg, e.g. --opt avg_steps=8")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    kw = {}
    for kv in args.opt:
        k, v = kv.split("=", 1)
        kw[k] = json.loads(v)
    tg = Target(args.model, args.like)
    tag = args.tag or f"{args.model}_{args.sampler}_{args.like}_c{args.chains}_s{args.seed}_{jax.default_backend()}"
    print(f"[{tag}] D={tg.D} start", flush=True)
    t0 = time.time()
    res = SAMPLERS[args.sampler](tg, args.chains, args.warmup, args.samples, args.seed, **kw)
    summ, X = summarize(tg, res, args.thin)
    summ.update(
        model=args.model, sampler=args.sampler, seed=args.seed, like=args.like, opts=kw, warmup=args.warmup,
        samples=args.samples, wall_total=time.time() - t0, env=env_info(),
    )
    RUNS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / f"{tag}.npz", x=X.astype(np.float32), names=np.array(tg.names),
                        grads_per_draw=res["grads_per_draw"], accept=res["accept"])
    (OUT / f"{tag}.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("env", "extra")}, default=float), flush=True)


if __name__ == "__main__":
    main()
