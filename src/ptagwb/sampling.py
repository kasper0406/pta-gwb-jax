"""Gradient-based posterior sampling (NumPyro NUTS) for the marginalised PTA likelihood.

Why NumPyro NUTS
----------------
* It accepts a plain ``potential_fn`` on a flat unconstrained vector, so our likelihood (custom
  VJPs, float64) is used as is; no probabilistic-program rewrite.
* Stan-style windowed warmup with **dense** mass-matrix adaptation. The 67 (log10_A, gamma) IRN
  pairs are strongly correlated (banana-shaped where the IRN is detected), and the common
  (gamma, log10_A) pair is correlated at f_ref = 1/yr; a dense metric absorbs the linear part.
* Multinomial NUTS with float64 energies: the accept/reject (trajectory sampling) uses the exact
  float64 Hamiltonian, so a deterministic approximate gradient (``grad_precision="mixed"``)
  changes efficiency but not the stationary distribution (leapfrog with any deterministic
  position-only force field is reversible and volume preserving). ``tests/test_sampling.py``
  checks this on a toy problem.
* Chains can run ``"vectorized"`` (one batched GPU call per leapfrog step for all chains) or
  ``"sequential"``; numpyro's ``MCMC.warmup`` / ``post_warmup_state`` let us sample in blocks
  and checkpoint to disk.

BlackJAX would work as well, but would need our own warmup/multi-chain driver for the same
features.

Parameterisation
----------------
Every parameter has a uniform prior on a box [lo, hi] (``config.PRIORS``). The sampler works on
z in R^D with

    x = lo + (hi - lo) * sigmoid(z),
    log p(z) = log L(x(z)) + log pi(x) + log|dx/dz|,
    log|dx/dz| = sum_i [log(hi_i - lo_i) - softplus(-z_i) - softplus(z_i)],
    log pi(x)  = -sum_i log(hi_i - lo_i)     (inside the box)

so the uniform-prior density and the log(hi - lo) of the Jacobian cancel; we keep both terms
explicit (``BoxTransform``) so that ``log_prior + log_jacobian`` can be tested against autodiff.

Parameter layout (flat vector): ``[rn_log10_A (P), rn_gamma (P), common...]`` with ``common`` =
``log10_A`` (fixed gamma), ``gamma, log10_A`` (varied gamma) or ``log10_rho_0..n-1`` (free
spectrum). Names follow the enterprise chains (``<psr>_red_noise_log10_A``, ``gw_log10_A``...).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import PRIORS, REPO_ROOT, enable_x64

enable_x64()


# ---------------------------------------------------------------------- transform


@dataclass(frozen=True)
class BoxTransform:
    """Logistic bijection R^D -> prod_i (lo_i, hi_i) for uniform box priors."""

    names: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        lo, hi = np.asarray(self.lo, np.float64), np.asarray(self.hi, np.float64)
        if lo.shape != (len(self.names),) or hi.shape != lo.shape:
            raise ValueError("lo/hi must have one entry per parameter name")
        if not np.all(np.isfinite(lo) & np.isfinite(hi) & (hi > lo)):
            raise ValueError("box bounds must be finite with hi > lo")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def log_volume(self) -> float:
        return float(np.sum(np.log(self.hi - self.lo)))

    def to_constrained(self, z):
        xp = jnp if isinstance(z, jax.Array) else np
        sig = 1.0 / (1.0 + xp.exp(-z)) if xp is np else jax.nn.sigmoid(z)
        return self.lo + (self.hi - self.lo) * sig

    def to_unconstrained(self, x):
        u = (np.asarray(x, np.float64) - self.lo) / (self.hi - self.lo)
        if np.any((u <= 0) | (u >= 1)):
            raise ValueError("point outside the open prior box")
        return np.log(u) - np.log1p(-u)

    def log_jacobian(self, z):
        """log|dx/dz| summed over the last axis."""
        z = jnp.asarray(z)
        return jnp.sum(jnp.log(self.hi - self.lo) - jax.nn.softplus(-z) - jax.nn.softplus(z), axis=-1)

    def log_prior(self, x):
        """Normalised uniform log density (-inf outside the closed box)."""
        x = jnp.asarray(x)
        inside = jnp.all((x >= self.lo) & (x <= self.hi), axis=-1)
        return jnp.where(inside, -self.log_volume, -jnp.inf)


# ---------------------------------------------------------------------- model specification


@dataclass
class ModelSpec:
    """Which likelihood/prior to sample.

    ``gamma``: fixed common spectral index (e.g. 13/3) or ``None`` for a varied gamma.
    ``prior_overrides``: {"gw_log10_A": (lo, hi), ...} to change ``config.PRIORS`` entries.
    """

    orf: str = "curn"  # "curn" or "hd" (any likelihood ORF string)
    common: str = "powerlaw"  # or "freespec"
    gamma: float | None = None
    n_common: int = 14
    n_modes: int = 30
    position: str = "enterprise"  # "enterprise" (as the released chains) or "icrs"
    grad_precision: str = "float64"
    prior_overrides: dict = field(default_factory=dict)

    def prior_bounds(self) -> dict[str, tuple[float, float]]:
        pri = dict(PRIORS)
        for k, v in self.prior_overrides.items():
            if k not in pri:
                raise KeyError(f"unknown prior key {k!r}")
            pri[k] = tuple(float(a) for a in v)
        return pri


def parameter_names(spec: ModelSpec, psr_names: list[str]) -> list[str]:
    names = [f"{p}_red_noise_log10_A" for p in psr_names] + [f"{p}_red_noise_gamma" for p in psr_names]
    if spec.common == "freespec":
        names += [f"gw_log10_rho_{i}" for i in range(spec.n_common)]
    elif spec.gamma is None:
        names += ["gw_gamma", "gw_log10_A"]
    else:
        names += ["gw_log10_A"]
    return names


def build_transform(spec: ModelSpec, psr_names: list[str]) -> BoxTransform:
    pri = spec.prior_bounds()
    P = len(psr_names)
    lo = [pri["rn_log10_A"][0]] * P + [pri["rn_gamma"][0]] * P
    hi = [pri["rn_log10_A"][1]] * P + [pri["rn_gamma"][1]] * P
    if spec.common == "freespec":
        lo += [pri["freespec_log10_rho"][0]] * spec.n_common
        hi += [pri["freespec_log10_rho"][1]] * spec.n_common
    elif spec.gamma is None:
        lo += [pri["gw_gamma"][0], pri["gw_log10_A"][0]]
        hi += [pri["gw_gamma"][1], pri["gw_log10_A"][1]]
    else:
        lo += [pri["gw_log10_A_fixed_gamma"][0]]
        hi += [pri["gw_log10_A_fixed_gamma"][1]]
    return BoxTransform(tuple(parameter_names(spec, psr_names)), np.array(lo), np.array(hi))


def unpack(x, spec: ModelSpec, P: int) -> dict:
    """Flat constrained vector (..., D) -> ``PTALikelihood.logL`` argument dict."""
    out = {"rn_log10_A": x[..., :P], "rn_gamma": x[..., P : 2 * P]}
    rest = x[..., 2 * P :]
    if spec.common == "freespec":
        out["log10_rho"] = rest
    elif spec.gamma is None:
        out["gamma"], out["log10_A"] = rest[..., 0], rest[..., 1]
    else:
        out["log10_A"] = rest[..., 0]
        out["gamma"] = jnp.full(rest.shape[:-1], spec.gamma) if isinstance(x, jax.Array) else np.full(rest.shape[:-1], spec.gamma)
    return out


def pack_named(named: dict, names: list[str]) -> np.ndarray:
    """Named sample (enterprise-style keys) -> flat constrained vector in ``names`` order."""
    return np.array([float(named[n]) for n in names])


class Posterior:
    """log posterior on the unconstrained space for a ``PTALikelihood`` and a ``ModelSpec``."""

    def __init__(self, like, spec: ModelSpec):
        self.like, self.spec = like, spec
        self.P = like.P
        self.transform = build_transform(spec, like.names)
        self.names = list(self.transform.names)
        if spec.common == "freespec" and like.common != "freespec":
            raise ValueError("spec/likelihood common-spectrum mismatch")

        def logL_x(x):
            return like._logL(unpack(x, spec, self.P))

        self._init_from(logL_x)

    def _init_from(self, logL_x):
        tr = self.transform

        def logpost_z(z):
            return logL_x(tr.to_constrained(z)) + tr.log_jacobian(z) - tr.log_volume

        self.logL_x_raw = logL_x
        self.logL_x = jax.jit(logL_x)
        self.logL_x_batched = jax.jit(jax.vmap(logL_x))
        self.logpost_z = logpost_z
        self.potential_fn = lambda z: -logpost_z(z)

    @classmethod
    def generic(cls, transform: BoxTransform, logL_x) -> Posterior:
        """Posterior for an arbitrary log-likelihood of the constrained vector (tests, toys)."""
        self = cls.__new__(cls)
        self.like, self.spec, self.P = None, None, 0
        self.transform, self.names = transform, list(transform.names)
        self._init_from(logL_x)
        return self

    def logL_samples(self, X: np.ndarray, batch: int = 64) -> np.ndarray:
        """log-likelihood at constrained samples X (N, D), batched on the device."""
        X = np.asarray(X, np.float64)
        out = np.empty(len(X))
        for i in range(0, len(X), batch):
            xb = X[i : i + batch]
            n = len(xb)
            if n < batch:  # pad to a fixed shape (one compilation)
                xb = np.concatenate([xb, np.repeat(xb[-1:], batch - n, axis=0)])
            out[i : i + n] = np.asarray(self.logL_x_batched(jnp.asarray(xb)))[:n]
        return out


# ---------------------------------------------------------------------- run driver

LIKELIHOOD_IMPLS = ("production", "fast")


def make_likelihood(terms, T, spec: ModelSpec, impl: str = "production"):
    """The likelihood object for ``spec`` (``RunConfig.likelihood_impl``)."""
    kw = {"n_modes": spec.n_modes, "n_common": spec.n_common, "orf": spec.orf, "common": spec.common, "grad_precision": spec.grad_precision}
    if impl == "production":
        from .likelihood import PTALikelihood

        return PTALikelihood(terms, T, **kw)
    if impl == "fast":
        from .perf_likelihood import FastPTALikelihood

        return FastPTALikelihood(terms, T, reduce="hh", tri_inv="levels", **kw)
    raise ValueError(f"unknown likelihood_impl {impl!r}; one of {LIKELIHOOD_IMPLS}")



@dataclass
class RunConfig:
    name: str
    model: dict  # ModelSpec fields
    num_chains: int = 4
    num_warmup: int = 1000
    num_samples: int = 1000
    block: int = 250  # samples per checkpoint block
    seed: int = 0
    target_accept: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    chain_method: str = "vectorized"
    init: str = "prior"  # "prior" or "run:<name>" (random posterior draws of an earlier run)
    init_radius: float = 2.0  # prior init: z ~ U(-r, r)
    # "adapt": numpyro windowed adaptation (dense or diagonal per ``dense_mass``);
    # "run:<name>[,<name>...]": pooled sample covariance of earlier runs' unconstrained draws
    # (parameters missing there get unit variance), optionally further adapted.
    metric: str = "adapt"
    adapt_mass_matrix: bool = True
    # initial NUTS step size and whether warmup adapts it (numpyro defaults: 1.0, True). With
    # adapt_step_size=False the given step size is used throughout (e.g. a value adapted earlier).
    step_size: float = 1.0
    adapt_step_size: bool = True
    # exact hybrid kernel (``hybrid.HybridNUTS``): path (relative to the repo) of a frozen block
    # proposal file (scripts/fs_fit_proposals.py); "" = plain NUTS. ``jump_sweeps`` block-MH sweeps
    # follow every NUTS transition (warmup included).
    jumps: str = ""
    jump_sweeps: int = 1
    # Metropolised conditional-grid moves (``hybrid.make_grid_moves``) for these free-spectrum bins
    # (0-based), after the block sweep; ``grid_kw``: n_coarse, n_fine, half_width, w_uniform.
    grid_bins: list = field(default_factory=list)
    grid_kw: dict = field(default_factory=dict)
    # hard cap on the sampling phase: stop after the first checkpoint block that ends beyond it
    # (0 = none); the run then holds fewer than num_samples draws (meta: stopped_by_cap)
    max_sampling_seconds: float = 0.0
    # init "run:<name>" only: each free-spectrum bin of each chain is independently moved, with this
    # probability, to a uniform draw in [lo + 0.5, -10] (deliberately diverse region starts)
    init_rho_low_frac: float = 0.0
    progress_bar: bool = False
    # sampler backend; only "nuts" (numpyro) is implemented. The model/init/chain/draw fields above
    # are backend-independent; target_accept, max_tree_depth, dense_mass, metric are NUTS-specific.
    sampler: str = "nuts"
    # likelihood implementation: "production" (``likelihood.PTALikelihood``, default) or "fast"
    # (opt-in ``perf_likelihood.FastPTALikelihood(reduce="hh", tri_inv="levels")``: same
    # quantities, exact within the budgets of tests/test_perf_likelihood.py; docs/PERF.md).
    likelihood_impl: str = "production"
    notes: str = ""

    def __post_init__(self):
        if self.likelihood_impl not in LIKELIHOOD_IMPLS:
            raise ValueError(f"unknown likelihood_impl {self.likelihood_impl!r}; one of {LIKELIHOOD_IMPLS}")

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        d = json.loads(Path(path).read_text())
        return cls(**d)

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(**self.model)


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "dirty_files": [ln for ln in run("git", "status", "--porcelain", "--untracked-files=no").splitlines() if ln],
    }


RUNS_DIR = REPO_ROOT / "runs"


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def load_run(name: str) -> dict:
    """Samples + sidecar of a finished (or partially finished) run."""
    d = run_dir(name)
    with np.load(d / "samples.npz", allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    out["meta"] = json.loads((d / "meta.json").read_text())
    out["names"] = out["meta"]["names"]
    return out


def _init_points(cfg: RunConfig, post: Posterior, rng: np.random.Generator):
    """Unconstrained initial points (C, D)."""
    C, D = cfg.num_chains, post.transform.dim
    if cfg.init == "prior":
        return rng.uniform(-cfg.init_radius, cfg.init_radius, size=(C, D))
    if cfg.init.startswith("run:"):
        src = load_run(cfg.init[4:])
        names = post.names
        X = src["x"].reshape(-1, src["x"].shape[-1])
        idx = {n: i for i, n in enumerate(src["names"])}
        sel = rng.choice(len(X), size=C, replace=False)
        z0 = np.empty((C, D))
        for c, s in enumerate(sel):
            # map each parameter separately (bounds may differ between models)
            for j, n in enumerate(names):
                if n in idx:
                    lo, hi = post.transform.lo[j], post.transform.hi[j]
                    u = np.clip((X[s, idx[n]] - lo) / (hi - lo), 1e-6, 1 - 1e-6)
                    z0[c, j] = np.log(u) - np.log1p(-u)
                else:
                    z0[c, j] = rng.uniform(-cfg.init_radius, cfg.init_radius)
        if cfg.init_rho_low_frac > 0:
            for j, n in enumerate(names):
                if n.startswith("gw_log10_rho_"):
                    lo, hi = post.transform.lo[j], post.transform.hi[j]
                    for c in range(C):
                        if rng.uniform() < cfg.init_rho_low_frac:
                            u = (rng.uniform(lo + 0.5, -10.0) - lo) / (hi - lo)
                            z0[c, j] = np.log(u) - np.log1p(-u)
        return z0
    raise ValueError(f"unknown init {cfg.init!r}")


def metric_from_runs(run_names: list[str], names: list[str]) -> np.ndarray:
    """Pooled covariance of earlier runs' unconstrained draws, embedded in ``names`` order.

    Parameters absent from all runs get unit variance and no correlation. Shared parameters must
    have the same prior box (so that z means the same thing); checked.
    """
    D = len(names)
    blocks = []
    for rn in run_names:
        src = load_run(rn)
        idx = {n: i for i, n in enumerate(src["names"])}
        lo_s, hi_s = np.array(src["meta"]["lo"]), np.array(src["meta"]["hi"])
        Z = src["z"].reshape(-1, src["z"].shape[-1])
        E = np.full((len(Z), D), np.nan)
        for j, n in enumerate(names):
            if n in idx:
                E[:, j] = Z[:, idx[n]]
        blocks.append((E - np.nanmean(E, axis=0), idx, lo_s, hi_s))
    E = np.concatenate([b[0] for b in blocks])
    have = ~np.isnan(E)
    E0 = np.where(have, E, 0.0)
    cnt = have.T.astype(float) @ have.astype(float)
    C = (E0.T @ E0) / np.maximum(cnt - 1, 1)
    known = np.diag(cnt) > 10
    cov = np.where(np.outer(known, known), C, 0.0)
    cov[np.diag_indices(D)] = np.where(known, np.diag(C), 1.0)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    w = np.maximum(w, 1e-8 * w.max())
    return (V * w) @ V.T


def run_nuts(cfg: RunConfig, post: Posterior, log=print) -> Path:
    """Warmup + block-wise sampling with checkpoints in ``runs/<name>/``.

    Files: ``meta.json`` (config, git SHA, timings, adapted metric summary) and ``samples.npz``
    with ``x`` (C, N, D) constrained draws, ``z`` unconstrained, ``potential_energy``,
    ``diverging``, ``num_steps``, ``accept_prob``, ``tree_depth``, ``logL``, plus the adapted
    ``step_size`` (C,) and ``inverse_mass_matrix`` (C, D, D).
    """
    from numpyro.infer import MCMC, NUTS

    if cfg.sampler != "nuts":
        raise NotImplementedError(f"sampler {cfg.sampler!r} not implemented (only 'nuts')")
    out = run_dir(cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    z0 = _init_points(cfg, post, rng)
    kw = {}
    if cfg.metric.startswith("run:"):
        imm0 = metric_from_runs(cfg.metric[4:].split(","), post.names)
        kw["inverse_mass_matrix"] = jnp.asarray(imm0 if cfg.dense_mass else np.diag(imm0))
        np.save(out / "initial_inverse_mass_matrix.npy", imm0)
    elif cfg.metric != "adapt":
        raise ValueError(f"unknown metric {cfg.metric!r}")
    nuts_kw = dict(
        dense_mass=cfg.dense_mass,
        target_accept_prob=cfg.target_accept,
        max_tree_depth=cfg.max_tree_depth,
        adapt_mass_matrix=cfg.adapt_mass_matrix,
        step_size=cfg.step_size,
        adapt_step_size=cfg.adapt_step_size,
        **kw,
    )
    fields = ("potential_energy", "diverging", "num_steps", "accept_prob")
    if cfg.jumps or cfg.grid_bins:
        from .hybrid import BlockProposals, HybridNUTS

        prop = BlockProposals.from_json(cfg.jumps, post.names) if cfg.jumps else None
        kernel = HybridNUTS(post, prop, sweeps=cfg.jump_sweeps, grid_bins=list(cfg.grid_bins), grid_kw=dict(cfg.grid_kw),
                            **nuts_kw)
        fields = fields + ("trajectory_length",)  # per-block accepted jumps (see hybrid.py)
    else:
        kernel = NUTS(potential_fn=post.potential_fn, **nuts_kw)
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.block,
        num_chains=cfg.num_chains,
        chain_method=cfg.chain_method,
        progress_bar=cfg.progress_bar,
    )
    meta = {
        "config": asdict(cfg),
        "git": git_state(),
        "names": post.names,
        "lo": post.transform.lo.tolist(),
        "hi": post.transform.hi.tolist(),
        "log_volume": post.transform.log_volume,
        "likelihood": {
            k: getattr(post.like, a, None)
            for k, a in (
                ("orf", "orf_name"),
                ("common", "common"),
                ("n_common", "n_common"),
                ("n_modes", "n_modes"),
                ("convention", "convention"),
                ("grad_precision", "grad_precision"),
                ("reduce", "reduce_name"),
                ("tri_inv", "tri_inv"),
            )
        }
        | {"class": type(post.like).__name__ if post.like is not None else None},
        "host": os.uname().nodename,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(dv) for dv in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    key = jax.random.PRNGKey(cfg.seed)
    t0 = time.time()
    init = jnp.asarray(z0 if cfg.num_chains > 1 else z0[0])
    mcmc.warmup(key, init_params=init, extra_fields=fields, collect_warmup=True)
    t_warm = time.time() - t0
    wx = mcmc.get_extra_fields(group_by_chain=True)
    warm_steps = int(np.sum(np.asarray(wx["num_steps"])))
    st = mcmc.last_state.adapt_state
    step_size = np.atleast_1d(np.asarray(st.step_size))
    imm = st.inverse_mass_matrix
    if isinstance(imm, dict):  # numpyro >= 0.13 may key the metric by site tuple
        imm = next(iter(imm.values()))
    imm = np.asarray(imm)
    if cfg.num_chains == 1:
        imm = imm[None]
    meta.update(
        warmup_seconds=t_warm,
        warmup_grad_evals=warm_steps,
        warmup_divergences=int(np.sum(np.asarray(wx["diverging"]))),
        step_size=step_size.tolist(),
    )
    if cfg.jumps or cfg.grid_bins:
        wa = np.asarray(wx["trajectory_length"])
        meta.update(warmup_jump_accept_rate=(wa.reshape(-1, wa.shape[-1]).mean(axis=0) / cfg.jump_sweeps).tolist())
    log(f"[{cfg.name}] warmup done in {t_warm:.0f} s, {warm_steps} grad evals, step sizes {np.round(step_size, 4)}")

    chunks: dict[str, list] = {k: [] for k in ("z",) + fields}
    n_done, t_samp, samp_steps = 0, 0.0, 0
    while n_done < cfg.num_samples:
        if cfg.max_sampling_seconds and t_samp >= cfg.max_sampling_seconds:
            meta["stopped_by_cap"] = {"sampling_seconds": t_samp, "draws": n_done}
            log(f"[{cfg.name}] sampling cap {cfg.max_sampling_seconds:.0f} s reached after {n_done} draws")
            break
        mcmc.post_warmup_state = mcmc.last_state
        t1 = time.time()
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=fields)
        t_samp += time.time() - t1
        z = np.asarray(mcmc.get_samples(group_by_chain=True))
        ex = mcmc.get_extra_fields(group_by_chain=True)
        if cfg.num_chains == 1 and z.ndim == 2:
            z = z[None]
        chunks["z"].append(z)
        for k in fields:
            a = np.asarray(ex[k])
            chunks[k].append(a if a.ndim == (3 if k == "trajectory_length" else 2) else a[None])
        n_done += z.shape[1]
        samp_steps += int(np.sum(chunks["num_steps"][-1]))
        _save(out, chunks, post, meta, step_size, imm, t_samp, samp_steps, n_done)
        log(
            f"[{cfg.name}] {n_done}/{cfg.num_samples} samples, {t_samp:.0f} s, "
            f"div {int(np.sum([np.sum(a) for a in chunks['diverging']]))}, "
            f"mean steps {np.mean(chunks['num_steps'][-1]):.1f}"
        )
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out


def _save(out, chunks, post, meta, step_size, imm, t_samp, samp_steps, n_done):
    z = np.concatenate(chunks["z"], axis=1)
    _C, _N, _D = z.shape
    x = np.asarray(post.transform.to_constrained(z))
    pe = np.concatenate(chunks["potential_energy"], axis=1)
    logjac = np.asarray(post.transform.log_jacobian(jnp.asarray(z)))
    logL = -pe - logjac + post.transform.log_volume  # potential = -(logL + logJ - logV)
    ns = np.concatenate(chunks["num_steps"], axis=1)
    np.savez(
        out / "samples.tmp.npz",
        x=x,
        z=z,
        potential_energy=pe,
        logL=logL,
        diverging=np.concatenate(chunks["diverging"], axis=1),
        num_steps=ns,
        tree_depth=np.floor(np.log2(np.maximum(ns, 1))).astype(np.int32) + 1,
        accept_prob=np.concatenate(chunks["accept_prob"], axis=1),
        step_size=step_size,
        inverse_mass_matrix=imm,
        **({"jump_accept": np.concatenate(chunks["trajectory_length"], axis=1)} if "trajectory_length" in chunks else {}),
    )
    os.replace(out / "samples.tmp.npz", out / "samples.npz")
    meta.update(sampling_seconds=t_samp, sampling_grad_evals=samp_steps, n_samples_per_chain=n_done)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
