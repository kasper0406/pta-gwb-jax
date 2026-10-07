"""Performance variants (``ptagwb.perf_likelihood``, docs/PERF.md) must reproduce the production
likelihood: value (without the parameter-independent constant) to <= 1e-9 absolute and every
gradient component to <= 1e-8 relative (to max(|g|, 1)), single and vmapped, compiled -- or, where
production's own reproducibility floor is larger than that (``noise_floor``: production vs exact
identities of itself -- pulsars permuted, a different split_fraction, vmapped vs single kernels;
monopole / dipole ORFs, regularised with diag_eps = 1e-5, reach ~1e-5 in the value on the GPU),
within 10x that floor.

Matrix: common spectrum {power law, free spectrum} x {14, 30} common modes x ORF {curn, hd,
dipole, monopole}; points: interior draws, IRN/common prior corners, free-spectrum bound profiles
(all-low, all-high, alternating, reviewer's -1.1); on the default backend (GPU here) and, via a
subprocess with JAX_PLATFORMS=cpu, on XLA:CPU. The CPU run is a regression test for the
XLA:CPU YNNPACK-fusion miscompilation of the instantiated-zero backward terms (CURN free
spectrum / 30-mode CURN gradients ~1e281 before ``perf_likelihood._reduce_bwd_sz``).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from synthetic import make_pta, tspan

from ptagwb import orf as orfs
from ptagwb.likelihood import PTALikelihood, _tri_inv_lower, precompute
from ptagwb.perf_likelihood import FastPTALikelihood, _hh_bottom, _tri_inv_levels

NM = 30
COMMONS = [("powerlaw", 14), ("powerlaw", 30), ("freespec", 14), ("freespec", 30)]
ORFS = ["curn", "hd", "dipole", "monopole"]
RHO_LO, RHO_HI = -15.5, -1.0


def _strip(like):
    like.const_total = 0.0
    like.value_and_grad = jax.jit(jax.value_and_grad(like._logL))
    return like


def _flat(g):
    return np.concatenate([np.atleast_1d(np.asarray(g[k])) for k in sorted(g)])


def points(P, common, nc, seed=0, n_interior=3):
    """Interior draws + IRN corners x common corners / free-spectrum bound profiles."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_interior):
        p = {"rn_log10_A": rng.uniform(-16, -12.5, P), "rn_gamma": rng.uniform(1, 6, P)}
        if common == "powerlaw":
            p.update(log10_A=np.asarray(rng.uniform(-15.5, -13.5)), gamma=np.asarray(rng.uniform(2, 6)))
        else:
            p["log10_rho"] = rng.uniform(-9, -5, nc)
        pts.append(p)
    irn = [(-20.0, 0.0), (-20.0, 7.0), (-11.0, 0.0), (-11.0, 7.0), (-13.5, 3.0)]
    if common == "powerlaw":
        com = [{"log10_A": np.asarray(a), "gamma": np.asarray(g)} for a, g in itertools.product((-18.0, -11.0), (0.0, 7.0))]
        com.append({"log10_A": np.asarray(-11.1), "gamma": np.asarray(6.9)})
    else:
        alt = np.where(np.arange(nc) % 2 == 0, RHO_LO, RHO_HI)
        com = [{"log10_rho": v} for v in (np.full(nc, RHO_LO), np.full(nc, RHO_HI), alt, alt[::-1].copy(),
                                          np.full(nc, -1.1), np.full(nc, -7.0))]
    for (a, g), cm in itertools.product(irn, com):
        pts.append({"rn_log10_A": np.full(P, a), "rn_gamma": np.full(P, g), **cm})
    return pts


def compare(prod, fast, pts):
    """Worst (value abs diff, gradient rel diff) over pts, single and vmapped; asserts finiteness."""
    stacked = {k: jnp.stack([jnp.asarray(p[k], dtype=jnp.float64) for p in pts]) for k in pts[0]}
    vb, gb = jax.jit(jax.vmap(jax.value_and_grad(fast._logL)))(stacked)
    wv = wg = 0.0
    for i, p in enumerate(pts):
        v0, g0 = prod.value_and_grad(p)
        a = _flat(g0)
        for v1, g1 in (fast.value_and_grad(p), (vb[i], {k: gb[k][i] for k in gb})):
            b = _flat(g1)
            assert np.isfinite(float(v1)) and np.all(np.isfinite(b)), (i, float(v1))
            wv = max(wv, abs(float(v1) - float(v0)))
            wg = max(wg, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))))
    return wv, wg


def _orf(name, terms):
    return name if name == "curn" else orfs.ORFS[name](np.stack([t.pos for t in terms]))


def noise_floor(terms, T, kw_fn, pts, perm_seed=1):
    """Production's own reproducibility for this configuration: the worst |logL| and gradient
    differences between the production likelihood and three exact identities of it that change
    the rounding -- (a) the pulsars permuted, (b) split_fraction 0.45 / 0.55 instead of 0.5
    (Gamma = lam0 I + Gamma' is an exact split), (c) the vmapped evaluation (batched cuSOLVER
    kernels) instead of the single one. Worst over pts."""
    perm = np.random.default_rng(perm_seed).permutation(len(terms))
    inv = np.argsort(perm)
    tp = [terms[i] for i in perm]
    A = _strip(PTALikelihood(terms, T, **kw_fn(terms)))
    others = [(_strip(PTALikelihood(tp, T, **kw_fn(tp))), True)]
    if not isinstance(kw_fn(terms)["orf"], str):  # correlated ORF: split_fraction is used
        others += [(_strip(PTALikelihood(terms, T, split_fraction=f, **kw_fn(terms))), False) for f in (0.45, 0.55)]
    stacked = {k: jnp.stack([jnp.asarray(p[k], dtype=jnp.float64) for p in pts]) for k in pts[0]}
    vb, gb = jax.jit(jax.vmap(jax.value_and_grad(A._logL)))(stacked)
    fv = fg = 0.0
    for i, p in enumerate(pts):
        v0, g0 = A.value_and_grad(p)
        a = _flat(g0)
        cands = [(vb[i], {k: gb[k][i] for k in gb})]
        for L, permuted in others:
            if permuted:
                pp = {k: (np.asarray(v)[perm] if k.startswith("rn_") else v) for k, v in p.items()}
                v1, g1 = L.value_and_grad(pp)
                g1 = {k: (np.asarray(v)[inv] if k.startswith("rn_") else v) for k, v in g1.items()}
            else:
                v1, g1 = L.value_and_grad(p)
            cands.append((v1, g1))
        for v1, g1 in cands:
            b = _flat(g1)
            fv = max(fv, abs(float(v1) - float(v0)))
            fg = max(fg, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))))
    return fv, fg


def run_matrix(terms, T, variants=(("hh", "levels"),), combos=None):
    """{key: (worst dv, worst dg, floor dv, floor dg)} over the COMMONS x ORFS matrix (or ``combos``)."""
    out = {}
    for (common, nc), orf_name in combos or itertools.product(COMMONS, ORFS):
        def kw_fn(tt, common=common, nc=nc, orf_name=orf_name):
            return {"n_modes": NM, "n_common": nc, "orf": _orf(orf_name, tt), "common": common}

        prod = _strip(PTALikelihood(terms, T, **kw_fn(terms)))
        pts = points(len(terms), common, nc)
        floor = noise_floor(terms, T, kw_fn, pts)
        for red, inv in variants:
            fast = _strip(FastPTALikelihood(terms, T, reduce=red, tri_inv=inv, **kw_fn(terms)))
            out[f"{common}{nc}/{orf_name}/{red}+{inv}"] = (*compare(prod, fast, pts), *floor)
    return out


@pytest.fixture(scope="module")
def pta():
    psrs, nd = make_pta(5, seed=0, n_epochs=70, signal=3e-7)
    T = tspan(psrs)
    return T, precompute(psrs, nd, T, n_modes=NM)


def test_hh_bottom_equals_qr():
    rng = np.random.default_rng(1)
    n, m = 12, 25
    top = np.triu(rng.standard_normal((n, n))) * np.logspace(-6, 6, n)[None, :]
    top = np.concatenate([top, rng.standard_normal((n, m - n))], axis=1)
    R = np.asarray(_hh_bottom(jnp.asarray(top)))
    R0 = np.linalg.qr(np.concatenate([top, np.concatenate([np.eye(n), np.zeros((n, m - n))], 1)]), mode="r")[:n]
    s = np.sign(np.diag(R)) * np.sign(np.diag(R0))
    np.testing.assert_allclose(R * s[:, None], R0, rtol=1e-10, atol=1e-12 * np.abs(R0).max())


@pytest.mark.parametrize("n", [37, 64, 130, 300])
def test_tri_inv_levels(n):
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n))
    L = jnp.linalg.cholesky(jnp.asarray(A @ A.T / n + np.eye(n)))
    for depth in (0, 3):
        X = _tri_inv_levels(L, leaf=16, depth=depth)
        np.testing.assert_allclose(np.asarray(X), np.asarray(_tri_inv_lower(L)), atol=1e-12 * float(jnp.abs(X).max()))


def test_method_b_rejected(pta):
    T, terms = pta
    with pytest.raises(ValueError, match="method"):
        FastPTALikelihood(terms, T, n_modes=NM, n_common=14, orf="hd", method="B")


def _assert_matrix(res, tol_v=1e-9, tol_g=1e-8, floor_factor=10.0):
    """Criterion: dv <= 1e-9 and dg <= 1e-8, or, where production's own reproducibility floor
    (``noise_floor``) already exceeds that (ill-conditioned monopole/dipole ORFs), within 10x it."""
    bad = {k: v for k, v in res.items()
           if not (v[0] <= max(tol_v, floor_factor * v[2]) and v[1] <= max(tol_g, floor_factor * v[3]))}
    assert not bad, bad


@pytest.mark.parametrize("common,nc", COMMONS)
def test_synthetic_matrix_default_backend(pta, common, nc):
    T, terms = pta
    _assert_matrix(run_matrix(terms, T, combos=[((common, nc), o) for o in ORFS]))


def test_synthetic_other_variants(pta):
    T, terms = pta
    combos = [(("powerlaw", 14), "hd"), (("freespec", 30), "curn"), (("freespec", 30), "hd")]
    _assert_matrix(run_matrix(terms, T, variants=(("hh", "recursive"), ("prod", "levels")), combos=combos))


def _main_cpu_subprocess():  # executed in the subprocess (JAX_PLATFORMS=cpu)
    psrs, nd = make_pta(5, seed=0, n_epochs=70, signal=3e-7)
    T = tspan(psrs)
    terms = precompute(psrs, nd, T, n_modes=NM)
    res = run_matrix(terms, T)
    print("RESULT " + json.dumps({"backend": jax.default_backend(), "res": res}))


def test_synthetic_matrix_cpu():
    """The full matrix on XLA:CPU in a fresh process (the default backend here is the GPU)."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    code = f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); import test_perf_likelihood as t; t._main_cpu_subprocess()"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=1800, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT "))
    out = json.loads(line[len("RESULT "):])
    assert out["backend"] == "cpu"
    _assert_matrix({k: tuple(v) for k, v in out["res"].items()})


@pytest.mark.oracle
@pytest.mark.slow
def test_ng15_matrix(ours, noisedict):
    """All 67 NG15 pulsars, the full common x ORF matrix (default backend). The CPU counterpart
    and per-configuration numbers: bench/check_exact.py (results in bench/results/exact_*.json)."""
    from ptagwb.data import get_tspan

    T = get_tspan(ours)
    terms = precompute(ours, noisedict, T, position="enterprise")
    _assert_matrix(run_matrix(terms, T))
