"""Numerical accuracy across the full prior box, including its corners (IRN and common
log10_A -> -11 with gamma -> 7, where I + Phi^1/2 A Phi^1/2 reaches condition numbers ~1e20).

References: (i) the mathematical identity single-pulsar HD == CURN (and HD with Gamma = I ==
CURN for the whole PTA), scanned over all 67 pulsars and all corners; (ii) a 50-digit
``decimal`` evaluation of the joint (unreduced) coefficient-space likelihood from the same
stage-1 square-root contractions, with 50-digit central-difference gradients; (iii) for the
correlated HD likelihood, the same 50-digit joint reference on subsets of the worst-conditioned
pulsars, fed either our float64 stage-1 terms or long-double TOA-level contractions.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from decimal_reference import grad_fd_dec, loglike_dec

from ptagwb import orf as orfs
from ptagwb.basis import fourier_frequencies, powerlaw
from ptagwb.data import get_tspan
from ptagwb.likelihood import PTALikelihood, precompute

pytestmark = [pytest.mark.oracle, pytest.mark.slow]  # needs the NG15 data (no oracle packages)

CORNERS = list(itertools.product((-20.0, -11.0), (0.0, 7.0), (-18.0, -11.0), (0.0, 7.0)))
REVIEW_POINTS = [(-11.1, 6.9, -11.1, 6.9), (-11.0, 7.0, -11.0, 7.0)]


@pytest.fixture(scope="module")
def setup(ours, noisedict):
    T = get_tspan(ours)
    return ours, T, precompute(ours, noisedict, T)


def worst_conditioned(terms, T, k=3):
    """Pulsars with the largest max_k phi_k A_kk at the IRN corner (log10_A=-11, gamma=7)."""
    f, df = fourier_frequencies(30, T)
    phi = powerlaw(f, df, -11.0, 7.0)
    score = [float(np.max(phi * np.diag(t.A))) for t in terms]
    return [terms[i].name for i in np.argsort(score)[::-1][:k]]


def _p(P, la, g, lac, gc):
    return {"rn_log10_A": jnp.full(P, la), "rn_gamma": jnp.full(P, g), "log10_A": jnp.asarray(lac), "gamma": jnp.asarray(gc)}


RHO_LO, RHO_HI = -15.5, -1.0  # adopted free-spectrum prior (config.PRIORS["freespec_log10_rho"])


def _rho_profiles(nc):
    alt = np.where(np.arange(nc) % 2 == 0, RHO_LO, RHO_HI)
    return {"lo": np.full(nc, RHO_LO), "hi": np.full(nc, RHO_HI), "alt": alt, "alt2": alt[::-1].copy(),
            "review": np.full(nc, -1.1), "mid": np.linspace(-6.0, -9.0, nc)}


def _points(common, nc):
    """(IRN corners) x (common corners / rho bound profiles), plus the reviewers' points."""
    irn = [(-20.0, 0.0), (-20.0, 7.0), (-11.0, 0.0), (-11.0, 7.0), (-11.1, 6.9)]
    if common == "powerlaw":
        com = [(-18.0, 0.0), (-18.0, 7.0), (-11.0, 0.0), (-11.0, 7.0), (-11.1, 6.9)]
        return [(a, g, {"log10_A": jnp.asarray(la), "gamma": jnp.asarray(gc)}) for a, g in irn for la, gc in com]
    return [(a, g, {"log10_rho": jnp.asarray(r)}) for a, g in irn for r in _rho_profiles(nc).values()]


@pytest.mark.parametrize("common,nc", [("powerlaw", 30), ("freespec", 14), ("freespec", 30)])
def test_scan_identity_orf_equals_curn_domain(setup, common, nc):
    """As below, for the power law with 30 common modes and the free spectrum with 14 and 30
    modes over the adopted rho prior bounds [-15.5, -1] (all-low, all-high, alternating, the
    reviewer's interior point log10_rho = -1.1, a mid profile) x IRN corners."""
    _, T, terms = setup
    P = len(terms)
    kw = {"common": common, "n_common": nc}
    curn = PTALikelihood(terms, T, orf="curn", **kw)
    others = [PTALikelihood(terms, T, orf=np.eye(P), **kw), PTALikelihood(terms, T, orf=np.eye(P), method="B", **kw)]
    worst = (0.0, None)
    for a, g, com in _points(common, nc):
        p = {"rn_log10_A": jnp.full(P, a), "rn_gamma": jnp.full(P, g), **com}
        v0, g0 = curn.value_and_grad(p)
        for L in others:
            v1, g1 = L.value_and_grad(p)
            assert abs(float(v1) - float(v0)) <= 1e-6, (a, g, L.method, float(v1) - float(v0))
            for k in g0:
                x, y = np.atleast_1d(np.asarray(g0[k])), np.atleast_1d(np.asarray(g1[k]))
                err = np.abs(x - y) / np.maximum(1.0, np.abs(x))
                i = int(np.argmax(err))
                worst = max(worst, (float(err[i]), (a, g, L.method, k, i)), key=lambda t: t[0])
                assert err[i] <= 1e-7, (a, g, L.method, k, i, x[i], y[i])
    print(f"{common}/{nc}: worst relative gradient discrepancy {worst}")


def test_scan_identity_orf_equals_curn_all_corners(setup):  # power law, 14 common modes
    """Gamma = I through the reduced HD path vs separable CURN, all 67 pulsars, 16 corners plus the
    reviewer's points: values and every gradient component (per-pulsar IRN gradients localise
    any failure to a pulsar)."""
    _, T, terms = setup
    P = len(terms)
    curn = PTALikelihood(terms, T, orf="curn")
    eye_s = PTALikelihood(terms, T, orf=np.eye(P))
    eye_b = PTALikelihood(terms, T, orf=np.eye(P), method="B")
    worst = (0.0, None)
    for pt in CORNERS + REVIEW_POINTS:
        p = _p(P, *pt)
        v0, g0 = curn.value_and_grad(p)
        for L in (eye_s, eye_b):
            v1, g1 = L.value_and_grad(p)
            assert abs(float(v1) - float(v0)) <= 1e-6, (pt, L.method, float(v1) - float(v0))
            for k in g0:
                a, b = np.atleast_1d(np.asarray(g0[k])), np.atleast_1d(np.asarray(g1[k]))
                err = np.abs(a - b) / np.maximum(1.0, np.abs(a))
                i = int(np.argmax(err))
                if err[i] > worst[0]:
                    worst = (float(err[i]), (pt, L.method, k, terms[i].name if a.size == P else None))
                assert err[i] <= 1e-7, (pt, L.method, k, i, a[i], b[i])
    print("worst relative gradient discrepancy:", worst)


@pytest.mark.parametrize("which", ["review", "worst"])
def test_single_pulsar_vs_decimal(setup, which):
    _, T, terms = setup
    tmap = {t.name: t for t in terms}
    names = ["J2043+1711", "B1937+21"] if which == "review" else worst_conditioned(terms, T)
    pts = REVIEW_POINTS + [(-11.0, 7.0, -18.0, 0.0), (-20.0, 0.0, -11.0, 7.0), (-13.5, 3.0, -14.6, 13 / 3)]
    for name in names:
        t = tmap[name]
        Ls = [PTALikelihood([t], T, orf="curn"), PTALikelihood([t], T, orf=np.eye(1))]
        c = -0.5 * t.const()
        for pt in pts:
            prm = {"rn_log10_A": [pt[0]], "rn_gamma": [pt[1]], "log10_A": pt[2], "gamma": pt[3]}
            ref = float(loglike_dec([t], T, [pt[0]], [pt[1]], pt[2], pt[3]))
            gref = grad_fd_dec([t], T, prm, [("rn_log10_A", 0), ("rn_gamma", 0), ("log10_A", None), ("gamma", None)])
            for L in Ls:
                v, g = L.value_and_grad(_p(1, *pt))
                assert abs(float(v) - c - ref) <= 1e-8, (name, pt, L.orf_name, float(v) - c - ref)
                for (k, i), gr in gref.items():
                    an = float(g[k][0] if i is not None else g[k])
                    assert abs(an - gr) <= 1e-7 * max(1.0, abs(gr)), (name, pt, L.orf_name, k, an, gr)


def _hd_subset_check(terms, T, pts, tol_v, tol_g, ref_terms=None):
    ref_terms = ref_terms or terms
    P = len(terms)
    pos = np.stack([t.pos for t in terms])
    G = orfs.hd(pos)
    L = PTALikelihood(terms, T, orf="hd")
    LB = PTALikelihood(terms, T, orf="hd", method="B")
    c = -0.5 * sum(t.const() for t in terms)
    for pt in pts:
        prm = {"rn_log10_A": [pt[0]] * P, "rn_gamma": [pt[1]] * P, "log10_A": pt[2], "gamma": pt[3]}
        ref = float(loglike_dec(ref_terms, T, prm["rn_log10_A"], prm["rn_gamma"], pt[2], pt[3], Gamma=G))
        which = [("rn_gamma", 0), ("rn_log10_A", P - 1), ("log10_A", None), ("gamma", None)]
        gref = grad_fd_dec(ref_terms, T, prm, which, Gamma=G) if tol_g is not None else {}
        for Lx in (L, LB):
            v, g = Lx.value_and_grad(_p(P, *pt))
            print(f"HD subset {[t.name for t in terms]} {pt} {Lx.method}: value - ref = {float(v) - c - ref:+.2e}")
            assert abs(float(v) - c - ref) <= tol_v, (pt, Lx.method, float(v) - c - ref)
            for (k, i), gr in gref.items():
                an = float(g[k][i] if i is not None else g[k])
                assert abs(an - gr) <= tol_g * max(1.0, abs(gr)), (pt, Lx.method, k, i, an, gr)


def test_hd_correlated_vs_decimal(setup):
    """Correlated HD (3 worst-conditioned pulsars, plus the two review pulsars) vs the
    50-digit joint reference, at the common-prior corner and a posterior-like point."""
    _, T, terms = setup
    tmap = {t.name: t for t in terms}
    sub = [tmap[n] for n in worst_conditioned(terms, T)[:2] + ["J2043+1711", "B1937+21"]]
    pts = REVIEW_POINTS + [(-14.0, 4.0, -11.0, 7.0), (-13.5, 3.0, -14.6, 13 / 3)]
    _hd_subset_check(sub, T, pts, tol_v=1e-8, tol_g=1e-7)


def test_hd_correlated_vs_long_double_toa_level(setup, noisedict):
    """Full chain in extended precision: long-double TOA-level stage 1 (whitening, Gram-Schmidt,
    contractions) + 50-digit joint HD stage 2, vs our float64 pipeline."""
    from extended_precision import pulsar_contractions_ld

    psrs, T, terms = setup
    pmap = {p.name: p for p in psrs}
    names = ["J2043+1711", "B1937+21", "J1713+0747"]
    sub = [t for t in terms if t.name in names]
    ld = []
    for t in sub:
        c = pulsar_contractions_ld(pmap[t.name], noisedict, T)
        ld.append(SimpleNamespace(A=c["A"], b=c["b"], s=c["s"], RA=None, const=lambda c=c: c["const"]))
    pts = [(-13.5, 3.0, -14.6, 13 / 3), (-12.0, 5.0, -12.0, 5.0), (-11.0, 7.0, -11.0, 7.0), (-20.0, 0.0, -11.0, 7.0)]
    # tolerance covers float64 vs long-double *stage-1* rounding (TOA-level whitening/QR of up to
    # 6e4 TOAs): measured 1e-10 at posterior-like points, 8e-8 at the prior corners.
    _hd_subset_check(sub, T, pts, tol_v=3e-7, tol_g=None, ref_terms=ld)


@pytest.mark.parametrize("orf", ["hd", "dipole", "monopole"])
@pytest.mark.parametrize("common", ["powerlaw", "freespec"])
def test_correlated_all_orfs_vs_decimal(setup, orf, common):
    """J2043+1711 + B1937+21 (the reviewers' pulsars) with a genuinely correlated ORF, 30 common
    modes, vs the 50-digit joint reference: value and IRN / common gradients."""
    _, T, terms = setup
    tmap = {t.name: t for t in terms}
    sub = [tmap["J2043+1711"], tmap["B1937+21"]]
    G = orfs.ORFS[orf](np.stack([t.pos for t in sub]))
    c = -0.5 * sum(t.const() for t in sub)
    if common == "freespec":
        cases = [{"log10_rho": np.full(30, -1.1)}, {"log10_rho": _rho_profiles(30)["alt"]}]
        which = [("log10_rho", 0), ("log10_rho", 2), ("log10_rho", 29)]
    else:
        cases = [{"log10_A": -11.1, "gamma": 6.9}, {"log10_A": -11.0, "gamma": 7.0}]
        which = [("log10_A", None), ("gamma", None)]
    which = [("rn_log10_A", 0), ("rn_gamma", 0), ("rn_log10_A", 1)] + which
    Ls = [PTALikelihood(sub, T, orf=G, common=common, n_common=30, method=m) for m in ("sigma", "B")]
    for com in cases:
        prm = {"rn_log10_A": [-11.1, -11.0], "rn_gamma": [6.9, 7.0], **com}
        ref = float(loglike_dec(sub, T, prm["rn_log10_A"], prm["rn_gamma"], com.get("log10_A"), com.get("gamma"),
                                Gamma=G, n_common=30, log10_rho=com.get("log10_rho")))
        gref = grad_fd_dec(sub, T, prm, which, Gamma=G, n_common=30)
        for L in Ls:
            v, g = L.value_and_grad({k: jnp.asarray(v_) for k, v_ in prm.items()})
            assert abs(float(v) - c - ref) <= 1e-8, (orf, common, L.method, float(v) - c - ref)
            for (k, i), gr in gref.items():
                an = float(np.atleast_1d(np.asarray(g[k]))[0 if i is None else i])
                assert abs(an - gr) <= 1e-7 * max(1.0, abs(gr)), (orf, common, L.method, k, i, an, gr)
