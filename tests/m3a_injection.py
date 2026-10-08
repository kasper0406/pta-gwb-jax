"""Gate G8: signal / null injections on the real TOA sampling of multi-leg pulsars (staggered
spans), recovered with option A (one leg per pulsar), B (per-leg timing) and C (shared timing).

Simulation (per realisation): white noise (fixed EFAC/EQUAD per namespaced system, ECORR per
system) + intrinsic red noise (fixed, on each pulsar's span) + an uncorrelated common process
(power law, gamma = 13/3, amplitude A_true or ~0) on the array grid; no timing-model term (every
likelihood is exactly invariant to it by construction). Each configuration marginalises its own
timing model on the same TOAs.

Statistic: the score s = d lnL / d(A^2) at the true A^2 (for the null: at A = 1e-18, where the
common term is negligible). For a correct likelihood E[s] = 0 (unbiased estimating equation) and
Var[s] = E[-d^2 lnL / d(A^2)^2] = I (information equality). Pass (fixed in advance):
|mean(s)| / (sd(s) / sqrt(N)) < 3.5 and Var(s) / mean(-H) within 1 +- 4.5 sqrt(2 / N) for every
configuration. Reported: the Fisher information I_C / I_B / I_A for the amplitude.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ptagwb.basis import powerlaw
from ptagwb.combined import GeneralPTALikelihood, PulsarGPModel, precompute_general
from ptagwb.gp import FourierBlock
from ptagwb.noise import EcorrTerm, build_general_white_noise

GAMMA = 13.0 / 3.0


def match(mp_ref, mp):
    """Indices into mp of mp_ref's TOAs, matched by (pta, name); TOAs absent in mp -> dropped."""
    key = {(a, b): i for i, (a, b) in enumerate(zip(np.asarray(mp.flags["pta"]).astype(str),
                                                     np.asarray(mp.flags["name"]).astype(str), strict=True))}
    ref = list(zip(np.asarray(mp_ref.flags["pta"]).astype(str), np.asarray(mp_ref.flags["name"]).astype(str), strict=True))
    keep = np.array([k in key for k in ref])
    return keep, np.array([key[k] for k, kp in zip(ref, keep, strict=True) if kp])


class _View:
    """A pulsar-like view (subset of TOAs, given timing matrix)."""

    def __init__(self, base, rows, M, name=None):
        self.name = name or base.name
        self.toas, self.freqs = base.toas[rows], base.freqs[rows]
        self.toaerrs, self.backend_flags = base.toaerrs[rows], base.backend_flags[rows]
        self.residuals = np.zeros(len(rows))
        self.Mmat = M[rows] if M.shape[0] != len(rows) else M
        self.Mmat = self.Mmat[:, np.linalg.norm(self.Mmat, axis=0) > 0]
        self.pos = base.pos
        self.pos_enterprise = None


def build_configs(mpC_list, mpB_list, seed=0, n_rn=20, n_c=10):
    """Per pulsar: views for C, B and A (A = the leg with the most TOAs, its own timing columns)
    on the common TOA set; white noise and GP models shared by all configurations."""
    rng = np.random.default_rng(seed)
    out = {"A": [], "B": [], "C": []}
    allt = np.concatenate([m.toas for m in mpC_list])
    Tarr = float(allt.max() - allt.min())
    common = FourierBlock("gw", n_c, Tarr)
    truth = []
    for mpC, mpB in zip(mpC_list, mpB_list, strict=True):
        keep, ib = match(mpC, mpB)
        rows = np.flatnonzero(keep)
        vC = _View(mpC, rows, mpC.Mmat)
        vB = _View(mpC, rows, mpB.Mmat[ib])
        ptas = np.asarray(mpC.flags["pta"]).astype(str)[rows]
        best = max(set(ptas.tolist()), key=lambda p: np.sum(ptas == p))
        legrows = rows[ptas == best]
        legcols = np.linalg.norm(mpB.Mmat[ib][ptas == best], axis=0) > 0
        vA = _View(mpC, legrows, mpB.Mmat[ib][ptas == best][:, legcols], name=mpC.name)
        systems = sorted(set(vC.backend_flags.tolist()))
        efeq = {s: (1.0 + 0.2 * rng.random(), -7.0 - rng.random()) for s in systems}
        ec = {s: -7.0 - rng.random() for s in systems}
        span = float(vC.toas.max() - vC.toas.min())
        rn = FourierBlock("red_noise", n_rn, span)
        for cfg, v in (("C", vC), ("B", vB), ("A", vA)):
            sy = sorted(set(v.backend_flags.tolist()))
            wn = build_general_white_noise(v.toas, v.toaerrs, v.backend_flags, {s: efeq[s] for s in sy},
                                           [EcorrTerm("ecorr", {s: v.backend_flags == s for s in sy}, {s: ec[s] for s in sy})])
            model = PulsarGPModel(sampled={"rn": rn}, common=common)
            out[cfg].append((v, wn, model))
        truth.append({"rn_log10_A": -14.3, "rn_gamma": 3.0, "rows_A": np.searchsorted(rows, legrows)})
    return out, truth, Tarr, common, rn


def simulate(cfg_list, truth, common, log10_A, R, rng):
    """R realisations of the C-configuration TOAs (white + IRN + CURN); returns list (per pulsar)
    of (n, R) arrays."""
    sims = []
    fc, dfc = common.frequencies()
    phic = powerlaw(fc, dfc, log10_A, GAMMA) if log10_A is not None else np.zeros(len(fc))
    for (v, wn, model), tr in zip(cfg_list, truth, strict=True):
        rn = model.sampled["rn"]
        f, df = rn.frequencies()
        phir = powerlaw(f, df, tr["rn_log10_A"], tr["rn_gamma"])
        Fr = rn.basis(v.toas, v.freqs)
        Fc = common.basis(v.toas, v.freqs)
        y = wn.colour(rng.standard_normal((len(v.toas), R)))
        y += Fr @ (np.sqrt(phir)[:, None] * rng.standard_normal((len(phir), R)))
        y += Fc @ (np.sqrt(phic)[:, None] * rng.standard_normal((len(phic), R)))
        sims.append(y)
    return sims


def score_stats(cfg_list, truth, sims, log10_A_eval, chunk=200):
    """Score s = dlnL/d(A^2) at log10_A_eval for every realisation, and mean(-d2lnL/d(A^2)^2)."""
    terms = [precompute_general(v, wn, m, projector=True) for v, wn, m in cfg_list]
    like = GeneralPTALikelihood(terms, orf="curn")
    P = len(terms)
    base = {"rn_log10_A": jnp.full(P, truth[0]["rn_log10_A"]), "rn_gamma": jnp.full(P, truth[0]["rn_gamma"]),
            "gamma": jnp.asarray(GAMMA)}
    A2 = 10.0 ** (2 * log10_A_eval)

    def f(a2, c, sp):
        return like._logL_cs(dict(base, log10_A=0.5 * jnp.log10(a2)), c, sp)

    g = jax.jit(jax.vmap(jax.grad(f), in_axes=(None, 0, 0)))
    h = jax.jit(jax.vmap(jax.hessian(f), in_axes=(None, 0, 0)))
    R = sims[0].shape[1]
    s, H = [], []
    for lo in range(0, R, chunk):
        hi = min(R, lo + chunk)
        c, sp = like.residual_terms([y[:, lo:hi] for y in sims])
        c = np.moveaxis(c, -1, 0)  # (R, P, K)
        sp = np.moveaxis(sp, -1, 0)
        s.append(np.asarray(g(jnp.asarray(A2), jnp.asarray(c), jnp.asarray(sp))))
        H.append(np.asarray(h(jnp.asarray(A2), jnp.asarray(c), jnp.asarray(sp))))
    return np.concatenate(s), np.concatenate(H)


def run(mpC_list, mpB_list, R=1000, log10_A_true=-14.0, seed=1):
    cfgs, truth, Tarr, common, _rn = build_configs(mpC_list, mpB_list, seed=seed)
    out = {"R": R, "log10_A_true": log10_A_true, "gamma": GAMMA, "Tarr_yr": Tarr / 3.15576e7, "results": {}}
    rng = np.random.default_rng(seed + 100)
    for label, lA in (("signal", log10_A_true), ("null", None)):
        simsC = simulate(cfgs["C"], truth, common, lA, R, rng)
        for cfg in ("C", "B", "A"):
            sims = simsC if cfg != "A" else [y[tr["rows_A"]] for y, tr in zip(simsC, truth, strict=True)]
            leval = lA if lA is not None else -18.0
            s, H = score_stats(cfgs[cfg], truth, sims, leval)
            info = float(np.mean(-H))
            z = float(np.mean(s) / (np.std(s, ddof=1) / np.sqrt(len(s))))
            ratio = float(np.var(s, ddof=1) / info)
            tol = 4.5 * np.sqrt(2.0 / len(s))
            out["results"][f"{label}/{cfg}"] = {"z_mean_score": z, "var_over_info": ratio, "info_A2": info,
                                                 "ratio_tol": float(tol),
                                                 "pass": bool(abs(z) < 3.5 and abs(ratio - 1) < tol)}
    for label in ("signal", "null"):
        r = out["results"]
        if f"{label}/A" not in r:
            continue
        out[f"info_ratio_C_over_B_{label}"] = r[f"{label}/C"]["info_A2"] / r[f"{label}/B"]["info_A2"]
        out[f"info_ratio_C_over_A_{label}"] = r[f"{label}/C"]["info_A2"] / r[f"{label}/A"]["info_A2"]
    out["pass"] = all(v["pass"] for v in out["results"].values())
    return out
