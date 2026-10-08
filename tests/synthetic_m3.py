"""Small synthetic multi-PTA arrays for the M3a likelihood tests (no data files needed).

``make_array``: pulsars with one to three legs from different "PTAs" with staggered spans,
namespaced systems (equal raw backend labels in different PTAs), TN and T2 EQUAD legs,
overlapping ECORR terms (per-system + global, as PPTA), a timing model with shared columns
(offset per leg, F0/F1-like polynomials, an annual sinusoid) and per-leg detector columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptagwb.combined import PulsarGPModel
from ptagwb.data import Pulsar
from ptagwb.gp import FourierBlock
from ptagwb.noise import EcorrTerm, build_general_white_noise, namespace

DAY = 86400.0
YR = 365.25 * DAY
T0 = 53000.0 * DAY


@dataclass
class LegSpec:
    pta: str
    start_yr: float
    span_yr: float
    n_epochs: int
    systems: tuple
    convention: str


def make_leg(rng, spec: LegSpec):
    epochs = np.sort(T0 + (spec.start_yr + rng.uniform(0, spec.span_yr, spec.n_epochs)) * YR)
    toas, sys_, freqs = [], [], []
    for e in epochs:
        s = spec.systems[rng.integers(len(spec.systems))]
        nch = int(rng.integers(1, 5))
        toas.extend(e + 0.1 * np.arange(nch) + rng.uniform(0, 0.05, nch))
        sys_.extend([namespace(spec.pta, s)] * nch)
        freqs.extend(rng.uniform(700, 3000, nch))
    return np.array(toas), np.array(sys_), np.array(freqs)


def make_pulsar(rng, name, legs: list[LegSpec], shared=True, signal=0.0):
    parts = [make_leg(rng, s) for s in legs]
    toas = np.concatenate([p[0] for p in parts])
    systems = np.concatenate([p[1] for p in parts])
    freqs = np.concatenate([p[2] for p in parts])
    leg = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
    isort = np.argsort(toas, kind="mergesort")
    toas, systems, freqs, leg = toas[isort], systems[isort], freqs[isort], leg[isort]
    n = len(toas)
    err = rng.uniform(0.3, 2.0, n) * 1e-6
    tt = (toas - toas.mean()) / (10 * YR)
    shared_cols = [tt, tt**2, np.sin(2 * np.pi * toas / YR), np.cos(2 * np.pi * toas / YR), (1400.0 / freqs) ** 2]
    cols = []
    for i in range(len(legs)):  # per-leg offset (phase JUMP between PTAs) and per-leg FD-like column
        cols.append((leg == i).astype(float))
        cols.append((leg == i) * np.log(freqs / 1400.0))
    if shared:
        cols += shared_cols
    else:
        for i in range(len(legs)):
            cols += [(leg == i) * c for c in shared_cols]
    M = np.column_stack(cols)
    res = err * rng.normal(size=n)
    if signal:
        res = res + signal * np.sin(2 * np.pi * (toas - T0) / (8 * YR))
    pos = rng.normal(size=3)
    pos /= np.linalg.norm(pos)
    psr = Pulsar(name=name, toas=toas, stoas=toas, residuals=res, toaerrs=err, freqs=freqs, freqs_topo=freqs,
                 backend_flags=systems.astype("U"), telescope=np.array(["x"] * n), Mmat=M,
                 fitpars=[f"c{i}" for i in range(M.shape[1])], pos=pos)
    psr.meta["leg"] = leg
    return psr


def white_noise(rng, psr, legs: list[LegSpec], overlapping=True):
    efeq, conv = {}, {}
    for s in legs:
        for raw in s.systems:
            lab = namespace(s.pta, raw)
            efeq[lab] = (rng.uniform(0.8, 1.5), rng.uniform(-7.5, -6.0))
            conv[lab] = s.convention
    present = set(psr.backend_flags.tolist())
    efeq = {k: v for k, v in efeq.items() if k in present}
    conv = {k: v for k, v in conv.items() if k in present}
    terms = [EcorrTerm("ecorr", {s: psr.backend_flags == s for s in efeq}, {s: rng.uniform(-7.0, -6.0) for s in efeq})]
    if overlapping:  # PPTA-like global ECORR over all systems of the first PTA, overlapping the above
        pta0 = legs[0].pta
        m = np.char.startswith(psr.backend_flags.astype(str), pta0 + ":")
        terms.append(EcorrTerm("ecorr_all", {pta0: m}, {pta0: rng.uniform(-7.0, -6.5)}))
    return build_general_white_noise(psr.toas, psr.toaerrs, psr.backend_flags, efeq, terms, convention=conv)


def make_array(seed=0, n_rn=5, n_dm=6, n_common=3, distinct=True, fixed=True, overlapping=True):
    """Two pulsars, three legs: A (PTA "X" 0-10 yr + PTA "Y" 7-13 yr, same raw backend label "BE"
    in both), B ("X" only, 2-9 yr). IRN on each pulsar's own span (``distinct``) or the array span,
    DM on its own grid for A only, common process on the array span, a fixed band-noise block on A."""
    rng = np.random.default_rng(seed)
    specA = [LegSpec("X", 0.0, 10.0, 18, ("BE", "B2"), "t2"), LegSpec("Y", 7.0, 6.0, 14, ("BE",), "tn")]
    specB = [LegSpec("X", 2.0, 7.0, 15, ("BE", "B3"), "t2")]
    A = make_pulsar(rng, "J0001+0001", specA)
    B = make_pulsar(rng, "J0002-0002", specB)
    psrs = [A, B]
    wns = [white_noise(rng, A, specA, overlapping), white_noise(rng, B, specB, overlapping)]
    Tarr = max(p.toas.max() for p in psrs) - min(p.toas.min() for p in psrs)
    common = FourierBlock("gw", n_common, Tarr)
    models = []
    for p in psrs:
        Trn = p.toas.max() - p.toas.min() if distinct else Tarr
        sampled = {"rn": FourierBlock("red_noise", n_rn, Trn)}
        if p is A:
            sampled["dm"] = FourierBlock("dm_gp", n_dm, 1.3 * Trn, chrom_idx=2.0)
        fx = []
        if fixed and p is A:
            blk = FourierBlock("band_low", 3, Trn, selection=("freq", 0.0, 1200.0))
            f, _df = blk.frequencies()
            fx.append((blk, 1e-13 * (f * YR) ** -2.0))
        models.append(PulsarGPModel(sampled=sampled, common=common, fixed=fx))
    return psrs, wns, models, Tarr


def random_params(rng, models, common="powerlaw", n_common=3):
    out = {}
    names = []
    for m in models:
        for nm in m.sampled:
            if nm not in names:
                names.append(nm)
    for nm in names:
        k = sum(nm in m.sampled for m in models)
        out[f"{nm}_log10_A"] = rng.uniform(-14.5, -12.8, k)
        out[f"{nm}_gamma"] = rng.uniform(1.5, 5.5, k)
    if common == "powerlaw":
        out["log10_A"] = np.asarray(rng.uniform(-14.5, -13.0))
        out["gamma"] = np.asarray(rng.uniform(2.0, 5.0))
    else:
        out["log10_rho"] = rng.uniform(-7.5, -6.3, n_common)
    return out
