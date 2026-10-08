"""Multi-leg pulsar container: options B (per-leg timing models) and C (shared timing columns)
through one code path (docs/M3_PLAN.md Sec. 2.2, 3.2; component L4).

A multi-leg pulsar stacks the TOAs of its legs (sorted by barycentric TOA), keeps every leg's
white noise in its own namespace (``<pta>:<system>``), and shares the GP basis automatically
(the Fourier/DM columns are evaluated on the stacked TOAs with one time origin). Only the timing
design matrix differs:

* **per_leg (B, MetaPulsar "composite"):** every leg keeps its own (canonical) timing model and
  residuals; M = blockdiag(M_p), every column named ``<param>:<pta>``.
* **shared (C, YA-v3 / MetaPulsar-0.9.3-shared):** every leg's par is rewritten from the
  *reference* leg before PINT computes residuals (``consistent_par``), then the design-matrix
  columns of the shared parameters are merged across legs (one column, stacked) and all other
  columns stay per leg. The rewrite reproduces MetaPulsar v0.9.3 ``ParameterManager``
  (``make_parfiles_consistent``): for the components astrometry, spindown, binary
  ("pulsar_system") and dispersion ("dispersion_constant"), remove every parameter of those
  components (from either model) from the target and insert the reference's lines (values and
  fit flags, epochs included); remove DMX everywhere; set DM = reference DM (free),
  DMEPOCH = reference DMEPOCH (55000 if absent; frozen), DM1 = DM2 = 0 (free) in *every* leg
  including the reference; EPHEM and CLOCK from the reference; TCB legs are converted to TDB
  first. Each leg keeps its detector parameters (phase offset, JUMPs, FD, solar wind ...).
  All legs are evaluated with the reference leg's evaluator profile.

The reference is the first PTA of ``REFERENCE_ORDER`` (NG15 > EPTA > PPTA > MPTA; InPTA never)
unless given explicitly (gate G6 swaps it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Pulsar
from .m3data import REFERENCE_ORDER
from .noise import namespace

SHARED_CATEGORIES = ("astrometry", "spindown", "pulsar_system", "dispersion_constant")
PROJ_RTOL = 1e-10  # declared rank threshold of complement_projector


def complement_projector(WM: np.ndarray, rtol: float = PROJ_RTOL):
    """x -> (I - P) x with P the orthogonal projector onto span(WM); rank-revealing (thin SVD of the
    column-normalised matrix, singular values > rtol * s_max kept). ``proj.rank``/``proj.ncols``."""
    WM = WM[:, np.linalg.norm(WM, axis=0) > 0]
    U, sv, _ = np.linalg.svd(WM / np.linalg.norm(WM, axis=0), full_matrices=False)
    U = U[:, sv > rtol * sv[0]]

    def proj(x):
        y = x - U @ (U.T @ x)
        return y - U @ (U.T @ y)

    proj.rank = int(U.shape[1])
    proj.ncols = int(WM.shape[1])
    return proj


# configurations (recorded in MultiLegPulsar.meta["config"])
CONFIG_B = "B/per-leg (MetaPulsar composite)"
CONFIG_YA = "C/YA-v3 (MetaPulsar-0.9.3 shared DM): the YA reproduction target"
CONFIG_LOCAL_DM = "C/local-DM (MetaPulsar-main exclude_from_shared=DM): NOT the YA target"
ADMISSIBILITY_RMS = 0.1  # max whitened rms of the linearisation residual vs the leg's own model


class InadmissibleBuildError(RuntimeError):
    """An option-C build whose residuals are not a linear re-parameterisation of each leg's own
    phase-connected model (e.g. phase wraps after the shared-DM copy)."""


def admissibility(mp_leg_psr, own_psr) -> dict:
    """Linearisation check of one C leg against the leg's own model (same CLOCK/EPHEM and clock
    profile, DMX columns free; built in ``_c_worker``), matched by tim
    record: e = P_perp[W M_own, W M_C] W (r_C - r_own) with W = 1/sigma (raw errors). For a linear,
    phase-connected re-parameterisation r_C - r_own lies in the union span and e ~ 0; wraps and
    nonlinearity leave e large. Admissible iff rms(e) <= ADMISSIBILITY_RMS."""
    rc = np.asarray(mp_leg_psr.meta["record_index"])
    ro = np.asarray(own_psr.meta["record_index"])
    common, ic, io = np.intersect1d(rc, ro, return_indices=True)
    sig = mp_leg_psr.toaerrs[ic]
    W = 1.0 / sig
    P = complement_projector(np.hstack([own_psr.Mmat[io] * W[:, None], mp_leg_psr.Mmat[ic] * W[:, None]]))
    e = P((mp_leg_psr.residuals[ic] - own_psr.residuals[io]) * W)
    rms = float(np.sqrt(np.mean(e**2)))
    return {"n_matched": len(common), "union_rank": P.rank, "union_ncols": P.ncols,
            "linearisation_rms_whitened": rms, "linearisation_rms_ns": float(np.sqrt(np.mean((e * sig) ** 2)) * 1e9),
            "admissible": bool(rms <= ADMISSIBILITY_RMS), "threshold": ADMISSIBILITY_RMS}
YA_V3_CONFIG = {
    "name": "YA-v3/MetaPulsar-0.9.3-shared",
    "metapulsar": "v0.9.3 (d2067ab, 2025-11-17)",
    "shared_categories": list(SHARED_CATEGORIES),
    "dm": "shared: reference DM (free), DMEPOCH from reference (frozen; 55000 if absent), DM1 = DM2 = 0 free in all legs",
    "dmx": "removed in all legs",
    "epochs": "PEPOCH/POSEPOCH/DMEPOCH/binary epochs from the reference",
    "fit_flags": "reference's for shared parameters; each leg's own for detector parameters",
    "clock_ephem": "reference leg's CLOCK and EPHEM",
    "units": "TDB (TCB legs converted by PINT)",
    "evaluator": "reference leg's evaluator profile for all legs",
    "reference_order": list(REFERENCE_ORDER),
}


def choose_reference(ptas: list[str]) -> str:
    for p in REFERENCE_ORDER:
        if p in ptas:
            return p
    raise ValueError(f"no admissible reference PTA among {ptas} (InPTA is never the reference)")


def _par_lines(text: str) -> list[tuple[str, str]]:
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append((s.split()[0], ln))
    return out


def component_params(model, categories) -> set[str]:
    names = set()
    for comp in model.components.values():
        if getattr(comp, "category", None) in categories:
            names.update(comp.params)
    return names


def consistent_par(ref_model, tgt_model, clock: str | None = None, ephem: str | None = None,
                   local_dm: bool = False) -> tuple[str, dict]:
    """MetaPulsar-v0.9.3 'consistent' rewrite of ``tgt_model`` from ``ref_model`` (both PINT models,
    TDB). Returns (par text, info). Works for tgt is ref as well (DMX removal, DM rules).
    ``clock``/``ephem``: override the reference's CLOCK/EPHEM (gate G6 holds them fixed while
    swapping the reference; v0.9.3 itself always takes the reference's)."""
    cats = tuple(c for c in SHARED_CATEGORIES if not (local_dm and c == "dispersion_constant"))
    drop = component_params(ref_model, cats) | component_params(tgt_model, cats) | {"BINARY"}
    drop |= component_params(tgt_model, ("dispersion_dmx",))
    drop |= {"EPHEM", "CLOCK", "CLK"} | (set() if local_dm else {"DM", "DMEPOCH", "DM1", "DM2"})
    ref_keep = component_params(ref_model, cats) | {"BINARY"}

    def canon(model, k):
        try:
            return model.match_param_aliases(k)
        except Exception:  # noqa: BLE001 - not a model parameter (header keyword)
            return k

    ref_lines = [(k, ln) for k, ln in _par_lines(ref_model.as_parfile())
                 if (k in ref_keep or canon(ref_model, k) in ref_keep)
                 and canon(ref_model, k) not in ("DM", "DMEPOCH", "DM1", "DM2")]
    tgt_lines = [(k, ln) for k, ln in _par_lines(tgt_model.as_parfile()) if k not in drop
                 and canon(tgt_model, k) not in drop and not re.fullmatch(r"DMX(R[12])?_\d+", k)]
    ref_all = dict(_par_lines(ref_model.as_parfile()))
    dm_str = ref_all["DM"].split()[1] if "DM" in ref_all else "0"
    dmepoch = ref_all["DMEPOCH"].split()[1] if "DMEPOCH" in ref_all else "55000"
    dm_lines = [] if local_dm else [f"DM {dm_str} 1", f"DMEPOCH {dmepoch}", "DM1 0.0 1", "DM2 0.0 1"]
    extra = dm_lines + [
        f"EPHEM {ephem or ref_model.EPHEM.value}",
        f"CLOCK {clock or ref_model.CLOCK.value}",
    ]
    shared_free = sorted(p for p in component_params(ref_model, cats)
                         if p in ref_model.free_params and p not in ("DM", "DM1", "DM2"))
    shared_free += [] if local_dm else ["DM", "DM1", "DM2"]
    text = "\n".join([ln for _, ln in tgt_lines] + [ln for _, ln in ref_lines] + extra) + "\n"
    info = {"shared_free": shared_free, "dropped_target_params": sorted(component_params(tgt_model, cats)),
            "dmx_removed": sum(1 for k, _ in _par_lines(tgt_model.as_parfile()) if re.fullmatch(r"DMX_\d+", k))}
    return text, info


@dataclass
class LegInfo:
    pta: str
    dataset: str
    label: str
    ntoa: int
    span_mjd: tuple
    fitpars: list
    meta: dict = field(default_factory=dict)


@dataclass
class MultiLegPulsar:
    """Stacked multi-leg pulsar; attribute-compatible with ``data.Pulsar`` for stage 1."""

    name: str
    timing: str
    reference: str | None
    legs: list
    toas: np.ndarray
    stoas: np.ndarray
    residuals: np.ndarray
    toaerrs: np.ndarray
    freqs: np.ndarray
    freqs_topo: np.ndarray
    backend_flags: np.ndarray
    telescope: np.ndarray
    Mmat: np.ndarray
    fitpars: list
    pos: np.ndarray
    leg: np.ndarray  # leg index per TOA
    flags: dict = field(default_factory=dict)
    pos_enterprise: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ntoa(self) -> int:
        return len(self.toas)

    def __repr__(self) -> str:
        return (f"<MultiLegPulsar {self.name} [{self.timing}, ref={self.reference}]: "
                f"{'+'.join(f'{lg.pta}:{lg.ntoa}' for lg in self.legs)} TOAs, {self.Mmat.shape[1]} timing columns>")

    def leg_slices(self) -> dict[str, np.ndarray]:
        return {lg.pta: np.flatnonzero(self.leg == i) for i, lg in enumerate(self.legs)}


def stack_legs(name: str, legs: list[Pulsar], ptas: list[str], *, timing: str, shared: list[str] | None = None,
               reference: str | None = None) -> MultiLegPulsar:
    """Stack leg Pulsars (each already evaluated with the right model) into one multi-leg pulsar.

    ``timing="per_leg"``: block-diagonal M. ``"shared"``: columns named in ``shared`` are merged
    across legs (each leg must have them); every other column stays per leg.
    """
    if timing not in ("per_leg", "shared"):
        raise ValueError(f"unknown timing mode {timing!r}")
    if len(set(ptas)) != len(ptas):
        raise ValueError(f"{name}: one leg per PTA expected, got {ptas}")
    shared = list(shared or []) if timing == "shared" else []
    n = [p.ntoa for p in legs]
    off = np.concatenate([[0], np.cumsum(n)])
    cols, names = [], []
    for nm in shared:
        col = np.zeros(off[-1])
        for i, p in enumerate(legs):
            if nm not in p.fitpars:
                raise ValueError(f"{name}: shared parameter {nm} missing from the {ptas[i]} leg")
            col[off[i]:off[i + 1]] = p.Mmat[:, p.fitpars.index(nm)]
        cols.append(col)
        names.append(nm)
    for i, p in enumerate(legs):
        for j, nm in enumerate(p.fitpars):
            if nm in shared:
                continue
            col = np.zeros(off[-1])
            col[off[i]:off[i + 1]] = p.Mmat[:, j]
            cols.append(col)
            names.append(f"{nm}:{ptas[i]}")
    M = np.column_stack(cols)
    cat = lambda a: np.concatenate([np.asarray(getattr(p, a)) for p in legs])
    systems = np.concatenate([[namespace(pta, s) for s in p.backend_flags] for p, pta in zip(legs, ptas, strict=True)])
    leg_idx = np.concatenate([np.full(p.ntoa, i) for i, p in enumerate(legs)])
    toas = cat("toas")
    isort = np.argsort(toas, kind="mergesort")
    flag_names = sorted({k for p in legs for k in p.flags})
    flags = {}
    for k in flag_names:
        flags[k] = np.concatenate([np.asarray(p.flags.get(k, np.full(p.ntoa, "")), dtype="U") for p in legs])[isort]
    flags["pta"] = np.concatenate([np.full(p.ntoa, pta) for p, pta in zip(legs, ptas, strict=True)])[isort]
    ref_i = ptas.index(reference) if reference in ptas else 0
    infos = [LegInfo(pta, p.meta.get("dataset", ""), p.meta.get("label", p.name), p.ntoa,
                     (float(p.toas.min() / 86400), float(p.toas.max() / 86400)), list(p.fitpars), dict(p.meta))
             for p, pta in zip(legs, ptas, strict=True)]
    return MultiLegPulsar(
        name=name, timing=timing, reference=reference, legs=infos,
        toas=toas[isort], stoas=cat("stoas")[isort], residuals=cat("residuals")[isort], toaerrs=cat("toaerrs")[isort],
        freqs=cat("freqs")[isort], freqs_topo=cat("freqs_topo")[isort], backend_flags=systems[isort].astype("U"),
        telescope=cat("telescope")[isort], Mmat=M[isort], fitpars=names, pos=np.asarray(legs[ref_i].pos),
        leg=leg_idx[isort], flags=flags, pos_enterprise=legs[ref_i].pos_enterprise,
        meta={"shared": shared, "ptas": list(ptas)},
    )


# ---------------------------------------------------------------------- duplicate observations (G7)

SITE_ALIASES = {
    # tim observatory codes -> physical site
    "ao": "arecibo", "arecibo": "arecibo", "3": "arecibo",
    "gbt": "gbt", "1": "gbt", "gb": "gbt",
    "vla": "vla", "6": "vla", "chime": "chime", "y": "chime",
    "pks": "parkes", "parkes": "parkes", "7": "parkes", "pk": "parkes",
    "meerkat": "meerkat", "mk": "meerkat", "m": "meerkat",
    "gmrt": "gmrt", "gm": "gmrt", "r": "gmrt",
    "eff": "effelsberg", "effelsberg": "effelsberg", "g": "effelsberg", "ef": "effelsberg",
    "jbo": "jodrell", "jb": "jodrell", "8": "jodrell", "jbodfb": "jodrell", "jboroach": "jodrell", "jbafb": "jodrell",
    "ncy": "nancay", "nancay": "nancay", "f": "nancay", "nc": "nancay", "ncyobs": "nancay",
    "wsrt": "wsrt", "i": "wsrt", "we": "wsrt", "srt": "srt", "z": "srt", "leap": "leap",
}


def site_of(obs: str) -> str:
    return SITE_ALIASES.get(obs.lower(), obs.lower())


def _obs_table(recs) -> tuple[np.ndarray, list]:
    """(t [s since MJD 50000], freq [MHz], duration [s], bandwidth [MHz]) per record, and the
    physical sites."""
    rows, sites = [], []
    for r in recs:
        day, sec = r.mjd_parts()
        tobs, bw = r.flag_values("-tobs"), r.flag_values("-bw")
        rows.append(((day - 50000) * 86400.0 + float(sec), r.freq, float(tobs[0]) if tobs and _isf(tobs[0]) else 0.0,
                     abs(float(bw[0])) if bw and _isf(bw[0]) else 0.0))
        sites.append(site_of(r.obs))
    return np.array(rows, dtype=np.float64).reshape(-1, 4), sites


def find_duplicates(legs: dict, *, dfreq_mhz: float = 1.0, min_duration_s: float = 1.0, freq_mode: str = "channel") -> list[dict]:
    """Same observation in two legs (review M3a #2: symmetric, order-independent).

    Criterion for a pair (a, b): same physical site (or one of them LEAP, the coherent sum of the
    EPTA telescopes), observation intervals centred on the TOAs overlapping,
    |t_a - t_b| < (d_a + d_b) / 2 with d the ``-tobs`` duration (at least ``min_duration_s``), and
    radio frequencies within ``dfreq_mhz`` (``freq_mode="channel"``: the same TOA counted twice), or
    overlapping bands, |f_a - f_b| < (bw_a + bw_b) / 2 from the ``-bw`` flags (``"band"``:
    simultaneous recordings of the same photons by different backends or channelisations).
    Candidates are enumerated with the window
    (d_a + max_b d_b) / 2, so every pair satisfying the criterion is found whatever the leg order.
    ``legs``: label -> list of TimRecord. Returns one entry per unordered pair of labels with the
    matched TOAs (file, line) on both sides."""
    keys = sorted(legs)
    tab = {k: _obs_table(legs[k]) for k in keys}
    out = []
    for ia, a in enumerate(keys):
        for b in keys[ia + 1:]:
            Xa, Sa = tab[a]
            Xb, Sb = tab[b]
            if len(Xa) == 0 or len(Xb) == 0:
                continue
            da = np.maximum(Xa[:, 2], min_duration_s)
            db = np.maximum(Xb[:, 2], min_duration_s)
            ob = np.argsort(Xb[:, 0], kind="mergesort")
            tb = Xb[ob, 0]
            dbmax = float(db.max())
            pairs = []
            for i in range(len(Xa)):
                w = 0.5 * (da[i] + dbmax)
                lo, hi = np.searchsorted(tb, Xa[i, 0] - w), np.searchsorted(tb, Xa[i, 0] + w, side="right")
                for jj in range(lo, hi):
                    j = ob[jj]
                    same = Sa[i] == Sb[j] or "leap" in (Sa[i], Sb[j])
                    df_tol = dfreq_mhz if freq_mode == "channel" else max(dfreq_mhz, 0.5 * (Xa[i, 3] + Xb[j, 3]))
                    if (same and abs(Xa[i, 0] - Xb[j, 0]) < 0.5 * (da[i] + db[j])
                            and abs(Xa[i, 1] - Xb[j, 1]) < df_tol):
                        ra, rb = legs[a][i], legs[b][j]
                        pairs.append({"a": [Path(ra.file).name, ra.lineno, Sa[i], float(Xa[i, 2])],
                                      "b": [Path(rb.file).name, rb.lineno, Sb[j], float(Xb[j, 2])],
                                      "dt_s": float(Xa[i, 0] - Xb[j, 0]), "freq_mhz": [float(Xa[i, 1]), float(Xb[j, 1])],
                                      "mjd_a": ra.sat})
            if pairs:
                out.append({"a": a, "b": b, "n_pairs": len(pairs), "pairs": pairs})
    return out


def _isf(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------- builders (PINT, subprocesses)


def _canonical_model(dataset: str, label: str, evaluator):
    import io
    from collections import Counter

    from pint.models import get_model

    from .legs import canonical_leg_texts
    from .m3data import leg_files

    par, tim = leg_files(dataset)[label]
    _, _, text, _ = canonical_leg_texts(par, tim, Counter())
    return get_model(io.StringIO(text), **evaluator.pint_kwargs())


def _c_worker(args):
    import os
    import traceback

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    dataset, label, ref, clock, tag, pin, force = args
    from pint.logging import setup

    from .legs import LegResult, leg_cache_path, load_leg, resolve_nharms, save_leg
    from .m3data import leg_files
    from .profiles import evaluator_for_par

    setup(level="ERROR")
    try:
        ref_par = leg_files(ref[0])[ref[1]][0]
        ev = evaluator_for_par(ref_par.read_text(errors="replace"))
        ref_model = _canonical_model(ref[0], ref[1], ev)
        tgt_model = ref_model if (dataset, label) == tuple(ref) else _canonical_model(dataset, label, ev)
        text, info = consistent_par(ref_model, tgt_model, **(force or {}))
        # the binary model is the reference's: its harmonic count comes from the reference's
        # original par under the reference's evaluator convention, not from PINT's serialised
        # (forced >= 7) NHARMS (review M3a #1)
        nh = None
        if "NHARMS" in ref_model.params and getattr(ref_model, "H4", None) is not None and ref_model.H4.quantity is not None:
            nh = resolve_nharms(ref_par.read_text(errors="replace"), ev)
            if nh is not None:
                text = "\n".join(f"NHARMS {nh}" if ln.split()[:1] == ["NHARMS"] else ln for ln in text.splitlines()) + "\n"
        res = load_leg(dataset, label, clock=clock, evaluator=ev, par_text=text, pin=pin, tag=tag, nharms=nh)
        res.meta["consistent"] = info
        res.meta["reference"] = list(ref)
        # admissibility reference: the leg's OWN canonical model, evaluated with the same CLOCK and
        # EPHEM as the rewritten leg and the same clock-file profile, with every DMX column free
        # (the rewrite removes DMX by design; DM variations go to the DM GP). Differences between
        # the two are then due to the rewrite alone.
        from collections import Counter as _C

        from .legs import canonical_leg_texts
        from .profiles import ClockProfile

        par, tim = leg_files(dataset)[label]
        _, _, own_text, _ = canonical_leg_texts(par, tim, _C(), dataset, label)
        cl = [ln.split()[1] for ln in text.splitlines() if ln.split()[:1] == ["CLOCK"]]
        ep = [ln.split()[1] for ln in text.splitlines() if ln.split()[:1] == ["EPHEM"]]
        own_prof = ClockProfile(clock.name, cl[0] if cl else None, ep[0] if ep else None, clock.overrides, clock.note)
        own_text, _ = own_prof.apply_to_par(own_text)
        own = load_leg(dataset, label, clock=clock, par_text=own_text, pin=pin, tag=tag + "-own", free_dmx=True,
                       nharms=resolve_nharms(par.read_text(errors="replace"), evaluator_for_par(par.read_text(errors="replace")))
                       if "H4" in {ln.split()[0] for ln in own_text.splitlines() if ln.split()} else None)
        if own.ok:
            save_leg(own, leg_cache_path(tag + "-own", dataset, label))
        res.meta["admissibility_reference"] = {"tag": tag + "-own", "ok": own.ok, "error": own.error,
                                               "clock": own_prof.clock, "ephem": own_prof.ephem}
    except Exception as ex:  # noqa: BLE001
        res = LegResult(dataset, "", label, False, None, {"error": repr(ex), "traceback": traceback.format_exc()}, repr(ex))
    save_leg(res, leg_cache_path(tag, dataset, label))
    return res


def _b_worker(args):
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    dataset, label, clock, tag, pin = args
    from .legs import leg_cache_path, load_leg, save_leg

    res = load_leg(dataset, label, clock=clock, pin=pin, tag=tag)
    save_leg(res, leg_cache_path(tag, dataset, label))
    return res


def build_multileg(name: str, legs: list[tuple[str, str]], *, timing: str, clock, reference: str | None = None,
                   tag: str | None = None, jobs: int = 4, pin: bool = False, force: dict | None = None,
                   allow_inadmissible: bool = False):
    """Ingest the legs ((dataset, label) list) of one pulsar for option B or C and stack them.

    Returns (MultiLegPulsar, {pta: LegResult}). ``clock``: the clock profile for every leg (the
    combined profile for B; for C a profile with clock = ephem = None, since the rewrite copies the
    reference's CLOCK/EPHEM). Each leg runs in a fresh spawned process.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from .m3data import pta_of

    ptas = [pta_of(d) for d, _ in legs]
    ctx = multiprocessing.get_context("spawn")
    if timing == "shared":
        reference = reference or choose_reference(ptas)
        ref = legs[ptas.index(reference)]
        tag = tag or (f"C-ref{reference}" + ("-forced" if force and (force.get("clock") or force.get("ephem")) else "")
                      + ("-localDM" if force and force.get("local_dm") else ""))
        work = [(d, lab, ref, clock, tag, pin, force) for d, lab in legs]
        fn = _c_worker
    else:
        tag = tag or "B"
        work = [(d, lab, clock, tag, pin) for d, lab in legs]
        fn = _b_worker
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, max_tasks_per_child=1) as ex:
        results = list(ex.map(fn, work))
    bad = [r for r in results if not r.ok]
    if bad:
        raise RuntimeError(f"{name}: leg ingestion failed: {[(r.dataset, r.label, r.error[:300]) for r in bad]}")
    shared = None
    if timing == "shared":
        shared = results[ptas.index(reference)].meta["consistent"]["shared_free"]
        shared = [p for p in shared if all(p in r.psr.fitpars for r in results)]
    for r in results:  # provenance carried into LegInfo.meta (serialised by save_multileg)
        r.psr.meta.update({k: r.meta.get(k) for k in ("profile", "nharms_used", "ell1h_nharms_config", "consistent",
                                                     "reference", "tag", "site_profile", "clock_policy")})
        r.psr.meta["n_clock_excluded"] = len(r.meta.get("clock_excluded", []))
    mp = stack_legs(name, [r.psr for r in results], ptas, timing=timing, shared=shared, reference=reference)
    if timing == "per_leg":
        mp.meta.update(config=CONFIG_B, admissible=True)
    else:
        mp.meta["config"] = CONFIG_LOCAL_DM if (force and force.get("local_dm")) else CONFIG_YA
        mp.meta["config_force"] = dict(force or {})
        from .data import Pulsar
        from .legs import leg_cache_path

        adm = {}
        for p_, r in zip(ptas, results, strict=True):
            ref_info = r.meta.get("admissibility_reference", {})
            if not ref_info.get("ok"):
                adm[p_] = {"admissible": False, "linearisation_rms_whitened": float("inf"), "n_matched": 0,
                           "union_rank": 0, "union_ncols": 0, "linearisation_rms_ns": float("inf"),
                           "threshold": ADMISSIBILITY_RMS, "error": ref_info.get("error", "no reference load")}
                continue
            own = Pulsar.load(leg_cache_path(ref_info["tag"], r.dataset, r.label))
            adm[p_] = admissibility(r.psr, own)
        for lg in mp.legs:
            lg.meta["admissibility"] = adm[lg.pta]
        mp.meta["admissibility"] = adm
        mp.meta["admissible"] = all(a["admissible"] for a in adm.values())
        if not mp.meta["admissible"]:
            bad = {k: round(v["linearisation_rms_whitened"], 3) for k, v in adm.items() if not v["admissible"]}
            if not allow_inadmissible:
                raise InadmissibleBuildError(
                    f"{name} [{mp.meta['config']}]: legs not a linear re-parameterisation of their own models "
                    f"(whitened linearisation rms {bad} > {ADMISSIBILITY_RMS}); pass allow_inadmissible=True for a "
                    "diagnostic build")
            mp.meta["diagnostic_override"] = True
    return mp, dict(zip(ptas, results, strict=True))


def _published_leg(dataset: str, label: str):
    """The leg evaluated with its own (canonical, published-profile) model, from the cache or fresh."""
    from .data import Pulsar
    from .legs import leg_cache_path, load_legs

    p = leg_cache_path("published", dataset, label)
    if not p.exists():
        load_legs([(dataset, label)], jobs=1, tag="published")
    return Pulsar.load(p)


def save_multileg(mp: MultiLegPulsar, path) -> None:
    import json
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: getattr(mp, k) for k in ("toas", "stoas", "residuals", "toaerrs", "freqs", "freqs_topo", "Mmat", "pos", "leg")}
    arrays["backend_flags"] = mp.backend_flags.astype("U")
    arrays["telescope"] = np.asarray(mp.telescope).astype("U")
    arrays.update({f"flag_{k}": np.asarray(v).astype("U") for k, v in mp.flags.items()})
    if mp.pos_enterprise is not None:
        arrays["pos_enterprise"] = mp.pos_enterprise
    legs = [{"pta": lg.pta, "dataset": lg.dataset, "label": lg.label, "ntoa": lg.ntoa, "span_mjd": list(lg.span_mjd),
             "fitpars": lg.fitpars, "meta": {k: v for k, v in lg.meta.items() if k != "record_index"}} for lg in mp.legs]
    header = {"name": mp.name, "timing": mp.timing, "reference": mp.reference, "fitpars": mp.fitpars, "legs": legs,
              "meta": mp.meta}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, _header=np.array(json.dumps(header, default=str)), **arrays)
    os.replace(tmp, path)


def load_multileg(path) -> MultiLegPulsar:
    import json

    with np.load(path, allow_pickle=False) as z:
        h = json.loads(str(z["_header"]))
        flags = {k[5:]: z[k] for k in z.files if k.startswith("flag_")}
        legs = [LegInfo(lg["pta"], lg["dataset"], lg["label"], lg["ntoa"], tuple(lg["span_mjd"]), lg["fitpars"],
                        lg.get("meta", {})) for lg in h["legs"]]
        return MultiLegPulsar(
            name=h["name"], timing=h["timing"], reference=h["reference"], legs=legs, toas=z["toas"], stoas=z["stoas"],
            residuals=z["residuals"], toaerrs=z["toaerrs"], freqs=z["freqs"], freqs_topo=z["freqs_topo"],
            backend_flags=z["backend_flags"], telescope=z["telescope"], Mmat=z["Mmat"], fitpars=h["fitpars"],
            pos=z["pos"], leg=z["leg"], flags=flags,
            pos_enterprise=z["pos_enterprise"] if "pos_enterprise" in z.files else None, meta=h["meta"])
