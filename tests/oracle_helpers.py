"""Helpers shared by the oracle tests and scripts (may import oracle packages lazily)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from ptagwb.config import RAW_DIR
from ptagwb.data import Pulsar

HAVE_DISCOVERY = importlib.util.find_spec("discovery") is not None
HAVE_ENTERPRISE = (
    importlib.util.find_spec("enterprise") is not None and importlib.util.find_spec("sksparse") is not None
)


def discovery_data_dir() -> Path:
    return next((RAW_DIR / "discovery_repo" / "extracted").glob("discovery-*")) / "data"


def feather_paths() -> list[Path]:
    return sorted(discovery_data_dir().glob("v1p1_de440_pint_bipm2019-*.feather"))


def pulsar_from_feather(path: Path) -> Pulsar:
    """Released discovery/enterprise feather -> our Pulsar container (no oracle import)."""
    import pyarrow.feather as pf

    t = pf.read_table(path)
    meta = json.loads(t.schema.metadata[b"json"])
    col = lambda c: t[c].to_numpy()
    M = np.array([col(c) for c in t.column_names if c.startswith("Mmat_")]).T.copy()
    flags = {c[6:]: col(c).astype("U") for c in t.column_names if c.startswith("flags_")}
    return Pulsar(
        name=meta["name"],
        toas=col("toas").astype(np.float64),
        stoas=col("stoas").astype(np.float64),
        residuals=col("residuals").astype(np.float64),
        toaerrs=col("toaerrs").astype(np.float64),
        freqs=col("freqs").astype(np.float64),
        freqs_topo=np.full(len(col("toas")), np.nan),
        backend_flags=col("backend_flags").astype("U"),
        telescope=np.array([""] * len(col("toas")), dtype="U"),
        Mmat=M,
        fitpars=list(meta.get("fitpars") or []),
        pos=np.asarray(meta["pos"], dtype=np.float64),
        pos_enterprise=np.asarray(meta["pos"], dtype=np.float64),  # feathers carry enterprise positions
        flags=flags,
        meta={"source": str(path), "noisedict": meta.get("noisedict", {})},
    )


def feather_pulsars() -> list[Pulsar]:
    return [pulsar_from_feather(p) for p in feather_paths()]


def toa_keys(psr: Pulsar) -> np.ndarray:
    """Per-TOA identity from tim-file flags (observation name, subband, subint, system)."""
    f = psr.flags
    return np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in zip(f["name"], f["chan"], f["subint"], f["f"])])


def match_toas(ours: Pulsar, ref: Pulsar) -> np.ndarray:
    """perm such that ours.<array>[perm] lines up with ref.<array> (keys must be unique)."""
    k1, k2 = toa_keys(ours), toa_keys(ref)
    if len(set(k1)) != len(k1) or len(set(k2)) != len(k2) or len(k1) != len(k2):
        raise ValueError(f"{ours.name}: TOA keys not unique or counts differ")
    o1, o2 = np.argsort(k1), np.argsort(k2)
    if not np.all(k1[o1] == k2[o2]):
        raise ValueError(f"{ours.name}: TOA key sets differ")
    perm = np.empty(len(k1), dtype=np.int64)
    perm[o2] = o1
    return perm


def to_discovery_pulsar(psr: Pulsar, noisedict: dict | None = None):
    """Our Pulsar -> discovery.Pulsar (for feeding our arrays to the oracle)."""
    import discovery as ds

    p = ds.Pulsar()
    p.name = psr.name
    p.toas, p.stoas = np.asarray(psr.toas), np.asarray(psr.stoas)
    p.toaerrs, p.residuals = np.asarray(psr.toaerrs), np.asarray(psr.residuals)
    p.freqs, p.backend_flags = np.asarray(psr.freqs), np.asarray(psr.backend_flags).astype("U")
    p.Mmat = np.asarray(psr.Mmat)
    p.pos = np.asarray(psr.pos)
    p.flags = dict(psr.flags)
    p.fitpars = list(psr.fitpars)
    p.mintoa, p.maxtoa = p.toas.min(), p.toas.max()
    if noisedict is not None:
        p.noisedict = {k: v for k, v in noisedict.items() if k.startswith(psr.name + "_")}
    return p


def load_chain(model: str):
    """Released discovery-repo NG15 chain ('m2a' = CURN^gamma, 'm3a' = HD^gamma)."""
    import pyarrow.feather as pf

    return pf.read_table(discovery_data_dir() / f"NG15yr-{model}-chain.feather").to_pandas()


# ---------------------------------------------------------------------- front-end comparison


def principal_angle_sin(A: np.ndarray, B: np.ndarray) -> float:
    """Largest sine of the principal angles between span(A) and span(B) (column-normalised)."""
    Ua, _, _ = np.linalg.svd(A / np.linalg.norm(A, axis=0), full_matrices=False)
    Ub, _, _ = np.linalg.svd(B / np.linalg.norm(B, axis=0), full_matrices=False)
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))


def compare_frontend(ours: Pulsar, ref: Pulsar, noisedict: dict | None = None) -> dict:
    """Per-pulsar differences ours - ref after matching TOAs by tim-file key."""
    from ptagwb.likelihood import timing_basis
    from ptagwb.noise import build_white_noise

    perm = match_toas(ours, ref)
    o = {k: np.asarray(getattr(ours, k))[perm] for k in ("toas", "residuals", "toaerrs", "freqs", "backend_flags")}
    M = np.asarray(ours.Mmat)[perm]
    dr = o["residuals"] - ref.residuals
    U = timing_basis(ref.Mmat, "svd")
    dr_perp = dr - U @ (U.T @ dr)
    out = {
        "name": ours.name,
        "ntoa": ours.ntoa,
        "ntoa_ref": ref.ntoa,
        "max_dtoa_s": float(np.max(np.abs(o["toas"] - ref.toas))),
        "max_dtoaerr_s": float(np.max(np.abs(o["toaerrs"] - ref.toaerrs))),
        "max_dfreq_mhz": float(np.max(np.abs(o["freqs"] - ref.freqs))),
        "backend_flags_equal": bool(np.all(o["backend_flags"] == ref.backend_flags)),
        "max_dres_s": float(np.max(np.abs(dr))),
        "rms_dres_s": float(np.sqrt(np.mean(dr**2))),
        "max_dres_perp_s": float(np.max(np.abs(dr_perp))),
        "rms_dres_perp_s": float(np.sqrt(np.mean(dr_perp**2))),
        "rms_res_s": float(np.sqrt(np.mean(ref.residuals**2))),
        "ncol": M.shape[1],
        "ncol_ref": ref.Mmat.shape[1],
        "M_span_sin": principal_angle_sin(M, ref.Mmat) if M.shape[1] == ref.Mmat.shape[1] else float("nan"),
        "pos_angle_rad": float(np.arccos(np.clip(np.dot(ours.pos, ref.pos), -1, 1))),
    }
    if noisedict is not None:
        w1 = build_white_noise(ours, noisedict)
        w2 = build_white_noise(ref, noisedict)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        # canonical bucket sets in ref-row indices
        b1 = sorted(tuple(sorted(inv[np.flatnonzero(w1.epoch == e)])) for e in range(w1.n_epoch))
        b2 = sorted(tuple(np.flatnonzero(w2.epoch == e)) for e in range(w2.n_epoch))
        out["n_epochs"] = w1.n_epoch
        out["ecorr_buckets_equal"] = b1 == b2
    return out


# ---------------------------------------------------------------------- likelihood oracles


def discovery_model(dpsrs, orf: str, T: float, ecorr_enterprise: bool = True, n_rn: int = 30, n_gw: int = 14):
    """discovery CURN^gamma / HD^gamma as in discovery.models.nanograv, with the ECORR
    quantisation selectable (``enterprise=True`` = nmin 2, as enterprise / the NG15 runs)."""
    import discovery as ds

    psls = [
        ds.PulsarLikelihood(
            [
                p.residuals,
                ds.makenoise_measurement(p, p.noisedict),
                ds.makegp_ecorr(p, p.noisedict, enterprise=ecorr_enterprise),
                ds.makegp_timing(p, svd=True),
            ]
        )
        for p in dpsrs
    ]
    if orf == "curn":
        cgp = ds.makecommongp_fourier(
            dpsrs, ds.makepowerlaw_crn(n_gw), n_rn, T=T, name="red_noise", common=["crn_log10_A", "crn_gamma"]
        )
        return ds.ArrayLikelihood(psls, commongp=cgp)
    rngp = ds.makecommongp_fourier(dpsrs, ds.powerlaw, n_rn, T=T, name="red_noise")
    hdgp = ds.makeglobalgp_fourier(dpsrs, ds.powerlaw, ds.hd_orf, n_gw, T=T, name="gw")
    return ds.ArrayLikelihood(psls, commongp=rngp, globalgp=hdgp)


def to_discovery_params(named: dict, orf: str) -> dict:
    """Chain-style names (gw_log10_A, gw_gamma) -> discovery names for discovery_model."""
    out = dict(named)
    if orf == "curn":
        out["crn_log10_A"], out["crn_gamma"] = named["gw_log10_A"], named["gw_gamma"]
    return out


def parameter_points(
    names: list[str],
    chain=None,
    n_chain: int = 6,
    n_prior: int = 6,
    seed: int = 0,
    rn_log10_A_range: tuple[float, float] = (-20.0, -11.0),
) -> list[dict]:
    """Chain samples (realistic) + prior draws (IRN log10_A in ``rn_log10_A_range`` (prior:
    U[-20,-11]), gamma U[0,7]; common log10_A U[-18,-13], gamma U[0,7])."""
    rng = np.random.default_rng(seed)
    pts = []
    if chain is not None and n_chain:
        for i in rng.choice(len(chain), n_chain, replace=False):
            row = chain.iloc[int(i)]
            pts.append({k: float(row[k]) for k in chain.columns if k.endswith(("_log10_A", "_gamma"))} | {"_chain_row": int(i)})
    for _ in range(n_prior):
        d = {}
        for n in names:
            d[f"{n}_red_noise_log10_A"] = rng.uniform(*rn_log10_A_range)
            d[f"{n}_red_noise_gamma"] = rng.uniform(0, 7)
        d["gw_log10_A"], d["gw_gamma"] = rng.uniform(-18, -13), rng.uniform(0, 7)
        pts.append(d)
    return pts


def to_enterprise_pulsar(psr: Pulsar):
    """Our Pulsar -> enterprise FeatherPulsar-like object (arrays already sorted)."""
    from enterprise.pulsar import FeatherPulsar

    p = FeatherPulsar()
    p.name = psr.name
    p.toas, p.stoas = np.asarray(psr.toas), np.asarray(psr.stoas)
    p.toaerrs, p.residuals = np.asarray(psr.toaerrs), np.asarray(psr.residuals)
    p.freqs = np.asarray(psr.freqs)
    p.backend_flags = np.asarray(psr.backend_flags).astype("U")
    p.telescope = np.asarray(psr.telescope)
    p.Mmat = np.asarray(psr.Mmat)
    p.pos = np.asarray(psr.pos)
    p.flags = {k: np.asarray(v) for k, v in psr.flags.items()}
    p.fitpars = list(psr.fitpars)
    p.planetssb = p.sunssb = p.pos_t = None
    p.sort_data()
    return p


def enterprise_pta(psrs: list[Pulsar], noisedict: dict, T: float, orf: str, n_rn: int = 30, n_gw: int = 14):
    """enterprise CURN^gamma / HD^gamma exactly as the NG15 tutorial (parameter_est.ipynb):
    MarginalizingTimingModel(use_svd=True), MeasurementNoise(efac, log10_t2equad) and
    EcorrKernelNoise by backend, 30-component power-law IRN, 14-component common process."""
    from enterprise.signals import (
        gp_signals,
        parameter,
        selections,
        signal_base,
        utils,
        white_signals,
    )

    sel = selections.Selection(selections.by_backend)
    efac, t2equad, ecorr = parameter.Constant(), parameter.Constant(), parameter.Constant()
    mn = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=sel)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=sel)
    rn = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw(log10_A=parameter.Uniform(-20, -11), gamma=parameter.Uniform(0, 7)),
        components=n_rn,
        Tspan=T,
    )
    cpl = utils.powerlaw(log10_A=parameter.Uniform(-18, -11)("gw_log10_A"), gamma=parameter.Uniform(0, 7)("gw_gamma"))
    if orf == "curn":
        gw = gp_signals.FourierBasisGP(spectrum=cpl, components=n_gw, Tspan=T, name="gw")
    else:
        gw = gp_signals.FourierBasisCommonGP(cpl, orf=utils.hd_orf(), components=n_gw, Tspan=T, name="gw")
    tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    s = tm + mn + ec + rn + gw
    pta = signal_base.PTA([s(to_enterprise_pulsar(p)) for p in psrs])
    pta.set_default_params(noisedict)
    return pta
