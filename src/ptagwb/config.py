"""Global configuration helpers (precision, paths)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def enable_x64() -> None:
    """PTA likelihoods need float64 (residuals ~1e-7 s, covariances span ~30 decades)."""
    import jax

    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------- priors
# Uniform prior bounds. Verified against the released production chains where possible (see
# docs/M1_VALIDATION.md, Sec. 7, and the errata in docs/SPEC_astra.md). ``log10_rho`` is the
# enterprise/discovery free-spectrum parameter: log10 of the coefficient *RMS* rho in seconds,
# coefficient variance phi_k = rho_k^2 = 10^(2 log10_rho_k).
FREESPEC_LOG10_RHO_PRIORS = {
    # What the released HD free-spectrum chains sampled: the Fig. 1(a) core
    # 30fCP_30fiRN_3A_freespec_chain.core and the tutorial hd_30f_fs.core. In both, ln(prior)
    # is constant (mean -357.8144861503552 in the figure core). With 67 x U[-20,-11] x U[0,7]
    # IRN priors and equal widths, that implies 14.499999999675 per rho; samples reach -15.50
    # and go no lower. INFERRED, not a recovered sampler config. The Ceffyl HD KDE grids
    # (Zenodo 8060824) span exactly [-15.5, -1.0].
    "production": (-15.5, -1.0),
    # Paper Table 1, "log-Uniform in rho_i [-18, -8]" read as log10(phi_k / s^2), i.e.
    # log10_rho in [-9, -4]. This is also the histogram range the Fig. 1(a) notebook uses.
    "paper_table1": (-9.0, -4.0),
}
PRIORS = {
    "rn_log10_A": (-20.0, -11.0),  # paper Table 1; m2a/m3a chain priors
    "rn_gamma": (0.0, 7.0),  # paper Table 1; m2a/m3a chain priors
    "gw_log10_A": (-18.0, -11.0),  # varied gamma: paper Table 1; m2a/m3a and 14f_PL_hd_crn priors
    "gw_gamma": (0.0, 7.0),
    # fixed gamma = 13/3. RESOLVED in M2 (docs/M2_RESULTS.md, Sec. "Priors"): every released
    # fixed-gamma product whose prior we can recover used U[-18,-11], not the U[-18,-14] of paper
    # Table 1: SplineORF_FixedGamma_NL.core stores 'gw_crn_log10_A:Uniform(pmin=-18, pmax=-11)' and
    # its lnpost-lnlike = -67 ln 63 - ln 7 - 7 ln 1.8; the tutorial product-space core curn_hd.core
    # (CURN^13/3 vs HD^13/3) has lnpost-lnlike = 2 (-67 ln 63) - 2 ln 7 and inactive-model amplitude
    # samples spanning [-18.00, -11.00]. The posterior never reaches -14 (released HD^13/3 chain max
    # -14.36), so the choice only shifts evidences against IRN by ln(7/4); it cancels in HD/CURN.
    "gw_log10_A_fixed_gamma": (-18.0, -11.0),
    "freespec_log10_rho": FREESPEC_LOG10_RHO_PRIORS["production"],
}
# Alternatives kept for sensitivity checks (ModelSpec.prior_overrides).
PRIOR_ALTERNATIVES = {
    "gw_log10_A_fixed_gamma": {"paper_table1": (-18.0, -14.0)},
    "freespec_log10_rho": FREESPEC_LOG10_RHO_PRIORS,
}
# CURN free spectrum: no released CURN^free chain (and so no normaliser) exists in any bundle we
# have. The only provenance is the Ceffyl v1 CP KDE grid, which spans [-15.1, -0.9] (the HD grids
# span exactly the HD^free prior [-15.5, -1.0]). INFERRED, weak: CURN^free most likely used a
# similarly wide U[-15.1, -0.9]-ish prior; bins with signal (f_1..f_8) are likelihood dominated
# and insensitive to it. Not needed for the M2 deliverables (HD^free only).
CURN_FREESPEC_LOG10_RHO_INFERRED = (-15.1, -0.9)
