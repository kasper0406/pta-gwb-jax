# Reproduction plan: NANOGrav 15-yr GWB with our own JAX pipeline

Target paper: Agazie et al. 2023, *The NANOGrav 15 yr Data Set: Evidence for a
Gravitational-wave Background*, ApJL 951 L8, arXiv:2306.16213 (called **GWB** below).
Companion: Agazie et al. 2023, *Detector Characterization and Noise Budget*, ApJL 951 L10,
arXiv:2306.16218 (called **DETCHAR**). Data set paper: arXiv:2306.16217 (ApJL 951 L9).

Numbers below come from the arXiv LaTeX source of GWB (v-latest as of 2026-10-06) unless
stated otherwise. **[UNCERTAIN]** marks things I could not pin down from the text; those must
be checked against the reference code/chains before we rely on them.

Figure numbers follow the order of the figures in the GWB main text (Fig. 1 summary,
2 Bayes factors, 3 phase-shift backgrounds, 4 OS S/N, 5 DMGP vs DMX, 6 free spectrum,
7 OS Legendre, 8 dropout, 9 time slicing, 10 telescope split, 11 astrophysical comparison).
This matches the `data_release/figure_N` folders of `nanograv/15yr_stochastic_analysis`.
Appendix tables/figures are cited by appendix and LaTeX label, because AASTeX's
`appendixfloats` numbering is **[UNCERTAIN]**.

---

## 1. Data and analysis settings

| Item | Setting | Source |
|---|---|---|
| Data | NG15 **narrowband** TOAs + PINT par files, `narrowband/par`, `narrowband/tim` | GWB §2 |
| Release used | Zenodo concept 10.5281/zenodo.7967584. We use v2.1.0 (record 16051178). The narrowband par/tim are **byte-identical** to v1.0.1 (record 8104459, June 2023); checked with `diff -r` | own check |
| Pulsars | 68 in the data set; **67** with timing baseline >= 3 yr are used. The excluded one is **J0614-3329** (span 2.41 yr in our smoke test) | GWB §2, App. A |
| Split-telescope files (`*ao`, `*gbt`) | Only for the Arecibo/GBT cross-validation (GWB §5.4, Fig. 10): 33 AO + 35 GBT pulsars. Not used in the baseline | GWB App. A |
| Time span T | **16.03 yr** = first-to-last TOA over the whole array. The same T sets f_i = i/T for every pulsar and every process. Our smoke test gets 16.030 yr | GWB §2 |
| Ephemeris / clock | **JPL DE440**, **TT(BIPM2019)**. Every par file has `EPHEM DE440`, `CLOCK TT(BIPM2019)`, `UNITS TDB`, `PLANET_SHAPIRO Y`, `CORRECT_TROPOSPHERE Y` | GWB §2; par files |
| BayesEphem | Not in the baseline. The text says Jovian-orbit uncertainties "are now negligible" with DE440 **[UNCERTAIN: confirm in reference model code]** | GWB §2 |
| Timing model | Linearised around the par-file solution. Analytically marginalised with design matrix M and a flat infinite prior on the coefficients. M covers spin, astrometry including PM and PX for every pulsar, binary terms, **DMX** (piecewise-constant DM), FD terms and JUMPs. discovery uses `makegp_timing(psr, svd=True)`, i.e. SVD-orthonormalised M | GWB §2 |
| DM model | **DMX** in the timing model, no DM Gaussian process. DMGP is only a robustness check (Fig. 5) | GWB §2, §5 |
| White noise | Per receiver/backend: EFAC, EQUAD (enterprise `log10_t2equad`, i.e. sigma^2 = EFAC^2 (sigma_TOA^2 + EQUAD^2)), ECORR (epoch-correlated across sub-bands). **Fixed** to MAP values from single-pulsar noise runs | GWB §2, DETCHAR §4 |
| White-noise dictionary | `v1p1_wn_dict.json` (697 entries) from the GWB Fig. 1 data bundle. Values are identical to `tutorials/data/15yr_wn_dict.json` in `nanograv/15yr_stochastic_analysis`. Other copies: `v1p1_all_dict.json` (WN + RN, CW release) and the noisedicts embedded in the discovery feather files | own check |
| ECORR grouping | Per backend, enterprise `create_quantization_matrix(dt=1 s, nmin=2)`: a bucket opens at a TOA and collects TOAs within 1 s of it; singletons get no ECORR. discovery's default keeps singletons (logL +0.02). **Resolved in M1** (`docs/M1_VALIDATION.md`) | enterprise source; production-chain logl |
| Intrinsic red noise (IRN) | Power law in every one of the 67 pulsars, **30** Fourier frequencies (i = 1..30, 2-59 nHz) | GWB §2 |
| Common process | **14** frequencies (i = 1..14, 2-28 nHz). 14 is the MAP break frequency of a broken-power-law CURN fit: f_break MAP = 2.75e-8 Hz ~ 14/T. Median 3.2(+5.4/-1.2)e-8 Hz, 90% CI | GWB §2, App. C |
| PSD | phi_i = A^2/(12 pi^2) (1/T) (f_i/f_ref)^(-gamma) f_ref^-3, f_ref = 1/yr. Same form for IRN, CURN and HD | GWB Eq. 5-7 |
| HD ORF | Gamma(xi) = 3/2 x ln x - x/4 + 1/2 + delta_ab/2, x = (1 - cos xi)/2 | GWB Eq. 4 |
| Likelihood | Gaussian. Marginalise analytically over the Fourier coefficients c and the timing coefficients epsilon. C = white-noise covariance (EFAC, EQUAD, ECORR) | GWB Eq. 1-3 |
| Sampler (theirs) | PTMCMC. Convergence: Gelman-Rubin R-hat < 1.01. Bayes factors from product-space sampling (HD vs CURN), thermodynamic integration, Savage-Dickey, reweighting (Hourihane & Meyers 2022) | GWB App. B |

### Priors (GWB Appendix B, `tab:priors`)

| Parameter | Prior |
|---|---|
| IRN log10 A_red | U[-20, -11] |
| IRN gamma_red | U[0, 7] |
| Common power law, gamma = 13/3: log10 A | Paper: U[-18, -14]. **Resolved in M2: the released fixed-gamma products used U[-18, -11]** (normalisers and stored priors; `docs/M2_RESULTS.md` Sec. 2). Posteriors are unaffected |
| Common power law, varied gamma: log10 A | U[-18, -11] |
| Common gamma (varied) | U[0, 7] |
| Free spectrum rho_i [s^2] | The paper says "log-Uniform in rho_i [-18, -8]", i.e. log10_rho in U[-9, -4]. **Resolved (M1 review): the released HD free-spectrum chains sampled enterprise `log10_rho` (log10 RMS in s) ~ U[-15.5, -1.0].** Evidence: constant ln-prior giving width 14.5 per rho, samples bottoming out at -15.5, and Ceffyl HD grids spanning [-15.5, -1]. [-9, -4] is only the Fig. 1(a) histogram range. CURN^free is unverified. See `docs/M1_VALIDATION.md` Sec. 7 and `ptagwb.config.PRIORS` |
| Broken power law | log10 A U[-18,-11], gamma U[0,7], delta = 0, log10 f_bend U[-8.7,-7], ell = 0.1 |
| t-process | log10 A U[-18,-11], gamma U[0,7], x_i ~ InvGamma(1,1) |
| Turnover | log10 A U[-18,-11], gamma U[0,7], kappa U[0,7], log10 f0 U[-9,-7] |
| Spline ORF | 7 knots at (1e-3, 25, 49.3, 82.5, 121.8, 150, 180) deg, y U[-0.9, 0.9] |
| White noise (single-pulsar runs only) | EFAC U[0,10], log10 EQUAD U[-8.5,-5], log10 ECORR U[-8.5,-5] |

### Model names (GWB notation)

IRN (intrinsic only). CURN^gamma / CURN^13/3 (common uncorrelated red noise with varied or
fixed gamma). HD^gamma / HD^13/3 (Hellings-Downs correlated). CURN^free / HD^free (free
spectrum). The common term in CURN uses the autocorrelation only (Phi_ab = delta_ab Phi).
In HD it is Gamma(xi_ab) Phi.

---

## 2. Headline target numbers

Each entry gives the quoted value, where it is quoted, and (where we have one) the released
chain or product to compare against. Checks marked "own check" were run on the released
files during M0.

### 2.1 Parameter estimation

| Quantity | Quoted value | Where | Released reference |
|---|---|---|---|
| **HD^13/3 amplitude** (gamma = 13/3, 14 freq, f_ref = 1/yr) | **A = 2.4(+0.7/-0.6) x 10^-15** (median, 90% credible) | Abstract; §3; Fig. 1(b) dashed curve | `figure1_data/nano15_hd_chain_fg_long_050523.npy` (log10 A): own check gives 2.40e-15 [1.82, 3.07]e-15 (5/50/95%) |
| **HD^gamma 2-D (gamma, A)** | **A_HD = 6.4(+4.2/-2.7) x 10^-15**, **gamma_HD = 3.2 +/- 0.6** (median, 5-95%). gamma = 13/3 lies at the 99% credible boundary. Essentially no support for A_HD < 1e-15 | §3; Fig. 1(b) | `figure1_data/nano15_hd_chain_long_050523.npy` (cols: gamma, log10 A): own check gives gamma 3.25 [2.66, 3.84], A 6.39e-15 [3.70e-15, 1.06e-14]. Also discovery `NG15yr-m3a-chain.feather` (HD^gamma): log10 A -14.197, gamma 3.26 |
| **CURN^gamma** | No numbers in the text. Fig. 5 and DETCHAR show contours only | Fig. 5 (DMX vs DMGP) | discovery `NG15yr-m2a-chain.feather` (CURN^gamma, 67 psr, full IRN): own check gives log10 A = -14.17 [-14.39, -13.98], gamma = 3.35 [2.80, 3.88]. Also `15yr_stochastic_analysis/tutorials/presampled_cores/curn_14f_pl_vg.core` and `figure5 .../curn_variedgamma.h5` |
| **CURN^13/3 amplitude** | **[UNCERTAIN] Not quoted in the GWB text or figure captions as far as I can find.** Do not use a number from memory. Get it from a released CURN^13/3 chain if one turns up, or from our own run checked against the discovery oracle | - | Provisional only: in the tutorial product-space core `presampled_cores/curn_hd.core` (params `gw_crn_log10_A`, `gw_hd_log10_A`, `nmodel`, no gamma, so presumably gamma = 13/3), the CURN-active samples (nmodel < 0.5) give log10 A_CURN = -14.56 [-14.68, -14.48] (5/50/95%), i.e. A ~ 2.7e-15. **[UNCERTAIN: tutorial data are described as 'reduced'; gamma = 13/3 is inferred from the missing gamma parameter; the inactive-model amplitude ranges up to -11.3, which suggests a U[-18,-11] prior rather than the U[-18,-14] of `tab:priors`. The raw nmodel count ratio (0.66) is NOT the Bayes factor, because the product-space model weights are unknown.]** |
| Energy density at gamma = 13/3 | Omega_gw = 9.3(+5.8/-4.0) x 10^-9, rho_gw = 7.7(+4.8/-3.3) x 10^-17 erg cm^-3 (90%; total integrated over the sensitive band, H0 = 70 km/s/Mpc) | §1 (intro) | derived from HD^13/3 |
| t-process CURN | gamma_CURN = 3.5 +/- 1.0 (median, 5-95%) | §5.2 (Spectral analysis); App. D, `fig:tps` | - |
| Turnover CURN vs CURN^gamma | BF = 1.46 +/- 0.02 (inconclusive) | App. E | - |
| Arecibo-only / GBT-only (varied gamma, 68%) | log10 A = -14.02(+0.18/-0.22) (AO), -14.2(+0.15/-0.17) (GBT); gamma = 2.78(+0.70/-0.64) (AO), 3.37(+0.40/-0.38) (GBT); noise-marginalised HD S/N 2.9 (AO), 3.3 (GBT) | §5.4 (Dropout and cross-validation), Fig. 10 | `ng15_gwb_fig10_telescopes` bundle |

### 2.2 Model comparison

| Quantity | Quoted value | Where |
|---|---|---|
| **HD^gamma vs CURN^gamma** | **200 (14 freq) to 1,000 (5 freq)** in the text. Fig. 2 prints **226 +/- 70** and "(965 with 5 freqs.)". Results are similar for HD^13/3 vs CURN^13/3 | §3; Fig. 2 (numbers taken from the figure's PDF text layer) |
| **CURN^gamma vs IRN** | Fig. 2 prints **10^(12.1 +/- 0.1)**. The abstract says HD (power law) vs independent pulsar noise > 10^14 (about 10^12.1 x 226) | Abstract; Fig. 2 |
| Other Fig. 2 edges | Labels 0.6 +/- 0.2, 0.78 +/- 0.09, 0.48 +/- 0.01 (HD^gamma + dipole / + monopole / + sinusoid relative to HD^gamma), and < 1e-7, < 1e-8 (dipole, monopole vs CURN^gamma). **[UNCERTAIN: which label goes with which edge; the PDF text layer does not say. Read off the rendered figure]** | Fig. 2 |
| Bayesian phase-shift background for HD/CURN BF | p = 1e-3 (5 of 5,000 shifts exceed) | §3; Fig. 3 left |
| Sky scrambles (BF) | p = 1.6e-3 (8 of 5,000) | §3; App. F |

### 2.3 Optimal statistic (frequentist)

| Quantity | Quoted value | Where |
|---|---|---|
| **Noise-marginalised OS S/N** | **5 +/- 1** (over CURN^gamma posteriors), **4 +/- 1** (over CURN^13/3 posteriors), 14 frequencies (mean +/- std over the noise posterior) | §4; Fig. 4 |
| OS p-values | Phase shifts p = 5e-5 (19 of 400,000). CURN^gamma simulations (27,000, MAP hyperparameters) p = 1.8e-4. Generalised chi^2 p = 1.9e-4. Sky scrambles p < 1e-4 | §4; Fig. 3 right; App. F |
| Multiple-component OS (gamma = 13/3, noise-marginalised) | HD only: A^2_HD = 6.8(9) x 10^-30, S/N 4(1). Monopole only: 1.1(1) x 10^-30, S/N 4(1). Dipole only: 1.5(3) x 10^-30, S/N 4(1). HD + monopole: 5.5(8)e-30 / S/N 3.4(8) and 8(1)e-31 / 2.9(8). Full table incl. AIC probabilities in `tab:mc_optstat` | App. G (Multiple-correlation OS), `tab:mc_optstat` |
| **Binned HD correlations** | 2,211 pairs, **15 bins** with equal numbers of pairs. Uses MAP CURN^13/3 noise parameters and accounts for pair covariance (Allen & Romano). chi^2 against HD = **8.1**: p = 0.75 (HD^gamma simulations), 0.92 (canonical chi^2 with 15 dof). 8-20 bins all give p > 0.3 | §4; Fig. 1(c) |
| Reference products | `figure1_data/optstat_ml_gamma4p33.json`, `os_covariance_matix_between_rhos.npy`; `ng15_gwb_fig7_os/figure7_data/correlations_{gamma4p33,variedgamma}_nfreq{14,5}.npz` (noise-marginalised rho_ab, sigma_ab for 20,000 posterior draws x 2,211 pairs; the nfreq14 files are also in the Fig. 4 bundle); tutorial `curn_14f_pl_vg_os.npz` | |
| Own check on released rho_ab/sigma_ab | S/N (mean +/- std over draws): gamma = 13/3, 14f: **4.49 +/- 1.02**, mean A^2 = 6.81e-30 (matches `tab:mc_optstat` HD-only 6.8(9)e-30, S/N 4(1)). Varied gamma, 14f: **4.98 +/- 1.10**. Consistent with the quoted 4 +/- 1 and 5 +/- 1 | own check |

### 2.4 Free spectrum and correlation reconstruction

| Quantity | Quoted value | Where |
|---|---|---|
| **CURN^free / HD^free** (30 freq) | Excess power in bins 1-8 (bin 6 marginal). HD-correlated power in bins 1-5 and 8. None above f_8. Best-fit power law gamma ~ 3.2. gamma = 13/3 overshoots bin 1 and undershoots bin 8. f_2 = 3.95 nHz | §5.2; Fig. 6; Fig. 1(a) |
| Monopole hint | Monopole power only in bin 2 (3.95 nHz) in the CURN^free + HD^free + MP^free + DP^free model | §5.3; Fig. 6 right |
| Spline ORF | 7 knots. Consistent with HD. The joint posterior at the HD zero crossings is consistent with (0,0) within 1 sigma | §3; Fig. 1(d) |
| Reference products | `figure1_data/30fCP_30fiRN_3A_freespec_chain.core`, `SplineORF_{Varied,Fixed}Gamma_NL.core`, `14f_PL_hd_crn.core`; tutorial `presampled_cores/hd_30f_fs.core`, `spline_orf_vg.core`; Ceffyl KDEs (Zenodo 8060824 / 21844115) | |

### 2.5 Likelihood-level oracle (from M0)

The discovery `NG15yr-m2a-chain.feather` stores `logl` for each sample. With the
`oracle` group installed, `scripts/oracle_sanity.py` builds discovery's CURN^gamma
likelihood (67 psr, 30 IRN + 14 common freq, fixed WN, SVD timing model) on the GPU. At 8
random chain samples it reproduces the stored **absolute** `logl`, about 7,973,1xx, to within
the float32 rounding of the stored column (|diff| < 0.2). One GPU evaluation takes about 2 ms.
So we have an end-to-end absolute target for our own likelihood: same inputs, same answer
to <~1e-6 relative. M1 tightened this with float64 discovery and enterprise evaluations: our
likelihood agrees with both to below 4e-7 absolute (`docs/M1_VALIDATION.md`).

---

## 3. Milestones

- **M0 (done)**: environment (JAX CUDA 13 on the RTX 5090, float64 OK), data fetched with
  checksums, PINT loads all 68 (+8 split) pulsars, plan.
- **M1 (done, 2026-10-06; covers the original M1-M3 below)**: PINT ingestion and cache,
  fixed WN, bases, ORFs, JAX CURN/HD likelihood with gradients. It matches discovery and
  enterprise to below 4e-7 absolute (8e6-sized logL) and the production chains' float32
  `logl`. HD value + gradient takes 12 ms (float64) or 7.7 ms (mixed-precision gradient).
  See `docs/M1_VALIDATION.md`. The original sub-plan was:
- *data layer.* PINT -> per-pulsar arrays: residuals, TOA errors, backend flags,
  radio frequencies, positions, design matrix. Store them as our own on-disk format.
  Compare to the discovery/tutorial feathers (`v1p1_de440_pint_bipm2019-*`), which come
  from the same par/tim via enterprise+PINT: residuals to <~1 ns, identical TOA counts and
  flag partitions, M column spans (compare projectors, not raw columns). Parse the WN
  dictionary into our noise parameters.
- *single-pulsar likelihood* in JAX: white noise (EFAC/EQUAD/ECORR), IRN with 30
  freqs, timing-model marginalisation (Woodbury / Schur, float64). Cross-check against
  discovery and enterprise at random parameter points to ~1e-8 relative.
- *PTA likelihoods.* CURN (block-diagonal) and HD (dense 67 x 28 inter-pulsar block)
  on the GPU. Match discovery to float64 precision and the m2a chain `logl` (float32).
  Time per evaluation.
- **M2 (in progress; HD^free closed 2026-10-08 as partially reproduced; covers sampling
  plus the BF/OS parts of M3 below)**: NumPyro NUTS
  for CURN^13/3, CURN^gamma (14 and 5 modes), HD^13/3, HD^gamma (14 and 5 modes, enterprise and
  ICRS positions) and HD^free (30 modes); HD vs CURN Bayes factors (reweighting, reverse
  reweighting, bridge); OS (fixed and noise-marginalised), pair-covariance-aware binned
  correlations; Figs. 1a-c and 4. Fixed-gamma amplitude prior resolved to U[-18,-11]. See
  `docs/M2_RESULTS.md`. HD^free: Fig. 1a partially reproduced: principal peak locations agree; tail occupancies and full posterior convergence remain unestablished (campaign closed after
  three pilot rounds, no further GPU runs; `docs/FS_PILOT.md` Sec. 15 has lessons for M3 and open sampler ideas); several IRN nuisance parameters at R-hat 1.01-1.03 (paper
  criterion < 1.01); BF uncertainty is conditional (estimator spread larger). Not in scope: CURN
  vs IRN BF, multi-component OS, phase-shift / sky-scramble backgrounds. The original M2 plan was:
- *M2: sampling.* CURN^13/3, CURN^gamma, HD^13/3, HD^gamma, CURN^free/HD^free (30f).
  NUTS via numpyro, plus our own sampler if needed. Compare marginals with the released
  chains (quantiles, KS / Wasserstein distance, 2-D (gamma, log10 A) contours).
- **M3: model comparison and OS.** HD vs CURN Bayes factor (target 226 +/- 70 at 14 freq,
  about 965 at 5 freq) by product-space and/or reweighting, CURN vs IRN about 10^12.1. OS
  S/N (5 +/- 1, 4 +/- 1), binned HD (15 bins, chi^2 = 8.1), multi-component OS table.
  Optionally phase-shift / sky-scramble backgrounds (expensive; GPU makes the OS ones
  cheap).
- **M4 (optional)**: EPTA DR2new / PPTA DR3 / InPTA with the same pipeline.

## 4. Known risks and open questions

1. **PINT version.** The par files were written with PINT 0.8.4 and the original analysis
   used PINT 0.9.1; we use PINT 1.1.7. Residuals could differ slightly (clock files, IERS,
   DDK conventions). M1 compares against the feathers built at the time.
2. **ECORR epoch definition** and the **free-spectrum prior scale**, both marked
   [UNCERTAIN] above.
3. **Fig. 2 edge labels** and the **CURN^13/3 amplitude** are not in the text.
4. **Timing-model marginalisation numerics.** M columns span many decades. SVD /
   normalisation is required in float64. Never use float32 for the likelihood.
5. **Bayes factors** need evidence estimation (product space / TI / reweighting). These are
   the most expensive and least portable numbers. Plan to spend GPU time here.
6. The HD vs CURN BF depends strongly on the number of common frequencies (5 vs 14), so
   we must match the setup exactly before comparing.
