# M3b plan: reproducing each PTA's own published GWB result (Stage 1)

M3b is Stage 1 of [`M3_PLAN.md`](M3_PLAN.md) (Sec. 5.2): reproduce each individual PTA's own
published GWB result with our pipeline, as far as the public products allow. M3a delivered the
infrastructure ([`M3A_VALIDATION.md`](M3A_VALIDATION.md): multi-leg container, tempo2-semantics
ingestion, profiles, general multi-block likelihood, gates G1-G9; approved in review round 3 with
E2-C's original all-oracle criterion unmet and E7/E8 open).

**This is a design and pre-registration document. Nothing has been implemented, sampled or run on
a GPU for it.** The only computations behind it are CPU reads of released products (reference-chain
quantiles, effective sample sizes and mode counts; Appendix A). Every compute number is a
**projection** unless it is marked as measured and cited. Tolerances in Sec. 6 are fixed here,
before any M3b run. They change only through a documented revision that is made before the run it
affects and is reviewed.

Notation follows M3_PLAN: **leg** = one PTA's par + tim of one pulsar; **K_a** = sampled GP columns
of pulsar a; **N_c** = number of common-process frequencies; **[UNCERTAIN]** = not pinned down from
papers or released files. "Reference chain" means the PTA's released posterior samples.

---

## 0. Summary

| PTA (release) | Published target (headline) | Reference products | Class we can claim | Order |
|---|---|---|---|---|
| **EPTA DR2new** (GitLab 2911d0e) | HD log10 A = -13.94 (+0.23/-0.48), gamma = 2.71 (+1.18/-0.71) (90 %); BF(HD/CURN) = 60 | full noise model in machine-readable files; released CURN/HD chains with **all** noise parameters (Zenodo 8091568) | **full reproduction** | 1 |
| **PPTA DR3** (GitHub fdbe6eb) | CURN log10 A = -14.50 (+0.14/-0.16), gamma = 3.87 +- 0.36 (68 %) | max-likelihood noise JSONs, 3-sigma prior files, the model builder (`commonNoise.py`), CURN / product-space chains with all ~260 parameters | **conditional implementation check** first (cheap); **full reproduction** only if the compute decision D5 allows it | 2 |
| **MPTA 4.5-yr** | ER CURN log10 A = -14.25 (+0.21/-0.36), gamma = 3.60 (+1.31/-0.89) | par/tim; noise only in rounded paper tables and an unofficial WN dictionary | **approximate cross-check** | 3 |
| **InPTA DR2** | 95 % UL log10 A < -13.47 (gamma = 13/3, uniform-A prior) | par/tim with T2EFAC; RN/DM/chromatic/SW table in the noise paper; no WN values, no chains | **approximate cross-check** | 4 |

Key decisions proposed here (Sec. 8 lists all human decision points):

* **E8 legs (Sec. 4.1).** For per-PTA published-model reproductions, use **tempo2 (libstempo, the
  pinned oracle env) as the primary evaluator of non-NG legs**, as the published analyses did, and
  keep PINT as an independent secondary evaluator whose effect is quantified per PTA as an
  "engine sensitivity". PINT-vs-tempo2 parity becomes a non-blocking research item. This
  re-scopes E8 and needs the user's and the reviewer's agreement (D1).
* **Stage 1 needs neither option C nor E7.** Each PTA has one leg per pulsar. J1022+1001 /
  J0437-4715 admissibility and E7 move to the M3c preparation (Sec. 4.4).
* **Published TOA sets.** Reproductions use the TOAs the PTA analysed, so M3a's clock-coverage
  exclusions (65 EPTA TOAs) and conservative duplicate removals (2 EPTA, 3 PPTA) are **not** applied
  in published profiles. Their effect is a reported robustness check (Sec. 4.2).
* **Sampling (Sec. 5).** NUTS with a fixed dense metric plus the exact frozen block-MH jumps of
  FS_PILOT for weakly constrained (log10 A, gamma) pairs and deterministic-event blocks. HD
  posteriors and BF(HD/CURN) come first from importance reweighting of CURN draws, with PSIS k-hat
  and Kish ESS checks; an HD NUTS run is made only if those checks fail.
* **The PPTA full model is the compute risk.** Its per-pulsar GP blocks reach K_a ~ 1,000-1,400
  columns. The projected cost is about 760x the NG15 per-pulsar cubic work, i.e. ~0.1-1.5 s per
  value+gradient and tens to hundreds of GPU-hours with NUTS (Sec. 5.5). A measured benchmark
  (M3b-0) comes before any commitment. A coefficient-Gibbs kernel is the main alternative lever.
* **Budget (projection).** M3b without the PPTA full model: **~6-50 GPU-h**. The PPTA full model
  adds **~5-60 GPU-h** with a validated Gibbs kernel or **~50-800 GPU-h** with plain NUTS + jumps
  (Sec. 7).

---

## 1. Scope

### 1.1 Three kinds of comparison (never mixed)

From the M3 plan review (round 2, item 6) and M3_PLAN Sec. 5.2:

* **Full reproduction.** Our implementation of the PTA's *published* model (same data, timing
  profile, noise components, sampled and fixed parameters, priors, N_c and grid), compared with
  the released chains under the Sec. 6 criteria. Only EPTA DR2new and PPTA DR3 release enough for
  this.
* **Conditional implementation check.** All non-target noise is fixed at released values. Our
  conditional posterior (or likelihood surface) of the common-process parameters is compared with
  enterprise evaluated on **the same conditional model and inputs**. This validates code, not
  published numbers. A conditional posterior is **not** expected to match the published
  noise-marginalised chains.
* **Approximate cross-check.** Products needed for an exact configuration are missing (MPTA,
  InPTA). We compare against published summaries with loose, pre-declared tolerances. Nothing
  else depends on the outcome.

### 1.2 Ordering

EPTA first: highest value (BF ~60 is the strongest European evidence; the chains contain every
noise parameter) and highest feasibility (67 sampled parameters, K_a <= 416, 45k TOAs). PPTA
second: complete products but a much larger model. Its conditional check is cheap and comes
first, and the full model waits for a compute decision. MPTA and InPTA come last as approximate
cross-checks. They are not prerequisites for Stage 2.

### 1.3 Out of scope for M3b

NG15 (done in M1/M2). Combinations (Stage 2+). Sampled white noise in array runs. BayesEphem
(EPTA and PPTA robustness variants only). Free spectra, binned/spline ORFs, OS and pair-wise
statistics, unless listed as optional in Sec. 2. Sky scrambles and phase shifts. EPTA DR2full and
DR2new+ (DR2new+ contains InPTA DR1; never combine it with InPTA).

---

## 2. Per-PTA specifications

Numbers from papers were checked against the arXiv sources (survey Sec. 3.7-3.10) and, for this
plan, re-checked against the arXiv abstracts or HTML (2306.16214, 2306.16215, 2306.16229,
2412.01153) and the InPTA GWB LaTeX (2608.02808v1). Reference-chain numbers were recomputed from
the released files on 2026-10-09 (Appendix A).

### 2.1 EPTA DR2new: full reproduction

**Targets** (Antoniadis et al. 2023, *EPTA DR2 III: search for GW signals*, A&A 678 A50,
arXiv:2306.16214, DR2new section and summary table; 90 % credible, enterprise):

| ID | quantity | published | reference chain (ours, Appendix A) |
|---|---|---|---|
| E-1 | HD^gamma log10 A | -13.94 (+0.23/-0.48) | -13.935 [-14.418, -13.702] (5/50/95), MCSE(q50) 0.005 |
| E-2 | HD^gamma gamma | 2.71 (+1.18/-0.71) | 2.710 [1.978, 3.894] |
| E-3 | CURN^gamma log10 A | -14.00 (+0.28/-0.77) | -13.997 [-14.766, -13.718] |
| E-4 | CURN^gamma gamma | 2.91 (+1.72/-0.87) | 2.906 [2.038, 4.626] |
| E-5 | gamma = 13/3 log10 A | -14.61 (+0.11/-0.12) | no released 13/3 chain in DR2new **[UNCERTAIN: check the Zenodo record and the GitLab repository again in M3b-0]** |
| E-6 | BF(HD/CURN) | 60 (enterprise); 62 (fortytwo); re-estimates 66, 56, 62 | not derivable from the released chains (separate runs, no evidences) |
| E-7 (optional) | OS (13/3) A^2 = 10.0 (+5.1/-4.9) x 10^-30, S/N 3.5 (+2.4/-1.7) | `os/hd_amp_sn.txt` | |

The released CURN and HD chains (`DR2new/{crn_pl,hd_pl}/chain_1.txt`, 29,990 rows, 67 parameters
+ 4 sampler columns) reproduce the published E-1 to E-4 to the last printed digit (Appendix A).
They are the reference for E-1 to E-4 and for every noise parameter.

**Published-analysis profile.**
* Data: DR2new par/tim (25 pulsars, 45,428 TOAs, 2011.1-2021.5, T_array = 10.33 yr).
* Clock and ephemeris: TT(BIPM2021), DE440, the release's corrected Nancay clock file.
* Units: TCB, evaluated natively by tempo2 under the D1 strategy.
* Timing model: analytically marginalised (`tm_svd=True, tm_marg=True`).
* WN: EFAC + TNEQUAD per `-group`, fixed at `noisefiles/DR2new/*.json`, no ECORR.
* Model code: `EPTA-DR2/scripts_gwb/model_single.py`, i.e. `enterprise_extensions.models.model_general`
  from EPTA's modified fork (tag `EPTADR2-v1.1`, gitlab.in2p3.fr; **not fetched**, decision D6).

**Noise model per pulsar** (released `red_dict.json`, `dm_dict.json`, `chrom_dict.json`; counts
on each pulsar's own span):

| component | where | basis | status in the GWB run |
|---|---|---|---|
| achromatic RN | 8 pulsars (10-99 modes) | Fourier, pulsar span | **sampled** (log10 A, gamma) |
| DM GP | 22 pulsars (11-100 modes) | nu^-2, TempoNest normalisation (`tndm=True`) | **sampled** |
| scattering GP | J1600-3053 (93 modes) | nu^-4 (`chrom_kernel='diag'`) | **sampled** |
| exponential dip | J1713+0747, one event (t0 in [57490, 57530], chromatic index 1) | deterministic | **sampled** (log10 Amp, log10 tau, t0) |
| solar wind | all, deterministic n_earth = 7.9 cm^-3 in the par (J1022 fitted) | timing model | fixed (in par) |
| band/system noise, ECORR | none | - | - |

Sampled parameters: 62 GP + 3 dip + 2 common = **67**, matching the chain columns. The common
process uses N_c = **9** modes (paper) on T_array; `freq_bins/freqs_dr2new.txt` gives the
spacing 3.066e-9 Hz = 1/(10.33 yr). Priors inferred from the chain ranges, to be pinned from the
fork: noise log10 A U(-18, -10), gamma U(0, 7); dip log10 tau U(0, 2.5); common log10 A
U(-18, -11) **[UNCERTAIN: upper bound -11 vs -10]**, gamma U(0, 7).

*Discrepancy recorded:* the GW paper gives dip chromatic indices 4 and 2. The released code
(`dm_expdip_idx=[1,4]`, windows [57490, 57530] and [54650, 54850]) and the noise file key
`expd-1.0_57490_57530` give index **1** for the 2016 event, the only one inside DR2new. We follow
the code, because it produced the reference chain.

**M3a support and gaps.**
* Supported: per-block spans and mode counts (`FourierBlock`), TempoNest DM normalisation, nu^-4
  chromatic blocks, TN EQUAD, namespacing, distinct common and noise grids (prefix identity broken),
  and padding for variable K_a.
* To add: the sampled exponential dip (N3); the EPTA model builder with exact enterprise semantics
  (N1); stage-2 bucketing (N8).
* Largest K_a: J0900-3144, 398 noise columns + 18 common = 416. Sum over pulsars of K_a^3 is
  ~16x NG15's with size buckets and ~124x if every pulsar is padded to 416.

**Comparisons.**
* Conditional implementation check E-C0: noise fixed at the noisefile values, the 141 x 141
  (log10 A, gamma) CURN and HD likelihood surfaces against enterprise built with the same fixed
  noise. CPU only.
* Full reproduction of E-1 to E-6, with E-5 against the paper's interval only (no chain).
* Optional E-7.

### 2.2 PPTA DR3: conditional check first, full reproduction gated

**Targets** (Reardon et al. 2023, *Search for an isotropic GWB with the PPTA DR3*, ApJL 951 L6,
arXiv:2306.16215; 68 % credible). Reference: GitHub fdbe6eb
`analysis_codes/data/all/chains/chain_commonNoise_*_DE440.npy`.

| ID | quantity | published | reference chain (Appendix A) |
|---|---|---|---|
| P-1 | CURN^gamma log10 A | -14.50 (+0.14/-0.16) | -14.496 (+0.138/-0.159), 2,978 draws, bulk ESS 549 |
| P-2 | CURN^gamma gamma | 3.87 +- 0.36 | 3.869 (+0.359/-0.364), ESS 619 |
| P-3 | CURN 13/3 log10 A | -14.69 +- 0.05 (A = 2.04 (+0.25/-0.22) x 10^-15) | -14.681 (+0.047/-0.054), ESS 1,569 |
| P-4 | HD^gamma log10 A, gamma | -14.51 (+0.18/-0.20); 3.87 +- 0.47 | HD-active product-space draws: -14.514 (+0.177/-0.195); 3.906 (+0.483/-0.462), ESS ~350 (approximate: subsequence of a product-space chain) |
| P-5 | HD 13/3 log10 A | -14.68 +- 0.06 | -14.675 (+0.060/-0.059) |
| P-6 | BF(HD/CURN) | ~1.5 (free gamma), ~2 (13/3) | product-space count ratios 1.85 / 1.69 **[UNCERTAIN: assumes equal model weights (HyperModel default); not the paper's numbers]** |
| P-7 (diagnostic) | "basic" noise model CURN | -14.08 +- 0.06, gamma 2.9 +- 0.2 | no chain |

**Published-analysis profile.**
* Data: 30 pulsars (J1741+1351 and J1824-2452A excluded), GitHub par/tim, 112,556 TOAs
  (canonical PINT count), 2004.1-2022.2.
* Clock: TT(BIPM2020) with the release's `pks2gps.clk`.
* Ephemeris: **DE440** (the GW paper's override of the par's DE436).
* Units: TCB (implicit; no UNITS line).
* Timing model: marginalised (`MarginalizingTimingModel(use_svd=True)`).
* WN fixed at `noiseFiles_maxlike/*_singlePsrNoise_sw_nesw0_noise.json`: EFAC + TNEQUAD per
  backend, plus ECORR per band split (40/20/10 cm, UWL and non-UWL separately), a global ECORR
  over UWL and a global ECORR over non-UWL TOAs (`global_ecorr`), and group ECORRs. These are
  overlapping selections. ECORR epochs follow enterprise's quantisation (dt = 1 s, nmin = 2) per
  selection.

**Noise model** (from `commonNoise.py` and `ppta_dr3_utils.py`; spans T_p are per pulsar, modes =
int(T_p / cadence)):

| component | where | basis | status |
|---|---|---|---|
| achromatic RN | all 30 | T_p/240 d | **sampled**, informed prior |
| DM GP | all 30 | nu^-2, T_p/60 d | **sampled**, informed prior |
| chromatic GP | 7 (J0437, J0613, J1017, J1045, J1600, J1643, J1939) | nu^-4, T_p/240 d | **sampled** |
| band noise low (<= 960 MHz) | 9; J0437 also mid and high | T_p/60 d, frequency row mask | **sampled** |
| HF achromatic | 8 (J0437, J1017, J1022, J1600, J1713, J1744, J1909, J2241) | T_p/30 d | **sampled** |
| group noise | pulsars in `psr_groupnoise_dict_dict['all']` | flag mask, f_max = 1/30 d on the masked span **[UNCERTAIN: span of the masked TOAs vs T_p; to be pinned against enterprise]** | **sampled**, one shared prior per pulsar |
| deterministic SW | n_earth sampled in 17 pulsars (prior from the 3-sigma file, within [0, 20]); fixed at 4 cm^-3 otherwise | solar-wind geometry, linear in n_earth | **sampled** where listed |
| SW GP | 10 (J0437, J0711, J0900, J1024, J1643, J1713, J1730, J1744, J1909, J2145) | `createfourierdesignmatrix_solar_dm`, T_p/60 d | **sampled** |
| exponential dips | J1713 (2: t0 in [54650, 54850], [57400, 57600]; index U(1, 3), U(0, 2)), J0437 (1, index U(-1, 2)), J1643 (1, U(-2, 0)), J2145 (1, U(-2, 2)) | deterministic, chromatic | **sampled** (log10 Amp U(-10, -2), log10 tau U(0, 2.5), t0, index) |
| annual DM sinusoid | J0613 | deterministic nu^-2 | **sampled** (log10 Amp, phase) |
| Gaussian DM event | J1603 (epoch U(53800, 54000)) | deterministic nu^-2 | **sampled** |
| J1600 20-cm bump, achromatic quadratic | defined, **off** in the GWB run (`bump_1600 = False`, `do_quad` unset) | - | not modelled |

The common process uses N_c = int(T/240 d) modes on the array span. Our 30-pulsar span is
6,604.7 d, so **N_c = 27**; the paper prints 28 (survey [UNCERTAIN]). The released builder fixes 27
from the data, and we follow it. Priors: common log10 A U(-18, -11), gamma U(0, 7). The fixed-gamma
runs use **gamma = 4.333** (`gamma_val=4.333`), not 13/3 exactly, and we replicate that. Informed
noise priors: log10 A in [max(-18, q_0.15% - 2), min(-11, q_99.85% + 1)], gamma in
[max(0, q - 0.5), min(7, q + 0.5)], from `noiseFiles/3sig` (`get_informed_rednoise_priors`).
About 260 sampled parameters; the CURN^gamma chain has 262 including an inert `nmodel`.

**M3a support and gaps.**
* Supported: overlapping ECORR (L3), TN EQUAD, band and flag row masks, nu^-4 blocks, per-block
  spans.
* To add: informed-prior configuration (N9); SW GP basis (N4); deterministic SW with a finite
  prior (N5); sampled dips with free chromatic index, annual DM and Gaussian DM (N3);
  enterprise-exact group-noise span semantics (N1); bucketed stage 2 at K_a up to ~1,400 (N8).
* Projected size: K_a reaches 1,042-1,388 for J1713, J1909, J1600 and J0437 before group noise.
  Sum of K_a^3 is ~760x NG15's.

**Comparisons.**
* **P-C0 conditional implementation check (first deliverable):** all noise hyperparameters,
  including dips, SW and n_earth, fixed at the max-likelihood JSON values; CURN/HD surfaces in
  (log10 A, gamma) and the 13/3 amplitude, ours vs enterprise built with the released
  `ppta_dr3_models.py` functions on the same inputs. A small GPU run, or a CPU grid.
* **Full reproduction of P-1 to P-6** only after decision D5 (Sec. 5.5).
* P-7 is an optional diagnostic: it exercises the same code with fewer blocks.

### 2.3 MPTA 4.5-yr: approximate cross-check

**Targets** (Miles et al. 2025, *MPTA 4.5-yr GW search*, MNRAS, arXiv:2412.01153; median and 68 %):

| ID | quantity | published |
|---|---|---|
| M-1 | DATA CURN log10 A, gamma | -14.25 (+0.21/-0.34); 3.52 (+1.12/-0.90) |
| M-2 | ER CURN log10 A, gamma | -14.25 (+0.21/-0.36); 3.60 (+1.31/-0.89) |
| M-3 | DATA HD log10 A, gamma | -14.28 (+0.23/-0.30); 4.50 (+1.00/-0.93) |
| M-4 (info) | OS (DATA, fixed parameters) A^2 = (5.7 +- 1.2) x 10^-29, i.e. the abstract's h_c,yr = 7.5e-15 at alpha = -0.26 and 4.8e-15 at alpha = -2/3 | not a Bayesian posterior; not compared |

**Profile.**
* 83 pulsars (J1825-0319 loads under tempo2; it is quarantined in the PINT path), 245,907 TOAs,
  single system KAT_MKBF.
* Clock: TT(BIPM2020) from the pars as the default; BIPM2022 (the paper's) as a variant (D7).
* Ephemeris DE440; WN fixed (paper).
* Noise (Miles et al. 2025, arXiv:2412.01148, tables): evidence-selected EFAC/TNEQUAD/ECORR; RN
  (12 pulsars; all pulsars in ER); DM GP; chromatic GP with fitted or fixed (4) beta; solar wind
  n_earth and SW GP; Gaussian chromatic events (15); annual chromatic terms (8).
* No machine-readable release: tables are rounded and MattTMiles/MPTAGW `example_noise.json` is
  unofficial.
* N_c and per-pulsar T for the Bayesian common process **[UNCERTAIN]**.

**What we do.** A conditional approximation: WN from the tables (the unofficial JSON as a cross-check
of transcription); RN/DM/chromatic/SW/event parameters fixed at the tabulated MAP values, with
beta fixed at its table value (no per-call basis rebuild); CURN^gamma and HD^gamma on the array
span with N_c from the MPTAGW scripts (or 30 if unresolvable; flagged). Comparing a fixed-noise
posterior with noise-marginalised published intervals is approximate by construction.

**M3a support:** the components above, with fixed hyperparameters and absorbed in stage 1 (cheap).
**To add:** Gaussian chromatic and annual chromatic deterministic terms as *fixed* residual
corrections (N3 fixed mode), and the SW GP basis (N4).

### 2.4 InPTA DR2: approximate cross-check

**Targets** (InPTA DR2 GWB paper, arXiv:2608.02808v1, abstract and Table "solar cuts"):

| ID | quantity | published |
|---|---|---|
| I-1 | 95 % UL at gamma = 13/3, uniform-in-A prior (LinearExp by reweighting a log-uniform chain) | log10 A < -13.47 (A < 3.4 x 10^-14) |
| I-2 | the same, log-uniform prior quantile | -13.55 |
| I-3 | CURN^gamma (broad, prior-dominated) | log10 A = -13.71 (+1.06/-3.28), gamma = 2.98 (+3.62/-2.70) |
| I-4 (info) | Savage-Dickey BF(CURN/none) | 2.5 |

**Profile.**
* 27 pulsars, 83,120 TOAs, TT(BIPM2023), DE440, TCB.
* DMX removed, DM + DM1 + DM2 (`InPTA.DR2.NA` pars).
* Noise from the noise paper (arXiv:2512.20455) table: ARN, DMN, free-chromatic noise with sampled
  chi, deterministic n_earth (+ derivative for J1909).
* Single-pulsar noise parameters **fixed** in the UL run (paper Sec. "upper limit").
* WN: only `T2EFAC` per group in the release pars. No EQUAD/ECORR values are public (**known
  gap**: our WN will under-model the white noise where EQUAD/ECORR were selected).

**What we do.** Fixed noise from the table (chi fixed at its tabulated value), EQUAD = ECORR = 0,
UL by the paper's reweighting recipe, and the Kish ESS reported.

**M3a support:** all fixed components. **To add:** deterministic SW with n_earth-dot (N5 fixed
mode), and free-chromatic blocks at a fixed chi (supported by `FourierBlock` idx).

---

## 3. New components (implementation list for M3b-0)

| # | component | needed for | design | validation |
|---|---|---|---|---|
| N1 | Per-PTA model builders reproducing the published enterprise semantics: block spans (pulsar span vs masked-subset span), `int(T/cadence)` mode counts, enterprise ECORR quantisation per selection, `tndm`, `gamma_val=4.333` | EPTA, PPTA | config-driven (`configs/m3b/<pta>_model.json`), generated from the released dicts/JSONs, never hand-typed | block-by-block column-space and prior comparison with enterprise `PTA` objects (sin < 1e-10 per block) |
| N2 | tempo2 evaluator path: per-leg residuals, design matrix (fit flags as tempo2), errors, freqs, flags from libstempo in the oracle env, written to our leg format with provenance | all non-NG reproductions | `scripts/t2py` export; the container already accepts arbitrary per-leg arrays (M3A open issue 1) | gate T1 (Sec. 4.1) |
| N3 | Parameter-dependent deterministic residual terms: exponential dip (free index), Gaussian DM event, annual DM sinusoid; also a fixed mode (residual correction in stage 1) | EPTA, PPTA (sampled); MPTA (fixed) | **affine residual update**: with all fixed blocks and WN absorbed in stage 1, c(theta) = c_0 - G d(theta) and the orthogonal-complement term s_perp(theta) = norm(Q_perp^T W (r - d(theta)))^2, with G and Q_perp^T W precomputed. Cost O(N_toa x K_a) per call for affected pulsars only | dense oracle with a dip; enterprise; gradient vs FD |
| N4 | SW GP block (`createfourierdesignmatrix_solar_dm` geometry x Fourier) | PPTA, MPTA | new basis in `gp.FourierBlock` | column equality vs enterprise basis |
| N5 | Deterministic solar wind with finite prior: n_earth enters linearly, so it is a sampled scalar on a fixed column (not a flat timing column); n_earth-dot variant | PPTA (sampled), InPTA/MPTA (fixed) | column s_SW(t, nu) precomputed; contributes like N3 with d = n_earth s_SW | vs enterprise `solar_wind` |
| N6 | Free chromatic index at fixed value | InPTA, MPTA | existing `FourierBlock` idx | existing tests |
| N7 | Product-space-free BF tooling: reweighting with PSIS k-hat, Kish ESS, block bootstrap; bridge when both chains exist | EPTA, PPTA | extend `ptagwb.evidence` | synthetic tests with known BF |
| N8 | Stage-2 size buckets for K_a up to ~1,400 (pad within buckets, not to the global max) and a memory model | EPTA, PPTA | L6 of M3_PLAN | padding invariance (G5-style, 1e-10) |
| N9 | Per-parameter prior boxes (informed PPTA priors; EPTA fork priors) | EPTA, PPTA | `ModelSpec` prior table from config | prior recovery under a flat likelihood |
| N10 | Hybrid kernel blocks for M3 parameter names (pairs, dip blocks) and the generalised fail-closed gate | all sampled runs | reuse `ptagwb.hybrid`, `diagnostics.freespec_gate` logic | existing invariance tests + new block shapes |
| N11 | Reference-product loaders with frozen identity (sha256, burn-in, thinning) | EPTA, PPTA | `scripts/m3b_reference.py` | reproduces Appendix A |
| N12 | Coefficient-Gibbs kernel (candidate; only if D5 selects it): sample Fourier coefficients b given hyperparameters (one Cholesky per pulsar), then hyperparameters given b (per-block conditionals; MH for deterministic terms), exact by construction (van Haasteren & Vallisneri 2014, PRD 90, 104012) | PPTA full model | separate design note before implementation | invariance on toys + agreement with NUTS on EPTA |

Not added in M3b: sampled chromatic index in a GP basis (per-call basis rebuild). The PPTA GW run
has none, since its chromatic GP uses idx = 4. InPTA/MPTA fix it.

---

## 4. Remaining M3a obligations that block Stage 1

### 4.1 E8 tempo2-parity legs: strategy

**The problem.** E8 (G3 < 1 ns and < 0.01 sigma; G4 sin < 1e-6) fails for every non-NG validation
leg. Most legs have a 1-5 ns floor; PPTA J1022 and J0437 (DDK) are at ~30 ns. The named blocking
legs:
* EPTA and PPTA J1600-3053 (T2 -> DD / DDH);
* PPTA J1713+0747 (T2 -> DDK);
* PPTA J2241-5236 (PB + FB series; G3 0.93 ns but G4 3.5e-5);
* the DDK (10) and DDH (21) classes in general.

The post-hoc likelihood impact reaches 34 nats (PPTA J1909) and 58 nats (PPTA J0437) in an RN+DM
model, almost all of it from the residuals rather than the design matrix (M3A Sec. 4).

**Options.**

| option | what | pro | con |
|---|---|---|---|
| (a) fix PINT parity | find and fix each PINT-vs-tempo2 difference: DDK Kopeikin terms, DDH, FB series, the TCB -> TDB "approximate" conversion, unsupported DILATEFREQ/TIMEEPH | keeps one engine; physically clean | open-ended; the 1-5 ns floor on *ordinary* legs has no identified cause (TCB -> TDB is a suspect, unproven); blocks Stage 1 indefinitely |
| (b) tempo2 as evaluator for non-NG legs | residuals and design matrices from tempo2/libstempo (pinned oracle env), as the EPTA, PPTA, MPTA and InPTA analyses did, and as YA did for non-NG legs | matches the evaluator of the published target, so the engine difference stops being a reproduction error; works for signed H3 (MPTA J1825), DDK, DDH and FB series as released | tempo2 is also the G3 oracle, so it is no longer an independent check of the timing evaluation; tempo2 2026.04.1 is not the 2023 versions the PTAs used |
| (c) quarantine with quantified impact | drop failing legs; report a dropout shift | simple | E8 fails for *all* non-NG legs, so quarantine would empty the PTA; for the named legs alone it changes the target data set |

**Decision proposed (D1): (b) for published-model reproductions, with PINT kept as an independent
secondary evaluator and (c) as the fallback for any leg that fails gate T1.** Concretely:

* **Evaluator profile `published-tempo2-v1`.** Non-NG legs are evaluated by tempo2 2026.04.1 /
  libstempo 2.5.1 (oracle env, hashes pinned):
  * with the leg's native units (TCB) and published clock/ephemeris profile (Sec. 2);
  * from our canonical flat tim (G1-identical TOAs);
  * with tempo2's site coordinates, which are the release-reproduction convention already adopted
    in M3a.
  * Residuals are pre-fit at the par values, as enterprise's libstempo `Pulsar` does. The design
    matrix is libstempo's `designmatrix()` for the par's fit flags.
  * NG15 stays on PINT (its published profile).
* **Gate T1 (exactness of the export).** For every leg in a reproduction, our exported arrays equal
  enterprise `Pulsar(par, tim, timing_package='tempo2')` arrays built in the same env from the
  same files: residuals and errors |diff| <= 1e-12 s; design matrix column space sin <= 1e-10;
  flags and frequencies identical. A failing leg is quarantined (option c) with a dropout shift
  reported.
* **Gate T2 (engine sensitivity, reported, not gating).** For each PTA, the conditional
  posterior of the common process (as in M3A Sec. 12b: fixed released noise, 141 x 141 grids) is
  computed with the tempo2 and the PINT evaluator for every leg that loads in PINT. Reported: shift
  of each Sec. 6 quantity in units of sigma68 and w90, per PTA and with the E8-named legs swapped
  one at a time. **Pre-declared interpretation:** a shift <= 0.1 sigma68 means the engine choice is
  immaterial at the posterior level for that PTA; a larger shift is reported as an
  evaluator-dependence of the published result, and the tempo2 result stays the reproduction.
* **Gate T3 (sanity against the release).** Per leg, the tempo2 post-fit weighted rms against the
  par's TRES (screen only, as in the survey). The legs the survey flagged as "residual excess"
  (J1600 EPTA/PPTA, PPTA J1713, J2241) are explained or listed.
* **E8 is re-scoped, not declared passed.** G3/G4 stay in the suite as PINT-vs-tempo2 diagnostics
  with the strict xfail kept. Statements about *PINT* timing parity remain restricted. PINT-parity
  work (DDK/DDH formulae, TCB -> TDB) continues as a non-blocking research item.
* **Version risk.** tempo2 2026.04.1 differs from the 2023 tempo2 builds inside the PTAs'
  containers (clock/EOP files, bug fixes). No released residuals exist to check against (EPTA's
  whitened residuals are PDF plots). The published chains themselves are the end-to-end check
  (Sec. 6). A mismatch is traced first to this item.

This keeps the oracle structure honest. G1 (TOA identity) stays independent of tempo2: our reader
vs PINT vs libstempo. G5 stays independent: our likelihood vs the long-double arbiter vs
enterprise, all fed the same arrays. What we give up is an independent check that tempo2's delays
are physically right, which a reproduction of tempo2-based published results does not need.

### 4.2 Published TOA sets vs M3a removals

M3a's default ingestion excludes 65 EPTA TOAs without nanosecond clock coverage
(`exclude-uncovered`; J1600-3053 has 54) and removes 7 within-leg same-channel pairs (EPTA 2,
PPTA 3, NG15 2) under a conservative project policy. The published analyses used those TOAs.

**Rule for published profiles:** `clock_coverage = "keep"` (tempo2's own interpolation, which is
what the PTA's tempo2 did) and **no duplicate removal**. Both choices are recorded in provenance.
The M3a defaults are run as a robustness variant (conditional posterior shift, as in T2). For the
combined profile (Stage 2) the M3a defaults stay.

### 4.3 Band-overlap covariance/pruning inventory

Status from M3a: simultaneous recordings in overlapping bands by different backends (same photons,
different channels) are correlated, not identical. There are 16,186 such PPTA channel pairs
(Medusa/UWL vs PDFB4 in 20 pulsars, CASPSR vs Medusa in 13), 235 in EPTA and 19 in NG15.

**Stage-1 position.** The published analyses kept these data and modelled them with their own
noise models. A *reproduction* must keep them too, so this item does not block Stage 1
acceptance. It is a required M3b deliverable for Stage 2 and for model adequacy:

1. **Inventory** (CPU, `scripts/m3b_band_overlap.py`). Per pulsar and system pair:
   * the number of pairs and their share of the pulsar's total TOA weight;
   * channel overlap fraction;
   * whether the published model already correlates them. For PPTA, check whether both TOAs fall
     in the same epoch of the global non-UWL / UWL ECORR under enterprise's dt = 1 s quantisation.
     Simultaneous PDFB4/CASPSR TOAs with different timestamps are probably **not** grouped
     [UNCERTAIN]. Also check whether both lie in one band-ECORR selection.
2. **Sensitivity (conditional, reported):** the PPTA and EPTA conditional posteriors (T2 set-up)
   with the overlapping-band partner pruned (keep the newer backend, EPTA's rule) vs kept. Shift in
   sigma68 units.
3. **Decision for Stage 2** (not M3b): pruning rule or an explicit cross-backend correlated-noise
   term, chosen from 1-2.

### 4.4 Other open M3a items, and why they do not block Stage 1

* **E7 (G6 reference-swap linearity) and option C.** Each per-PTA reproduction has exactly one leg
  per pulsar, so no reference model is copied. E7 matters for M3c only. Its remedy (re-linearise at
  a joint fit of all legs, iterated) is scheduled in the M3c preparation.
* **J1022+1001 / J0437-4715 admissibility for later combination.** Stage 1 does not need option C.
  Plan for M3c-prep (not M3b):
  1. align binary models across legs before any copy (J1022: PPTA DDH with frozen H3/STIG vs NG15
     DD; J0437: DDK KIN/KOM free in PPTA, frozen in MPTA);
  2. local or per-leg DM offsets (shared DM wraps both pulsars);
  3. joint re-linearisation;
  4. re-run admissibility (threshold 0.1 whitened, unchanged).

  Under the D1 strategy the value copy happens in tempo2 par space for non-NG legs, as in YA.
  Option B remains available for any multi-leg check before then.
* **E2-C (original all-oracle criterion).** It stays on record. For M3b the likelihood gate per
  PTA (G5-PTA, Sec. 5.3) is the independent long-double arbiter, extended to N3/N5 terms (shape
  <= 1e-6 nats, gradient <= 1e-8 relative), plus enterprise on the PTA's own model code. Enterprise
  is reported against a pre-declared budget of 1e-4 nats shape at > 40k TOAs (M3a measured up to
  7.6e-5 at 87k).
* **Signed H3 (MPTA J1825-0319).** Evaluated by tempo2 under D1. If T1 fails, it is quarantined for
  the MPTA cross-check (1 of 83 pulsars).

---

## 5. Sampling strategy

### 5.1 Kernel per model

| model | dims | kernel | why |
|---|---|---|---|
| EPTA CURN^gamma, CURN 13/3 | 67 / 66 | NUTS (fixed dense metric from a pilot) + frozen block-MH jumps for (log10 A, gamma) pairs whose reference marginal has >= 0.5 % mass within 1 dex of the lower prior bound, and a 3-D dip block | the reference chain shows RN/DM amplitudes reaching -18.0 (shelves at the prior floor). FS_PILOT showed plain NUTS trapping on peak + shelf targets (CURN^free A/B: max R-hat 1.470 -> 1.007 with jumps) |
| EPTA HD^gamma | 67 | **reweighting of CURN draws first** (Sec. 5.4); NUTS + jumps only if the overlap checks fail | ln BF ~ 4.1; overlap likely adequate, to be verified |
| PPTA P-C0 (conditional) | 2 / 1 | deterministic grid (141 x 141; 1,401 at 4.333) | fixed noise |
| PPTA full CURN^gamma, 13/3 | ~262 | decision D5: (i) NUTS + jumps; (ii) N12 coefficient-Gibbs; (iii) not run | Sec. 5.5 |
| PPTA HD | ~262 | reweighting of CURN draws (ln BF ~ 0.4-0.7, so near-identical posteriors are expected) | |
| MPTA, InPTA | 2 / 1 | grid, or NUTS on 2 parameters | fixed noise |

Jump proposals follow FS_PILOT Sec. 8: 20 % prior box + 80 % equal-mass histogram, exact density,
MH ratio in physical coordinates. They are **fitted from our own pilot draws only** (never the
reference chain under test) and committed before the production run (frozen file + sha256).

### 5.2 Warmup, initialisation, metric (FS_PILOT and PERF lessons)

* Never start dual averaging at step 1.0. Start at a step size measured in the pilot. The v2
  pilots saturated at tree depth 10 from a poor start, and the CURN A/B warmup cost 1.3-1.6x a
  sampling iteration when started at a workable step.
* Fixed dense metric from the pilot's draws (M2 recipe). No windowed adaptation in production.
* Inits: overdispersed, from our pilot draws with independently randomised shelf/peak region starts
  (`init_rho_low_frac`-style, generalised to noise pairs). Never from the reference chain. PPTA's
  own practice of initialising from single-pulsar chains is allowed for *pilots* only.
* 4-8 vectorised chains. Lockstep loss and step-size landing are measured and reported (PERF
  Sec. 4).
* Every run has a committed config, a recorded SHA and a pre-set abort rule (projected cost > 1.5x
  its allocation: stop and report).

### 5.3 Pre-registered convergence gate (fail-closed)

`ptagwb.diagnostics` generalised from `freespec_gate` (N10). A run gets PASS only if **all** of the
following hold; otherwise FAIL or INCONCLUSIVE, never a silent pass:

1. **Every sampled parameter**: rank-normalised split R-hat < 1.01; bulk and tail ESS >= 400.
2. **Target parameters** (common log10 A, gamma): bulk and tail ESS >= 1,000, and MCSE of every
   Sec. 6 quantile <= 0.05 sigma68_ref, so that MC noise uses at most a third of the tolerance.
3. **Region transport.** For every (parameter, region) declared relevant, the declaration is frozen
   from the reference chain with its sha256 and burn-in before the run. A region is relevant if the
   reference puts >= 0.5 % of its mass below log10 A_min + 1 (shelf) or in the complementary
   peak. Each relevant region needs >= 10 entries and >= 10 exits pooled, in >= 2 chains, with the
   longest sojourn <= 50 % of its draws and occupancy MCSE <= 0.01.
4. **Zero divergences** after warmup; any divergence makes the run INCONCLUSIVE pending
   explanation.
5. **No undeclared configuration drift**: dirty files none, run metadata matching the config.
6. **G5-PTA passed** for the exact model and SHA, before the run.

The EPTA and PPTA reference chains contain every noise parameter, so relevance can be declared
from them for all parameters, as was done for Fig. 1a.

### 5.4 HD posteriors and Bayes factors: overlap diagnostics before anything expensive

Order, cheapest first, stopping when a step passes:

1. **CURN -> HD reweighting.** w_i = L_HD(x_i) / L_CURN(x_i) at the CURN draws (identical noise
   parameters, priors and N_c). BF = mean(w). HD posterior = CURN draws weighted by w.
   * Diagnostics: Kish ESS = (sum w)^2 / sum w^2; PSIS Pareto k-hat of the log-weights; a block
     bootstrap over chains and batches for the SE of ln BF.
   * **Accept** if k-hat < 0.7 **and** Kish ESS >= 1,000 (posterior use) or >= 200 (BF only).
   * Cost: one HD *value* per CURN draw (value only, no gradient).
2. If 1 fails: **HD NUTS run** with the same gate, then reverse reweighting (HD -> CURN) and
   bridge sampling with both chains. Their spread is reported as part of the estimator
   uncertainty, as in M2.
3. If the estimators disagree beyond their SEs: an evidence ladder (stepping stones along
   Gamma_lambda = (1 - lambda) I + lambda Gamma_HD, as M3_PLAN L14 describes). Needs human
   approval (D8).

No nested sampling or product-space runs are planned. PERF Sec. 4 measured them as expensive, and
the M2 reweighting/bridge estimators sufficed on NG15.

### 5.5 Cost projections (labelled projections; to be replaced by the M3b-0 benchmark)

**Measured anchors** (RTX 5090, float64; PERF Sec. 2-4, M2_RESULTS Sec. 4):
* NG15 HD value+gradient: 15.0 ms (B = 1) and 5.8 ms per chain (B = 16), of which 3.5 ms is the
  per-pulsar stage (67 pulsars, K = 60, 0.14 GF at 0.04 TFLOP/s).
* NG15 CURN: 3.6 ms (B = 1).
* M2 CURN^gamma (4 x 1500 draws): 8.3e5 chain-gradients in 25 min. HD^13/3 (4 x 500): 3.0e5
  chain-gradients in 61 min.

**Projection model.** Per-call cost is roughly (per-pulsar stage) + (core). The per-pulsar stage is
cubic in K_a: square-root absorption of a K_a x K_a block, ~3.3 K_a^3 flops forward and ~2x more
for the gradient. The core is cubic in N_psr x 2 N_c for HD and negligible for CURN. Efficiency is
bracketed between the measured small-block rate (0.04 TFLOP/s) and the measured batched-Cholesky
rate (~1.3 TFLOP/s).

| model | sum K_a^3 / NG15 | core dim (HD) | projected value+grad, B = 1 | chain-gradients for the gate (projection) | projected GPU-h per production run |
|---|---|---|---|---|---|
| EPTA CURN^gamma (bucketed) | ~16 | - | 10-60 ms | 1-3 x 10^6 | **~3-40** (central ~5) |
| EPTA HD reweighting, 2 x 10^4 draws | ~16 | 450 | 5-30 ms (value) | - | **< 0.2** |
| PPTA P-C0 grids (fixed noise; stage 1 absorbs everything) | - | 1,620 | ~2-10 ms (value) | 2 x 10^4 grid points | **< 0.1** (or CPU) |
| PPTA full CURN^gamma, NUTS + jumps | ~760 (before group noise) | - | 0.1-1.5 s | 2-5 x 10^6 | **~50-800** |
| PPTA full, coefficient-Gibbs (N12, unvalidated) | one Cholesky per pulsar per sweep, ~0.3 K_a^3 | - | 5-40 ms per sweep | 10^5-10^6 sweeps | **~5-60** |
| MPTA / InPTA conditional | fixed noise | 82 x 2N_c | ~2-20 ms (value) | grid | **< 0.5 each** |

Assumptions and caveats:
* The chain-gradient counts assume NUTS tree sizes like M2 (88-128 leapfrog steps). The stricter
  all-parameter gate needs about 1.5-3x M2's draws, and jumps add ~10 %.
* Lockstep loss of 10-30 % is included in the range. Memory for batched PPTA blocks (~16 MB per
  1,400^2 block per chain) is projected to fit, but is not yet measured.
* The PPTA-NUTS range is wide because both the efficiency and the step count at ~260 dimensions are
  unknown. It is the main reason for D5.
* The Gibbs row assumes autocorrelation times of 10^2-10^3 sweeps for weakly constrained
  amplitudes. That is not measured on our data; it is a literature-informed guess.

**M3b-0 benchmark (first GPU use, D3, <= 1 GPU-h).** EPTA and PPTA full-model value+gradient and
value-only throughput, and peak memory, at B in {1, 4, 8}, production and bucketed kernels, plus 50
fixed-step NUTS transitions per model to measure tree sizes. The table above is then rewritten
with measured per-call costs. Ranges on chain-gradient counts stay ranges until a pilot exists.

---

## 6. Pre-registered acceptance criteria

### 6.1 General rules (all PTAs)

* **Quantities.** For EPTA the published 90 % intervals (q05, q50, q95); for PPTA the 68 %
  intervals (q16, q50, q84). All are computed from our draws, with the reference values recomputed
  from the reference chain (Appendix A), not from the rounded paper numbers.
* **Definitions.** Delta = q_ours - q_ref. s = sqrt(MCSE_ours^2 + MCSE_ref^2), using quantile MCSE
  (Vehtari et al. 2021), as in M2. sigma68_ref = (q84 - q16)/2 of the reference.
* **Tier A, agreement within MC error:** |Delta| <= 3.5 s for every listed quantile (the same
  threshold as the M2 / Fig. 1a agreement rule).
* **Tier B, equivalence within a practical margin:**
  * medians: |Delta| <= 0.15 sigma68_ref;
  * outer quantiles: |Delta| <= 0.25 sigma68_ref.
  * The margins are motivated by the measured posterior-level effect of evaluator conventions
    (NHARMS: <= 0.10 sigma68, M3A Sec. 12b) plus software-version differences. They are fixed here
    and confirmed by the user (D2) before any run.
* **Verdicts per PTA:**
  * **REPRODUCED** = convergence gate PASS + G5-PTA PASS + T1 PASS + Tier A and Tier B for all
    headline quantities.
  * **REPRODUCED WITHIN TOLERANCE** = as REPRODUCED, but Tier A fails while Tier B passes. This is
    plausible because tempo2 versions differ.
  * **NOT REPRODUCED** = Tier B fails for any headline quantity. The discrepancy is traced (order:
    evaluator version, TOA set, model-builder semantics, priors, N_c) and reported. No retuning of
    tolerances.
  * **INCONCLUSIVE** = convergence gate not passed, or MCSE above the bound.
* **Reported, not gating:**
  * 2-D (log10 A, gamma) KS and energy distances;
  * per-noise-parameter z for all ~65 (EPTA) or ~260 (PPTA) parameters (count of |z| > 3.5
    reported, with the expected number under independence);
  * T2 engine sensitivity;
  * the M3a-default TOA-set variant;
  * the band-overlap sensitivity.

### 6.2 EPTA DR2new

| ID | quantity | reference q05 / q50 / q95 | sigma68_ref | Tier B margin (median / tails) | role |
|---|---|---|---|---|---|
| E-1 | HD log10 A | -14.418 / -13.935 / -13.702 | 0.183 | 0.027 / 0.046 | headline |
| E-2 | HD gamma | 1.978 / 2.710 / 3.894 | 0.525 | 0.079 / 0.131 | headline |
| E-3 | CURN log10 A | -14.766 / -13.997 / -13.718 | 0.254 | 0.038 / 0.064 | headline |
| E-4 | CURN gamma | 2.038 / 2.906 / 4.626 | 0.669 | 0.100 / 0.167 | headline |
| E-5 | 13/3 log10 A | paper -14.61 (+0.11/-0.12) | ~0.07 | median within +-0.035 of -14.61; 90 % bounds within +-0.03 of the paper's | headline (no chain: rounding of 0.01 is the reference uncertainty) |
| E-6 | ln BF(HD/CURN) | ln 60 = 4.09; published estimator range [ln 56, ln 66] = [4.03, 4.19] | - | \|ln BF_ours - 4.09\| <= 0.30 + 2 SE_ours, with the Sec. 5.4 overlap checks passed | headline |

The E-6 margin of 0.30 nats is ~4x the half-width of EPTA's own estimator range. It is fixed here,
before any run. If the overlap checks fail, E-6 is INCONCLUSIVE until step 2 of Sec. 5.4 is done.

### 6.3 PPTA DR3

Full reproduction (only if D5 authorises it):

| ID | quantity | reference q16 / q50 / q84 | sigma68_ref | Tier B margin (median / tails) |
|---|---|---|---|---|
| P-1 | CURN log10 A | -14.655 / -14.496 / -14.358 | 0.149 | 0.022 / 0.037 |
| P-2 | CURN gamma | 3.505 / 3.869 / 4.228 | 0.362 | 0.054 / 0.090 |
| P-3 | CURN 4.333 log10 A | -14.735 / -14.681 / -14.634 | 0.051 | 0.008 / 0.013 |
| P-4 | HD log10 A, gamma (reweighted) | -14.709 / -14.514 / -14.337; 3.444 / 3.906 / 4.389 | 0.186; 0.473 | 0.028 / 0.047; 0.071 / 0.118 (Tier A uses the product-space subsample's approximate MCSE) |
| P-5 | HD 4.333 log10 A | -14.734 / -14.675 / -14.615 | 0.060 | 0.009 / 0.015 |
| P-6 | ln BF(HD/CURN) | paper ~1.5 / ~2 (ln 0.41 / 0.69); count ratios 1.85 / 1.69 | - | \|Delta ln BF\| <= 0.40 + 2 SE_ours against the paper's value; secondary (rounded, weights uncertain) |

The P-3 margin (0.008 dex) is tight because the reference is narrow. The reference MCSE(q50) is
0.0014, so Tier A is also achievable. This is deliberate: P-3 is the most precise test of the full
model.

**P-C0 conditional implementation check** (gating for the PPTA code, not a reproduction claim).
Ours vs enterprise on identical inputs and fixed noise:
* lnL-shape differences over the grids <= 1e-6 nats (arbiter) and <= 1e-4 nats (enterprise);
* conditional posterior quantiles (q05/q50/q95 of log10 A and gamma) agree to <= 0.002 dex /
  0.005, i.e. grid resolution.

### 6.4 MPTA and InPTA (approximate cross-checks)

Declared outcomes are CONSISTENT, TENSION or UNAVAILABLE. They never gate anything else.

* **MPTA** (M-1, M-2, M-3).
  * CONSISTENT if, for log10 A and gamma, our conditional median lies inside the published 68 %
    interval **and** the published median lies inside our 68 % interval.
  * TENSION otherwise, reported with the shift in published-sigma68 units.
  * Also reported: the BIPM2020 vs BIPM2022 variant and the J1825-0319 inclusion/exclusion shift.
  * Our fixed-noise intervals are expected to be narrower than the published ones. This is not
    counted against consistency.
* **InPTA.**
  * I-1: CONSISTENT if |log10 A_UL,ours - (-13.47)| <= 0.15 dex (the paper's own cut-to-cut and
    band-to-band spread is 0.03-0.05 dex; our WN is incomplete), with Kish ESS of the LinearExp
    reweighting >= 1,000.
  * I-2: the same rule with -13.55.
  * I-3: CONSISTENT if the published median lies within our 68 % interval (broad, prior-dominated
    posterior).

### 6.5 Frozen before any run (committed in M3b-0, reviewed)

`configs/m3b/acceptance_{epta,ppta,mpta,inpta}.json` hold every number in Sec. 6.2-6.4 and the
reference identities (path, sha256, burn-in rule: EPTA first 25 % of `chain_1.txt` discarded;
PPTA files as released). Alongside them:
* relevance declarations (`configs/m3b/relevance/*.json`);
* model configs (N1) and evaluator/clock profiles;
* the T2 report template.

Code-level tests assert that the gate reads these files and fails closed without them.

---

## 7. Milestones and budget

Each sub-milestone ends with an independent GPT-6 Astra review (`codex exec`, read-only; fallback:
independent Opus reviewers with the same VERDICT rule). A **REQUEST_CHANGES** blocks the next
sub-milestone. GPU numbers are projections (Sec. 5.5) and become allocations only after D3/D4/D5.

| sub-milestone | content | GPU (projection) | review gate | human decision |
|---|---|---|---|---|
| **M3b-0** infrastructure (mostly CPU) | N1-N11; tempo2 evaluator path + T1; G5-PTA (arbiter + enterprise) for EPTA and PPTA models; reference loaders; frozen acceptance/relevance files; band-overlap inventory (4.3 step 1); T2 engine sensitivity (conditional, CPU grids); published vs M3a-default TOA sets; GPU benchmark at the end | **0.5-1** (benchmark only) | Astra R-M3b-0 (incl. the E8 re-scope) | D1, D2, D3, D6 |
| **M3b-EPTA** | E-C0 (CPU); pilot (<= 2 GPU-h, abort rules); jump proposals frozen; production CURN^gamma and CURN 13/3; HD by reweighting (HD NUTS only if Sec. 5.4 fails); BF; optional OS | **5-45** (2 pilot + 3-40 production + HD fallback) | Astra R-M3b-EPTA | D4 (production allocation after the pilot), D8 if an evidence ladder is needed |
| **M3b-PPTA-C** | P-C0 conditional check; band-overlap sensitivity (4.3 step 2) | **< 0.5** | Astra R-M3b-PPTA-C | D5 (full-model path) |
| **M3b-PPTA-F** (conditional on D5) | (ii) Gibbs: design note -> toys -> agreement with NUTS on EPTA -> PPTA; or (i) NUTS + jumps; production CURN^gamma, 4.333; HD reweighting; BF | **5-60** (Gibbs, if validated) or **50-800** (NUTS) | Astra R-M3b-PPTA-F (+ Gibbs design review) | D5, D4-style allocation |
| **M3b-MPTA** | table transcription (double-entry checked against MPTAGW JSON); fixed-noise CURN/HD; clock variant | **< 0.5** | Astra R-M3b-X (joint with InPTA) | D7 |
| **M3b-InPTA** | fixed-noise UL and CURN | **< 0.5** | (joint) | - |
| **M3b close-out** | `docs/M3B_RESULTS.md`; updates to M3_PLAN Stage 2 inputs (pruning rule, evaluator profile, J1022/J0437 plan) | 0 | Astra final | go/no-go for M3c-prep |

**Total GPU (projection):** **~6-50 GPU-h without PPTA-F** (sum of the rows: 0.5-1 + 5-45 + three items < 0.5 each). With PPTA-F, add 5-60 (Gibbs) or
50-800 (NUTS). The NUTS upper end is not a plan: D5 is expected to choose between a validated Gibbs
kernel, a capped NUTS run, or conditional-only PPTA, on the measured M3b-0 numbers.

---

## 8. Human decision points

| # | decision | when | default if no answer |
|---|---|---|---|
| D1 | Adopt tempo2/libstempo as primary evaluator of non-NG legs for published reproductions and re-scope E8 (Sec. 4.1); needs reviewer agreement | before M3b-0 implementation | no non-NG reproduction claim |
| D2 | Confirm the Sec. 6 tolerances (Tier B 0.15/0.25 sigma68; E-6 0.30 nats; P-6 0.40; InPTA 0.15 dex) | before any sampling run | not run |
| D3 | First GPU use: M3b-0 benchmark (<= 1 GPU-h) | end of M3b-0 | not run |
| D4 | EPTA production allocation after the pilot (cap = pilot-measured projection x 1.5) | after the EPTA pilot | stop after pilot |
| D5 | PPTA full model: (i) capped NUTS + jumps, (ii) implement and validate the coefficient-Gibbs kernel, or (iii) conditional check only | after M3b-0 benchmark and P-C0 | (iii) |
| D6 | Fetch EPTA's modified enterprise/enterprise_extensions fork (tag EPTADR2-v1.1, gitlab.in2p3.fr) into the oracle env, so that the EPTA oracle and priors are exact | M3b-0 | rebuild the EPTA model from standard enterprise signals; priors marked inferred |
| D7 | MPTA clock default (par BIPM2020 vs paper BIPM2022) | M3b-MPTA | BIPM2020 default, BIPM2022 variant, both reported |
| D8 | Any evidence ladder or HD NUTS run beyond the projected envelope | if Sec. 5.4 step 1 fails | report INCONCLUSIVE |

---

## 9. Risks

| # | risk | likelihood / impact | mitigation |
|---|---|---|---|
| R1 | tempo2 2026.04.1 vs the PTAs' 2023 builds (clock/EOP files, fixes) shift residuals by ns, so Tier A fails | medium / low-medium | Tier B margins; T2-style evaluator comparison; trace order in Sec. 6.1 |
| R2 | Losing the independent timing oracle under D1 hides a tempo2 error | low / medium | PINT kept as secondary evaluator (T2); G1 and G5 stay independent; published chains are the end-to-end check |
| R3 | PPTA full model too expensive (K_a ~ 1,400, ~260 parameters) | high / high | M3b-0 benchmark; size buckets; reweighting for HD; Gibbs kernel; conditional-only fallback (D5) |
| R4 | Enterprise-semantics mismatches in the builders: masked-subset spans of group/band noise, ECORR quantisation, `int()` mode counts, `gamma_val=4.333`, informed priors | medium / medium | N1 block-by-block comparison against enterprise objects; G5-PTA |
| R5 | EPTA fork semantics unknown (priors, dip index, TempoNest DM) | medium / medium | D6; inferred priors from chain ranges are flagged; the released chain is the arbiter |
| R6 | Shelves and funnels in weakly constrained noise amplitudes (RN/DM reaching -18 in the EPTA chain; IRN funnels in M2) | high / medium | frozen block-MH jumps, region-transport gate, diverse inits, pilot-first |
| R7 | Reweighting overlap fails for EPTA HD (ln BF ~ 4) | low-medium / low | PSIS k-hat / Kish checks; HD NUTS + bridge fallback |
| R8 | Reference chains have modest ESS (PPTA 549-1,569; product-space HD ~350; EPTA ~1,000-1,500), which inflates Tier A's s and weakens agreement claims | certain / low | Tier A uses s; Tier B uses sigma68_ref margins; reported honestly |
| R9 | Target numbers inconsistent across paper and products (EPTA dip index; PPTA N_c 27 vs 28; PPTA BF with unknown model weights; EPTA 13/3 chain missing) | certain / low | follow the code that produced the chains; flagged [UNCERTAIN]; alternatives run where cheap (N_c = 28 as a variant) |
| R10 | Published TOA sets include clock-uncovered TOAs and possible duplicates | certain / low | Sec. 4.2 rule + robustness variant |
| R11 | MPTA/InPTA inputs incomplete (rounded tables; no InPTA WN) | certain / low | approximate class only; loose declared tolerances |
| R12 | GPU contention on the shared box | medium / low | record contention in run metadata; abort rules; ESS per hour is never claimed as a kernel speedup without matched arms |
| R13 | Scope creep into Stage 2 (combination) or free spectra | medium / medium | Sec. 1.3; each extension needs a plan revision |

---

## 10. Open questions

1. Is a DR2new gamma = 13/3 chain released anywhere (Zenodo 8091568 lists crn_pl, hd_pl, crn_fs,
   hd_fs, ORF variants and os)? If not, E-5 is compared with the paper's rounded interval only.
2. PPTA group-noise basis span (masked TOAs vs pulsar span) and the exact `psr_groupnoise_dict_dict['all']`
   and `psr_groupecorr_dict_dict['all']` contents: read from `ppta_dr3_utils.py` in M3b-0.
3. The PPTA BF definition behind "~1.5 / ~2" (product-space weights, which chain).
4. MPTA: N_c and T for the Bayesian common process; RN modes per pulsar ("120 components up to
   ~1/(14 d)").
5. InPTA DR2: N_c of the UL run; whether ECORR/EQUAD values can be obtained ("available on request").

---

## Appendix A. Reference-chain checks done for this plan (CPU, 2026-10-09)

Computed with `ptagwb.diagnostics` (`ess_bulk`, `ess_tail`, `mcse_quantile`) on the released files.
The producer becomes `scripts/m3b_reference.py` (N11) in M3b-0, which must reproduce these numbers.

**EPTA DR2new** (`epta_dr2_gwb_chains/.../DR2new/{hd_pl,crn_pl}.tar.gz`, `chain_1.txt`, 29,990
rows x 71 columns = 67 parameters + lnpost, lnlike, acceptance, PT acceptance; first 25 %
discarded, 22,493 kept; medians change by <= 0.001 with no burn-in):

| chain | parameter | q05 | q16 | q50 | q84 | q95 | bulk / tail ESS | MCSE q50 / q16 / q05 |
|---|---|---|---|---|---|---|---|---|
| hd_pl | log10 A | -14.418 | -14.149 | -13.935 | -13.783 | -13.702 | 1,331 / 1,120 | 0.0046 / 0.0103 / 0.0368 |
| hd_pl | gamma | 1.978 | 2.251 | 2.710 | 3.300 | 3.894 | 1,556 / 1,213 | 0.0121 / 0.0098 / 0.0127 |
| crn_pl | log10 A | -14.766 | -14.323 | -13.997 | -13.815 | -13.718 | 984 / 1,108 | 0.0067 / 0.0263 / 0.0581 |
| crn_pl | gamma | 2.038 | 2.364 | 2.906 | 3.701 | 4.626 | 1,139 / 1,257 | 0.0169 / 0.0142 / 0.0190 |

Published (90 %): HD -13.94 (+0.23/-0.48), 2.71 (+1.18/-0.71); CURN -14.00 (+0.28/-0.77),
2.91 (+1.72/-0.87). All reproduced to the printed digit. Chain ranges imply noise log10 A
U(-18, -10), gamma U(0, 7), dip log10 tau in [0, 2.5], dip t0 in [57490, 57530] (the dip is at
MJD ~57510). The CURN amplitude reaches -17.97 and RN/DM amplitudes reach -18.0 (prior floor).

**PPTA DR3** (GitHub fdbe6eb, `analysis_codes/data/all/chains/`):

| chain | parameter | draws | q16 | q50 | q84 | bulk / tail ESS | MCSE q50 / q16 |
|---|---|---|---|---|---|---|---|
| pl_nocorr_freegam_DE440 | log10 A | 2,978 | -14.655 | -14.496 | -14.358 | 549 / 727 | 0.0053 / 0.0088 |
| pl_nocorr_freegam_DE440 | gamma | 2,978 | 3.505 | 3.869 | 4.228 | 619 / 910 | 0.0151 / 0.0220 |
| pl_nocorr_fixgam_DE440 | log10 A | 2,967 | -14.735 | -14.681 | -14.634 | 1,569 / 2,342 | 0.0014 / 0.0017 |
| ..._v_pl_hd_freegam (nmodel > 0.5) | HD log10 A | 3,643 | -14.709 | -14.514 | -14.337 | ~354 | - |
| ..._v_pl_hd_freegam (nmodel > 0.5) | HD gamma | 3,643 | 3.444 | 3.906 | 4.389 | ~319 | - |
| ..._v_pl_hd_fixgam (nmodel > 0.5) | HD log10 A | 3,226 | -14.734 | -14.675 | -14.615 | ~1,728 | - |

The HD-active fractions are 0.649 (free gamma) and 0.628 (fixed), i.e. count ratios 1.85 and 1.69.
The fixed-gamma product-space chain bounds the amplitudes to [-18, -14].

**Mode counts and spans** (from `data/processed/m3_survey/*.csv` and the released dictionaries):
* EPTA: T_array = 10.33 yr, 45,428 TOAs; K_a (noise) max 398 (J0900-3144), median 50; sum of K_a^3
  (incl. 18 common columns) ~16x NG15's with buckets, ~124x padded.
* PPTA (30 pulsars): T = 6,604.7 d, so N_c = 27; 112,556 TOAs. K_a counts RN, DM, chromatic,
  band, HF and SW-GP blocks plus 54 common columns, before group noise. The top five are J1744-1134
  988, J1600-3053 1,042, J1713+0747 1,146, J1909-3744 1,208 and J0437-4715 1,388; the median is
  ~540. Sum of K_a^3 is ~760x NG15's, and ~5,500x if every pulsar is padded to the maximum.
* These counts are projections from the builder's rules applied to the survey spans; N1 replaces
  them with the enterprise-exact counts.
