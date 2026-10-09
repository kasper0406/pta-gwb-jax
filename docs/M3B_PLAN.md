# M3b plan: reproducing each PTA's own published GWB result (Stage 1)

M3b is Stage 1 of [`M3_PLAN.md`](M3_PLAN.md) (Sec. 5.2): reproduce each individual PTA's own
published GWB result with our pipeline, as far as the public products allow. M3a delivered the
infrastructure ([`M3A_VALIDATION.md`](M3A_VALIDATION.md): multi-leg container, tempo2-semantics
ingestion, profiles, general multi-block likelihood, gates G1-G9; approved in review round 3 with
E2-C's original all-oracle criterion unmet and E7/E8 open).

**This is a design and pre-registration document. Nothing has been implemented, sampled or run on
a GPU for it.** The only computations behind it are CPU reads of released products: reference-chain
quantiles, MCSEs, prior-volume columns, mode counts and builder source code (Appendix A). Every
compute number is a **projection** unless it is marked as measured and cited. The decision rules
in Sec. 6 are fixed here. The practical margins are **provisional** until the user confirms them
(D2); after confirmation they change only through a documented revision made before the run it
affects, and that revision is reviewed.

**Revision 2 (after independent review of 25436e9, REQUEST_CHANGES).** The review endorsed
EPTA-first, the tempo2 evaluator, keeping E8 open and deferring option C / E7. Changes:
1. The **PPTA common-process grid** is not fixed by the mode count. `get_crn_model_dict` does not
   pass `Tspan` to `enterprise_extensions.blocks.common_red_noise_block`, and the resulting grid
   depends on the extensions version and differs between CURN and HD (Sec. 2.2). The chain-producing
   versions must be established, and frequencies, normalisations and bases are exported separately
   for CURN and HD. The evidence-ladder endpoint is fixed for unequal grids (Sec. 5.4).
2. A complete **version-pinned prior/model manifest** is an M3b-0 exit condition (Sec. 4.5). The
   EPTA fork audit (enterprise *and* enterprise_extensions) is mandatory for a full-reproduction
   claim; there is no inferred-prior fallback. PPTA exceptions are added: fixed-gamma amplitude
   U(-18, -14), confirmed from the chains' prior columns; SW-GP bound rules.
3. **T1 covers every consumed array** (barycentric epochs, row order and deletion masks, ORF
   positions, SW geometry, timing-model rank, frequency convention) and checks original released
   inputs against canonical ones. Acceptance needs the complete roster; a dropout is a separately
   labelled sensitivity analysis (Sec. 4.1).
4. **Sampled event epochs** leave NUTS and are updated by exact MH, validated against a numerical
   reference. Divergence rules apply to NUTS sub-steps only (Sec. 5.1, 5.3).
5. The **transport gate** requires bidirectional transport only when both regions have material
   reference support. It is defined separately for amplitudes, event epochs and other parameters
   (Sec. 5.3).
6. **Reweighting**: raw and PSIS estimators are specified, with autocorrelation-aware MCSE for
   weights and weighted quantiles, independent-chain stability, precision gates on HD outputs, and
   an ordered-chain bootstrap of the PPTA product-space reference (Sec. 5.4).
7. **Acceptance** uses the uncertainty of both estimates: EQUIVALENT / INCOMPATIBLE / INCONCLUSIVE,
   where equivalence means the difference's 90 % interval lies inside the margin. Reference-limited
   quantities are identified now (Sec. 6).
8. **BF**: cross-model lnL_HD - lnL_CURN oracle checks including normalisations, a maximum BF
   uncertainty, and an uncertainty-aware classification (Sec. 6.2, 6.3).
9. **PPTA cost**: group-selection columns added (J0437 -> 1,816); non-harmonic group grids; the
   arithmetic is corrected; Gibbs is labelled as unvalidated speculation; no dense Q_perp
   (Sec. 3, 5.5).
10. **Target corrections**: the EPTA HD gamma lower error and the paper's internal discrepancy;
    J1022 NE_SW is fitted and marginalised; PPTA samples 16 mean SW densities; E-5 is an explicit
    fixed-gamma HD target with provenance (Sec. 2).
11. **Structure**: M3b-0 is split so that EPTA is not blocked by PPTA or cross-check
    infrastructure. PPTA defaults to conditional-only, and PPTA sampler development is a separate,
    separately reviewed milestone. OS is optional. D1-D8 carry the reviewer's recommendations,
    marked as such (Sec. 8).

**Revision 3 (after review of f47b66d, REQUEST_CHANGES; M3b-0 may start with these
incorporated).**
1. The **fingerprint gate** fails closed on any scatter beyond a frozen error budget. The budget
   comes from storage precision (float64 values with six decimals, never cast to float32),
   parameter rounding and evaluator error. There is no "investigate but keep the claim" band, and
   grid discrimination requires the alternative to exceed that budget. Fingerprints are supporting
   evidence, not a substitute for source identification. Model-dependent offsets c_HD - c_CURN are
   covered by the cross-model oracle checks, not by the fingerprint (Sec. 4.5, 6.2).
2. **PPTA product-space prior.** The logged offsets (-935.2116, -930.2006) are twice the
   single-model ones, because `HyperModel.get_lnprior` sums the constituent models' priors. The
   physical normalised prior is now distinguished from the sampler's logged density. The informed
   prior rules reproduce the single-model volumes to -0.003872 nats; the earlier 2.541-nat claim
   was our band-noise key-mapping bug (Sec. 2.2, 4.5, Appendix A).
3. **Zero reference occupancy** uses a boundary-aware, dependence-adjusted upper bound; if
   dependence prevents a defensible bound, support is "unresolved". This also applies to
   event-epoch intervals (Sec. 5.3).
4. **PSIS.** Acceptance is restricted to k-hat < 0.5 with the raw estimator. Above that the
   existing fallback (HD run) applies. A validated PSIS uncertainty procedure would need a plan
   revision (Sec. 5.4).

**Revision 4 (after review of 353941b; one MAJOR open).** The zero-visit bound is changed. Zero-visit
(and few-event) region support is **UNRESOLVED** unless an independently justified bound exists. The
only admissible one is the Rao-Blackwellised conditional occupancy, with explicit validity
conditions. Bulk amplitude/epoch ESS is never used. All-visit cases go through the complementary
region, so p_hi = 1. This applies to reference chains, our chains and event intervals. A decision
table states how UNRESOLVED propagates: it gives INCONCLUSIVE, never a silent pass or a hard fail
(Sec. 5.3).

**Revision 5 (after review of 0dcb5ff).**
* The Rao-Blackwellised conditional-occupancy bound is withdrawn as a classifier and kept as a
  supporting diagnostic. Only certified bounds (over all conditioning states, or a proven global
  mixing bound) could resolve zero- or few-event regions, and none is planned.
* An explicit precedence rule makes any UNRESOLVED occupancy, in the reference or in our run, for
  amplitude regions or event intervals, give INCONCLUSIVE before the transport-FAIL rows are
  evaluated.
* INCONCLUSIVE blocks acceptance.
* The consequence for EPTA (REPRODUCED unattainable while its 17 unvisited shelves stay
  unresolved) is flagged as decision D9 (Sec. 5.3, 8).

**User decisions, 2026-10-09 (D1-D9).** The user accepted every decision as the reviewer
recommended it (D1-D8: review of 25436e9; D9: review round 5, `review_m3b_r5.out`). The table in
Sec. 8 records each decision; the binding wording is:
* **D1:** tempo2/libstempo is the evaluator for non-NANOGrav legs, with the expanded T1 and
  complete-roster acceptance (Sec. 4.1). E8 stays open.
* **D2:** the decision rule of Sec. 6.1 is confirmed. The margins stay **provisional** under the
  EQUIVALENT / INCOMPATIBLE / INCONCLUSIVE rule.
* **D3:** **one** GPU benchmark of **<= 1 GPU-h**. It may run only after the exact-model, T1 and
  G5-PTA checks have passed (this replaces the "<= 0.5 GPU-h each in M3b-0E and M3b-0P" of
  Sec. 5.5 and Sec. 7 for the EPTA path).
* **D4:** the EPTA production allocation is decided **after the benchmark**, from its measured
  cost. Work stops and is reported before any production run (pilot included).
* **D5:** PPTA is conditional-only (P-C0); no PPTA sampler work.
* **D6:** the EPTA fork audit (enterprise **and** enterprise_extensions) is mandatory for a
  full-reproduction claim.
* **D7:** MPTA uses BIPM2022 for the published-target comparison; the native-par clocks are a
  sensitivity.
* **D8:** evidence runs are capped, with the INCONCLUSIVE fallback.
* **D9:** the claim class **"CONDITIONALLY EQUIVALENT TO RELEASED RESULTS"** is adopted exactly
  as specified in the round-5 review: frozen union U, zero visits in both chains,
  mixture-envelope sensitivity for p in {0.001, 0.01, 0.05, 0.1}, BF sensitivity, and the
  unconditional verdict stays INCONCLUSIVE. Template and conditions: Sec. 6.6.

Notation follows M3_PLAN: **leg** = one PTA's par + tim of one pulsar; **K_a** = sampled GP columns
of pulsar a; **N_c** = number of common-process frequencies; **[UNCERTAIN]** = not pinned down from
papers or released files. "Reference chain" = the PTA's released posterior samples.

---

## 0. Summary

| PTA (release) | Published target (headline) | Reference products | Class we can claim | Order |
|---|---|---|---|---|
| **EPTA DR2new** (GitLab 2911d0e) | HD log10 A = -13.94 (+0.23/-0.48), gamma = 2.71 (+1.18/-0.73 or -0.71; paper inconsistent) (90 %); BF(HD/CURN) = 60 | machine-readable noise files and dicts; model script; released CURN/HD chains with **all** 67 parameters plus lnlike and lnpost columns (Zenodo 8091568) | **full reproduction**, conditional on the fork audit and the manifest (Sec. 4.5); otherwise downgraded | 1 |
| **PPTA DR3** (GitHub fdbe6eb) | CURN log10 A = -14.50 (+0.14/-0.16), gamma = 3.87 +- 0.36 (68 %) | max-likelihood noise JSONs, 3-sigma prior files, model builder, CURN / product-space chains with all ~260 parameters plus lnlike/lnpost | **conditional implementation check by default**; full reproduction only as a separate milestone (M3b-PPTA-S/F, D5) | 2 |
| **MPTA 4.5-yr** | ER CURN log10 A = -14.25 (+0.21/-0.36), gamma = 3.60 (+1.31/-0.89) | par/tim; noise only in rounded paper tables and an unofficial WN dictionary | **approximate cross-check** | 3 |
| **InPTA DR2** | 95 % UL log10 A < -13.47 (gamma = 13/3, uniform-A prior) | par/tim with T2EFAC; RN/DM/chromatic/SW table in the noise paper; no WN values, no chains | **approximate cross-check** | 4 |

Key decisions (Sec. 8 lists all human decision points):

* **E8 legs (Sec. 4.1).** Published-model reproductions use **tempo2 (libstempo, pinned oracle env)
  as the primary evaluator of non-NG legs**, as the published analyses did. PINT is kept as an
  independent secondary evaluator whose effect is reported per PTA. E8 stays open (re-scoped as a
  PINT-parity diagnostic). Reproduction acceptance needs an expanded T1 on the **complete roster**.
* **Model identity (Sec. 4.5).** The primary evidence is source-model identification: an audited,
  version-pinned manifest. Two checks support it and fail closed:
  * a **chain fingerprint**: our lnL at released draws minus the stored lnlike must be constant
    within a frozen error budget;
  * a **prior-volume check**: our log prior must reproduce the logged lnpost - lnlike, including
    the product-space sampler's logging convention.

  First CPU checks:
  * EPTA's prior volume matches noise log10 A U(-18, -10), gamma U(0, 7), the dip boxes, and a
    **common log10 A U(-18, -10)**, to ~1e-9 nats in the mean;
  * PPTA's fixed-gamma amplitude is U(-18, -14);
  * the released informed-prior rules reproduce PPTA's single-model volumes to a constant
    **-0.003872 nats**. This is still unresolved; revision 2's "2.541 nats" was a key-mapping error
    of ours.
  * PPTA's product-space chains log **twice** the single-model prior: HyperModel sums both
    models' priors (Appendix A).
* **Stage 1 needs neither option C nor E7** (one leg per pulsar per PTA, Sec. 4.4).
* **Published TOA sets** are used (no M3a clock exclusions or duplicate removals), and the M3a
  defaults are a reported sensitivity (Sec. 4.2).
* **Sampling (Sec. 5).** NUTS with a fixed dense metric on the continuous parameters. Frozen block-MH
  jumps for shelf-prone (log10 A, gamma) pairs. Exact MH for event epochs. HD and BF(HD/CURN) by
  importance reweighting first (raw estimator, accepted only for Pareto k-hat < 0.5, autocorrelation-aware MCSE), and an HD run
  only if that fails.
* **PPTA full model is the compute risk.** Up to K_a ~ 1,800 columns per pulsar (J0437). Projected
  ~56-2,100 GPU-h per NUTS run (Sec. 5.5). A coefficient-Gibbs kernel is unvalidated research. PPTA
  therefore defaults to the conditional check.
* **Budget (projection).** M3b without PPTA sampler work: **~7-100 GPU-h** (central ~15). The PPTA full model is a
  separate decision (D5).

---

## 1. Scope

### 1.1 Three kinds of comparison (never mixed)

* **Full reproduction.** Our implementation of the PTA's *published* model: same data, timing
  evaluator, TOA roster, noise components, sampled and fixed parameters, priors, frequency grids
  and normalisations. Compared with the released chains under Sec. 6. Requires the manifest
  (Sec. 4.5) to be complete and verified. Only EPTA DR2new and PPTA DR3 release enough for this.
* **Conditional implementation check.** All non-target noise fixed at released values. Our
  conditional likelihood surface / posterior of the common process is compared with enterprise on
  **the same conditional model and inputs**. This validates code, not published numbers. It is
  **not** expected to match the published noise-marginalised chains.
* **Approximate cross-check.** Products needed for an exact configuration are missing (MPTA,
  InPTA). Comparison against published summaries with loose, pre-declared rules. Nothing depends on
  it.

A claim is downgraded, never stretched. If the manifest of a full reproduction stays unresolved,
or the roster is incomplete, the result is reported as an "approximate comparison" with the reason.

### 1.2 Ordering

EPTA first: highest value (BF ~60; chains contain every parameter plus lnlike/lnpost) and highest
feasibility (67 parameters, K_a <= 416, 45k TOAs). PPTA second, conditional check by default. MPTA
and InPTA last as approximate cross-checks; they are not prerequisites for Stage 2.

### 1.3 Out of scope for M3b

NG15 (done). Combinations (Stage 2+). Sampled white noise in array runs. BayesEphem. Free spectra and
ORF reconstructions. Sky scrambles and phase shifts. EPTA DR2full and DR2new+ (DR2new+ contains
InPTA DR1). The OS is optional everywhere.

---

## 2. Per-PTA specifications

Numbers from papers were checked against the arXiv sources (survey Sec. 3.7-3.10) and re-checked for
this plan against arXiv abstracts or HTML (2306.16214, 2306.16215, 2306.16229, 2412.01153) and the
InPTA GWB LaTeX (2608.02808v1). Reference-chain numbers were recomputed from the released files
(Appendix A).

### 2.1 EPTA DR2new: full reproduction (subject to Sec. 4.5)

**Targets** (Antoniadis et al. 2023, *EPTA DR2 III*, A&A 678 A50, arXiv:2306.16214; 90 % credible,
enterprise):

| ID | quantity | published | reference (Appendix A) | role |
|---|---|---|---|---|
| E-1 | HD^gamma log10 A | -13.94 (+0.23/-0.48) | -13.935 [-14.418, -13.702] | headline |
| E-2 | HD^gamma gamma | 2.71 (+1.18 / -0.73 in one passage, -0.71 in another; see note) | 2.710 [1.978, 3.894] (lower error 0.732) | headline |
| E-3 | CURN^gamma log10 A | -14.00 (+0.28/-0.77) | -13.997 [-14.766, -13.718] | headline |
| E-4 | CURN^gamma gamma | 2.91 (+1.72/-0.87) | 2.906 [2.038, 4.626] | headline |
| E-5 | HD log10 A at gamma = 13/3 | -14.61 (+0.11/-0.12) | no released chain | secondary (provenance below) |
| E-6 | BF(HD/CURN) | 60 (enterprise); 62 (fortytwo); re-estimates 66, 56, 62 | not derivable from released chains | headline |
| E-7 (optional) | OS (13/3): A^2 = 10.0 (+5.1/-4.9) x 10^-30, S/N 3.5 | `os/hd_amp_sn.txt` | optional |

Notes:
* **E-2 discrepancy.** The released chain gives a lower error of 0.732. The paper prints -0.73 and
  -0.71 in different places: the reviewer reads Table 2 as -0.71 and the main text as -0.73, and
  our survey extraction recorded -0.71. The chain is the reference; the discrepancy is recorded.
  The released chains reproduce E-1, E-3 and E-4 to the printed precision and E-2 to the -0.73
  variant.
* **E-5 provenance.** The paper says that "when fixing the spectral index to 13/3" the amplitude is
  log10 A = -14.61 (+0.11/-0.12). It describes this as a slice of the 2-D posterior, the lower error
  differs between passages, and the slice method is not specified. **Our target definition:** the
  posterior of log10 A in the **HD model with gamma fixed at 13/3** and the same amplitude prior.
  This is exactly the conditional p(log10 A | gamma = 13/3, data) that a slice estimates. We obtain
  it from a fixed-gamma CURN run reweighted to HD (Sec. 5.4). Because the reference method and its
  uncertainty are not reconstructible, E-5 is **secondary**: classified and reported, not part of
  the REPRODUCED verdict.

**Published-analysis profile.**
* Data: DR2new par/tim (25 pulsars, 45,428 TOAs, 2011.1-2021.5, T_array = 10.33 yr).
* Clock and ephemeris: TT(BIPM2021), DE440, the release's corrected Nancay clock.
* Units: TCB, evaluated natively by tempo2 (D1).
* Timing model: marginalised (`tm_svd=True, tm_marg=True`). Solar wind is the par's deterministic
  NE_SW = 7.9 cm^-3 for all pulsars except **J1022+1001**, whose NE_SW = 9.70 +- 0.55 is
  **fitted** (fit flag 1). Its design column is therefore analytically marginalised with the
  timing model (flat prior), not fixed.
* WN: EFAC + TNEQUAD per `-group`, fixed at `noisefiles/DR2new/*.json`, no ECORR.
* Model code: `EPTA-DR2/scripts_gwb/model_single.py`, which calls `model_general` from EPTA's
  modified enterprise and enterprise_extensions (tag `EPTADR2-v1.1`, gitlab.in2p3.fr; **audit
  mandatory**, Sec. 4.5).

**Noise model per pulsar** (released `red_dict.json`, `dm_dict.json`, `chrom_dict.json`):

| component | where | basis (to be confirmed by the fork audit) | status |
|---|---|---|---|
| achromatic RN | 8 pulsars (10-99 modes) | Fourier on the pulsar span | **sampled** |
| DM GP | 22 pulsars (11-100 modes) | nu^-2, TempoNest normalisation (`tndm=True`) | **sampled** |
| scattering GP | J1600-3053 (93 modes) | nu^-4 (`chrom_kernel='diag'`) | **sampled** |
| exponential dip | J1713+0747, one event (t0 in [57490, 57530], chromatic index 1) | deterministic, Heaviside onset at t0 | **sampled** (log10 Amp, log10 tau continuous; **t0 by MH**, Sec. 5.1) |
| solar wind | NE_SW in the par; J1022 fitted (above) | timing model | fixed / marginalised |
| common process | 9 modes **[UNCERTAIN until audit: array span via `model_general`'s `Tspan`]**; `freq_bins/freqs_dr2new.txt` lists multiples of 3.066e-9 Hz = 1/(10.33 yr) | | sampled |

Sampled parameters: 62 GP + 3 dip + 2 common = **67** (matches the chain). The prior volume in the
chain (lnpost - lnlike = -135.4959, constant up to six-decimal rounding) equals, to ~1e-9 nats in the mean:
* noise log10 A U(-18, -10) and gamma U(0, 7) for all 31 processes;
* dip log10 Amp width 8, log10 tau U(0, 2.5), t0 width 40 d;
* common log10 A **U(-18, -10)** and gamma U(0, 7).

The alternative common upper bound -11 gives -135.362 and is excluded. A volume match is a
consistency check, not an identification of each bound, so the audit (Sec. 4.5) is still
required.

*Discrepancy recorded:* the GW paper gives dip indices 4 and 2. The released code
(`dm_expdip_idx=[1,4]`) and the noise file key `expd-1.0_57490_57530` give index **1** for the 2016
event, the only one inside DR2new. We follow the code that produced the chains.

**M3a support and gaps.**
* Supported: per-block spans and mode counts, TempoNest DM normalisation, nu^-4 blocks, TN EQUAD,
  namespacing, distinct grids, padding.
* To add: the dip with MH t0 (N3, N13); the EPTA builder (N1); stage-2 buckets (N8); the manifest
  and fingerprint tooling (N11).
* Largest K_a = 416 (J0900-3144: 398 + 18 common). Sum of K_a^3 ~16x NG15's with buckets.

**Comparisons.**
* E-C0, a conditional implementation check: fixed noisefile values; CURN/HD surfaces on 141 x 141
  grids vs enterprise from the audited fork (CPU).
* Full reproduction of E-1 to E-4 and E-6.
* E-5 secondary; E-7 optional.

### 2.2 PPTA DR3: conditional implementation check by default

**Targets** (Reardon et al. 2023, ApJL 951 L6, arXiv:2306.16215; 68 %). Reference: GitHub fdbe6eb
`analysis_codes/data/all/chains/chain_commonNoise_*_DE440.npy`.

| ID | quantity | published | reference (Appendix A) |
|---|---|---|---|
| P-1 | CURN^gamma log10 A | -14.50 (+0.14/-0.16) | -14.496 (+0.138/-0.159), 2,978 draws, bulk ESS 549 |
| P-2 | CURN^gamma gamma | 3.87 +- 0.36 | 3.869 (+0.359/-0.364) |
| P-3 | CURN fixed-gamma (4.333) log10 A | -14.69 +- 0.05 | -14.681 (+0.047/-0.054) |
| P-4 | HD^gamma log10 A, gamma | -14.51 (+0.18/-0.20); 3.87 +- 0.47 | HD-active product-space draws: -14.514 (+0.177/-0.195); 3.906 (+0.483/-0.462) |
| P-5 | HD fixed-gamma log10 A | -14.68 +- 0.06 | -14.675 (+0.060/-0.059) |
| P-6 | BF(HD/CURN) | ~1.5 (free), ~2 (fixed) | product-space count ratios 1.85 / 1.69 **[UNCERTAIN: model weights; Sec. 6.3]** |

**Published-analysis profile.**
* Data: 30 pulsars (J1741+1351 and J1824-2452A excluded), 112,556 TOAs, 2004.1-2022.2.
* Clock: TT(BIPM2020) with `pks2gps.clk`. Ephemeris: DE440 (override of the par's DE436). Units:
  TCB (implicit).
* Timing model marginalised (SVD).
* WN fixed at `noiseFiles_maxlike/*`: EFAC + TNEQUAD per backend; band-split ECORR (40/20/10 cm,
  UWL and non-UWL); global UWL and global non-UWL ECORR (`global_ecorr`); group ECORRs. These are
  overlapping selections, with enterprise quantisation per selection.

**Common-process grid: unresolved, and it differs between models.**
* `get_crn_model_dict(tspan)` computes `components = int(T_array / 240 d)` = **27**. Our 30-pulsar
  span is 6,604.7 d; the paper prints the inconsistent "floor(6605/240) = 28".
* It does **not** pass `Tspan` to `crn_block` = `enterprise_extensions.blocks.common_red_noise_block`
  (imported in `ppta_dr3_models.py`). The frequencies then depend on the extensions version:
  * **2023-era extensions** (reviewer's reading and two-pulsar check): CURN (`orf=None`) builds
    `FourierBasisGP`, i.e. 27 harmonics of **each pulsar's own span**. HD builds
    `FourierBasisCommonGP`, which infers the **array span**.
  * **Installed extensions 3.0.3** (read here): both use `createfourierdesignmatrix_red(Tspan=None)`
    inside `BasisGP` / `BasisCommonGP`, i.e. per-pulsar spans for **both**.
* The chain-producing versions are in the unreleased Singularity image `entpptadr3_bilby_nano.sif`
  (the repository README says it will be on the CSIRO DAP).
* **Resolution (M3b-0P exit condition):**
  1. obtain the image or its package list;
  2. as supporting evidence, run the **chain fingerprint test** (Sec. 4.5): evaluate each grid
     hypothesis at >= 50 released draws against the stored lnlike column, with the frozen error
     budget and the fail-closed discrimination rule;
  3. export frequency labels, normalisations and bases **separately for CURN and HD**, and handle
     per-pulsar vs array spans explicitly.

  If the grids differ between CURN and HD, both are reproduced as published, and the HD/CURN
  reweighting and any evidence ladder use the correct endpoints (Sec. 5.4).

**Noise model** (`commonNoise.py`, `ppta_dr3_utils.py`; modes = int(T / cadence); spans to be
confirmed per block by N1):

| component | where | basis | status |
|---|---|---|---|
| achromatic RN | all 30 | T_p/240 d | **sampled**, informed prior |
| DM GP | all 30 | nu^-2, T_p/60 d | **sampled**, informed prior |
| chromatic GP | 7 (J0437, J0613, J1017, J1045, J1600, J1643, J1939) | nu^-4, T_p/240 d | **sampled** |
| band noise low (<= 960 MHz) | 9; J0437 also mid and high | T_p/60 d, frequency mask | **sampled** |
| HF achromatic | 8 (J0437, J1017, J1022, J1600, J1713, J1744, J1909, J2241) | T_p/30 d | **sampled** |
| group noise | J0437 (4 groups), J1017 (2), J1022 (2), J1713 (4), J1909 (1) (`psr_groupnoiselist_dict_all`) | `FourierBasisGP_ppta(fmax=1/30 d)` on the **masked** TOAs: frequencies `linspace(1/T_mask, fmax, n)`, **not integer harmonics**; one shared prior per pulsar from the sup/inf of the groups' 3-sigma bounds | **sampled** |
| deterministic SW | n_earth **sampled in 16 pulsars** (the code lists 17, including the excluded J1824); prior [max(0, q_lo - 1), min(20, q_hi + 1)]; fixed at 4 cm^-3 otherwise | geometry x n_earth (linear) | **sampled** |
| SW GP | 10 (J0437, J0711, J0900, J1024, J1643, J1713, J1730, J1744, J1909, J2145) | `createfourierdesignmatrix_solar_dm`, T_p/60 d | **sampled**; prior log10 A [max(-10, q_lo - 1), min(-3, q_hi + 1)], gamma [max(-4, q_lo - 0.5), min(4, q_hi + 0.5)], **negative slopes allowed** (released chains contain them) |
| exponential dips | J1713 (2; index U(1, 3), U(0, 2)), J0437 (1, U(-1, 2)), J1643 (1, U(-2, 0)), J2145 (1, U(-2, 2)) | deterministic, Heaviside onset | log10 Amp U(-10, -2), log10 tau U(0, 2.5), index sampled continuously; **t0 (200-d windows) by MH** |
| annual DM sinusoid | J0613 | deterministic nu^-2 | **sampled** |
| Gaussian DM event | J1603 (epoch U(53800, 54000), log10 sigma U(0, 3)) | deterministic nu^-2 | **sampled** (smooth in epoch: NUTS) |
| J1600 bumps, achromatic quadratic | off in the GW run | - | not modelled |

Priors: common log10 A U(-18, -11) and gamma U(0, 7) for free gamma. **Fixed-gamma runs use
gamma = 4.333 and log10 A U(-18, -14).** The width-4 box is confirmed: the freegam and fixgam
chains' prior columns differ by exactly ln(49/4) = 2.5055. Informed noise priors:
* log10 A in [max(-18, q_0.15% - 2), min(-11, q_99.85% + 1)];
* gamma in [max(0, q - 0.5), min(7, q + 0.5)], from `noiseFiles/3sig`;
* SW and group rules as in the table.

A reconstruction of the complete box set has been made with the released rules: informed bounds
looked up under the builder's *noisename* keys (`band_noise_low`, not the parameter's
`band_noise_low_low`), and shared per-pulsar group bounds. It gives -467.601929 (free gamma) and
-465.096403 (fixed). The logged single-model volumes are -467.605801 and -465.100276: a constant
residual of **-0.003872 nats** in both. This agrees with the reviewer's independent reconstruction.
The residual is far above the six-decimal storage rounding (row spread 5e-7), so some bound is
still slightly off (a width ratio of ~1.0039). Resolving it to <= 1e-5 nats is an M3b-0P exit
condition (Sec. 4.5). Revision 2's "2.541 nats" came from our own key-mapping error.

**Comparisons.**
* **P-C0 conditional implementation check (default deliverable).** All noise hyperparameters
  (incl. dips, SW, n_earth) fixed at the max-likelihood JSONs. CURN and HD surfaces on the
  established grids, ours vs enterprise built with the released `ppta_dr3_models.py` under the
  established extensions version.
* **Full reproduction (P-1 to P-6)** only via the separate milestones M3b-PPTA-S/F (D5).

### 2.3 MPTA 4.5-yr: approximate cross-check

**Targets** (Miles et al. 2025, MNRAS, arXiv:2412.01153; median and 68 %): M-1 DATA CURN -14.25
(+0.21/-0.34), gamma 3.52 (+1.12/-0.90). M-2 ER CURN -14.25 (+0.21/-0.36), gamma 3.60 (+1.31/-0.89).
M-3 DATA HD -14.28 (+0.23/-0.30), gamma 4.50 (+1.00/-0.93). The abstract's h_c,yr = 7.5e-15
(alpha = -0.26) and 4.8e-15 (alpha = -2/3) come from the fixed-parameter OS (A^2 = (5.7 +- 1.2) x
10^-29). They are not Bayesian posteriors and are not compared.

**Profile.**
* 83 pulsars under tempo2 (J1825-0319's signed H3 is handled by tempo2), 245,907 TOAs, KAT_MKBF.
* Clock: decision D7. The reviewer recommends the paper's BIPM2022 for the published-target
  comparison, with the par's BIPM2020 as a sensitivity.
* DE440; WN fixed. Noise from the arXiv:2412.01148 tables (rounded); unofficial MPTAGW
  `example_noise.json` as a transcription cross-check.
* N_c and T of the Bayesian common process **[UNCERTAIN]**.

**What we do.** A conditional approximation: WN from the tables; RN/DM/chromatic/SW/event
parameters fixed at the tabulated MAP; beta fixed; CURN and HD on the array span. Fixed-noise
posteriors vs noise-marginalised published intervals is approximate by construction.

### 2.4 InPTA DR2: approximate cross-check

**Targets** (arXiv:2608.02808v1): I-1 95 % UL at gamma = 13/3, uniform-in-A prior (LinearExp by
reweighting a log-uniform chain): log10 A < -13.47. I-2 log-uniform quantile -13.55. I-3 CURN^gamma
log10 A = -13.71 (+1.06/-3.28), gamma = 2.98 (+3.62/-2.70). Info: Savage-Dickey BF 2.5.

**Profile.**
* 27 pulsars, 83,120 TOAs, TT(BIPM2023), DE440, TCB; DM + DM1 + DM2 (`InPTA.DR2.NA`).
* Noise from the arXiv:2512.20455 table: ARN, DMN, free-chromatic noise at the tabulated chi,
  deterministic n_earth (+ derivative for J1909), all fixed.
* WN: only release `T2EFAC`; EQUAD = ECORR = 0, a **known gap**.

**What we do.** The fixed-noise UL by the paper's recipe, with an autocorrelation-aware ESS of the
reweighted chain (Sec. 5.4).

---

## 3. New components (implementation list)

| # | component | needed for | design | validation |
|---|---|---|---|---|
| N1 | Per-PTA model builders with enterprise-exact semantics: block spans (pulsar span, array span, masked-subset span), `int(T/cadence)` counts, non-harmonic `linspace` grids for PPTA group noise, enterprise ECORR quantisation per selection, `tndm`, `gamma_val=4.333`, separate CURN and HD common grids | EPTA, PPTA | generated from the manifest (Sec. 4.5); never hand-typed | per-block frequencies, normalisations and column space vs enterprise objects of the pinned versions (sin <= 1e-10) |
| N2 | tempo2 evaluator export of every consumed array (Sec. 4.1, T1) | non-NG legs | `scripts/t2py` + oracle env | T1 |
| N3 | Parameter-dependent deterministic residuals: exponential dip (free index), Gaussian DM event, annual DM; fixed mode for MPTA events | EPTA, PPTA, MPTA | **affine update without a dense complement**. With fixed blocks and WN absorbed, c(theta) = c_0 - G d(theta), where G = R^-T [M F]^T N^-1 is applied through the stage-1 whitening operator and the thin QR factors. s_perp(theta) = (r - d)^T N^-1 (r - d) - norm(Q_[M F]^T W (r - d))^2 is evaluated with thin factors only. Cost O(N_toa (n_M + K_a)) per call for affected pulsars; memory O(N_toa (n_M + K_a)) | dense oracle with a dip; enterprise; gradients in the continuous parameters vs FD away from onset crossings |
| N4 | SW GP basis (`createfourierdesignmatrix_solar_dm`) and SW geometry export | PPTA, MPTA | `gp.FourierBlock` | column equality vs enterprise; geometry in T1 |
| N5 | Deterministic SW with finite prior (linear in n_earth; n_earth-dot variant) | PPTA (sampled), InPTA/MPTA (fixed) | precomputed column times scalar, via N3 | vs enterprise `solar_wind` |
| N6 | Free chromatic index at a fixed value | InPTA, MPTA | existing | existing |
| N7 | Reweighting / BF tooling (Sec. 5.4): raw estimator with k-hat diagnostic, batch-means MCSE, weighted-quantile MCSE, per-chain stability, ordered block bootstrap | EPTA, PPTA | `ptagwb.evidence` | synthetic targets with known BF and quantiles, including correlated draws |
| N8 | Stage-2 size buckets for K_a up to ~1,800 and a memory model | EPTA, PPTA | M3_PLAN L6 | padding invariance 1e-10 |
| N9 | Per-parameter prior table from the manifest | EPTA, PPTA | `ModelSpec` | prior-volume check vs chain lnprior (Sec. 4.5) |
| N10 | Hybrid kernel generalisation, the generalised fail-closed gate with the UNRESOLVED class and precedence rule, and conditional-occupancy diagnostics (Sec. 5.3) | sampled runs | `ptagwb.hybrid`, `diagnostics` | invariance tests incl. negative controls |
| N11 | Reference loaders, manifest and fingerprint tooling (Sec. 4.5) | EPTA, PPTA | `scripts/m3b_reference.py`, `scripts/m3b_fingerprint.py` | reproduces Appendix A |
| N12 | Coefficient-Gibbs kernel: **unvalidated research**, only via D5 | PPTA-S | separate design note. Plain conditional updates of power-law hyperparameters mix poorly (van Haasteren & Vallisneri 2014, arXiv:1407.1838, Sec. VI.3); collapsed joint updates are needed | toys with PPTA-like overlapping weak processes, then agreement with NUTS on EPTA |
| N13 | Masked NUTS (NUTS over the continuous coordinates, conditional on the MH-updated event epochs) composed with exact MH for t0 | EPTA, PPTA | Metropolis-within-Gibbs; NUTS caches refreshed after each MH update (as in `ptagwb.hybrid`) | small numerical reference (Sec. 5.1) |
| N14 | CURN common block on per-pulsar spans with shared hyperparameters (if the PPTA grid resolves that way) | PPTA | a per-pulsar sampled block with tied (A, gamma) | dense oracle |

---

## 4. Remaining M3a obligations that block Stage 1

### 4.1 E8 tempo2-parity legs: strategy and the expanded T1

**Problem (unchanged).** E8 fails for every non-NG validation leg: a 1-5 ns floor, and ~30 ns for
PPTA DDK. The named legs are EPTA/PPTA J1600-3053, PPTA J1713+0747 and J2241-5236, plus the DDK (10)
and DDH (21) classes. The post-hoc RN+DM likelihood impact is up to 34 / 58 nats (PPTA J1909 /
J0437), almost entirely from the residuals.

**Options considered.**
* (a) Fix PINT parity: open-ended. The floor on ordinary legs has no identified cause.
* (b) tempo2 as evaluator for non-NG legs: matches the published evaluator; handles signed H3,
  DDK, DDH and FB series as released.
* (c) Quarantine: would empty the PTAs, and changes the target data set.

**Decision proposed (D1): (b), PINT as secondary, E8 kept open.** The reviewer endorses this, with
the conditions below.

* **Evaluator profile `published-tempo2-v1`.** tempo2 2026.04.1 / libstempo 2.5.1 (oracle env,
  hashes pinned):
  * native units and the published clock/ephemeris profile;
  * tempo2 site coordinates (the release convention already adopted in M3a);
  * pre-fit residuals at the par values, as enterprise's libstempo `Pulsar` produces them.

  NG15 stays on PINT.
* **Gate T1 (exactness of every consumed array).** For every leg of a reproduction, compare our
  export with `enterprise.Pulsar(par, tim, ephem=..., timing_package='tempo2')` of the **pinned
  chain-producing enterprise version**, built in the same env. Compared:
  * TOA epochs used by the bases (`psr.toas`, with the time scale stated);
  * residuals and errors (|diff| <= 1e-12 s);
  * **row order and the deletion mask** (TOAs deleted by tempo2 or dropped by enterprise; counts
    and indices identical);
  * radio frequencies **with the convention stated**. Enterprise uses SSB frequencies
    (`ssbfreqs`) and our M3a export uses site frequencies. The published model's chromatic and
    band selections use whatever enterprise used, so we use that convention and record it.
  * flags and backend selections;
  * design matrix column space (sin <= 1e-10) **and numerical rank**;
  * ORF sky positions (`psr.pos`);
  * the time-dependent geometry used by solar-wind terms (`pos_t`, `sunssb`, `planetssb`, or the
    derived impact angle; |diff| <= 1e-12 relative).
* **Released vs canonical inputs.** Enterprise is built twice, from the **original released**
  par/tim (as the PTA did) and from our canonical files, and every array above is compared. This
  catches shared canonicalisation errors (e.g. tempo2 reading a released quirk differently from our
  canonical rewrite). Differences must be zero or explained by a listed rule.
* **Complete roster.** Reproduction acceptance requires every intended pulsar and TOA of the
  published roster to pass T1. A failing leg means **no full-reproduction claim** for that PTA.
  Results without it are a separately labelled **dropout sensitivity analysis**.
* **Gate T2 (engine sensitivity, reported).** Conditional posteriors (fixed released noise, grids)
  with the tempo2 and the PINT evaluator for every leg that loads in PINT, and with the E8-named
  legs swapped one at a time. Shifts in sigma68 / w90 units. A shift <= 0.1 sigma68 means the engine
  choice is immaterial for that PTA. Larger shifts are reported as an evaluator dependence; the
  tempo2 result stays the reproduction.
* **Gate T3 (screen).** tempo2 post-fit wrms vs the par's TRES per leg. The survey's
  residual-excess legs are explained or listed.
* **E8 status.** Re-scoped, not passed. G3/G4 stay as PINT-vs-tempo2 diagnostics with the strict
  xfail. PINT-parity work continues as a non-blocking item.
* **Version risk.** tempo2 2026.04.1 is not the PTAs' 2023 build. Residual-level differences show
  up as fingerprint scatter (Sec. 4.5). Scatter beyond the frozen budget **fails closed**: the
  reproduction claim is withdrawn unless a reviewed manifest revision (e.g. the PTA's own tempo2
  build) brings the scatter within budget.

### 4.2 Published TOA sets vs M3a removals

Published profiles use `clock_coverage = "keep"` (tempo2's own interpolation) and **no duplicate
removal**. The 65 EPTA clock-uncovered TOAs and the conservative removals (EPTA 2, PPTA 3) are not
applied. The M3a defaults are run as a reported conditional sensitivity. The combined profile
(Stage 2) keeps the M3a defaults. T1's roster check uses the published set.

### 4.3 Band-overlap covariance/pruning inventory

The published analyses kept simultaneous overlapping-band recordings (PPTA 16,186 channel pairs,
EPTA 235, NG15 19). A reproduction keeps them too, so this does not block Stage 1 acceptance. M3b
deliverables (in M3b-0X, not on the EPTA path):
1. Inventory per pulsar and system pair:
   * pair counts and their share of the TOA weight;
   * channel overlap;
   * whether the published model already correlates the pair. For PPTA, check whether the pair
     falls in one epoch of the global UWL / non-UWL ECORR under enterprise's dt = 1 s quantisation
     [UNCERTAIN for PDFB4/CASPSR], and in a band-ECORR selection.
2. Conditional sensitivity: the partner pruned (newer backend kept) vs kept.
3. The Stage-2 decision (pruning rule or a cross-backend term) is deferred to M3c-prep.

### 4.4 Other open items, and why they do not block Stage 1

* **E7 / option C.** One leg per pulsar per PTA, so no reference copy. Moved to M3c-prep, together
  with the re-linearisation remedy.
* **J1022 / J0437 admissibility.** M3c-prep:
  1. align binary models across legs (J1022 PPTA DDH with frozen H3/STIG vs NG15 DD; J0437 DDK
     KIN/KOM free vs frozen);
  2. per-leg DM;
  3. joint re-linearisation;
  4. admissibility re-run (threshold 0.1 unchanged).

  Under D1 the value copy happens in tempo2 par space, as in YA.
* **E2-C** stays on record. Per PTA, **G5-PTA** applies:
  * value and gradient of our likelihood against the long-double arbiter, extended to N3/N5 terms
    (shape <= 1e-6 nats, gradient <= 1e-8 relative, at points away from onset crossings);
  * against enterprise (pinned version), with a pre-declared 1e-4-nat budget at > 40k TOAs;
  * the **cross-model check** (Sec. 6.2).
* **Signed H3 (MPTA J1825-0319)**: under tempo2. If T1 fails it is not admitted, and the MPTA
  cross-check is labelled with the roster change.

### 4.5 Model identity: version-pinned prior/model manifest (M3b-0E / M3b-0P exit condition)

For each full-reproduction target, `configs/m3b/manifest_<pta>.json` must hold, for every
parameter and block, the following with its provenance (repository, commit, file, line):
* prior family and bounds;
* basis type, frequency labels and spacing rule, span definition and normalisation;
* row selection and ECORR quantisation;
* chain-producing dependency versions: enterprise, enterprise_extensions, PTMCMCSampler, tempo2,
  libstempo and their clock/EOP files, where recoverable.

It is verified in three ways:

1. **Source audit.**
   * EPTA: the modified **enterprise and enterprise_extensions** of tag `EPTADR2-v1.1`
     (gitlab.in2p3.fr) are fetched and pinned in the oracle env (D6). Every function `model_general`
     calls on the DR2new path is read; frequency/span rules, `tndm`, dip parameterisation and all
     priors are recorded.
   * PPTA: the extensions version from the Singularity image (CSIRO DAP) or its package list.
     Failing that, a grid hypothesis is accepted only if (i) the code of the candidate version
     that produces it is identified *and* (ii) the item-3 discrimination is decisive. The
     fingerprint alone never identifies the source model.

   Running the fork in production is not required, if our replacement is shown equivalent
   (items 2-3). **Source identification is the primary evidence.** Items 2-3 are supporting checks
   at finitely many released points; they cannot establish the prior bounds individually, the
   absolute normalisation, or agreement away from the evaluated draws.
2. **Prior-volume check (physical vs logged prior).**
   * The **physical** prior of a model is the normalised product of its parameter priors:
     ln pi(x) = -sum_j ln(width_j) for uniform boxes, over the model's *unique* parameters.
   * The **logged** density is whatever the chain-producing sampler wrote as lnpost - lnlike.
     * Single-model PTMCMC runs log the physical prior.
     * **Product-space runs** (`enterprise_extensions.hypermodel.HyperModel`) log
       `get_lnprior`, which **sums the constituent models' log priors**. Shared noise priors are
       counted twice, each model's own common block once, plus the model-index prior (`nmodel`
       uniform on [-0.5, 1.5], log density 0 under the logged convention).
     * The released PPTA product-space offsets are -935.211603 (free gamma) and -930.200551
       (fixed) = 2 x the single-model -467.605801 and -465.100276. Both models share all noise
       priors and have common blocks of equal width.
   * The manifest records the physical prior and, separately, the **logging convention** of each
     reference file: multiplicities, model-index convention, constants.
   * The check compares our reconstruction *of the logged quantity* with every row of every
     released chain (EPTA crn_pl, hd_pl; PPTA freegam, fixgam and both product-space chains).
     Tolerance: <= 1e-5 nats in the mean, and a row spread consistent with six-decimal rounding
     (sd <= 1e-6).
   * Physical bounds are never adjusted to absorb a logging convention.
   * Status: EPTA passes for the candidate manifest (mean agreement ~1e-9). PPTA single-model
     volumes are reproduced to a constant -0.003872 nats (unresolved). The product-space factor of
     2 is explained by the convention above, which the check must reproduce explicitly.
3. **Chain fingerprint (supporting, fail-closed).** At >= 50 released draws per chain, spread over
   the chain, compute Delta_i = lnL_ours(x_i) - lnlike_ref(x_i) in float64. Reference values are
   read at their stored precision and never cast to float32: at these magnitudes float32 alone
   adds ~0.04-nat scatter.
   * **Frozen error budget per draw:** sigma_i^2 = sigma_store^2 + sigma_par,i^2 + sigma_eval^2.
     * sigma_store: storage rounding of lnlike (six decimals: uniform +-5e-7, sd 2.9e-7).
     * sigma_par,i: propagated parameter rounding, |grad lnL(x_i)| . delta_x_i. Here delta_x_i is
       the stored precision: EPTA text values carry ~22 significant digits; PPTA values are
       float64.
     * sigma_eval: our evaluator error, bounded by the arbiter-validated 1e-6 nats (G5-PTA).
   * **Pass:** the standardised scatter of Delta_i about its mean, chi^2 = sum((Delta_i - mean)^2 /
     sigma_i^2) / (n - 1), is <= 2. Equivalently, the observed scatter is within the budget.
   * **Anything above is FAIL** (closed), whatever its suspected cause, including tempo2-version
     residual differences. A failing model cannot carry a full-reproduction claim. It can be
     rescued only by a revised, re-reviewed manifest (e.g. the PTA's actual tempo2 build in the
     oracle env) that brings the scatter within budget. Otherwise the result is downgraded.
   * **Grid discrimination (PPTA CURN/HD spans):** a hypothesis is accepted only if it passes
     **and** every alternative fails with a standardised scatter >= 100 (its disagreement exceeds
     the error budget by >= 10x in rms). If both pass, or neither, the grid is unresolved (R3).
   * **Model-dependent offsets.** Each model m has a fitted constant c_m = mean_i(Delta_i). The
     fingerprint does not constrain c_HD - c_CURN, which shifts ln BF one-for-one. The BF
     normalisation is established by the **cross-model oracle check** (Sec. 6.2): ours vs the
     pinned enterprise, including all constants. The fingerprint's c_HD - c_CURN is reported and
     must agree with the oracle-predicted difference within the combined budget.
   * Cost: ~50-200 likelihood values per chain (CPU, or < 0.1 GPU-h).

**No inferred-prior fallback.** If any item stays unresolved, the PTA's result is downgraded to
"approximate comparison (model identity unresolved: <items>)" and is not called a reproduction.

---

## 5. Sampling strategy

### 5.1 Kernel per model

| model | parameters | kernel |
|---|---|---|
| EPTA CURN^gamma | 67 | NUTS (fixed dense metric from a pilot) on the 66 continuous coordinates **excluding the dip t0**. Exact MH for t0 each iteration (N13): random walk within the window plus a 20 % independence draw from the prior window. Frozen block-MH jumps for shelf-prone (log10 A, gamma) pairs (Sec. 5.3 definition) and a joint (t0, log10 tau, log10 Amp) independence block fitted from pilot draws |
| EPTA CURN gamma = 13/3 (for E-5) | 66 | same |
| EPTA HD^gamma, HD 13/3 | 67 / 66 | reweighting of the corresponding CURN draws first (Sec. 5.4); the same kernel as CURN only if reweighting fails |
| PPTA P-C0 | 2 / 1 | deterministic grids on the established CURN and HD grids |
| PPTA full (M3b-PPTA-F only) | ~262 | D5 |
| MPTA, InPTA | 2 / 1 | grids, or NUTS on 2 parameters |

**Event epochs.** The released exponential waveform switches on with a Heaviside step at t0. The
likelihood is discontinuous in t0 whenever t0 crosses a TOA, so neither NUTS dynamics nor
derivative checks are valid in t0. t0 is therefore never a NUTS coordinate. The other event
parameters are continuous given t0 (amplitude, timescale, chromatic index; the Gaussian DM event's
epoch is smooth). **Validation** of the composed kernel:
* a single-pulsar toy with one dip and realistic TOA gaps, against a numerical reference: 2-D
  quadrature over (t0, log10 Amp) with the other parameters fixed, plus a 3-D grid with log10 tau.
  t0 interval occupancies (the inter-TOA intervals with >= 1 % mass) must agree within 3.5 combined
  SEs, and the marginal quantiles of t0 and log10 Amp likewise;
* a negative control: the same kernel with the MH ratio's proposal term removed must fail;
* the EPTA J1713 conditional: t0 | others on a fine grid vs the kernel's conditional draws.

### 5.2 Warmup, initialisation, metric (FS_PILOT and PERF lessons)

* Never start dual averaging at step 1.0. Use the step size measured in the pilot.
* Fixed dense metric from pilot draws. No windowed adaptation in production.
* Overdispersed inits from our own pilot draws, with randomised shelf/peak starts. Never from the
  reference chain under test.
* 4-8 vectorised chains. Lockstep loss is reported.
* Committed configs, recorded SHA and pre-set abort rules: stop at 1.5x the allocation.
* Jump proposals: 20 % prior + 80 % equal-mass histogram, fitted from **our** pilot draws only,
  frozen with sha256 before production.

### 5.3 Pre-registered convergence gate (fail-closed)

PASS only if all of the following hold; otherwise FAIL or INCONCLUSIVE.

1. **All sampled parameters** (including MH-updated ones): rank-normalised split R-hat < 1.01;
   bulk and tail ESS >= 400.
2. **Target parameters:** bulk and tail ESS >= 1,000. MCSE of every Sec. 6 quantity small enough
   that the decision is not limited by our sampling (Sec. 6.1, max_our_MCSE column).
3. **Transport, by parameter class.** Region occupancies and their MCSEs in the *reference* are
   computed as below, frozen with the reference's sha256 and burn-in. The same rules apply to our
   own chains.
   * **(a) Shelf-prone amplitudes** (noise and common log10 A). Shelf S = [lower bound, lower
     bound + 1 dex], peak P = the complement.
     * **Occupancy intervals, applied identically to reference chains and our chains.**
       * **Estimable case.** The region indicator 1[x in R] has >= 10 entries and >= 10 exits pooled.
         Then its own integrated autocorrelation time tau_R is estimable, and the 95 % interval
         [p_lo, p_hi] is the Wilson score interval on n_eff = N / tau_R. tau_R comes from the
         indicator itself, never from the amplitude.
       * **Few-event case.** Some visits, but fewer than 10 entries or exits. tau_R is not
         estimable, so the interval is **UNRESOLVED**.
       * **Zero-visit case.** The interval is **UNRESOLVED**. The only exception is a *certified*
         bound (below). The amplitude's (or t0's) bulk ESS, split-half agreement and similar
         within-region mixing diagnostics are **not** such a bound. Fast mixing
         inside the peak says nothing about entries into an unvisited shelf. Example (from the
         review): a two-region chain with entry and exit probabilities 1e-5 / 9e-5 per draw has 10 %
         shelf mass and zero visits in 1,000 draws with probability 0.89. Revision 3's ESS-based
         bound would have given 0.003 there; it is withdrawn.
       * **All-visit case.** Handled as the zero-visit case of the complementary region, so
         p_hi(R) = 1 always. R's support then follows from the complement's status, and is
         **UNRESOLVED** when the complement's is.
     * **Certified bounds only.** An UNRESOLVED interval can be resolved without visits only by
       a bound whose justification does not rest on empirical convergence diagnostics:
       * a bound on P(R | y) that is certified over **all** conditioning states y in the prior box
         (e.g. by rigorous global optimisation of the conditional with validated numerics); or
       * a proven global mixing or error bound for the kernel used.

       Neither is available or planned in M3b, so in practice **zero- and few-event regions stay
       UNRESOLVED**.
     * **Rao-Blackwellised conditional occupancy is a supporting diagnostic only** (revision 4's
       escape hatch is withdrawn). At >= 2,000 retained draws compute
       pi_i = P(x_R in R | all other parameters at x_i):
       * 2-D quadrature over (log10 A, gamma) for amplitudes;
       * an exact piecewise sum over inter-TOA intervals for t0.

       Report mean(pi), its batch-means MCSE, ESS, maximum single-draw contribution and split-half
       agreement. These estimate p(R) only over the conditioning states the chains visited. The
       review's analytic counterexample shows the risk: two slowly communicating basins of mass
       0.9 / 0.1 with conditional occupancies ~1e-5 / 0.99 give a true mass of 0.099, while chains
       confined to the first basin pass every one of these checks with an estimate of ~1e-5. The
       diagnostic is reported, never used to classify support.
       * Cost (projection): CPU or < 0.2 GPU-h per PTA.
     * **Support classes** (from the resolved interval): **material** if p_lo >= 0.01; **absent**
       if p_hi < 0.01; **ambiguous** otherwise; **UNRESOLVED** as above.
     * **Gate decision per (parameter, region pair)**, combining the reference and our run, evaluated
       **in this order**:
       1. **Precedence rule.** If *any* occupancy interval entering the decision is UNRESOLVED
          (reference or ours, for either region of an amplitude pair or any event interval), the
          status is **INCONCLUSIVE**. The rows below are not evaluated, so an unresolved occupancy
          can never produce FAIL or PASS.
       2. Otherwise, with all intervals resolved:

       | reference support | our run (all intervals resolved) | transport status |
       |---|---|---|
       | both regions material | bidirectional transport (>= 10 entries and >= 10 exits pooled, >= 2 chains, longest sojourn <= 50 %, occupancy MCSE <= 0.01) and interval overlap with the reference | PASS |
       | both material | transport criteria not met, or intervals disjoint | FAIL |
       | one region absent | our interval for that region overlaps the reference's and p_hi,ours < 0.02 | PASS |
       | one region absent | our interval disjoint from the reference's, or p_lo,ours >= 0.02 | FAIL |
       | ambiguous | bidirectional transport and interval overlap | PASS |
       | ambiguous | otherwise | INCONCLUSIVE |

       Because of the precedence rule, an UNRESOLVED occupancy never passes and never hard-fails.
     * **Gate aggregation:**
       * The run's convergence gate is FAIL if any item is FAIL; INCONCLUSIVE if any item is
         INCONCLUSIVE and none FAIL; PASS only if all pass.
       * **INCONCLUSIVE blocks acceptance exactly as FAIL does.** A PTA whose gate is INCONCLUSIVE
         gets the verdict INCONCLUSIVE (Sec. 6.1), never REPRODUCED.
       * Its agreement numbers are still computed and reported, labelled "not accepted: unresolved
         regions <list>".
       * Remedies: run longer, or add targeted jump proposals for the unresolved block. Tolerances
         and classes are not changed.
     * **Consequence (flagged for the user, D9).** Shelves the posterior genuinely does not visit
       stay UNRESOLVED in both the reference and our chains, because no certified bound exists.
       * 17 EPTA CURN-chain amplitudes have zero shelf visits after burn-in. PPTA's free-gamma
         common amplitude has a reference minimum of -16.27.
       * Under this rule EPTA (and PPTA-F) **cannot reach REPRODUCED** unless those shelves are
         genuinely visited.
       * The expected EPTA outcome is therefore INCONCLUSIVE with the unvisited regions listed,
         even if every headline quantity is EQUIVALENT.
       * Whether to define a separately labelled, reviewed claim class (e.g. "equivalent
         conditional on negligible mass in the listed unvisited regions") is decision D9.
         **Adopted by the user on 2026-10-09** as the class "CONDITIONALLY EQUIVALENT TO
         RELEASED RESULTS" (Sec. 6.6). It is reported *alongside* the unconditional verdict,
         which stays INCONCLUSIVE; it changes none of the rows or the precedence rule above.
   * **(b) Event epochs t0.** Intervals = the inter-TOA gaps that hold >= 1 % reference mass, plus
     one "rest of window" bin. Interval occupancies use exactly the construction and decision table
     of (a), including the precedence rule: estimable from the interval indicator's own events,
     otherwise UNRESOLVED (no certified bound is planned); the all-visit case goes through the
     complement. The conditional t0 occupancy is a supporting diagnostic only.
     Every material interval must also be visited by >= 2 chains with occupancy MCSE <= 0.02. R-hat
     and ESS of t0 as in item 1.
   * **(c) Other parameters** (gamma, timescales, indices, n_earth, phases): items 1-2 only.
4. **NUTS sub-steps:** zero divergences after warmup; any divergence makes the run INCONCLUSIVE
   pending explanation. This is a conservative rule, not a proof of correctness, and it does not
   apply to the MH/Gibbs components. Those report per-block acceptance rates and pass item 3.
5. No configuration drift (no dirty files, run metadata matching the config).
6. G5-PTA, T1 and the manifest checks (Sec. 4.5) passed for the exact model and SHA before the
   run.

### 5.4 HD posteriors and Bayes factors

Applies wherever HD (or HD 13/3) is obtained from CURN draws x_i (i = 1..N, C chains).

**Precondition.** Identical shared parameters and priors in both models. The two models may differ
in **ORF and common grid** (PPTA; possibly EPTA), so the weight is the full likelihood ratio
w_i = L_HD(x_i) / L_CURN(x_i), with each model's own common basis and normalisation. Any
normalisation difference must pass the cross-model oracle check (Sec. 6.2).

**Estimators.**
* **Accepted estimator: raw only.** BF_raw = mean(w); HD quantiles are weighted quantiles with
  raw normalised weights.
* Pareto k-hat (PSIS tail fit; Vehtari et al., arXiv:1507.02646) is computed on the pooled
  log-weights and per chain, as a **diagnostic only**.
* **Acceptance requires k-hat < 0.5**, pooled and in every chain. In that regime the raw estimator
  has finite variance, and the dependence-aware MCSEs below are meaningful.
* k-hat >= 0.5: reweighting is **not accepted**, and the existing fallback applies (more CURN
  draws do not help; sample HD directly, then reverse reweighting / bridge). PSIS-smoothed
  estimates may be reported for information, never used for acceptance.
* Using PSIS for acceptance in 0.5 <= k-hat < 0.7 would need its own validated uncertainty
  procedure (e.g. PSIS refitted inside a validated moving-block resampling of the ordered chains,
  with coverage tested on synthetic heavy-tailed dependent weights). That requires a reviewed plan
  revision.

**Uncertainty (autocorrelation-aware).** Kish ESS is reported but never gates, because it ignores
serial dependence.
* MCSE of ln BF comes from overlapping batch means on each chain's ordered weight series
  (batch length >= 5x the integrated autocorrelation time of w), combined across chains, and
  cross-checked by a moving-block bootstrap.
* Weighted quantile q_p is a ratio estimator: sum_i w_i 1[x_i <= q] / sum_i w_i = p. Its MCSE
  comes from batch means of the pair (w_i 1[x_i <= q_p], w_i) with the delta method, converted to
  quantile units by the weighted density at q_p (kernel estimate, bandwidth recorded).

**Independent-chain stability.** Per-chain ln BF and weighted medians must be mutually
consistent: chi-squared of the per-chain estimates about their mean, p > 0.01, using per-chain
MCSEs. No single chain may carry > 50 % of the total weight.

**Precision gates on the HD outputs.** Sec. 6's max_our_MCSE for each HD quantile, and
MCSE(ln BF) <= 0.10. If either fails: more CURN draws (cheap), then an HD run.

**Fallback.** An HD NUTS run, then reverse reweighting and bridge sampling. Their spread enters the
reported estimator uncertainty.

**Evidence ladder (only via D8).** If the CURN and HD grids are equal, an ORF path
Gamma_lambda = (1 - lambda) I + lambda Gamma_HD connects the endpoints. If they **differ**, the
ORF-only path has the wrong CURN endpoint. Use the covariance path
C_lambda = (1 - lambda) C_CURN + lambda C_HD (both common blocks present, tied hyperparameters).
Its endpoints are exactly the two published models, and every intermediate covariance is positive
definite.

**PPTA product-space reference.**
* BF_ref: the HD-active count ratio requires equal model weights. The weights are verified from
  the HyperModel configuration in the manifest; if they are unresolved, P-6 is "reference
  unresolved".
* Its SE comes from a moving-block bootstrap of the **original ordered chain**, keeping model
  occupancy: blocks >= 5x the autocorrelation time of the nmodel indicator.
* HD-active reference quantiles are computed as indicator-weighted quantiles of the full ordered
  chain, with the same ratio-estimator MCSE as above. They are not computed from an extracted
  subsequence treated as a chain.

### 5.5 Cost projections (projections; replaced by measured M3b-0 benchmarks)

**Measured anchors** (RTX 5090, float64; PERF Sec. 2-4, M2_RESULTS Sec. 4):
* NG15 HD value+gradient: 15.0 ms (B = 1) and 5.8 ms per chain (B = 16), of which the per-pulsar
  stage is 3.5 ms (67 pulsars, K = 60, 0.14 GF at 0.04 TFLOP/s). NG15 CURN: 3.6 ms.
* M2 CURN^gamma (4 x 1500): 8.3e5 chain-gradients in 25 min. HD^13/3 (4 x 500): 3.0e5 in 61 min.

**Projection model.** The per-pulsar stage is ~3.3 K_a^3 flops forward and ~2x more for the
gradient. Efficiency is bracketed between 0.04 TFLOP/s (measured small blocks) and ~1.3 TFLOP/s
(measured batched Cholesky). Chain-gradient counts are 1.5-3x M2's, for the stricter
all-parameter gate.

| item | sum K_a^3 / NG15 | projected value+grad (B = 1) | chain-gradients per run | GPU-h per run (projection) |
|---|---|---|---|---|
| EPTA CURN^gamma or CURN 13/3 (bucketed) | ~16 | 10-60 ms | 1-3 x 10^6 | 3-50 (central ~5) |
| EPTA HD reweighting (2 x 10^4 draws, value only) | ~16 | 5-30 ms | - | < 0.2 |
| PPTA P-C0 grids (fixed noise) | - | 2-10 ms (value) | 2 x 10^4 points per grid x (CURN, HD) | < 0.2 |
| PPTA full model, NUTS + jumps | ~760 before group noise; J0437 K_a 1,388 -> **1,816** with its ~428 group columns | 0.1-1.5 s | 2-5 x 10^6 | **56 (2e6 x 0.1 s) to 2,083 (5e6 x 1.5 s)** |
| PPTA full model, coefficient-Gibbs (N12) | - | - | - | **not projected: unvalidated speculation** (cost per sweep ~1 Cholesky per pulsar, but mixing unknown; collapsed updates required) |
| MPTA / InPTA conditional | fixed noise | 2-20 ms (value) | grids | < 0.5 each |

PPTA full-model totals would multiply by the number of runs: free gamma and 4.333, plus any HD
fallback. Batching (B = 4) is assumed to give no per-chain gain at these block sizes. Memory for
batched 1,800^2 blocks is not yet measured.

**Benchmarks (D3).** M3b-0E: EPTA value+gradient, value-only, memory at B in {1, 4, 8}, and 50
fixed-step NUTS transitions (<= 0.5 GPU-h). M3b-0P: the same for P-C0 grids (and, only if D5 asks,
the full PPTA model) (<= 0.5 GPU-h).

---

## 6. Pre-registered acceptance criteria

### 6.1 Decision rule (all PTAs)

For each quantity: D = q_ours - q_ref, SE_D = sqrt(MCSE_ours^2 + MCSE_ref^2), with the MCSEs from
Sec. 5.4 (quantile MCSE; ratio-estimator MCSE for weighted outputs and for product-space
references). Let I_D = D +- 1.645 SE_D (a 90 % interval: two one-sided tests at 5 %). With
practical margin m:

* **EQUIVALENT** if I_D lies within [-m, m];
* **INCOMPATIBLE** if I_D lies entirely outside [-m, m];
* **INCONCLUSIVE** otherwise. It is **reference-limited** if 1.645 MCSE_ref >= m (more sampling on
  our side cannot resolve it), and **ours-limited** otherwise. To make "ours-limited" avoidable, the
  convergence gate requires MCSE_ours <= max_our_MCSE = sqrt((m / 1.645)^2 / 4 - MCSE_ref^2), floored
  at 0.2 m / 1.645. This means our sampling uses at most half the decision budget where the
  reference allows it.

Also reported, not gating: Tier A, i.e. statistical distinguishability |D| > 3.5 SE_D (an
EQUIVALENT but distinguishable result is reported as such); 2-D KS and energy distances;
per-noise-parameter differences (count beyond 3.5 SE_D vs the expected count); T2; TOA-set and
band-overlap sensitivities.

**Margins (provisional, D2):**
* m = 0.15 sigma68_ref for medians and 0.25 sigma68_ref for outer quantiles;
* ln BF: m = 0.30 nats (EPTA), 0.40 (PPTA);
* InPTA UL: 0.15 dex.

**Verdict per PTA.**
* **REPRODUCED**: every *decidable* headline quantity is EQUIVALENT; no headline is INCOMPATIBLE;
  the convergence gate, G5-PTA, T1 (complete roster) and the manifest checks pass.
* **NOT REPRODUCED**: any headline INCOMPATIBLE. The cause is traced (evaluator version, TOA set,
  builder semantics, priors, grid) and tolerances are not retuned.
* **INCONCLUSIVE**: any headline ours-limited INCONCLUSIVE, or a gate unmet.
* **CONDITIONALLY EQUIVALENT TO RELEASED RESULTS** (D9, Sec. 6.6): a separately labelled,
  weaker claim reported *in addition to* an unconditional INCONCLUSIVE verdict, only when that
  verdict is caused exclusively by zero-visit regions listed in the frozen union U and every
  condition of Sec. 6.6 holds. It is never reported as REPRODUCED.

Reference-limited quantities are listed as such, and are moved out of the headline set *now*,
before any run (tables below), so that the verdict cannot hinge on reference MC noise.

### 6.2 EPTA DR2new

Reference MCSEs from Appendix A. "Decidable" = 1.645 MCSE_ref < m.

| ID | quantile | q_ref | MCSE_ref | m | 1.645 MCSE_ref | decidable | role |
|---|---|---|---|---|---|---|---|
| E-1 | HD log10 A q05 | -14.418 | 0.0368 | 0.046 | 0.061 | **no** (reference-limited) | reported |
| E-1 | q50 | -13.935 | 0.0046 | 0.027 | 0.008 | yes | headline |
| E-1 | q95 | -13.702 | 0.0038 | 0.046 | 0.006 | yes | headline |
| E-2 | HD gamma q05 | 1.978 | 0.0127 | 0.131 | 0.021 | yes | headline |
| E-2 | q50 | 2.710 | 0.0121 | 0.079 | 0.020 | yes | headline |
| E-2 | q95 | 3.894 | 0.0704 | 0.131 | 0.116 | yes (narrowly) | headline |
| E-3 | CURN log10 A q05 | -14.766 | 0.0581 | 0.064 | 0.096 | **no** | reported |
| E-3 | q50 | -13.997 | 0.0067 | 0.038 | 0.011 | yes | headline |
| E-3 | q95 | -13.718 | 0.0040 | 0.064 | 0.007 | yes | headline |
| E-4 | CURN gamma q05 | 2.038 | 0.0190 | 0.167 | 0.031 | yes | headline |
| E-4 | q50 | 2.906 | 0.0169 | 0.100 | 0.028 | yes | headline |
| E-4 | q95 | 4.626 | 0.1069 | 0.167 | 0.176 | **no** | reported |
| E-5 | HD (gamma = 13/3) log10 A q50, q05, q95 | -14.61 / -14.73 / -14.50 (paper, rounded, slice method unknown) | not available | 0.035 / 0.03 / 0.03 | - | classified with SE_D = MCSE_ours plus a reference rounding term of 0.005 (half the last digit), labelled "reference uncertainty incomplete" | secondary |
| E-6 | ln BF(HD/CURN) | ln 60 = 4.094 | - (see below) | 0.30 | - | yes | headline |

**E-6 rule.**
* Reference treatment: the published 60 is a point target with no calibrated SE. EPTA's other
  estimates (56-66, i.e. ln 4.03-4.19) are context, and the 0.30-nat margin is ~3.75x their
  half-range.
* Our SE: MCSE(ln BF) from Sec. 5.4 plus the spread of our estimators (raw, and reverse reweighting and bridge if
  run) added in quadrature. **Maximum allowed SE: 0.10 nats**; above it E-6 is INCONCLUSIVE.
* Classification: EQUIVALENT if ln BF_ours +- 1.645 SE lies within [4.094 - 0.30, 4.094 + 0.30];
  INCOMPATIBLE if the interval lies entirely outside; otherwise INCONCLUSIVE. Valid only after the
  overlap rules (Sec. 5.4) and the cross-model check pass.

**Cross-model oracle check (G5-PTA addition).** At >= 12 random shared-parameter points:
* (lnL_HD - lnL_CURN)_ours vs the arbiter (<= 1e-6 nats) and vs enterprise of the pinned version
  (<= 1e-4 nats), **including every normalisation constant**: the timing marginalisation, GP-block
  log-determinants, and each model's common-grid normalisation;
* both likelihoods' absolute values against enterprise with a common constant convention, so that
  no model-dependent constant can hide;
* the fingerprint (Sec. 4.5) on both released chains is supporting evidence of each model's
  likelihood *shape* at the released draws only. Its constants c_HD and c_CURN are free, so it does
  not validate the BF normalisation; the oracle checks above do. The fingerprint's c_HD - c_CURN
  must agree with the oracle-predicted difference within the budget.

### 6.3 PPTA DR3

**P-C0 (default; gating for PPTA code, not a reproduction claim).** On identical inputs, fixed
noise, and the established grids:
* lnL-shape differences <= 1e-6 nats (arbiter) and <= 1e-4 nats (enterprise);
* cross-model lnL_HD - lnL_CURN within the same tolerances;
* conditional posterior quantiles within grid resolution (0.002 dex in log10 A, 0.005 in gamma).

**Full reproduction (only M3b-PPTA-F).**

| ID | quantile | q_ref | MCSE_ref | m | 1.645 MCSE_ref | decidable |
|---|---|---|---|---|---|---|
| P-1 | CURN log10 A q16 / q50 / q84 | -14.655 / -14.496 / -14.358 | 0.0088 / 0.0053 / 0.0078 | 0.037 / 0.022 / 0.037 | 0.014 / 0.009 / 0.013 | yes |
| P-2 | CURN gamma q16 / q50 / q84 | 3.505 / 3.869 / 4.228 | 0.0220 / 0.0151 / 0.0155 | 0.090 / 0.054 / 0.090 | 0.036 / 0.025 / 0.025 | yes |
| P-3 | CURN 4.333 log10 A q16 / q50 / q84 | -14.735 / -14.681 / -14.634 | 0.0017 / 0.0014 / 0.0017 | 0.013 / 0.008 / 0.013 | 0.003 / 0.002 / 0.003 | yes |
| P-4 | HD log10 A, gamma (reweighted) | -14.709 / -14.514 / -14.337; 3.444 / 3.906 / 4.389 | ordered-chain ratio-estimator MCSE (Sec. 5.4), to be frozen in M3b-0P | 0.047 / 0.028 / 0.047; 0.118 / 0.071 / 0.118 | TBD at freeze | decided at freeze |
| P-5 | HD 4.333 log10 A | -14.734 / -14.675 / -14.615 | as P-4 | 0.015 / 0.009 / 0.015 | TBD | decided at freeze |
| P-6 | ln BF(HD/CURN) | paper ~1.5 / ~2; count ratios 1.85 / 1.69 | ordered block bootstrap | 0.40 | - | secondary: classified against the count ratio if the model weights are verified, else "reference unresolved"; the paper's rounded values are context |

### 6.4 MPTA and InPTA (approximate cross-checks)

Outcomes: CONSISTENT / TENSION / UNAVAILABLE. They never gate anything.
* **MPTA.** CONSISTENT if, for log10 A and gamma, our conditional median lies in the published
  68 % interval and the published median lies in ours. Reported: clock variants and
  J1825-0319 inclusion.
* **InPTA.** I-1: CONSISTENT if |UL_ours - (-13.47)| <= 0.15 dex, with the autocorrelation-aware
  ESS of the LinearExp reweighting >= 400. I-2 the same with -13.55. I-3: the published median lies
  in our 68 % interval.

### 6.5 Frozen before any run (committed, reviewed)

* `configs/m3b/acceptance_<pta>.json`: every Sec. 6 number, the decidability table and
  max_our_MCSE;
* the manifests (Sec. 4.5);
* reference identities (path, sha256; burn-in: EPTA first 25 % of `chain_1.txt`; PPTA as
  released);
* relevance/transport declarations (`configs/m3b/relevance/*.json`);
* model configs and profiles;
* the T2 template.

The gate code fails closed without these files.

### 6.6 D9 claim class: "CONDITIONALLY EQUIVALENT TO RELEASED RESULTS" (adopted 2026-10-09)

Adopted by the user as specified in the round-5 review (`review_m3b_r5.out`). The unconditional
verdict of Sec. 6.1 is unchanged and stays INCONCLUSIVE; this class is reported next to it.

**Reporting template (verbatim; the bracketed parts are filled from the frozen files and the
results):**

> **CONDITIONALLY EQUIVALENT TO RELEASED RESULTS.** All predeclared decidable headline
> quantities meet the equivalence criteria conditional on excluding the explicitly listed
> regions U. Both retained sample sets contain zero visits to U; its posterior mass remains
> unresolved. Extension to the unrestricted posterior assumes P_m(U) <= epsilon_m for each
> relevant model, with numerical thresholds and sensitivity results reported below. These mass
> assumptions have not been established by the chains. This claim is weaker than REPRODUCED; the
> unconditional verdict remains INCONCLUSIVE.

**Conditions (all required):**
1. **Scope frozen before production.** Every excluded parameter region and event interval is
   listed with its exact boundaries, the model it applies to, the reference file and its sha256,
   and the burn-in, in `configs/m3b/acceptance_<pta>.json` (`d9.excluded_regions`), committed
   before any production run. **U is defined as their union** (per model): seventeen individually
   small masses need not have a small union mass. Regions are added to U only from the reference
   occupancy table frozen with it; U is never edited after our chains are seen.
2. **Eligibility.**
   * Zero retained visits to every listed region, in **both** the reference and our retained
     samples (after burn-in / warmup). Warmup visits are reported separately.
   * A few-event case (some visits, < 10 entries or exits) or asymmetric visitation (a region
     visited by one sample set and not the other) is not eligible; it needs a separate review.
     Samples are never discarded to qualify.
   * Every other applicable check passes: model identity (manifest, prior volume, fingerprint),
     numerical validation (T1, G5-PTA incl. the cross-model check), the convergence gate with
     every item other than the UNRESOLVED occupancies of regions in U (i.e. mixing within the
     retained domain U^c), reweighting diagnostics (k-hat, MCSE, per-chain stability, Sec. 5.4),
     and headline equivalence (every decidable headline quantity EQUIVALENT under Sec. 6.1,
     computed from the retained samples, i.e. conditional on U^c).
   * This class is distinct from the fixed-noise "conditional implementation check" (Sec. 1.1).
3. **Missing-mass sensitivity (each model m and each headline marginal).** With F_0 the
   retained-sample CDF (conditional on U^c) and G the unknown distribution inside U,
   F_p(x) = (1 - p) F_0(x) + p G(x), so (1 - p) F_0(x) <= F_p(x) <= (1 - p) F_0(x) + p. Inverting
   the envelope gives worst-case bounds of the alpha-quantile:
   q_lo(alpha; p) = F_0^-1((alpha - p) / (1 - p)) (the prior's lower bound if alpha <= p) and
   q_hi(alpha; p) = F_0^-1(alpha / (1 - p)) (the prior's upper bound if alpha / (1 - p) >= 1).
   * Evaluated at **p in {0.001, 0.01, 0.05, 0.10}**, for our estimate and the reference
     estimate **separately** (their missing-region effects are not assumed to cancel): the
     worst-case difference interval is [q_lo,ours - q_hi,ref, q_hi,ours - q_lo,ref] widened by
     +- 1.645 SE_D; the equivalence conclusion survives at p if that interval lies in [-m, m].
   * Reported per quantity: **p*** = the smallest mass at which the conclusion can change
     (bisection on p, with its Monte Carlo uncertainty from the quantile MCSEs).
   * **epsilon_m** (per model) = the largest grid value p <= min over that model's headline
     quantities of the lower 90 % Monte Carlo bound of p*. The class requires
     epsilon_m >= 0.001 for every relevant model; otherwise it is not available.
   * Reweighting existing draws cannot recover an unvisited region. Any hypothetical G used for
     illustration is labelled as an assumption.
4. **Evidence sensitivity.** With the same excluded domain and shared priors, B_0 = the ratio of
   the evidence integrals over U^c, and B_full = B_0 (1 - p_CURN) / (1 - p_HD). Reported: ln B_full
   - ln B_0 on the grid p_CURN, p_HD in {0, 0.001, 0.01, 0.05, 0.10}. E-6 qualifies only if
   ln B_0 +- 1.645 SE, widened by [ln(1 - epsilon_CURN), -ln(1 - epsilon_HD)], lies within the
   E-6 margin. Small missing CURN mass alone protects neither the HD summaries nor the BF; HD
   (reweighted from CURN draws) uses its own epsilon_HD, with U evaluated on the HD-weighted
   draws (zero visits) and the HD reference chain.
5. **Unconditional verdict.** It stays INCONCLUSIVE (Sec. 5.3 precedence rule) and is reported
   first; this class is reported next to it with U, epsilon_m, every p* and the BF table.

---

## 7. Milestones and budget

Every sub-milestone ends with an independent GPT-6 Astra review (fallback: independent Opus
reviewers, same VERDICT rule). REQUEST_CHANGES blocks the dependent milestones only.

| sub-milestone | content | depends on | GPU (projection) | review | decisions |
|---|---|---|---|---|---|
| **M3b-0E** EPTA infrastructure (CPU + one benchmark, D3) | fork audit + EPTA manifest; tempo2 export + T1 (complete EPTA roster, released vs canonical); prior-volume and fingerprint checks; N1 (EPTA), N3, N7, N8, N9, N11, N13; event-epoch kernel validation; conditional-occupancy diagnostics on the reference chains (supporting only, Sec. 5.3); G5-EPTA incl. cross-model; reference loaders; EPTA acceptance/relevance files; EPTA T2 (CPU grids); benchmark | - | <= 1 (D3: one benchmark) | R-M3b-0E | D1, D2, D3, D6 |
| **M3b-EPTA** | E-C0 (CPU); pilot (<= 2 GPU-h, abort rules); proposals frozen; production CURN^gamma and CURN 13/3; HD^gamma and HD 13/3 by reweighting (HD run only if Sec. 5.4 fails); BF; E-7 optional | M3b-0E | 6-100 (2 pilot + two runs at 3-50 each, central ~12) | R-M3b-EPTA | D4, D8 |
| **M3b-0P** PPTA infrastructure (CPU) | extensions version / image; grid resolution by fingerprint; PPTA manifest (resolve the -0.003872-nat residual; product-space logging convention); N1 (PPTA), N4, N5, N14; T1 (PPTA roster); P-C0 tooling; product-space reference bootstrap | M3b-0E tooling | <= 0.5 (grids + fingerprint) | R-M3b-0P | - |
| **M3b-PPTA-C** | P-C0; band-overlap sensitivity | M3b-0P | < 0.5 | R-M3b-PPTA-C | D5 |
| **M3b-PPTA-S** (separate experiment, only via D5) | sampler development for ~262 parameters: coefficient-Gibbs design note (collapsed updates), toys with overlapping weak processes, agreement with NUTS on EPTA; or a capped NUTS feasibility pilot | D5 | capped by D5 (pilot <= 5) | separate Astra design + result reviews | D5 |
| **M3b-PPTA-F** (only after PPTA-S passes) | production PPTA full model; P-1 to P-6 | PPTA-S | from PPTA-S measurements | R-M3b-PPTA-F | new allocation |
| **M3b-0X** cross-check infrastructure (CPU) | MPTA/InPTA transcription (double entry, MPTAGW JSON cross-check); band-overlap inventory (4.3 step 1) | - | 0 | joint with M3b-X | D7 |
| **M3b-X** | MPTA and InPTA cross-checks | M3b-0X | < 1 | R-M3b-X | - |
| **close-out** | `docs/M3B_RESULTS.md`; Stage-2 inputs (evaluator profile, pruning rule, J1022/J0437 plan) | all run milestones | 0 | final | go/no-go for M3c-prep |

The EPTA path (M3b-0E -> M3b-EPTA) does not depend on any PPTA or cross-check work.

**Total GPU (projection):** **~7-100 GPU-h** without M3b-PPTA-S/F, central ~15. The EPTA production
range dominates. PPTA-S/F are allocated separately (D5).

---

## 8. Human decision points

The reviewer's recommendations (review of 25436e9) are marked **[R]**. **On 2026-10-09 the user
accepted every recommendation (D1-D8 as marked, D9 as specified in review round 5)**; the last
column records the decision (binding wording at the top of this document).

| # | decision | when | default if no answer | reviewer recommendation | **user decision 2026-10-09** |
|---|---|---|---|---|---|
| D1 | tempo2/libstempo as primary evaluator of non-NG legs; E8 re-scoped, kept open | before M3b-0E | no non-NG reproduction claim | **[R]** adopt tempo2, with expanded T1 and complete-roster acceptance | adopted: tempo2/libstempo evaluator, expanded T1, complete roster; E8 open |
| D2 | Confirm the decision rule (Sec. 6.1) and the provisional margins | before any sampling | not run | **[R]** revise the statistical rules before approval (done in revision 2); margins can stay provisional | adopted: rule confirmed, margins provisional |
| D3 | First GPU use: benchmarks (<= 0.5 GPU-h each in M3b-0E and M3b-0P) | after the exact models are established | not run | **[R]** support the capped benchmark after exact models are established | adopted: one benchmark, <= 1 GPU-h, only after exact-model, T1 and G5-PTA pass |
| D4 | EPTA production allocation from the pilot | after the pilot | stop after the pilot | **[R]** allocate from measured cost-to-precision for all required EPTA runs | adopted: allocation after the benchmark from measured cost; stop and report before any production run |
| D5 | PPTA: conditional-only, or a separately reviewed sampler experiment (Gibbs or capped NUTS) | after M3b-PPTA-C | conditional-only | **[R]** default to conditional-only; Gibbs as a separately reviewed experiment | adopted: conditional-only |
| D6 | Fetch and audit EPTA's modified enterprise + enterprise_extensions (EPTADR2-v1.1) | M3b-0E | **no full-reproduction claim for EPTA** (downgrade) | **[R]** mandatory for full reproduction; no inferred-prior fallback | adopted: mandatory (enterprise and enterprise_extensions) |
| D7 | MPTA clock for the published-target comparison | M3b-0X | par BIPM2020 default, BIPM2022 variant | **[R]** prefer paper BIPM2022; keep native-par clocks as a sensitivity | adopted: BIPM2022 for the published comparison; native-par clocks as sensitivity |
| D8 | Any evidence ladder or HD run beyond the envelope | if Sec. 5.4 fails | INCONCLUSIVE | **[R]** keep the cap and the INCONCLUSIVE fallback; validate the evidence-path endpoints first | adopted: capped, INCONCLUSIVE fallback |
| D9 | Whether to define a separately labelled claim class for results that are EQUIVALENT on all headline quantities but INCONCLUSIVE only because of unvisited regions (Sec. 5.3); needs reviewer agreement | before EPTA results are reported | no such class: verdict INCONCLUSIVE | review round 5: adopt, with the template and conditions of Sec. 6.6 | adopted: Sec. 6.6 (frozen union U, zero visits in both, mixture envelope p in {0.001, 0.01, 0.05, 0.1}, BF sensitivity; unconditional verdict INCONCLUSIVE) |

---

## 9. Risks

| # | risk | likelihood / impact | mitigation |
|---|---|---|---|
| R1 | tempo2 2026.04.1 vs the PTAs' 2023 builds shift residuals | medium / medium | fingerprint scatter quantifies it; T2; trace order |
| R2 | Loss of an independent timing oracle under D1 | low / medium | PINT secondary (T2); G1/G5 independent; fingerprint |
| R3 | PPTA common grid irrecoverable (image unavailable, fingerprint ambiguous) | medium / high for PPTA | P-C0 on both hypotheses; PPTA claim downgraded |
| R4 | EPTA fork unavailable or unreadable | low-medium / high for EPTA | D6 default = downgrade; ask EPTA |
| R5 | Builder mismatches (masked spans, non-harmonic grids, ECORR quantisation, `int()` counts, informed priors; the -0.003872-nat residual) | medium / medium | manifest checks, fingerprint, N1 per-block comparison |
| R6 | Shelves and funnels | high / medium | frozen jumps, transport gate by class, diverse inits |
| R7 | Event-epoch discontinuities mishandled | medium / medium | t0 by MH only; numerical-reference validation with a negative control |
| R8 | Reweighting overlap or precision inadequate | low-medium / low | k-hat rules, batch-means MCSE, per-chain stability, fallback |
| R9 | Reference precision limits decisions (EPTA outer tails) | certain / low | decidability fixed now; reference-limited quantities moved out of the headline set |
| R10 | PPTA full model too expensive | high / high | conditional-only default; separate PPTA-S |
| R11 | Paper/product inconsistencies (E-2, dip index, N_c 27/28, E-5 slice, PPTA BF weights) | certain / low | follow the chain-producing code; recorded; E-5 and P-6 secondary |
| R12 | Incomplete MPTA/InPTA inputs | certain / low | approximate class only |
| R13 | GPU contention | medium / low | recorded; no unmatched speed claims |
| R14 | Scope creep | medium / medium | Sec. 1.3; revisions required |

---

## 10. Open questions

1. Chain-producing versions of enterprise / enterprise_extensions / tempo2 for EPTA (fork tag) and
   PPTA (Singularity image on the CSIRO DAP).
2. Is a DR2new gamma = 13/3 chain released anywhere? Without one, E-5 stays secondary.
3. PPTA product-space model weights; the source of "~1.5 / ~2".
4. MPTA: N_c, T and RN mode counts of the Bayesian runs.
5. InPTA DR2: N_c of the UL run; WN values ("available on request").

---

## Appendix A. Reference-product checks done for this plan (CPU, 2026-10-09)

Tools: `ptagwb.diagnostics` (`ess_bulk`, `ess_tail`, `mcse_quantile`) and numpy, on the released
files. The producer becomes `scripts/m3b_reference.py` (N11), which must reproduce these numbers.

**EPTA DR2new** (`DR2new/{hd_pl,crn_pl}.tar.gz`, `chain_1.txt`: 29,990 rows x 71 columns = 67
parameters + lnpost, lnlike, acceptance, PT acceptance; first 25 % discarded, 22,493 kept; medians
change by <= 0.001 without burn-in):

| chain | parameter | q05 | q16 | q50 | q84 | q95 | bulk / tail ESS | MCSE q05 / q16 / q50 / q84 / q95 |
|---|---|---|---|---|---|---|---|---|
| hd_pl | log10 A | -14.418 | -14.149 | -13.935 | -13.783 | -13.702 | 1,331 / 1,120 | 0.0368 / 0.0103 / 0.0046 / 0.0039 / 0.0038 |
| hd_pl | gamma | 1.978 | 2.251 | 2.710 | 3.300 | 3.894 | 1,556 / 1,213 | 0.0127 / 0.0098 / 0.0121 / 0.0259 / 0.0704 |
| crn_pl | log10 A | -14.766 | -14.323 | -13.997 | -13.815 | -13.718 | 984 / 1,108 | 0.0581 / 0.0263 / 0.0067 / 0.0041 / 0.0040 |
| crn_pl | gamma | 2.038 | 2.364 | 2.906 | 3.701 | 4.626 | 1,139 / 1,257 | 0.0190 / 0.0142 / 0.0169 / 0.0506 / 0.1069 |

**Prior-volume column.** lnpost - lnlike = **-135.4959** in every row of both chains. The candidate
manifest (31 noise processes with log10 A U(-18, -10) and gamma U(0, 7); dip widths 8, 2.5, 40 d;
common log10 A U(-18, -10), gamma U(0, 7)) gives 135.4959. With a common upper bound of -11 it
gives 135.3623. The lnlike column ranges over 531,435-536,713 (HD) and is the fingerprint target.
The CURN amplitude reaches -17.97 and noise amplitudes reach -18.0.

**PPTA DR3** (GitHub fdbe6eb, `analysis_codes/data/all/chains/`):

| chain | parameter | draws | q16 | q50 | q84 | bulk / tail ESS | MCSE q16 / q50 / q84 |
|---|---|---|---|---|---|---|---|
| pl_nocorr_freegam_DE440 | log10 A | 2,978 | -14.655 | -14.496 | -14.358 | 549 / 727 | 0.0088 / 0.0053 / 0.0078 |
| pl_nocorr_freegam_DE440 | gamma | 2,978 | 3.505 | 3.869 | 4.228 | 619 / 910 | 0.0220 / 0.0151 / 0.0155 |
| pl_nocorr_fixgam_DE440 | log10 A | 2,967 | -14.735 | -14.681 | -14.634 | 1,569 / 2,342 | 0.0017 / 0.0014 / 0.0017 |
| ..._v_pl_hd_freegam (nmodel > 0.5) | HD log10 A | 3,643 of 5,616 | -14.709 | -14.514 | -14.337 | (subsequence ESS ~354; to be replaced by the ordered-chain method) | - |
| ..._v_pl_hd_freegam (nmodel > 0.5) | HD gamma | 3,643 | 3.444 | 3.906 | 4.389 | (~319) | - |
| ..._v_pl_hd_fixgam (nmodel > 0.5) | HD log10 A | 3,226 of 5,139 | -14.734 | -14.675 | -14.615 | (~1,728) | - |

* HD-active fractions are 0.649 / 0.628, i.e. count ratios 1.85 / 1.69.
* The free-gamma common amplitude's minimum is -16.27 (zero shelf visits), so its reference shelf
  support is UNRESOLVED (Sec. 5.3).
* **Prior-volume column:** freegam -467.6058, fixgam -465.1003. The difference is
  2.5055 = ln(49/4): gamma width 7 and amplitude width 7 vs amplitude width 4, which confirms the
  fixed-gamma U(-18, -14).
* **Reconstruction** with the released rules (informed bounds under the builder's noisename keys,
  e.g. `band_noise_low`, not the parameter's `band_noise_low_low`; shared per-pulsar group bounds):
  -467.601929 (free) and -465.096403 (fixed), against the logged -467.605801 and -465.100276. That
  is a constant **-0.003872** nats in both, matching the reviewer's independent value. Row spread
  of the logged column: 5.0e-7 (six-decimal storage). Revision 2's "2.5408" came from looking the
  band-noise bounds up under the parameter names, which silently fell back to (-18, -11) / (0, 7).
* **Product-space logged prior:** -935.211603 (free) and -930.200551 (fixed), i.e. 2 x the
  single-model values. This is `HyperModel.get_lnprior` summing both constituent models' priors.
* Storage: all PPTA `.npy` arrays are float64; lnlike and lnpost carry six decimals.
* Group noise in the 30-pulsar array: J0437 8 parameters (4 groups), J1017 4, J1022 4, J1713 8,
  J1909 2.

**Mode counts and spans** (survey CSVs and released dictionaries; projections until N1 replaces
them with enterprise-exact counts):
* EPTA: T_array = 10.33 yr, 45,428 TOAs; noise K_a max 398 (J0900-3144), median 50; sum of K_a^3
  (with the 18 common columns) ~16x NG15's with buckets, ~124x padded.
* PPTA (30 pulsars): T = 6,604.7 d, so int(T/240 d) = 27; 112,556 TOAs. K_a with RN, DM,
  chromatic, band, HF and SW-GP blocks and 54 common columns: J1744 988, J1600 1,042, J1713 1,146,
  J1909 1,208, J0437 1,388 (**1,816** with its ~428 group columns, before any basis consolidation);
  median ~540. Sum of K_a^3 ~760x NG15's before group noise.

**Builder facts read for revision 2.**
* PPTA `ppta_dr3_models.py` line 21 imports `enterprise_extensions.blocks.common_red_noise_block as
  crn_block`. `get_crn_model_dict` passes `components` but not `Tspan`.
* The installed enterprise_extensions 3.0.3 builds both CURN (`BasisGP`) and HD (`BasisCommonGP`)
  from `createfourierdesignmatrix_red(Tspan=None)`.
* EPTA J1022+1001 DR2new par: `NE_SW 9.7014335675808864986 1 0.55246717142622825403`.
