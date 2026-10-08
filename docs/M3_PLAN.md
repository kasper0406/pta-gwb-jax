# M3 plan: combining the five public PTA data sets

Goal: combine NANOGrav 15-yr (NG15), EPTA DR2, PPTA DR3, InPTA DR2 and the MeerKAT PTA 4.5-yr
release (MPTA) in our own JAX pipeline. First reproduce each PTA's published GWB result and the
public five-PTA combination of Yu & Allen (arXiv:2512.08666, called **YA** below). Only then go
beyond it.

This is a design document. Nothing in `src/ptagwb` has changed on this branch. The inventory,
overlap matrix and PINT load results are in [`M3_SURVEY.md`](M3_SURVEY.md); the literature
summary (Sec. 3 of the survey) gives the per-paper method details cited below.

Notation: **leg** = one PTA's data set (par + tim) of one pulsar. **K_a** = number of
stochastic (GP) basis columns of pulsar a. **N_f** = number of common-process frequencies.
**[UNCERTAIN]** marks things not pinned down from the papers or the released files.

---

## 0. Summary and recommendation

* **Data.** All five PTAs are on disk. 122 unique pulsars (InPTA DR2 set), or 121 with YA's
  inputs, which reproduces YA's count exactly. 234 legs (YA: 222); 60 pulsars (YA: 56) have
  two or more legs.
* **Recommended option: (C) TOA-level combination in coefficient space, with (B) as the same
  code path.** We build one "multi-leg pulsar" per physical pulsar: its legs' TOAs stacked,
  per-leg white noise (block-diagonal N), a shared Fourier basis for intrinsic red noise (IRN),
  DM and the GWB, and a timing design matrix whose columns are either shared across legs
  (astrometry, spin, binary: **C**, YA's MetaPulsar "shared" mode) or kept per leg
  (**B**, FrankenStat / MetaPulsar "per_pta"). Our stage-1/stage-2 split makes the per-call
  cost of B and C identical: it depends only on K_a and N_psr, not on the TOAs or on how M
  is shared. The two options differ only in the one-time stage-1 precompute and in the
  par-file engineering. (A), one leg per pulsar (Lite), is a cheap baseline and a sanity check.
* **Why C (shared timing columns) as the default.** It is what YA did, so reproducing it needs
  it. It is also the only option that keeps a short leg extending past the end of the longest
  leg phase-coherent with it: MPTA and InPTA run up to 3.5 yr past NG15. With per-leg spin
  parameters (B), each leg's own F0/F1 absorbs that leg's low frequencies. FrankenStat's
  "<1 % S/N loss" comes from simulations where every split PTA spans the same 20 yr; it does
  not cover this. Stage 2c quantifies B versus C on the real legs before any sampling.
* **New likelihood components needed** (Sec. 4): the TN-EQUAD white-noise convention;
  per-PTA white-noise dictionaries and selections; Fourier GPs with chromatic scaling (DM GP
  with nu^-2, chromatic/scattering with nu^-beta) and with row masks (band and system noise);
  a solar-wind GP and deterministic n_earth; fixed deterministic chromatic events (exponential
  dips, Gaussian bumps, annual terms); the multi-leg container; and a stage-2 reduction for a
  larger sampled per-pulsar block (K_a ~ 60 to 260 instead of 60). Plus generalized-chi^2
  p-values for the OS, NP and NPMV statistics, and a better Bayes-factor estimator for
  ln BF ~ 10.
* **Cost.** Per value+gradient call, HD on 121 pulsars x 14 modes (a 3388-dim common system)
  with sampled DM GPs (K_a = 260) is projected at **~60-150 ms** on the 5090, versus 8.8 ms
  for NG15 with the fast likelihood. A YA-style HD NUTS run is then **~10-35 GPU hours**; with
  DM hyperparameters fixed it is **~3-6 h** (Sec. 6). These numbers must be confirmed by the
  Stage-0 micro-benchmark before we commit GPU time.

---

## 1. What we are combining (short; details in M3_SURVEY.md)

| PTA (release used) | Pulsars | TOAs | Span | Clock / ephemeris in par | Units | DM in the released model | Released noise model |
|---|---|---|---|---|---|---|---|
| NG15 v2.1.0 narrowband | 68 (67 used) | 676,397 | 2004.6-2020.6 (15.9 yr) | TT(BIPM2019) / DE440 | TDB (PINT) | DMX | WN dict + IRN (fixed-WN GWB analysis) |
| EPTA DR2new (GitLab) | 25 | 45,428 | 2011.1-2021.4 (10.3 yr) | TT(BIPM2021) / DE440 | TCB | DM GP (custom bins), DM1/DM2 in par | `noisefiles/DR2new/*.json` (EFAC, TNEQUAD, RN, DM, chromatic) + bin-count dicts |
| PPTA DR3 (CSIRO / GitHub) | 32 (30 used) | 114,823 / 113,951 | 2004.1-2022.2 (18.1 yr) | TT(BIPM2020) / **DE436** | TCB (no UNITS line) | DM GP + DM1/DM2 | `noisefiles/*.json` (full max-likelihood models) + chains on GitHub |
| InPTA DR2 (GitHub) | 27 | 83,120 | 2017.1-2024.3 (7.2 yr) | TT(BIPM2023) / DE440 | TCB | DMX in `.DMX.par`; DM12 pars in DR2.NA | paper table (arXiv:2512.20455); T2EFAC in par |
| InPTA DR1 (YA only) | 14 | 8,529 | 2018.4-2021.8 (3.4 yr) | TT(BIPM2019) / DE440 | TCB | DMX | T2EFAC in par |
| MPTA 4.5-yr (Data Central) | 83 | 245,907 | 2019.1-2023.6 (4.5 yr) | TT(BIPM2020) (paper: BIPM2022) / DE440 | TCB | DM1/DM2 + DM GP (noise model) | paper table (arXiv:2412.01148) only |

The whole-array span is 19.5 yr (YA's inputs) or 20.2 yr (with InPTA DR2): f_1 = 1.6 nHz and
14 modes reach 22-23 nHz. NG15's 14 modes over 16.03 yr reach 28 nHz.

---

## 2. Options for overlapping pulsars

### 2.1 Our likelihood in one paragraph

Per pulsar a (M1 docs, `likelihood.py`): residuals r = M eps + F c + n with flat-prior timing
offsets eps, Gaussian Fourier coefficients c (IRN on 30 modes; the first 14 also carry the
common process) and white noise N (EFAC, T2EQUAD, ECORR). **Stage 1** (host, once): whiten
with N^-1/2, project out span(M), QR the projected basis F_p = Q_F R_F, and keep R_F, c = Q_F^T r_p,
s_perp, log|N| and log|M^T N^-1 M|. Then A = F^T P F = R_F^T R_F and b = F^T P r = R_F^T c.
**Stage 2** (JAX, per call): square-root absorption of the per-pulsar prior Phi_a in a batched
QR, which leaves a 28 x 28 common-mode precision E_a and score d_a; CURN is separable, and HD
solves one (N_psr * 2 N_f) system. Stage 2 never touches TOAs.

### 2.2 One physical pulsar with several legs

Stack the legs' TOAs: r = [r_1; ...; r_P]. White noise is per leg, so N = blockdiag(N_p). The
pulsar's stochastic processes are physical properties of the pulsar and the line of sight,
so they share one coefficient vector across legs: F = [F_1; ...; F_P], built on the same
frequencies f_k = k/T and the same time origin. For DM and chromatic GPs, F_p carries the
per-TOA factor (nu_ref/nu)^beta. **Sharing Fourier coefficients across legs is therefore
exactly "stacked TOAs, block-diagonal noise, shared basis"**, as the task statement says.
The only modelling choice left is the timing design matrix:

* **per-leg (B):** M = blockdiag(M_p). Then the projector is block-diagonal too,
  P = blockdiag(P_p), and the stage-1 contractions **add up**:

      A = sum_p R_p^T R_p,   b = sum_p R_p^T c_p,   r^T P r = sum_p (s_perp,p + |c_p|^2),
      log|N| = sum_p log|N_p|,   log|M^T N^-1 M| = sum_p log|M_p^T N_p^-1 M_p|.

  In square-root form: QR the stacked factors, [R_1; ...; R_P] = Q R; then c = Q^T [c_1; ...; c_P]
  and s_perp = sum_p s_perp,p + |[c_1; ...]|^2 - |c|^2 (computed as the norm of the stacked
  vector's component outside span(Q), not by subtraction). So **B is a pure stage-1
  aggregation of per-leg terms that we can already compute** with today's `precompute_pulsar`,
  given a shared basis definition. This is FrankenStat (arXiv:2512.14807, eq:franken-N/M).
* **shared (C):** M has one column per physical parameter (sky position, proper motion,
  parallax, spin, binary, DM1/DM2) with support on all legs, plus per-leg columns for
  detector terms (phase offset / inter-PTA JUMP, backend JUMPs, FD, per-leg DM reference
  value). P is no longer block-diagonal, so stage 1 runs on the stacked whitened TOAs with the
  combined M. That is one host QR of an (n_TOA x m) matrix per pulsar, with n_TOA up to a few
  x 10^4 (J1909-3744 summed over 5 PTAs) and m ~ 500 (DMX removed): seconds per pulsar, once. The
  shared columns require the legs' residuals to be computed at **one common nominal model**:
  with each leg's own par values the offsets M_p (beta_0,ref - beta_0,p) are not in the span
  of the shared columns. YA/MetaPulsar therefore copy the reference PTA's astrometric, spin,
  binary and DM values (priority NG > EPTA > PPTA > MPTA) into the other legs' pars and
  recompute their residuals. After that only the column space of M enters the marginalised
  likelihood (MetaPulsar METHOD_DESCRIPTION, "Statistical equivalence"), as long as each leg
  stays in the linear regime around the copied values.

Stage 2 is **identical** for B and C: per pulsar it sees one (R_F, c, s_perp) of size K_a. The
per-call cost does not depend on the number of legs.

### 2.3 Option (A): one leg per pulsar (Lite, arXiv:2503.20949)

Pick the leg with the largest FoM = sqrt(T_obs / (<sigma_TOA>^2 <Delta t>)^(3/13)) (harmonic
mean error, geometric mean cadence), or a PTA priority order as in 3P+ (arXiv:2309.00693).
Every leg keeps its own released timing and noise model.

* *Correctness:* trivially valid; no cross-PTA consistency issues except a common clock and
  ephemeris.
* *Information:* throws away the other legs. On IPTA DR2, Lite's CURN evidence was 10^3.0
  against 10^9.1 for the full combination, its 95 % (A, gamma) area was 2.25x larger, A(13/3)
  was biased 23 % high, and B(HD/CURN) was 0.66 against 1.39 (2503.20949 tab:crn_params,
  tab:hd_BFs).
* *Cost:* lowest. N_psr = 122 with K_a from each leg's native model.
* *Use:* baseline and cross-check. It runs as soon as each leg's noise model is implemented.

### 2.4 Option (B): per-leg timing model, shared IRN/DM/GW (FrankenStat)

* *Correctness:* valid but conservative. Each leg's timing parameters are free separately, so
  the likelihood marginalises over more nuisance directions than physics allows. Mixed DM
  treatments are a trap: if a leg keeps DMX in its timing model while a DM GP is shared, the
  DMX columns project out the DM GP on that leg, which is consistent but wasteful. If DMX is
  removed (YA), the per-leg M must still contain DM, DM1 and DM2.
* *Information loss:* small where legs interleave over the same span (FrankenStat simulations:
  combined vs FrankenStat OS S/N 11.23 vs 11.14, CURN log10 A difference -0.03 +- 0.19). It
  can be large for **non-overlapping extensions**. An MPTA leg covering 2019-2023 next to a
  NANOGrav leg ending in 2020.6 has its own F0/F1 (+ astrometry). Those absorb power below
  ~1/(4.5 yr) in the extension, so the extension adds little at the GWB's lowest frequencies.
* *Engineering:* minimal. No par-file surgery and no unit conversion, because each leg is
  timed with its own model. Only clock/ephemeris unification (Sec. 7).
* *Cost:* as C.

### 2.5 Option (C): full TOA-level combination with shared astrophysical timing columns

Classic IPTA (DR2) combination re-derives one timing solution per pulsar with inter-PTA
JUMPs and refits it. MetaPulsar's "shared" mode (used by YA) gets **the same marginalised
likelihood without a refit**: it copies the reference leg's astrometric, spin, binary and DM
values into the other legs' pars, converts TCB to TDB, unifies EPHEM/CLK and conventions
(NE_SW, ECL, T2CMETHOD, ...), removes DMX, and builds M with merged columns. The phase
offset, JUMPs, FD and the per-leg DM reference value stay per leg.

* *Correctness:* equivalent to a manual combination under the linear-timing assumption.
  Caveats: (i) a nominal model far from a leg's own best fit can break linearity, so we
  check the per-leg pre-fit residual RMS after the value swap; (ii) PINT and tempo2 must
  evaluate the same delays for the same pars (YA mix PINT for NG15 and tempo2 for the rest;
  MetaPulsar forces an explicit common profile: UNITS TDB, T2CMETHOD IAU2000B, TIMEEPH
  FB90, DILATEFREQ N, CORRECT_TROPOSPHERE N, PLANET_SHAPIRO N, NE_SW explicit); (iii) phase
  connection between legs, which YA had to fix by adding pulse numbers (their SM Sec. C:
  phase disconnections had produced anomalous WN parameters in an earlier draft).
* *Information:* maximal.
* *Engineering:* highest. Par-file normalisation per pulsar, with every rule checked by the
  residual-equality test in Stage 0.
* *Cost:* as B per call. Stage 1 is a bit more expensive (joint QR) but runs once.

### 2.6 Comparison

| | (A) Lite | (B) per-leg M | (C) shared M |
|---|---|---|---|
| Pulsars / legs used | 122 / 122 | 122 / 234 | 122 / 234 |
| Same-pulsar coefficients shared | n/a | yes | yes |
| Timing columns | leg's own | blockdiag per leg | shared astrophysical + per-leg detector |
| Low-frequency information from short extension legs | none | mostly lost | kept |
| Par-file engineering | none | clock/ephem only | TCB->TDB, value copy, DMX removal, convention profile |
| Stage-1 (host, once) | per leg | per leg, then aggregate | joint per pulsar |
| Stage-2 cost per call | lowest K_a | same as C | K_a, N_psr only |
| Reproduces YA | no | no (different M) | yes |
| Main risk | biased by single-leg systematics | conservative; mixed DM treatments | linearity / phase connection / PINT-tempo2 parity |

**Recommendation.** Implement the multi-leg container with a `timing_sharing = {"per_leg",
"shared"}` switch, so that B and C share all likelihood code. Use **C as the default** (YA
reproduction, maximal information) and **B as a robustness check**: B-versus-C differences
flag timing-model problems. Keep **A as the day-one baseline**. The extra engineering for C is
par-file normalisation and a parity test. MetaPulsar's documented rules (stripped terms,
TCB->TDB, ecliptic transformation, NE_SW alignment, ELL1H/H3/STIG handling) give us a checklist
and an oracle: the `metapulsar` tags are fetched, and its outputs can be used as a reference
in tests only, like enterprise and discovery in M1.

---

## 3. What reproducing YA requires (target definition)

From the arXiv:2512.08666 v3 LaTeX (`manuscript.tex`; EM = End Matter, SM = Supplementary
Material), cross-checked against our survey:

| Item | YA setting | Our check / status |
|---|---|---|
| Inputs | EPTA DR2new (GitLab 1506123), InPTA **DR1** (GitHub 2c400d5), MPTA 4.5 yr (Data Central), NG15 v1.0.0, PPTA DR3 (GitHub danielreardon/PPTA-DR3 fdbe6eb) | All fetched (`inpta_dr1`, `ppta_dr3_github`). EPTA par/tim identical between 1506123 and our 2911d0e (GitLab compare). NG15 narrowband par/tim identical v1.0.1 vs v2.1.0 (M0). |
| Pulsars | 121 = 25 EPTA + 14 InPTA + 83 MPTA + 68 NG + 32 PPTA; mean 1.8 PTAs per pulsar | **121** unique by 30" position matching; mean 1.83 (survey). Note YA use all 68 NG15 and all 32 PPTA pulsars, including J0614-3329 (NG, 2.4 yr) and J1741+1351 (PPTA) |
| TOAs | N = 1,090,206; 976 backends | Our text count of the same files: **1,090,212** (+6) [UNCERTAIN: which 6; PINT counts in survey] |
| Combination | MetaPulsar "direct combination": reference PTA NG > EPTA > PPTA > MPTA; copy astrometry/spin/binary/DM values; TCB->TDB; DMX -> DM0 + DM1 + DM2 (initialised 0); one JUMP per target PTA; FD and detector parameters per PTA; ELL1H without H3 -> ELL1; pulse numbers added | Option C. PINT for NG15 and tempo2 for the rest in YA; we use PINT for all legs (risk R3) |
| White noise | Per pulsar/backend EFAC, EQUAD (T2EQUAD convention), ECORR for NG, PPTA, MPTA and 7 EPTA backends; EFAC+EQUAD only for InPTA and the other EPTA backends. **Re-fit on the combined data** by MAP (JAXopt) with the Table-1 priors, then fixed | They released no WN dictionary; we must refit (Stage 3a) |
| Pulsar noise | Per pulsar: IRN power law, **30 bins**; DM GP power law, **100 bins**. Both sampled. Priors gamma U(1,7), log10 A U(-20,-11) | No chromatic, band, system or SW-GP terms (a deliberate simplification versus EPTA/PPTA/MPTA). 4 x 121 + 2 = 486 parameters |
| Common process | HD and CURN, power law, **14 bins, "2 <~ f/nHz <~ 30"**; gamma U(1,7) or 13/3; log10 A U(-20,-11) | **[UNCERTAIN]** the basis span: with T = whole-array span (19.5 yr) the 14 bins cover 1.6-22.8 nHz, not 2-30. "2-30 nHz" matches NG15's T = 16.03 yr. Must be resolved (author contact or code release) before comparing numbers |
| Sampler | discovery NUTS (NumPyro), 20,480 posterior samples per model | Same family as our M2 |
| Evidence | Generalized stepping-stone sampling, K = 8 chains | We need GSS or an equivalent (Sec. 5, Stage 3d) |

**YA headline targets** (tab:pos: medians and 68 % intervals; tab:pvals; EM:6):

| Quantity | YA value |
|---|---|
| CURN, gamma free | log10 A = -14.37 (+0.11 / -0.12), gamma = 3.63 (+0.27 / -0.26) |
| CURN, gamma = 13/3 | log10 A = -14.69 +- 0.04 |
| HD, gamma free | log10 A = -14.45 (+0.13 / -0.15), gamma = 3.76 (+0.31 / -0.29) |
| HD, gamma = 13/3 | log10 A = -14.72 +- 0.04 |
| ln BF(HD/CURN) | 10.18 +- 0.13 (BF = 26,000 +- 3,000), GSS; ln(s^N E_HD) = 12,500,504.39 +- 0.10, ln(s^N E_CURN) = 12,500,494.21 +- 0.09 |
| OS p-value (gamma free, CURN draws) | mean 4.3 sigma (-log10 p = 5.2); median 4.8 sigma; 68 % range 4.3-5.4 sigma |
| NP / NPMV p-value | mean 3.3 sigma / 3.3 sigma; NPMV gamma = 13/3: mean 3.4 sigma (p = 3.7e-4) |
| HD reconstruction | 15 bins x 484 pairs, reduced chi^2 = 0.74 |
| Robustness | EFAC cuts [1/Q, Q] for Q = 10, 4, 3, 2 drop 0.18-1.96 % of TOAs; posteriors stable, amplitude rises slightly (SM tab:efac_cuts) |

The absolute evidences give a constant-sensitive check of our likelihood normalisation (cf.
M1's `convention="chain"`), once we know their constant convention.

---

## 4. What the M1 likelihood needs

Ordered by when each item is first needed. "Stage 1" means absorbed in the host precompute
(fixed hyperparameters); "Stage 2" means sampled (per-call cost).

| # | Component | Needed for | Where it goes | Notes |
|---|---|---|---|---|
| L1 | **EQUAD convention switch** `t2` (EFAC^2(sigma^2+Q^2): NG15, InPTA DR1/DR2 noise papers) / `tn` (EFAC^2 sigma^2 + Q^2: EPTA, PPTA, MPTA; enterprise `log10_tnequad`) | all non-NG | `noise.build_white_noise` | trivial. YA use T2EQUAD for everything they refit |
| L2 | **Per-PTA WN dictionaries and selections**: key schemes `<psr>_<group>_efac` (EPTA `-group`, PPTA `-group` + `basis_ecorr_*`, MPTA `KAT_MKBF`), PTA-specific ECORR rules, no ECORR for InPTA/most EPTA | per-PTA reproductions | `noise` | PPTA's ECORR is a basis GP over band/system *groups* that may overlap per TOA, so we treat it as a fixed GP in stage 1 (Woodbury), not as disjoint epoch blocks |
| L3 | **Multi-leg pulsar** container: stacked arrays, leg index, per-leg flags/WN, combined M with `per_leg`/`shared` columns, common time origin | B, C | `data` (new `MultiPulsar`) + `precompute_pulsar` | B = aggregation of per-leg stage-1 terms (Sec. 2.2); C = joint stage 1 |
| L4 | **Generic Fourier GP block**: basis = Fourier(t; k/T, n modes) x per-TOA scale (nu_ref/nu)^beta x row mask; prior = power law (or free spectrum) per mode | DM GP (beta = 2), chromatic (beta = 4 or fitted), band noise (mask on frequency band), system/group noise (mask on backend), IRN | `basis` + `likelihood` | basis frequency grid per block: whole-array T (YA, 3P+) or per-pulsar T (EPTA, PPTA, MPTA single-pulsar conventions) |
| L5 | **Sampled per-pulsar block of size K_a > 60** in the stage-2 square-root reduction, with per-pulsar *variable* K_a (padding or bucketing) | YA (K_a = 60 + 200), EPTA (custom counts), MPTA | `likelihood` (+ `perf_likelihood`) | the main compute change (Sec. 6). Mixed sizes need padding with zero-information columns or size buckets |
| L6 | **Fixed GP absorption in stage 1** (Woodbury into N): any GP whose hyperparameters are held at released values | per-PTA "noise fixed" reproductions; PPTA band/system/hf noise; ECORR-as-GP | `likelihood.precompute_pulsar` | keeps stage 2 at K_a = 60 while all released noise terms are honoured |
| L7 | **Solar wind**: deterministic n_earth (timing-model column or fixed delay) and the SW GP (PPTA `gp_sw`, MPTA SW_Full) with basis = geometry factor x Fourier | PPTA, MPTA, (3P+) | `basis` | geometry from PINT's solar-wind delay derivative per TOA |
| L8 | **Deterministic chromatic events** held fixed and subtracted: exponential dips (PPTA `dmexp_*`, J1713+0747 in EPTA/3P+), Gaussian bumps and annual chromatic terms (MPTA tab "deterministic models") | PPTA, MPTA, EPTA | stage 1 (residual correction) | nonlinear in t0/tau, so fix at released MAP; sample only if a reproduction needs it |
| L9 | **Chromatic index beta**: fixed per pulsar at the released value | EPTA, PPTA, MPTA | L4 | a sampled beta changes the basis every call and breaks stage 1; Lite and PPTA also fix it in the array analysis |
| L10 | **Common clock and ephemeris** at ingestion: one BIPM realisation and DE440 for all legs (PPTA ships DE436) | everything | `data` | PINT `include_bipm`/`bipm_version`, EPHEM override; residual check (Sec. 7) |
| L11 | **tempo2-par canonicalisation** for PINT (survey Sec. 5, F1-F13: archive names, indented comments, END semantics, valueless flags, implicit UNITS TCB, TRACK -2, negative M2/H3, PB+FBn, DMXR flags, DMX_0001 template, `ell1h_shapiro="absorbed"`) | every non-NG leg | `data` (new canonicaliser) | prototyped in `scripts/m3_survey.py`: all 330 par/tim pairs (9 data-set variants) load, PINT TOA counts equal the text counts |
| L12 | **WN MAP refit** on combined data (YA) | YA | new `noise_fit` (JAX, optax/L-BFGS) | uses the same likelihood with WN sampled. Reuse the stage-1 split: WN changes N, so this runs per pulsar with full TOA-level linear algebra |
| L13 | **Generalized chi^2 p-values** for quadratic statistics (OS, NP, NPMV; Hazboun et al. 2023, van Haasteren et al. 2509.06489) per posterior draw | YA tab:pvals | `optstat` | eigenvalues of the null covariance of Q; Imhof/Davies integration |
| L14 | **Evidence for ln BF ~ 10**: GSS (Zahraoui et al.) or thermodynamic integration along Gamma_lambda = (1 - lambda) I + lambda Gamma_HD | YA BF | `evidence` | M2's bridge/reweighting needs overlap that degrades at ln BF ~ 10 |
| L15 | **Free per-pulsar Tspan option** for GP frequency grids | per-PTA reproductions | `basis` | EPTA, PPTA and MPTA define single-pulsar grids on each pulsar's own span |

Out of scope for M3 unless a reproduction needs it: sampled white noise in array runs,
BayesEphem, wideband TOAs (MetaPulsar does not support them; all five releases ship
narrowband or sub-band TOAs).

---

## 5. Staged plan (each stage has a gate; GPU-heavy stages need explicit approval)

### Stage 0: ingestion and infrastructure (CPU)

1. Canonicaliser (L11) moved from the survey script into `ptagwb.data`; ingestion of all legs
   with an explicit common clock (BIPM2019 or the latest, one choice for all) and DE440.
   **Gate:** every leg loads; per-leg PINT pre-fit wrms within 10 % of the par file's TRES
   (raw errors) or explained (survey Sec. 5 lists the current exceptions); TOA counts equal
   the par NTOA where the par was written from the same tim.
2. **PINT vs tempo2 parity** on a stratified sample (one pulsar per binary model per PTA), if
   a tempo2/libstempo build is available (oracle group only). **Gate:** residual differences
   < 1 ns RMS after removing a fit of the design matrix columns (the marginalised likelihood
   only sees the projection).
3. Micro-benchmark of the stage-2 reduction at K_a in {60, 120, 260} and N_psr = 122, CURN
   and HD (N_f = 14), value+gradient, B = 1/4/16. Replaces the projections in Sec. 6.
4. **B-versus-C Fisher check** (cheap, deterministic): for each multi-leg pulsar, compute
   the per-mode information F^T P F of the GWB modes under per-leg vs shared M, at fixed
   released noise. **Gate:** report the per-frequency information ratio; this quantifies
   Sec. 2.4's claim before any sampling.

### Stage 1: each PTA alone, with its released noise model (M2-style)

Configuration "released-noise": white noise and all non-IRN noise terms fixed at the released
values (L6), IRN and the common process sampled (as NG15 GWB). Where the paper sampled more,
a second "as-published" configuration samples DM (and chromatic) hyperparameters (L5). Compare
with released chains where they exist (quantiles, KS, MCSE-aware z-scores as in M2).

| Sub-stage | Data | Paper settings | Targets (paper) | Released reference |
|---|---|---|---|---|
| 1a EPTA | DR2new, 25 psr, DE440, array T = 10.3 yr | custom per-pulsar RN/DM/SV (nu^-4) models from `noisefiles/DR2new` (mode counts in `red_dict.json`/`dm_dict.json`/`chrom_dict.json`, on each pulsar's own span; DM amplitudes in TempoNest normalisation, `tndm=True`), sampled with the CRS; WN EFAC + TNEQUAD per `-group`, fixed, no ECORR; NE_SW = 7.9 fixed; J1713+0747 exponential dips (indices 4 and 1 in the released code); CRS on **9** frequencies; priors log10 A U(-18,-10), gamma U(0,7) (2306.16214 tab:priors) | HD^gamma: log10 A = -13.94 (+0.23 / -0.48), gamma = 2.71 (+1.18 / -0.71); CURN: -14.00 (+0.28 / -0.77), 2.91 (+1.72 / -0.87) (90 %, tab:pl); gamma = 13/3: log10 A = -14.61 (+0.11 / -0.12) (Sec. 4.1; the conclusions print -0.15); BF(HD/CURN) = 60 (enterprise), 62, re-estimates 66 / 56 / 62 (tab:bf_dr2); OS S/N_HD = 3.5 (+2.4 / -1.7), A^2_HD = 10.0 (+5.1 / -4.9) x 10^-30 (tab:os); DR2new+: BF 65, S/N 4.1 | Zenodo 8091568 chains `chains/DR2new/{crn_pl,hd_pl,hd_fs,os,...}` (fetched) |
| 1b PPTA | DR3 GitHub fdbe6eb (its TOA counts match the paper's Table tb:dr3, 113,951; the CSIRO v2 files differ), 30 psr (excludes J1824-2452A, steep intrinsic RN, and J1741+1351, 16 observations; 2306.16215 Sec. 2), DE440 override (pars say DE436) | full PPTA models (2306.16229 + released `commonNoise.py`): RN (T_p/240 d modes), DM GP (T_p/60 d), HFF RN (T_p/30 d) in 8 psr, nu^-4 scattering in 7, low/mid/high band noise, group noise, SW GP + n_earth (4 cm^-3 where unconstrained), exponential dips (J0437, J1643, J2145, 2x J1713) with sampled index, annual DM (J0613), Gaussian events (J1603, J1600 20-cm bump), overlapping band/broadband ECORRs; WN TNEQUAD fixed at max-likelihood (`noiseFiles_maxlike`); tailored priors (99.7 % single-pulsar interval, -2/+1 dex, +-0.5); CRS on floor(6605 d / 240 d) modes, printed as 28 (the floor is 27: [UNCERTAIN]) | CURN (DE440): log10 A = -14.50 (+0.14 / -0.16), gamma = 3.87 +- 0.36 (68 %); CURN 13/3: log10 A = -14.69 +- 0.05 (A = 2.04 (+0.25 / -0.22) x 10^-15); HD: log10 A = -14.51 (+0.18 / -0.20), gamma = 3.87 +- 0.47; HD 13/3: -14.68 +- 0.06; BF(HD/CURN) ~ 1.5 (free gamma), ~ 2 (13/3); pairwise log10 Delta L = 1.1, sky-scramble p <~ 0.014-0.018; 'basic' noise model instead gives log10 A = -14.08 +- 0.06, gamma = 2.9 +- 0.2 (2306.16229 Sec. 4), a useful intermediate target | GitHub `analysis_codes/data/all/chains/chain_commonNoise_*_DE440.npy` (CURN free/fixed gamma, HD, free spectrum, spline ORF) |
| 1c MPTA | 83 psr, DE440, T = 4.5 yr | noise table of 2412.01148 (EFAC, TNEQUAD, ECORR; RN in 12 psr; DM GP; chromatic with beta; SW n_earth and SW GP; Gaussian events in 15 and annual chromatic in 8 psr); ~120 Fourier components [UNCERTAIN: per-pulsar vs array T]; WN fixed in the GW search | ER CURN: log10 A = -14.25 (+0.21 / -0.36), gamma = 3.60 (+1.31 / -0.89); DATA CURN: -14.25 (+0.21 / -0.34), 3.52 (+1.12 / -0.90); DATA HD: -14.28 (+0.23 / -0.30), 4.50 (+1.00 / -0.93); ln B(CURN/IRN) = 3.17; ln B(HD+CURN / CURN) = -0.21; fixed-parameter OS (DATA): A^2 = (5.7 +- 1.2) x 10^-29, S/N = 4.6; noise-marginalised OS S/N mean 0.96 (sd 0.81) (2412.01153 Sec. 3) | none (no chains released); MPTAGW scripts give model code |
| 1d InPTA | DR2, 27 psr, per-pulsar T (7.2 yr max) | noise models of 2512.20455 (DMX removed, DM + DM1 + DM2 with DM GP, free-chromatic GP with sampled chi, deterministic n_earth (+ derivative for J1909-3744), per-process mode counts on each pulsar's own span; WN T2EQUAD with ECORR in some pulsars). The table gives RN/DM/chromatic/SW values but **no WN values**, so WN must be refit (L12) | InPTA DR2 III (arXiv:2608.02808): CURN log10 A = -13.71 (+1.06 / -3.28), gamma = 2.98 (+3.62 / -2.70); BF(CURN / no common) = 2.5; HD indistinguishable from CURN (no BF quoted); NMOS S/N distribution peaks HD -0.46, monopole -0.34, dipole -0.42; 95 % upper limit at gamma = 13/3: log10 A < -13.47 (uniform-A prior), < -13.55 (log-uniform), A < 3.4e-14 | none. "As far as feasible": the InPTA-only GWB result is an upper limit, so this is mainly a noise-model and upper-limit check |
| 1e NG15 | M2 | unchanged | M2 regression: likelihood value bit-identical through the new container (single leg, no new terms) | M2 runs |

**Gate per sub-stage:** power-law posteriors within MC error of the released chains (EPTA,
PPTA), or within the published intervals (MPTA, InPTA), with differences explained; BF
within the paper's estimator spread.

### Stage 2: the five-PTA data set, deterministic checks (CPU, then small GPU)

a. Build all 121/122 multi-leg pulsars in both B and C modes; check the timing-rank and the
   per-leg linearity (pre-fit wrms of each leg under the reference values).
b. Oracle check against MetaPulsar (reference only, oracle group): same column space of M
   (principal angles ~ 0) and same stage-1 contractions for a sample of pulsars, using
   MetaPulsar's Enterprise/discovery-compatible output.
c. B vs C information (Stage 0.4 on the full set), and the A/B/C CURN log-likelihood maps on a
   (log10 A, gamma) grid at fixed noise. Cheap.

### Stage 3: reproduce YA (GPU; needs approval)

a. **WN refit** (L12) on the combined data, YA priors (EFAC U(0.01,10), log10 EQUAD and
   log10 ECORR U(-8.5,-5)), MAP per pulsar. Compare the EFAC histogram with YA's SM
   Fig. (n_backends, n_ToAs) and the cut counts in tab:efac_cuts (7 PPTA backends outside
   [0.1, 10], ..., 1 EPTA + 24 InPTA + 4 NG + 18 PPTA outside [0.5, 2]).
b. CURN^gamma, CURN^13/3, HD^gamma, HD^13/3 NUTS runs: 30 IRN + 100 DM bins per pulsar,
   14 common bins, YA priors. Targets in Sec. 3.
c. OS / NP / NPMV with generalized-chi^2 p-values over CURN draws (L13). Targets: OS mean 4.3
   sigma, NPMV mean 3.3 sigma.
d. ln BF(HD/CURN): GSS (K = 8) or the ORF-lambda TI ladder (L14). Target 10.18 +- 0.13.
e. HD reconstruction in 15 bins (reduced chi^2 0.74) and the EFAC-cut robustness runs.

**Gate:** amplitudes and gamma within the YA 68 % intervals (MC-error aware); ln BF within
~3 sigma of 10.18 with our own estimator spread reported; OS p-value distribution consistent.
Any mismatch is first traced to the open settings (T for the common basis, WN refit, the
PINT-for-all difference).

### Stage 4: beyond YA (only after Stage 3 passes; separate proposal)

Candidate directions, none chosen yet: InPTA DR2 instead of DR1; per-PTA noise models
(chromatic, band, system, SW) instead of YA's RN+DM only, i.e. a combination that honours
each PTA's noise budget (YA list the lack of hierarchical noise models as a limitation);
B-vs-C information accounting; HD free spectrum and ORF reconstruction on the combined data.
These are not novelty-checked here and need the usual novelty check before any claim.

---

## 6. Compute and GPU estimates

Measured anchors (docs/PERF.md, RTX 5090, float64): NG15 HD (67 psr x 28 = 1876 common
dimensions, K_a = 60) value+gradient 15.0 ms (production) / 8.8 ms (fast) at B = 1, and 4.4 ms
per chain at B = 16. Breakdown (fast, B = 1): Cholesky of the 1876 system 4.9 ms (0.44 TFLOP/s),
level-batched triangular inverse 1.9 ms, per-pulsar Householder reduction 1.0 ms. CURN fast
1.0 ms. M2 HD^13/3: ~3e5 chain-gradients, 61 min wall (21 + 40).

Scaling (projections; Stage 0.3 replaces them):

| Configuration | Common system | Per-pulsar block | HD value+grad, B = 1 | CURN value+grad | NUTS HD run (5e5-1e6 chain-grads, B = 4) |
|---|---|---|---|---|---|
| NG15 (measured) | 1876 (2.2 GF chol) | 67 x K = 60 | 8.8 ms | 1.0 ms | ~1 h |
| 5-PTA, DM fixed (stage-1 absorbed) | 3388-3416 (13 GF chol, 92 MB) | 122 x 60 | ~25-45 ms | ~2 ms | ~3-6 h |
| 5-PTA, YA (DM sampled, 100 bins) | 3388 | 121 x 260 | ~60-150 ms | ~30-90 ms | ~10-35 h |
| 5-PTA, YA with 30 DM bins | 3388 | 121 x 120 | ~35-70 ms | ~5-15 ms | ~5-15 h |

How the ranges were obtained: the common Cholesky and the triangular inverse scale as
n^3 (x5.9 from 1876 to 3388), with efficiency rising toward the ~1.3 TFLOP/s seen in batched
runs. The per-pulsar square-root reduction scales as ~K_a^3 per pulsar (x81 from K = 60 to
260, x1.8 for the pulsar count): ~8-20 GF forward plus the VJP. The measured rate of the
batched small QR at K = 60 is only 0.04 (production) to 0.13 TFLOP/s (structured
Householder); assuming larger blocks run at 0.2-0.5 TFLOP/s gives 30-100 ms. That makes the
per-pulsar reduction the bottleneck in the YA configuration. A Cholesky of the K x K
precision costs ~10x fewer flops than the structured QR, but M1 rejected normal-equation
forms for conditioning at prior corners; a blocked or mixed approach is a Stage-0 benchmark
item.
NUTS cost: 486 parameters instead of 135 means ~1.4x more leapfrog steps per trajectory
(d^1/4) and more funnel-limited mixing (IRN and DM amplitudes in 121 pulsars), so we budget
2-3x M2's gradient count.

Total M3 GPU budget (projection): Stage 1 ~10-25 h (four PTAs, CURN + HD, two configs);
Stage 3 ~4 production runs x 10-35 h plus evidence (GSS with K = 8 tempered runs: ~8x a
posterior run, or a TI ladder of similar size): **~100-400 GPU hours** at YA fidelity. Hence
the cheaper "DM fixed" configuration first (~20-40 h for all four models) to debug the
pipeline, then YA fidelity. Memory: one 3388^2 float64 matrix is 92 MB, so B = 16 chains fit
easily in 32 GB.

---

## 7. Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **Noise models outside our family**: PPTA band/group/HFF noise, overlapping ECORR groups, exponential dips with sampled chromatic index and tailored priors; MPTA chromatic beta, Gaussian events, annual chromatic and SW GP; EPTA per-pulsar mode counts on per-pulsar spans, TempoNest DM normalisation, J1713 dips (chromatic indices 4 and 1 in code and noise paper, 4 and 2 in the GW paper); InPTA free-chromatic chi | per-PTA reproductions fail or are biased | L4-L9 implement them as fixed stage-1 terms first. Unit-test each noise-file convention against enterprise (oracle) on a pulsar where the PTA's own code is available (EPTA scripts_gwb, PPTA `ppta_dr3_models.py`, MPTAGW) |
| R2 | **Clock and ephemeris mismatches**: PPTA pars use DE436; clock realisations BIPM2019/2020/2021/2022/2023 across PTAs (MPTA par says BIPM2020, its paper BIPM2022); observatory clock files must exist for every site (gmrt, meerkat, pks, ncyobs, jbroach, effix, wsrt, leap, ...) | spurious common (monopolar or dipolar) signals; the PPTA reproduction depends on its ephemeris | one BIPM realisation and DE440 for all legs at ingestion (L10); per-leg residual comparison before and after; check that PINT's global clock repository covers every site and epoch (PINT warns on missing corrections; survey records warnings) |
| R3 | **PINT vs tempo2**: four PTAs time with tempo2. Differences in TCB->TDB conversion (PINT warns it is approximate), T2 binary resolution (allow_T2), ELL1H H3/STIG conventions (Freire & Wex eq. 28 vs 29: **observed** in the survey, several-us residuals in MPTA/PPTA ELL1H legs until `ell1h_shapiro="absorbed"`), DMMODEL/FDJUMP support, tropospheric and planetary Shapiro defaults, implicit NE_SW = 4 in tempo2 | residual systematics from ns to us | Stage 0.2 parity test where tempo2 is available; PINT's `ell1h_shapiro`, explicit NE_SW and UNITS; MetaPulsar's rule list as a checklist |
| R4 | **Phase connection** across legs and within legs (TRACK -2 pars without pulse numbers; YA's early anomalous WN came from phase disconnections) | wrong residuals, absurd EFACs | compare pre-fit wrms to TRES (survey); add pulse numbers from the leg's own model before swapping in the reference values |
| R5 | **Linearity under the value swap (option C)** | biased residuals for pulsars whose reference and leg models differ by more than the linear regime | per-leg wrms check; fall back to per-leg columns (B) for that pulsar |
| R6 | **YA settings not fully specified**: the common-basis T; NG15 v1.0.0 vs the 67-pulsar selection; WN refit details; 6-TOA count difference; evidence constant convention | cannot match their numbers exactly | resolve via a code release or author contact; meanwhile report sensitivity to each choice |
| R7 | **Compute**: K_a = 260 reduction and 486-parameter NUTS | 10-35 GPU h per run | DM-fixed debug configuration; benchmark first; DM-bin count sensitivity (100 vs 30) |
| R8 | **Mixed DM treatments** in B (DMX on some legs, DM GP shared) | consistent but wasteful; can hide DM mismodelling | remove DMX everywhere for the combined runs (as YA and 3P+ do); keep DMX only in the per-PTA reproductions that used it (NG15, InPTA DR2) |
| R9 | **Data versions**: PPTA DR3 exists in two public variants. GitHub fdbe6eb matches the data paper's TOA table exactly (113,951; J1939+2134 1,473 TOAs; J1741+1351 present). CSIRO DAP v2 has 114,823 (J1939+2134 2,456 TOAs, no J1741+1351 tim, par uncertainty columns stripped). EPTA `noisefiles_t2equad/` only renames the key (`log10_tnequad` -> `log10_t2equad`) without converting the value; EPTA DR2new noise files switched DM amplitudes to TempoNest units (commit 9728272, `tndm=True`) | silently wrong noise or data | use GitHub fdbe6eb for both the PPTA and the YA reproduction; use the TN-convention EPTA files with the TN DM normalisation; both variants recorded in the manifest |

---

## 8. Open questions for review

1. Which T defines the YA common basis (whole array 19.5 yr, or something else)?
2. PPTA: the GitHub variant matches the paper's TOA table; is CSIRO v2's extra J1939+2134 data (983 TOAs) worth a separate check?
3. Is a tempo2/libstempo build acceptable in the oracle group for the parity test (R3)?
4. Stage-3 GPU budget (~100-400 h at YA fidelity): run the DM-fixed configuration first?
