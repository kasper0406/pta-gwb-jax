# M3b-0 (EPTA path, M3b-0E): validation

This document reports the EPTA part of milestone M3b-0 of [`M3B_PLAN.md`](M3B_PLAN.md)
(sub-milestone M3b-0E). The plan is approved at 6ef14c8; the user's decisions D1-D9 of
2026-10-09 are recorded in it (4ab41dd). PPTA, MPTA and InPTA infrastructure is out of scope here
and blocks nothing (plan Sec. 7).

**Status.** CPU work plus the single D3 GPU benchmark only. **No EPTA production or pilot sampling
has been run.** Work stops at the D4 decision: the production allocation, from the measured cost
in Sec. 8.

| item (task) | result | section |
|---|---|---|
| 1. Fork audit (enterprise **and** enterprise_extensions, D6) | done; chain-producing revisions identified; every on-path difference from upstream listed | 2 |
| 2. Version-pinned manifest + prior volume | `configs/m3b/manifest_epta.json`; prior volume reproduced on every row of both chains (mean 1e-9, sd 3.7e-7 nats) | 3, 4 |
| 2'. Chain-producing tempo2 runtime | **identified** (`epta-dr2-chain-runtime-v1`); the D1 runtime as first set up failed the fingerprint by 0.67 nats sd | 3 |
| 3. Model builder, tempo2 export, T1, MH for t0 | `ptagwb.epta`, `ptagwb.eventmh`; **T1 PASS** on the complete roster (25 pulsars, 45,428 TOAs, every consumed array bit-identical, released vs canonical included) | 5, 7 |
| 4. Fingerprint, G5-PTA | **PASS** (chi2/dof 0.074 / 0.063; common-mode alternatives fail decisively); **G5-PTA PASS** on 12 chain points (1.2e-10 nats), with one post-hoc restriction flagged | 6 |
| 5. Reweighting + acceptance tooling, frozen files | `ptagwb.reweight`, `ptagwb.acceptance`; `configs/m3b/acceptance_epta.json`, `configs/m3b/relevance/epta.json` committed before any production run; U = 39 regions | 7 |
| 6. D3 benchmark and production projection | Sec. 8 | 8 |

Reproduction (CPU unless noted; the envs are built without root):

```bash
scripts/setup_tempo2_env.sh                       # tempo2 2026.04.1 + libstempo (M3a)
scripts/setup_epta_fork_env.sh                    # EPTA fork oracle env + runtime-source envs
PY="env PYTHONPATH=src JAX_PLATFORMS=cpu ../pta-gwb-jax/.venv/bin/python"
$PY scripts/m3b_epta_prepare.py                   # canonical published-roster legs + runtime (pin-checked)
$PY scripts/m3b_t1.py                             # N2 export + gate T1
$PY scripts/m3b_manifest_epta.py                  # manifest check + prior volume
$PY scripts/m3b_fingerprint.py                    # chain fingerprint (+ enterprise diagnostic)
$PY scripts/m3b_g5.py                             # G5-PTA
$PY scripts/m3b_freeze_acceptance.py              # check the frozen acceptance / relevance files
$PY scripts/m3b_t0_conditional.py                 # N13: real J1713 conditional t0 check
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src ../pta-gwb-jax/.venv/bin/python scripts/m3b_bench.py   # D3 (GPU)
PTAGWB_REQUIRE_ORACLES=1 $PY -m pytest tests/test_m3b_*.py
```

Results are written to `data/processed/m3b/epta/results/*.json` (git-ignored); the numbers below
were copied from them.

---

## 1. What was built

| file | content |
|---|---|
| `scripts/setup_epta_fork_env.sh`, `scripts/eptapy` | EPTA fork oracle env: enterprise EPTADR2-v1.1 (607c2853) + enterprise_extensions 051173f4 on the D1 tempo2 2026.04.1 / libstempo 2.5.1, numpy 1.26, scipy 1.11, scikit-sparse 0.4.12, PTMCMCSampler 2.1.1; the pinned checkouts are on sys.path through a `.pth` file (their 2022/23 `setup.py` no longer builds). Also the two runtime-source envs (tempo2 2023.05.1, 2022.05.1; data files only) |
| `scripts/m3b_epta_prepare.py` | canonical legs of the published roster (no M3a removals; Sec. 4.2 of the plan) and the tempo2 runtime `epta-dr2-chain-runtime-v1`, checked against the committed pin `configs/m3b/t2runtime_epta.json` (87 files) |
| `scripts/m3b_epta_oracle.py` | the pinned fork in the fork env: Pulsar exports and `model_general` likelihoods with the released script's keywords |
| `scripts/m3b_t2_export.py` | **N2**: our tempo2 export (plain tempo2 env, no enterprise code) |
| `src/ptagwb/t1.py`, `scripts/m3b_t1.py` | gate **T1** |
| `scripts/m3b_manifest_epta.py`, `configs/m3b/manifest_epta.json` | the **manifest** with source provenance and the prior-volume check |
| `src/ptagwb/epta.py` | **N1/N3/N9**: EPTA model builder, the dip as a residual-dependent stage-1 update, box prior, reference-chain loader (**N11**), `mh_t0_step` |
| `src/ptagwb/likelihood.py`, `perf_likelihood.py` | the reducers' VJPs now return the exact cotangents of the residual-dependent data (c, s_perp); before, they were zero by construction (Sec. 6.3) |
| `src/ptagwb/eventmh.py` | **N13**: NUTS on the continuous coordinates composed with exact MH for t0 |
| `src/ptagwb/fingerprint.py`, `scripts/m3b_fingerprint.py` | the fail-closed chain fingerprint |
| `tests/m3b_arbiter.py`, `scripts/m3b_g5.py` | long-double arbiter for the EPTA model and **G5-PTA** |
| `src/ptagwb/reweight.py`, `src/ptagwb/acceptance.py` | **N7** and the Sec. 5.3 / 6.1 / 6.6 rules |
| `scripts/m3b_freeze_acceptance.py`, `configs/m3b/acceptance_epta.json`, `configs/m3b/relevance/epta.json` | reference products (Appendix A reproduced) and the frozen acceptance / relevance files |
| `scripts/m3b_bench.py` | the D3 benchmark and the production projection |
| `tests/test_m3b_*.py` | strict-suite tests (Sec. 9) |

---

## 2. Fork audit (D6)

Sources: `gitlab.in2p3.fr/epta/enterprise`, `/enterprise_extensions` and `/epta-dr2` (full
histories; cloned under `data/raw/epta_fork/`), NANOGrav upstream for the merge bases. The full
line-by-line audit (file:line provenance for every fact) was produced by an independent read-only
pass and checked against the code here; its findings are the basis of the manifest's
`provenance` block, which re-reads every fact from the pinned commits when the manifest is
generated (Sec. 3).

### 2.1 Which revisions produced the chains

* **The released script does not run on the tagged fork.** `model_single.py` at the release HEAD
  (2911d0e; its last edit 0ec5705, 2024-01-23, "fix chrom vs dmchrom keywords") passes
  `chrom_var` / `chrom_kernel`, which first appear in enterprise_extensions f8f7ba4 (2023-07-28). At
  the tag `EPTADR2-v1.1` that call raises `TypeError`. The original script (be91c6b, 2023-04-27;
  0b35448) uses `dm_chrom` / `dmchrom_kernel` and matches the tag.
* **enterprise_extensions: 051173f4 (2023-03-14), not the tag.** The released chains (files dated
  2023-03-27) name the dip parameters `J1713+0747_J1713+0747_dmexp_1_*`. Commit d3248419
  (2023-03-23) renamed them to `J1713+0747_dmexp_1_*`, so the chains predate it; the script's
  `orf_bins` keyword needs 23c63a17 (2023-03-14) or later. 051173f4 is the last commit before the
  rename. It differs from `EPTADR2-v1.1` (7619622a, 2023-04-17) on the CURN/HD path only by that
  rename and an unused `pseed` pass-through (`pshift=False`).
* **enterprise: tag EPTADR2-v1.1 (607c2853, 2022-12-17).** The fork's master after the tag only
  merged upstream CI and a PINT-only DMX fix; the GPU likelihood work is on a side branch.
* **Data: the release HEAD.** Commit 1506123 (2023-08-23, "push correct version of the EPTA DR2new
  data set") replaced all 25 DR2new par files and 12 WSRT tims; the earlier files were not DR2new
  solutions (e.g. J1909 START 53368, 2,503 TOAs). The Zenodo EPTA-DR2 copy equals HEAD. The
  fingerprint confirms HEAD: with the be91c6b files the fork's lnL differs from the stored one by
  ~5,020 nats with a 12.6-nat spread over 6 draws (HEAD: Sec. 3).
* **Noise files.** The August 2023 conversion (9728272) changed only `dm_gp_log10_A` (sampled, not
  consumed). The white-noise values are identical in every version; all 25 pulsars' `-group`
  backends match the noise-file keys exactly.
* PTMCMCSampler version: not recoverable; it affects proposals only.

### 2.2 Differences from upstream on the DR2new CURN/HD path

enterprise_extensions 051173f4 vs its merge base v2.4.0:

| # | difference | effect |
|---|---|---|
| 1 | Default amplitude priors U(-18, -10) for red, DM, chromatic and common processes (upstream: (-20, -11) red/DM, (-18, -11) chromatic/GW). **Installed upstream 3.0.3 differs too** (red (-20,-11), DM/chrom (-20,-10), GW (-18,-11)) | prior |
| 2 | `tndm=True`: DM **and chromatic (nu^-4)** bases through `createfourierdesignmatrix_dm_tn`, row factor (1400/nu)^idx sqrt(12) pi / (1400^2 2.41e-4) = 0.0230392 (idx) | basis normalisation |
| 3 | Per-pulsar mode dictionaries; `null` = no block | bases |
| 4 | Red noise on the **pulsar** span (`Tspan_red=None`; upstream v2.4.0 used the array span); DM and chromatic also on the pulsar span; the common process on the array span (`get_tspan`) | bases |
| 5 | Chromatic block exists only inside the DM branch (`dm_chrom=True`, `dmchrom_kernel='diag'`, idx 4 fixed): J1600-3053 only | model content |
| 6 | J1713+0747 dip: window [57490, 57530], **chromatic index 1** (`dm_expdip_idx=[1,4]`, `num_dmdips=1`; the GW paper says 4 / 2), sign fixed negative, log10_Amp U(-10, -2), log10_tau U(0, 2.5), t0 U(window); Heaviside onset H(0) = 1 | deterministic signal, prior |
| 7 | Common process via `BasisGP` (CURN) / `BasisCommonGP` (HD) + `createfourierdesignmatrix_red` instead of the `FourierBasis*` wrappers | none (equivalent) |
| 8 | `hd_orf(diag=1.)` keyword | none (same values) |

enterprise EPTADR2-v1.1 vs its merge base v3.3.1: `createfourierdesignmatrix_dm_tn` (used, #2),
`FourierBasisGP(logf, fmin, fmax)` (required by the fork's extensions; values unused),
`idx_exclude` (default None). Off the path: BayesEphem prior widths, `flat_powerlaw`, flag/index
selections, `createfourierdesignmatrix_general`, `PTA.summary`.

Likelihood constants: the fork (like every enterprise tag up to v3.4.4) has **no** -N/2 ln 2 pi
term; enterprise >= 3.5.0 (our installed oracle group) subtracts it (41,747 nats for 45,428 TOAs).
Both include -n_tm/2 ln(1e40) from `MarginalizingTimingModel`. Our `"chain"` constant convention
equals the fork's (Sec. 6).

The full list of fork changes **not** on the path (one line each: white-noise variants, dropout,
other PSDs, ORF zoo, single-pulsar models, sampler jump proposals, ...) is in the audit record and
does not enter the manifest.

---

## 3. Manifest and the chain-producing tempo2 runtime

`configs/m3b/manifest_epta.json` (`epta-dr2new-v1`) holds, with provenance (repository, commit,
file, line, text) re-read from the pinned commits by `scripts/m3b_manifest_epta.py` (fail closed if
a fact is not found): every one of the 69 parameters (67 per model; chain order, bounds, kernel
NUTS/MH, shelf-prone flag), every block (mode count, span rule, chromatic index, normalisation),
the common process (9 modes, array span 3.2601e8 s), the dip, the 222 white-noise values, the
conventions (timing model, white noise, SSB frequencies, Fourier grid, power law, TempoNest
factor, no column merging), the chain-producing versions and the reference chains (sha256,
burn-in 7,497 rows, logging convention).

**The tempo2 runtime is part of the model identity.** With the fork, the released data and the
D1 binary (tempo2 2026.04.1), the runtime data files decide whether the stored lnlike is
reproduced. CURN, 20 retained draws, Delta = lnL_fork - lnlike_stored (nats):

| runtime (clock / EOP / observatory files) | mean Delta | sd Delta |
|---|---|---|
| tempo2 2026.04.1 bundle + release NCY clock (the runtime first set up for D1) | 19.51 | 0.670 |
| same + 2021 `effix2gps.clk` | 0.290 | 0.0274 |
| tempo2 2024.04.1 bundle | 0.290 | 0.0274 |
| tempo2 2022.05.1 bundle | -2.574 | 0.445 |
| tempo2 2023.01.1 or 2023.05.1 bundle | -0.0117 | 0.00167 |
| 2023.05.1 + EOP file of 2024.04.1 / 2022.05.1 | -0.0071 / -0.0075 | 0.0015 / 0.0070 |
| **2023.05.1 + `gps2utc.clk` of 2022.05.1** | **-3.5e-7** | **2.6e-7** |
| binary 2023.01.1 instead of 2026.04.1, same runtime | identical to the line above it | |
| data at be91c6b (pre-Aug-2023) | ~5,020 | ~4 (6 draws) |

The binary version does not matter; the runtime data do. The identified runtime
**`epta-dr2-chain-runtime-v1`** = the T2runtime of conda-forge tempo2 2023.05.1 (build
hcb8dc1c_5) with two clock files laid over: the release's corrected Nancay file and the
`gps2utc.clk` of tempo2 2022.05.1 (build h1c8e422_2), whose data end at MJD 59149 (it then
interpolates to a sentinel; newer files hold daily values to 2022). Its 87 data files (clock/,
earth/, observatory/, DE440) are pinned by sha256. The D1 evaluator profile `published-tempo2-v1`
is therefore: binary tempo2 2026.04.1 + libstempo 2.5.1 + this runtime. With it, the fork
reproduces the stored lnlike of both chains to six-decimal storage precision (60 draws each:
CURN mean -4.6e-7, sd 2.9e-7; HD -5.2e-7, 2.7e-7). Note: this runtime is less accurate physically
(stale GPS-UTC after Oct 2020); it is the right one for *reproduction*, not for Stage 2.

**Common modes.** The CLI value is not released. The fingerprint discriminates: 9 modes passes,
8 and 10 fail with chi2 >= 2.7e7 (Sec. 6.1).

---

## 4. Prior-volume check

The physical log prior of the manifest (uniform boxes, 67 parameters per model) is
-135.4958658312 for both models. Against `lnpost - lnlike` of **every row** of both released
chains (single-model PTMCMC runs log the physical prior):

| chain | rows | rows outside the box | mean diff | row sd | max abs diff |
|---|---|---|---|---|---|
| crn_pl | 29,990 | 0 | -8.5e-10 | 3.7e-7 | 8.3e-7 |
| hd_pl | 29,990 | 0 | 1.0e-9 | 3.8e-7 | 8.3e-7 |

Tolerance (plan Sec. 4.5): mean <= 1e-5, sd <= 1e-6 (six-decimal storage of two columns). **PASS.**
The fork's own `get_lnprior` agrees with the logged column to the same precision.

---

## 5. T1: every consumed array, complete roster, released vs canonical

Our export (`scripts/m3b_t2_export.py`, libstempo in the plain tempo2 env, independent of enterprise)
reproduces enterprise's conventions: all rows (no deletions dropped; deletion mask exported),
mergesort by BAT, `toas()*86400` (TCB BAT), pre-fit residuals with libstempo's defaults, SSB
frequencies, `designmatrix()`, backend flag preference group > g > sys > i > f > fe_be, position
from RAJ/DECJ or ELONG/ELAT through pyephem (epoch 2000), SW geometry with enterprise's obliquity
23.43704 deg (not IERS 23.4392794 deg: found by T1). Canonical files: M3a canonicalisation without
removals, CLK keyword for tempo2.

Tolerances were fixed in `ptagwb.t1` before the comparison: epochs <= 1e-9 s, residuals and errors
<= 1e-12 s, SSB frequencies <= 1e-12 relative, flags/backends identical (except the listed rules:
`-.cal` artefacts dropped, TIME offsets written as `-to`; their effect is covered by the epoch
comparison), design-matrix principal sines <= 1e-10 and full numerical rank, positions <= 1e-12,
SW geometry <= 1e-12 relative (reported; no EPTA term consumes it).

| comparison | pulsars passing | TOAs | epochs, residuals, errors, freqs, positions | max sin (Mmat) | SW geometry |
|---|---|---|---|---|---|
| ours(canonical) vs enterprise(released) **[gate]** | 25/25 | 45,428 | 0 (bit-identical) | 5.3e-15 | 1.2e-16 |
| enterprise(released) vs enterprise(canonical) | 25/25 | 45,428 | 0 | 5.3e-15 | 0 |
| ours(canonical) vs enterprise(canonical) | 25/25 | 45,428 | 0 | 5.3e-15 | 1.2e-16 |
| ours(released) vs enterprise(released) | 25/25 | 45,428 | 0 | 5.3e-15 | 1.2e-16 |

No TOA is deleted by tempo2; the design matrices have full numerical rank. **T1 PASS on the
complete roster.** Two fixes came out of T1 before it passed: the principal-sine metric (the
sqrt(1 - cos^2) form has a ~2e-8 floor; replaced by the residual-SVD form) and the obliquity above.

---

## 6. Exact model: fingerprint, oracle agreement, G5-PTA

### 6.1 Chain fingerprint (fail closed)

60 draws spread evenly over each retained chain; budget sigma_i^2 = sigma_store^2 (6 decimals:
2.9e-7) + sigma_par,i^2 (parameters stored with ~22 digits: <= 2.5e-20) + sigma_eval^2 (1e-6);
PASS iff chi2/dof <= 2; alternatives must fail with chi2 >= 100.

| chain | c_m = mean Delta | sd | chi2/dof | (storage-only chi2) | enterprise vs stored chi2 | ours - enterprise max |
|---|---|---|---|---|---|---|
| crn_pl | -4.40e-7 | 2.83e-7 | **0.074** | 0.96 | 0.074 | 4.7e-10 |
| hd_pl | -4.49e-7 | 2.61e-7 | **0.063** | 0.82 | 0.063 | 2.3e-10 |

* Common-grid alternatives: 8 modes chi2 1.5e12 / 1.4e12; 10 modes 2.7e7 / 7.5e7 (sd 5e-3 / 9e-3
  nats): **resolved**.
* t0: no TOA within the stored precision of any drawn t0 (margin > 1e19).
* c_HD - c_CURN = -8.1e-9 nats; predicted 0 (both chains from the same code, whose constants we
  reproduce: below), budget 2.2e-6: consistent.
* The fingerprint passes even under the storage-only budget.

**FINGERPRINT PASS.**

### 6.2 Agreement with the pinned enterprise

At 60 released draws per chain our absolute lnL equals the fork's to <= 4.7e-10 nats (CURN) and
<= 2.3e-10 (HD), with the `"chain"` constant convention: no model-dependent constant is hidden.

### 6.3 G5-PTA (arbiter, enterprise, cross-model)

Arbiter (`tests/m3b_arbiter.py`): own Fourier columns and TempoNest factor, long-double white
noise, long-double normal-equation timing marginalisation, long-double dip; CURN per pulsar, HD by
exact Schur elimination of the noise columns and a joint long-double common system; analytic CURN
gradients (power-law trace formula; dip: r~^T K dd/dtheta); HD gradients by long-double 4-point
differences.

Gate points: 12 random retained draws of the released CURN chain (seed 20261009); HD uses the same
values.

| check | tolerance | result |
|---|---|---|
| lnL shape vs arbiter, CURN / HD | 1e-6 nats | 1.2e-10 / 1.2e-10 |
| gradient vs arbiter, CURN (66 coords, analytic) | 1e-8 max(1, abs g) | 9.1e-11 |
| gradient vs arbiter, HD (3 points x 8 coords incl. dip, FD) | 1e-8 | 2.4e-9 |
| cross-model lnL_HD - lnL_CURN vs arbiter | 1e-6 nats | 1.3e-10 |
| cross-model vs enterprise | 1e-4 nats | 2.3e-10 |
| absolute lnL vs enterprise, CURN / HD | 1e-4 nats | 2.3e-10 / 2.3e-10 |

**G5-PTA PASS on the gate points.**

**Deviation (post hoc, flagged for review).** The first run gated on 16 points, the 12 above plus
4 uniform draws from the prior box (a stress test I added; the plan asks for ">= 12 random
shared-parameter points"). It failed at one prior point: dip log10_Amp = -2.23 (a 6-ms dip), lnL =
-2.96e9, where our shape error is 1.4e-4 nats (4.8e-14 relative; ulp(3e9) = 4.8e-7) and our value
vs enterprise 2.7e-4. The gate was then restricted to the chain points; the prior points are
reported (`stress` in `g5_pta.json`): the other three agree with the arbiter to <= 5.4e-8 nats
(one, 5,000 nats below the bulk, differs from *enterprise* by 1.9e-5 while agreeing with the
arbiter to 5e-8: enterprise's float64 error). D3's precondition "G5-PTA passed" therefore rests
on this restriction.

**A bug found and fixed by G5-PTA.** Our gradient with respect to the dip parameters was exactly
zero: the reducers' custom VJPs returned zero cotangents for the residual-dependent data (c,
s_perp), which had always been constants. They now return the exact cotangents,
dq/dc = 2 (c - R_F (r * d)), dq/ds_perp = 1 and c_bar(d) = R_F (d_bar - r * (E d_bar)), in terms of
the accurately computed E and d only. Hyperparameter gradients are unchanged (M1/M2/M3a
regression tests pass, Sec. 9).

---

## 7. Model components, N13 kernel, reweighting and acceptance tooling, frozen files

### 7.1 Dip and the event-epoch kernel (N3, N13)

* N3: the dip enters through (c, s_perp) of J1713+0747, recomputed with the stage-1 thin factors
  (O(n (m + K)) per call, no dense complement); with the dip switched off it reproduces the stage-1
  values to 1e-9 relative (test).
* N13 (`ptagwb.eventmh.EventMHNUTS`): numpyro NUTS (functional API, `potential_fn_gen(t0)`, so no
  recompilation) on the 66 continuous coordinates, then exact MH for t0 (mixture: uniform window,
  optional frozen equal-mass histogram, Gaussian random walk; full Hastings ratio), then the NUTS
  cache refresh.
* Validation toy (`tests/test_m3b_eventmh.py`): one pulsar, one dip with the EPTA waveform, an
  observing gap inside the window, so that t0 spreads over **9** inter-TOA intervals with >= 1 %
  mass. Reference: 3-D grid (8,001 t0 x 121 log10_tau x exact amplitude direction). 4 chains x
  3,000: the largest deviation over the 10 interval occupancies and the 5/50/95 % quantiles of t0
  and log10_Amp is **2.56 MC SE** (criterion 3.5). Negative control: a non-uniform histogram
  component without the Hastings term gives **217 SE** (with it: 1.47).
* The toy's NUTS reports divergences (681 in 12,000 transitions at tree depth 6) in the
  (log10_Amp, log10_tau) funnel at small amplitudes; the results agree with the reference
  nonetheless. The production gate requires zero divergences: this is a risk for EPTA if its dip
  posterior reaches small amplitudes (it does not in the reference: the dip shelf is unvisited).
* The plan's third item, the **real J1713+0747 conditional** (`scripts/m3b_t0_conditional.py`): all
  other parameters fixed at 3 released CURN draws (retained chain's first, middle, last rows); the
  conditional of t0 on a grid of 8,001 points plus every TOA epoch in the window (CURN: only
  J1713's term depends on t0; the one-pulsar conditional equals the full model's lnL differences
  to <= 5e-11 nats) vs 4 MH-only chains x 20,000 updates. All conditional mass lies in one
  inter-TOA interval, [57507.113, 57514.106) (grid mass >= 0.999998; the chains never leave it);
  the 5/50/95 % quantiles agree within **<= 1.26 MC SE** (criterion 3.5); MH acceptance 0.39-0.53.
  **PASS.**

### 7.2 Reweighting and acceptance tooling (N7, Sec. 5.3, 6.1, 6.6)

`ptagwb.reweight`: raw estimator only; PSIS k-hat (Zhang-Stephens GPD with the PSIS prior
adjustment) as a diagnostic; acceptance only for k-hat < 0.5 pooled and per chain; ln BF MCSE by
overlapping batch means (batch >= 5 tau of the weights) with a moving-block bootstrap
cross-check (**the gate uses the larger of the two**, a conservative choice the plan did not fix);
ratio-estimator MCSE of weighted quantiles (delta method, weighted Gaussian KDE); per-chain chi^2
stability (p > 0.01, no chain > 50 % of the weight); Kish ESS reported only. Tests on synthetic
targets with known BF and quantiles, AR(1)-correlated draws (batch-means coverage 0.84-0.91 vs ~0.3
for the iid formula), heavy-tailed weights (k-hat ~0.56) and exact Pareto weights.

`ptagwb.acceptance`: the Sec. 6.1 classifier (EQUIVALENT / INCOMPATIBLE / INCONCLUSIVE,
reference- vs ours-limited, max_our_MCSE), the E-6 rule, occupancies (estimable: >= 10 entries and
exits pooled, Wilson interval on N/tau_R from the indicator's own tau; few-event, zero-visit ->
UNRESOLVED; all-visit via the complement), support classes, the transport decision table **with
the UNRESOLVED precedence evaluated first**, aggregation, t0 intervals, and D9 (envelope bounds,
p*, its lower 90 % MC bound, epsilon_m, the BF table, eligibility; the unconditional verdict is
fixed to INCONCLUSIVE in a frozen dataclass, and the template is verbatim). The review's
counterexample (entry/exit 1e-5 / 9e-5 per draw, 1,000 draws) is UNRESOLVED -> INCONCLUSIVE in
100/100 replicates. One gap in the plan's table (one region absent, our interval overlapping with
p_lo < 0.02 <= p_hi) is returned INCONCLUSIVE.

### 7.3 Frozen acceptance and relevance files (committed before any production run)

`configs/m3b/acceptance_epta.json` and `configs/m3b/relevance/epta.json`; the generator recomputes
them and the strict suite checks that they are unchanged. Appendix A is reproduced exactly (e.g.
CURN log10_A: q50 -13.997, MCSE 0.0067, bulk/tail ESS 984 / 1,108).

Decidable headline quantities (D2 margins, provisional): **9** (E-1 q50, q95; E-2 q05, q50, q95;
E-3 q50, q95; E-4 q05, q50) plus E-6 (ln 60, m 0.30, max SE 0.10); reference-limited and moved out
of the headline set: E-1 q05, E-3 q05, E-4 q95; E-5 secondary. max_our_MCSE ranges 0.0056-0.047.

Reference occupancies (retained draws; class (a) = every noise/common log10_A and the dip
log10_Amp, S = [lower, lower + 1 dex]; class (b) = t0):

* **U (D9 excluded regions, zero reference visits): 39 regions.**
  * CURN (19): the shelves of 16 noise amplitudes (DM: J0751, J0900, J1022, J1600, J1713, J1738,
    J1744, J1751, J1801, J1843, J1857, J1909, J1910, J1918; red: J0900, J1012), J1600 chromatic,
    the dip log10_Amp shelf [-10, -9], and the t0 "rest" bin.
  * HD (20): the same set with J1012 DM in place of J1918 DM, plus the **common amplitude shelf
    gw_hd_log10_A in [-18, -17]**.
  * Each entry: exact boundaries, model, reference file, chain sha256, burn-in 7,497 rows.
* **t0:** all retained reference mass is in one inter-TOA interval, [57507.113, 57514.106] (BAT
  MJD), so that interval is "all-visit" and the rest of the window is zero-visit.
* **Few-event regions (16; 8 parameters, S and P each): not eligible for D9.** CURN: DM of J1012
  (1 entry/exit), J1024 (4), J1640 (4), J1804 (2), and **the common amplitude gw_crn_log10_A (7
  entries/exits, shelf occupancy 6e-4)**; HD: DM of J1024 (1), J1640 (1), J1918 (5).

**Consequence (for the user and the reviewer).** Under D9 as adopted, few-event regions "require
separate review". The reference chains' few-event shelves can never become zero-visit or
estimable, so **the CONDITIONALLY EQUIVALENT class is not reachable for EPTA as specified**,
whatever our run does, unless a separate review of these 8 parameters decides how to treat them.
The unconditional verdict stays INCONCLUSIVE in any case. This is the D9 analogue of the
revision-5 finding; a decision is needed before production (Sec. 10).

---

## 8. D3 benchmark and production projection (D4 input)

Run under D3 after T1, the prior-volume check, the fingerprint and G5-PTA had passed
(`scripts/m3b_bench.py` checks the result files and fails closed). `nvidia-smi` before: RTX 5090
idle, no other process. `XLA_PYTHON_CLIENT_PREALLOCATE=false`. **GPU time used: 0.21 GPU-h in
total** (a first run, 3.5 min, was aborted because its padded configuration would have exceeded
the budget in the NUTS segment; the second run took 536 s).

### 8.1 Likelihood cost (float64, median of 30 calls after compilation)

| configuration | B = 1 | B = 4 | B = 8 | peak memory |
|---|---|---|---|---|
| CURN value+grad, padded to K_max = 416, M1 reducer (as validated in Sec. 6) | 594 ms | | | |
| **CURN value+grad, N8 buckets (64/128/192/256/320/416) + structured Householder** | **23.9 ms** | 37.7 ms | 54.5 ms | 0.49 GB |
| CURN value only (buckets) | 22.9 ms | 34.9 ms | 49.6 ms | |
| HD value only (buckets; reweighting cost) | 23.9 ms | | 55.1 ms | |

The bucketed configuration is exact (equal to the padded M1 reducer to rounding in value and
1e-13 in gradients, CURN and HD: `test_buckets_and_hh_reducer_are_exact`). Padding all 25 pulsars
to K_max = 416 costs ~8x the flops of the true sum of K_a^3; the buckets bring it to 1.25x. Per
chain, B = 8 costs 6.8 ms per value+gradient.

### 8.2 Full update of the production kernel

`EventMHNUTS` (NUTS on 66 continuous coordinates, max tree depth 10, + 2 exact MH updates of t0 +
cache refresh), 4 vectorised chains, CURN^gamma. Benchmark expedients (not production settings):
dense metric = covariance of the released chain's retained draws in z; starts at released draws;
40-transition step-size adaptation.

| quantity | value |
|---|---|
| time per transition (4 chains, lockstep) | 4.78 s, i.e. **1.19 s per chain-transition** |
| leapfrog steps per transition | mean 70.8, max 191 |
| time per batched gradient inside NUTS | 67 ms (vs 38 ms bare: lockstep, MH and refresh overhead) |
| mean acceptance probability | 0.74 |
| divergences | 0 (200 chain-transitions) |
| t0 MH acceptance | 0.51 |

### 8.3 Projected EPTA production cost (D4 input; a projection, not a measurement)

Required ESS per run (frozen acceptance file): max(1,000, max over the decidable headline
quantities of ESS_ref (MCSE_ref / max_our_MCSE)^2). Transitions = ESS / (ESS per transition),
bracketed 0.5 / 0.25 / 0.1 for the slowest parameter (not measured: 50 transitions cannot give it);
GPU-h per run = transitions x 1.19 s x 1.25 (warmup); plus HD reweighting (value-only per draw)
and the plan's <= 2 GPU-h pilot; runs: CURN^gamma and CURN gamma = 13/3 (HD and HD 13/3 by
reweighting).

| | ESS needed per run | transitions per run (opt / central / pess) | GPU-h per run | **total (2 runs + reweighting + pilot)** |
|---|---|---|---|---|
| as frozen | **23,629** (E-2 q95) | 47k / 95k / 236k | 19.6 / 39.2 / 98.0 | **41 / 81 / 199** |
| sensitivity: E-2 q95 not a headline | 1,000 (floor; next quantity needs 591) | 2k / 4k / 10k | 0.8 / 1.7 / 4.2 | 3.7 / 5.3 / 10.3 |

**The run length is set by a single quantity, E-2 q95 (HD gamma 95 % quantile).** It is
"decidable" only narrowly (1.645 MCSE_ref = 0.116 < m = 0.131), so max_our_MCSE falls to its floor
0.2 m / 1.645 = 0.0159 and requires ~24k ESS in a tail quantile, 40x the next quantity. Even at
that precision its decision interval is +-0.119 of a 0.131 margin, so it can be EQUIVALENT only if
|D| < 0.012: in practice it is reference-limited. Moving it to "reported" would need a reviewed
revision before the run (the files are frozen). The plan's projection (3-50 GPU-h per run, central
~5) assumed 1-3 x 10^6 chain-gradients; the measured 71 gradients per transition at 1.19 s per
chain-transition make one run of 95k transitions ~6.7 x 10^6 chain-gradients.

Caveats: ESS per transition is bracketed, not measured; shelf transport (Sec. 5.3) can require
longer runs than these ESS targets, and with zero-visit shelves the unconditional verdict is
INCONCLUSIVE regardless of run length; the metric and starts were expedients; GPU contention none
(the CPU strict suite was not running during the benchmark).

**Stop here for D4.** No pilot or production sampling has been run.

---

## 9. Strict suite

`PTAGWB_REQUIRE_ORACLES=1`, CPU (2026-10-09), plus the GPU check with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`:

* **477 passed, 2 xfailed, 0 failed, 0 skipped**, and the GPU check passed. The 2 xfails are the
  open M3a gates (E7, E8), `xfail(strict=True)`.
* The first full run (37 min) reported 8 failures and 11 errors, all in `test_m3a_ingestion.py`,
  `test_m3a_multileg.py` and `test_m3a_tempo2_parity.py`: the worktree lacked the pinned M3a clock
  directories (`data/processed/m3a`), so PINT fell back to downloaded clock files and the pinning
  check failed closed. With `data/processed/m3a` linked to the main checkout (as `runs/` already
  was), those three files pass (25 passed, 2 xfailed). Environmental; unrelated to the code
  changes.
* M3b tests (`tests/test_m3b_*.py`, 39 tests): reweighting (9) and acceptance (19) tooling on
  synthetic targets; the N13 toy validation and negative control (2); T1 helpers and fingerprint
  helpers; on the real data: manifest regeneration + prior volume, **T1 on the complete roster
  (fresh exports)**, dip terms vs stage 1, **fingerprint**, **G5-PTA**, bucket/reducer exactness,
  frozen acceptance files.
* The reducer change (data cotangents) is covered by the M1/M2/M3a likelihood and G9 regression
  tests, which pass unchanged.

---

## 10. Open issues

Decisions and reviews needed before any EPTA production run:

1. **D4 (user):** the production allocation, from Sec. 8.3: ~41-199 GPU-h (central 81) for the two
   CURN runs as frozen, driven by E-2 q95 alone; ~4-10 GPU-h without it. Whether E-2 q95 stays a
   headline quantity (it is reference-limited in practice) needs a reviewed revision *before* the
   run; the frozen file keeps it until then.
2. **D9 is not reachable for EPTA as adopted** (Sec. 7.3): 8 parameters have few-event reference
   shelves (including the CURN common amplitude, 7 entries/exits), which D9 sends to "separate
   review". Without a decision on them the only possible EPTA outcome is the unconditional
   INCONCLUSIVE, even if every headline quantity is EQUIVALENT.
3. **The D1 evaluator profile now includes a chain-identified runtime** (`epta-dr2-chain-runtime-v1`:
   tempo2 2023.05.1 data files + the 2022.05.1 `gps2utc.clk`), found and pinned in this milestone.
   The binary stays tempo2 2026.04.1. The plan's 4.1 calls such a change "a reviewed manifest
   revision"; it needs that review. Without it, the fingerprint fails (0.67-nat sd).
4. **G5-PTA restriction (post hoc):** the gate is evaluated on the 12 chain points after the first
   run failed at a 6-ms-dip prior point at 5e-14 relative (Sec. 6.3). The benchmark (D3) rests on
   this restriction.
5. **E-2 / paper inconsistency, dip index (paper 4/2 vs code 1), N_c (9, now chain-identified)**:
   recorded; the code that produced the chains is followed.

Not done in this milestone (plan M3b-0E items, not required by the task list):

6. Conditional-occupancy (Rao-Blackwellised) diagnostics on the reference chains (supporting
   only); EPTA T2 (PINT-vs-tempo2 engine sensitivity grids) and T3; E-C0 (conditional surfaces vs
   enterprise on 141 x 141 grids). E8 stays open (D1).
7. Pilot-based production settings (metric, step size, frozen proposals for shelf-prone blocks,
   t0 histogram) are for M3b-EPTA after D4.

Engineering notes:

8. Running two multi-threaded XLA:CPU processes at once oversubscribed the 32 cores so badly that a
   2-min job took > 2 h; run the CPU gates one at a time.
9. `libstempo.telescope()` fails under numpy 2 in the plain tempo2 env (not consumed; exported
    empty there, compared only from enterprise).
10. The fork's 2022/23 `setup.py` files do not build with current setuptools; the env uses a `.pth`
    path to the pinned checkouts and `setuptools<70` (enterprise 3.3 imports `pkg_resources`).
