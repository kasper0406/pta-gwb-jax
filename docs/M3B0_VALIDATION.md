# M3b-0 (EPTA path, M3b-0E): validation

This document reports the EPTA part of milestone M3b-0 of [`M3B_PLAN.md`](M3B_PLAN.md)
(sub-milestone M3b-0E). The plan is approved at 6ef14c8; the user's decisions D1-D9 of
2026-10-09 are recorded in it (4ab41dd). PPTA, MPTA and InPTA infrastructure is out of scope here
and blocks nothing (plan Sec. 7).

**Status.** CPU work plus the single D3 GPU benchmark only. **No EPTA production or pilot sampling
has been run.** The pilot waits for the independent review of this revision and the
coordinator's confirmation.

**Revision 2 (after the review of 2ee1bb7, REQUEST_CHANGES; `review_m3b0.out`), and the user's
second set of decisions of 2026-10-09** (recorded in the plan):
1. **G5-PTA**: the post-hoc restriction is withdrawn. All 16 points are kept, 8 fixed cases are
   added (4 shelf/corner, 4 dip-onset-adjacent), the reviewer's mixed criterion was frozen before
   the rerun (827d99d), and both the padded and the bucketed production configuration are gated:
   **PASS** (Sec. 6.3). The first run's "float64 representability" was wrong: the error was about
   300 ulp.
2. **Cost**: per-run provisional scenarios with each run's own targets, weighted-quantile MCSE
   requirements from reweighting the released CURN draws, and the complete kernel cost
   (amplitude-block and dip-block MH included) (Sec. 8.3).
3. **Fail-closed hygiene**:
   * the fingerprint verdict now requires grid discrimination and c_HD - c_CURN consistency;
   * every gate result is bound to hashes of the evaluated source, configs, runtime and exported
     inputs (`ptagwb.binding`), and the benchmark and run driver refuse stale, missing or failed
     evidence;
   * D9 eligibility requires complete inventories and rejects non-finite inputs;
   * negative tests were added for each.
4. **Provenance wording**: these are pinned, source-audited configurations reproducing the
   chain-generating model, with the historical equivalence class stated. 23c63a17 is corrected to
   23c63a15. The 2026-runtime scatter is reported as a parameter-dependent sensitivity, together
   with a paired-runtime displacement analysis (Sec. 3).
5. **Scope**: T2 and the conditional-occupancy diagnostics get an explicit deferral proposal for
   review (plan, top). They are not done.
6. **User decisions** (second set, 2026-10-09), implemented:
   * **D9 revised** to a common-domain conditional comparison (24 exclusions; 38 / 78 reference
     draws in U), with p* / epsilon_m retired;
   * **headline rule** 1.645 MCSE_ref <= m/2, so E-2 q95 is reported-only;
   * **D4 cap of 12 GPU-h**, enforced by the run driver (`scripts/m3b_run_epta.py`, `ptagwb.budget`).
     Pilot configs are committed but not started.

   The acceptance file v2 was frozen before any production run (Sec. 7.3).

| item (task) | result | section |
|---|---|---|
| 1. Fork audit (enterprise **and** enterprise_extensions, D6) | done; pinned, source-audited configurations reproducing the chain-generating model (historical equivalence classes stated); every on-path difference from upstream listed | 2 |
| 2. Version-pinned manifest + prior volume | `configs/m3b/manifest_epta.json`; prior volume reproduced on every row of both chains (mean 1e-9, sd 3.7e-7 nats) | 3, 4 |
| 2'. tempo2 runtime | pinned runtime `epta-dr2-chain-runtime-v1` reproducing the stored likelihoods; the 2026 bundle gives a parameter-dependent 0.67-nat scatter, reported as a sensitivity | 3 |
| 3. Model builder, tempo2 export, T1, MH for t0 | `ptagwb.epta`, `ptagwb.eventmh`; **T1 PASS** on the complete roster (25 pulsars, 45,428 TOAs, every consumed array bit-identical, released vs canonical included) | 5, 7 |
| 4. Fingerprint, G5-PTA | **PASS** (chi2/dof 0.074 / 0.063; discrimination resolved; c_HD - c_CURN consistent); **G5-PTA PASS** on all 24 points (16 random + 8 fixed), mixed criterion frozen before the rerun, padded and bucketed production configurations | 6 |
| 5. Reweighting + acceptance tooling, frozen files | `ptagwb.reweight`, `ptagwb.acceptance`; acceptance file v2 (revised D9 common domain, 24 exclusions; revised headline rule: 8 + E-6) and relevance file committed before any production run | 7 |
| 6. D3 benchmark and production projection | benchmark 0.21 GPU-h; per-run provisional projection 4.1 / 6.1 / 12.3 GPU-h incl. a 2 GPU-h pilot; D4 cap 12 GPU-h enforced by the driver | 8 |

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
$PY scripts/m3b_reference_weights.py              # HD/CURN and paired-runtime reweighting of the released draws
$PY scripts/m3b_projection.py                     # per-run provisional cost scenarios
$PY scripts/m3b_run_epta.py configs/m3b/run_configs/epta_pilot_curn_freegamma.json --dry-run   # driver checks only
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
* **enterprise_extensions: a pinned, source-audited configuration of the chain-generating model,
  not a uniquely identified commit.** The released chains (files dated 2023-03-27) name the dip
  parameters `J1713+0747_J1713+0747_dmexp_1_*`; commit d3248419 (2023-03-23) renamed them to
  `J1713+0747_dmexp_1_*`, so the chains predate it. The original script's `orf_bins` keyword needs
  23c63a15 (2023-03-14) or later. The **historical equivalence class** compatible with both facts
  is {23c63a15, 051173f4}; 051173f4 only changes the binned-ORF function, which the CURN/HD path
  does not use, so the two are identical on this path. We pin 051173f4 as its representative.
  `EPTADR2-v1.1` (7619622a, 2023-04-17) differs on the path only by the rename and an unused
  `pseed` pass-through (`pshift=False`), so it also reproduces the model up to parameter names.
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

The two tested binaries (2026.04.1 and 2023.01.1) agree on the tested configuration; the runtime
data files decide. The pinned runtime **`epta-dr2-chain-runtime-v1`** reproduces the stored
likelihoods; it does not uniquely identify every file EPTA had installed (other files with no
effect on these data are equally compatible). It is the T2runtime of conda-forge tempo2 2023.05.1 (build
hcb8dc1c_5) with two clock files laid over: the release's corrected Nancay file and the
`gps2utc.clk` of tempo2 2022.05.1 (build h1c8e422_2), whose data end at MJD 59149 (it then
interpolates to a sentinel; newer files hold daily values to 2022). Its 87 data files (clock/,
earth/, observatory/, DE440) are pinned by sha256. The D1 evaluator profile `published-tempo2-v1`
is therefore: binary tempo2 2026.04.1 + libstempo 2.5.1 + this runtime. With it, the fork
reproduces the stored lnlike of both chains to six-decimal storage precision (60 draws each:
CURN mean -4.6e-7, sd 2.9e-7; HD -5.2e-7, 2.7e-7). Note: this runtime is less accurate physically
(stale GPS-UTC after Oct 2020); it is the right one for *reproduction*, not for Stage 2.

**Runtime sensitivity (parameter-dependent; it cannot be absorbed into a constant).** With the
tempo2 2026 bundle runtime (the D1 runtime as first set up), lnL changes by +19.67 nats on
average, with sd **0.633** nats over the 22,493 retained CURN draws (range 17.1-22.3); for HD the
values are +19.78 and sd 0.627 (every 5th draw). A paired-runtime reweighting of the released draws
(`scripts/m3b_reference_weights.py`; CPU, no sampling; weights = the lnL difference) gives good
overlap (k-hat 0.18 CURN / -0.02 HD; Kish ESS 15,275 / 3,123). The displacement of the headline
quantiles if the 2026 runtime had been used is below. It is a sensitivity of the reproduction to
the evaluator runtime, not a posterior result.

| quantity | displacement (MC SE) | margin m |
|---|---|---|
| CURN log10 A q50 / q95 | +0.023 (0.006) / +0.009 (0.005) | 0.038 / 0.063 |
| CURN gamma q05 / q50 | -0.032 (0.020) / -0.057 (0.017) | 0.167 / 0.100 |
| HD log10 A q50 / q95 | +0.007 (0.005) / +0.001 (0.005) | 0.027 / 0.046 |
| HD gamma q05 / q50 / (q95, reported) | +0.019 (0.015) / -0.014 (0.015) / -0.079 (0.066) | 0.131 / 0.079 / 0.131 |

The runtime alone moves the CURN medians by 0.6x their margins. This is why the runtime is part of
the model identity.

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

Arbiter (`tests/m3b_arbiter.py`):
* own Fourier columns and TempoNest factor;
* long-double white noise, long-double normal-equation timing marginalisation, long-double dip;
* CURN per pulsar; HD by exact Schur elimination of the noise columns and a joint long-double
  common system;
* analytic CURN gradients (the power-law trace formula; for the dip, r~^T K dd/dtheta);
* HD gradients by long-double 4-point differences.

**Points (revision 2: all kept).**
* 12 random retained draws of the released CURN chain (`chain`).
* 4 uniform prior-box draws (`prior`).
* 4 fixed shelf/corner cases (`corner`):
  * every amplitude 0.05 dex above its lower bound, gamma 0.05;
  * the same with gamma 6.95;
  * every amplitude 0.05 dex below its upper bound, gamma 0.05;
  * the chain point with only the common process at (-17.95, 6.95).
* 4 dip-onset-adjacent cases (`onset`): t0 1e-5 d on either side of the two J1713 TOAs bounding the
  reference interval. Across a TOA lnL jumps by up to 8 nats.

HD uses the same values. Both configurations are evaluated: padded M1 and the bucketed production
configuration (N8 + structured Householder).

**Mixed criterion** (the reviewer's recommendation, frozen in 827d99d before the rerun):
* shape error relative to the chain-point constant: <= 1e-6 nats at `chain`/`onset` points and
  <= max(1e-6, 1e-12 S) at `prior`/`corner` points;
* absolute value vs enterprise: <= 1e-4 / max(1e-4, 1e-12 S);
* S = |lnL| with every parameter-independent constant removed (the arbiter's value);
* gradients <= 1e-8 max(1, |g|) everywhere;
* cross-model differences <= 1e-6 (arbiter) / 1e-4 (enterprise) everywhere.

| check (both configurations; bucketed shown) | result |
|---|---|
| shape vs arbiter, chain / onset points | 1.2e-10 / 1.2e-10 nats |
| shape vs arbiter, prior/corner points | 1.4e-4 nats at S = 2.96e9 (tolerance 3.0e-3; 4.8e-14 relative); all others <= 5.4e-8 |
| absolute vs enterprise, chain / onset points | <= 2.3e-10 nats |
| absolute vs enterprise, prior/corner points | 2.7e-4 at S = 2.96e9 (tolerance 3.0e-3); 1.9e-5 at S = 2.8e4 (enterprise's own float64 error: the arbiter agrees with ours to 5e-8); others <= 5.5e-7 |
| cross-model vs arbiter / enterprise, all 24 points | <= 4.8e-7 / 2.3e-5 nats |
| CURN gradient vs arbiter, all 24 points | <= 1.2e-9 relative |
| HD gradient vs arbiter FD (5 points incl. 2 onset, 8 coordinates incl. dip) | <= 2.4e-9 |

**G5-PTA PASS on every point, in both configurations.** The prior point with the 6-ms dip
(log10_Amp = -2.23, lnL = -2.96e9) failed the absolute tolerance in the first run. It sits about
300 ulp of its magnitude away from the arbiter (ulp(2.96e9) = 4.8e-7). Revision 1 called this
"float64 representability"; that was wrong. Its gradients (3.7e-12) and cross-model difference
(4.8e-7) agree.

**A bug found and fixed by G5-PTA (revision 1).** The dip gradients were exactly zero: the reducers'
custom VJPs returned zero cotangents for the residual-dependent data (c, s_perp). They now return
dq/dc = 2 (c - R_F (r * d)), dq/ds_perp = 1 and c_bar(d) = R_F (d_bar - r * (E d_bar)), using only
the accurately computed E and d. The reviewer confirmed the scope: earlier M1/M2/M3a uses treated
(c, s_perp) as constants, so no earlier result is affected.

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
* The complete kernel (NUTS + t0 MH + prior-independence block MH on (log10_Amp, log10_tau) and the
  joint (t0, log10_tau, log10_Amp) block, as in the pilot configs) also matches the reference
  within 3.5 SE (`test_event_mh_kernel_with_block_moves_matches_quadrature`).
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

**`ptagwb.reweight`**
* Raw estimator only. PSIS k-hat (Zhang-Stephens GPD with the PSIS prior adjustment) is a
  diagnostic; acceptance needs k-hat < 0.5, pooled and per chain.
* ln BF MCSE by overlapping batch means (batch >= 5 tau of the weights), cross-checked by a
  moving-block bootstrap. **The gate uses the larger of the two**, a conservative choice the plan
  did not fix.
* Ratio-estimator MCSE of weighted quantiles, now with an optional domain mask (w 1[D]) on the
  ordered chains.
* Per-chain chi^2 stability: p > 0.01 and no chain carrying more than 50 % of the weight.
* Kish ESS is reported only.
* Tests: synthetic targets, AR(1) coverage 0.84-0.91, heavy and Pareto tails.

**`ptagwb.acceptance`**
* The Sec. 6.1 classifier and the E-6 rule.
* Occupancies, support classes, and the transport table with the UNRESOLVED precedence evaluated
  first. The review's counterexample is UNRESOLVED -> INCONCLUSIVE in 100/100 replicates.
* **Revised D9 (user decision 2026-10-09):**
  * the frozen exclusion validator (`d9_exclusions`);
  * the common-domain indicator under the CURN<->HD mapping (`domain_indicator`);
  * conditional quantiles with the ordered-chain MCSE (`conditional_quantile`). Coverage is tested
    on AR(1) chains with a correlated domain indicator: 0.85-0.99 required, met;
  * eligibility: complete check / headline / model inventories, excluded-draw counts reported,
    non-integer or NaN counts rejected; the unconditional verdict is fixed to INCONCLUSIVE in a
    frozen dataclass.
* **Retired:** p*, p_star_lo90, epsilon_m and the epsilon-qualified BF test. No coverage claim on
  missing mass remains. The mixture envelope (`descriptive_envelope`) and the BF domain correction
  ln(1 - p_CURN) - ln(1 - p_HD) (`bf_domain_correction`) are labelled descriptive.

### 7.3 Frozen acceptance and relevance files (v2, committed before any production run)

`configs/m3b/acceptance_epta.json` (version 2) and `configs/m3b/relevance/epta.json`. The generator
recomputes them, and the strict suite checks that they are unchanged. Appendix A is reproduced
exactly in `reference_summary_unconditional` (e.g. CURN log10_A: q50 -13.997, MCSE 0.0067,
bulk/tail ESS 984 / 1,108).

**Common domain D (revised D9).** U is the union of **24 predeclared exclusions**, identical for CURN
and HD under the gw_crn <-> gw_hd mapping:
* **23 amplitude shelves** [lower, lower + 1 dex] whose retained reference occupancy is zero-visit
  or few-event in either model:
  * DM: J0751, J0900, J1012, J1022, J1024, J1600, J1640, J1713, J1738, J1744, J1751, J1801, J1804,
    J1843, J1857, J1909, J1910, J1918;
  * red noise: J0900, J1012;
  * J1600 chromatic;
  * the dip log10_Amp shelf [-10, -9];
  * the common amplitude.
* **The dip-epoch rest region**: t0 outside [57507.11314794429, 57514.10554332977) (BAT MJD).

Each entry records both models' reference cases (zero-visit, few-event, or for J1804's HD shelf
estimable). **The union holds 38 of 22,493 retained CURN reference draws and 78 HD draws**; they are
kept and their row numbers are recorded.

**Conditional reference quantities and the revised headline rule.** All quantities are recomputed
conditional on D, with the ordered-chain ratio-estimator MCSE. Headline iff 1.645 MCSE_ref <= m/2:

| quantity | q_ref (D) | MCSE_ref (D) | m | headline | max_our_MCSE |
|---|---|---|---|---|---|
| E-1 HD log10 A q05 / q50 / q95 | -14.419 / -13.935 / -13.702 | 0.036 / 0.0041 / 0.0036 | 0.046 / 0.027 / 0.046 | no / yes / yes | 0.0056 / 0.0073 / 0.0134 |
| E-2 HD gamma q05 / q50 / q95 | 1.978 / 2.709 / 3.894 | 0.011 / 0.0115 / 0.070 | 0.131 / 0.079 / 0.131 | yes / yes / **no** | 0.038 / 0.021 / 0.016 |
| E-3 CURN log10 A q05 / q50 / q95 | -14.761 / -13.997 / -13.719 | 0.052 / 0.0067 / 0.0044 | 0.063 / 0.038 / 0.063 | no / yes / yes | 0.0077 / 0.0094 / 0.019 |
| E-4 CURN gamma q05 / q50 / q95 | 2.038 / 2.906 / 4.627 | 0.017 / 0.0174 / 0.105 | 0.167 / 0.100 / 0.167 | yes / yes / no | 0.048 / 0.025 / 0.020 |

So there are **8 headline quantities plus E-6**. E-2 q95 is reported-only; its estimate and
classification will still be reported. E-6 uses ln B_D (common domain); the correction to the
unrestricted BF is reported descriptively. For reference only: the released CURN draws reweighted
to HD give ln BF_raw = 4.224 (MCSE 0.037, k-hat 0.34) against the published ln 60 = 4.094.

**Frozen claim wording.** The text is `ptagwb.acceptance.D9_TEMPLATE`, copied into the file. It
says "CONDITIONALLY EQUIVALENT ON THE COMMON DOMAIN D", is not presented as the original
zero-visit claim, makes no claim about the unconditional posterior, and keeps the unconditional
verdict INCONCLUSIVE.

**Superseded (revision 1).** The model-specific zero-visit lists (39 regions) and the conclusion
"not reachable because of 16 few-event regions" are replaced by the common domain above.

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

### 8.3 Projected EPTA production cost (provisional scenarios; revision 2)

The script is `scripts/m3b_projection.py`. Each run is projected separately, against its own frozen
targets.

**Run A: CURN^gamma**, with HD^gamma and E-6 by reweighting.
* Direct CURN quantities: ESS_req = ESS_ref (MCSE_ref / max_our_MCSE)^2.
* Reweighted HD quantities: ESS_req = ESS_ref,CURN (MCSE_rw / max_our_MCSE)^2. MCSE_rw is the
  weighted-quantile MCSE on the released CURN draws reweighted to HD (every 5th retained draw;
  k-hat 0.34, Kish ESS 909 of 4,499). This prices the reweighting inefficiency.
* E-6: ESS_req from MCSE(ln BF) = 0.037 on those draws.
* The largest requirement is E-1 q50 at 987, below the 1,000 convergence floor. **Run A therefore
  needs ESS 1,000.**

**Run B: CURN gamma = 13/3** (E-5, secondary). It has an illustrative ESS of 1,000 because there is
no released chain to set it.

**Complete kernel cost:**
* 1.194 s per chain-transition, measured (NUTS + 2 t0 MH + refresh);
* plus 33 block-MH value evaluations (31 noise pairs, the common pair, the dip block) at B = 4:
  0.288 s;
* **1.48 s per chain-transition in total**, plus 25 % warmup and 6.9 ms of HD reweighting per draw.

ESS per transition (0.5 / 0.25 / 0.1) is **not measured**; the pilot measures it.

| scenario | run A GPU-h | run B GPU-h | production total | incl. 2 GPU-h pilot | within the D4 production cap (8 GPU-h)? |
|---|---|---|---|---|---|
| optimistic (0.5) | 1.03 | 1.03 | 2.07 | 4.07 | yes |
| central (0.25) | 2.07 | 2.07 | 4.13 | 6.13 | yes |
| pessimistic (0.1) | 5.17 | 5.17 | 10.33 | 12.33 | **no**: run A first (priority), run B within what remains |

These are provisional scenarios, not an allocation. Production is released only if the pilot (the
complete kernel) shows adequate mixing and projects these targets within the cap (D4, second set).

**Superseded (revision 1).** The 41 / 81 / 199 GPU-h projection charged E-2 q95's free-gamma tail
requirement (ESS 23,629) to both runs and omitted the block-MH cost.

### 8.4 D4: mechanical cap and run driver (user decision 2026-10-09, second set)

`ptagwb.budget` and `scripts/m3b_run_epta.py`:
* **Ledger.** An append-only GPU-time ledger. Caps: total 12, pilot 2, production 8, contingency 2
  GPU-h. A run must fit its `max_gpu_hours` in what is left of its phase and of the total. A
  run_id is never repeated or resumed. A heartbeat keeps a crashed run's time counted.
* **Deadline.** max_gpu_hours minus two watchdog grace periods. Before every chunk of 10
  transitions the previous chunk's time must fit. A watchdog thread writes the ledger and
  terminates the process if a chunk overruns.
* **Fixed configuration.** The driver refuses an uncommitted config or any work-tree change in
  src/scripts/tests/configs, unknown config keys, inits from the reference chain, stale or failed
  gate evidence (bound hashes), and a GPU used by another process.
* **Stop rules.** The deadline, the transition count, or a non-finite lnL.
* **Pilot configs** (committed, **not started**): `configs/m3b/run_configs/epta_pilot_curn_freegamma.json`
  (1.5 GPU-h) and `epta_pilot_curn_g433.json` (0.5 GPU-h), together exactly the 2 GPU-h pilot phase.
  Both use the complete kernel with prior-independence block proposals. Production configs
  (frozen metric, histogram proposals fitted from the pilot) are written only after the pilot and
  its review; a pilot that suggests a change to a committed config ends the work.

The D3 benchmark (0.21 GPU-h, revision 1) is not part of the ledger. Its evidence preconditions
were met only under the then-unbound check and the since-withdrawn G5 restriction. Its timings
remain valid as timing evidence (review).

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

Needed before the pilot:

1. **Independent review of this revision.** It covers the frozen acceptance file v2, the revised D9
   wording, the run driver and its cap enforcement, and the G5 mixed criterion. The coordinator's
   confirmation must follow. The pilot does not start before both.
2. **Proposed deferral of T2 and the conditional-occupancy diagnostics** to M3b-EPTA, before the
   pilot (plan, top). This needs the reviewer's acceptance; otherwise both are done in M3b-0E first.
3. **The chain-reproducing runtime** (`epta-dr2-chain-runtime-v1`) is part of the D1 profile.
   The reviewer supported the manifest revision with the provenance wording now used.

Needed after the pilot (before production):

4. Production configs: a frozen dense metric and step size from the pilot, histogram block
   proposals fitted from pilot draws only (frozen with sha256), and inits from pilot draws. These
   need a review.
5. The pilot's measured ESS per transition replaces the bracket in Sec. 8.3. Production is released
   only if the targets project within the remaining cap.

Recorded, no action:

6. E-2 / paper inconsistency; dip index (paper 4/2, code 1); N_c = 9 (fingerprint-discriminated).
   The code that produced the chains is followed.
7. Reference reweighting gives ln BF_raw = 4.224 +- 0.037 against the published ln 60 = 4.094. This
   is context for E-6 and not a classification.
8. E-C0 belongs to M3b-EPTA (review). E8 stays open (D1).

Engineering notes:

9. Two multi-threaded XLA:CPU processes running at once oversubscribed the 32 cores badly; run the
   CPU gates one at a time.
10. `libstempo.telescope()` fails under numpy 2 in the plain tempo2 env. It is not consumed.
11. The forks' 2022/23 `setup.py` files no longer build. The env puts the pinned checkouts on the
    path through a `.pth` file and pins `setuptools<70`.
12. `.gitignore` ignores every `runs/` directory, so the run configs live in
    `configs/m3b/run_configs/`.
