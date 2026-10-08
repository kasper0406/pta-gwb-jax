# HD free-spectrum (30 modes) re-run: pilot before committing to hd_fs30_v2

Question: would the planned re-run `configs/m2/hd_fs30_v2.json` (8 chains, 500 warmup + 750 draws,
independent overdispersed init z ~ U(-4, 4), diagonal windowed adaptation) pass the free-spectrum
acceptance gate (docs/M2_RESULTS.md Sec. 10), and at what cost? Budget: ~2 GPU-hours (RTX 5090),
abort above 2.5 h. **Used: 1.93 GPU-hours** (five pilots, 10:23-12:21 UTC on 2026-10-08, including setup).

**Status after review round 1 (2026-10-08, afternoon).** Round 1 (Secs. 1-6, corrected below) stopped v2.
Round 2 (Secs. 7-11) fixed the acceptance gate, built and tested an exact hybrid kernel (NUTS + block
Metropolis-Hastings jumps), and compared it with plain NUTS on CURN^free and HD^free.
Round 2 used **~2.6 GPU-hours** (budget 3.5). Round 3 (Secs. 13-15, ~3.0 GPU-h) added proposal validation, the exact
conditional-grid move, a controlled analysis of what drives f_3, and an enhanced HD pilot. Current recommendation:
Sec. 15.

**Round-1 findings that stand.** Do not launch v2 as planned:
1. The v2-style warmup was expensive in every attempt. From warmup iteration ~8 on, at least one of the 8 lockstep
   chains reached tree depth 10 (1023 leapfrog steps, 164 s per 8-chain iteration) in most iterations (pilots A-D,
   all aborted under their pre-set rules). The pilots are confounded, though (Sec. 3): the step size started at
   numpyro's default 1.0 without a heuristic search, and A/B stopped before their first mass-matrix update. The
   v2 warmup cost is therefore **not established**. The measured cost suggests a conditional
   scenario of ~11-18 h warmup + 9.4 h sampling (~20-28 GPU-h) if v2 behaved like pilots A-E.
2. With plain NUTS, the bins whose posterior has a small second region are crossed rarely (M2 run and pilot E).
   The 0.7% low-power region at f_3 was never visited in 1,400 post-warmup draws. Whether a v2-length plain-NUTS
   run would pass the gate is **unknown**: the per-kernel spread of the measured rates is large (Sec. 5), so
   I do not give a passing budget.
3. The geometry is consistent with a scale mismatch and slow excursions. For the detected bins (f_1, f_3, f_4, f_8),
   each has a narrow likelihood peak (sd(z) 0.035-0.074) next to a broad prior-dominated shelf
   (sd(z) 1.0-1.2). A classic latent-coefficient funnel was not demonstrated, but not excluded either (Sec. 4).

## 1. Code and configs (all committed before the runs they affect; the only dirty file at launch is noted in Sec. 9)

| commit | content |
|---|---|
| 396f2bc | `RunConfig.likelihood_impl`: `"production"` (default, unchanged) or `"fast"` (opt-in `FastPTALikelihood(reduce="hh", tri_inv="levels")`, exact within the docs/PERF.md budgets); `sampling.make_likelihood` used by `scripts/m2_run.py`; run metadata now also records the likelihood class / reducer / triangular inverse. Tests: default + validation, fast vs production potential and gradient (HD free spectrum, mixed precision; HD power law, float64) on a synthetic PTA. Config pilot A. |
| e9532d3, 3c83d31, 7b56c29 | configs pilots B, C, D (each written after the previous abort, with its own abort rule) |
| db6e9be | `RunConfig.step_size` / `adapt_step_size` (defaults = numpyro's 1.0 / True, so no existing config changes behaviour) + test; config pilot E |
| cd720dd | `scripts/fs_pilot_diag.py` (crossings, IRN indicators, projection), round-1 document |
| c0c1182 | round 2: fail-closed gate (`diagnostics.py`, `m2_freespec_diag.py`, tests); `ptagwb.hybrid` (HybridNUTS, block proposals) + `tests/test_hybrid.py`; `RunConfig.jumps` / `jump_sweeps` / `init_rho_low_frac`; frozen proposals `configs/m2/proposals/{curn,hd}_fs30_v1.json`; `bench/bench_jumps.py`; configs of the CURN A/B and the HD hybrid pilot |
| 0813235 | `scripts/fs_compare_kernels.py`, `scripts/fs_schur_prototype.py`, slow marker for the GP-toy test (HD hybrid pilot launched at this SHA) |
| 7360f80 | round-2 document |
| e5f8311 | round 3: proposal validation, `ptagwb.conditional` + tests, conditional-grid moves (one-bin and joint), `scripts/fs_conditional_analysis.py`, frozen relevance declaration + `--relevance-file`, `RunConfig.max_sampling_seconds`, `bench/bench_grid.py`, proposals v2, grid-pilot config |
| 67db032 | fixed-step grid-pilot config (run at this SHA) |
| this commit | round-3 document |

Run metadata check: every pilot's `meta.json` has `backend: gpu`, `devices: [cuda:0]`,
`xla_flags: --xla_cpu_experimental_ynn_fusion_type=`, `likelihood.class: FastPTALikelihood`,
`dirty_files: []` (SHAs 396f2bc, e9532d3, 3c83d31, 7b56c29, db6e9be).

## 2. Gradient cost (measured before sizing)

Value+gradient of the HD^free(30) potential (mixed-precision backward pass, as in M2), `vmap` over
B chains, median of 10 calls after JIT, GPU, identical at posterior and U(-4, 4) points:

| B | production | fast | per chain (fast) |
|---|---|---|---|
| 1 | 24.5 ms | 20.5 ms | 20.5 ms |
| 4 | 105.7 ms | 100.8 ms | 25.2 ms |
| 8 | 166.2 ms | **159.5 ms** | **19.9 ms** |

The fast likelihood gains only 4% at B = 8 for the 30-mode free spectrum. Its 4020-dim core
Cholesky and triangular inverse dominate (~22 GFLOP per chain-gradient, ~55% of the measured FP64 DGEMM peak), not
the per-pulsar stage that `FastPTALikelihood` accelerates. Inside NUTS, one 8-chain leapfrog step cost
165 ms (pilot E), i.e. 41 s per iteration at depth 8, 164 s at depth 10. The suggested
300 warmup + 250 draws would take ~10-14 h at the warmup and sampling costs measured below (estimated ~6-8 h before
launch); pilot A was sized to 100 + 75.

## 3. Pilots

| pilot | design | outcome | GPU time |
|---|---|---|---|
| A `hd_fs30_v2_pilot` | **exact v2 recipe** (8 chains, U(-4,4) seed 31, unit initial metric, diagonal windowed adaptation), 100 warmup + 75 draws | aborted at warmup iteration 16 (rule: warmup projection > 1.75 h). Per-iteration seconds: 18 (iteration 1, incl. compile), 0-5 (2-6), then **82, 81, 41, 164, 164, 163, 164, 164, 82, 163** (7-16): depth 9-10 almost every iteration; projected warmup ~3.5 h | 27 min |
| B `..._pilotB` | A + initial diagonal metric = diagonal of the M2 run hd_fs30's z-covariance, 75 warmup | aborted at iteration 11: 82, 163, 164, 164 s from iteration 8 | 12 min |
| C `..._pilotC` | 8 chains started at hd_fs30 draws, fixed diagonal hd_fs30 metric, 30 step-size-only warmup | aborted at 13/30: 41, 41, 40, 21, 82, 82, 163, 82, 41, 164 s | 16.5 min |
| D `..._pilotD` | C with the dense hd_fs30 metric | aborted at 14/30 (rule: warmup > 20 min): 41, 40, 41, 21, 40, 164, 164, 82, 164, 81, 164 s | 20.5 min |
| E `..._pilotE` | 8 chains at hd_fs30 draws, fixed dense hd_fs30 metric, **fixed step size 0.025** (M2 adapted 0.026-0.030), 1 non-adapting warmup iteration, 100 draws planned | stopped at the 50-draw checkpoint (rule: last checkpoint before 12:30). **0 divergences, accept 0.945, mean 215 steps per draw, depth {7: 132, 8: 263, 9: 5}, lockstep max 273 steps (efficiency 0.79), 45.1 s per iteration** | 39.5 min |

Pilot E is the only one with post-warmup draws (8 x 50). It is NOT independent of hd_fs30 (init and
metric), so its between-chain differences are partly inherited. It measures transition rates and ESS
per gradient in the regime v2 would reach after a successful warmup, with a dense rather than
v2's diagonal metric. The M2 run hd_fs30 (4 x 250, dense CURN metric, 100 step-size warmup) is analysed with the
same script for comparison.

**Why the warmup is so expensive.** (i) The U(-4, 4) starting points lie 0.8-2.3 x 10^4 nats above the typical set.
The metric-whitened gradient norm there is 1,060-1,770, against 24-62 at posterior draws, and is dominated by
high-frequency bins started at far too much power (log10 rho up to -1.26). Dual averaging shrinks the step
size during this relaxation. (ii) Even near the posterior, dual averaging (which starts at
log(10 eps0) and oscillates) repeatedly passes through small step sizes. In a vectorised run, each iteration costs the
longest of the 8 trees, so a single chain in a small-step phase costs the whole batch 1023 steps.
For comparison, the M2 CURN^free run (4 chains, U(-2, 2), 500 diagonal-adaptation warmup) averaged 475 lockstep steps
per warmup iteration against 267 when sampling; its warmup is the closest analogue of v2's.

**Confounders (review round 1).**
* Pilots A-D started from numpyro's default step size 1.0. The installed numpyro has
  `find_heuristic_step_size=False` and the driver did not override it, so dual averaging began far from the
  workable ~0.025. C/D therefore do not test adaptation that starts near a workable step size.
* A and B stopped before their first mass-matrix update (numpyro schedule for 100 warmup iterations: first
  update after iteration 89). A 100-iteration schedule also differs from v2's 500-iteration one, so A is not the v2
  adaptation recipe merely shortened.
* E changed metric, initialisation, adaptation and step size at once. Its acceptance (0.945, against a target
  of 0.8) shows the step size was smaller than a tuned one, so its cost per draw is not that of an adapted v2.
* 50-draw chains cannot establish asymptotic ESS rates, especially with inherited initialisation and one long
  excursion.

The CURN A/B runs of Sec. 9 started step-size adaptation at a measured workable value (0.02). There, 150 adaptation
iterations cost 1.6-2.0 s each (295 / 241 s), about 1.3-1.6x a sampling iteration, with no saturation episodes.
This suggests the HD warmup blow-up in A-D owes much to the poor step-size start. That has not been tested on HD.

## 4. Diagnostics of the post-warmup draws (pilot E, 8 x 50; M2 hd_fs30, 4 x 250)

**Gate (round-1 version)** (`scripts/m2_freespec_diag.py --run hd_fs30_v2_pilotE`, exit 1): convergence FAIL (163 of
164 parameters, 30 of 30 bins), reproduction agreement PASS (0 bins disagreeing; at these SEs that establishes little).
The one passing parameter, J2322+2057 gamma (bulk / tail ESS 805 / 433), shows that 400 draws *can* give ESS > 400
(negative autocorrelation), contrary to the round-1 remark. Max R-hat 1.173 (f_26), worst bins f_26,
f_1 (1.152), f_25 (1.111), f_4 (1.104). M2 hd_fs30: convergence FAIL (59 parameters), agreement PASS.
**Under the fail-closed gate (Sec. 7)** both runs fail convergence with "no exploration evidence" at f_3
(both regions; E also at f_29), and agreement is INCONCLUSIVE (our occupancy SEs > 0.01).

**Occupancy and transitions between regions.** Transition = completed move between log10 rho < -10
and > -8 (the chain must cross the -10..-8 gap; jitter at -9 does not count). Bins with both regions
carrying >= 0.5% of the released core's mass must show transitions in a converged run. Excerpt (all 30 bins
are in `outputs/m2/fs_pilot_hd_fs30_v2_pilotE.txt`):

| bin | released P(<-9) | released mass <-10 / >-8 | E occupancy per chain (%) | E transitions per chain | M2 occupancy per chain (%) | M2 transitions per chain | R-hat E / M2 |
|---|---|---|---|---|---|---|---|
| f_1 | 17.3% | 14.7 / 80.1% | 2 58 20 52 18 20 14 14 | 2 10 8 7 8 6 4 6 | 23 3 20 20 | 13 1 8 10 | 1.152 / 1.082 |
| f_3 | 0.9% | 0.7 / 99.0% | 0 0 0 0 0 0 0 0 | **0 0 0 0 0 0 0 0** | 0 0 0 0 | **0 0 0 0** | 1.020 / 1.003 |
| f_4 | 10.2% | 8.5 / 87.6% | 8 0 8 0 2 0 28 0 | 4 0 2 0 1 0 5 0 | 3 14 2 6 | 4 14 2 8 | 1.104 / 1.021 |
| f_5 | 49.0% | 41.3 / 42.0% | 46 64 54 58 34 70 36 70 | 9 14 8 12 10 8 9 10 | 64 29 48 58 | 39 21 48 23 | 1.046 / 1.044 |
| f_8 | 6.0% | 4.9 / 91.4% | 0 14 8 2 0 0 0 4 | 0 4 2 0 0 0 0 2 | 14 0 0 0 | 1 0 0 0 | 1.046 / 1.033 |
| f_13 | 89.4% | 75.5 / 0.6% | 92 80 90 84 92 84 94 88 | 0 0 0 0 0 0 0 2 | 88 88 90 89 | 4 2 0 2 | 1.011 / 1.006 |
| f_14 | 88.8% | 75.4 / 0.8% | 96 90 90 86 90 92 82 86 | 0 2 4 0 0 2 2 0 | 87 90 88 91 | 4 2 4 2 | 1.009 / 1.005 |
| f_28 | 88.8% | 75.2 / 0.7% | 90 86 86 92 86 90 86 92 | 0 0 0 0 0 2 0 2 | 84 86 90 93 | 2 8 6 4 | 1.022 / 1.008 |
| f_29 | 88.3% | 74.6 / 0.6% | 90 96 88 88 92 80 90 80 | **0 0 0 0 0 0 0 0** | 86 88 88 92 | 4 0 2 6 | 1.022 / 1.000 |
| f_26 | 50.0% | 42.4 / 29.4% | 38 70 40 54 22 68 62 40 | 9 2 6 10 6 5 10 6 | 58 43 42 60 | 29 29 25 26 | 1.173 / 1.013 |

* **Zero transitions in all chains** (pilot E): f_3 and f_29 among the bins with two relevant regions (f_2 has
  a single region). f_3's low-power region (0.7% of the released mass) has never been visited by any
  HD^free chain of ours (M2: 0 / 1000 draws, E: 0 / 400). Bins where only a minority of chains transition: f_4 (4 of 8),
  f_8 (3 of 8), f_13 (1 of 8), f_14 (4), f_28 (2). Every chain transitions at f_1, f_5-f_7, f_10-f_12, f_16-f_20 and f_24-f_26,
  f_30 (typically 2-21 times per 50 draws); 5-7 of 8 chains at f_9, f_15, f_21-f_23.
* **f_8 vs the released 5.96%:** pilot E pooled 3.5% (per chain 0/14/8/2/0/0/0/4%). M2 hd_fs30 also pooled 3.5% (14/0/0/0%).
  Three of E's chains reached the low region within 50 draws (transition rate 8 / 400 draws vs
  1 / 1000 in M2). Pooled over both runs, 9 transitions in 1,400 draws.
* Gate weakness: a bin whose minority region is never visited (f_3) had an "unavailable" indicator
  (warning only) and fine parameter ESS. An all-zero 8 x 750 indicator gives agreement |z| ~ 2.4 < 3.5 (the
  reviewer's computation; round 1 said 2.8), so **the round-1 gate could pass a run that never visits f_3's
  low-power region**. Fixed in Sec. 7.
* Zero crossings alone are weak evidence for the truncated tails: with the released P(f_29 > -8) = 0.64%,
  400 *independent* draws contain none with probability 7.7%.

**Why the transitions are rare (released core, sampler coordinate z): a plausible description, not a measured barrier.**
For the detected bins, the conditional sd(z) is 0.074 in the peak (> -8) vs 1.08 on the shelf (< -10) at f_1,
0.035 vs 1.01 at f_3, 0.046 vs 1.17 at f_4 and 0.044 vs 1.08 at f_8. One global scale per coordinate cannot serve both.
Round 1 also quoted "potential rises" of 3.7-7.8 nats. Those are differences of *marginal* log density
(60-bin histograms in z: mean shelf density vs the peak bin), not joint Hamiltonian barriers, and they are not
reproduced by the committed script, so treat them as illustrative only. For f_13 / f_29 the > -8 population is a
truncated **upper tail** (released counts decrease through it), not a demonstrated separate mode.

**IRN parameters.** In pilot E, 92 IRN parameters have R-hat >= 1.01 (M2: 26; short chains). Worst:
J0610-2100 log10_A R-hat 1.123, tail ESS 19; J2234+0611 1.100; J0437-4715 1.092; J1853+1303 1.081;
J0645+5158 1.057; J1713+0747 1.055. Funnel indicators (pulsar-wise, both runs):
* conditional width of z_gamma in the lowest vs highest log10_A tercile: median ratio 1.12, worst 2.2-2.6
  (J1853+1303, J1713+0747, J0613-0200): mildly funnel-shaped;
* per-draw mobility of z_gamma in units of its conditional sd: 0.9-2.1 in both terciles. The
  Spearman correlation of the step-size-normalised move |dz_gamma| / (eps sqrt(M^-1)) with log10_A
  has median -0.02 (worst -0.25), and tree length vs log10_A has |rho| <= 0.14. These are weak diagnostics,
  though: NUTS varies its trajectory length, the terciles average over broad amplitude ranges, and a rare neck
  that chains seldom visit cannot be excluded. So **a classic funnel limit was not demonstrated**, which is not the
  same as being excluded;
* the worst amplitudes behave like the bins: long excursions. Example: J0610-2100's released
  posterior has 0.6% below log10_A = -14. Pilot E chain 0 went to -19.5 and spent 56% of its 50 draws below -14,
  while the other 7 chains stayed in [-13.8, -11.8] (M2: chain 0 8%, others 0%). The mobility of z_A in its low-A
  tercile is 0.47 sd per draw. IRN amplitudes also correlate moderately with low-frequency bins (Spearman up to
  -0.34 with f_1), so their slow movement partly inherits the bins' metastability.

## 5. Projection (corrected: a conditional scenario, no passing budget)

* **Cost scenario** for v2 as planned (500 + 750), *if* v2 behaved like pilots A-E: warmup ~11-18 h
  (80-127 s per iteration, pilots A, C, D) + sampling 9.4 h (45 s per iteration, pilot E) = ~20-28 GPU-h. The
  confounders of Sec. 3 (step-size start at 1.0, E's 0.945 acceptance) make this an upper-side scenario,
  not a prediction.
* **Passing budget: unknown.** The two plain-NUTS kernels measured give very different rates. At f_8 the two-state
  estimate 800 p(1-p)/f for an indicator ESS of 400 is ~487 draws per chain from pilot E's transition rate
  (8 / 400 draws) but ~9,737 from the M2 run's (1 / 1,000; its single crossing was an exit with no later entry).
  This spread dominates any pooled number, so the round-1 "1,500-2,700 draws per chain / 30-50 GPU-h" and
  "f_3 >= 650 draws" figures are withdrawn. The two-state formula also assumes stationarity and a Poisson crossing
  process, which paired entries / exits from inherited starts do not satisfy.

## 6. Round-1 recommendation (superseded by Sec. 15; kept with corrections)

**Corrections (review round 1).**
* *Shelf warp:* the benefit has to be evaluated in the **final NUTS coordinate**
  u = logit((g(x) - g(lo)) / (g(hi) - g(lo))), not by subtracting log(1/c) in the old coordinate. In the near-linear shelf
  the logistic map largely undoes the compression. On released draws (c = 0.05, delta = 0.2, x_floor = -9; verified),
  the shelf/peak sd ratio goes from 14.5 to 6.4 (f_1), 28.6 to 11.4 (f_3), 25.2 to 9.2 (f_4) and 24.2 to 8.3 (f_8), not
  to ~1. The shelf sd itself barely changes (f_3: 1.011 -> 0.904). The warp is a valid exact bijection, but its
  benefit is not demonstrated; it remains a controlled competitor, not the selected remedy.
* *Pivot amplitude:* with the gamma-dependent bounds proposed, a_p = l - k gamma + (h - l) sigmoid(u) implies
  log10_A = l + (h - l) sigmoid(u). The pivot **cancels exactly** and reproduces the original coordinate (an algebraic identity; the reviewer also checked
  it numerically on saved draws, to ~1e-15). It does not remove the A-gamma correlation in NUTS coordinates. (Also, a_p is a half-log-PSD-like
  quantity, not "log10 PSD".)
* The cost numbers in the original text below are superseded by Sec. 5.

Original round-1 text:

**Reparameterisation sketch (exact: a bijection plus its log-Jacobian; it changes efficiency only).**
The Fourier coefficients are marginalised, so a non-centred form for them is unavailable. The
structure to remove is one coordinate covering a narrow likelihood peak and a 25x wider prior shelf.

1. *Free-spectrum bins: shelf compression.* For bin k with x = log10 rho_k, sample w = g_k(x) with
   g_k'(x) = c_k + (1 - c_k) sigmoid((x - x_k^floor) / delta). Identity above the bin's noise floor
   x_k^floor; compressed by c_k ~ 0.03-0.1 below it (c_k ~ peak scale / shelf scale), then the usual
   logistic box map on [g(lo), g(hi)]. log p(w) gains -log g'(x). In w the shelf's width matches the peak's, and the
   peak-to-shelf rise drops by log(1/c_k) ~ 2.3-3.5 nats (f_8: 5.4 -> ~2-3; f_3: 7.8 -> ~4.5-5.5; f_1:
   3.7 -> ~0.5-1.5). x_k^floor and c_k come from the target itself: a 1-D likelihood profile per bin at one
   reference point (30 x ~20 evaluations), or the bin's Fisher information. No other run is needed, so the
   v2 independence argument survives. Wrong choices cost efficiency, never correctness.
2. *IRN: pivot amplitude + the same shelf compression.* Replace (log10_A, gamma) by
   (a_p, gamma) with a_p = log10 PSD at a pulsar-specific pivot f_p (the best-measured IRN frequency):
   a_p = log10_A - (gamma / 2) log10(f_p / f_yr). This removes the linear A-gamma banana. The box
   log10_A in [-20, -11] becomes gamma-dependent bounds on a_p (logistic map conditional on gamma; triangular Jacobian).
   Then compress a_p below the pulsar's detection floor as in 1 (J0610-2100, J0437-4715, J2234+0611,
   J1853+1303, J0645+5158, J1713+0747 first). gamma needs no non-centring: its conditional width changes by only
   1.1-2.6x and it mixes locally.
3. *Warmup.* (i) Before NUTS, relax each U(-4, 4) point with a few hundred deterministic optimiser steps,
   stopping while still ~D nats above the mode, so that the init stays overdispersed but does not start
   10^4 nats out. (ii) Use one shared step size for the 8 vectorised chains during adaptation. The
   lockstep max of 8 independently oscillating dual-averaging step sizes is what produces the depth-10
   iterations. (iii) Use a dense metric after the first window.
4. *Validation, cheapest first:* unit tests (Jacobian vs autodiff, prior recovered under a flat likelihood,
   NUTS on a toy peak + shelf target vs its exact marginal); then a CURN^free A/B pilot (same bimodal bins, ~8x cheaper
   gradient), measuring transitions per gradient at f_1, f_3, f_4, f_8 vs the current coordinates; then a 2-h HD pilot
   like E. Success criterion for launching a full HD run: worst-parameter ESS per draw >= 0.2 and f_3 low-region visits in
   most chains, which would bring the gate within reach at ~8 x 250-500 draws (~3-6 h sampling).

If the warp does not raise transition rates enough, the next step is a mode-jumping move between the two
regions per bin (an independence Metropolis proposal from the shelf's prior to the peak's Laplace
approximation, interleaved with NUTS), or tempering, at higher implementation cost.

## 7. The gate now fails closed (`ptagwb.diagnostics.freespec_gate`)

Changes (review round 1, item 6; tests in `tests/test_freespec_gate.py`, 238 passing):
* **Regions:** per bin, "low" (log10 rho < -10) and "high" (> -8), with a hysteresis state per chain, in addition to
  the existing < -9 occupancy. Entries, exits and sojourns are counted from the state (`region_states`,
  `region_events`).
* **Relevance is predeclared:** a region is relevant if the released core puts >= 0.5% of its mass there
  (`--relevance-ref hd_fs30`, also in convergence-only mode), or it is given explicitly. Round 3 froze the declaration in
  `configs/m2/relevance/hd_fs30_released_v1.json` (`scripts/fs_freeze_relevance.py`). It records the reference
  identity: path, sha256 8ca24e87..., stored la_forge burn-in 122,500 of 490,000 samples, 367,500 retained. The gate
  reads it with `--relevance-file`. The pooled-draw declaration used for the CURN A/B (Sec. 9) was exploratory, not
  predeclared. With no declaration and no
  reference, convergence is INCONCLUSIVE, never PASS.
* **Every relevant region must show exploration evidence**, else convergence FAILs and names the missing evidence:
  * >= 10 entries and >= 10 exits pooled, in >= 2 chains;
  * the longest single sojourn holds <= 50% of the region's draws;
  * the region indicator 1[x in R] has R-hat < 1.01 and ESS >= 400;
  * occupancy MCSE <= 0.01. This is tied to the objective of reproducing the Fig. 1a probability masses to about
    +-2 percentage points at 95%. The same MCSE bound applies to every bin's < -9 occupancy.

  A region that is the bin's only relevant one needs >= 2 visiting chains and the MCSE bound.
* **Agreement:** z keeps the reference SE (sqrt(SE_ours^2 + SE_ref^2)), and the verdicts are FAIL / INCONCLUSIVE / PASS /
  UNAVAILABLE. Agreement is INCONCLUSIVE when our SE of any compared occupancy (< -9 and the relevant regions) exceeds
  0.01. The released core's own SE reaches 0.010 (f_1 high region): it limits how precisely agreement can be
  established, so it is reported but does not block. Exit 0 only for convergence PASS and agreement PASS. The thresholds are operational safeguards, not convergence guarantees. MCSE 0.01 targets absolute
  (percentage-point) accuracy and is weak for the *relative* accuracy of a 0.7% event. Agreement PASS means
  compatibility under this rule, not demonstrated equivalence within +-2 percentage points.
* **Results:**
  * M2 hd_fs30 now fails at f_3 with "no exploration evidence" (low and high).
  * Pilot E fails at f_3 and f_29.
  * Agreement is INCONCLUSIVE for both runs.

## 8. Exact hybrid kernel: NUTS + frozen block-MH independence jumps (`ptagwb.hybrid`)

**Design.** Each iteration is one numpyro NUTS transition (unchanged, warmup adaptation included), followed by one sweep
of block Metropolis-Hastings independence moves. The blocks are each of the 30 free-spectrum bins (1-D) and the
(log10_A, gamma) pairs of 6 problem pulsars (2-D): J0610-2100, J2234+0611, J0437-4715, J1853+1303, J0645+5158
and J1713+0747, chosen from pilot E / M2 R-hat, not from the released chain.
* **Proposal q_b:** 20% uniform on the block's prior box + 80% equal-mass histogram of pilot draws. 1-D blocks
  use 40 quantile bins; 2-D blocks use 12 quantile bins in log10_A, each split into 8 conditional quantile
  bins in gamma, uniform within cells. The density is continuous and exact with full support. Proposals are
  fitted by `scripts/fs_fit_proposals.py` from our own draws only:
  * CURN: from curn_fs30;
  * HD: from hd_fs30 + pilot E.

  The fit was frozen and committed (`configs/m2/proposals/*.json`, c0c1182) before any run that uses it.
* **Acceptance:** min(1, pi(x') q(x_b) / (pi(x) q(x_b'))) in **physical coordinates**, where pi is proportional to L inside
  the box. This equals the MH ratio in the sampler's logistic coordinate because the Jacobians of target and proposal
  cancel.
* **Cache refresh:** after the sweep, NUTS's cached potential energy and gradient are recomputed at the final point.
* **Invariance:** each move satisfies detailed balance and NUTS preserves pi, so the composition is pi-invariant.
* **Config:** `RunConfig.jumps` (proposal file) and `jump_sweeps`. Per-block accepted moves are stored as
  `jump_accept` in `samples.npz`.

**Tests** (`tests/test_hybrid.py`, 6 passing):
* proposal density normalised, and the sampler consistent with it (KS);
* invariance on a 1-D peak (95%) + shelf (5%) target: 40,000 exact draws, then 1 or 5 sweeps with a deliberately
  biased proposal still pass KS against the exact CDF with the shelf mass kept. The moves are real: acceptance
  0.05-0.95, and regions are crossed;
* **power check:** a naive acceptance without the q ratio fails the same KS test (p < 1e-6);
* invariance on a 2-D (log10_A, gamma)-like target (correlated peak + low-amplitude shelf; marginal KS, shelf mass, and
  the in-peak correlation preserved);
* proposal JSON round-trip and schema check;
* full NUTS + jumps through `run_nuts` (4 vectorised chains) on a 2-bin Gaussian-process toy with a low-power shelf
  (y ~ N(0, I + sum_k 10^(2 rho_k) F_k F_k^T)): quantiles and the shelf occupancy match a 600 x 600 grid
  within 5 MCSE (marked slow).

**Forward-only cost** (GPU, B = 8, `bench/bench_jumps.py`, results in `bench/results/jumps_*_B8.json`):

| model | value+grad | value only | sweep (36 proposals) + refresh | acceptance at posterior draws |
|---|---|---|---|---|
| CURN^free | 2.75 ms | 2.34 ms | 89 ms | 0.71 |
| HD^free | 159 ms | **135 ms** | **5.0 s** | 0.73 |

The value-only HD evaluation is not much cheaper than value+gradient (the 4020-dim Cholesky dominates). One sweep
adds ~11% to an iteration at pilot E's 45 s.

## 9. CURN^free screening: plain NUTS vs hybrid (matched arms)

`configs/m2/curn_fs30_ab_{nuts,hybrid}.json` (c0c1182). Both arms: 8 vectorised chains, the same seed-41 starts
at curn_fs30 draws with every bin independently moved into the low region [-15, -10] with probability 0.3
(deliberately diverse region starts), fixed dense curn_fs30 metric, and step-size adaptation (150 iterations)
starting at the measured workable 0.02. Then 1500 draws. GPU time 67 min for both arms, including setup.
The two arms differ only in the jump sweep. Arm B's metadata lists one dirty test file (`tests/test_hybrid.py`,
test sizes only; no run code).

| | curn_fs30_ab_nuts | curn_fs30_ab_hybrid |
|---|---|---|
| chains | 8 | 8 |
| draws per chain | 1500 | 1500 |
| sampling wall [h] | 0.45 | 0.34 |
| mean leapfrog steps | 228 | 251 |
| lockstep steps | 276 | 255 |
| accept | 0.951 | 0.960 |
| divergences | 0 | 0 |
| max R-hat | 1.470 | 1.007 |
| params R-hat >= 1.01 | 19 | 0 |
| worst parameter (min bulk/tail ESS) | gw_log10_rho_1 | B1855+09_red_noise_log10_A |
| its ESS | 15 | 793 |
| worst ESS per sampling hour | 34 | 2327 |
| median param min ESS | 6173 | 7461 |

Selected bins (all 30: `outputs/m2/fs_compare_curn_fs30_ab_nuts_curn_fs30_ab_hybrid.json`). Excursions = entries +
exits of the minority region, pooled, with the number of chains that have any in parentheses:

| bin | NUTS: minority occupancy, excursions (chains), longest sojourn, MCSE(< -9), R-hat | hybrid: same |
|---|---|---|
| f_1 | low 10.3%, 243 (6), 0.05, 0.032, 1.203 | low 11.6%, 546 (8), 0.03, 0.010, 1.006 |
| f_2 | low 20.0%, 2 (2), 0.53, 0.148, 1.470 | low 0.2%, 8 (4), 0.60, 0.002, 1.001 |
| f_3 | low 4.5%, 2 (2), 0.64, 0.043, 1.050 | low 0.9%, 26 (7), 0.17, 0.003, 1.002 |
| f_4 | low 24.8%, 980 (8), 0.01, 0.094, 1.171 | low 10.3%, 1019 (8), 0.01, 0.008, 1.002 |
| f_8 | low 4.2%, 67 (8), 0.15, 0.013, 1.007 | low 9.3%, 602 (8), 0.02, 0.007, 1.001 |
| f_13 | high 0.7%, 178 (8), 0.03, 0.003, 1.001 | high 0.7%, 164 (8), 0.04, 0.003, 1.000 |
| f_17 | low 41.9%, 1798 (8), 0.01, 0.023, 1.012 | low 39.9%, 3767 (8), 0.01, 0.006, 1.000 |
| f_29 | high 0.4%, 88 (8), 0.04, 0.003, 1.000 | high 0.4%, 84 (8), 0.07, 0.003, 1.000 |

* **Plain NUTS stays trapped where it started.** At f_2, the chains started on the low shelf spent 69% and 89% of their
  1500 post-warmup draws there; the other six never went (R-hat 1.47, ESS 15). At f_3, two chains spent 23% / 13% low.
  These are the same metastable excursions as in the HD runs, now with matched, controlled starts.
* **The hybrid removes most of them** (it still fails 4 region criteria, below). Max R-hat 1.007 over all 164
  parameters (NUTS: 1.470, 19 parameters >= 1.01). Worst-parameter min(bulk, tail) ESS is 793 against 15, i.e. 2,327 vs
  34 per recorded sampling hour. That **~70x is an observed ESS-per-hour ratio, not an isolated kernel speedup**: the
  NUTS arm overlapped another GPU job at its start, and the arms adapted different step sizes. f_3's low region (0.9%)
  is crossed 26 times in 7 chains, with a longest sojourn of 17%. Jump acceptance is 0.42-0.87 for bins and 0.51-0.82 for
  IRN pairs. In bins where plain NUTS already mixes (f_9-f_30), both arms agree on the occupancies.
* **Fail-closed gate**, applied as a function (the CLI is HD-only) with relevance declared from the pooled draws of both
  arms:
  * NUTS: FAIL, 20 parameters, 25 of 58 relevant regions.
  * Hybrid: FAIL on only 4 region criteria:
    * f_1 occupancy MCSE 0.0104-0.0113 (bound 0.01);
    * f_2 low / high with 4 entries / exits each. f_2's low region is "relevant" only because NUTS's trapped
      chains inflate the pooled relevance; the hybrid alone puts 0.2% there;
    * f_3 indicator R-hat 1.010-1.011.
  * All parameters pass in the hybrid arm.
* Wall time: the hybrid arm sampled in 0.34 h against 0.45 h for NUTS (smaller lockstep cost: 255 vs 276 steps;
  arm A also overlapped with another agent's GPU test run at its start). The jump sweep itself costs ~0.09 s of
  ~0.8-1.1 s per iteration.

## 10. HD^free hybrid pilot

`configs/m2/hd_fs30_hybrid_pilot.json` (c0c1182; run at the clean SHA 0813235). The setup is identical to pilot E: the same
seed-31 starts at hd_fs30 draws, fixed dense hd_fs30 metric, fixed step size 0.025, 8 vectorised chains and the fast
exact likelihood. One jump sweep per iteration is added (proposals `configs/m2/proposals/hd_fs30_v1.json`, fitted from
hd_fs30 + E). 100 draws per chain: 0 divergences, acceptance 0.945, 48.5 s per iteration (E: 45.1 s; the sweep adds ~5 s).
GPU time 83 min including setup. Matched comparison over the first 50 draws, plus the full run:

| | E: plain NUTS, 8 x 50 | hybrid, first 8 x 50 | hybrid, 8 x 100 |
|---|---|---|---|
| sampling wall [h] | 0.63 | 0.67 (half the full run's time, not separately measured) | 1.35 |
| max R-hat | 1.173 | 1.125 | 1.062 |
| parameters with R-hat >= 1.01 | 120 | 102 | 58 |
| worst parameter, min(bulk, tail) ESS | J0610-2100 log10_A, 19 | f_8, 38 | f_1, 140 |
| worst ESS per sampling hour | 30 | 57 | 104 |
| median parameter min ESS | 264 | 294 | 560 |
| f_1 low: occupancy, excursions (chains) | 21.8%, 51 (8) | 21.5%, 45 (8) | 17.6%, 102 (8) |
| f_3 low (released 0.70%) | 0, 0 (0) | 0, 0 (0) | **0, 0 (0)** |
| f_4 low | 4.8%, 12 (4) | 7.5%, 23 (7) | 7.5%, 50 (8) |
| f_8 low (released P(< -9) 5.96%) | 3.2%, 8 (3) | 5.8%, 20 (6) | 4.8%, 42 (7) |
| f_13 / f_14 / f_29 high | 0.2% 2 (1) / 1.2% 10 (4) / 0 | 0.2% 2 (1) / 0.8% 6 (2) / 0 | 0.2% 4 (2) / 0.6% 10 (4) / 0.2% 4 (1) |

Jump acceptance: bins 0.46-0.86, IRN pairs 0.42-0.78. All bins: `outputs/m2/fs_compare_hd_fs30_v2_pilotE_hd_fs30_hybrid_pilot{,_N50}.json`.

* **Gains on HD are real but smaller than on CURN.** At matched length: worst ESS 2x, 15% fewer parameters with
  R-hat >= 1.01, and 2-2.5x more excursions at f_4 and f_8. At f_1, whose low region plain NUTS already crosses,
  there is no gain. Pilot E's long J0610-2100 excursion does not recur, but the random numbers differ, so that alone
  is not evidence.
* **Gate** (fail-closed, relevance from the released core): convergence FAIL (65 parameters, 30 bins, 52 of 58
  relevant regions; "no exploration evidence" only at f_3), agreement INCONCLUSIVE (78 comparisons with our SE > 0.01;
  0 disagreeing). Exit 1, as expected for 800 draws.
* **Why f_3 is not visited (corrected in round 3; the round-2 "only jointly with f_4" diagnosis was false).** The exact
  conditional P(f_3 < -10 | all other parameters) at 40 hybrid draws (5 per chain) is ~0 at 39 of them (median 2e-9)
  and 0.41 at one (chain 6, draw 99). That single draw contributes 99.9% of the 40-point sum. Varying f_4 alone at that
  state barely changes it: P = 0.4106 at the saved f_4 = -14.8 and 0.5164 at f_4 = -7.5 (reviewer, refined
  quadrature). In the released chain, 1,084 of the 2,585 f_3-low draws have f_4 > -8. Conclusion as of round 2:
  **the f_3 conditional mass varies strongly with the remaining parameters; the controlling coordinates were
  unidentified.** Round 3 identifies them (Sec. 13). The Rao-Blackwellised average (1.0%) is **exploratory only**:
  one point carries 99.9% of it, so its SE is about the size of the estimate itself, and it says nothing about
  agreement with the released 0.7%. The ~61 f_3 shelf proposals of the block sweep were made at intermediate states,
  not at the saved endpoints, so "none accepted" does not show that 1-D moves cannot work.
* **Projection with this kernel (indicative, from one 8 x 100 run).** The worst-parameter ESS rate (0.18 per draw)
  would reach 400 at ~290 draws per chain (~4 h). The fail-closed gate is much more demanding, though:
  * f_1's occupancy MCSE (0.044 at 800 draws; indicator ESS ~0.1 per draw) needs ~2,000 draws per chain (~27 h)
    to reach 0.01;
  * the rare high tails (f_13, f_28, f_29) need >= 10 entries and exits: ~500-1,000 draws per chain;
  * f_3 needs moves that reach its conditional low mass, which is ~0 at most states.

  **A full run with this kernel would most likely still fail the gate.**

## 11. Prototype: cached conditional likelihood of one bin (groundwork, not used in any run)

`scripts/fs_schur_prototype.py`. Changing rho_k changes only the 134 coefficients of bin k (2 per pulsar). Order the
coefficients by (frequency, pulsar). Phi is block-diagonal over frequency (Phi_j = diag(phi_irn[:, j]) + phi_j Gamma,
L_j = chol(Phi_j)), and logL = -1/2 [s - z^T B^-1 z + log|B|] + const with B = I + L^T A L and z = L^T b.
Split the coordinates into U (bin k, 134) and R (the rest):

    log|B| = log|B_RR| + log|S|,  z^T B^-1 z = z_R^T B_RR^-1 z_R + t^T S^-1 t,
    S = I + L_U^T M L_U,  t = L_U^T (b_U - h),  M = A_UU - A_UR L_R B_RR^-1 L_R^T A_RU,  h = A_UR L_R B_RR^-1 z_R.

Only L_U depends on rho_k. S >= I bounds its smallest eigenvalue below in exact arithmetic, but **not its condition
number**: the reviewer measured cond(S) ~ 1.6e7 near rho = -2 .. -1 at the exceptional state. The setup is one
3886-dim Cholesky (1.8-6.2 s, numpy on a loaded CPU); after it, each rho_k value costs a 134-dim Cholesky (9-50 ms,
unoptimised).

Check against the production `PTALikelihood._logL`: 40-point grids over rho_k in [-15.4, -2] at hd_fs30 draws, comparing
differences from the first grid point (which cancels the constant). The largest disagreement is a few 1e-9 at
logL ranges of ~1e3:

| bin | draw | logL range over the grid | max abs difference |
|---|---|---|---|
| f_3 | 0 | 1223 | 1.6e-9 |
| f_8 | 500 | 1343 | 8.2e-10 |
| f_1 | 900 | 851 | 9.0e-9 |
| f_29 | 300 | 1389 | 9.9e-10 |

(Round 2 also compared a single-chain CPU grid without its rebuild cost against an 8-chain GPU likelihood benchmark.
That comparison did not establish the cost of an 8-chain grid move and is withdrawn. The measured complete-update
cost of the JAX implementation is in Sec. 13.) The cache is valid only for the current values of all other parameters
and must be rebuilt per state.

## 12. Round-2 recommendation (superseded by Sec. 15)

**Recommendation: (b), modified: do not launch a full HD run yet. Adopt the hybrid kernel, then a short, targeted round of
kernel work and one more 2-h HD pilot before committing GPU-days.**

1. **Keep the exact hybrid kernel (NUTS + frozen block-MH jumps).** It is cheap (+11% per HD iteration), tested for
   invariance. On CURN^free it removed most of the trapping (max R-hat 1.470 -> 1.007, worst ESS 15 -> 793, ~70x
   observed ESS per recorded hour, with GPU contention in the NUTS arm). On HD it gave ~2x at matched length.
2. **Add joint bin-pair blocks**, e.g. (f_3, f_4), (f_1, f_2) and (f_7, f_8), with 2-D proposals that include
   "both low" cells, plus bin + IRN-pair blocks for the pulsars whose amplitudes co-move with the low bins. (Round 3:
   the f_3/f_4 premise was false; see Sec. 13 for the coordinates that do matter.) Fit them from the hybrid pilot's
   draws, never the released chain.
3. **Use Metropolised conditional-grid moves for the low-frequency bins**, built on the cached conditional of
   Sec. 11 (exact via an MH correction of the interpolated grid density). They track the current conditional,
   unlike frozen marginal proposals. The same conditional gives Rao-Blackwellised occupancy estimates. Lower
   pointwise variance does not by itself imply lower MCSE (autocorrelation matters), so they are exploratory until
   their own R-hat / ESS / MCSE and contribution concentration have been assessed. If they are ever admitted to the
   gate, the transport requirements stay.
4. **Warmup:** start step-size adaptation near the measured workable value (0.02-0.025). On CURN that gave a warmup
   costing only 1.3-1.6x a sampling iteration. Keep diverse region starts (`init_rho_low_frac`). Shared-step adaptation is optional:
   averaging acceptance can hide a difficult chain.
5. **Next pilot (~2 GPU-h):** HD 8 x 100 with items 2-4. Proceed to a full run only if f_3 shows replicated
   entries/exits and the projected f_1 MCSE length is <= ~800 draws per chain.

Not (a): the v2 recipe is untested at a sensible step-size start, but its plain-NUTS kernel is the one that fails here. Not
(c) as formulated in round 1: the shelf warp's benefit in the final coordinate is ~2.5x in scale ratio, not ~25x, and the
pivot cancels. The warp remains a controlled competitor for later.

## 13. Round 3: proposal validation, exact conditional-grid moves, what controls f_3

**Proposal validation** (`hybrid.validate_proposals`, called before initialisation by `make_sweep`, `make_grid_moves` and
`HybridNUTS`). A proposal file must:
* match the target's parameter names;
* use distinct, in-range block indices, with no parameter in two blocks;
* have prior bounds equal to the target's (`posterior.transform`);
* have w_prior in (0, 1];
* have finite, strictly increasing edges inside the bounds.

Otherwise it is rejected with a ValueError. Every uniform component (the prior part and each histogram cell) is now
sampled and evaluated on the same trimmed interval [a + d, b - d] (d = 1e-12 (b - a)), and log q = -inf outside the
support. Tests cover mismatched (narrower) bounds and nine malformed variants.

**Cached conditional in JAX** (`ptagwb.conditional`, the Sec. 11 formulation, jit/vmap-able). Agreement with the
production likelihood depends on how much power the other common bins carry:
* at posterior states (other bins log10 rho <= -6): <= 1e-9 relative on the synthetic PTA (tests), and <= 6e-8 in
  logL on the real data (f_1, f_3, f_4, f_8 at four hybrid draws; `--validate`);
* at prior draws z ~ U(-4, 4) with several bins near the maximal power: up to 7e-4. A long-double dense reference on
  the synthetic PTA shows that the dense Schur form (not production) loses precision there, because
  B = I + L^T A L is very ill conditioned.

The conditional is therefore used **only to build proposals**. The Metropolis-Hastings acceptance always uses the
production likelihood.

**Metropolised conditional-grid move** (`hybrid.make_grid_moves`, `RunConfig.grid_bins` / `grid_kw`). For bin k at the
current values of all other parameters:
1. Rebuild the cache (it is independent of rho_k; tested) and evaluate the conditional on a grid: 48 uniform nodes on
   the trimmed prior range [a, b] (d = 1e-9 (hi - lo)) plus 32 nodes on [x* - 0.75, x* + 0.75], x* the coarse argmax.
2. Propose rho' from the independence density
   q(rho | rest) = 0.05 / (b - a) + 0.95 q_grid(rho | rest) on [a, b]. q_grid is the normalised piecewise log-linear
   interpolant of the conditional through the nodes, sampled exactly by a per-cell inverse CDF. The 5% uniform part is
   the only explicit extra component; there is no separate low-region component.
3. Accept with min(1, L(x') q(rho | rest) / (L(x) q(rho' | rest))), L the production likelihood.

*Joint variant* (`grid_kw.companions`): one or more pulsars' IRN (log10_A, gamma) pairs are drawn from their frozen
2-D block proposals q2, then rho' from q(. | rest with the new pairs). The reverse density uses q2 at the current
pairs and q(. | current rest). This is an exact independence proposal for the joint block, at the cost of two cache
builds per move.

**Tests** (`tests/test_conditional.py`, `tests/test_hybrid.py`; 267 passing in total with the gate and sampling tests):
* grid density normalisation and its sampler (KS);
* invariance of the one-bin move from exact conditional draws (KS), also with a deliberately coarse grid;
* a q-ratio-free variant fails the same test (p < 1e-6);
* joint-move invariance against a 3-D grid reference (marked slow);
* an end-to-end HybridNUTS run with grid bins.

**Complete 8-chain update** (GPU, `bench/bench_grid.py`, `bench/results/grid_update_hd_fs30_grid_pilot.json`). Cost per
iteration of everything after NUTS:

| component | cost per iteration |
|---|---|
| block sweep (30 bins + 8 IRN pairs) | 5.1 s |
| grid moves (f_1, f_2, f_4, f_8 one-bin; f_3 three times jointly with J1713+0747 / J1909-3744 / J0030+0451) | 2.6 s |
| **complete update, incl. potential/gradient refresh** | **7.9 s** |
| one leapfrog step (value + grad), for scale | 0.16 s |

At posterior draws, one-bin grid moves accept 0.88-1.0 and the joint f_3 move about 0.13.

**What controls P(f_3 < -10 | rest)?** (`scripts/fs_conditional_analysis.py --bin 2`, CPU, 40 hybrid-pilot states
(chain c, draw n = 20, 39, 59, 79, 99); output `outputs/m2/fs_conditional_hd_fs30_hybrid_pilot_bin2.json`.)
* **Quadrature:** at S* = (chain 6, draw 99), 241 nodes give 0.4127 and 961 give 0.4110.
* **The reviewer's f_4 check, reproduced:** P = 0.413 at the saved f_4 vs 0.519 at f_4 = -7.5.
* **Controlled swaps:** every other group (29 bins, 67 IRN pairs) was swapped between S* and three reference states T
  with median conditional mass (P ~ 2-4e-9: (2, 99), (1, 79), (5, 20)). Changes are in the log-odds of the conditional
  low mass.

| group | log-odds change when S* takes T's value | when T takes S*'s value |
|---|---|---|
| J1713+0747 IRN | -72 / -5 / -72 | +9.6 / +7.8 / +12.6 |
| J1909-3744 IRN | -23 / -30 / -78 | +5.6 / +13.8 / +5.8 |
| J0030+0451 IRN | -6.5 / - / - | +5.1 / - / - |
| f_1, f_4, f_5 (bins) | ~+0.1-0.5 | -60 / -15 / -9 (f_1, f_4, f_5 at T (2, 99)) |

  The two best-timed pulsars dominate. In S*, J1713+0747 and J1909-3744 have IRN amplitudes at the top of their
  pilot distributions (log10_A = -14.3 and -14.5 vs pilot medians -15.6 and -17.2; gamma ~ 3), so their intrinsic noise
  can absorb the f_3 power. Inserting S*'s values of these two pairs, plus J0030+0451 / J1744-1134 and some bins,
  cumulatively into T raises P from ~1e-9 to 0.17-0.46.
* **Strong interactions:** S*'s low f_1 / f_4 / f_5 values *lower* the f_3 low mass when inserted alone into T, so
  the effect is not additive.
* **Conclusion:** the controlling coordinates are mainly the IRN of J1713+0747 and J1909-3744 (secondarily J0030+0451
  and J1744-1134), conditional on the low-frequency bins. This is evidence from three state pairs around one
  exceptional state, not a complete map. It motivates the joint f_3 + IRN moves used in the pilot.
* **Same analysis for f_8** (`--bin 7`; `outputs/m2/fs_conditional_hd_fs30_hybrid_pilot_bin7.json`). The f_8 conditional
  low mass is far less concentrated: mean 0.050, largest single state 25% of the sum, median 1.3e-3. Again
  J1713+0747's IRN dominates in all three swap pairs (+4.0 to +5.5 log-odds into T), followed by J2043+1711.
  Setting f_4 to -14.78 / -7.5 at S* gives P = 0.43 / 0.51, so f_4 does not control f_8 either.

## 14. Enhanced HD pilot

**Rules fixed before launch.** The configs were committed first (e5f8311, 67db032). The relevance declaration and
the proposals (`hd_fs30_v2.json`, fitted from hd_fs30 + pilot E + the hybrid pilot, never the released chain) were
frozen. Success = replicated f_3 entries and exits (>= 2 chains) plus improved occupancy MCSE vs the hybrid pilot.

**Attempt 1: `hd_fs30_grid_pilot`, aborted.** Step-size adaptation (dual averaging) for 25 iterations, starting at
the workable 0.025, with the same starts and fixed dense metric as before. The pre-set rule was to abort if warmup
took more than 35 min. Per-iteration times were 171, 8, 29, 173, 90, 91, 49, 173, 91, 173, 90, 173, 90, 90, 172, 172,
90, 90 s: most iterations ran 511-1023 lockstep leapfrog steps. The rule fired at 18/25 iterations (38 min of GPU).
So on HD, even starting near a workable step size, dual-averaging adaptation of 8 vectorised chains repeatedly
drives some chain to tree depth 9-10. This resolves the round-1 confounder: the poor 1.0 start was not the
cause. (CURN^free did not show this, Sec. 9.)

**Attempt 2: `hd_fs30_grid_pilot_fixed`, the result.** The attempt-1 config said "relaunch nothing". I deviated from
that and documented it in the new config before launch: it is the same pilot with a **fixed step size 0.025** and no
adaptation, exactly matching pilot E and the hybrid pilot. Run at the clean SHA 67db032:
* 140 draws per chain;
* the 2-h sampling cap is checked between 10-draw blocks, so sampling actually ran 2.11 h (6.5 min over);
* 54 s per iteration (hybrid pilot: 48.5 s), 0 divergences, acceptance 0.942.

Grid-move acceptance: 0.95-0.96 for the one-bin moves (f_1, f_2, f_4, f_8) and 0.24 for the three joint f_3 + IRN moves.

| | hybrid pilot, 8 x 100 | grid pilot, first 8 x 100 | grid pilot, 8 x 140 |
|---|---|---|---|
| sampling wall [h] | 1.35 | 1.53 (pro rata) | 2.11 |
| max R-hat / parameters >= 1.01 | 1.062 / 58 | 1.079 / 51 | 1.066 / 25 |
| worst min(bulk, tail) ESS (parameter) | 140 (f_1) | 73 (J1944+0907 log10_A) | 89 (J1944+0907 log10_A) |
| median parameter min ESS | 560 | 555 | 754 |
| f_1 low: occupancy, excursions (chains), MCSE(< -9) | 17.6%, 102 (8), 0.044 | 18.1%, 113 (8), **0.025** | 17.9%, 164 (8), 0.029 |
| f_3 low (released 0.70%): occupancy, entries / exits (chains) | 0, 0 / 0 (0) | 0.1%, 1 / 1 (1) | **0.18%, 2 / 2 (2)** |
| f_4 low: occupancy, excursions, MCSE | 7.5%, 50 (8), 0.018 | 7.9%, 85 (8), 0.021 | 7.7%, 119 (8), 0.014 |
| f_8 low: occupancy, excursions, MCSE | 4.8%, 42 (7), 0.016 | 5.6%, 60 (8), **0.010** | 6.0%, 91 (8), 0.010 |
| f_13 / f_28 high: excursions (chains) | 4 (2) / 2 (1) | 12 (4) / 6 (3) | 12 (4) / 15 (5) |

**Gate** (fail-closed, frozen relevance file): convergence FAIL (28 parameters, 29 bins, 49 of 58 relevant regions),
**no "no exploration evidence" region any more**. f_3 low now fails on its event count (2 entries and 2 exits < 10),
an unavailable indicator, and MCSE 0.017 > 0.01. Agreement INCONCLUSIVE (74 comparisons with our SE > 0.01).
Exit 1.

**Against the pre-set success criterion:**
* **f_3 entries/exits: met only at the literal minimum.** There were 2 entries and 2 exits in 2 chains, each a
  single-draw visit (chain 3, draw 118; chain 7, draw 4). Both visits occur with J1713+0747 at a high IRN amplitude
  (-14.1, -14.5), consistent with Sec. 13. The f_3 low occupancy (0.18%, from 2 of 1,120 draws) is far too poorly
  determined to compare with the released 0.7%.
* **Occupancy MCSE: improved for f_1 and f_8.** At matched length, f_1 went 0.044 -> 0.025 and f_8 0.016 -> 0.010,
  with ~1.4-1.6x more excursions per draw at f_4 / f_8 / f_13 / f_28. f_4 was mixed (0.018 -> 0.021 matched, 0.014 at
  140 draws). f_3's MCSE rises (0.009 -> 0.017-0.026), because the sparse-count SE grows once the region is visited
  at all.
* **Not improved:** the worst-parameter ESS (an IRN amplitude, J1944+0907) and the ESS per hour (iterations are ~11%
  more expensive).

**Indicative projection with this kernel** (from one 8 x 140 run; not a calibrated budget):
* f_3 needs >= 10 entries and exits; at 2 per 1,120 draws that is ~700 draws per chain (~10.5 h);
* f_1's occupancy MCSE 0.029 at 140 draws scales to 0.01 at ~1,200 draws per chain (~18 h);
* the gate would therefore need **~18-20 GPU-h at least**, with f_3's rate resting on two events.

GPU time this round: complete-update benchmarks ~10 min, attempt 1 38 min, attempt 2 2.15 h (setup + 1 iteration +
2.11 h sampling), **~3.0 GPU-h** in total.

## 15. Recommendation (round 3)

**(b), modified again: do not launch a full HD run yet.**

* **Use this kernel next:** NUTS + block sweep + Metropolised conditional grids is exact, tested, and better than the
  hybrid baseline where it matters most. f_3's low region is visited for the first time (2 independent entries and
  exits), and f_1 / f_8 occupancy MCSE drops by ~40% at matched length. It does not change the worst IRN ESS.
* **f_3 is still the bottleneck.** Its conditional low mass is ~0 except where J1713+0747 and J1909-3744 (and
  J0030+0451, J1744-1134) sit at high IRN amplitude together with particular low-frequency bins (Sec. 13). Joint
  moves built from marginal histograms rarely propose such a configuration. Two candidates, to be benchmarked
  before any longer run:
  1. a joint f_3 move whose companion proposals are the *conditional* grids of those IRN amplitudes (rebuilt per
     state), not frozen histograms;
  2. parallel tempering (2-4 temperatures), now that a multi-coordinate bottleneck is evidenced. Judge it by
     cold-chain f_3 excursions and temperature round trips.
* **Warmup:** on HD, dual-averaging step-size adaptation with 8 lockstep chains saturates the tree depth even when it
  starts at 0.025 (attempt 1). Use a fixed step size from a short pilot (0.025 works, acceptance 0.94), or adapt one
  shared step size from pooled acceptance, which still needs a test.
* **Full-run decision:** launch only after a ~2 GPU-h pilot shows f_3 entries/exits at a rate projecting >= 10 within
  ~600 draws per chain. The current kernel projects >= 18-20 GPU-h with a fragile f_3 estimate.

## 16. Reproduction

```bash
uv run --no-sync python scripts/m2_run.py configs/m2/hd_fs30_v2_pilotE.json   # stopped after 50 draws (2 blocks)
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run hd_fs30_v2_pilotE
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run hd_fs30     # (existing)
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_pilot_diag.py --run hd_fs30_v2_pilotE
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_pilot_diag.py --run hd_fs30
```

Round 2:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests/test_hybrid.py tests/test_freespec_gate.py tests/test_sampling.py  # incl. -m slow
uv run --no-sync python scripts/fs_fit_proposals.py --runs curn_fs30 --out configs/m2/proposals/curn_fs30_v1.json
uv run --no-sync python scripts/fs_fit_proposals.py --runs hd_fs30,hd_fs30_v2_pilotE --out configs/m2/proposals/hd_fs30_v1.json
uv run --no-sync python bench/bench_jumps.py --model curn && uv run --no-sync python bench/bench_jumps.py --model hd
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python scripts/m2_run.py configs/m2/curn_fs30_ab_nuts.json
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python scripts/m2_run.py configs/m2/curn_fs30_ab_hybrid.json
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python scripts/m2_run.py configs/m2/hd_fs30_hybrid_pilot.json
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_compare_kernels.py curn_fs30_ab_nuts curn_fs30_ab_hybrid
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_compare_kernels.py hd_fs30_v2_pilotE hd_fs30_hybrid_pilot [--draws 50]
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run hd_fs30_hybrid_pilot
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_schur_prototype.py --bin 2 --draw 0   # also --bin 7/0/28
```

The f_3 Rao-Blackwellised check (Sec. 10) used `BinConditional` from `scripts/fs_schur_prototype.py` on 40 hybrid-pilot
draws (5 per chain at evenly spaced positions after the first 20%) with a 241-point grid on [-15.5, -1] and the uniform prior.

Round 3:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests/test_conditional.py tests/test_hybrid.py   # incl. -m slow
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_conditional_analysis.py --validate
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_conditional_analysis.py --run hd_fs30_hybrid_pilot --bin 2   # also --bin 7
uv run --no-sync python scripts/fs_fit_proposals.py --runs hd_fs30,hd_fs30_v2_pilotE,hd_fs30_hybrid_pilot \
    --pairs J0610-2100,J2234+0611,J0437-4715,J1853+1303,J0645+5158,J1713+0747,J1909-3744,J0030+0451 --out configs/m2/proposals/hd_fs30_v2.json
uv run --no-sync python scripts/fs_freeze_relevance.py --out configs/m2/relevance/hd_fs30_released_v1.json
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python bench/bench_grid.py --config configs/m2/hd_fs30_grid_pilot.json
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --no-sync python scripts/m2_run.py configs/m2/hd_fs30_grid_pilot_fixed.json
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_compare_kernels.py hd_fs30_hybrid_pilot hd_fs30_grid_pilot_fixed [--draws 100]
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run hd_fs30_grid_pilot_fixed \
    --relevance-file configs/m2/relevance/hd_fs30_released_v1.json
```

Pilots A-D: `configs/m2/hd_fs30_v2_pilot{,B,C,D}.json`. Their logs (`runs/logs/*.aborted.log`) and run directories
(`runs/*.aborted/`, metadata only) are git-ignored. The per-iteration warmup times above are read from the
numpyro progress bar (`progress_bar: true`). One 8-chain step costs 0.16 s, so 41 / 82 / 164 s correspond to 255 / 511 / 1023
lockstep leapfrog steps.
