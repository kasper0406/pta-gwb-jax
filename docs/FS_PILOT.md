# HD free-spectrum (30 modes) re-run: pilot before committing to hd_fs30_v2

Question: would the planned re-run `configs/m2/hd_fs30_v2.json` (8 chains, 500 warmup + 750 draws,
independent overdispersed init z ~ U(-4, 4), diagonal windowed adaptation) pass the free-spectrum
acceptance gate (docs/M2_RESULTS.md Sec. 10), and at what cost? Budget: ~2 GPU-hours (RTX 5090),
abort above 2.5 h. **Used: 1.93 GPU-hours** (five pilots, 10:23-12:21 UTC on 2026-10-08, including setup).

**Verdict: do not launch v2 as planned (option c: reparameterise first, and fix the warmup).**
1. The v2 *warmup* does not fit its budget. With the overdispersed init, from warmup iteration ~8
   onwards at least one of the 8 lockstep chains reaches tree depth 10 (1023 leapfrog steps,
   164 s per 8-chain iteration) in most iterations. The same happened with an initial diagonal or
   dense metric taken from the M2 run, and even with chains started at M2 posterior draws, as long as
   the step size was still being adapted (pilots A-D, all aborted under their pre-set rules).
   Projected v2 warmup: **~11-18 h**, against ~6-9 h in the M2 cost estimate (docs/M2_RESULTS.md Sec. 10).
2. Even after a good warmup, the free-spectrum bins whose posterior has a small second region
   (low power at f_1, f_3, f_4, f_8; signal at f_13, f_14, f_28, f_29) are crossed rarely. The worst
   parameters reach only 0.02-0.05 bulk/tail ESS per draw, and the 0.7% low-power region at f_3 was never
   visited in 1,400 post-warmup draws (M2 + pilot E). v2's 6,000 draws would give a worst-parameter
   ESS of ~115-285 < 400: **expected gate result FAIL (convergence)**. Meeting R-hat < 1.01 and ESS >= 400 with
   the current parameterisation needs ~1,500-2,700 draws per chain (8 chains): **~30-50 GPU-hours
   including warmup**, with no guarantee for f_3.
3. Neither limitation is a classic IRN funnel (no step-size-normalised slow-down in the neck).
   In the sampler's coordinate, every problematic parameter has **two regions whose scales differ by 15-35x**:
   a narrow likelihood peak (sd(z) 0.02-0.07) and a broad prior-dominated shelf (sd(z) 1.0-1.2).
   The marginal potential rises by 3.7-7.8 nats from the peak to the shelf. A coordinate warp that compresses
   the shelf (Sec. 6) targets exactly this and is cheap to test on CURN^free first.

## 1. Code and configs (all committed before the runs they affect; every run recorded a clean SHA)

| commit | content |
|---|---|
| 396f2bc | `RunConfig.likelihood_impl`: `"production"` (default, unchanged) or `"fast"` (opt-in `FastPTALikelihood(reduce="hh", tri_inv="levels")`, exact within the docs/PERF.md budgets); `sampling.make_likelihood` used by `scripts/m2_run.py`; run metadata now also records the likelihood class / reducer / triangular inverse. Tests: default + validation, fast vs production potential and gradient (HD free spectrum, mixed precision; HD power law, float64) on a synthetic PTA. Config pilot A. |
| e9532d3, 3c83d31, 7b56c29 | configs pilots B, C, D (each written after the previous abort, with its own abort rule) |
| db6e9be | `RunConfig.step_size` / `adapt_step_size` (defaults = numpyro's 1.0 / True, so no existing config changes behaviour) + test; config pilot E |
| this commit | `scripts/fs_pilot_diag.py` (crossings, IRN indicators, projection), this document |

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

## 4. Diagnostics of the post-warmup draws (pilot E, 8 x 50; M2 hd_fs30, 4 x 250)

**Gate** (`scripts/m2_freespec_diag.py --run hd_fs30_v2_pilotE`, exit 1): **convergence FAIL** (163 of
164 parameters, 30 of 30 bins; expected: 400 draws cannot reach ESS 400). **Reproduction agreement PASS** (0 bins
disagreeing, max |z| 2.0 at f_6; the SEs are large at this length). Max R-hat 1.173 (f_26), worst bins f_26,
f_1 (1.152), f_25 (1.111), f_4 (1.104). M2 hd_fs30 for reference: convergence FAIL (59 parameters), agreement PASS.

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
* Gate weakness noted in passing: a bin whose minority region is never visited (f_3) has an
  "unavailable" indicator (warning only) and fine parameter ESS. With 6,000 never-visiting draws, its
  sparse-count agreement SE gives |z| ~ 2.8 < 3.5, so **the gate could pass a run that never visits f_3's
  low-power region**.

**Why the transitions are rare (released core, sampler coordinate z).** Scales and potential rise
from the signal peak to the low-power shelf: f_1 sd(z) 0.074 vs 1.08, dU 3.7 nats; f_3 0.035 vs 1.01,
dU 7.8; f_4 0.046 vs 1.17, dU 4.7; f_8 0.044 vs 1.08, dU 5.4 (f_13 / f_29: 0.03 / 0.02 vs 1.1, dU 0.6).
One global scale per coordinate cannot serve both regions, and the chain must climb several nats
onto a shelf ~25x wider than the peak.

**IRN parameters.** In pilot E, 92 IRN parameters have R-hat >= 1.01 (M2: 26; short chains). Worst:
J0610-2100 log10_A R-hat 1.123, tail ESS 19; J2234+0611 1.100; J0437-4715 1.092; J1853+1303 1.081;
J0645+5158 1.057; J1713+0747 1.055. Funnel indicators (pulsar-wise, both runs):
* conditional width of z_gamma in the lowest vs highest log10_A tercile: median ratio 1.12, worst 2.2-2.6
  (J1853+1303, J1713+0747, J0613-0200): mildly funnel-shaped;
* per-draw mobility of z_gamma in units of its conditional sd: 0.9-2.1 in both terciles. The
  Spearman correlation of the step-size-normalised move |dz_gamma| / (eps sqrt(M^-1)) with log10_A
  has median -0.02 (worst -0.25), and tree length vs log10_A has |rho| <= 0.14: **NUTS is not locally slowed
  in the neck**, so this is not a classic funnel limit;
* instead the worst amplitudes show the same **two-region metastability** as the bins. Example: J0610-2100's released
  posterior has 0.6% below log10_A = -14. Pilot E chain 0 went to -19.5 and spent 56% of its 50 draws below -14,
  while the other 7 chains stayed in [-13.8, -11.8] (M2: chain 0 8%, others 0%). The mobility of z_A in its low-A
  tercile is 0.47 sd per draw. IRN amplitudes also correlate moderately with low-frequency bins (Spearman up to
  -0.34 with f_1), so their slow movement partly inherits the bins' metastability.

## 5. Projection

Inputs: 8 chains; post-warmup 45.1 s per iteration with a dense metric (pilot E; a diagonal metric is
unlikely to be cheaper, cf. CURN^free 213 steps per draw with diagonal adaptation); warmup 80-127 s per iteration
(pilot A iterations 7-16: 127 s; pilots C/D during step-size adaptation: ~80 s; CURN^free analogue: 1.8x
its sampling lockstep cost, i.e. ~81 s at 8 chains).

| quantity | value |
|---|---|
| v2 as planned (500 + 750) | warmup ~11-18 h + sampling 9.4 h = **~20-28 GPU-h** (plan: 13-21 h) |
| v2 expected worst-parameter ESS | 6,000 draws x 0.019-0.047 ESS per draw (M2 f_8 tail; E J0610-2100 log10_A tail) = **~115-285 < 400 -> gate FAIL** |
| draws per chain for ESS >= 400 on all 164 | 1,050 (E rate) - 2,650 (M2 rate) |
| draws per chain for indicator ESS >= 400 (two-state estimate 800 p(1-p)/f, released p, pooled M2+E transition rate f) | f_8: ~1,500; f_4: ~560; f_1: ~440; f_13/f_29: ~200; **f_3: >= 650 (lower bound; no transition ever observed)** |
| cost to pass with the current parameterisation | ~1,500-2,700 draws per chain: 19-34 h sampling + 11-18 h warmup = **~30-50 GPU-h**, f_3 not guaranteed |

Caveats: E's ESS estimates come from 50-draw chains initialised at hd_fs30 draws (noisy, and between-chain differences
partly inherited); M2's from 250-draw chains. The ranges are indicative, but both runs point the same way, and the
v2 configuration cannot reach the target at the planned length under any of them.

## 6. Recommendation: (c) reparameterise first; fix the warmup as well

Not (a): v2 would cost ~20-28 GPU-h and would almost certainly fail convergence. Not (b) alone: more
chains or a longer run means ~30-50 GPU-h with plain NUTS, and still no f_3 visits. A dense metric from the pilot does not change the
two-scale geometry: pilot E already used one.

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

## 7. Reproduction

```bash
uv run --no-sync python scripts/m2_run.py configs/m2/hd_fs30_v2_pilotE.json   # stopped after 50 draws (2 blocks)
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run hd_fs30_v2_pilotE
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run hd_fs30     # (existing)
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_pilot_diag.py --run hd_fs30_v2_pilotE
JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_pilot_diag.py --run hd_fs30
```

Pilots A-D: `configs/m2/hd_fs30_v2_pilot{,B,C,D}.json`. Their logs (`runs/logs/*.aborted.log`) and run directories
(`runs/*.aborted/`, metadata only) are git-ignored. The per-iteration warmup times above are read from the
numpyro progress bar (`progress_bar: true`). One 8-chain step costs 0.16 s, so 41 / 82 / 164 s correspond to 255 / 511 / 1023
lockstep leapfrog steps.
