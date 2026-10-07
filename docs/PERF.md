# Performance study: likelihood, backends, samplers (branch `perf-bench`)

Goal: find where the HD value+gradient time goes, whether CPU or batching helps, which *exact*
likelihood speedups exist, and which sampler gives the most effective samples per second on the
HD^13/3 posterior. Nothing in the production code paths (`ptagwb.likelihood`, `ptagwb.sampling`)
was changed; all new code is in `src/ptagwb/perf_likelihood.py` (exact variants, opt-in),
`bench/` (harness, results) and `tests/test_perf_likelihood.py`.

Hardware: RTX 5090 (FP64 DGEMM measured **1.90 TFLOP/s**, FP32 69 TFLOP/s), Ryzen 9 9950X3D
(16 Zen5 cores), jax 0.11.2, BlackJAX 1.7.1, NumPyro 0.22.0. Model: 67 pulsars, 30 IRN modes,
14 common modes, enterprise positions (exactly as `scripts/m2_run.py`). All times are medians
after JIT, float64.

## Summary

* **Profile (HD, 1 chain, 14.3 ms value+grad, GPU busy 98%)**: 1876-dim Cholesky 4.9 ms
  (cuSOLVER, 0.44 TFLOP/s), backward triangular inverse 5.2 ms (production recursion, 32 serial
  leaf trsm + dense GEMMs), per-pulsar QR stage 3.5 ms for only 0.14 GF (batched cuSOLVER QR at
  ~15 GFLOP/s + Q formation). ~6.9 GF at 0.48 TFLOP/s = 25% of the measured 1.9 TFLOP/s DGEMM
  peak. Not launch-bound.
* **CPU vs GPU**: CPU best 40 ms per HD gradient (16 BLAS threads), batching/pmap/multi-process
  do not beat ~26 ms; GPU 14.4 ms single, **5.5-5.8 ms per chain at B = 16-64**. Even the
  OpenBLAS ceiling (16 single-thread processes) is ~2x below the GPU batched path.
* **Exact likelihood speedups** (bit-level agreement: value 5.8e-11 abs = ulp of the
  parameter-dependent logL, gradient 1e-13 rel, at corners/posterior/prior draws, single and
  vmapped): structured Householder for the per-pulsar stage (3.5 -> 1.0 ms) + level-batched
  triangular inverse (5.2 -> 1.9 ms). HD **14.4 -> 8.5 ms** (B = 1), 9.4 -> 7.2 (B = 4),
  5.8 -> 4.4 (B = 16); CURN **3.6 -> 1.07 ms**, 0.70 -> 0.31 at B = 16.
* **Samplers** (same metric/init/likelihood): BlackJAX NUTS gives 2x the ESS/s of NumPyro NUTS on
  CURN (1.5x ESS per gradient from a better-adapted step size, plus lower overhead) and on HD
  1.8x / 2.7x / 2.3x the M2 production ESS/s (common / worst bulk / worst tail). MAMS matches or slightly
  beats it on CURN, loses on HD (and its tuner costs 26 min on HD); ChEES/MEADS do not pay at the FP64-bound
  batched gradient cost; unadjusted MCLMC is 3x faster but **biased** (0.43 sd on an IRN mean,
  0.15 sd on the common amplitude).
* **Recommendation**: GPU x exact `hh + levels` likelihood x BlackJAX NUTS (4 chains, CURN dense
  metric, continuous step-size DA): an HD^13/3 run at M2-equal ESS in **~25-35 min instead of
  ~61 min**. The remaining large lever is an IRN reparameterisation (worst-parameter ESS is set by
  rare funnel excursions for every exact sampler).

## 1. Profile: where the HD value+gradient time goes

`bench/profile_hd.py` (stage timings, separately jitted) and a `jax.profiler` kernel trace
(`bench/parse_trace.py`; Nsight Systems 2025.6 could not import traces of the pip CUDA 13.4
runtime) at an HD^13/3 posterior draw, production code:

| stage (production, 1 chain) | time [ms] | flops | achieved |
|---|---|---|---|
| value | 8.67 | | |
| **value + gradient** | **14.29** | ~6.9 GF | 0.48 TFLOP/s (25% of DGEMM peak) |
| per-pulsar square-root stage, forward (67 x [QR 120x60 + Q formation + trsm]) | 3.54 | 0.14 GF | **0.04 TFLOP/s** |
|   of which batched geqrf (`geqr2_batch_kernel`) / orgqr | 2.61 / 0.73 | | |
| 1876-dim Cholesky (cuSOLVER potrf, `getrf_wo_pivot` kernel) | 4.85-4.98 | 2.2 GF | 0.44 TFLOP/s |
| backward triangular inverse L^-1 (production recursion) | 5.24 | 4.4 GF | 0.84 TFLOP/s |
|   of which 32 leaf `trsm` calls / GEMMs | 2.18 / ~2.6 | | |
| diag blocks of Sigma'^-1 (einsum) + 2 trsv | 0.21 + 0.31 | 0.2 GF | |
| per-pulsar analytic VJP (`_reduce_bwd`) | 0.08 | 0.03 GF | |

* The GPU is busy 14.0 ms of the 14.3 ms call (232 kernels per call): the evaluation is **not
  launch/latency bound** at the call level; it is bound by three slow library kernels.
* The per-pulsar stage does 2% of the flops in 25% of the time: cuSOLVER's batched small QR runs
  at ~15 GFLOP/s and the explicit Q formation adds another kernel.
* The production backward tri-inverse is a depth-first recursion: the top-level products are
  dense 938^3 GEMMs (the triangular zeros are multiplied), and its 32 small leaf `trsm`s run
  one at a time (68 us each).
* XLA's `cost_analysis` reports 4.7 GF for value+grad but does not count the cuSOLVER custom calls.

## 2. CPU vs GPU

`bench/bench_backends.py`: value+grad at B distinct posterior draws, vmapped (`B = 1` un-vmapped).
"per chain" = call time / B, i.e. the per-chain-gradient cost a vectorised sampler pays.

**HD (production code), ms per chain-gradient**

| backend / threads | B = 1 | B = 4 | B = 16 | B = 64 |
|---|---|---|---|---|
| GPU (RTX 5090) | 14.4 | 9.4 | 5.8 | 5.5 |
| CPU, XLA defaults (OpenBLAS 32 threads incl. SMT) | 81.0 | 67.7 | | |
| CPU, `OPENBLAS_NUM_THREADS=16` | **39.6** | 41.5 | | |
| CPU, 16 threads + `--xla_cpu_multi_thread_eigen=true` | 40.2 | 40.8 | | |
| CPU, `OPENBLAS_NUM_THREADS=1` | 55.4 | 57.7 | | |
| CPU, pmap over 16 host devices, 1 BLAS thread each | | | 33.4 | 32.6 |
| CPU, 16 pinned single-thread processes (1 chain each) | | | ~26 (aggregate) | |

**CURN (production code)**: GPU 3.6 / 1.17 / 0.70 / 1.11 ms (B = 1/4/16/64); CPU 16 threads
5.7 / 5.7 / 5.4 / 5.4 ms.

* XLA:CPU does not parallelise a vmapped batch of LAPACK custom calls (they run one after the
  other), and its own GEMMs (the backward recursion) are slow when single-threaded.
* The CPU *hardware* ceiling is also below the GPU: scipy/OpenBLAS at n = 1876 does dpotrf in
  9.9 ms and dtrtri in 8.4 ms with 16 threads (0.22-0.26 TFLOP/s; DGEMM 0.9-1.1 TFLOP/s), vs
  4.9 ms potrf on the GPU. 16 concurrent single-threaded processes reach 1 potrf+trtri per
  ~140 ms each, i.e. an aggregate ~9-10 ms per chain-gradient for the core alone (cache/memory
  contention; the two CCDs differ by 25%). Even a hand-written LAPACK pipeline on all 16 cores
  would be ~2x slower than the GPU batched path, so CPU co-processing was not pursued.
* GPU batching pays: per-chain HD cost drops 2.6x from B = 1 to B = 16 (the Cholesky becomes a
  batched potrf at 1.3 TFLOP/s and the GEMMs fill the GPU). Beyond B = 16 the gain is small
  (FP64-flop bound: B = 64 runs at ~1.3 TFLOP/s overall).

## 3. Exact likelihood speedups

New opt-in class `ptagwb.perf_likelihood.FastPTALikelihood(terms, T, reduce=..., tri_inv=...)`,
a subclass of the production likelihood with identical maths, identical analytic VJPs and the
same square-root (Householder) numerics. Two changes:

1. **`reduce="hh"`: structured Householder for the per-pulsar square-root stage.** Production
   needs T (R of [Y; I], Y = R_F Phi^1/2), Q1^T R_F and Q1^T c and gets them from a batched QR
   that forms Q. Instead, take the R factor of the augmented matrix [[Y, R_F, c], [I, 0, 0]]:
   R = Q_full^T M, so its top rows are exactly [T, Q1^T R_F, Q1^T c] with no Q formed. Because
   Y is upper triangular, Householder step k touches only top row k and the 60 x 121 bottom block
   (the top block stays triangular), so the whole QR is 60 rank-1 updates of one small block
   (`_hh_bottom`, a `fori_loop` vectorised over pulsars and chains). Same reflections, same
   backward stability; row signs cancel in E, d, q and log|T|.
   67-pulsar forward stage: 3.54 -> **1.04 ms**. (Feeding the augmented matrix to cuSOLVER's
   R-only QR instead is *slower*, 9.4 ms; kept as `reduce="aug"` for the record.)
2. **`tri_inv="levels"`: level-batched triangular inverse for the backward pass.** Pad 1876 to
   1888 = 59 x 32 with an identity block, invert all 32 diagonal 59x59 blocks in *one* batched
   trsm, then merge pairs bottom-up, [[A,0],[L21,C]]^-1 = [[A^-1,0],[-C^-1 L21 A^-1, C^-1]], with
   batched GEMMs whose triangular operands skip their zero upper blocks three levels deep
   (`_mm_lower_left/right`; 44% fewer flops than dense products). 5.2 -> **1.86 ms**
   (B = 16: 2.63 -> 1.60 ms per chain).

**Exactness** (`bench/check_exact.py`, results in `bench/results/exact_gpu.json`): all 67 NG15
pulsars, HD and CURN, at 8 HD posterior draws, the 16 prior corners + 2 reviewer points of
`tests/test_corners.py` (where I + Phi^1/2 A Phi^1/2 reaches cond ~1e20) and 8 random prior draws,
single and vmapped. Values are compared **without the parameter-independent constant** (the full
logL ~ -8e6 has an ulp of 1.9e-9, which would make any difference below 1e-9 invisible).

| variant | max abs value diff | max gradient diff / max(abs g, 1) | criterion |
|---|---|---|---|
| hh + levels (HD) | 5.8e-11 | 9.2e-14 | 1e-9 / 1e-8: pass |
| hh + recursive (HD) | 5.8e-11 | 9.2e-14 | pass |
| prod + levels (HD) | 5.8e-11 | 1.1e-13 | pass |
| hh (CURN) | 5.8e-11 | 1.1e-13 | pass |

5.8e-11 is the ulp of the parameter-dependent part of logL (~1e5-1e6), i.e. bit-level agreement.
`tests/test_perf_likelihood.py` pins this (synthetic PTA incl. all 16 corners for CURN/HD x three
variants, the QR and tri-inverse kernels separately, and an NG15 oracle test with the same 18
corner points + prior draws).

**Speed (GPU, ms per chain-gradient)**

| variant | B = 1 | B = 4 | B = 16 | B = 64 |
|---|---|---|---|---|
| HD production | 14.4 | 9.4 | 5.8 | 5.5 |
| **HD hh + levels (exact)** | **8.5** (1.69x) | **7.2** (1.31x) | **4.4** (1.32x) | **4.0** (1.37x) |
| HD hh + levels, value only | 6.2 | | | |
| HD production, `grad_precision="mixed"` (approx. gradient) | 9.5 | 6.4 | 3.3 | 3.1 |
| HD hh + levels, `grad_precision="mixed"` (approx. gradient) | 7.0 | 5.8 | 2.9 | 2.5 |
| CURN production | 3.6 | 1.17 | 0.70 | 1.11 |
| **CURN hh (exact)** | **1.07** (3.4x) | **0.51** (2.3x) | **0.31** (2.2x) | 0.56 |

(The CPU does not benefit: the `fori_loop` Householder is a GPU optimisation; on XLA:CPU the
production LAPACK path stays faster, 5.4 vs 7.4 ms for CURN.)

What remains for exact HD at B = 1: Cholesky 4.9 ms (cuSOLVER, 0.44 TFLOP/s) + reduce 1.0 +
tri-inverse 1.9 + small kernels. At B >= 16 the evaluation is FP64-flop bound (~1.3 TFLOP/s of
the 1.9 peak).

Considered and rejected:

* **Kronecker / eigendecomposition of Gamma.** Sigma' = Gamma'^-1 (x) I_28 + blockdiag(E_a):
  the E_a differ per pulsar, so the sum has no Kronecker structure and rotating by Gamma's
  eigenvectors fills the block diagonal; the 1876 system is genuinely dense. (Kronecker
  structure would exist only for identical E_a, i.e. no intrinsic red noise.)
* **float32 Cholesky with float64 refinement for the value.** The value needs log|Sigma'| to
  ~1e-9 absolute. A float32 factor L32 has relative backward error ~6e-8 cond(Sigma'_eq); the
  correction log|I + L32^-1 (Sigma' - L32 L32^T) L32^-T| needs the float64 residual, i.e. a
  triangular product of the same n^3/3 flops as the float64 Cholesky itself. No gain.
* **`grad_precision="mixed"`** (float32 tri-inverse of the equilibrated factor; existing
  production option) cuts B = 16 cost to 2.9 ms but the gradient is approximate (max component
  error 1e-4 relative, M1 Sec. 4). It is *exact-posterior-safe* for HMC/NUTS (deterministic in
  the position, value float64), so it is a legitimate opt-in for the samplers below, but it is
  not an "exact likelihood" speedup and is listed separately.
* trsm against the identity for L^-1 (8.0 ms), cuSOLVER R-only QR on the augmented matrix
  (9.4 ms), XLA:CPU (Sec. 2).

* **`tri_inv="fused"`: recursive GEMM Cholesky that returns L and L^-1 together** (so the
  backward pass needs no inverse; L21 = S21 L11^-T via the explicit L11^-1). Exact (same
  check: 5.8e-11 / 9.2e-14) but slower: chol + inverse 10.9-12.8 ms (leaf 128-1024) vs
  potrf + `levels` 7.0 ms; value+grad 12.5 ms (B = 1) vs 8.5. cuSOLVER's potrf wins at n = 1876.
  Kept as a measured negative result (`_chol_inv_rec`).

## 4. Samplers

### Set-up (`bench/samplers.py`)

* **Target**: the production posterior `Posterior.logpost_z` (logistic box transform, float64
  energies), with the exact fast likelihood (`hh` for CURN, `hh + levels` for HD) for every
  sampler -- so ESS/gradient differences are the samplers', ESS/s differences include their
  overheads.
* **Same preconditioning for all**: production HD NUTS uses the pooled covariance C of the M2
  CURN^13/3 draws (in z) as a fixed dense metric. Every BlackJAX sampler runs in whitened
  coordinates w = L_C^-1 (z - mu) with an identity metric (identical to the dense metric for
  HMC/NUTS; it gives the diagonal-metric methods MCLMC/MAMS/ChEES/MEADS the same dense
  preconditioner). NumPyro runs through the production driver `ptagwb.sampling.run_nuts` with
  `metric=run:curn_g433_14f`, `adapt_mass_matrix=False` (the M2 HD recipe).
* **Same initialisation**: random draws of the M2 CURN^13/3 run, so warmup is tuning only.
* **Tuning** (each sampler's own): NUTS -- dual averaging of the step size only (target 0.8;
  NumPyro: its windowed schedule; BlackJAX: one continuous DA run, our 20-line driver).
  MCLMC -- `mclmc_find_L_and_step_size` (3 phases incl. the ESS-based L). MAMS =
  `adjusted_mclmc_dynamic` (random trajectory length, MH-corrected) with
  `adjusted_mclmc_find_L_and_step_size` (target accept 0.9); BlackJAX 1.7 pins L = avg x step,
  and a pilot on CURN (`pilot_curn_mams_avg*`) picked avg = 8 integration steps (ESS/kgrad min
  tail 0.42 / 1.41 / 0.58 / 0.32 for avg 2 / 8 / 16 / 32). ChEES-HMC -- `chees_adaptation`
  (Adam lr 0.025, Halton jitter; all chains share the step count, so no lockstep loss). MEADS --
  `meads_adaptation` (4 folds) + GHMC.
* **Gradient accounting**: leapfrog step = 1 gradient; isokinetic McLachlan step (MCLMC, MAMS)
  = 2 gradients. ESS: rank-normalised bulk / tail ESS (`ptagwb.diagnostics`, as in M2) of the
  constrained parameters; "common" = gw_log10_A, "min" = worst of all 135 parameters. ESS/s uses
  the sampling phase only (warmup listed separately). Thinned runs (MCLMC x10, MEADS x2) count
  all gradients, i.e. their ESS/grad is conservative.
* **Budgets**: CURN ~0.4 M chain-gradients of sampling per run, seeds 1-3 (+ seed-0 pilots);
  HD 0.08-0.13 M, one seed (GPU budget), plus the M2 production run.

### CURN^13/3 (cheap proxy), 3 seeds, mean +- sd over seeds

| sampler (chains) | ESS/s common | ESS/s min bulk | ESS/s min tail | ESS/kgrad common | ESS/kgrad min bulk | ESS/kgrad min tail | grads/draw | warmup [s] | max R-hat | exact? bias check |
|---|---|---|---|---|---|---|---|---|---|---|
| NumPyro NUTS, M2 production (4; diag. metric adapted from prior) | 1.03 | 0.12 | 0.023 | 1.73 | 0.20 | 0.039 | 92 | 660 | 1.03 | reference |
| NumPyro NUTS (4; CURN metric, step-size warmup) | 2.76 +- 0.09 | 1.32 +- 0.52 | 0.82 +- 0.73 | 1.87 +- 0.05 | 0.89 +- 0.36 | 0.56 +- 0.50 | 93 | 58 | 1.02 | yes; pass |
| **BlackJAX NUTS** (4; same) | **5.51 +- 0.53** | **3.27 +- 0.55** | **2.39 +- 1.9** | **2.76 +- 0.24** | **1.64 +- 0.27** | **1.20 +- 0.94** | 63 | 44 | 1.01 | yes; pass |
| **MAMS**, avg 8 steps (4) | **6.65 +- 0.71** | 3.34 +- 2.4 | 2.07 +- 1.7 | **3.39 +- 0.36** | 1.70 +- 1.2 | 1.05 +- 0.87 | 16 | 130-160 | 1.02 | yes; pass |
| ChEES-HMC (64) | 4.22 +- 0.48 | 2.59 +- 0.37 | 2.29 +- 0.56 | 2.38 +- 0.27 | 1.46 +- 0.21 | 1.30 +- 0.32 | 6.4 | 107 | 1.10 | yes; pass |
| MEADS (64) | 2.71 +- 2.1 | 1.92 +- 1.5 | 0.92 +- 0.74 | 1.53 +- 1.2 | 1.08 +- 0.88 | 0.52 +- 0.42 | 1 | 73 | 1.06 / **50** / 1.08 | yes; **seed 2 failed** (stuck chains, R-hat 50) |
| MCLMC, unadjusted (4) | 17.9 +- 1.6 | 9.85 +- 0.92 | 6.31 +- 4.2 | 9.07 +- 0.83 | 4.99 +- 0.47 | 3.2 +- 2.1 | 2 | 12 | 1.01 | **no; biased** (below) |

CURN wall-clock throughput per 4-chain leapfrog step (same likelihood, B = 4: 2.05 ms):
BlackJAX NUTS 1.97 ms, NumPyro 2.74 ms (NumPyro's per-step overhead is ~0.7 ms, negligible at
HD cost). ChEES/MEADS run 64 chains (B = 64, 0.56 ms per chain-gradient).

Observations:

* **NumPyro vs BlackJAX NUTS (same metric, same likelihood, same target 0.8)**: BlackJAX's
  continuous dual averaging ends at a ~25% larger step (0.054 vs 0.043 in whitened units;
  realised acceptance 0.88 vs 0.93), so trees have 63 instead of 93 steps (depth 6 instead of
  6-7) at the same ESS per draw: 1.5x ESS per gradient, and 2x ESS/s with the lower per-step
  overhead. Likely cause: NumPyro's step-size-only warmup restarts dual averaging at its window
  boundaries and ends on a short last window, which biases the step size low (the M2 doc's "acceptance
  0.92-0.94 overshoots" observation). No divergences in any NUTS run.
* **MAMS** has the best common-parameter ESS/grad and ESS/s, but its worst-parameter ESS is
  noisier (seed 3: J0610-2100 funnel, tail ESS 26). Its BlackJAX tuner is expensive: per
  gradient 4-6x slower than sampling (130-160 s on CURN, **26 min on HD**).
* **ChEES** (64 chains) is robust (smallest seed-to-seed spread of the worst-parameter ESS) but
  adapts short trajectories (~12 leapfrog steps at step 0.06) and its R-hat over 64 short chains
  is 1.10 (B1855+09 IRN amplitude). **MEADS** is unreliable here: one of three seeds left chains
  stuck (R-hat 50, posterior sd x42 on one parameter).
* **Worst-parameter (tail) ESS is dominated by rare funnel excursions** (IRN amplitudes of
  J0610-2100, J0437-4715, B1855+09, J2145-0750) for *every* exact sampler, hence the large
  seed-to-seed sd. Example: the M2 CURN reference chain 3 spent 410 draws in the low-amplitude
  tail of J2145-0750 (log10 A down to -19.7; 15% of that chain, 3.9% of the run); none of the
  28 perf runs (all initialised from posterior draws, 0.4-0.8x the reference's gradients per chain)
  entered it, so
  that tail's weight is not established by any of them (it is why the posterior-sd ratio vs
  the reference reaches 0.12 for that one parameter in every run; the quantile z-tests are
  underpowered there). A funnel-aware reparameterisation of the IRN (M2 open issue 4) is the
  lever for the worst-parameter ESS, not the sampler.

**Bias checks** (`bench/analyze_samplers.py`; vs the M2 production chains): 51 quantile tests per
run (q05/q50/q95 of gw_log10_A and 16 IRN parameters of 8 pulsars incl. the funnel ones),
z = delta q / sqrt(MCSE_run^2 + MCSE_ref^2), plus an all-parameter mean test chi^2 =
sum_j z_j^2 (135 parameters). Exact samplers: max |z| 1.7-2.9, no |z| > 3 in 51 tests, chi^2/135
= 0.6-1.3 (all NUTS, MAMS, ChEES runs; MEADS seeds 1, 3). The common amplitude quantiles of the
pooled exact runs, [-14.672, -14.565, -14.476], match M2 CURN [-14.670, -14.563, -14.474].

**Unadjusted MCLMC is biased** (as expected, flagged): all three seeds fail the mean test
(chi^2/135 = 1.7, 2.4, 2.8; max |z| 6.3-10.5, always J1713+0747's IRN amplitude). Against the
pooled exact runs: the J1713+0747 IRN-amplitude mean is off by **0.43 posterior sd**, the common
amplitude mean by **+0.15 sd** (median -14.557 vs -14.565) and its sd is 6% too small. Its
apparent 3x ESS/s advantage is therefore not usable for production; it would need the
MH-adjusted version (MAMS) or a much smaller step (energy-error tuning `desired_energy_var`),
which removes the advantage.

### HD^13/3 (target), one seed

| sampler (chains) | likelihood | ESS/s common | ESS/s min bulk | ESS/s min tail | ESS/kgrad common | ESS/kgrad min bulk | ESS/kgrad min tail | grads/draw | warmup | sampling | max R-hat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NumPyro NUTS, **M2 production run** (4) | production | 0.181 | 0.104 | 0.202 | 1.93 | 1.11 | 2.15 | 113 | 21 min | 40 min / 500 draws | 1.021 |
| **BlackJAX NUTS** (4) | hh + levels | **0.334** | **0.283** | **0.460** | 2.42 | 2.05 | 3.33 | 63 | 12 min (from step 0.25) | 9 min / 300 draws | 1.042 |
| BlackJAX NUTS (16), *projected* | hh + levels | (0.53) | (0.45) | (0.73) | (2.42) | (2.05) | (3.33) | 63 | ? | | measured run aborted (budget), see below |
| MAMS, avg 8 (4) | hh + levels | 0.282 | 0.252 | 0.104 | 2.03 | 1.82 | 0.75 | 16 | 26 min | 16 min / 2062 draws | 1.022 |
| ChEES-HMC (16), short pilot | hh + levels | (0.50) | (0.35) | (0.22) | 2.21 | 1.55 | 0.99 | 4.4 | 2 min | 1.5 min / 300 draws | **1.42: not converged** |

HD bias checks (vs M2 hd_g433_14f, 51 quantile tests + all-parameter mean test): BlackJAX NUTS
max |z| 2.6, chi^2/135 = 1.04; MAMS 2.5, 0.96; ChEES (unconverged) 3.1 with one |z| > 3 in 51 tests, chi^2/135 = 0.73. The HD runs are short (one seed), so
their ESS/s carry ~+-30% (bulk) to ~+-100% (worst tail) uncertainty judging from the CURN
seed-to-seed spread.

The 16-chain NUTS row is a projection: per-chain-gradient cost 4.38 ms at B = 16 vs 7.18 ms at
B = 4 (Sec. 3) times the lockstep efficiency estimated by resampling the measured HD tree sizes
(99% of trees have 63 leapfrog steps; efficiency 0.99 for 4 chains, 0.955 for 16, 0.84 for 64).
A measured 16-chain HD run (100 warmup from step 0.05 + 150 draws) was **aborted after 52 min**
without finishing, against ~30 min projected; it was stopped to stay inside the GPU budget, so
16-chain NUTS on HD (most likely its warmup, where one chain with a 1023-step tree stalls all 16)
is unverified and should be piloted before use.

### Not run

* **flowMC** (normalising-flow MCMC) and **nested sampling** (BlackJAX `nss`, jaxns) for the
  HD vs CURN evidence: a 135-dim nested-sampling run needs ~1e7-1e8 likelihood calls (500-1000
  live points x ~D slice steps x ~50 e-folds); at the measured 2.0 ms per batched HD value that is
  6-60 GPU hours, beyond the budget and far above the ~1 min of the M2 reweighting/bridge
  estimators (BF 178-228). flowMC's flow training on a 135-dim funnel-shaped posterior needs
  many global-proposal acceptances to pay off; with ESS/s already limited by IRN funnels it was
  deprioritised. Both are worth revisiting only for the product-space BF question (M2 open
  issue 2).

## 5. Recommendation

**Production configuration: GPU x `FastPTALikelihood(reduce="hh", tri_inv="levels")` (exact,
float64) x BlackJAX NUTS, 4 vectorised chains, fixed dense metric from the CURN run (whitened
coordinates), one continuous dual-averaging step-size warmup started from the CURN-tuned step
size.** Everything else in the M2 recipe stays (CURN run first for init + metric, target accept
0.8, max tree depth 10).

| | M2 production (measured) | recommended (measured HD run, extrapolated to equal ESS) |
|---|---|---|
| per-chain gradient (B = 4) | 9.4 ms | 7.2 ms (exact, bit-level agreement) |
| leapfrog steps / draw | 113 | 63 |
| ESS/s common / min bulk / min tail | 0.18 / 0.10 / 0.20 | 0.33 / 0.28 / 0.46 |
| sampling time to the M2 HD ESS (common bulk 435, min bulk 250) | 40 min | **~22 min** (435 / 0.334 s^-1) |
| warmup | 21 min (incl. compile) | 12 min measured from step 0.25; ~3-5 min expected from the CURN step size (0.054 vs HD 0.051 in whitened units; 100 DA iterations x 1.8 s), not measured |
| **HD^13/3 run at equal ESS** | **~61 min** | **~25-35 min (1.8-2.4x)** |

Why not the others: MAMS has the best common-parameter ESS/grad on CURN but its BlackJAX tuner
costs 26 min on HD and its worst-parameter ESS is erratic; on HD it was below NUTS on every
metric. ChEES/MEADS only pay with many chains, and at B = 16-64 the HD gradient is already
FP64-flop bound (4.0-4.4 ms per chain, only 1.6-1.8x cheaper than B = 4), which does not make up
for ChEES' short adapted trajectories and slow many-chain convergence (R-hat 1.10 on CURN with 64
chains, 1.42 on the HD pilot); MEADS failed on 1 of 3 CURN seeds. Unadjusted MCLMC is biased
(0.43 sd on an IRN mean, 0.15 sd on the common amplitude). CPU is 3-7x slower than the GPU and
also below it at the LAPACK ceiling.

Further options, in order of expected gain:

1. **IRN reparameterisation** (funnel-aware / non-centred). Every exact sampler's worst-parameter
   ESS is set by rare funnel excursions (J0610-2100, J0437-4715, B1855+09, J2145-0750), and the
   J2145-0750 low-amplitude tail is not established by any short run. This is the largest
   remaining lever and a correctness issue for tail quantiles, independent of speed.
2. `grad_precision="mixed"` on top of `hh + levels`: 5.8 vs 7.2 ms at B = 4 (2.9 vs 4.4 at
   B = 16). Exact posterior for NUTS (deterministic gradient, float64 energies), but an
   approximate gradient; opt-in only.
3. 16 vectorised NUTS chains: projected 1.6x more ESS/s (0.53 / 0.45 / 0.73), but the measured
   16-chain run stalled (Sec. 4); pilot first.

**What it enables** (GPU, at the recommended config):

* Full HD^13/3 posterior runs (e.g. one per sky-scrambled or phase-shifted ORF, re-sampled
  from scratch at M2-level ESS): ~30 min each, **~40-55 per day** (M2 production: ~24 per day).
* Null-distribution Bayes factors by **reweighting** the 6,000 CURN draws (as M2's reweighting
  estimator): value-only HD logL costs 2.0 ms per draw batched (production 2.6 ms;
  `bench/results/value_only_hd_gpu.json`), i.e. ~12 s per scrambled realisation, **~7,000 per
  day** (production ~5,500), valid where the scrambled-HD/CURN weights keep enough ESS.
* Optimal-statistic phase-shift / sky-scramble nulls need no new likelihood evaluations (seconds
  per realisation over 6,000 draws, M2 Sec. 9): ~10^4+ per day either way.


## 6. Reproduction

Worktree set-up (data and M2 runs are symlinked from the main checkout; nothing is written there):

```bash
git worktree add ../pta-gwb-jax-perf -b perf-bench main && cd ../pta-gwb-jax-perf
for d in raw cache processed; do ln -s ../../pta-gwb-jax/data/$d data/$d; done
mkdir runs && for r in curn_g433_14f hd_g433_14f; do ln -s ../../pta-gwb-jax/runs/$r runs/$r; done
scripts/setup_oracle_env.sh   # or: uv sync --group oracle --group perf (with the SuiteSparse env of that script)
uv sync --group oracle --group perf
```

All commands below: `uv run --no-sync python ...`; GPU unless `JAX_PLATFORMS=cpu`. The first call
builds `bench/cache/terms_*.pkl` (stage-1 precompute, ~1-4 min).

```bash
# Q1 profile
python bench/profile_hd.py                                   # -> bench/results/profile_gpu.json
python bench/profile_hd.py --jax-trace 20 && python bench/parse_trace.py bench/traces/jax_hd_gpu 20
python bench/trace_batched.py hd hh+levels 16 && python bench/parse_trace.py bench/traces/batched_hd_hh+levels_float64_16 10
# Q2 backends / batching
python bench/bench_backends.py --orf hd --batch 1 4 16 64 --variant prod
python bench/bench_backends.py --orf curn --batch 1 4 16 64 256 --variant prod
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=16 python bench/bench_backends.py --orf hd --batch 1 4 --variant prod --tag _ob16
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 XLA_FLAGS=--xla_force_host_platform_device_count=16 \
    python bench/bench_backends.py --orf hd --batch 16 --variant prod --pmap --tag _pmap16_ob1_B16
# Q3 exact variants: exactness and speed
python bench/check_exact.py                                  # -> bench/results/exact_gpu.json
python bench/bench_backends.py --orf hd --batch 1 4 16 64 --variant hh+levels --tag _v2
python bench/bench_backends.py --orf hd --batch 1 4 16 64 --variant prod hh+levels --grad-precision mixed --tag _v2_mixed
python bench/bench_backends.py --orf curn --batch 1 4 16 64 256 --variant hh --tag _v1
uv run --no-sync pytest tests/test_perf_likelihood.py
# Q4 samplers (one line per run; seeds 1-3 for CURN, seed 1 for HD; CURN pilots: --seed 0 --tag pilot_curn_<sampler>,
# MAMS pilots with --opt avg_steps=2/8/16/32)
python bench/samplers.py --model curn --like hh --seed 1 --sampler numpyro --chains 4 --warmup 150 --samples 1000
python bench/samplers.py --model curn --like hh --seed 1 --sampler bj_nuts --chains 4 --warmup 150 --samples 1000
python bench/samplers.py --model curn --like hh --seed 1 --sampler mams --chains 4 --warmup 1000 --samples 6250 --opt avg_steps=8
python bench/samplers.py --model curn --like hh --seed 1 --sampler chees --chains 64 --warmup 600 --samples 700
python bench/samplers.py --model curn --like hh --seed 1 --sampler mclmc --chains 4 --warmup 3000 --samples 50000 --thin 10
python bench/samplers.py --model curn --like hh --seed 1 --sampler meads --chains 64 --warmup 2000 --samples 6250 --thin 2
python bench/samplers.py --model hd --like hh+levels --seed 1 --sampler bj_nuts --chains 4 --warmup 150 --samples 300
python bench/samplers.py --model hd --like hh+levels --seed 1 --sampler mams --chains 4 --warmup 600 --samples 2062 --opt avg_steps=8
python bench/samplers.py --model hd --like hh+levels --seed 1 --sampler chees --chains 16 --warmup 400 --samples 300
# aborted (budget): --model hd --seed 2 --sampler bj_nuts --chains 16 --warmup 100 --samples 150 --opt init_step=0.05
python bench/bench_value_only.py
JAX_PLATFORMS=cpu python bench/m2_reference.py               # M2 production runs in the same format
JAX_PLATFORMS=cpu python bench/analyze_samplers.py           # bias checks + bench/results/samplers_summary.json
```

Raw results: `bench/results/*.json` (throughput, profile, exactness), `bench/results/samplers/*.json`
(one per sampler run, incl. bias checks), `bench/results/samplers_summary.json` (aggregated).
Draws are in `bench/runs/*.npz` (git-ignored, ~5-50 MB each).
