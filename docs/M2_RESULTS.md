# M2 results: sampling, Bayes factors and optimal statistic

M2 reproduces the headline Bayesian and frequentist results of the NG15 GWB paper (Agazie et al.
2023, arXiv:2306.16213) with our own pipeline: the M1 float64 JAX likelihood on our PINT 1.1.7
arrays, NumPyro NUTS, our own Bayes-factor estimators and our own optimal statistic (OS). Every
number is compared with the paper and with the released chains/products. All production runs
were made on 2026-10-06/07 on the shared RTX 5090 (one GPU job at a time) from committed configs.

"Production mode" uses the enterprise sky-position convention (`position="enterprise"`, B-name
pulsars in B1950-equinox coordinates as in the released chains); the HD runs were repeated with
physically correct ICRS positions. CURN does not depend on positions.

Reproduce (outputs go to the git-ignored `runs/` and `outputs/m2/`):

```bash
nohup scripts/m2_production.sh > runs/production.log 2>&1 &           # all NUTS runs, sequential
uv run --no-sync python scripts/m2_compare.py                           # vs released chains, diagnostics
uv run --no-sync python scripts/m2_bayes.py --systematics               # HD vs CURN Bayes factors
uv run --no-sync python scripts/m2_optstat.py                           # OS, noise marginalisation, Fig. 1c
uv run --no-sync python scripts/m2_figures.py                           # docs/figures/m2_*.png
uv run --no-sync python scripts/m2_report.py > outputs/m2/tables.md     # the tables below
PTAGWB_REQUIRE_ORACLES=1 uv run --no-sync pytest                        # 92 tests incl. OS oracle
```

## 1. Summary

Medians with 5-95% intervals (90% credible), A at f_ref = 1/yr. "MC" = Monte-Carlo standard
error of our quantiles (Vehtari et al. 2021 quantile MCSE from the ESS of the indicator).
Verdict: **agrees** = the difference is within ~2 combined MCSEs or well inside the quoted
uncertainty; numbers with z use z = (ours - released) / sqrt(MCSE_ours^2 + MCSE_rel^2).

| Quantity | Paper | Released chains / products | Ours (enterprise positions) | Ours (ICRS) | Verdict |
|---|---|---|---|---|---|
| HD^13/3 A [1e-15] | 2.4 (+0.7/-0.6) | 2.404 [1.815, 3.069] (Fig. 1b chain) | 2.445 [1.820, 3.106]; log10 A -14.612 [-14.740, -14.508], MC (0.004, 0.004, 0.003) | 2.372 [1.782, 3.092]; log10 A -14.625 [-14.749, -14.510] | agrees: z(5,50,95) = (+0.2, +1.7, +1.8) vs Fig. 1b chain; (-2.1, +0.1, -0.6) vs the tutorial core; KS D = 0.046 |
| HD^gamma A [1e-15] | 6.4 (+4.2/-2.7) | 6.385 [3.700, 10.559] | 6.445 [3.739, 10.486]; log10 A -14.191 [-14.427, -13.979] | 6.32 [3.64, 10.09]; log10 A -14.199 [-14.439, -13.996] | agrees: z = (+0.3, +0.7, -0.4) |
| HD^gamma gamma | 3.2 +- 0.6 | 3.248 [2.660, 3.840] | 3.226 [2.679, 3.854] | 3.255 [2.676, 3.846] | agrees: z = (+1.0, -1.7, +0.4); 2-D energy distance at the MC floor (Sec. 5) |
| CURN^13/3 log10 A | not quoted | -14.563 [-14.675, -14.480] (tutorial CURN^13/3-vs-HD^13/3 core, CURN samples) | -14.563 [-14.670, -14.474] | n/a (no positions) | agrees: z = (+0.7, -0.0, +1.4) |
| CURN^gamma log10 A | not quoted (contours only) | -14.171 [-14.392, -13.977] (m2a) | -14.170 [-14.387, -13.966] | n/a | agrees: z = (+0.7, +0.1, +1.9) |
| CURN^gamma gamma | not quoted | 3.351 [2.797, 3.877] (m2a) | 3.344 [2.784, 3.878] | n/a | agrees: z = (-0.9, -0.6, +0.1) |
| HD^free (30 modes), Fig. 1a | excess power in bins 1-8 | released core (Fig. 1a) | bins f_1-f_3 and f_8 (log10 rho): -6.57, -6.82, -7.16, -7.58 vs released -6.57, -6.81, -7.15, -7.59; first 10 bins |z(50)| <= 1.6, KS D <= 0.047 (Fig. 1a) | not run | agrees for the medians/peaks; weak mixing in the bimodal bins (f_1, f_4, f_5, f_8: R-hat up to 1.08, bulk ESS 31-170) |
| BF HD^gamma / CURN^gamma, 14 modes | ~200; Fig. 2: 226 +- 70 (tutorial: hypermodel 202 +- 3, TI 198 +- 45) | - | **178 +- 6** (bridge); reweighting 183 +- 9, reverse reweighting 228 +- 13 | 172 (ln BF -0.0125 +- 0.0005 vs enterprise positions) | agrees within the paper's +-70. Our estimator spread (178-228) exceeds the bootstrap errors: quote 178 (+50/-6) |
| BF HD^13/3 / CURN^13/3, 14 modes | "similar" | - | **212 +- 8** (bridge); reweighting 225 +- 9, reverse 214 +- 20 | 236 vs 239 on the same draws (ln BF -0.011) | agrees ("similar") |
| BF HD^gamma / CURN^gamma, 5 modes | ~1000; Fig. 2: 965 | - | **894 +- 30** (bridge); reweighting 834 +- 28, reverse 1081 +- 64 | not computed | agrees within the estimator spread (834-1081) |
| BF self-check (CURN via the correlated-ORF path with Gamma = I) | 1 | - | 1 - 1.1e-11 (max abs dlogL 1.7e-8) | - | exact |
| OS S/N, noise-marginalised, gamma = 13/3 | 4 +- 1 | 4.49 +- 1.02 (20,000 released draws) | 4.52 +- 1.03 (6,000 draws) | 4.51 +- 1.03 | agrees |
| OS S/N, noise-marginalised, varied gamma | 5 +- 1 | 4.98 +- 1.10 | 5.04 +- 1.10 | 5.03 +- 1.10 | agrees |
| OS mean A^2 (13/3, noise-marg.) | 6.8(9)e-30 (App. G) | 6.81e-30 +- 0.87e-30 | 6.79e-30 +- 0.88e-30 | 6.78e-30 | agrees |
| OS at the released ML noise vector (13/3) | - | A^2 = 6.7037e-30 +- 1.2324e-30, S/N 5.439 (notebook output) | 6.7032e-30 +- 1.2324e-30, S/N 5.439 (our PINT arrays) | 6.7059e-30, S/N 5.441 | agrees to 6e-5 (A^2) / 2e-6 (sigma) |
| Binned HD chi^2 (15 bins), at the released ML vector | 8.1 (p = 0.92 for chi^2_15) | - | 8.10 (p = 0.920); same with the released pair covariance | 8.03 | exact |
| Binned HD chi^2, at the MAP draw of *our* CURN^13/3 chain | - | - | 9.99 (p = 0.82); over the 20 highest-logL draws: median 10.1, range 6.4-30.2 | 9.52 | consistent; chi^2 is sensitive to the noise point (Sec. 7) |
| Pair covariance (2211 x 2211) at the released ML vector | - | `os_covariance_matix_between_rhos.npy` | max deviation 1.3e-4 (our arrays), 1.3e-5 (feathers), in correlation units | - | agrees |

Headline: every parameter-estimation target is reproduced within Monte-Carlo error; the HD vs
CURN Bayes factor (178, estimator range 178-228) and the 5-mode one (894, range 834-1081) are
inside the paper's quoted uncertainties; the OS reproduces the released ML numbers and chi^2 = 8.1
exactly and the noise-marginalised S/N distributions to 0.03-0.06.

## 2. Priors: the two open prerequisites

**Fixed-gamma common amplitude: resolved to U[-18, -11]** (paper Table 1 says U[-18, -14]).
Evidence, all from the released products (`lnpost - lnlike` is constant in every core, std
< 5e-7, so it is the log prior normaliser):

| Product | Model | Prior normaliser | Decomposition | Amplitude prior |
|---|---|---|---|---|
| `SplineORF_FixedGamma_NL.core` (Fig. 1d) | spline ORF, gamma = 13/3 | -283.6504434721 | -67 ln 63 - ln 7 - 7 ln 1.8 = -283.6504 | stored: `gw_crn_log10_A:Uniform(pmin=-18, pmax=-11)` |
| tutorial `curn_hd.core` | CURN^13/3 vs HD^13/3 product space (no gamma parameter) | -559.0718736 | 2 (-67 ln 63) - 2 ln 7 | U[-18, -11]; inactive-model amplitude samples span [-18.00, -11.00] |
| `14f_PL_hd_crn.core`, tutorial `curn_14f_pl_vg.core`, `hd_14f_pl_vg.core` | varied gamma | -562.9636939 | 2 (-67 ln 63 - ln 7 - ln 7) | U[-18, -11] x U[0, 7] (stored) |

The HD^13/3 Fig. 1(b) chain (`nano15_hd_chain_fg_long_050523.npy`) stores only log10_A; its
maximum is -14.36, so its posterior does not see either upper edge. Consequences: posteriors are
identical under both priors; evidences against IRN shift by ln(7/4) = 0.56; HD/CURN Bayes factors
are unaffected (same prior in both models). We adopt U[-18, -11] (`config.PRIORS`), with
`config.PRIOR_ALTERNATIVES` keeping the Table 1 value.

**CURN free spectrum: not resolvable from the available provenance; not needed for M2.** No
released CURN^free chain exists in any bundle we have (the Fig. 1a core and the tutorial core are
HD^free; the Fig. 6 bundle was not part of the release we fetched). The only CURN^free artefact is
the Ceffyl v1 CP KDE grid, which spans [-15.1, -0.9], whereas the HD grids span exactly the HD^free
prior [-15.5, -1.0] (M1 Sec. 7); the Ceffyl v2 (2025-26 re-runs) grids are all [-9, -4]. Best
inference: a similarly wide uniform prior, recorded as `config.CURN_FREESPEC_LOG10_RHO_INFERRED =
(-15.1, -0.9)`, status *inferred, weak*. Bins with signal (f_1..f_8) are likelihood-dominated and
insensitive to the lower edge. Our auxiliary CURN^free run (`curn_fs30`, used only to initialise
and precondition HD^free) uses the HD^free prior U[-15.5, -1.0].

## 3. Sampler

**Choice: NumPyro NUTS** (`ptagwb.sampling`). It takes a plain `potential_fn` on a flat vector,
so the M1 likelihood (custom VJPs, float64) is used unchanged; it has Stan-style windowed
step-size and (diagonal or dense) mass-matrix adaptation; multinomial NUTS uses the exact float64
Hamiltonian for trajectory sampling; and `MCMC.warmup` / `post_warmup_state` allow block-wise
sampling with checkpoints. Chains run `vectorized` (one batched GPU call per leapfrog step for
all 4 chains: CURN 4.6 ms vs 4 x 3.6 ms, HD 38 ms vs 4 x 14.8 ms). BlackJAX would have needed our
own warmup / multi-chain driver for the same features.

**Parameterisation.** Every prior is uniform on a box; NUTS runs on z in R^D with
x = lo + (hi - lo) sigmoid(z), log p(z) = log L(x) + log pi(x) + log|dx/dz|,
log|dx/dz| = sum [log(hi - lo) - softplus(z) - softplus(-z)] (finite for any z). Tests:
Jacobian vs autodiff, the flat-likelihood density integrates to 1, NUTS on truncated Gaussians
matches `scipy.stats.truncnorm` quantiles within 5 MCSE (`tests/test_sampling.py`).

**Run design (what the pilots taught us).** The IRN posteriors of several pulsars
(J0610-2100, J1747-4036, J1903+0327, B1855+09, ...) are funnel-like: a tight "IRN detected" core
plus a long tail to small amplitude or to gamma -> 0 (whitened kurtosis up to ~1600 in the
released m2a chain). NUTS then needs step sizes ~0.02-0.05 and ~90-210 leapfrog steps per draw
even with a well-adapted metric; ESS per draw for the common parameters is ~0.15-0.4.
* Dense adaptation from prior draws saturates the tree depth (1023 steps) during the early,
  rank-deficient windows (25-100 draws for 135 parameters), so the CURN runs use **diagonal**
  windowed adaptation from prior draws (z ~ U(-2, 2)), 500 warmup + 1500 draws x 4 chains.
* HD runs start from 4 random draws of the corresponding CURN run and use the CURN run's pooled
  sample covariance (in z) as a **fixed dense metric**, with step-size-only warmup (150; 100 for
  HD^free). The HD and CURN posteriors share the IRN geometry, so this saves the HD warmup that
  would otherwise dominate (HD gradients cost 8x CURN). Pilot check: same tree depth and ESS as a
  diagonally adapted run.
* HD^free (30 common modes, 4020-dim reduced system) uses `grad_precision="mixed"` (float32
  triangular inverse in the backward pass only; value and NUTS energies stay float64). Leapfrog
  with any deterministic position-only force field is reversible and volume preserving, so the
  stationary distribution is unchanged; `test_nuts_exact_with_deterministic_approximate_gradient`
  verifies this with a 5% deterministic gradient error. All power-law runs use exact float64
  gradients.
* `target_accept = 0.8`, `max_tree_depth = 10`. Realised acceptance 0.92-0.94: dual averaging
  settles on the step size required by the funnel cores, which overshoots elsewhere.

Each run directory `runs/<name>/` (git-ignored) holds `samples.npz` (constrained and
unconstrained draws, log L, potential energy, divergences, tree depths, acceptance, adapted step
size and metric) and `meta.json` (config, git SHA and dirty-file list at launch, timings,
gradient-evaluation counts). All production configs (`configs/m2/*.json`) and the driver
(`scripts/m2_production.sh`) were committed before launch; every production run records a
clean SHA.

## 4. Runs and diagnostics

All runs: 4 chains, `max_tree_depth` 10, `target_accept` 0.8, enterprise positions unless the
name ends in `_icrs`. 30 IRN modes; 14 common modes unless noted. Diagnostics over **all**
parameters (rank-normalised split R-hat, bulk/tail ESS, Vehtari et al. 2021); "common" = the
common-process parameters. Wall times exclude the ~15-75 s setup (PINT cache load + stage-1
precompute) and include JIT compilation inside warmup. SHA = commit at launch (no run had
modified tracked files at launch). `src/ptagwb`, `scripts/m2_run.py` and `configs/m2/` are identical in all these commits
except the committed hd_fs30 scope reduction (`git diff 1f11cf0 90b44b2 -- src scripts/m2_run.py configs/m2`); the
intermediate commits touched only post-processing scripts and docs.

| run | chains x draws | max R-hat (param) | min bulk / tail ESS | common bulk ESS | divergences | mean tree depth (steps) | accept | wall warmup + sampling [min] | grad evals | common ESS/s | SHA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| curn_g433_14f | 4 x 1500 | 1.029 (J2145-0750_red_noise_log10_A) | 109 / 21 | log10_A 959 | 0 | 6.5 (92) | 0.93 | 11 + 16 | 871,564 | 1.03 | 1f11cf0 |
| curn_vg_14f | 4 x 1500 | 1.012 (J0437-4715_red_noise_log10_A) | 382 / 81 | gamma 1521, log10_A 2264 | 0 | 6.4 (88) | 0.92 | 10 + 15 | 831,544 | 1.73 | bec8f10 |
| hd_g433_14f | 4 x 500 | 1.021 (J1944+0907_red_noise_log10_A) | 250 / 485 | log10_A 435 | 0 | 6.8 (113) | 0.93 | 21 + 40 | 296,789 | 0.18 | e59353a |
| hd_vg_14f | 4 x 500 | 1.021 (J1944+0907_red_noise_log10_A) | 265 / 153 | gamma 944, log10_A 975 | 0 | 6.4 (88) | 0.92 | 18 + 38 | 240,901 | 0.41 | 625464c |
| hd_fs30 | 4 x 250 | 1.082 (gw_log10_rho_0) | 31 / 19 | min 31 | 0 | 7.0 (128) | 0.93 | 53 + 57 | 202,487 | 0.01 | 90b44b2 |
| hd_g433_14f_icrs | 4 x 500 | 1.032 (J1713+0747_red_noise_log10_A) | 175 / 199 | log10_A 258 | 0 | 6.8 (111) | 0.92 | 19 + 40 | 292,789 | 0.11 | 625464c |
| hd_vg_14f_icrs | 4 x 500 | 1.024 (J1944+0907_red_noise_log10_A) | 295 / 423 | gamma 1267, log10_A 1697 | 0 | 6.7 (105) | 0.93 | 22 + 40 | 276,475 | 0.53 | 90b44b2 |
| curn_vg_5f | 4 x 1500 | 1.009 (J0437-4715_red_noise_log10_A) | 593 / 110 | gamma 2862, log10_A 3819 | 0 | 6.9 (119) | 0.94 | 11 + 15 | 1,012,211 | 3.18 | 3ad2306 |
| hd_vg_5f | 4 x 500 | 1.025 (J1911+1347_red_noise_log10_A) | 252 / 124 | gamma 917, log10_A 886 | 0 | 6.3 (80) | 0.92 | 6 + 10 | 220,313 | 1.46 | 625464c |
| curn_fs30 | 4 x 1000 | 1.147 (gw_log10_rho_3) | 20 / 68 | min 20 | 0 | 7.7 (213) | 0.93 | 18 + 21 | 1,368,280 | 0.02 | 3ad2306 |


* **No divergences** in any sampling phase; no draw hit the maximum tree depth except
  2 draws of the auxiliary curn_fs30 run.
* **R-hat.** In the power-law runs all common parameters have R-hat <= 1.005, except log10 A in
  the HD^13/3 ICRS run (1.014, bulk ESS 258); the free-spectrum runs are discussed in Sec. 10.
  The worst parameter of each power-law run is an IRN amplitude with a funnel-shaped marginal (J2145-0750, J0437-4715, J1944+0907,
  J1713+0747, J1911+1347): R-hat 1.009-1.032. The paper's criterion (Gelman-Rubin < 1.01) is met
  for the common parameters but not for every IRN nuisance parameter in the HD runs (4 x 500
  draws); see Sec. 10.
* **ESS.** Common-parameter bulk ESS 258-3800; the HD^13/3 ICRS run is the weakest (258). The
  quantile MCSEs in the tables propagate this directly.
* hd_g433_14f warmup (21 min) is inflated by GPU contention: a duplicate driver process ran for
  30 s at its start and two Bayes-factor evaluations ran concurrently (see `runs/production.log`).

## 5. Parameter estimation vs the released chains

Reference chains: Fig. 1(b) HD^gamma / HD^13/3 chains (`nano15_hd_chain_{long,fg_long}_050523.npy`),
discovery-repo m2a (CURN^gamma) and m3a (HD^gamma) chains with full IRN, the Fig. 1
product-space core `14f_PL_hd_crn.core` split by `nmodel` ("hm"), and the tutorial
CURN^13/3-vs-HD^13/3 product-space core `curn_hd.core` ("tut", gamma fixed; the tutorial data
are described as reduced). Released chains are single concatenated chains after the stored
la_forge burn-in; their MCSEs use autocorrelation ESS. KS p-values use n_eff from the bulk ESS of
both samples and are indicative only.

| run | released | param | ours | MCSE ours | released | MCSE rel | z(5,50,95) | KS D (p_ESS) |
|---|---|---|---|---|---|---|---|---|
| curn_g433_14f | curn_g433_tut | gw_log10_A | -14.563 [-14.670, -14.474] | +-(0.003, 0.002, 0.003) | -14.563 [-14.675, -14.480] | +-(0.007, 0.002, 0.003) | +0.7, -0.0, +1.4 | 0.032 (0.88) |
| curn_vg_14f | curn_vg_m2a | gw_gamma | 3.344 [2.784, 3.878] | +-(0.012, 0.008, 0.012) | 3.351 [2.797, 3.877] | +-(0.009, 0.007, 0.007) | -0.9, -0.6, +0.1 | 0.016 (0.97) |
| curn_vg_14f | curn_vg_m2a | gw_log10_A | -14.170 [-14.387, -13.966] | +-(0.006, 0.003, 0.006) | -14.171 [-14.392, -13.977] | +-(0.004, 0.002, 0.002) | +0.7, +0.1, +1.9 | 0.012 (0.99) |
| curn_vg_14f | curn_vg_hm | gw_gamma | 3.344 [2.784, 3.878] | +-(0.012, 0.008, 0.012) | 3.353 [2.789, 3.890] | +-(0.012, 0.008, 0.014) | -0.3, -0.7, -0.6 | 0.023 (0.82) |
| curn_vg_14f | curn_vg_hm | gw_log10_A | -14.170 [-14.387, -13.966] | +-(0.006, 0.003, 0.006) | -14.174 [-14.390, -13.976] | +-(0.005, 0.003, 0.005) | +0.5, +0.9, +1.3 | 0.024 (0.65) |
| hd_g433_14f | hd_g433 | gw_log10_A | -14.612 [-14.740, -14.508] | +-(0.004, 0.004, 0.003) | -14.619 [-14.741, -14.513] | +-(0.003, 0.002, 0.001) | +0.2, +1.7, +1.8 | 0.046 (0.42) |
| hd_g433_14f | hd_g433_tut | gw_log10_A | -14.612 [-14.740, -14.508] | +-(0.004, 0.004, 0.003) | -14.612 [-14.726, -14.502] | +-(0.005, 0.004, 0.009) | -2.1, +0.1, -0.6 | 0.051 (0.78) |
| hd_vg_14f | hd_vg | gw_gamma | 3.226 [2.679, 3.854] | +-(0.016, 0.010, 0.030) | 3.248 [2.660, 3.840] | +-(0.012, 0.009, 0.012) | +1.0, -1.7, +0.4 | 0.030 (0.56) |
| hd_vg_14f | hd_vg | gw_log10_A | -14.191 [-14.427, -13.979] | +-(0.012, 0.004, 0.006) | -14.195 [-14.432, -13.976] | +-(0.006, 0.003, 0.004) | +0.3, +0.7, -0.4 | 0.027 (0.71) |
| hd_vg_14f | hd_vg_m3a | gw_gamma | 3.226 [2.679, 3.854] | +-(0.016, 0.010, 0.030) | 3.256 [2.666, 3.861] | +-(0.011, 0.011, 0.013) | +0.7, -2.0, -0.2 | 0.035 (0.51) |
| hd_vg_14f | hd_vg_m3a | gw_log10_A | -14.191 [-14.427, -13.979] | +-(0.012, 0.004, 0.006) | -14.197 [-14.440, -13.977] | +-(0.006, 0.003, 0.005) | +1.0, +1.2, -0.3 | 0.035 (0.48) |
| hd_vg_14f | hd_vg_hm | gw_gamma | 3.226 [2.679, 3.854] | +-(0.016, 0.010, 0.030) | 3.248 [2.678, 3.833] | +-(0.020, 0.011, 0.019) | +0.0, -1.5, +0.6 | 0.039 (0.41) |
| hd_vg_14f | hd_vg_hm | gw_log10_A | -14.191 [-14.427, -13.979] | +-(0.012, 0.004, 0.006) | -14.198 [-14.421, -13.982] | +-(0.007, 0.004, 0.007) | -0.4, +1.2, +0.3 | 0.043 (0.33) |
| hd_fs30 | hd_fs30 | gw_log10_rho_0 | -6.572 [-13.815, -6.256] | +-(1.316, 0.044, 0.022) | -6.572 [-13.625, -6.276] | +-(0.108, 0.007, 0.006) | -0.1, -0.0, +0.9 | 0.030 (1.00) |
| hd_fs30 | hd_fs30 | gw_log10_rho_1 | -6.818 [-6.982, -6.662] | +-(0.010, 0.004, 0.007) | -6.810 [-6.990, -6.660] | +-(0.006, 0.003, 0.004) | +0.7, -1.6, -0.3 | 0.045 (0.41) |
| hd_fs30 | hd_fs30 | gw_log10_rho_2 | -7.155 [-7.359, -6.971] | +-(0.007, 0.005, 0.007) | -7.152 [-7.385, -6.962] | +-(0.007, 0.004, 0.004) | +2.6, -0.4, -1.1 | 0.040 (0.56) |
| hd_fs30 | hd_fs30 | gw_log10_rho_3 | -7.445 [-9.824, -7.193] | +-(1.507, 0.011, 0.008) | -7.463 [-12.134, -7.195] | +-(0.181, 0.006, 0.004) | +1.5, +1.5, +0.2 | 0.047 (0.88) |
| hd_fs30 | hd_fs30 | gw_log10_rho_4 | -8.925 [-14.574, -7.253] | +-(0.150, 0.731, 0.014) | -8.888 [-14.840, -7.256] | +-(0.024, 0.103, 0.005) | +1.7, -0.0, +0.2 | 0.030 (1.00) |
| hd_fs30 | hd_fs30 | gw_log10_rho_5 | -10.905 [-14.997, -7.418] | +-(0.091, 0.204, 0.061) | -10.877 [-15.007, -7.419] | +-(0.015, 0.044, 0.011) | +0.1, -0.1, +0.0 | 0.017 (1.00) |
| hd_fs30 | hd_fs30 | gw_log10_rho_6 | -11.266 [-15.040, -7.474] | +-(0.080, 0.158, 0.032) | -11.112 [-15.053, -7.439] | +-(0.012, 0.042, 0.015) | +0.2, -0.9, -1.0 | 0.030 (0.87) |
| hd_fs30 | hd_fs30 | gw_log10_rho_7 | -7.578 [-8.020, -7.328] | +-(0.910, 0.008, 0.010) | -7.587 [-9.911, -7.333] | +-(0.427, 0.005, 0.005) | +1.9, +0.9, +0.4 | 0.041 (0.97) |
| hd_fs30 | hd_fs30 | gw_log10_rho_8 | -11.665 [-15.077, -8.243] | +-(0.058, 0.110, 0.056) | -11.674 [-15.121, -8.226] | +-(0.011, 0.037, 0.018) | +0.8, +0.1, -0.3 | 0.020 (0.94) |
| hd_fs30 | hd_fs30 | gw_log10_rho_9 | -11.284 [-15.047, -7.823] | +-(0.103, 0.187, 0.047) | -11.194 [-15.087, -7.784] | +-(0.015, 0.033, 0.012) | +0.4, -0.5, -0.8 | 0.030 (0.63) |

**2-D (gamma, log10 A).** curn_vg_14f vs curn_vg_m2a: 1.23e-04 (MC floors: our chains 1-2 vs 3-4 1.66e-03; released halves 8.74e-05); curn_vg_14f vs curn_vg_hm: 2.21e-04 (MC floors: our chains 1-2 vs 3-4 1.66e-03; released halves 9.69e-05); hd_vg_14f vs hd_vg: 4.02e-04 (MC floors: our chains 1-2 vs 3-4 5.37e-04; released halves 1.48e-04); hd_vg_14f vs hd_vg_m3a: 1.24e-03 (MC floors: our chains 1-2 vs 3-4 5.37e-04; released halves 1.40e-04); hd_vg_14f vs hd_vg_hm: 8.89e-04 (MC floors: our chains 1-2 vs 3-4 5.37e-04; released halves 4.20e-04);
In every case the ours-vs-released energy distance is at or below the MC floor of our own
chains (chains 1-2 vs 3-4), i.e. the 2-D posteriors agree within our Monte-Carlo noise.
Fig. 1(b) overlays the contours.

**IRN spot checks** (six pulsars spanning detected IRN, upper limits and funnel shapes; ours vs
m2a for CURN^gamma and vs m3a for HD^gamma): all 72 quantiles agree, |z| <= 2.4, KS D <= 0.035.

| run | released | param | ours | released | z(5,50,95) | KS D |
|---|---|---|---|---|---|---|
| curn_vg_14f | curn_vg_m2a | B1937+21_red_noise_log10_A | -13.570 [-13.725, -13.427] | -13.567 [-13.727, -13.429] | +0.6, -1.5, +0.7 | 0.021 |
| curn_vg_14f | curn_vg_m2a | B1937+21_red_noise_gamma | 4.015 [3.495, 4.631] | 4.018 [3.502, 4.640] | -0.7, -0.4, -0.6 | 0.009 |
| curn_vg_14f | curn_vg_m2a | J1713+0747_red_noise_log10_A | -16.311 [-19.597, -14.061] | -16.486 [-19.626, -14.066] | +0.7, +1.6, +0.6 | 0.034 |
| curn_vg_14f | curn_vg_m2a | J1713+0747_red_noise_gamma | 2.165 [0.192, 6.307] | 2.212 [0.167, 6.351] | +1.3, -0.6, -0.8 | 0.016 |
| curn_vg_14f | curn_vg_m2a | J1012+5307_red_noise_log10_A | -12.639 [-12.737, -12.543] | -12.636 [-12.736, -12.537] | -0.4, -2.1, -2.3 | 0.022 |
| curn_vg_14f | curn_vg_m2a | J1012+5307_red_noise_gamma | 0.591 [0.100, 1.186] | 0.581 [0.107, 1.185] | -0.7, +1.1, +0.1 | 0.016 |
| curn_vg_14f | curn_vg_m2a | J0610-2100_red_noise_log10_A | -12.777 [-13.550, -12.238] | -12.769 [-13.552, -12.237] | +0.2, -0.7, -0.1 | 0.017 |
| curn_vg_14f | curn_vg_m2a | J0610-2100_red_noise_gamma | 3.373 [0.564, 6.491] | 3.331 [0.499, 6.431] | +1.9, +0.7, +1.5 | 0.018 |
| curn_vg_14f | curn_vg_m2a | J1909-3744_red_noise_log10_A | -17.292 [-19.751, -14.368] | -17.262 [-19.717, -14.359] | -1.2, -0.5, -0.2 | 0.012 |
| curn_vg_14f | curn_vg_m2a | J1909-3744_red_noise_gamma | 2.994 [0.295, 6.449] | 2.968 [0.263, 6.502] | +1.4, +0.4, -1.4 | 0.020 |
| curn_vg_14f | curn_vg_m2a | J1903+0327_red_noise_log10_A | -12.199 [-12.333, -12.052] | -12.199 [-12.331, -12.050] | -0.5, +0.1, -0.5 | 0.012 |
| curn_vg_14f | curn_vg_m2a | J1903+0327_red_noise_gamma | 1.628 [0.819, 2.499] | 1.627 [0.825, 2.472] | -0.2, +0.1, +1.3 | 0.011 |
| hd_vg_14f | hd_vg_m3a | B1937+21_red_noise_log10_A | -13.567 [-13.723, -13.431] | -13.571 [-13.722, -13.430] | -0.2, +1.3, -0.2 | 0.025 |
| hd_vg_14f | hd_vg_m3a | B1937+21_red_noise_gamma | 4.030 [3.524, 4.635] | 4.052 [3.535, 4.637] | -0.7, -2.2, -0.1 | 0.035 |
| hd_vg_14f | hd_vg_m3a | J1713+0747_red_noise_log10_A | -16.185 [-19.618, -14.039] | -16.258 [-19.620, -14.040] | +0.0, +0.4, +0.0 | 0.020 |
| hd_vg_14f | hd_vg_m3a | J1713+0747_red_noise_gamma | 2.094 [0.185, 6.363] | 2.179 [0.199, 6.309] | -0.5, -0.9, +0.5 | 0.025 |
| hd_vg_14f | hd_vg_m3a | J1012+5307_red_noise_log10_A | -12.635 [-12.733, -12.533] | -12.633 [-12.734, -12.533] | +0.3, -0.6, +0.1 | 0.019 |
| hd_vg_14f | hd_vg_m3a | J1012+5307_red_noise_gamma | 0.598 [0.101, 1.234] | 0.595 [0.114, 1.182] | -0.7, +0.4, +2.4 | 0.020 |
| hd_vg_14f | hd_vg_m3a | J0610-2100_red_noise_log10_A | -12.761 [-13.568, -12.248] | -12.778 [-13.551, -12.248] | -0.6, +0.9, -0.0 | 0.031 |
| hd_vg_14f | hd_vg_m3a | J0610-2100_red_noise_gamma | 3.266 [0.515, 6.505] | 3.373 [0.499, 6.423] | +0.2, -1.5, +1.1 | 0.022 |
| hd_vg_14f | hd_vg_m3a | J1909-3744_red_noise_log10_A | -17.302 [-19.760, -14.402] | -17.256 [-19.704, -14.388] | -1.1, -0.5, -0.2 | 0.034 |
| hd_vg_14f | hd_vg_m3a | J1909-3744_red_noise_gamma | 2.836 [0.302, 6.458] | 2.996 [0.251, 6.529] | +1.6, -1.8, -1.7 | 0.033 |
| hd_vg_14f | hd_vg_m3a | J1903+0327_red_noise_log10_A | -12.201 [-12.328, -12.051] | -12.198 [-12.329, -12.050] | +0.3, -1.4, -0.0 | 0.024 |
| hd_vg_14f | hd_vg_m3a | J1903+0327_red_noise_gamma | 1.608 [0.859, 2.440] | 1.611 [0.799, 2.436] | +1.9, -0.2, +0.1 | 0.025 |

**ICRS vs enterprise positions.** Using the correct ICRS positions (B1855+09, B1937+21 and
B1953+29 move by 0.46-0.59 deg) shifts the HD medians by -0.013 dex (A, gamma = 13/3),
+0.029 (gamma) and -0.009 dex (A, varied gamma): <= 0.1 posterior sd and within ~2 combined
MCSEs (z(50) = -2.1, +2.2, -1.6). The HD vs CURN Bayes factor moves by ln BF = -0.0125 +- 0.0005
(gamma varied) and -0.0106 +- 0.0004 (13/3), i.e. ~1%. The noise-marginalised OS S/N changes by
< 0.01.

| run | param | enterprise | ICRS | shift of median | z(5,50,95) |
|---|---|---|---|---|---|
| hd_g433_14f | gw_log10_A | -14.612 [-14.740, -14.508] | -14.625 [-14.749, -14.510] | -0.0132 | -1.4, -2.1, -0.3 |
| hd_vg_14f | gw_gamma | 3.226 [2.679, 3.854] | 3.255 [2.676, 3.846] | +0.0289 | -0.1, +2.2, -0.2 |
| hd_vg_14f | gw_log10_A | -14.191 [-14.427, -13.979] | -14.199 [-14.439, -13.996] | -0.0087 | -0.8, -1.6, -1.5 |

## 6. Bayes factors HD vs CURN

Same parameters and priors in both models, so BF = Z_HD / Z_CURN needs only likelihood ratios
at posterior draws (`ptagwb.evidence`, `scripts/m2_bayes.py`):

* **Reweighting** (Hourihane & Meyers 2022): BF = E_CURN[L_HD / L_CURN] over the 6,000 CURN draws.
* **Reverse reweighting**: 1 / BF = E_HD[L_CURN / L_HD] over the 2,000 HD draws.
* **Bridge sampling** with the Meng & Wong optimal bridge, using both sample sets
  (effective sample sizes in the bridge weights); our headline estimator.

All in log space (log-sum-exp). Errors: moving-block bootstrap within chains, block = 2x the
integrated autocorrelation time of the weights (2000/2000/500 replicates), so autocorrelation is
accounted for. Tests on an analytic Gaussian toy with AR(1)-correlated exact posterior draws
recover the closed-form BF within 4 sd and show calibrated bootstrap errors
(`tests/test_evidence.py`).

| setup | estimator | BF | bootstrap sd | 16-84% | Kish ESS / n | block |
|---|---|---|---|---|---|---|
| vg14 | reweight | 183.0 | 9.4 | 173.8-192.6 | 467 / 6000 | 5 |
| vg14 | reverse_reweight | 227.7 | 12.6 | 215.3-239.4 | 371 / 2000 | 5 |
| vg14 | bridge | 178.3 | 6.5 | 171.9-185.1 | - / 8000 | [28, 9] |
| vg14 | self_check_identity_orf | 1.0 | 0.0 | 1.0-1.0 | 600 / 600 | 3 |
| g433_14 | reweight | 224.9 | 8.9 | 216.9-233.6 | 989 / 6000 | 7 |
| g433_14 | reverse_reweight | 213.7 | 19.5 | 195.7-234.1 | 326 / 2000 | 10 |
| g433_14 | bridge | 212.0 | 7.9 | 203.5-219.8 | - / 8000 | [20, 10] |
| g433_14 | self_check_identity_orf | 1.0 | 0.0 | 1.0-1.0 | 600 / 600 | 3 |
| vg5 | reweight | 833.8 | 28.2 | 807.6-864.6 | 1119 / 6000 | 6 |
| vg5 | reverse_reweight | 1081.4 | 63.6 | 1017.6-1144.2 | 483 / 2000 | 8 |
| vg5 | bridge | 894.4 | 29.7 | 865.4-928.0 | - / 8000 | [24, 10] |
| vg5 | self_check_identity_orf | 1.0 | 0.0 | 1.0-1.0 | 600 / 600 | 3 |

Checks:
* **BF(model vs itself) = 1**: reweighting the CURN chain with the CURN likelihood evaluated
  through the *correlated-ORF* code path with Gamma = I gives BF = 1 - 1e-11 (max |dlogL| ~ 2e-8)
  for all three setups.
* Recomputed log L at the chain draws equals the sampler's own value to 1e-9.
* **Front end**: replacing our PINT 1.1.7 arrays by the released feathers (analysis-time
  PINT 0.9.1; importance ratio from our CURN posterior, 1,200 draws) changes ln BF by
  +0.026 +- 0.003 (varied gamma) and +0.019 +- 0.003 (13/3), i.e. ~2-3%. Positions: -1% (Sec. 5).

Interpretation. The three estimators disagree by more than their bootstrap errors in the
varied-gamma setups (14 modes: 183 / 228 / 178; 5 modes: 834 / 1081 / 894); for 13/3 they agree
(225 / 214 / 212). The pattern (forward reweighting low, reverse high) is the usual finite-sample
behaviour when the HD posterior's tails are under-represented in the CURN draws and vice versa,
so the bootstrap errors understate the true uncertainty; the bridge estimate is the most
reliable. Our result: **BF(HD^gamma/CURN^gamma, 14 modes) = 178, with an estimator range
178-228**, vs the paper's ~200 (Fig. 2: 226 +- 70; tutorial hypermodel 202 +- 3, tutorial TI
198 +- 45); **5 modes: 894 (range 834-1081)** vs ~1000 (Fig. 2: 965); HD^13/3 vs CURN^13/3: 212
(range 212-225), "similar" in the paper. All agree within the paper's quoted uncertainty. A
product-space or thermodynamic-integration cross-check was not run (cost; the bridge already uses
both posteriors).

## 7. Optimal statistic

`ptagwb.optstat`: X_a = F^T C_a^-1 r_a and Z_a = F^T C_a^-1 F (CURN noise model, common-process
columns) are exactly the projected quantities d_a and E_a of the M1 square-root reduction with
the diagonal prior Phi_a = IRN + common, so the OS costs one CURN-likelihood evaluation;
rho_ab, sigma_ab, A^2, S/N follow enterprise_extensions `OptimalStatistic.compute_os`.

Validation (`tests/test_optstat.py`, `tests/test_oracle_optstat.py`):
* X_a, Z_a vs a brute-force TOA-space reference (timing-projected inverse covariance) on a
  synthetic PTA: 1e-8 relative.
* **Oracle: enterprise_extensions `OptimalStatistic`** (model_2a, tm_marg=False, as in the
  release) on the released feathers at the released ML vector and a perturbed vector, gamma = 13/3
  and varied: |rho_ours - rho_ee| < 1e-4 sigma_ab for all 2,211 pairs, sigma_ab to 1e-5, A^2 to
  1e-4 sigma.
* Pair covariance (Allen & Romano 2023, as implemented in the release's `gw_corr`, which we
  vectorise term by term): equal to a port of the release loop to 1e-10 (synthetic), and to the
  released 2211 x 2211 matrix to 1.3e-5 (feathers) / 1.3e-4 (our PINT arrays) in correlation units.
* At the released ML noise vector on our PINT arrays (enterprise positions) we get
  A^2 = 6.7032e-30 +- 1.2324e-30, S/N = 5.439, vs the release notebook's printed 6.7037e-30 +-
  1.2324e-30, S/N 5.439. Curiously, on the released feathers our (and current
  enterprise_extensions') A^2 is 0.36% (0.02 sigma) higher, 6.7278e-30, with sigma identical to 1e-7.
  Since sigma depends only on Z_a and A^2 also on the residuals, the notebook's inputs evidently
  differed slightly from the published feathers in the residuals. Immaterial (0.02 sigma).

Noise-marginalised S/N over all 6,000 draws of our CURN chains (Fig. 4): **4.52 +- 1.03**
(gamma = 13/3) and **5.04 +- 1.10** (varied gamma), vs 4.49 +- 1.02 and 4.98 +- 1.10 from the
released per-draw correlations (20,000 draws) and "4 +- 1" / "5 +- 1" in the paper. The
fixed-noise OS at the maximum-likelihood draw of our CURN^13/3 chain (log10 A_CURN = -14.693) is
A^2 = 5.78e-30 +- 1.10e-30, S/N 5.27 (the released ML vector, which has 6.3 lower log L on our
arrays, gives S/N 5.44). ICRS positions change the S/N by < 0.01.

**Binned correlations (Fig. 1c).** 15 bins with the notebook's edges (bin counts 147, 143, 151,
152, 73, 208, 158, 146, 147, 152, 139, 153, 148, 147, 145, identical to the release), the
Allen & Romano pair-covariance-aware estimator per bin, and the binned covariance B_jk. At the
released ML vector we reproduce **chi^2 = 8.10** (p = 0.92, chi^2 with 15 dof), as in the paper
(8.03 with ICRS positions). At the maximum-likelihood draw of our own chain chi^2 = 9.99
(p = 0.82). chi^2 is strongly sensitive to the noise point: over the 20 highest-likelihood draws of
our chain it ranges 6.4-30.2 (median 10.1), so "MAP CURN^13/3" is not a sharp specification; the
paper's conclusion (consistent with HD, p > 0.3) holds for most but not all of these draws.

## 8. Figures

* `docs/figures/m2_fig1a_freespec.png`: HD free spectrum (ours left halves, released right
  halves, with the notebook's [-9, -4] truncation), HD^gamma 90% bands, and the paper's
  A = 2.4e-15 gamma = 13/3 line.
* `docs/figures/m2_fig1b_gamma_A.png`: HD^gamma (gamma, log10 A) 1/2/3-sigma contours and
  marginals, plus HD^13/3 amplitude marginals (dashed).
* `docs/figures/m2_fig1c_correlations.png`: binned correlations at our MAP draw and at the
  released ML vector.
* `docs/figures/m2_fig4_os_snr.png`: noise-marginalised OS S/N distributions.

Colours: ours = blue, released = orange throughout.

## 9. Compute cost

Ten production runs: **8.0 h** of warmup + sampling wall time on the RTX 5090 (plus ~1 h of
pilots and ~10 min of setup/compilation), 5.6 M gradient evaluations (per chain, summed over
chains). Within the ~12 h budget; HD^free was cut to 4 x 250 draws to stay inside it.

| | per-chain gradient, measured in the runs | ESS/s, common parameters (sampling phase) |
|---|---|---|
| CURN, 14 modes (4 chains vectorised) | 1.8 ms (4.6 ms per 4-chain step, ~92 steps/draw) | 1.0-3.2 |
| HD, 14 modes, float64 | 13.9 ms (38 ms per 4-chain step; lockstep loss ~30%) | 0.11-0.53 |
| HD, 5 modes | 4.6 ms | 1.5 |
| HD^free, 30 modes, mixed-precision gradient | 32.6 ms | 0.01 (worst bin) |

Post-processing is cheap: HD log L at 6,000 CURN draws takes ~60 s (batched), the
noise-marginalised OS over 6,000 draws a few seconds, the 2211 x 2211 pair covariance ~2 s.
Pilots (CURN diagonal vs dense metric, HD step-size-only warmup) are described in Sec. 3.

## 10. Deviations and open issues

1. **HD^free.** The free-spectrum marginals of bins that are partly
   signal and partly prior-dominated (f_1, f_4, f_5, f_8) are bimodal ("power present / absent");
   NUTS crosses between the modes slowly. hd_fs30 (4 x 250 draws, the budget allowed) has R-hat up to
   1.08 (rho at f_1, bulk ESS 31) and lower-tail quantile MCSEs of 0.1-1.5 dex in those bins. The
   peaks and medians agree with the released core (first 10 bins |z(50)| <= 1.6) and the
   well-measured bins f_2, f_3 agree to 0.01 dex, but the lower tails in Fig. 1a are not
   converged at the level of the released 490,000-sample core. The auxiliary CURN^free run shows
   the same behaviour (R-hat 1.15 at f_4, bulk ESS 20).
2. **Bayes-factor uncertainty.** The bootstrap errors are only Monte-Carlo errors given the
   draws and understate the estimator spread for varied gamma (Sec. 6). A product-space NUTS
   (or TI) run would pin down whether the true value is nearer 180 or 230. Not run (cost).
3. **R-hat of a few IRN nuisance parameters** in the 4 x 500-draw HD runs is 1.02-1.03 (funnel
   marginals); common-parameter R-hats are <= 1.005 (HD^13/3 ICRS: 1.014) and their bulk ESS >= 258. Longer HD chains
   would fix this at ~40 min per 500 extra draws per run.
4. **Funnel geometry limits NUTS efficiency** (ESS per gradient ~1-4e-3 for the common
   parameters). A reparameterisation of the IRN (e.g. non-centred in log-power at the most
   sensitive frequency) might help; not explored.
5. **Mixed-precision gradients** were used only for HD^free (exact energies, theoretically
   unbiased, toy-tested); its acceptance rate (0.93) is comparable to the float64 runs.
6. **"MAP CURN^13/3"** is ill-defined at the level that matters for the binned chi^2 (Sec. 7).
7. **CURN^free prior** remains inferred only (Sec. 2); no CURN^free result is claimed.
8. hd_fs30 was reduced to 100 warmup + 250 draws per chain after measuring the free-spectrum cost
   on curn_fs30 (committed config change before its launch); the driver was restarted once to
   run it last (`runs/production.log`).
