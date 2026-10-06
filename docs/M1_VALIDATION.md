# M1 validation: data layer and deterministic likelihood

M1 covers the PINT front end, the fixed white noise, the Fourier bases, the ORFs and the JAX
CURN / HD likelihoods. There is no sampling yet. All numbers below were measured on
2026-10-06 on the RTX 5090 (CUDA 13, JAX 0.11.2, float64) with PINT 1.1.7 and the oracle
group from `docs/ENVIRONMENT.md` (enterprise 3.5.0, enterprise_extensions 3.0.3, discovery 0.5
@ b26d2554).

How to reproduce:

```bash
uv run --no-sync python scripts/ingest.py           # PINT -> data/cache/pulsars (about 105 s on 16 cores)
uv run --no-sync pytest                              # 51 tests, about 4 min (oracle tests included)
uv run --no-sync python scripts/m1_validate.py --enterprise   # numbers below -> outputs/m1_validation.json
```

Always use `uv run --no-sync`. A plain `uv run` re-syncs the default groups and uninstalls
the oracle packages.

## 1. What was built

| Module | Content |
|---|---|
| `ptagwb.data` | PINT ingestion with enterprise `PintPulsar` conventions. Outputs barycentric TOAs, site TOAs, pre-fit residuals, raw TOA errors, SSB and topocentric radio frequencies, enterprise-resolved backend flags (`group` > `g` > `sys` > `i` > `f` > `fe_be`), the design matrix with column names, and the ICRS unit vector. Arrays are merge-sorted by barycentric TOA. Each pulsar is cached as `data/cache/pulsars/<clock>/<psr>.npz`, keyed by a sha256 over the par file, tim file, clock files, PINT version and schema. The 67-pulsar selection is checked against the paper. |
| `ptagwb.noise` | EFAC and T2 EQUAD: `EFAC^2 (sigma^2 + EQUAD^2)`. ECORR sits outside EFAC. Epochs are quantised per backend with enterprise's rule: a bucket opens at a TOA and takes later TOAs while `t - t_open < 1 s`; buckets with fewer than 2 TOAs are dropped. Exact blockwise whitening, solve and logdet. Missing parameters raise an error, and so do unused white-noise entries for a pulsar. |
| `ptagwb.basis` | sin/cos-interleaved basis with `f_k = k/T` on the whole-array `T`. enterprise `powerlaw` coefficient variance including `Delta f`, and the free spectrum. |
| `ptagwb.orf` | HD (auto-correlation 1), CURN, monopole and dipole (diagonal regulariser 1e-5 as in enterprise, 1e-6 as in discovery). |
| `ptagwb.likelihood` | Stage 1, once, on the host in numpy float64: whiten, QR-project off the timing model, then store `s = r^T P r`, `b = F^T P r` and `A = F^T P F` (60 x 60) plus the constants. Stage 2, per call, in JAX: absorb the intrinsic RN per pulsar. CURN is then separable. HD solves one 67 x 28 = 1876 system. |
| `scripts/ingest.py`, `scripts/m1_validate.py` | Cache builder and the validation run behind the numbers below. |
| `tests/` | Unit tests (synthetic PTA, dense brute-force reference), oracle tests (`-m oracle`), and a long-double reference (`tests/extended_precision.py`). |

### Likelihood layout and why

All TOA-level work depends only on the fixed white noise, so it runs once per pulsar (about
11 s for all 67 on the host). After that every pulsar is described by arrays of the same
shape (`A`: 60 x 60, `b`: 60, scalars). The hot path is therefore a dense stacked batch with
no padding, no ragged TOA arrays and no masks:

* `vmap` over pulsars of a 60 x 60 Cholesky of `I + Phi^1/2 A Phi^1/2`. The `I + ...` form is
  well conditioned and avoids `Phi^-1`.
* CURN: the common variance is added to the first 28 diagonal entries, giving a sum of 67
  independent terms.
* HD: the per-pulsar Woodbury step gives the RN-projected common-mode blocks `E_a` (28 x 28)
  and `d_a`. We then factorise

      Sigma' = Gamma^-1 (x) I_28 + blockdiag(phi^1/2 E_a phi^1/2)      (1876 x 1876)

  using `log|Q| + log|Q^-1 + E| = log|Sigma'| + 28 log|Gamma|`. `Gamma^-1` is precomputed,
  so assembly needs no contraction over pulsar triples (P^3). A hand-written VJP needs only
  the 28 x 28 diagonal blocks of `Sigma'^-1`. They come from a GEMM-blocked triangular
  inverse of the forward Cholesky factor, about n^3/3 flops, where differentiating through
  `cholesky` costs about 3 n^3. The spec's form `B = I + Z^T E Z` is kept as a reference
  (`method="B"`). Its values match exactly (difference 0.0 on the full-PTA HD at a chain
  sample) and its gradients match to 4e-12 relative.

## 2. Front end: PINT 1.1.7 vs the released feathers

The reference is discovery's `v1p1_de440_pint_bipm2019-*.feather`. These are byte-identical to
the tutorial feathers and were written by enterprise's PINT front end at analysis time; the
release used PINT 0.9.1 and its par files were written by 0.8.4. TOAs are matched by
tim-file key (observation name, subband, subint, system), because TOAs inside one
observation coincide to within the float64 resolution of about 1 us and stable sorting can
order them differently. The likelihood does not depend on that order.

| Quantity (67 pulsars) | Result |
|---|---|
| TOA counts, TOA sets, raw TOA errors | identical (674,683 TOAs) |
| Backend flags, ECORR buckets (ours vs our code on the feathers vs enterprise `create_quantization_matrix` vs discovery `makegp_ecorr(enterprise=True)`) | identical for all pulsars |
| Barycentric TOAs | max difference 9.5e-7 s, which is 1 ulp at 4.6e9 s |
| SSB radio frequencies | max 2.8e-11 MHz |
| Whole-PTA span T | 505861299.1401644 s in both. This is the exact value hard-coded in the collaboration's figure code (16.0297772689 yr) |
| Residuals, all pulsars except J1713+0747 | rms difference median 1.1 ns, max 2.3 ns. Max abs median 4.7 ns, max 6.3 ns. After projecting out the timing model: rms median 0.49 ns, max 0.86 ns; max abs 5.7 ns |
| Design-matrix column space (largest principal-angle sine) | median 4.5e-8. ELL1 pulsars differ at 1e-5 to 7e-4 (EPS1/EPS2 partials: J1719-1438 7.3e-4, B1855+09 3.5e-5) |
| Clock files | release `clock/` and PINT's global repository give identical arrays. The default is the frozen release clock files |

Front-end findings and decisions:

1. **J1713+0747 (DDK).** PINT 1.1.7 gives a different KIN/KOM partial: the DDK model in
   ecliptic coordinates changed between PINT versions. Two TOAs that sit exactly on DMX range
   edges (`DMX_0240` is a 4 ms window, MJD 57508.36015548 to .36015553; `DMXR2_0241` is
   another) are no longer assigned to their bin. The residuals differ by 0.19 us rms and up
   to 0.68 us, almost entirely inside the timing-model span: the rms after projection is
   2.3 ns, but the two edge TOAs differ by 0.5 us. **Kept as is**, with PINT 1.1.7 as the
   front end. The likelihood impact is in the table below.
2. **J1024-0719 `F3 0 1`.** The par file marks F3 as fitted, but its value is exactly 0 with
   no uncertainty. The tempo2 version of the same par file has no F3, and the GWB-analysis
   feathers have 149 columns, i.e. no F3. PINT 1.1.7 would add an F3 column. **We freeze
   such placeholders**: a fit flag with value 0 and no uncertainty. In NG15 this hits only
   J1024-0719 F3, and the frozen names are recorded in `meta["frozen_placeholders"]`.
   Without this, J1024's logL shifts by about -58: an extra `log 1e40` timing column, plus a
   noise-dependent part with std 0.08 over the posterior.
3. **Sky positions of B-name pulsars.** enterprise (`BasePulsar._get_radec_from_ecliptic`,
   pyephem) converts ELONG/ELAT using the **B1950** equinox for names starting with "B". So
   the feather `pos` of B1855+09, B1937+21 and B1953+29 is 0.46 to 0.59 deg away from ICRS.
   We confirmed this by reproducing the feather vectors with pyephem `epoch='1950'` to 2e-8
   rad. For the J pulsars the pyephem vs PINT/IERS2010 obliquity difference is 2e-7 rad.
   **We use the physically correct ICRS position** from PINT `coords_as_ICRS`. The impact on
   HD is a logL shift of -0.004 to -0.017 at posterior samples, and up to 0.11 at prior
   draws. It does not affect CURN.

### Front-end impact on the likelihood

Here "ours on our PINT arrays" is compared with "ours on the feathers". The CURN likelihood
is separable, so its difference splits per pulsar. Statistics are over 20 posterior samples
of the CURN chain (m2a):

| Pulsar | mean dlogL | std over samples | peak-to-peak |
|---|---|---|---|
| J1713+0747 | -1.197 | 0.025 | 0.103 |
| B1937+21 | -0.213 | 0.014 | 0.061 |
| J1600-3053 | -0.009 | 0.006 | 0.025 |
| J1744-1134 | +0.125 | 0.006 | 0.022 |
| all others | below 0.04 | below 0.006 | |

Full PTA at the released chain samples: ours on our arrays minus the stored `logl` is
-1.42 +- 0.13 (CURN) and -1.47 +- 0.17 (HD). Most of the spread is the float32 rounding of
the stored column, ulp 0.5. A parameter-independent offset does not affect posteriors or
Bayes factors. The parameter-dependent part is O(0.03) over the posterior, small against
the O(1) logL differences that drive inference. At extreme prior draws (IRN log10_A near
-11), the front-end difference grows to a few units, again mostly J1713+0747.

## 3. Likelihood: oracle comparisons

Constants: `convention="chain"` reproduces discovery and enterprise 3.3.1, the version that
wrote the NG15 production chains. It includes `m log(1e40)` per pulsar and has no `2 pi`
term. `convention="enterprise"` subtracts `n_toa/2 log 2 pi`, as enterprise 3.5 does. With
the constants matched we compare **absolute** values, which is stronger than comparing
differences.

Test points: 6 chain samples (m2a for CURN, m3a for HD), 5 prior draws with IRN
`log10_A in [-20, -13]` and `gamma in [0, 7]`, and 5 draws over the full prior
`log10_A in [-20, -11]`. Common `log10_A in [-18, -13]`, `gamma in [0, 7]`.

| Comparison (67 pulsars, 30 IRN + 14 common frequencies) | CURN max abs diff (spread) | HD max abs diff (spread) |
|---|---|---|
| ours vs discovery, feathers, chain samples | 3.7e-9 (1.9e-9) | 4.7e-9 (8.4e-9) |
| ours vs discovery, feathers, prior with log10_A <= -13 | 2.1e-8 (1.9e-8) | 5.8e-8 (7.2e-8) |
| ours vs discovery, **our PINT arrays**, chain samples | 3.7e-8 (2.8e-9) | 4.0e-8 (1.7e-8) |
| ours vs discovery, our PINT arrays, prior with log10_A <= -13 | 6.2e-8 (2.4e-8) | 9.3e-8 (8.3e-8) |
| ours vs enterprise 3.5 (`convention="enterprise"`), feathers, 5 chain + 5 prior points | 3.5e-7 (3.9e-7) | 2.9e-7 (3.4e-7) |
| ours (feathers) vs stored chain `logl` (float32), 8 samples | max 0.25 | max 0.25 |
| ours vs discovery, full prior (log10_A up to -11) | 1.0e-4 | 1.6e-4 |

`pytest` asserts at most 1e-6 for both the absolute value and the spread on the first five
rows; these pass with a margin of at least 2.6x. In the pytest run itself the largest values were 4.7e-8 / 7.6e-8
(CURN feathers / ours), 3.8e-7 / 1.8e-7 (HD) and 1.6e-7 (enterprise). The last row is not
a bug on our side. For IRN amplitudes near the prior edge (log10_A > -12.5 with large
gamma), discovery and enterprise build `Phi^-1 + T^T N^-1 T` from normal equations, which
squares the condition number. A **long-double reference** (`tests/extended_precision.py`)
recomputes the same per-pulsar likelihood from the same float64 inputs with a 64-bit
mantissa; it whitens, runs Gram-Schmidt off the timing model and does the Cholesky in long
double. It agrees with ours to 2e-10 (J1705-1903), 2e-11 (J1751-2857) and 9e-9 (J2214+3000)
at such a point. discovery is off by 2.0e-4, 6.5e-6 and 1.7e-5 at the same points.

Other checks:

* Synthetic PTA (5 pulsars, multi-channel epochs, singleton epochs, 2 backends) vs
  brute-force dense TOA-space covariance with the projector formula, over CURN, HD,
  monopole and dipole, power law and free spectrum, both HD methods: worst relative error
  **4.9e-14** (test tolerance 1e-8).
* HD code path with `Gamma = I` vs the separable CURN path: equal to 1e-9 relative
  (synthetic) and 1e-7 absolute (full PTA).
* Gradients: JAX vs central finite differences on the synthetic PTA for every parameter
  (CURN and HD, power law and free spectrum); full PTA HD with a 5-point stencil for 5
  parameters. All pass at 1e-5 and 1e-4 relative respectively.
* ECORR convention matters. discovery's default `makegp_ecorr` keeps singleton epochs and
  shifts logL by +0.019 on average (spread 0.04 to 0.08 across points). With
  `enterprise=True` it matches enterprise and us bucket for bucket.
* Timing basis. The SVD basis (`use_svd=True`) and the unit-norm basis give the same
  projected quantities: `s` agrees to 2.6e-9 absolute and `A` to 1.3e-11 relative. They differ in
  the constant by 1665 in logL. Only the SVD basis reproduces the production chains'
  absolute `logl`, so the production runs used SVD. This settles the spec's "UNVERIFIED".
* The white-noise dictionary `v1p1_wn_dict.json` holds 697 entries. Its 645 EFAC/EQUAD/ECORR
  values for the 67 pulsars are **bit-identical** to the `Constant=` values in the enterprise
  runtime info of both production chains (m2a CURN, m3a HD). The remaining entries are 6 for
  J0614-3329 and 46 IRN values, which are ignored. Every one of the 215 systems has all
  three parameters, and `nanograv_backends` and `by_backend` select the same systems in NG15.

## 4. Timings (RTX 5090, float64, after JIT, our PINT arrays)

| | value | value + gradient |
|---|---|---|
| CURN (67 x 60 Cholesky, vmapped) | 0.24 ms | 0.56 ms |
| HD, exact float64 | 5.9 ms | **12.0 ms** |
| HD, `grad_precision="mixed"` | 5.9 ms | **7.7 ms** |
| discovery HD (value only, same GPU) | 7.3 ms | |
| discovery CURN (value only) | 1.6 ms | |
| Stage-1 precompute, host numpy, all 67 pulsars | about 11 s, once | |

On this GPU float64 runs at about 1/64 of the float32 rate. The float64 matmul peak measured
1.76 TFLOPS. The float64 Cholesky of the 1876 system takes 5.4 ms (cuSOLVER), and that sets
the floor for the HD value. Things that did not help: blocked Cholesky in JAX (8 to 9 ms);
recursive GEMM-based Cholesky (12 ms); cuBLAS FP64 emulation through
`CUBLAS_EMULATE_DOUBLE_PRECISION` (no effect through XLA).

**Mixed precision is opt-in and affects only the gradient.** With `grad_precision="mixed"`
the backward pass inverts the symmetrically equilibrated Cholesky factor in float32. The
likelihood value stays float64 and bit-identical. Measured gradient error, maximum
component over the largest component: 7e-6 at a posterior sample; at prior draws up to
3e-4, and up to 1e-2 on small components. HMC/NUTS stays exact with a deterministic
approximate gradient, because leapfrog with any deterministic gradient field is still
reversible and volume preserving and the Metropolis step uses the exact float64 value;
only the acceptance rate can suffer. Float64 gradients remain the default ("correctness
over speed"). The exact path misses the ~10 ms target for value + gradient (12.0 ms); the
mixed path meets it.

## 5. Deviations from / additions to `docs/SPEC_astra.md`

All load-bearing claims in the spec were checked against the installed enterprise and
discovery sources and against the data. **Confirmed:** T2 EQUAD convention; ECORR outside
EFAC with nmin=2 and the 1 s bucket opened at the first TOA; whole-PTA T (exact value
reproduced); enterprise `powerlaw` including `Delta f`; sin/cos ordering with absolute TOAs
as the common phase origin; HD with auto-correlation 1; the improper timing prior and the
`m log 1e40` constant; the 1876-dimensional reduced HD system; Cholesky-only algebra.

| Spec item | What we found / did |
|---|---|
| "Historical production SVD setting: UNVERIFIED" | Verified: SVD. Only the SVD basis reproduces the chains' absolute logl; the normed basis is off by 1665. |
| "Exact production dictionary bytes: UNVERIFIED" | Verified: `v1p1_wn_dict.json` equals the production runs' constants bit for bit. |
| ECORR selection `nanograv_backends` | Equivalent to `by_backend` for NG15, which is what the tutorial uses. Default `ecorr_selection="nanograv"`; strict mode fails if any dictionary entry goes unused. |
| Sky vector "at the timing-model reference position" | Correct in principle, but enterprise and the released feathers use B1950-equinox coordinates for B-name pulsars (0.5 deg error). We use ICRS and quantify the impact (Sec. 2). |
| "Current source includes n_TOA log 2 pi" | True for enterprise 3.5, not for 3.3.1 or discovery, which are what the chains used. Both conventions are offered; `chain` is the default. |
| Monopole/dipole diagonal 1 + 1e-5 | enterprise. discovery uses 1e-6. Exposed as `orf_diag_eps`. |
| HD system `B = I + L^T E L` | Implemented as the reference path. The default factorises the congruent `Sigma' = Gamma^-1 (x) I + phi^1/2 E phi^1/2` with a custom VJP: same value, 2.5x faster value + gradient. |
| Timing marginalisation | Exact projection via QR on the whitened basis. No `1e40` enters the numerics, only the constant. |
| Radio frequency | enterprise's `psr.freqs` is the **SSB** frequency (`barycentric_radio_freq`); exported as `freqs`, with topocentric `freqs_topo` alongside. |
| Split ao/gbt files | Not in the baseline. The discovery and tutorial data sets contain only the 67 combined pulsars, and the split files exist for the telescope cross-validation (GWB Sec. 5.4). The 8 files are B1937+21ao/gbt, J1600-3053gbt, J1643-1224gbt, J1713+0747ao/gbt, J1903+0327ao and J1909-3744gbt. They load on request: `load_pulsars([...names])` or `find_par_tim(split_only=True)`. |
| PINT version | Not in the spec. Front-end deviations are documented in Sec. 2: J1713+0747 DDK/DMX edges, the J1024-0719 F3 placeholder, ELL1 partials, and about 1 ns rms residual differences. |

## 6. Open concerns

* J1713+0747 differs systematically from the analysis-time front end (DDK convention, two
  DMX-edge TOAs). The impact is about 0.03 std in logL over the posterior, but a
  bit-faithful reproduction of the feathers would need the old PINT DDK convention.
* Mixed-precision gradient errors are larger at extreme prior points (up to 1e-2 relative on
  small components). M2 should check NUTS acceptance before relying on it.
* Precision in the IRN prior corner (log10_A -> -11, gamma -> 7). Our float64 numbers match
  long double to below 1e-8 there, but `I + Phi^1/2 A Phi^1/2` reaches a condition number
  around 1e20 in that corner. We have not proven bounds for every pulsar and parameter
  combination.
* The absolute-value checks against the released chains are limited by their float32 `logl`
  column (+-0.25).
