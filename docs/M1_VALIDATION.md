# M1 validation: data layer and deterministic likelihood

M1 covers the PINT front end, the fixed white noise, the Fourier bases, the ORFs and the JAX
CURN / HD likelihoods. There is no sampling yet. All numbers below were measured on
2026-10-06 on the RTX 5090 (CUDA 13, JAX 0.11.2, float64) with PINT 1.1.7 and the oracle
group from `docs/ENVIRONMENT.md` (enterprise 3.5.0, enterprise_extensions 3.0.3, discovery 0.5
@ b26d2554).

How to reproduce:

```bash
uv run --no-sync python scripts/ingest.py           # PINT -> data/cache/pulsars (about 105 s on 16 cores)
uv run --no-sync pytest                              # 74 tests, about 9 min (oracle tests included)
PTAGWB_REQUIRE_ORACLES=1 uv run --no-sync pytest     # validation mode: any skip is a failure
uv run --no-sync python scripts/m1_validate.py --enterprise   # numbers below -> outputs/m1_validation.json
```

Always use `uv run --no-sync`. A plain `uv run` re-syncs the default groups and uninstalls
the oracle packages. In validation mode (`--require-oracles` or `PTAGWB_REQUIRE_ORACLES=1`)
every skip counts as a failure, whatever the reason: missing data, oracle packages or GPU.
So a green run there means every oracle comparison actually ran.

**Revision 2 (after independent review, 2026-10-06).** Changes since the first M1 report:

1. The free-spectrum prior was resolved from provenance (Sec. 7; errata in `SPEC_astra.md`).
2. The RN reduction was rewritten in square-root form, with a diagonal ORF split, to remove
   catastrophic cancellation in the prior corners (Sec. 3b).
3. Timing-basis rank is now enforced.
4. J1024-0719 F3 is an explicit, checked exception rather than a general rule.
5. Validation mode was added.
6. Enterprise-convention sky positions are available as an option.

**Revision 3 (second review).**

1. The per-pulsar square-root reduction now has an analytic VJP. Gradients no longer go
   through JAX's generic QR rule, which lost up to ~1e-4 relative accuracy in the free-spectrum
   domain (Sec. 3c).
2. `split_fraction` must be finite and in (0, 1); non-finite or asymmetric ORF matrices are
   rejected.
3. The timing-rank check requires full column rank, including n_columns <= n_rows.
4. The prior normaliser is reported at full precision (Sec. 7).
5. New tests: free-spectrum and 30-mode identity scans, and correlated 50-digit checks for
   all ORFs.

All numbers below are from revision 3 unless marked otherwise.

## 1. What was built

| Module | Content |
|---|---|
| `ptagwb.data` | PINT ingestion with enterprise `PintPulsar` conventions. Outputs barycentric TOAs, site TOAs, pre-fit residuals, raw TOA errors, SSB and topocentric radio frequencies, enterprise-resolved backend flags (`group` > `g` > `sys` > `i` > `f` > `fe_be`), the design matrix with column names, the ICRS unit vector `pos`, and `pos_enterprise`, the vector enterprise and the released products use (pyephem; Sec. 2). Arrays are merge-sorted by barycentric TOA. Each pulsar is cached as `data/cache/pulsars/<clock>/<psr>.npz`, keyed by a sha256 over the par file, tim file, clock files, PINT version and schema. The 67-pulsar selection is checked against the paper. |
| `ptagwb.noise` | EFAC and T2 EQUAD: `EFAC^2 (sigma^2 + EQUAD^2)`. ECORR sits outside EFAC. Epochs are quantised per backend with enterprise's rule: a bucket opens at a TOA and takes later TOAs while `t - t_open < 1 s`; buckets with fewer than 2 TOAs are dropped. Exact blockwise whitening, solve and logdet. Missing parameters raise an error, and so do unused white-noise entries for a pulsar. |
| `ptagwb.basis` | sin/cos-interleaved basis with `f_k = k/T` on the whole-array `T`. enterprise `powerlaw` coefficient variance including `Delta f`, and the free spectrum. |
| `ptagwb.orf` | HD (auto-correlation 1), CURN, monopole and dipole (diagonal regulariser 1e-5 as in enterprise, 1e-6 as in discovery). |
| `ptagwb.likelihood` | Stage 1, once, on the host in numpy float64: whiten, QR-project off the timing model, QR the projected Fourier basis, `F_p = Q_F R_F`, then store the square-root contractions `R_F` (60 x 60), `c = Q_F^T r_p` and `s_perp`, plus the constants. `A = F^T P F = R_F^T R_F` is never formed. Fails clearly on a numerically rank-deficient timing matrix. Stage 2, per call, in JAX: square-root (QR) absorption of the intrinsic RN per pulsar. CURN is then separable; HD solves one 67 x 28 = 1876 system. `precompute(..., position="icrs"/"enterprise")` selects the sky-position convention. |
| `ptagwb.config` | `PRIORS` and `FREESPEC_LOG10_RHO_PRIORS`, with provenance (Sec. 7). |
| `scripts/ingest.py`, `scripts/m1_validate.py` | Cache builder and the validation run behind the numbers below. |
| `tests/` | Unit tests (synthetic PTA, dense brute-force reference), oracle tests (`-m oracle`), and a long-double reference (`tests/extended_precision.py`). |

### Likelihood layout and why

All TOA-level work depends only on the fixed white noise, so it runs once per pulsar (about
11 s for all 67 on the host). After that every pulsar is described by arrays of the same
shape (`A`: 60 x 60, `b`: 60, scalars). The hot path is therefore a dense stacked batch with
no padding, no ragged TOA arrays and no masks:

* `vmap` over pulsars of a reduced QR of the stacked 120 x 60 matrix `[R_F Phi^1/2; I]`,
  which yields `T` with `T^T T = I + Phi^1/2 A Phi^1/2`. The normal matrix is never formed,
  so its condition number (up to ~1e20 in the IRN prior corner) is never squared.
* CURN: the common variance is added to the first 28 diagonal entries, giving a sum of 67
  independent terms.
* HD: split `Gamma = lam0 I + Gamma'` with `lam0 = lambda_min(Gamma)/2`. The diagonal part
  `lam0 phi^CP` is absorbed per pulsar together with the IRN. This step gives the projected
  common-mode blocks `E_a = [R^-1/2 T^-1 Q1^T R_F]_GG` and `d_a = [R^-1/2 T^-1 y]_G`, where
  G is the common modes and `R = Phi^RN + lam0 phi^CP`. Neither involves a subtraction. We
  then factorise

      Sigma' = Gamma'^-1 (x) I_28 + blockdiag(phi^1/2 E_a phi^1/2)      (1876 x 1876)

  using `log|Q| + log|Q^-1 + E| = log|Sigma'| + 28 log|Gamma'|`. Thanks to the split,
  `phi^CP E_a <= 1/lam0` stays bounded even where the common process dominates. `Gamma^-1` is precomputed,
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
   **We use the physically correct ICRS position** from PINT `coords_as_ICRS` by default. The
   impact on HD is a logL shift of -0.004 to -0.017 at posterior samples, and up to 0.11 at
   prior draws. It does not affect CURN. For like-for-like comparisons with the released
   products, `pos_enterprise` reproduces enterprise's vectors to below 1e-12 (all 67
   pulsars; bit-identical to enterprise `PintPulsar` in the tests). Select it with
   `precompute(..., position="enterprise")`.

**Which position mode each number uses.**

* "feathers": the released arrays, which carry enterprise positions.
* "our PINT arrays": ICRS, the default, unless the text says "enterprise pos".
* discovery or enterprise fed our arrays: the same positions as ours.

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
-1.42 +- 0.13 (CURN) and -1.47 +- 0.17 (HD, ICRS), or -1.45 +- 0.17 (HD, enterprise pos). Most of the spread is the float32 rounding of
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
| ours vs discovery, **our PINT arrays**, chain samples | 3.6e-8 (1.9e-9) | 4.0e-8 (1.7e-8) |
| ours vs discovery, our PINT arrays, prior with log10_A <= -13 | 7.1e-8 (2.8e-8) | 9.6e-8 (4.6e-8) |
| ours vs enterprise 3.5 (`convention="enterprise"`), feathers, 5 chain + 5 prior points | 3.5e-7 (3.9e-7) | 2.9e-7 (3.4e-7) |
| ours (feathers) vs stored chain `logl` (float32), 8 samples | max 0.25 | max 0.25 |
| ours vs discovery, full prior (log10_A up to -11) | 1.0e-4 (feathers), 2.6e-4 (our arrays) | 1.6e-4, 7.9e-5 |

`pytest` asserts at most 1e-6 for both the absolute value and the spread on the first five
rows; these pass with a margin of at least 2.6x. In the pytest run itself the largest values were 4.7e-8 / 8.1e-8
(CURN feathers / ours), 3.8e-7 / 2.6e-7 (HD) and 1.6e-7 (enterprise). The last row is not
a bug on our side (Sec. 3b). For IRN amplitudes near the prior edge (log10_A > -12.5 with large
gamma), discovery and enterprise build `Phi^-1 + T^T N^-1 T` from normal equations, which
squares the condition number. A **long-double reference** (`tests/extended_precision.py`)
recomputes the same per-pulsar likelihood from the same float64 inputs with a 64-bit
mantissa; it whitens, runs Gram-Schmidt off the timing model and does the Cholesky in long
double. It agrees with ours to 2e-10 (J1705-1903), 4e-12 (J1751-2857) and 3e-11
(J2214+3000) at such a point. discovery is off by 2.0e-4, 6.5e-6 and 1.7e-5 at the same
points.

### 3b. Prior corners (review finding 2) and the square-root reduction

The first M1 version formed `E_a = A_GG - K^T K` and `d_a = b_G - K^T y`. When the intrinsic
RN dominates, this cancels catastrophically, with relative error ~ eps |A| / |R^-1|, i.e.
O(0.1) at log10_A = -11, gamma = 7. The reviewer's reproductions, rerun against a
**50-digit `decimal` reference**: `tests/decimal_reference.py` evaluates the *joint*,
unreduced enterprise-style system `Phi^-1 + blockdiag(A)` with central-difference
gradients, in the same precision.

| Point (single pulsar; HD with Gamma = 1 must equal CURN) | old CURN | old HD sigma | old HD "B" | **new, all three paths** |
|---|---|---|---|---|
| J2043+1711, both log10_A = -11.1, both gamma = 6.9: error in d logL / d gamma_RN (reference -1.505095) | 1.2e-5 | 1.1e-1 | 6.1e-3 | **5e-12** |
| same point, value error | 3.2e-6 | 1.5e-5 | 1.5e-5 | **9e-13** |
| B1937+21, both -11, both 7: value error (old HD - CURN was 6.98e-4) | 3e-9 | 7.0e-4 | 7.0e-4 | **5e-11** |
| same point, d/d gamma_RN error | 2e-9 | 2.6e-2 | 2.3e-2 | **4e-12** |

Two changes were needed:

* **Square-root stage 1/2** (`R_F`, `c`, `s_perp`; QR of `[R_F Phi^1/2; I]`). This removes
  both the `A`-squaring and the cancellation. The 50-digit reference is insensitive to
  0.5-ulp perturbations of `(R_F, c)` (changes of 1e-15 to 1e-12), so the remaining errors
  are algorithmic and now at 1e-12 to 1e-10.
* **Diagonal ORF split.** Without it, a common process far stronger than the IRN, e.g. IRN
  (-20, 0) with common (-11, 7), still produced errors up to 1e-4. There `Sigma'` inherits
  cond(A) ~ 1e12. With `lam0 phi^CP` absorbed per pulsar, `phi E <= 1/lam0` and the error
  drops to 1e-10.

Tests, all in `tests/test_corners.py` (`slow`, needs the data):

* **Scan, all 67 pulsars.** All 16 corners of the box {IRN log10_A -20/-11} x {gamma 0/7} x
  {common log10_A -18/-11} x {gamma 0/7}, plus the reviewer's points. The HD path with
  `Gamma = I` (both methods) must equal CURN in value and in **every** gradient component, including all 134 per-pulsar IRN components
  (<= 1e-7 relative; value tolerance 1e-6 for the 67-pulsar sum). Worst measured: 1.0e-10 (B1937+21, d/d gamma_RN, reviewer point).
* **Single pulsar vs the 50-digit reference.** Pulsars: J2043+1711, B1937+21, and the
  worst-conditioned pulsars by `max_k phi_k A_kk` at the IRN corner, chosen systematically
  (J1713+0747, J1909-3744, B1937+21). Points: the reviewer's two, (-11, 7, -18, 0),
  (-20, 0, -11, 7), and a posterior-like one. Checked: CURN and HD values (<= 1e-8) and
  gradients in all four parameters (<= 1e-7).
* **Correlated HD vs the 50-digit joint reference.** J1713+0747, J1909-3744, J2043+1711 and
  B1937+21 with their HD Gamma, at the corners and a posterior point, both methods: value
  errors 7e-12 to 1.8e-10, gradients within 1e-7.
* **Extended precision end to end.** Long-double TOA-level stage 1 feeding the 50-digit
  joint HD stage 2, vs our float64 pipeline (B1937+21, J1713+0747, J2043+1711): 1e-10 at
  posterior-like points and 3e-8 to 8e-8 at the corners. The corner residual comes from
  float64 vs long-double stage-1 rounding over up to 6e4 TOAs.
* **Full-PTA HD gradients** vs a 5-point stencil at a posterior sample, with the common
  process at (-11, 7), and with every IRN also at (-11, 7) (`test_full_pta_gradient_fd`).

### 3c. Free-spectrum domain (second review) and the analytic reduction VJP

Values in revision 2 were accurate everywhere, but the gradients still went through JAX's
generic QR derivative. Where the prior variances span many decades, that rule loses
accuracy. The reviewer's point: J2043+1711, `n_common = 30`, free spectrum with all
`log10_rho = -1.1` (inside the adopted prior), IRN (-11.1, 6.9). Errors there:

* d/d IRN log10_A: 5.9e-5 (HD) against a reference of -0.0926324.
* d/d log10_rho[2]: 6.8e-5.
* J2043+B1937 with HD or dipole: about 1e-5.

The reduction (`likelihood._reduce`) is now a `custom_vjp` taking the diagonal prior
variance `r`. Its backward pass uses only the accurately computed `E = (A^-1 + R)^-1` and
`d = E A^-1 b`:

    dq/dr_k = -d_k^2,   d log|I + R^1/2 A R^1/2| / dr_k = E_kk,
    dE/dr_k = -E e_k e_k^T E,   dd/dr_k = -E e_k d_k

These are exact identities for `q = r^T P r - b^T (R^-1 + A)^-1 b`. Everything upstream of
`r` (power law, free spectrum, `lam0 phi^CP`) and downstream (`Sigma'` core) is
differentiated as before. Errors against the 50-digit reference after the change, at the
reviewer's point:

| Case | value | d/d IRN log10_A | d/d IRN gamma | d/d log10_rho[2] | d/d log10_rho[0] |
|---|---|---|---|---|---|
| J2043, CURN | 2e-10 | 5e-12 | 3e-12 | 2e-11 | 4e-12 |
| J2043, HD Gamma = 1 (sigma and B) | 2e-10 | 9e-12 | 5e-12 | 2e-11 | 8e-12 |
| J2043+B1937, HD (sigma and B) | 4e-11 | 4e-12 | 2e-12 | 9e-12 | 3e-12 |
| J2043+B1937, dipole | 1e-11 | 1e-11 | 7e-12 | 7e-12 | 4e-12 |
| J2043+B1937, monopole (sigma / B) | 7e-11 / 1.3e-10 | 2e-11 | 1e-11 | 4e-13 | 4e-11 |

New tests (`tests/test_corners.py`):

* **Identity scans, all 67 pulsars, in three more configurations.** Each compares the
  `Gamma = I` HD path (both methods) with CURN, in value and every gradient component, at
  1e-7. Worst discrepancies:
  * power law, 30 common modes, 25 corner/interior points: 9e-14;
  * free spectrum, 14 modes: 8e-12;
  * free spectrum, 30 modes: 1.5e-10.

  The free-spectrum points cross 5 IRN settings (corners plus (-11.1, 6.9)) with 6 rho
  profiles. The profiles are all at -15.5, all at -1.0, alternating -15.5/-1.0 in both
  orders, all at -1.1 (the reviewer's point), and a mid-range ramp.
* **Correlated 50-digit checks**, J2043+1711 + B1937+21, `n_common = 30`, for HD, dipole and
  monopole, each with power law (-11.1, 6.9) and (-11, 7) and with free spectrum at -1.1 and
  alternating bounds. Value within 1e-8; gradients in two IRN amplitudes, an IRN slope and
  two or three common parameters within 1e-7 relative (12 test cases).

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
| CURN, 14-mode power law (67 batched 120 x 60 QRs) | 3.6 ms | 3.6 ms |
| HD, 14-mode power law, exact float64 | 9.2 ms | **14.9 ms** |
| HD, 14-mode power law, `grad_precision="mixed"` | 9.0 ms | **10.3 ms** |
| HD, 14-mode power law, `method="B"` (reference) | 9.6 ms | 32.5 ms |
| CURN, **30-mode free spectrum** | 3.7 ms | 3.6 ms |
| HD, **30-mode free spectrum** (67 x 60 = 4020 system), exact | 20.8 ms | **51.0 ms** |
| HD, 30-mode free spectrum, `grad_precision="mixed"` | 20.7 ms | **24.8 ms** |
| discovery HD (value only, same GPU) | 7.3 ms | |
| discovery CURN (value only) | 1.6 ms | |
| Stage-1 precompute, host numpy, all 67 pulsars | about 11 s, once | |

**Regression vs revision 1, accepted for correctness.** Revision 1 was 0.24/0.56 ms for CURN
and 5.9/12.0 ms for HD. The batched float64 QR (cuSOLVER) costs about 3.4 ms, where the
per-pulsar Cholesky cost 0.3 ms. A hand-written vectorised Householder sweep brings the value
down to 1.0 ms, but its autodiff gradient is slower (5.5 ms). Revision 3's analytic VJP for
the square-root stage made the gradient nearly free, so CURN value + gradient is 3.6 ms. The
remaining cost is the forward QR.

On this GPU float64 runs at about 1/64 of the float32 rate. The float64 matmul peak measured
1.76 TFLOPS. The float64 Cholesky of the 1876 system takes 5.4 ms (cuSOLVER), and that sets
the floor for the HD value. Things that did not help: blocked Cholesky in JAX (8 to 9 ms);
recursive GEMM-based Cholesky (12 ms); cuBLAS FP64 emulation through
`CUBLAS_EMULATE_DOUBLE_PRECISION` (no effect through XLA).

**Mixed precision is opt-in and affects only the gradient.** With `grad_precision="mixed"`
the backward pass inverts the symmetrically equilibrated Cholesky factor in float32. The
likelihood value stays float64 and bit-identical. Measured gradient error, maximum
component error over the largest component: 4e-5 at a posterior sample and 3e-6 to 9e-5 at
prior draws. Per component, relative to max(|g|, 1), it is at most 1.1e-4. HMC/NUTS stays exact with a deterministic
approximate gradient, because leapfrog with any deterministic gradient field is still
reversible and volume preserving and the Metropolis step uses the exact float64 value;
only the acceptance rate can suffer. Float64 gradients remain the default ("correctness
over speed"). Both 14-mode paths miss the ~10 ms target for value + gradient: exact
14.9 ms, mixed 10.3 ms. The 30-mode free-spectrum HD system is 4020-dimensional and takes
51 ms (24.8 ms mixed).

**Note added 2026-10-07 (performance study, `docs/PERF.md` Sec. 3a).** XLA:CPU in jaxlib 0.11.2
miscompiles a batched dot with a broadcast operand fused into a multiply + reduce (YNNPACK library
fusion). The likelihood's reducer VJP is affected on the **CPU backend** when it receives constant
(broadcast) cotangents on E; the production likelihood's own gradient was verified unaffected on
CPU (compiled vs eager <= 9e-11 at 52 NG15 points across CURN/HD/dipole/monopole and both spectra),
and the GPU backend, on which every number in this document was produced, does not use YNNPACK
(compiled = eager to <= 2.4e-11). The production backward rule now keeps the affected products
behind `optimization_barrier`s (values bit-identical, gradients <= 1e-14, GPU speed unchanged), and
`ptagwb` disables the fusion via `XLA_FLAGS` at import (warning if JAX was initialised first).

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
| HD system `B = I + L^T E L` | Implemented as the reference path. The default factorises the congruent `Sigma' = Gamma'^-1 (x) I + phi^1/2 E phi^1/2` with a custom VJP: same value, 2x faster value + gradient. |
| "First absorb intrinsic RN ... Woodbury" with `H`, `P`, `q`, `d`, `E` as written | Taken literally (normal equations, `E = G^T P G` by subtraction), this is numerically unstable in the prior corners (Sec. 3b). We use a square-root form (QR of `[R_F Phi^1/2; I]`, subtraction-free E and d) plus a diagonal ORF split. |
| Free-spectrum prior `log10_rho ~ U[-9, -4]` | **Wrong for the released production chains**: they sampled `U[-15.5, -1.0]` (Sec. 7; errata added to the spec). |
| Timing marginalisation "diagnose rank deficiency" | Enforced: `TimingRankError` if s_min/s_max < 1e-10 for the column-normalised design matrix. NG15 worst case: 4.9e-7 (J1853+1303). No silent truncation. |
| Timing marginalisation | Exact projection via QR on the whitened basis. No `1e40` enters the numerics, only the constant. |
| Radio frequency | enterprise's `psr.freqs` is the **SSB** frequency (`barycentric_radio_freq`); exported as `freqs`, with topocentric `freqs_topo` alongside. |
| Split ao/gbt files | Not in the baseline. The discovery and tutorial data sets contain only the 67 combined pulsars, and the split files exist for the telescope cross-validation (GWB Sec. 5.4). The 8 files are B1937+21ao/gbt, J1600-3053gbt, J1643-1224gbt, J1713+0747ao/gbt, J1903+0327ao and J1909-3744gbt. They load on request: `load_pulsars([...names])` or `find_par_tim(split_only=True)`. |
| PINT version | Not in the spec. Front-end deviations are documented in Sec. 2: J1713+0747 DDK/DMX edges, the J1024-0719 F3 placeholder, ELL1 partials, and about 1 ns rms residual differences. J1024-0719 F3 is now an explicit exception in `data.FROZEN_PARAMS`. Ingestion checks that F3 is free, zero-valued and without uncertainty, and that the result has 149 columns; otherwise it raises. |

## 6. Open concerns

* J1713+0747 differs systematically from the analysis-time front end (DDK convention, two
  DMX-edge TOAs). The impact is about 0.03 std in logL over the posterior, but a
  bit-faithful reproduction of the feathers would need the old PINT DDK convention.
* Mixed-precision gradients (up to 1e-4 relative per component) need a NUTS acceptance check
  in M2 before use.
* The corner tests cover all 67 pulsars through the identity scan (16 corners plus 2
  points), but the 50-digit comparisons cover only 2 to 4 pulsars at a time. The scan
  cannot detect an error common to the CURN and HD paths. Both use the same square-root
  stage, which was validated only on the 50-digit subsets.
* Speed regressed compared with revision 1 (Sec. 4). The forward batched QR (3.4 ms) and the
  4020-dimensional free-spectrum HD system (51 ms value + gradient) are the M2 throughput
  bottlenecks.
* Fixed-gamma common amplitude prior: paper U[-18, -14] vs the released fixed-gamma spline
  core U[-18, -11]. Unresolved for HD^13/3 and CURN^13/3; recorded in `config.PRIORS`, to be
  settled in M2.
* The absolute-value checks against the released chains are limited by their float32 `logl`
  column (+-0.25).

## 7. Priors: free-spectrum prior resolved from provenance (review finding 1)

`config.FREESPEC_LOG10_RHO_PRIORS["production"] = (-15.5, -1.0)` is the default
(`PRIORS["freespec_log10_rho"]`). Here `log10_rho` is enterprise's parameter: log10 of the
coefficient RMS in seconds, with `phi_k = 10^(2 log10_rho_k)`. Evidence:

1. **Released HD free-spectrum chains.** These are the Fig. 1(a) core
   `30fCP_30fiRN_3A_freespec_chain.core` (490,000 samples) and the tutorial
   `presampled_cores/hd_30f_fs.core`. In both, `lnpost - lnlike` is constant, with std
   3.6e-7. Its means are **-357.8144861503552** (figure core) and -357.8144861475138
   (tutorial core). The other 134 parameters are 67 x IRN `U[-20, -11] x U[0, 7]`, which
   their sample ranges confirm, contributing -67 ln 63 = -277.590026668233. Assuming equal,
   independent uniform widths w for the 30 rho, this leaves -30 ln w, so
   **w = 14.499999999675** (figure core) and 14.499999998301 (tutorial core). For comparison,
   -67 ln 63 - 30 ln 14.5 = -357.814486151029. The equal-width decomposition is an
   assumption; the normaliser itself only fixes the total prior volume.
2. **Lower edge.** In 28 of the 30 frequencies the samples reach **-15.50** (to 3
   decimals) and go no lower. Width 14.5 then fixes the upper edge at **-1.0**. No
   sample exceeds -5.25, so the upper edge is set by the prior width, not by sample extrema.
3. **Ceffyl KDEs** (Zenodo 8060824, our `ng15_kde_freespec_v1`). The HD, HD+MP+DP and
   HD+MP+DP+CP free-spectrum grids span exactly [-15.5, -1.0]. The CURN (CP) grid spans
   [-15.1, -0.9], and the DMGP variant [-10, -5.3].
4. **Not the plotting range.** The paper's Table 1 says "log-Uniform in rho_i [-18, -8]".
   Read as log10(phi/s^2), that is log10_rho in [-9, -4], and the Fig. 1(a) notebook
   (`upper_left.ipynb`) histograms the chain on `np.linspace(-9, -4, 400)` and resamples
   from those truncated, renormalised histograms. **Raw-chain reproduction** therefore needs
   `U[-15.5, -1]`. **Figure reproduction** additionally applies that [-9, -4]
   truncation/renormalisation to the marginals. In the raw chain, 72% of
   `gw_hd_log10_rho_29` samples lie below -9 (median -10.99).
5. enterprise_extensions defaults (3.0.3 installed: `Uniform(-10, -4)` for the
   `log-uniform` prior, `Uniform(-9, -4)` otherwise) match neither; the runs passed explicit
   bounds. The runtime info and model pickles of the free-spectrum run are not in any
   release bundle we have. The width argument rests on the stored `lnpost - lnlike`.

Status: HD^free is **inferred** as `U[-15.5, -1.0]` from three independent pieces of
evidence: the normaliser, the sample floor and the KDE grid. It is not a recovered sampler
configuration; no run script, runtime info or model pickle of the free-spectrum run is
available. **CURN^free is unverified**: no released
CURN free-spectrum chain is available, and the Ceffyl CP grid [-15.1, -0.9] differs. M2 must
check this before comparing CURN^free marginals or evidences. `paper_table1 = (-9, -4)` is
kept as an alternative.

Other priors in `config.PRIORS`:

* IRN `U[-20, -11] x U[0, 7]` and varied-gamma common `U[-18, -11] x U[0, 7]`: verified from
  the m2a/m3a and 14f_PL_hd_crn chain metadata.
* Fixed-gamma common amplitude: paper `U[-18, -14]` vs `U[-18, -11]` in the fixed-gamma
  spline-ORF core. Open (Sec. 6).
