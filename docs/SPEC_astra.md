The baseline is **fixed white noise + marginalized timing model/DMX + intrinsic RN + one common process**. Encode it explicitly: `model_2a` implements **CURN**, while HD requires a correlated common GP. Current library defaults are not a frozen specification of the published analysis. [Source: `model_2a`](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/models.py#L622-L798).

Below, **UNVERIFIED** means I could not establish the requested detail from the inspected sources. Numerical tolerances in the validation checklist are proposed engineering requirements.

1. **Data and timing front end**

| Item | Reproduction requirement | Source |
|---|---|---|
| Selection | **67 narrowband pulsars**, baseline **≥3 yr**; the conclusion also writes “>3 yr.” | [GWB §II, §VI](https://arxiv.org/html/2306.16213#S2) |
| Exact membership | All pulsars in the data paper’s **Table 5 except J0614−3329**, whose span is **2.4 yr**. Preserve the release’s pulsar names, including B names. | [Data paper, Table 5](https://arxiv.org/html/2306.16217) |
| Timing reference | **DE440**, **TT(BIPM2019)**. | [Data paper §IV](https://arxiv.org/html/2306.16217#S4) |
| Input products | Final selected narrowband TOAs and matching fitted par files. Preserve release cuts, jumps, fitted parameters and DMX intervals. | [Data paper §III–IV](https://arxiv.org/html/2306.16217) |
| Fourier span | \(T=\max_{a,i}t_{ai}-\min_{a,i}t_{ai}\): extrema over the **whole selected PTA**, not the longest individual pulsar span. | [`model_utils.get_tspan`](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/model_utils.py) |
| Reference value | **505861299.1401644 s**, explicitly used by the collaboration’s figure code; **16.0297772689 Julian yr**, calculated from it. Paper rounds to **16.03 yr**. | [Figure 1(a) notebook, frequency cell](https://github.com/nanograv/15yr_stochastic_analysis/blob/main/data_release/figure_1/upper_left.ipynb); [GWB §II](https://arxiv.org/html/2306.16213#S2) |

Do not substitute `16.03 * year` for the precise span. Recompute it from your exported TOAs and investigate any mismatch.

For the PINT export, reproduce the enterprise `PintPulsar` interface:

```text
t        = model.get_barycentric_toas(toas), converted to seconds
residual = Residuals(toas, model).time_resids, seconds
sigma    = toas.get_errors(), seconds, BEFORE EFAC/EQUAD scaling
M        = model.designmatrix(toas), including the phase-offset column
position = fixed unit sky vector at the timing-model reference position
```

Sort all associated arrays identically. Observatory MJDs and barycentric TOAs are separate quantities. [Source: `PintPulsar`](https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L345-L404).

Use the full release for the final comparison: the collaboration explicitly warns that tutorial data are reduced and may not reproduce the published result exactly. **Production file checksums and historical dependency commits: UNVERIFIED.** [Repository notice](https://github.com/nanograv/15yr_stochastic_analysis).

2. **Per-pulsar model**

For pulsar \(a\),

\[
r_a=M_a\epsilon_a+F_a c_a+n_a,
\]

with independent intrinsic RN parameters for every pulsar, shared common-process spectral parameters, and analytically integrated timing/Fourier coefficients.

**White noise.** For system \(s(i)\),

\[
N_{ij}
=\delta_{ij}E_{s(i)}^2\left[\sigma_i^2+Q_{s(i)}^2\right]
+\sum_{s,e}J_s^2U_{se,i}U_{se,j}.
\]

This is the **T2 EQUAD convention**:

- \(E=\mathrm{EFAC}\), dimensionless.
- \(Q=10^{\texttt{log10\_t2equad}}\) seconds.
- \(J=10^{\texttt{log10\_ecorr}}\) seconds.
- ECORR is outside the EFAC scaling.
- Set `tnequad=False`.

The alternative TempoNest convention is \(E^2\sigma^2+Q_{\rm TN}^2\); conversion requires \(Q_{\rm TN}=E Q_{\rm T2}\). Renaming a dictionary key alone is incorrect. [Noise paper §III.1, Eq. (2), footnote 2](https://arxiv.org/html/2306.16218); [`combined_ndiag`, `TNEquadNoise`](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/white_signals.py#L60-L98).

**System selection and ECORR:**

| Detail | Required behavior |
|---|---|
| Grouping | Receiver/backend combination, not backend hardware alone. |
| Flag resolution | Enterprise preference is nonempty `group`, then `g`, `sys`, `i`, `f`, then `fe+"_"+be`. Thus `-group` can override `-f`. |
| ECORR selection | `model_2a` enables ECORR for narrowband NANOGrav data and uses the block’s NANOGrav-backend selection. Verify masks against the oracle; missing `pta` metadata can accidentally disable ECORR. |
| Epoch algorithm | Within each selected system, sort **barycentric TOAs in seconds**. Start a bucket at its first TOA; append while \(t-t_{\rm first}<1\,\mathrm{s}\). Otherwise start another bucket. |
| Singleton buckets | Discard: `nmin=2`. |
| Epoch meaning | Not integer-MJD bins, not a sliding adjacent-gap rule, and not arbitrary half-hour bins. |

Sources: [`backend_flags`](https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L286-L309), [`selections`](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/selections.py#L118-L128), [`white_noise_block`](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/blocks.py#L35-L160), [`create_quantization_matrix`](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/utils.py#L706-L729), [`EcorrKernelNoise`](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/white_signals.py).

Hold WN parameters fixed at the released single-pulsar noise estimates. The tutorial loads `15yr_wn_dict.json` and calls `set_default_params`; independently verify its suitability for your full release. Require every selected-system WN parameter to resolve explicitly. **The exact production dictionary bytes and MAP extraction procedure are UNVERIFIED here.** Intrinsic RN remains sampled even if the dictionary contains RN values. [Parameter-estimation notebook](https://github.com/nanograv/15yr_stochastic_analysis/blob/main/tutorials/parameter_est.ipynb); [Methods §III.3.4](https://arxiv.org/html/2306.16223).

**RN and DM.** Use **30 intrinsic RN frequencies** and **14 common power-law frequencies**. Retain DMX in the timing model; baseline has no additional DM GP. [Collaboration model construction](https://github.com/nanograv/15yr_stochastic_analysis/blob/main/tutorials/parameter_est.ipynb); [`model_2a`](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/models.py#L717-L754).

**Timing marginalization.**

- Mathematical target: flat, improper prior on linear timing offsets.
- Enterprise’s basis representation uses coefficient variance **\(10^{40}\)**.
- Default `TimingModel`: normalize each design-matrix column by its Euclidean norm.
- `use_svd=True`: replace \(M\) by the thin-SVD left singular vectors.
- `MarginalizingTimingModel` projects timing parameters analytically and retains a normalization contribution \(m\log(10^{40})\).

These implementations must agree in parameter-dependent likelihood, but arbitrary timing-basis normalization changes additive constants under the improper-prior convention. Match the oracle’s basis and constants before demanding absolute likelihood equality. **Historical production SVD setting: UNVERIFIED**; the current tutorial explicitly uses SVD. [Timing-model source](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/gp_signals.py#L285-L316), [basis normalization](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/utils.py#L879-L913), [`MarginalizingNmat`](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/gp_signals.py#L948-L1007).

3. **Fourier basis and spectral normalization**

Let \(Y=365.25\times86400=31557600\) seconds and \(f_{\rm yr}=Y^{-1}\). Enterprise uses a Julian year. [Constants source](https://github.com/nanograv/enterprise/blob/master/enterprise/constants.py#L20-L23).

For \(k=1,\ldots,N_f\),

\[
f_k=k/T,\qquad
F_{i,2k-2}=\sin(2\pi f_k t_i),\qquad
F_{i,2k-1}=\cos(2\pi f_k t_i).
\]

No zero-frequency mode, Fourier-column normalization or extra \(\sqrt2\). Enterprise returns frequency labels repeated as \((f_1,f_1,f_2,f_2,\ldots)\). [Fourier source](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/gp_bases.py#L25-L87).

The one-sided residual PSD and coefficient variance are

\[
S_r(f)=\frac{A^2}{12\pi^2}f_{\rm yr}^{\gamma-3}f^{-\gamma},
\qquad
\phi_k=S_r(f_k)\Delta f_k.
\]

- \(A\): dimensionless characteristic strain at \(f_{\rm yr}\).
- \(S_r\): \(\mathrm{s^3}=\mathrm{s^2/Hz}\).
- \(\phi_k\): \(\mathrm{s^2}\).
- **Each** sine and cosine coefficient has variance \(\phi_k\).
- Regular grid: \(\Delta f_k=1/T\).
- Enterprise’s general implementation takes differences of distinct frequencies, prepending zero, then repeats each width for sine/cosine.

The function named `powerlaw` returns **coefficient variances**, including \(\Delta f\), rather than the continuous PSD alone. [Exact `powerlaw` implementation](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/gp_priors.py#L12-L15).

Construct the common basis as the first common-frequency columns of the RN basis. Use identical time origin and phases across pulsars. A common shift of time origin is harmless; independent pulsar shifts change HD cross-covariances unless the coefficient covariance is transformed correspondingly.

For shared modes,

\[
\operatorname{Cov}(c_{akq},c_{blq'})
=\delta_{kl}\delta_{qq'}
\left[\delta_{ab}\phi^{\rm RN}_{ak}
+\Gamma_{ab}\phi^{\rm CP}_k\right].
\]

Above the common cutoff only intrinsic RN remains. This is equivalently implemented with separate latent RN/common coefficients or a merged Fourier basis with **summed variances**.

4. **Overlap reduction functions**

Set \(x_{ab}=(1-\hat p_a\cdot\hat p_b)/2\).

| Process | Covariance multiplier \(\Gamma_{ab}\) |
|---|---|
| HD | \(1\) for \(a=b\); otherwise \(\frac32x\ln x-\frac14x+\frac12\) |
| CURN | \(\delta_{ab}\) |
| Monopole, ideal | \(1\) |
| Dipole, ideal | \(\hat p_a\cdot\hat p_b\) |
| Enterprise monopole/dipole | Above, with diagonal **\(1+10^{-5}\)** |

Thus HD tends to **\(1/2\)** for distinct, nearly aligned pulsars and has auto-correlation **1**. Implement \(x\log x\to0\) safely and identify auto terms by pulsar index. The monopole/dipole diagonal addition regularizes otherwise low-rank matrices; retain it for exact source-oracle comparisons. Do not substitute `gw_monopole` or `gw_dipole`, which are different functions. [ORF source](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/model_orfs.py#L220-L244); [Methods §II.1, Eqs. (15)–(18)](https://arxiv.org/html/2306.16223).

5. **Priors**

All intervals below refer to explicitly sampled variables.

| Variable | Prior |
|---|---|
| Each \(\log_{10}A_{\rm RN}\) | \(U[-20,-11]\) |
| Each \(\gamma_{\rm RN}\) | \(U[0,7]\) |
| Common \(\log_{10}A\), variable \(\gamma\) | \(U[-18,-11]\) |
| Common \(\gamma\) | \(U[0,7]\) |
| Common \(\log_{10}A\), fixed \(\gamma\) | \(U[-18,-14]\) |
| Fixed common \(\gamma\) | Exactly \(13/3\) |
| Free-spectrum \(\log_{10}(\phi_k/\mathrm{s^2})\) | \(U[-18,-8]\) |

Source: [GWB Appendix B, Table 1](https://arxiv.org/html/2306.16213#A2).

For enterprise’s free-spectrum parameter \(u_k=\texttt{log10\_rho}_k\),

\[
\phi_k=10^{2u_k}\mathrm{s^2}.
\]

Therefore the paper’s variance bounds translate to **\(u_k\sim U[-9,-4]\)**. Set them explicitly: current `common_red_noise_block` defaults to a different lower bound. [Free-spectrum conversion](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/gp_priors.py#L23-L32); [current block defaults](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/blocks.py#L1504-L1520).

Uniform-in-log amplitude is not uniform-in-amplitude. Preserve prior normalization in evidence calculations.

6. **Targets and statistical procedures**

Amplitudes below are at \(1\,\mathrm{yr}^{-1}\); intervals are marginal **5th–95th percentiles**, not joint posterior bounds.

| Target | Value | Citation |
|---|---|---|
| HD, \(\gamma=13/3\) | \(A=2.4^{+0.7}_{-0.6}\times10^{-15}\) | [GWB §III, Fig. 1(b)](https://arxiv.org/html/2306.16213#S3) |
| HD, variable \(\gamma\) | \(A=6.4^{+4.2}_{-2.7}\times10^{-15}\); \(\gamma=3.2^{+0.6}_{-0.6}\) | [GWB §III, Fig. 1(b)](https://arxiv.org/html/2306.16213#S3) |
| CURN amplitude/interval, either slope treatment | **UNVERIFIED separately** | Do not relabel HD summaries |
| HD/CURN BF, variable \(\gamma\) | Approximately **200**, 14 frequencies; **1000**, 5 frequencies | [GWB §III](https://arxiv.org/html/2306.16213#S3) |
| Fixed-\(\gamma\) BF | Described as similar; exact values **UNVERIFIED** | [GWB §III](https://arxiv.org/html/2306.16213#S3) |
| Noise-marginalized OS S/N | **\(5\pm1\)** variable \(\gamma\); **\(4\pm1\)** fixed \(\gamma\); mean ± SD | [GWB §IV, Fig. 4](https://arxiv.org/html/2306.16213#S4) |
| Fixed-noise OS S/N | **UNVERIFIED** | Requires specified fixed parameter vector |
| Bayesian phase-shift \(p\) | **\(10^{-3}\)** | [GWB §III, Fig. 3](https://arxiv.org/html/2306.16213#S3) |
| Bayesian sky-scramble \(p\) | **\(1.6\times10^{-3}\)** | [GWB Appendix F, Fig. 14](https://arxiv.org/html/2306.16213#A6) |
| OS \(p\): phase shifts / simulations / analytic | **\(5\times10^{-5}\) / \(1.8\times10^{-4}\) / \(1.9\times10^{-4}\)** | [GWB §IV, Fig. 3](https://arxiv.org/html/2306.16213#S4) |
| OS sky-scramble \(p\) | Reported **\(<10^{-4}\)** | [GWB Appendix F](https://arxiv.org/html/2306.16213#A6) |

The released posterior-summary notebook provides sharper regression references:

| HD model | Amplitude median [5%,95%], \(10^{-15}\) | Gamma median [5%,95%] |
|---|---|---|
| Fixed slope | 2.403918 [1.815212, 3.068923] | Fixed |
| Variable slope | 6.385421 [3.699923, 10.558530] | 3.247687 [2.659568, 3.839772] |

These are stored-chain summaries, not precision requirements for a new Monte Carlo run. [Figure 1(b) data notebook, quantile cells](https://github.com/nanograv/15yr_stochastic_analysis/blob/main/data_release/figure_1/lower_left.ipynb).

**Free spectrum:** reproduce Fig. 1(a) using **30 common frequencies**, confirmed by its released chain and plotting code. Plot \(\log_{10}\sqrt{\phi_k/\mathrm{s^2}}\), not log PSD. [Figure notebook](https://github.com/nanograv/15yr_stochastic_analysis/blob/main/data_release/figure_1/upper_left.ipynb). HD power appears in bins **1–5 and 8**. [GWB §V.2, Fig. 6](https://arxiv.org/html/2306.16213#S5.SS2).

**Binned HD:** Fig. 1(c), **15 bins**, MAP CURN\(^{13/3}\), pair-covariance-aware errors. [GWB §IV](https://arxiv.org/html/2306.16213#S4). Use the released bin edges and covariance calculation for exact plotting; an independent-pair weighted plot is a different diagnostic. [Figure 1(c) notebook](https://github.com/nanograv/15yr_stochastic_analysis/blob/main/data_release/figure_1/upper_right.ipynb).

**Sampler/evidence:** PTMCMC; HD/CURN product-space occupancy ratios, with reweighting also described. [GWB Appendix B](https://arxiv.org/html/2306.16213#A2). For implementation,

\[
B_{\rm HD,CURN}
=\frac{n_{\rm HD}}{n_{\rm CURN}}
\frac{\pi_{\rm CURN}}{\pi_{\rm HD}},
\]

including any artificial model-occupancy weighting correction. With identical parameter priors, importance reweighting gives

\[
B_{\rm HD,CURN}
=E_{\theta\sim p(\theta|d,\mathrm{CURN})}
\left[e^{\ell_{\rm HD}(\theta)-\ell_{\rm CURN}(\theta)}\right].
\]

Use log-sum-exp, weight effective sample size and autocorrelation-aware uncertainty. **The exact estimator/run provenance of every published BF is UNVERIFIED.** [Methods §III.5.1–III.5.2](https://arxiv.org/html/2306.16223).

For OS, include intrinsic RN **and common auto-power** in each pulsar’s filtering covariance. Evaluate the statistic separately for each posterior draw; average the resulting statistics, not their input parameters. Use exact fixed slope or each draw’s slope consistently. S/N is not itself Gaussian significance. [OS implementation](https://github.com/nanograv/enterprise_extensions/blob/master/enterprise_extensions/frequentist/optimal_statistic.py#L159-L265).

7. **JAX likelihood implementation**

The following is an algebraic implementation prescription; matrix dimensions are derived from the configuration above.

First absorb intrinsic RN independently:

\[
C_a=N_a+F_{{\rm RN},a}R_aF_{{\rm RN},a}^{T}.
\]

Apply Woodbury and determinant lemmas using the diagonal-plus-ECORR structure of \(N_a\). Then eliminate timing parameters:

\[
H_a=M_a^TC_a^{-1}M_a,
\]

\[
P_a=C_a^{-1}
-C_a^{-1}M_aH_a^{-1}M_a^TC_a^{-1}.
\]

Here \(P_a\) is a projected precision operator. Never materialize its full TOA-space matrix.

With \(G_a=F_{{\rm CP},a}\), compute

\[
q_a=r_a^TP_ar_a,\quad
d_a=G_a^TP_ar_a,\quad
E_a=G_a^TP_aG_a,
\]

\[
h_a=\log|C_a|+\log|H_a|.
\]

In pulsar-major coefficient ordering,

\[
Q=\Gamma\otimes
\operatorname{diag}(\phi_1,\phi_1,\ldots,\phi_{N_g},\phi_{N_g}),
\]

\[
\Sigma=Q^{-1}+\operatorname{blockdiag}(E_a),\qquad d=\operatorname{concat}(d_a).
\]

Then, up to a fixed normalization convention,

\[
\boxed{
\ell=-\frac12\left[
\sum_a(q_a+h_a)
+\log|Q|+\log|\Sigma|
-d^T\Sigma^{-1}d
\right].
}
\]

The final HD system has dimension

\[
67\times2\times14=\boxed{1876}.
\]

For CURN it separates by pulsar. Irregular sampling and timing projection couple frequencies within each pulsar, so the HD likelihood generally does **not** split into independent frequency-by-frequency systems.

A better-conditioned common-process calculation uses \(Q=LL^T\):

\[
B=I+L^TEL,\qquad z=L^Td,
\]

\[
\log|Q|+\log|\Sigma|=\log|B|,
\qquad
d^T\Sigma^{-1}d=z^TB^{-1}z.
\]

Cache fixed-WN, timing-projected residual/Fourier contractions; perform the RN updates in coefficient space. This avoids repeatedly processing all TOAs. The two-stage marginalization principle is described in [Methods §II.2](https://arxiv.org/html/2306.16223).

The direct enterprise formulation is

\[
\Sigma=T^TN^{-1}T+\Phi^{-1}
=\texttt{TNT}+\texttt{phiinv}.
\]

Its determinant and quadratic terms provide the oracle for the reduced calculation. [Likelihood source](https://github.com/nanograv/enterprise/blob/master/enterprise/signals/signal_base.py#L193-L246).

Implementation requirements:

- Enable JAX float64 before constructing arrays.
- Use Cholesky solves and log determinants; avoid explicit inverses.
- Do not construct \(10^{40}MM^T\) in TOA space.
- Diagnose rank deficiency before SVD truncation; arbitrary truncation changes the timing model.
- For absolute oracle equality, reproduce its Gaussian constants and timing-prior constant. Current source includes \(n_{\rm TOA}\log(2\pi)\); timing marginalization includes \(m\log(10^{40})\).
- Treat unexplained Cholesky failure as a bug until shown otherwise; arbitrary jitter changes the likelihood.
- Explicitly track coefficient ordering and common auto-power to prevent double counting.

8. **Validation order and discrepancy localization**

| Order | Checkpoint |
|---|---|
| 1 | Freeze par/tim/noise hashes, clock files, ephemeris, PINT/oracle versions and pulsar ordering. Compare per-pulsar TOA counts and exact masks. |
| 2 | Compare PINT-exported residuals, raw errors, barycentric TOAs, sky vectors and timing design-matrix column spaces to the oracle. Separate front-end differences from likelihood differences. |
| 3 | Compare every backend mask and ECORR bucket exactly. Compare \(N^{-1}v\) and \(\log|N|\) against dense calculations on small subsets. |
| 4 | Compare Fourier matrices and coefficient variances at fixed parameters. Confirm common columns coincide with the RN subset. |
| 5 | Single-pulsar WN+timing, then RN, then CURN likelihoods: target absolute difference **≤\(10^{-6}\)** with identical arrays, basis and constants. Also compare likelihood differences. |
| 6 | Small synthetic PTA: dense covariance versus Woodbury, merged versus separate bases, and reduced HD system. Check JAX gradients against finite differences away from boundaries. |
| 7 | Full-PTA fixed-parameter likelihoods against enterprise for CURN and HD, across plausible amplitudes/slopes. Replace HD ORF with identity and recover CURN. |
| 8 | Recover CURN posterior; compare released samples using Monte Carlo uncertainty. Check individual RN posteriors, not just common parameters. |
| 9 | Recover HD posterior and compare direct sampling with CURN reweighting. Check model-versus-itself BF equals unity. |
| 10 | Reproduce OS at identical parameter vectors, then its noise-marginalized distribution. |
| 11 | Reproduce free-spectrum and angular-correlation diagnostics. |
| 12 | Only then run phase-shift/sky-scramble backgrounds and published BF comparisons. |

The most damaging mistakes are wrong EQUAD scaling; already-scaled TOA errors; incorrect ECORR groups; local or rounded Fourier spans; independent common-process phase origins; missing \(\Delta f\); an extra factor of two; confusing coefficient RMS with variance; HD diagonal normalization errors; fixing intrinsic RN accidentally; adding a DM GP to DMX baseline; and comparing different priors, frequency cutoffs or likelihood constants.
---

## Errata (added 2026-10-06 after M1 validation and review; the text above is unchanged)

These corrections come from our own verification against the released NG15 products and the
installed enterprise and discovery sources. Details and evidence are in
`docs/M1_VALIDATION.md`.

1. **Sec. 5, free-spectrum prior: incorrect for the released chains.** The text derives
   `log10_rho ~ U[-9, -4]` from the paper's Table 1 ("log-Uniform in rho_i [-18, -8]"). The
   released HD free-spectrum production chains (Fig. 1(a) core
   `30fCP_30fiRN_3A_freespec_chain.core`, tutorial `hd_30f_fs.core`) actually sampled
   **`log10_rho ~ U[-15.5, -1.0]`** (inferred, not recovered from a sampler config).
   Evidence: `lnpost - lnlike` is constant, with mean -357.8144861503552 in the figure core.
   After the 67 IRN `U[-20,-11] x U[0,7]` priors, and assuming equal widths, that implies
   14.499999999675 per rho (exactly 14.5 gives -357.814486151029); samples reach -15.50 and go no lower; the Ceffyl HD KDE grids span exactly
   [-15.5, -1]. [-9, -4] is the range the Fig. 1(a) notebook uses to histogram, truncate and
   renormalise the marginals before plotting. Raw-chain reproduction needs [-15.5, -1];
   reproducing the figure additionally needs that truncation. The CURN^free prior is
   unverified. The variance-to-RMS conversion in Sec. 5 is correct.
2. **Sec. 7, RN absorption.** Implemented literally (normal equations, then
   `E = G^T P G` and `d = G^T P r` by subtracting the RN-projected part), the prescription
   cancels catastrophically when the intrinsic RN or the common process dominates, as at
   the prior corners log10_A -> -11, gamma -> 7. Gradient errors reach O(0.1) relative in
   float64. Use a square-root formulation, e.g. QR of `[R_F Phi^1/2; I]` with
   `F^T P F = R_F^T R_F`, subtraction-free `E` and `d`, and a diagonal split of Gamma (see
   `ptagwb/likelihood.py`).
3. **Sec. 2, "Historical production SVD setting: UNVERIFIED".** Verified: SVD. Only the SVD
   timing basis reproduces the absolute `logl` of the production chains; the unit-norm basis
   is off by 1665.
4. **Sec. 2, "exact production dictionary bytes: UNVERIFIED".** Verified: `v1p1_wn_dict.json`
   equals the 645 fixed WN constants in the production chains' enterprise runtime info.
5. **Sec. 1, sky vector.** enterprise (and therefore the released feathers and chains) uses
   B1950-equinox coordinates for pulsars whose name contains "B", about 0.5 deg from ICRS
   (pyephem `Equatorial(..., epoch="1950")`). Our default is ICRS;
   `position="enterprise"` reproduces the released vectors.
6. **Sec. 7, likelihood constants.** The `n_TOA log 2 pi` term is present in enterprise 3.5
   but not in enterprise 3.3.1 or discovery. The production chains used 3.3.1.
7. **Sec. 4, monopole/dipole regulariser.** discovery uses 1 + 1e-6, enterprise 1 + 1e-5.
8. **Sec. 5, fixed-gamma common amplitude.** Table 1 says `U[-18, -14]`, but the released
   fixed-gamma spline-ORF core used `U[-18, -11]`. **Resolved in M2: `U[-18, -11]`.** Every
   released fixed-gamma product whose prior is recoverable used it: the spline-ORF core stores
   `gw_crn_log10_A:Uniform(pmin=-18, pmax=-11)` and its `lnpost - lnlike` equals
   `-67 ln 63 - ln 7 - 7 ln 1.8`; the tutorial CURN^13/3 vs HD^13/3 product-space core has
   `lnpost - lnlike = 2(-67 ln 63) - 2 ln 7` and inactive-model amplitudes spanning
   [-18.00, -11.00]. Posteriors never approach -14, so the choice only moves evidences against
   IRN by ln(7/4) and cancels in HD/CURN (`docs/M2_RESULTS.md`).
9. **Sec. 5, CURN free spectrum (added in M2).** No released CURN^free chain or normaliser
   exists in the bundles we have; the only provenance is the Ceffyl v1 CP KDE grid
   [-15.1, -0.9]. Inferred (weak) and not needed for M2, which runs HD^free only.
