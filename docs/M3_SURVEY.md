# M3 survey: the five public PTA data sets, their methods, and what PINT makes of them

Companion to [`M3_PLAN.md`](M3_PLAN.md). Contents:

1. Downloads: what was fetched (MPTA status and sizes) and what does not exist.
2. Per-data-set inventory: pulsars, spans, TOAs, systems, NB/WB, clock/ephemeris, par/tim
   conventions, available noise models.
3. Method research: how each paper handles overlapping pulsars, noise, timing model and the
   common process, with headline numbers.
4. Pulsar overlap across the five PTAs.
5. PINT smoke load of every par/tim, with failures categorised and fixes.
6. Appendix: generated tables (`scripts/m3_survey_report.py`).

Reproduce (CPU only, about 1 h on 16-20 cores; nothing in `src/ptagwb` is used):

```bash
python scripts/fetch_data.py --group secondary m3          # data (raw files are git-ignored)
OMP_NUM_THREADS=1 python scripts/m3_survey.py --canonical  # -> data/processed/m3_survey/
python scripts/m3_survey_report.py > data/processed/m3_survey/report.md
```

**Revision 2 (after review of 256c0be).** The negative-H3 rewrite (old F9) is withdrawn; the
survey now records every warning, checks TOA identity per leg against a tempo2-semantics
reader, audits binary-model evaluator classes, fixes a TIME-offset leak (F14) and generates the
inventory from the selected configuration (PPTA from GitHub). See Sec. 5.

All numbers from papers were taken from the arXiv LaTeX sources (fetched to
`data/raw/m3_paper_sources` for the noise tables; the other sources were read from arXiv
e-prints). Literature extraction was done with an independent reader and spot-checked
against the LaTeX for every number used as a target in `M3_PLAN.md`. **[UNCERTAIN]** marks
claims we could not verify.

---

## 1. Downloads

New sources in `scripts/fetch_data.py` (group `m3`), with sha256 in `data/MANIFEST.json`:

| Source | What | Size | Checksum |
|---|---|---|---|
| `mpta_4p5yr` | **MPTA 4.5-yr** (doi:10.57891/j0vh-5g31, AAO Data Central documents 52 and 53): `partim.tar.gz` = 83 tempo2 par + 83 sub-banded tim files (245,907 TOAs); `portraits.tar.gz` = 84 frequency-resolved 2-D template portraits (PSRFITS) | 7,283,924 + 3,922,099 bytes | Data Central's HTTP ETag is the file's sha1; verified for both files |
| `mpta_4p5yr_archives` (large, not fetched) | MPTA full PSRFITS observation archives, `archives.tar.gz` (document 51) | 867,068,478 bytes | `--include-large` |
| (not kept) | `MPTA_Anisotropy_supplement.zip` (document 54): only ten .mp4 movies of anisotropy S/N maps | 443,539,544 bytes, sha1 676a7f41... | recorded in `not_available` |
| `mpta_gw_scripts` | MattTMiles/MPTAGW @ fb1d8c9: the MPTA authors' GW/noise scripts (enterprise model code, `example_noise.json` with per-pulsar EFAC/TNEQUAD/ECORR). Not an official product | 0.9 MB | commit sha |
| `inpta_dr1` | InPTA DR1 @ 2c400d5, the commit Yu & Allen cite (14 pulsars, NB + WB) | 6.4 MB | commit sha |
| `ppta_dr3_github` | danielreardon/PPTA-DR3 @ fdbe6eb (Yu & Allen's PPTA source): par/tim, max-likelihood noise JSONs, single-pulsar noise chains, CRN/HD/free-spectrum/spline-ORF chains (DE421/438/440, BayesEphem), time-slice chains, pairwise correlation chains, analysis code | 357 MB | commit sha |
| `metapulsar_v0.9.3`, `metapulsar_main` | vhaasteren/metapulsar, the "direct combination" code of Yu & Allen (v0.9.3, 2025-11-17, closest tag to their Zenodo record 17626664) and main @ 5adf316 (current METHOD_DESCRIPTION) | 10.5 + 12.1 MB | commit sha |
| `m3_paper_sources` | arXiv LaTeX of 2412.01148v1 (MPTA noise table), 2512.20455v2 (InPTA DR2 noise table), 2608.02808v1 (InPTA DR2 GWB), 2512.08666v3 (Yu & Allen); versions pinned in the fetch URLs | 53 MB | sha256 |

**MPTA noise models and chains: none are released.** The data availability statement of
2412.01148/2412.01153 lists sub-banded TOAs, archives, ephemerides and portraits only. The
noise model is the longtable "MPTA noise models" (MAP and 68 % intervals of EFAC, EQUAD,
ECORR, red/DM/chromatic/solar-wind power laws, chromatic index beta, n_earth, plus a
log10 A_13/3 column) and the "deterministic models" table (Gaussian events, annual
chromatic terms) in the LaTeX. The MPTAGW scripts give the model code and an unofficial
WN dictionary.

**Noise products of the other PTAs** (already fetched before M3 unless noted):

| PTA | Released noise products |
|---|---|
| NG15 | WN dictionary `v1p1_wn_dict.json`; WN+RN `v1p1_all_dict.json`; single-pulsar noise chains (v2.x release) |
| EPTA DR2 | `EPTA-DR2/noisefiles/{DR2full,DR2full+,DR2new,DR2new+}/<psr>_noise.json` (EFAC, `log10_tnequad`, RN, DM GP; DM amplitudes in TempoNest units since commit 9728272), `red_dict.json`/`dm_dict.json`/`chrom_dict.json` (Fourier counts), `noisefiles_t2equad/` (same values, keys renamed, **not converted**); posterior plots (Zenodo 8025019); GWB chains (Zenodo 8091568: crn_pl, hd_pl, crn_fs, hd_fs, bin/Chebyshev/Legendre ORF, os) |
| PPTA DR3 | `noisefiles/*_singlePsrNoise_sw_nesw0_noise.json` (32 max-likelihood dictionaries; identical in CSIRO and GitHub); GitHub adds `noiseFiles/{chains,stats,3sig}` and the common-noise chains; `*_singlePsrNoise_fit.par` with TempoNest TN* parameters |
| InPTA DR2 | no noise files: only `T2EFAC` in the DMX par files and DM12 par files (DM + DM1 + DM2) in InPTA.DR2.NA. The noise paper (2512.20455) gives RN/DM/chromatic/SW values in a table but **no white-noise values**; its data are "available on request" |
| InPTA DR1 | `T2EFAC` in par files; noise paper 2303.12105 has a parseable RN/DM/scattering table, no WN values |
| MPTA | paper tables only (above) |
| Yu & Allen | **nothing released** besides MetaPulsar (code "will be released upon publication"; no journal reference as of v3, 2026-04-28). No WN dictionary or chains |

---

## 2. Per-data-set inventory

Numbers are from our text parse of the released par/tim files (Appendix A) unless cited.

### NANOGrav 15-yr (v2.1.0 narrowband; M1/M2 inputs)

* 68 pulsars (67 used: J0614-3329 has 2.4 yr), 676,397 sub-band TOAs, 2004.6-2020.6.
* Arecibo (ASP, PUPPI), GBT (GASP, GUPPI), VLA (YUPPI); 2-10 systems per pulsar; 291-3985 MHz.
  Narrowband sub-band TOAs (a wideband set exists in the release but is not used).
* PINT par files: `UNITS TDB`, `EPHEM DE440`, `CLOCK TT(BIPM2019)`; DMX in every pulsar; FD
  in 59; 87 JUMP lines; EFAC/EQUAD/ECORR and RNAMP/RNIDX in the par (T2EQUAD convention).
* Noise: WN fixed at the released dictionary; IRN power law (30 modes); DMX in the timing
  model. Fully handled by M1/M2.

### EPTA DR2 (GitLab 2911d0e; Zenodo 8300645)

* 25 pulsars in four configurations: **DR2full** (24.5 yr, legacy + new backends, 56,202
  TOAs read with tempo2 semantics), **DR2new** (10.2 yr, modern backends only, 45,428), and the "+" versions with InPTA
  DR1 appended for 10 pulsars (DR2new+: 50,597). The GWB headline is DR2new.
* Telescopes: Effelsberg (`effix`, `eff`, old `g`), Jodrell Bank (`jbroach`, `jbmk2roach`,
  old `jb`/`8`), Nancay (`ncyobs`, old `ncy`), WSRT, LEAP; median 14 systems (`-group`) per
  pulsar in DR2new; 342-4882 MHz. Sub-banded narrowband TOAs (up to 4 per observation).
* tempo2 par files: `UNITS TCB`, `EPHEM DE440`, `CLK TT(BIPM2021)`, `DM_SERIES TAYLOR` with
  DM1/DM2, **no DMX**, NE_SW = 7.9 (fixed; J1022+1001 fitted), T2 binary for 13 pulsars,
  326 JUMP lines (DR2new), no FD (sub-band templates), no noise keywords in the par.
  The corrected Nancay clock file is shipped (`EPTA-DR2/clockfiles`).
* tim quirks: `TIME` offsets (+-0.001 s to +-60 s, some with trailing flags); `-padd` phase
  additions; `-pn` pulse numbers on part of J1911+1347's TOAs (PINT: "Some TOAs are missing
  pulse numbers, they will not be used"). `END` statements sit inside INCLUDEd files of
  five DR2full legs (none in the DR2new/DR2new+ trees that are actually INCLUDEd); tempo2
  ends only that file, PINT stops reading all TOAs. Indented `C` comment lines; DR2full also
  has a broken continuation line (61 orphan `-padd` lines in J1738+0333), comment lines with
  non-ASCII bytes, and one INCLUDEd file whose TIME statements do not sum to zero (F14).
* Noise: per-pulsar customised RN / DM GP / scattering (nu^-4) selection on each pulsar's own
  span, counts in `*_dict.json` (10-151 modes); WN = EFAC + TNEQUAD per `-group`, **no
  ECORR**; J1713+0747 two exponential dips. See Sec. 3.7.

### PPTA DR3 (CSIRO DAP 59374v2 and GitHub fdbe6eb)

* 32 pulsars (30 in the GW search), Parkes only, 2004.1-2022.2 (18.1 yr). Backends CPSR2,
  PDFB1-4, CASPSR, WBCORR, Medusa (UWL); up to 20 systems per pulsar; 662-3853 MHz.
  UWL data in 8 sub-bands x 4 channels (narrowband).
* **Two public variants.** GitHub fdbe6eb (used by Yu & Allen) has 113,951 TOAs and matches
  the data paper's Table tb:dr3 exactly (J1939+2134: 1,473; J1741+1351: 111). CSIRO DAP v2
  has 114,823 TOAs: J1939+2134 has 2,456 TOAs, and there is no `J1741+1351.tim`. Its par
  files have the uncertainty columns removed, and flags differ.
* tempo2 par files: **no UNITS line (tempo2 default TCB; PINT would assume TDB)**,
  `EPHEM DE436`, `CLK TT(BIPM2020)`, DM1/DM2, FD in all 32, 1,113 JUMP lines,
  `TRACK -2` in 6-7 pars without pulse numbers in the tims, NE_SW = 0 (solar wind is in the
  noise model), ELL1 with PB + FB1..FB17 for J2241-5236.
* Noise: the richest model of the five. See Sec. 3.8 and the 32 max-likelihood JSONs.

### InPTA DR2 (GitHub e2806fc) and DR1 (2c400d5)

* DR2: 27 pulsars, uGMRT band 3 (300-500 MHz) and band 5 (1260-1460 MHz), 2017.0-2024.2
  (7.2 yr), 83,120 sub-band TOAs (narrowband), 1-7 `-group` systems (band, bandwidth,
  pre/post cycle 36). DR1: 14 pulsars, 2018.4-2021.8 (3.4 yr), 8,529 NB TOAs (+ 606 WB TOAs
  in separate files).
* tempo2 par files: `UNITS TCB`, `EPHEM DE440`, `CLK TT(BIPM2023)` (DR2) / `TT(BIPM2019)`
  (DR1), **DMX** (epoch DMs from DMCalc), `T2EFAC` per group, FDJUMP in 2 DR2 pulsars
  (J1643-1224 `FDJUMPDM`), no JUMPs (one telescope).
* tim quirks: valueless flag `-cycle_post34` / `-cycle_pre34` on every DR2 TOA; J0751+1807
  has 5 DMX ranges without a DMX value; J1600-3053 and J1614-2230 have no `DMX_0001`.
* Noise (2512.20455): DMX removed; DM + DM1 + DM2 with DM GP; free-chromatic GP (chi
  sampled); deterministic n_earth; per-pulsar spans and mode counts; WN in the T2EQUAD
  convention (EFAC multiplies EQUAD), ECORR in some pulsars. See Sec. 3.10.

### MeerKAT PTA 4.5-yr

* 83 pulsars, MeerKAT L-band (856-1712 MHz; TOAs at 900-1660 MHz), single system
  `KAT_MKBF`, 32 sub-bands per observation, 2019.1-2023.6 (4.5 yr), 245,907 TOAs.
* tempo2 par files: `UNITS TCB` (82) / TDB (1), `EPHEM DE440`, `CLK TT(BIPM2020)` in 81
  pars (the paper says BIPM2022; one par each has BIPM2019 and BIPM2022), DM1/DM2
  (DM_SERIES TAYLOR in 29), FD in 65, 566 JUMP lines (`-MJD_...` flag JUMPs), NE_SW = 0,
  `TRACK -2` in 12 pars, ELL1H/DDH for 27 pulsars (J1825-0319 has a negative H3).
* Noise: per-pulsar selected EFAC/TNEQUAD/ECORR, RN (12 pulsars), DM GP, chromatic with
  fitted beta, solar wind (n_earth, SW GP), Gaussian chromatic events (15), annual chromatic
  (8). See Sec. 3.9.

### Clock, ephemeris and time units across PTAs

Checks done for this revision: (i) MPTA pars say TT(BIPM2020) (81 of 83; J0931-1902 says
BIPM2019, one says BIPM2022), the paper says BIPM2022. Over the MPTA span the two realisations
differ by 4.3 ns rms (11 ns peak to peak, mean removed; PINT's global BIPM files); BIPM2022 vs
BIPM2023 differ by 0.6 ns rms. Unresolved; see the plan's clock profiles. (ii) EPTA's shipped,
corrected Nancay clock file agrees with PINT's current global `ncyobs2obspm.clk` to <= 3.1 ns
(rms 0.1 ns) over DR2new; it will still be pinned. (iii) PINT warns that the Effelsberg clock
files (`leap2effix.clk` 36 legs, `effix2gps.clk` 18, `eff2gps.clk` 3) and `tai2tt_bipm2019.clk`
(MPTA J0931-1902) do not cover all TOAs; these must be resolved before any EPTA or MPTA leg is
used.

| | NG15 | EPTA DR2 | PPTA DR3 | InPTA DR2 / DR1 | MPTA |
|---|---|---|---|---|---|
| Timing package | PINT | tempo2 | tempo2 | tempo2 | tempo2 |
| UNITS | TDB | TCB | TCB (implicit) | TCB | TCB |
| EPHEM | DE440 | DE440 | **DE436** (GW run used DE440) | DE440 | DE440 |
| CLK | TT(BIPM2019) | TT(BIPM2021) | TT(BIPM2020) | TT(BIPM2023) / TT(BIPM2019) | TT(BIPM2020) (paper: 2022) |
| Solar wind in par | NE_SW 0 (SWM 0) | NE_SW 7.9 | NE_SW 0 (+ noise model) | NE_SW 0 | NE_SW 0 (+ noise model) |
| DM in timing model | DMX | DM, DM1, DM2 | DM, DM1, DM2 | DMX | DM, DM1, DM2 |
| EQUAD convention of the noise model | T2 | TN | TN | T2 | TN |
| Clock files shipped | release `clock/` | corrected `ncyobs2obspm.clk`, BIPM2020/2021 | `pks2gps.clk` (+ `tai2tt_bipm2021.clk` in CSIRO) | none | none |
| ECORR | yes | no | band/group ECORRs | some pulsars | some pulsars |

---

## 3. Method research

Short per-paper summaries focused on what we must reproduce. Labels are LaTeX labels.
Amplitudes are at f_ref = 1/yr.

### 3.1 Yu & Allen, arXiv:2512.08666 (v3): five-PTA "direct combination"

* **Data:** EPTA DR2new (GitLab 1506123), InPTA **DR1**, MPTA 4.5 yr, NG15 v1.0.0, PPTA DR3
  (GitHub fdbe6eb); 121 pulsars (25 + 14 + 83 + 68 + 32 before overlap), N = 1,090,206 TOAs,
  976 backends. Our survey reproduces 121 pulsars with a mean of 1.83 PTAs per pulsar (paper:
  "on average, 1.8") and counts 1,090,212 TOAs in the same files (+6, [UNCERTAIN]).
* **Overlapping pulsars:** TOA-level combination with MetaPulsar ("shared" mode). For each
  pulsar a reference PTA (NG > EPTA > PPTA > MPTA; InPTA never reference) supplies the
  astrophysical parameters (position, proper motion, spin, binary, DM), copied into the other
  PTAs' par files. TCB is converted to TDB. The DM model becomes DM0 + DM1 + DM2 (DMX
  removed). Detector parameters (FD, JUMPs) stay per PTA, plus one phase JUMP per target PTA.
  ELL1H without H3 becomes ELL1, and pulse numbers are added. Residuals: PINT for NG TOAs,
  tempo2 for the rest. Justification: after linear marginalisation only the column space of
  M matters (EM:3, SM Sec. A).
* **Noise:** WN per pulsar/backend, EFAC + EQUAD (T2EQUAD) + ECORR (NG, PPTA, MPTA, 7 EPTA
  backends), **re-fitted on the combined data** by MAP (JAXopt) and then fixed. Per pulsar
  IRN (30 bins) and DM GP (100 bins), both power laws and sampled. No chromatic, band,
  system or solar-wind GP. Priors: EFAC U(0.01,10), log10 EQUAD/ECORR U(-8.5,-5),
  gamma U(1,7), log10 A U(-20,-11).
* **Common process:** CURN and HD, 14 bins, "2 <~ f/nHz <~ 30", gamma free or 13/3.
  **[UNCERTAIN]:** 14 bins on the combined 19.5-yr span give 1.6-22.8 nHz; the quoted range
  matches NG15's 16.03 yr.
* **Inference:** discovery + NumPyro NUTS, 20,480 samples per model, 486 parameters;
  evidences by generalized stepping-stone sampling (K = 8); OS, NP and NPMV p-values from
  analytic generalized chi^2 per posterior sample.
* **Headline:** CURN gamma free log10 A = -14.37 (+0.11/-0.12), gamma 3.63 (+0.27/-0.26);
  CURN 13/3 -14.69 +- 0.04; HD gamma free -14.45 (+0.13/-0.15), gamma 3.76 (+0.31/-0.29);
  HD 13/3 -14.72 +- 0.04 (68 %, tab:pos). **ln BF(HD/CURN) = 10.18 +- 0.13, BF = 26,000 +-
  3,000**. OS mean p-value 4.3 sigma (median 4.8), NP 3.3 sigma, NPMV 3.3 sigma (tab:pvals).
  HD reconstruction reduced chi^2 0.74 (15 bins x 484 pairs). The 95.5 % (A, gamma) area is
  48 % of the 3P+ intersection.
* **Robustness:** dropping backends with EFAC outside [1/Q, Q] (Q = 10, 4, 3, 2; 0.18-1.96 %
  of TOAs) does not change the posteriors much (amplitude rises slightly).
* **Released:** nothing beyond MetaPulsar. FrankenStat's authors point out that an earlier
  draft had anomalous WN; YA traced this to phase disconnections fixed with pulse numbers.

### 3.2 MetaPulsar (vhaasteren/metapulsar, METHOD_DESCRIPTION.md)

The description below is of the **current main** (5adf316, 2026-09-16). It differs from v0.9.3
(2025-11-17, the tag closest to YA's Zenodo record) at least in the DM handling; the M3 plan
therefore pins a versioned YA configuration (plan Sec. 3.2) instead of "what MetaPulsar does".

Two strategies: **shared** (YA) and **per_pta** (= FrankenStat). Shared mode strips
deterministic terms that PINT and tempo2 do not both support (DMMODEL, CM/CMX/chromatic events,
WAVE/IFUNC, glitches, exponential dips, NE_SW_SIN, SWX, DMWaveX, ...). It converts to explicit
UNITS TDB and transforms ecliptic astrometry numerically to IERS2003 for mixed engines. It
forces a common profile for PINT + tempo2 stacks (T2CMETHOD IAU2000B, TIMEEPH FB90,
DILATEFREQ N, CORRECT_TROPOSPHERE N, PLANET_SHAPIRO N, SWM 0, explicit NE_SW; tempo2's
implicit NE_SW is 4). It aligns ELL1H Shapiro conventions (Freire & Wex eq. 28 vs 29) and
converts ELL1 to DD above a 1-ns gate. In the current main, DM stays per PTA (`DM_<pta>`,
`exclude_from_shared=("DM",)`, a CHANGELOG change); **v0.9.3, the version closest to YA,
copies the reference DM and DMEPOCH into every leg**. DMX is removed and DM1/DM2 are added free
in both. A zero-information column cull, position matching within 10 arcsec,
and narrowband TOAs only. "Statistical equivalence" argument: same column space of M as a
manual combination.

### 3.3 Lite, arXiv:2503.20949 (Larsen et al.)

* One PTA per pulsar, by FoM = sqrt(T_obs / (<sigma>^2 <Delta t>)^(3/13)) (harmonic mean
  error, geometric mean cadence). Tested by splitting IPTA DR2 VersionB (53 pulsars, 30.2 yr,
  DE436): DR2 Lite takes 33 EPTA, 8 NANOGrav and 12 PPTA pulsars; EDR2 = the 22 best pulsars
  fully combined.
* Noise (all variants): WN EFAC/EQUAD per `-group`, ECORR for NANOGrav only, MAP-fixed. IRN on
  30 array-span modes. DM GP from 1/T_obs to 150/T_DR2 plus DM1/DM2. A chromatic GP
  (chi ~ N(4, 0.5) x U(2.5, 10), MAP-fixed in array runs) for 7 pulsars. Solar wind:
  marginalised mean density plus a per-conjunction GP. J1713 exponential dip. Common process:
  13 modes on T_DR2.
* Results (95 %): A(13/3) = 4.8 +- 1.8 (Lite), 3.6 (+1.0/-1.2) (EDR2), 3.9 +- 1.0 (Full)
  x 10^-15; log10 B(CURN/noise) 3.0 / 6.4 / 9.1; B(HD/CURN) 0.66 / 2.84 / 1.39
  (reweighting). Lite's amplitude is biased high (23 %) and its 95 % area is 2.25x the
  full combination's.

### 3.4 IPTA DR2, arXiv:2201.03980 (Antoniadis et al.), and its noise treatment

* IPTA DR2 VersionB: EPTA DR1 + NANOGrav 9-yr + PPTA DR1 (+ extension, legacy); 65 pulsars,
  53 with > 3 yr; TOA-level combination with one tempo2 timing model per pulsar and
  inter-PTA JUMPs (data paper Perera et al. 2019, arXiv:1909.04534: reference data set =
  largest sum 1/sigma^2; PPTA backend offsets kept as fixed JUMPs; WN re-estimated on the
  combined TOAs; TN vs T2 EQUAD explicitly distinguished; DM = DM1 + DM2 + DM GP in
  VersionB, DMMODEL in VersionA). EPTA/PPTA TOAs band-averaged; NANOGrav sub-band TOAs
  with ECORR.
* GWB search: WN fixed at single-pulsar posterior medians; IRN + DM GP; J1713 dip; no
  system/band noise; common process 13 modes on ~30 yr, DE438.
* Results (95 %): CURN gamma free A = 5.1 (+6.7/-3.1) x 10^-15, gamma = 3.9 +- 0.9;
  CURN 13/3 A = (3.2 +- 1.0) x 10^-15; HD 13/3 A = 2.8 (+1.2/-0.9) x 10^-15; log10 BF(HD/CURN)
  = 0.31; log10 BF(CURN/noise) = 8.2 (factorised); NMOS S/N HD 0.6 (+1.2/-0.8), monopole
  2.0; HD false-alarm 0.25 (phase shifts). Chains: Zenodo 5787557.
* **Noise-modelling paper:** there is no dedicated IPTA *DR2* noise paper; the DR2 data paper
  (Perera et al. 2019) defines the VersionA/B noise treatments. The dedicated IPTA noise
  study is Lentati et al. 2016 (arXiv:1602.05570) on **DR1** (49 pulsars). It used TNEQUAD,
  per-pulsar T with T/n_c ~ 60 d (91-165 modes), and DM shapelet events (J1713, J1603). It
  introduced **system noise** (e.g. Nancay systems in 8 pulsars) and **band noise**
  (J0437 three bands, J1600, J1643, J1939), and noted that its DM amplitude has no 12 pi^2
  factor.

### 3.5 IPTA "3P+" comparison, arXiv:2309.00693

* Compares NG15 (67), EPTA+InPTA DR2new+ (25) and PPTA DR3 (30); no MPTA, no CPTA.
* Overlap handling: (i) posterior-density products (acknowledged as not independent);
  (ii) ceffyl free-spectrum KDE refits; (iii) "extended" pseudo-IPTA arrays that keep the
  base PTA's version of shared pulsars and add the others' unique pulsars (all 6 orderings,
  not FoM based); (iv) standardised factorised CURN. 27 pulsars overlap among the three.
* **Standardised settings** (the closest precedent to a common-settings combination):
  global T = 18.9 yr for every GP; IRN 30 modes; **DM GP 100 modes with NG DMX replaced by
  DM1/DM2**; deterministic solar wind with free n_earth; two J1713 exponential events; CURN
  15 modes, gamma = 13/3. NMOS with DM hyperparameters fixed at maximum likelihood.
* Published regional settings: 9 (EPTA+InPTA), 14 (NG15) and 28 (PPTA) common modes.
* Results: regional A(13/3) EPTA+InPTA 2.5 +- 0.7, NG15 2.4 (+0.7/-0.6) (90 %), PPTA 2.0
  (+0.3/-0.2) (68 %) x 10^-15; BF(HD/CURN) ~60, ~200, ~1.5. Standardised NMOS median S/N:
  2.2, 4.9, 1.0. Extended arrays: median 15 % narrower 68 % CURN interval and about 7 orders
  of magnitude more CURN evidence; NANOGrav-first orderings are best. FrankenStat attributes
  a combined "covariance approach" S/N ~6.8 to this work; that number is not in the 3P+
  LaTeX [UNCERTAIN].

### 3.6 FrankenStat, arXiv:2512.14807

* **Likelihood-level combination of the legs of one pulsar:** concatenate residuals and F,
  N = blockdiag(N_p), **M = blockdiag(M_p)**; Fourier coefficients (IRN, GWB) shared; truncate
  the sky position to the precision where legs agree. No merged timing model. This is our
  option B.
* Validation: 100 simulations of 126 pulsars, 20 yr, split into 3 PTAs by **interleaving
  TOAs** (every PTA covers the full 20 yr), injected GWB log10 A = -15.7. Combined vs
  FrankenStat: mean OS S/N 11.23 vs 11.14 (5.91 vs 5.86 sigma); single PTAs ~8.3; CURN
  log10 A difference -0.03 +- 0.19, gamma 0.05 +- 0.28; sensitivity curves within 0.05 %
  (TM only) to ~1 %. The S/N bound proof holds only for the nested case.
* Not covered: legs with different spans (MPTA/InPTA extensions), DM/chromatic noise, mixed
  DMX/DM GP. Code: github.com/davecwright3/frankenstat-paper-1; maintained in MetaPulsar.

### 3.7 EPTA DR2 III (2306.16214) and DR2 noise models (2306.16225)

* Data: 25 pulsars; DR2full 24.7 yr, DR2new 10.3 yr, "+" with InPTA DR1 for 10 pulsars
  (TOA-level merge with per-sub-band InPTA JUMPs in the released "+" pars).
* Noise: WN EFAC + TNEQUAD per `-group`, fixed at single-pulsar maximum likelihood, **no
  ECORR**. Per-pulsar Bayesian selection among {none, RN, DM, RN+DM, DM+SV, RN+DM+SV}
  (BF >= 150) with per-process mode counts (10-150, DR2new 10-100) on each pulsar's own
  span. DM amplitudes in TempoNest normalisation (`tndm=True`). Solar wind deterministic
  7.9 cm^-3 (J1022+1001 fitted 10.9). J1713+0747 two exponential dips (chromatic index 4
  and 1 in the noise paper and released code; 4 and 2 in the GW paper). RN/DM/SV
  hyperparameters sampled jointly with the common process; priors log10 A U(-18,-10),
  gamma U(0,7).
* Common process: f = n/T_array; **9 modes (DR2new)**, 24 (DR2full); DE440 fixed (BayesEphem
  as a robustness check); enterprise and fortytwo; `tm_svd=True`.
* Headline (90 %, enterprise): DR2new HD log10 A = -13.94 (+0.23/-0.48), gamma = 2.71
  (+1.18/-0.71); CURN -14.00 (+0.28/-0.77), 2.91 (+1.72/-0.87); 13/3: log10 A = -14.61
  (+0.11/-0.12); **BF(HD/CURN) = 60** (62 with fortytwo; re-estimates 66, 56, 62); OS
  (13/3) A^2_HD = 10.0 (+5.1/-4.9) x 10^-30, S/N 3.5 (+2.4/-1.7); BF phase-shift p = 0.0005,
  OS p < 1e-4 (phase shifts), 0.004 (sky scrambles). DR2new+: HD -14.10 (+0.25/-0.44),
  3.03 (+1.02/-0.67), BF 65, S/N 4.1. DR2full: BF 4, S/N 1.3. With BayesEphem the BF falls
  to 17 (DR2new).

### 3.8 PPTA DR3 GW search (2306.16215), noise (2306.16229), data (2306.16230)

* Data: 30 of 32 pulsars (no J1824-2452A, J1741+1351), ~1.2e5 TOAs, 18 yr, DE440 override.
* Noise (all sampled with the common process except WN, which is fixed at max-likelihood):
  WN EFAC + TNEQUAD per backend, ECORR per band (40/20/10 cm; UWL split at 960/2048 MHz) +
  broadband UWL + group ECORRs (overlapping selections); RN and DM GP in **every** pulsar (RN
  modes floor(T_p/240 d), DM floor(T_p/60 d)); HFF achromatic RN (T_p/30 d) in 8; nu^-4
  scattering in 7; band noise (low band in 9, J0437 also mid/high); group/system noise for
  selected `-group` subsets; deterministic solar wind (n_earth, 4 cm^-3 where unconstrained)
  plus a SW GP in ~10 pulsars; exponential dips (J0437, J1643, J2145, 2x J1713) with sampled
  chromatic index; annual DM (J0613); Gaussian DM event (J1603); 20-cm Gaussian bump (J1600);
  a JUMP at MJD 58925. About 260 parameters in the GW run. Tailored priors: 99.7 %
  single-pulsar interval widened by -2/+1 dex and +-0.5.
* Common process: floor(T/240 d) modes (printed as 28; the floor of 6605/240 is 27
  [UNCERTAIN]); log10 A U(-18,-11), gamma U(0,7).
* Headline (68 %): CURN log10 A = -14.50 (+0.14/-0.16), gamma 3.87 +- 0.36; CURN 13/3
  -14.69 +- 0.05 (A = 2.04 (+0.25/-0.22) x 10^-15); HD -14.51 (+0.18/-0.20), 3.87 +- 0.47;
  HD 13/3 -14.68 +- 0.06; BF(HD/CURN) ~1.5 (free) / ~2 (13/3); pairwise log10 Delta L = 1.1,
  sky-scramble p <~ 0.02. The first 9 yr give A < 1.2 x 10^-15 (95 %), in tension with the
  full-data amplitude (time dependence). With the "basic" noise model (RN + DM + events +
  fixed SW): log10 A = -14.08 +- 0.06, gamma = 2.9 +- 0.2.

### 3.9 MPTA (2412.01148 data + noise, 2412.01153 GW search)

* Data: 83 pulsars, 4.5 yr, 245,907 sub-band TOAs (32 sub-bands), S/N >= 8, DE440,
  TT(BIPM2022) per the paper. J1713+0747 cut before MJD 59319.
* Noise (per pulsar, evidence-selected): sigma^2 = E_F^2 sigma^2 + E_Q^2 (TN convention);
  ECORR in 29/83, EQUAD in 20/83; RN in 12; DM GP; chromatic GP with beta fitted or fixed at 4;
  solar wind deterministic n_earth (58 differ from 4 cm^-3) and SW GP (25); Gaussian
  chromatic events (15) and annual chromatic terms (8). Stated basis: 120 Fourier components
  up to ~1/(14 d); per-pulsar vs array T [UNCERTAIN]. WN fixed in the GW search.
* GW search variants: DATA (selected noise, 390 parameters), ER (+ RN in every pulsar,
  532), ALT (ER without J2129-5721's extra RN). Frequentist OS on the lowest 3 harmonics.
* Headline: ER CURN log10 A = -14.25 (+0.21/-0.36), gamma 3.60 (+1.31/-0.89); DATA CURN
  -14.25 (+0.21/-0.34), 3.52 (+1.12/-0.90); DATA HD -14.28 (+0.23/-0.30), 4.50 (+1.00/-0.93);
  ln B(CURN/IRN) = 3.17; ln B(HD+CURN/CURN) = -0.21 (ER); fixed-parameter OS (DATA)
  A^2 = (5.7 +- 1.2) x 10^-29, S/N 4.6 (p ~ 6e-4 to 8e-4, ~3.2-3.4 sigma), but
  noise-marginalised OS S/N mean 0.96 (sd 0.81). Fixed-13/3 amplitude log10 A = -14.28 +- 0.21
  (factorised), in 1.35-2.5 sigma tension with the other PTAs.

### 3.10 InPTA DR2 (2506.16769 data, 2512.20455 noise, 2608.02808 GWB) and DR1

* DR2 data: 27 pulsars, cycles 31-45 (2016-2024; cleaned MJD 57768-60399), narrowband sub-band
  TOAs, B3 + B5, DE440, TT(BIPM2023), TCB, DMX from DMCalc (epoch DMs), T2EFAC groups by band,
  bandwidth and pre/post cycle 36. J1713 cut at MJD 59309.
* DR2 noise: WN T2EQUAD convention with ECORR in some pulsars (selected; values not
  tabulated); DMX removed, DM1/DM2 + DM GP; free-chromatic GP (chi sampled: 0.6-6.9);
  deterministic n_earth (+ derivative for J1909); per-pulsar spans and per-process counts
  (up to 86 modes); priors EFAC U(0.1,8), log10 EQUAD U(-8,-1), log10 ECORR U(-9,-1),
  log10 A U(-20,-11) (ARN) / U(-20,-10) (DM, FCN), gamma and chi U(0,7), n_earth U(0,20).
* DR2 GWB (InPTA alone): CURN log10 A = -13.71 (+1.06/-3.28), gamma 2.98 (+3.62/-2.70);
  BF(CURN/none) = 2.5; HD indistinguishable from CURN; NMOS peaks HD -0.46; 95 % upper limit
  at 13/3: log10 A < -13.47 (uniform-A prior), A < 3.4 x 10^-14. No other PTA combined.
* DR1 (2206.09289, 2303.12105): 14 pulsars, 3.5 yr, NB (B3 + B5) and WB (B3 only) TOAs;
  T2EQUAD, no ECORR; per-pulsar mode counts. **EPTA DR2new+/full+ use InPTA DR1 (NB)**, so
  using InPTA DR1 and EPTA DR2new+ together would double-count.

---

## 4. Pulsar overlap

Pulsars matched by sky position (30 arcsec; B1855+09 = J1857+0943, B1937+21 = J1939+2134).
All counts below are generated by `scripts/m3_survey.py`
(`data/processed/m3_survey/inventory_*.json`, `overlap_*.md`).

**Selected configuration (NG15, EPTA DR2new, PPTA DR3 GitHub, InPTA DR2, MPTA): 122 unique
pulsars, 235 legs, 61 pulsars in two or more PTAs, 1,164,803 TOAs**; by number of PTAs: 61 in
one, 35 in two, 10 in three, 6 in four, 10 in all five (mean 1.93).

| | NG15 | EPTA DR2new | PPTA DR3 (GitHub) | InPTA DR2 | MPTA |
|---|---|---|---|---|---|
| NG15 | 68 | 21 | 18 | 24 | 32 |
| EPTA DR2new | 21 | 25 | 12 | 16 | 18 |
| PPTA DR3 | 18 | 12 | 32 | 16 | 28 |
| InPTA DR2 | 24 | 16 | 16 | 27 | 16 |
| MPTA | 32 | 18 | 28 | 16 | 83 |

(With the CSIRO PPTA variant, which has no J1741+1351 tim file: 234 legs, 60 multi-PTA
pulsars.)

**Yu & Allen set (InPTA DR1, PPTA GitHub): 121 unique pulsars, 222 legs, 56 multi-PTA
pulsars, 1,090,212 TOAs**, matching the paper's 121 pulsars; 65 / 32 / 10 / 7 / 7 in
1 / 2 / 3 / 4 / 5 PTAs (mean 1.835, paper "1.8"). Pairwise: NG15 x EPTA 21, NG15 x PPTA 18,
NG15 x InPTA 13, NG15 x MPTA 32, EPTA x PPTA 12, EPTA x InPTA 10, EPTA x MPTA 18,
PPTA x InPTA 12, PPTA x MPTA 28, InPTA x MPTA 10.

In all five PTAs (selected configuration, 10): J0030+0451, J0613-0200, J1022+1001,
J1024-0719, J1600-3053, J1713+0747, J1730-2304, J1744-1134, J1909-3744, J2124-3358 (YA set: 7,
without J0030+0451, J1024-0719, J1730-2304, which are not in InPTA DR1). Single-PTA pulsars
(selected configuration): 36 MPTA-only (southern), 24 NG15-only (northern, Arecibo/GBT), 1
InPTA-only; every EPTA DR2 and PPTA DR3 (GitHub) pulsar is also timed by another PTA.
Appendix A has the per-pulsar matrix with TOA counts and spans.

---

## 5. PINT ingestion audit

`scripts/m3_survey.py` loads every par/tim pair with PINT 1.1.7
(`get_model_and_toas(planets=True, allow_tcb=True, allow_T2=True)`, PINT's global clock
repository), first **as released**, then after a logged **canonicalisation** (`--canonical`;
copies in `data/processed/m3_survey/canon/`; tempo2 pars loaded with
`ell1h_shapiro="absorbed"`). For the canonical load it records:

* **every warning** (PINT's loguru messages and Python warnings, including those PINT's
  TCB->TDB path would otherwise hide, e.g. "PINT does not support 'DILATEFREQ Y'");
* a **TOA-identity check** against an independent reader of the *released* tim tree with
  tempo2 semantics (INCLUDE recursion, SKIP/NOSKIP, END ending only the current file, TIME
  offsets local to a file, the `-to` flag): count, arrival time without clock corrections
  (< 2 ns), radio frequency, uncertainty and system flag per TOA;
* a binary-model **evaluator audit** (Appendix A);
* the pre-fit weighted RMS against the par file's TRES (a screen only: TRES is tempo2's
  post-fit value with its own weighting and selection, so agreement is not an equivalence
  test and disagreement is not proof of an error).

Outcome (Appendix A has the per-data-set tables). Every loadable leg (329 of 330) passes the
TOA-identity check (max |dt| 0.81 ns, identical frequencies, uncertainties and system flags):

| data set | as released ok / fail | canonical ok / fail | TOA identity: identical / checked |
|---|---|---|---|
| NG15 | 68 / 0 | 68 / 0 | 68 / 68 |
| EPTA DR2new | 19 / 6 | 25 / 0 | 25 / 25 |
| EPTA DR2full | 16 / 9 | 25 / 0 | 25 / 25 |
| EPTA DR2new+ | 19 / 6 | 25 / 0 | 25 / 25 |
| PPTA DR3 (CSIRO) | 3 / 28 | 31 / 0 | 31 / 31 |
| PPTA DR3 (GitHub) | 3 / 29 | 32 / 0 | 32 / 32 |
| InPTA DR2 | 8 / 19 | 27 / 0 | 27 / 27 |
| InPTA DR1 | 13 / 1 | 14 / 0 | 14 / 14 |
| MPTA | 70 / 13 | 82 / 1 (J1825-0319, signed H3) | 82 / 82 |

### Canonicalisation rules

None of these rules is yet validated against tempo2 beyond TOA identity; they are intended to
reproduce tempo2's reading of the files. Equivalence of the *timing model* (residuals, column
space, likelihood) is the subject of the M3a gates (plan Sec. 5.1).

| # | Category | Where | Cause | Rewrite | Status |
|---|---|---|---|---|---|
| F1 | TOA line misparsed (`could not convert string to float`, `invalid literal for int`) | PPTA (22 per variant), EPTA (6 DR2new, 9 DR2full) | PINT classifies lines by their first characters (leading blank + '.' in column 42 = Parkes format; `[0-9a-z@] ` = Princeton; a command word = command); free-form archive names trip this | archive token -> `toaNNNNNNN`, leading blanks stripped | metadata only; TOA identity passes |
| F2 | Indented comment ` C ...` read as a TOA | EPTA (2 legs per configuration) | tempo2 comment rule vs PINT's column detection | comments written unindented | TOA identity passes |
| F3 | `END` inside an INCLUDEd file | EPTA DR2full (5 legs) | tempo2 ends that file only; PINT stops reading all TOAs (silently: DR2full J1744-1134 gave 392 of 1,949 TOAs) | END and the rest of that file commented out | TOA identity passes |
| F4 | Orphan flag line / non-ASCII comment marker | EPTA DR2full (J1738+0333: 61 `-padd` continuation lines; `C<0xA0>` lines) | broken lines | commented out | TOA identity passes |
| F5 | `TRACK -2` without pulse numbers | PPTA (6 CSIRO, 7 GitHub), MPTA (12) | par requests pulse-number tracking, tims have no `-pn` | TRACK dropped; PINT tracks the nearest pulse | **phase connection unverified**; production must add pulse numbers from each leg's own model |
| F6 | No UNITS line | PPTA (29 CSIRO / 31 GitHub pars) | tempo2 default TCB, PINT default TDB (J1909-3744 pre-fit wrms 850 us without, 0.39 us with the fix; TRES 0.33 us) | `UNITS TCB` added to tempo2 pars (EPHVER present) without UNITS | restores tempo2's reading; PINT then converts TCB->TDB "approximately" (its warning) |
| F7 | Valueless tim flag | InPTA DR2 (`-cycle_post34`/`-cycle_pre34`, all 27) | PINT requires flag/value pairs | value `1` | flag used for selection only |
| F8 | PB + FB1..FB17 without FB0 | PPTA J2241-5236 | tempo2 uses 1/PB as FB0; PINT requires FB0 | FB0 = 1/(PB x 86400 s) | loads, but pre-fit wrms 2.4x TRES: **unvalidated**, FB-series values and derivatives to be checked (PINT PR #2023 targets this class) |
| ~~F9~~ | ~~Negative H3 / M2~~ | MPTA J1825-0319 (DDH, H3 = -2.98e-7, H3 and STIG fitted) | | **withdrawn**: zeroing H3 also zeroes the STIG design column (the Shapiro delay is H3 g(STIG, t)), which changes the marginalised likelihood | leg not loadable with PINT 1.1.7: **blocking** for that leg; needs a signed-H3 evaluator (PR #2023) or tempo2 |
| F10 | DMXR ranges without DMX value | InPTA DR2 J0751+1807 (5) | inconsistent par | `DMX_xxxx 0` frozen added | tempo2 ignores such ranges |
| F11 | Fit flag on a DMXR bound | InPTA DR1 J0751+1807 | PINT: unfittable parameter | frozen | bounds are not parameters |
| F12 | DMX present but no DMX_0001 | InPTA DR2 J1600-3053, J1614-2230 | PINT's DMX component always holds a template DMX_0001 | empty frozen DMX_0001 range added | no TOA in the range |
| F13 | ELL1H with H3 + STIG: different Shapiro expression | MPTA J1802-2124 (pre-fit 11.0 us vs TRES 3.0), J1757-5322 (7.1 vs 2.4), J1525-5545 (8.6 vs 4.6), J1435-6100; PPTA J1545-4550 (2.4 vs 1.1), J1902-5105 (3.4 vs 1.6) | PINT's default `ell1h_shapiro="full"` = Freire & Wex eq. 29, tempo2 = eq. 28 | tempo2 pars loaded with `"absorbed"` | wrms then equals TRES within 4 % (3.00 / 2.41 / 4.63 / 1.18 / 1.55 us). **Does not cover** ELL1H with H3 + H4 (harmonic count, below) or DDH. In a combination the reference and target legs must use the same convention before parameter values are copied |
| F14 | TIME offsets leaking across INCLUDEs | EPTA DR2full J1713+0747 (`WSRT.P1.2273.tim` ends at a net -2 ms) | tempo2 keeps TIME local to the file; PINT carries the running offset into later INCLUDEd files (all later TOAs shifted by 2 ms) | compensating `TIME` appended to any file with a non-zero net offset; trailing tokens after a TIME value dropped | TOA identity passes; pre-fit wrms of that leg 965 us -> 1.76 us |

### Evaluator classes not fixed by canonicalisation

From the audit table in Appendix A (selected configuration and YA set):

* **Signed H3** (MPTA J1825-0319): blocking (above).
* **ELL1H with H3 + H4 and no NHARMS** (MPTA J0613-0200, J1327-0755, J1545-4550,
  J1804-2717, J2145-0750; PPTA J0613-0200; EPTA DR2new J0751+1807, J1012+5307; InPTA DR2
  J0751+1807, J1012+5307): PINT always uses at least 7 harmonics (checked: an explicit
  `NHARMS 4` in the par is raised to 7), tempo2 defaults to 4. The reviewer measured 56 ns to
  1.6 us RMS Shapiro-delay differences on MPTA pars. **Quarantined** until an evaluator with an
  explicit harmonic count passes parity.
* **DDH** (21 legs), **DDK** (10 legs) and **T2 -> {ELL1, ELL1H, DD, DDK}** resolutions (43
  legs): not affected by the ELL1H switch; to be validated against tempo2.

### Legs whose pre-fit wrms still differs from TRES (screen only)

In the selected configuration (and the YA set) after F1-F14:

| leg | PINT binary | pre-fit wrms / TRES [us] | note |
|---|---|---|---|
| EPTA DR2new J1600-3053 (also DR2new+) | T2 -> DD (M2, SINI, PBDOT, XDOT, OMDOT) | 1.46 / 0.87 | **blocks** EPTA reproduction and YA acceptance until parity passes |
| PPTA J1600-3053 | DDH | 2.28 / 1.47 | **blocks** as above |
| PPTA J1713+0747 | T2 -> DDK | 0.44 / 0.25 | **blocks** as above; exponential dips are in PPTA's noise model, not the par |
| PPTA J2241-5236 | ELL1 + FB0..FB17 | 0.54 / 0.22 | **blocks** as above (F8) |
| PPTA J1824-2452A | isolated | 30.9 / 15.8 | strong red noise, TRACK dropped; not in the PPTA GW search but in YA |
| PPTA J1741+1351 (GitHub only) | ELL1 | 1.67 / 0.83 | TRACK dropped; 111 TOAs; not in the PPTA GW search but in YA |
| MPTA J1514-4946 | ELL1H | 1.48 / 2.30 | par fitted to 1,017 TOAs, release has 713 |

EPTA DR2full (not used in M3) keeps eight legs with ratios of 1.8-48 after F14 (e.g.
J1744-1134 27 us, J1857+0943 47 us; legacy backends, `-padd` and phase-tracking conventions are
suspects). Par NTOA vs tim count mismatches (Appendix A) are expected where the par was fitted
before the final TOA selection (e.g. MPTA J2241-5236 6,688 vs 3,405; PPTA J0437-4715 20,836 vs
11,637).

### Warnings that need action (Appendix A has all of them)

* Clock coverage: `leap2effix.clk` (36 legs), `effix2gps.clk` (18), `eff2gps.clk` (3),
  `tai2tt_bipm2019.clk` (MPTA J0931-1902): "Data points out of range".
* Ignored tempo2 settings: `DILATEFREQ Y`, `TIMEEPH IF99` (262 legs).
* "Some TOAs are missing pulse numbers, they will not be used" (EPTA J1911+1347 in DR2new,
  DR2new+, DR2full: only part of its TOAs carry `-pn`).
* "overflow encountered in conversion from string" (14 MPTA/InPTA legs) and "divide by zero"
  (MPTA J0955-6150): not yet traced.
* "EFAC ... has no TOAs" (InPTA DR1 J1857+0943, J1939+2134), DMX range overlaps (8 legs).

---

## Appendix A: generated tables

Generated by `scripts/m3_survey_report.py` from the `--canonical` survey run (PINT 1.1.7,
2026-10-08).

<!-- GENERATED TABLES BELOW -->

### Per-data-set summary

| data set | pulsars | with tim | TOAs (text) | TOAs (PINT, canonical) | first-last TOA | max span [yr] | median span [yr] | systems/psr (median) | observatories | radio freq [MHz] | wideband TOAs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ng15 | 68 | 68 | 676,397 | 676,397 | 2004.58-2020.61 | 15.87 | 8.12 | 2 | arecibo, gbt, vla | 291-3985 | 0 |
| epta_dr2new | 25 | 25 | 45,428 | 45,428 | 2011.13-2021.47 | 10.22 | 9.92 | 14 | eff, effix, jbmk2roach, jbroach, leap, ncyobs, wsrt | 342-4882 | 0 |
| epta_dr2full | 25 | 25 | 56,202 | 56,202 | 1996.76-2021.47 | 24.46 | 16.10 | 19 | 8, eff, effix, g, jb, jbdfb, jbmk2roach, jbroach, leap, ncy, ncyobs, wsrt | 324-4882 | 0 |
| epta_dr2new+ | 25 | 25 | 50,597 | 50,597 | 2011.13-2022.18 | 11.02 | 10.08 | 15 | eff, effix, gmrt, jbmk2roach, jbroach, leap, ncyobs, wsrt | 302-4882 | 0 |
| ppta_dr3 | 32 | 31 | 114,823 | 114,823 | 2004.10-2022.18 | 18.08 | 17.75 | 15 | pks | 662-3853 | 0 |
| ppta_dr3_gh | 32 | 32 | 113,951 | 113,951 | 2004.10-2022.18 | 18.08 | 17.75 | 15 | pks | 662-3853 | 0 |
| inpta_dr2 | 27 | 27 | 83,120 | 83,120 | 2017.04-2024.24 | 7.20 | 5.85 | 4 | gmrt | 301-1457 | 0 |
| inpta_dr1 | 14 | 14 | 8,529 | 8,529 | 2018.36-2021.77 | 3.40 | 3.39 | 5 | gmrt | 302-1451 | 0 |
| mpta | 83 | 83 | 245,907 | 242,863 | 2019.11-2023.58 | 4.46 | 4.27 | 1 | meerkat | 900-1660 | 0 |

### Par-file conventions (counts of pulsars)

| data set | UNITS | CLK | EPHEM | BINARY | NE_SW | DM_SERIES | DMX psr | JUMP lines | FD psr | noise keys in par | tempo2-only keywords |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ng15 | {'TDB': 68} | {'TT(BIPM2019)': 68} | {'DE440': 68} | {'ELL1': 23, '-': 18, 'DD': 22, 'BT': 1, 'DDK': 1, 'ELL1H': 3} | {'0.0': 68} | {'-': 68} | 68 | 87 | 59 | {'ECORR': 68, 'EFAC': 68, 'EQUAD': 68, 'RNAMP': 23, 'RNIDX': 23} | {'DILATEFREQ': 68, 'SWM': 68, 'T2CMETHOD': 51, 'TIMEEPH': 51} |
| epta_dr2new | {'TCB': 25} | {'TT(BIPM2021)': 25} | {'DE440': 25} | {'-': 9, 'T2': 13, 'DDH': 3} | {'7.9000000000000000001': 24, '9.7014335675808864986': 1} | {'TAYLOR': 24, '-': 1} | 0 | 326 | 0 | - | {'DILATEFREQ': 25, 'T2CMETHOD': 25, 'TIMEEPH': 25} |
| epta_dr2full | {'TCB': 25} | {'TT(BIPM2021)': 25} | {'DE440': 25} | {'-': 9, 'T2': 13, 'DDH': 3} | {'7.9000000000000000001': 11, '7.9': 13, '10.403795260209403572': 1} | {'TAYLOR': 24, '-': 1} | 0 | 471 | 0 | - | {'DILATEFREQ': 25, 'T2CMETHOD': 25, 'TIMEEPH': 25} |
| epta_dr2new+ | {'TCB': 25} | {'TT(BIPM2021)': 25} | {'DE440': 25} | {'-': 9, 'T2': 13, 'DDH': 3} | {'7.9000000000000000001': 24, '7.2225507908424881583': 1} | {'TAYLOR': 24, '-': 1} | 0 | 688 | 0 | - | {'DILATEFREQ': 25, 'T2CMETHOD': 25, 'TIMEEPH': 25} |
| ppta_dr3 | {'TCB(default)': 30, 'TCB': 2} | {'TT(BIPM2020)': 32} | {'DE436': 32} | {'-': 9, 'ELL1': 5, 'T2': 13, 'DDH': 3, 'ELL1H': 2} | {'0': 9, '0.000': 22, '4.000': 1} | {'TAYLOR': 32} | 0 | 1113 | 32 | - | {'DILATEFREQ': 32, 'T2CMETHOD': 32, 'TIMEEPH': 32} |
| ppta_dr3_gh | {'TCB(default)': 31, 'TCB': 1} | {'TT(BIPM2020)': 32} | {'DE436': 32} | {'-': 9, 'ELL1': 5, 'T2': 13, 'DDH': 3, 'ELL1H': 2} | {'0': 8, '0.000': 24} | {'TAYLOR': 32} | 0 | 1116 | 32 | - | {'DILATEFREQ': 32, 'T2CMETHOD': 32, 'TIMEEPH': 32} |
| inpta_dr2 | {'TCB': 27} | {'TT(BIPM2023)': 27} | {'DE440': 27} | {'-': 8, 'T2': 13, 'ELL1': 3, 'DDH': 2, 'DD': 1} | {'0': 27} | {'-': 27} | 27 | 0 | 0 | {'T2EFAC': 27} | {'DILATEFREQ': 27, 'T2CMETHOD': 27, 'T2EFAC': 27, 'TIMEEPH': 27} |
| inpta_dr1 | {'TCB': 14} | {'TT(BIPM2019)': 14} | {'DE440': 14} | {'DDK': 2, 'ELL1': 4, 'ELL1H': 1, 'DDH': 2, 'DD': 2, '-': 3} | {'0': 14} | {'-': 14} | 14 | 0 | 0 | {'T2EFAC': 12} | {'DILATEFREQ': 14, 'T2CMETHOD': 14, 'T2EFAC': 12, 'TIMEEPH': 14} |
| mpta | {'TCB': 82, 'TDB': 1} | {'TT(BIPM2022)': 1, 'TT(BIPM2020)': 81, 'TT(BIPM2019)': 1} | {'DE440': 83} | {'-': 19, 'T2': 4, 'ELL1H': 15, 'ELL1': 18, 'DD': 14, 'DDH': 12, 'DDGR': 1} | {'0': 83} | {'-': 54, 'TAYLOR': 29} | 0 | 566 | 65 | - | {'DILATEFREQ': 83, 'T2CMETHOD': 83, 'TIMEEPH': 83} |

### PINT load outcome

| data set | as released: ok / fail | canonicalised: ok / fail | failure categories (as released) | canonical fixes applied (pulsars) |
|---|---|---|---|---|
| ng15 | 68 / 0 | 68 / 0 | - | tim:archive-name (31) |
| epta_dr2new | 19 / 6 | 25 / 0 | tim line misparsed (archive name / indented comment / continuation line) (6) | tim:archive-name (25); tim:indented-comment (2) |
| epta_dr2full | 16 / 9 | 25 / 0 | tim line misparsed (archive name / indented comment / continuation line) (9) | tim:archive-name (25); tim:END-in-file (5); tim:unparseable-line(commented) (3); tim:TIME-trailing-tokens(dropped) (2); tim:orphan-flag-line(dropped) (2); tim:indented-comment (1); tim:TIME-reset-at-end-of-file (1) |
| epta_dr2new+ | 19 / 6 | 25 / 0 | tim line misparsed (archive name / indented comment / continuation line) (6) | tim:archive-name (25); tim:indented-comment (2) |
| ppta_dr3 | 3 / 28 | 31 / 0 | tim line misparsed (archive name / indented comment / continuation line) (22); TRACK -2 in par but no pulse numbers in tim (5); PB + FBn (tempo2) without FB0 (1) | tim:archive-name (31); par:explicit-UNITS-TCB (29); tim:valueless-flag (18); par:drop-TRACK (6); par:PB->FB0 (1) |
| ppta_dr3_gh | 3 / 29 | 32 / 0 | tim line misparsed (archive name / indented comment / continuation line) (22); TRACK -2 in par but no pulse numbers in tim (6); PB + FBn (tempo2) without FB0 (1) | tim:archive-name (32); par:explicit-UNITS-TCB (31); tim:valueless-flag (18); par:drop-TRACK (7); par:PB->FB0 (1) |
| inpta_dr2 | 8 / 19 | 27 / 0 | valueless tim flag (-cycle_post34) (18); DMXR ranges without DMX value (1) | tim:valueless-flag (27); par:empty-DMX_0001-template (2); par:missing-DMX-value (1) |
| inpta_dr1 | 13 / 1 | 14 / 0 | fit flag on DMXR range bound (1) | tim:archive-name (13); par:freeze-DMXR (1) |
| mpta | 70 / 13 | 82 / 1 | TRACK -2 in par but no pulse numbers in tim (12); signed (negative) H3 in DDH: unsupported by PINT 1.1.7 (1) | tim:archive-name (83); par:drop-TRACK (12); par:signed-H3(unsupported (1); kept) (1) |

Still failing after canonicalisation: mpta/J1825-0319 (ValueError: Companion mass M2 cannot be negative (-0.4480919945782575 solMass))

Par without tim: ppta_dr3/J1741+1351

### TOA identity (canonicalised PINT load vs tempo2-semantics text records)

Per leg, PINT's TOAs (clock corrections removed, TIME/-to offsets kept) are compared with the records of the released tim tree read with tempo2 semantics (INCLUDE, SKIP, END per file, TIME, -to): count, arrival time (< 2 ns), frequency, uncertainty and system flag (-group, else -sys, else -f).

| data set | legs checked | identical | max dt [ns] | failures |
|---|---|---|---|---|
| ng15 | 68 | 68 | 0.81 | - |
| epta_dr2new | 25 | 25 | 0.80 | - |
| epta_dr2full | 25 | 25 | 0.80 | - |
| epta_dr2new+ | 25 | 25 | 0.80 | - |
| ppta_dr3 | 31 | 31 | 0.63 | - |
| ppta_dr3_gh | 32 | 32 | 0.63 | - |
| inpta_dr2 | 27 | 27 | 0.63 | - |
| inpta_dr1 | 14 | 14 | 0.71 | - |
| mpta | 82 | 82 | 0.80 | - |

### Binary-model evaluator audit (selected configuration and Yu & Allen set)

Legs whose binary parameterisation PINT and tempo2 evaluate differently or that needed a rewrite. None of these is validated against tempo2 yet (M3a gates).

| class | consequence | legs |
|---|---|---|
| signed (negative) H3 | PINT 1.1.7 rejects (DDH converts to M2 < 0); zeroing H3 would delete the STIG column: **blocking**, needs a signed-H3 evaluator (PINT PR #2023) or tempo2 | mpta/J1825-0319 |
| ELL1H H3 + H4, no NHARMS | PINT uses >= 7 harmonics, tempo2 defaults to 4 (reviewer measured 56 ns-1.6 us Shapiro differences): **quarantined** | epta_dr2new/J0751+1807, epta_dr2new/J1012+5307, inpta_dr2/J0751+1807, inpta_dr2/J1012+5307, mpta/J0613-0200, mpta/J1327-0755, mpta/J1545-4550, mpta/J1804-2717, mpta/J2145-0750, ppta_dr3_gh/J0613-0200 |
| ELL1H H3 + STIG | evaluated with ell1h_shapiro='absorbed' (tempo2 eq. 28) | epta_dr2new/J0613-0200, inpta_dr1/J0751+1807, inpta_dr2/J0613-0200, mpta/J1036-8317, mpta/J1435-6100, mpta/J1514-4946, mpta/J1525-5545, mpta/J1543-5149, mpta/J1757-5322, mpta/J1802-2124, ppta_dr3_gh/J1545-4550, ppta_dr3_gh/J1902-5105 |
| DDH | full DD Shapiro expression; not affected by the ELL1H switch; unvalidated | epta_dr2new/J1022+1001, epta_dr2new/J1640+2224, epta_dr2new/J1918-0642, inpta_dr1/J1022+1001, inpta_dr1/J1600-3053, inpta_dr2/J1022+1001, inpta_dr2/J1640+2224, mpta/J0900-3144, mpta/J1017-7156, mpta/J1022+1001, mpta/J1101-6424, mpta/J1125-5825, mpta/J1421-4409, mpta/J1455-3330, mpta/J1811-2405, mpta/J1903-7051, mpta/J1918-0642, mpta/J2150-0326, ppta_dr3_gh/J1017-7156, ppta_dr3_gh/J1022+1001, ppta_dr3_gh/J1600-3053 |
| DDK | Kopeikin terms; unvalidated | epta_dr2new/J1713+0747, inpta_dr1/J0437-4715, inpta_dr1/J1713+0747, inpta_dr2/J0437-4715, inpta_dr2/J1713+0747, mpta/J0437-4715, mpta/J2222-0137, ng15/J1713+0747, ppta_dr3_gh/J0437-4715, ppta_dr3_gh/J1713+0747 |
| PB + FB1..FBn (no FB0) | rewritten to FB0 = 1/PB; derivative structure unvalidated | ppta_dr3_gh/J2241-5236 |
| BINARY T2 | resolved by PINT allow_T2 to {'ELL1H': 7, 'ELL1': 18, 'DD': 11, 'DDK': 7} | 43 legs |

### PINT warnings (canonicalised load; all legs)

Every loguru and Python warning is recorded per leg (`canon_pint_warnings` in survey.json). Counts of legs per normalised message:

| legs | message |
|---|---|
| 262 | UserWarning: PINT does not support 'DILATEFREQ Y' |
| 262 | UserWarning: PINT only supports 'TIMEEPH FB#' |
| 261 | PINT does not support 'UNITS TCB' internally. Reading this par file nevertheless because the `allow_tcb` option was give |
| 260 | Converting this timing model from TCB to TDB. Please note that the TCB to TDB conversion is only approximate and the res |
| 259 | UserWarning: Unrecognized parfile line 'EPHVER #' |
| 163 | UserWarning: Unrecognized parfile line 'DM_SERIES TAYLOR' |
| 36 | UserWarning: Data points out of range in clock file 'leap#ffix.clk' |
| 36 | Found T# binary model. Gracefully converting T# to: ELL# |
| 21 | Found T# binary model. Gracefully converting T# to: DD. |
| 18 | UserWarning: Data points out of range in clock file 'effix#gps.clk' |
| 14 | UserWarning: DDK model uses KIN as inclination angle. SINI will not be used. This happens every time a DDK model is cons |
| 14 | Found T# binary model. Gracefully converting T# to: ELL#H. |
| 14 | RuntimeWarning: overflow encountered in conversion from string |
| 11 | Found T# binary model. Gracefully converting T# to: DDK. |
| 8 | Start of DMX_# (#) overlaps with DMX_# (#) |
| 8 | End of DMX_# (#) overlaps with DMX_# (#) |
| 4 | Invalid altitude calculated for # TOAS |
| 3 | UserWarning: Data points out of range in clock file 'eff#gps.clk' |
| 3 | Some TOAs are missing pulse numbers, they will not be used. |
| 2 | UserWarning: Unrecognized parfile line 'EPHVER # #' |
| 2 | UserWarning: EFAC maskParameter(EFAC# -sys GM_GWB_#_#_b# # () frozen=True) has no TOAs |
| 1 | UserWarning: Unrecognized parfile line 'FDJUMP_SCALE LOG' |
| 1 | UserWarning: Using A#DOT with a DDK model is not advised. |
| 1 | TZRMJD is not set.  Setting TZRMJD to first TOA after PEPOCH or last TOA before PEPOCH.  This may leave your residuals w |
| 1 | UserWarning: Data points out of range in clock file 'tai#tt_bipm#clk' |
| 1 | RuntimeWarning: divide by zero encountered in divide |

### Consistency flags (canonicalised PINT load)

TOA counts: PINT vs text count of the tim tree (INCLUDE/SKIP/END honoured) vs the par's NTOA (written by tempo2 when the par was last fitted; it can predate the released tim). Pre-fit wrms: PINT pre-fit weighted RMS with raw TOA errors vs the par's TRES (tempo2 post-fit wrms); a ratio > 1.5 flags a possible phase-connection or model problem.

| data set | psr | text | PINT | par NTOA | wrms [us] | TRES [us] | ratio | note |
|---|---|---|---|---|---|---|---|---|
| epta_dr2new | J1600-3053 | 2598 | 2598 | 2598 | 1.463 | 0.873 | 1.68 | wrms/TRES |
| epta_dr2full | J0030+0451 | 4069 | 4069 | 4071 | 2.456 | 2.456 | 1.00 | NTOA != PINT |
| epta_dr2full | J0613-0200 | 2917 | 2917 | 2909 | 3.844 | 1.432 | 2.68 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1022+1001 | 2453 | 2453 | 2445 | 9.892 | 1.022 | 9.68 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1024-0719 | 2522 | 2522 | 2515 | 1.284 | 1.139 | 1.13 | NTOA != PINT |
| epta_dr2full | J1640+2224 | 2006 | 2006 | 2007 | 1.277 | 1.142 | 1.12 | NTOA != PINT |
| epta_dr2full | J1713+0747 | 5011 | 5011 | 4991 | 1.757 | 0.228 | 7.70 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1730-2304 | 1329 | 1329 | 1315 | 10.783 | 0.974 | 11.07 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1738+0333 | 1024 | 1024 | 1019 | 2.801 | 2.722 | 1.03 | NTOA != PINT |
| epta_dr2full | J1744-1134 | 1949 | 1949 | 1931 | 26.936 | 0.567 | 47.51 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1751-2857 | 401 | 401 | 398 | 10.773 | 3.192 | 3.37 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1857+0943 | 1547 | 1547 | 1540 | 46.626 | 1.209 | 38.57 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1909-3744 | 2503 | 2503 | 2524 | 0.630 | 0.347 | 1.82 | NTOA != PINT, wrms/TRES |
| epta_dr2full | J1911+1347 | 886 | 886 | 882 | 1.166 | 1.007 | 1.16 | NTOA != PINT |
| epta_dr2new+ | J1600-3053 | 2868 | 2868 | 2868 | 1.485 | 0.911 | 1.63 | wrms/TRES |
| epta_dr2new+ | J2124-3358 | 2345 | 2345 | 2197 | 1.749 | 0.474 | 3.69 | NTOA != PINT, wrms/TRES |
| ppta_dr3 | J0437-4715 | 11637 | 11637 | 20836 | 0.477 | 0.447 | 1.07 | NTOA != PINT |
| ppta_dr3 | J1713+0747 | 5140 | 5140 | 5141 | 0.429 | 0.248 | 1.73 | NTOA != PINT, wrms/TRES |
| ppta_dr3_gh | J1600-3053 | 5146 | 5146 | 5146 | 2.278 | 1.466 | 1.55 | wrms/TRES |
| ppta_dr3_gh | J1713+0747 | 5140 | 5140 | 5141 | 0.435 | 0.248 | 1.76 | NTOA != PINT, wrms/TRES |
| ppta_dr3_gh | J1741+1351 | 111 | 111 | 111 | 1.673 | 0.829 | 2.02 | wrms/TRES |
| ppta_dr3_gh | J1824-2452A | 1284 | 1284 | 1284 | 30.879 | 15.772 | 1.96 | wrms/TRES |
| ppta_dr3_gh | J2241-5236 | 6238 | 6238 | 6238 | 0.538 | 0.224 | 2.40 | wrms/TRES |
| inpta_dr1 | J0751+1807 | 196 | 196 | 411 | 33.062 | 31.290 | 1.06 | NTOA != PINT |
| inpta_dr1 | J1643-1224 | 528 | 528 | 535 | 9.671 | 11.964 | 0.81 | NTOA != PINT |
| inpta_dr1 | J1713+0747 | 772 | 772 | 771 | 2.630 | 2.625 | 1.00 | NTOA != PINT |
| inpta_dr1 | J1857+0943 | 406 | 406 | 411 | 8.497 | 8.528 | 1.00 | NTOA != PINT |
| inpta_dr1 | J2124-3358 | 827 | 827 | 1290 | 6.883 | 8.542 | 0.81 | NTOA != PINT |
| inpta_dr1 | J2145-0750 | 1808 | 1808 | 1829 | 4.519 | 4.707 | 0.96 | NTOA != PINT |
| mpta | J0030+0451 | 2880 | 2880 | 2926 | 2.829 | 2.850 | 0.99 | NTOA != PINT |
| mpta | J0101-6422 | 1430 | 1430 | 1646 | 1.878 | 2.014 | 0.93 | NTOA != PINT |
| mpta | J0636-3044 | 2217 | 2217 | 3648 | 3.068 | 3.381 | 0.91 | NTOA != PINT |
| mpta | J0931-1902 | 1903 | 1903 | 2188 | 2.337 | 2.529 | 0.92 | NTOA != PINT |
| mpta | J1012-4235 | 2898 | 2898 | 3261 | 3.258 | 3.439 | 0.95 | NTOA != PINT |
| mpta | J1036-8317 | 2046 | 2046 | 2485 | 1.848 | 2.034 | 0.91 | NTOA != PINT |
| mpta | J1125-6014 | 2636 | 2636 | 2648 | 1.337 | 1.338 | 1.00 | NTOA != PINT |
| mpta | J1435-6100 | 3976 | 3976 | 4035 | 3.430 | 3.443 | 1.00 | NTOA != PINT |
| mpta | J1514-4946 | 713 | 713 | 1017 | 1.481 | 2.302 | 0.64 | NTOA != PINT, wrms/TRES |
| mpta | J1658-5324 | 1161 | 1161 | 1488 | 2.089 | 2.772 | 0.75 | NTOA != PINT |
| mpta | J1719-1438 | 2659 | 2659 | 2938 | 3.385 | 3.498 | 0.97 | NTOA != PINT |
| mpta | J1737-0811 | 3258 | 3258 | 3262 | 5.332 | 5.334 | 1.00 | NTOA != PINT |
| mpta | J1832-0836 | 1949 | 1949 | 2242 | 1.510 | 1.573 | 0.96 | NTOA != PINT |
| mpta | J1933-6211 | 2937 | 2937 | 3320 | 1.409 | 1.521 | 0.93 | NTOA != PINT |
| mpta | J1946-5403 | 1815 | 1815 | 2185 | 0.637 | 0.705 | 0.90 | NTOA != PINT |
| mpta | J2124-3358 | 3138 | 3138 | 3150 | 1.739 | 1.741 | 1.00 | NTOA != PINT |
| mpta | J2229+2643 | 2094 | 2094 | 2391 | 1.553 | 1.670 | 0.93 | NTOA != PINT |
| mpta | J2241-5236 | 3405 | 3405 | 6688 | 0.168 | 0.167 | 1.01 | NTOA != PINT |
| mpta | J2317+1439 | 1948 | 1948 | 2293 | 1.532 | 1.654 | 0.93 | NTOA != PINT |
| mpta | J2322+2057 | 1756 | 1756 | 2025 | 2.030 | 2.185 | 0.93 | NTOA != PINT |

### Per-pulsar overlap, main set (ng15, epta_dr2new, ppta_dr3_gh, inpta_dr2, mpta)

122 unique pulsars; number of PTAs per pulsar: {1: 61, 2: 35, 3: 10, 4: 6, 5: 10}. Cells: TOAs / span in yr.

| pulsar | ng15 | epta_dr2new | ppta_dr3_gh | inpta_dr2 | mpta | n | combined span [yr] |
|---|---|---|---|---|---|---|---|
| J0030+0451 | 19571 / 15.5 | 3347 / 9.8 | 593 / 3.2 | 530 / 1.8 | 2880 / 3.7 | 5 | 19.1 |
| J0613-0200 | 17124 / 15.0 | 1750 / 10.1 | 4927 / 18.1 | 5028 / 7.2 | 3067 / 4.3 | 5 | 20.1 |
| J1022+1001 | 3978 / 5.6 | 1804 / 10.1 | 5242 / 18.1 | 3346 / 7.2 | 2945 / 4.2 | 5 | 20.1 |
| J1024-0719 | 12635 / 10.5 | 2112 / 10.1 | 1841 / 18.1 | 83 / 1.1 | 2783 / 4.3 | 5 | 19.5 |
| J1600-3053 | 22955 / 12.5 | 2598 / 9.9 | 5146 / 18.1 | 1012 / 5.9 | 6200 / 4.3 | 5 | 20.1 |
| J1713+0747 | 59389 / 15.5 | 4001 / 10.1 | 5140 / 17.1 | 1103 / 4.2 | 1273 / 2.0 | 5 | 17.2 |
| J1730-2304 | 4870 / 3.4 | 1163 / 9.9 | 3306 / 18.1 | 5585 / 7.2 | 3030 / 4.3 | 5 | 20.1 |
| J1744-1134 | 17745 / 15.7 | 1541 / 9.7 | 5401 / 18.1 | 3075 / 2.9 | 2957 / 4.3 | 5 | 20.1 |
| J1909-3744 | 35037 / 15.5 | 2289 / 9.0 | 9644 / 18.1 | 4160 / 7.2 | 7199 / 4.5 | 5 | 20.1 |
| J2124-3358 | 4982 / 3.5 | 1601 / 9.7 | 3411 / 18.1 | 5153 / 5.9 | 3138 / 4.3 | 5 | 20.1 |
| J0437-4715 | 5830 / 4.8 |  | 11637 / 14.6 | 13283 / 6.8 | 3517 / 4.1 | 4 | 16.7 |
| J0900-3144 |  | 5525 / 9.9 | 1832 / 2.8 | 132 / 1.9 | 3262 / 4.3 | 4 | 12.9 |
| J1455-3330 | 10818 / 15.7 | 2433 / 9.3 |  | 250 / 2.2 | 2448 / 4.3 | 4 | 19.0 |
| J1643-1224 | 22144 / 15.7 |  | 4183 / 18.1 | 5021 / 5.9 | 3162 / 4.3 | 4 | 20.1 |
| J1857+0943 | 7758 / 15.6 | 1156 / 10.0 | 2594 / 18.0 | 1868 / 7.2 |  | 4 | 20.1 |
| J2145-0750 | 18675 / 15.5 |  | 4944 / 18.0 | 4651 / 5.9 | 2966 / 4.3 | 4 | 20.0 |
| J0614-3329 | 1714 / 2.4 |  | 698 / 3.2 |  | 3499 / 4.3 | 3 | 5.7 |
| J1012+5307 | 25837 / 15.5 | 4188 / 10.1 |  | 5952 / 7.1 |  | 3 | 19.4 |
| J1614-2230 | 18445 / 11.5 |  |  | 123 / 2.2 | 3027 / 4.3 | 3 | 14.9 |
| J1640+2224 | 14066 / 15.5 | 1546 / 10.2 |  | 155 / 2.1 |  | 3 | 16.3 |
| J1751-2857 | 2025 / 3.5 | 305 / 9.4 |  |  | 2967 / 4.3 | 3 | 12.3 |
| J1832-0836 | 7739 / 7.1 |  | 385 / 9.2 |  | 1949 / 4.3 | 3 | 10.7 |
| J1843-1113 | 4595 / 3.5 | 736 / 10.1 |  |  | 2886 / 4.3 | 3 | 12.4 |
| J1918-0642 | 18875 / 15.5 | 1138 / 10.1 |  |  | 3123 / 4.0 | 3 | 18.8 |
| J1939+2134 | 23023 / 15.9 |  | 1473 / 17.8 | 18191 / 5.9 |  | 3 | 20.0 |
| J2322+2057 | 3088 / 5.4 | 674 / 9.7 |  |  | 1756 / 3.7 | 3 | 12.1 |
| J0125-2327 |  |  | 2706 / 3.2 |  | 3170 / 4.3 | 2 | 4.6 |
| J0610-2100 | 4885 / 3.4 |  |  |  | 2758 / 3.8 | 2 | 6.7 |
| J0645+5158 | 10403 / 8.9 |  |  | 201 / 2.2 |  | 2 | 8.9 |
| J0711-6830 |  |  | 5538 / 18.1 |  | 2830 / 4.3 | 2 | 19.5 |
| J0740+6620 | 13401 / 6.3 |  |  | 509 / 1.9 |  | 2 | 10.3 |
| J0751+1807 |  | 2469 / 10.1 |  | 1518 / 7.2 |  | 2 | 13.1 |
| J0931-1902 | 5473 / 7.1 |  |  |  | 1903 / 4.0 | 2 | 10.4 |
| J1012-4235 | 797 / 3.4 |  |  |  | 2898 / 4.3 | 2 | 6.7 |
| J1017-7156 |  |  | 5887 / 11.6 |  | 3321 / 4.4 | 2 | 13.0 |
| J1045-4509 |  |  | 4316 / 18.1 |  | 3231 / 4.3 | 2 | 19.5 |
| J1125+7819 | 8723 / 6.3 |  |  | 159 / 1.4 |  | 2 | 10.3 |
| J1125-6014 |  |  | 2832 / 14.2 |  | 2636 / 4.3 | 2 | 15.6 |
| J1446-4701 |  |  | 732 / 11.1 |  | 2207 / 4.4 | 2 | 12.5 |
| J1545-4550 |  |  | 2454 / 10.8 |  | 4905 / 4.3 | 2 | 12.2 |
| J1603-7202 |  |  | 5141 / 18.1 |  | 3121 / 4.4 | 2 | 19.5 |
| J1719-1438 | 6356 / 3.4 |  |  |  | 2659 / 4.3 | 2 | 6.8 |
| J1738+0333 | 8790 / 10.7 | 749 / 10.0 |  |  |  | 2 | 11.3 |
| J1741+1351 | 5582 / 11.0 |  | 111 / 2.6 |  |  | 2 | 12.5 |
| J1747-4036 | 11055 / 8.1 |  |  |  | 3098 / 4.1 | 2 | 11.4 |
| J1801-1417 |  | 384 / 9.7 |  |  | 3069 / 4.3 | 2 | 12.3 |
| J1802-2124 | 6796 / 3.5 |  |  |  | 3039 / 4.4 | 2 | 6.8 |
| J1804-2717 |  | 648 / 9.5 |  |  | 1569 / 3.1 | 2 | 12.3 |
| J1811-2405 | 5266 / 3.5 |  |  |  | 6290 / 4.3 | 2 | 6.8 |
| J1902-5105 |  |  | 501 / 2.8 |  | 3438 / 4.3 | 2 | 4.3 |
| J1910+1256 | 6486 / 11.4 | 460 / 9.9 |  |  |  | 2 | 12.0 |
| J1911+1347 | 3786 / 7.0 | 811 / 9.9 |  |  |  | 2 | 9.9 |
| J1933-6211 |  |  | 893 / 3.2 |  | 2937 / 4.4 | 2 | 4.6 |
| J1944+0907 | 5328 / 12.5 |  |  | 1098 / 1.9 |  | 2 | 16.1 |
| J2010-1323 | 17077 / 10.5 |  |  |  | 3121 / 4.3 | 2 | 13.8 |
| J2129-5721 |  |  | 2921 / 17.7 |  | 3039 / 4.3 | 2 | 19.1 |
| J2229+2643 | 3711 / 7.0 |  |  |  | 2094 / 3.7 | 2 | 10.0 |
| J2234+0944 | 7535 / 7.1 |  |  |  | 2365 / 3.7 | 2 | 10.1 |
| J2241-5236 |  |  | 6238 / 12.1 |  | 3405 / 4.4 | 2 | 13.5 |
| J2302+4442 | 10211 / 8.1 |  |  | 313 / 6.7 |  | 2 | 11.6 |
| J2317+1439 | 13927 / 15.6 |  |  |  | 1948 / 3.7 | 2 | 18.6 |
| B1953+29 | 5126 / 11.1 |  |  |  |  | 1 | 11.1 |
| J0023+0923 | 15896 / 9.0 |  |  |  |  | 1 | 9.0 |
| J0034-0534 |  |  |  | 621 / 1.3 |  | 1 | 1.3 |
| J0101-6422 |  |  |  |  | 1430 / 3.8 | 1 | 3.8 |
| J0340+4130 | 11093 / 8.1 |  |  |  |  | 1 | 8.1 |
| J0406+3039 | 2446 / 3.6 |  |  |  |  | 1 | 3.6 |
| J0509+0856 | 2169 / 3.6 |  |  |  |  | 1 | 3.6 |
| J0557+1551 | 525 / 4.6 |  |  |  |  | 1 | 4.6 |
| J0605+3757 | 554 / 3.4 |  |  |  |  | 1 | 3.4 |
| J0636+5128 | 32222 / 6.3 |  |  |  |  | 1 | 6.3 |
| J0636-3044 |  |  |  |  | 2217 / 4.2 | 1 | 4.2 |
| J0709+0458 | 3030 / 4.6 |  |  |  |  | 1 | 4.6 |
| J0955-6150 |  |  |  |  | 6167 / 4.3 | 1 | 4.3 |
| J1036-8317 |  |  |  |  | 2046 / 4.1 | 1 | 4.1 |
| J1101-6424 |  |  |  |  | 3693 / 4.3 | 1 | 4.3 |
| J1125-5825 |  |  |  |  | 3233 / 4.4 | 1 | 4.4 |
| J1216-6410 |  |  |  |  | 3256 / 4.4 | 1 | 4.4 |
| J1231-1411 |  |  |  |  | 2129 / 4.3 | 1 | 4.3 |
| J1312+0051 | 1705 / 4.6 |  |  |  |  | 1 | 4.6 |
| J1327-0755 |  |  |  |  | 788 / 3.5 | 1 | 3.5 |
| J1421-4409 |  |  |  |  | 2994 / 4.3 | 1 | 4.3 |
| J1431-5740 |  |  |  |  | 3138 / 4.4 | 1 | 4.4 |
| J1435-6100 |  |  |  |  | 3976 / 4.4 | 1 | 4.4 |
| J1453+1902 | 2551 / 7.0 |  |  |  |  | 1 | 7.0 |
| J1514-4946 |  |  |  |  | 713 / 3.6 | 1 | 3.6 |
| J1525-5545 |  |  |  |  | 9504 / 3.6 | 1 | 3.6 |
| J1543-5149 |  |  |  |  | 2685 / 4.4 | 1 | 4.4 |
| J1547-5709 |  |  |  |  | 2870 / 4.2 | 1 | 4.2 |
| J1629-6902 |  |  |  |  | 3106 / 4.4 | 1 | 4.4 |
| J1630+3734 | 1815 / 3.5 |  |  |  |  | 1 | 3.5 |
| J1652-4838 |  |  |  |  | 3124 / 4.3 | 1 | 4.3 |
| J1653-2054 |  |  |  |  | 2686 / 4.3 | 1 | 4.3 |
| J1658-5324 |  |  |  |  | 1161 / 4.1 | 1 | 4.1 |
| J1705-1903 | 9871 / 3.7 |  |  |  |  | 1 | 3.7 |
| J1708-3506 |  |  |  |  | 2492 / 3.1 | 1 | 3.1 |
| J1721-2457 |  |  |  |  | 2112 / 3.3 | 1 | 3.3 |
| J1732-5049 |  |  |  |  | 4442 / 4.5 | 1 | 4.5 |
| J1737-0811 |  |  |  |  | 3258 / 4.3 | 1 | 4.3 |
| J1745+1017 | 3017 / 4.5 |  |  |  |  | 1 | 4.5 |
| J1757-5322 |  |  |  |  | 3323 / 4.2 | 1 | 4.2 |
| J1804-2858 |  |  |  |  | 2167 / 4.0 | 1 | 4.0 |
| J1824-2452A |  |  | 1284 / 17.0 |  |  | 1 | 17.0 |
| J1825-0319 |  |  |  |  | 3044 / 4.0 | 1 | 4.0 |
| J1843-1448 |  |  |  |  | 1969 / 4.3 | 1 | 4.3 |
| J1853+1303 | 4570 / 9.1 |  |  |  |  | 1 | 9.1 |
| J1903+0327 | 6856 / 10.7 |  |  |  |  | 1 | 10.7 |
| J1903-7051 |  |  |  |  | 2913 / 4.4 | 1 | 4.4 |
| J1911-1114 |  |  |  |  | 1804 / 3.0 | 1 | 3.0 |
| J1923+2515 | 3974 / 9.0 |  |  |  |  | 1 | 9.0 |
| J1946+3417 | 4743 / 5.7 |  |  |  |  | 1 | 5.7 |
| J1946-5403 |  |  |  |  | 1815 / 4.0 | 1 | 4.0 |
| J2017+0603 | 3512 / 8.3 |  |  |  |  | 1 | 8.3 |
| J2033+1734 | 3847 / 7.0 |  |  |  |  | 1 | 7.0 |
| J2039-3616 |  |  |  |  | 1917 / 4.0 | 1 | 4.0 |
| J2043+1711 | 7397 / 9.1 |  |  |  |  | 1 | 9.1 |
| J2150-0326 |  |  |  |  | 2091 / 4.0 | 1 | 4.0 |
| J2214+3000 | 7406 / 8.4 |  |  |  |  | 1 | 8.4 |
| J2222-0137 |  |  |  |  | 2993 / 4.0 | 1 | 4.0 |
| J2234+0611 | 3566 / 6.5 |  |  |  |  | 1 | 6.5 |
| J2236-5527 |  |  |  |  | 1599 / 4.3 | 1 | 4.3 |
| J2322-2650 |  |  |  |  | 1967 / 3.8 | 1 | 3.8 |

### Per-pulsar overlap, Yu & Allen set (ng15, epta_dr2new, ppta_dr3_gh, inpta_dr1, mpta)

121 unique pulsars; number of PTAs per pulsar: {1: 65, 2: 32, 3: 10, 4: 7, 5: 7}. Cells: TOAs / span in yr.

| pulsar | ng15 | epta_dr2new | ppta_dr3_gh | inpta_dr1 | mpta | n | combined span [yr] |
|---|---|---|---|---|---|---|---|
| J0613-0200 | 17124 / 15.0 | 1750 / 10.1 | 4927 / 18.1 | 399 / 3.4 | 3067 / 4.3 | 5 | 19.5 |
| J1022+1001 | 3978 / 5.6 | 1804 / 10.1 | 5242 / 18.1 | 477 / 3.4 | 2945 / 4.2 | 5 | 19.5 |
| J1600-3053 | 22955 / 12.5 | 2598 / 9.9 | 5146 / 18.1 | 171 / 3.4 | 6200 / 4.3 | 5 | 19.5 |
| J1713+0747 | 59389 / 15.5 | 4001 / 10.1 | 5140 / 17.1 | 772 / 2.9 | 1273 / 2.0 | 5 | 17.2 |
| J1744-1134 | 17745 / 15.7 | 1541 / 9.7 | 5401 / 18.1 | 208 / 0.4 | 2957 / 4.3 | 5 | 19.5 |
| J1909-3744 | 35037 / 15.5 | 2289 / 9.0 | 9644 / 18.1 | 446 / 3.4 | 7199 / 4.5 | 5 | 19.5 |
| J2124-3358 | 4982 / 3.5 | 1601 / 9.7 | 3411 / 18.1 | 827 / 3.4 | 3138 / 4.3 | 5 | 19.5 |
| J0030+0451 | 19571 / 15.5 | 3347 / 9.8 | 593 / 3.2 |  | 2880 / 3.7 | 4 | 18.5 |
| J0437-4715 | 5830 / 4.8 |  | 11637 / 14.6 | 451 / 0.8 | 3517 / 4.1 | 4 | 16.0 |
| J1024-0719 | 12635 / 10.5 | 2112 / 10.1 | 1841 / 18.1 |  | 2783 / 4.3 | 4 | 19.5 |
| J1643-1224 | 22144 / 15.7 |  | 4183 / 18.1 | 528 / 3.4 | 3162 / 4.3 | 4 | 19.5 |
| J1730-2304 | 4870 / 3.4 | 1163 / 9.9 | 3306 / 18.1 |  | 3030 / 4.3 | 4 | 19.5 |
| J1857+0943 | 7758 / 15.6 | 1156 / 10.0 | 2594 / 18.0 | 406 / 3.4 |  | 4 | 18.0 |
| J2145-0750 | 18675 / 15.5 |  | 4944 / 18.0 | 1808 / 3.4 | 2966 / 4.3 | 4 | 19.3 |
| J0614-3329 | 1714 / 2.4 |  | 698 / 3.2 |  | 3499 / 4.3 | 3 | 5.7 |
| J0900-3144 |  | 5525 / 9.9 | 1832 / 2.8 |  | 3262 / 4.3 | 3 | 12.3 |
| J1012+5307 | 25837 / 15.5 | 4188 / 10.1 |  | 357 / 3.4 |  | 3 | 17.0 |
| J1455-3330 | 10818 / 15.7 | 2433 / 9.3 |  |  | 2448 / 4.3 | 3 | 19.0 |
| J1751-2857 | 2025 / 3.5 | 305 / 9.4 |  |  | 2967 / 4.3 | 3 | 12.3 |
| J1832-0836 | 7739 / 7.1 |  | 385 / 9.2 |  | 1949 / 4.3 | 3 | 10.7 |
| J1843-1113 | 4595 / 3.5 | 736 / 10.1 |  |  | 2886 / 4.3 | 3 | 12.4 |
| J1918-0642 | 18875 / 15.5 | 1138 / 10.1 |  |  | 3123 / 4.0 | 3 | 18.8 |
| J1939+2134 | 23023 / 15.9 |  | 1473 / 17.8 | 1483 / 3.4 |  | 3 | 17.8 |
| J2322+2057 | 3088 / 5.4 | 674 / 9.7 |  |  | 1756 / 3.7 | 3 | 12.1 |
| J0125-2327 |  |  | 2706 / 3.2 |  | 3170 / 4.3 | 2 | 4.6 |
| J0610-2100 | 4885 / 3.4 |  |  |  | 2758 / 3.8 | 2 | 6.7 |
| J0711-6830 |  |  | 5538 / 18.1 |  | 2830 / 4.3 | 2 | 19.5 |
| J0751+1807 |  | 2469 / 10.1 |  | 196 / 3.4 |  | 2 | 10.6 |
| J0931-1902 | 5473 / 7.1 |  |  |  | 1903 / 4.0 | 2 | 10.4 |
| J1012-4235 | 797 / 3.4 |  |  |  | 2898 / 4.3 | 2 | 6.7 |
| J1017-7156 |  |  | 5887 / 11.6 |  | 3321 / 4.4 | 2 | 13.0 |
| J1045-4509 |  |  | 4316 / 18.1 |  | 3231 / 4.3 | 2 | 19.5 |
| J1125-6014 |  |  | 2832 / 14.2 |  | 2636 / 4.3 | 2 | 15.6 |
| J1446-4701 |  |  | 732 / 11.1 |  | 2207 / 4.4 | 2 | 12.5 |
| J1545-4550 |  |  | 2454 / 10.8 |  | 4905 / 4.3 | 2 | 12.2 |
| J1603-7202 |  |  | 5141 / 18.1 |  | 3121 / 4.4 | 2 | 19.5 |
| J1614-2230 | 18445 / 11.5 |  |  |  | 3027 / 4.3 | 2 | 14.9 |
| J1640+2224 | 14066 / 15.5 | 1546 / 10.2 |  |  |  | 2 | 16.3 |
| J1719-1438 | 6356 / 3.4 |  |  |  | 2659 / 4.3 | 2 | 6.8 |
| J1738+0333 | 8790 / 10.7 | 749 / 10.0 |  |  |  | 2 | 11.3 |
| J1741+1351 | 5582 / 11.0 |  | 111 / 2.6 |  |  | 2 | 12.5 |
| J1747-4036 | 11055 / 8.1 |  |  |  | 3098 / 4.1 | 2 | 11.4 |
| J1801-1417 |  | 384 / 9.7 |  |  | 3069 / 4.3 | 2 | 12.3 |
| J1802-2124 | 6796 / 3.5 |  |  |  | 3039 / 4.4 | 2 | 6.8 |
| J1804-2717 |  | 648 / 9.5 |  |  | 1569 / 3.1 | 2 | 12.3 |
| J1811-2405 | 5266 / 3.5 |  |  |  | 6290 / 4.3 | 2 | 6.8 |
| J1902-5105 |  |  | 501 / 2.8 |  | 3438 / 4.3 | 2 | 4.3 |
| J1910+1256 | 6486 / 11.4 | 460 / 9.9 |  |  |  | 2 | 12.0 |
| J1911+1347 | 3786 / 7.0 | 811 / 9.9 |  |  |  | 2 | 9.9 |
| J1933-6211 |  |  | 893 / 3.2 |  | 2937 / 4.4 | 2 | 4.6 |
| J2010-1323 | 17077 / 10.5 |  |  |  | 3121 / 4.3 | 2 | 13.8 |
| J2129-5721 |  |  | 2921 / 17.7 |  | 3039 / 4.3 | 2 | 19.1 |
| J2229+2643 | 3711 / 7.0 |  |  |  | 2094 / 3.7 | 2 | 10.0 |
| J2234+0944 | 7535 / 7.1 |  |  |  | 2365 / 3.7 | 2 | 10.1 |
| J2241-5236 |  |  | 6238 / 12.1 |  | 3405 / 4.4 | 2 | 13.5 |
| J2317+1439 | 13927 / 15.6 |  |  |  | 1948 / 3.7 | 2 | 18.6 |
| B1953+29 | 5126 / 11.1 |  |  |  |  | 1 | 11.1 |
| J0023+0923 | 15896 / 9.0 |  |  |  |  | 1 | 9.0 |
| J0101-6422 |  |  |  |  | 1430 / 3.8 | 1 | 3.8 |
| J0340+4130 | 11093 / 8.1 |  |  |  |  | 1 | 8.1 |
| J0406+3039 | 2446 / 3.6 |  |  |  |  | 1 | 3.6 |
| J0509+0856 | 2169 / 3.6 |  |  |  |  | 1 | 3.6 |
| J0557+1551 | 525 / 4.6 |  |  |  |  | 1 | 4.6 |
| J0605+3757 | 554 / 3.4 |  |  |  |  | 1 | 3.4 |
| J0636+5128 | 32222 / 6.3 |  |  |  |  | 1 | 6.3 |
| J0636-3044 |  |  |  |  | 2217 / 4.2 | 1 | 4.2 |
| J0645+5158 | 10403 / 8.9 |  |  |  |  | 1 | 8.9 |
| J0709+0458 | 3030 / 4.6 |  |  |  |  | 1 | 4.6 |
| J0740+6620 | 13401 / 6.3 |  |  |  |  | 1 | 6.3 |
| J0955-6150 |  |  |  |  | 6167 / 4.3 | 1 | 4.3 |
| J1036-8317 |  |  |  |  | 2046 / 4.1 | 1 | 4.1 |
| J1101-6424 |  |  |  |  | 3693 / 4.3 | 1 | 4.3 |
| J1125+7819 | 8723 / 6.3 |  |  |  |  | 1 | 6.3 |
| J1125-5825 |  |  |  |  | 3233 / 4.4 | 1 | 4.4 |
| J1216-6410 |  |  |  |  | 3256 / 4.4 | 1 | 4.4 |
| J1231-1411 |  |  |  |  | 2129 / 4.3 | 1 | 4.3 |
| J1312+0051 | 1705 / 4.6 |  |  |  |  | 1 | 4.6 |
| J1327-0755 |  |  |  |  | 788 / 3.5 | 1 | 3.5 |
| J1421-4409 |  |  |  |  | 2994 / 4.3 | 1 | 4.3 |
| J1431-5740 |  |  |  |  | 3138 / 4.4 | 1 | 4.4 |
| J1435-6100 |  |  |  |  | 3976 / 4.4 | 1 | 4.4 |
| J1453+1902 | 2551 / 7.0 |  |  |  |  | 1 | 7.0 |
| J1514-4946 |  |  |  |  | 713 / 3.6 | 1 | 3.6 |
| J1525-5545 |  |  |  |  | 9504 / 3.6 | 1 | 3.6 |
| J1543-5149 |  |  |  |  | 2685 / 4.4 | 1 | 4.4 |
| J1547-5709 |  |  |  |  | 2870 / 4.2 | 1 | 4.2 |
| J1629-6902 |  |  |  |  | 3106 / 4.4 | 1 | 4.4 |
| J1630+3734 | 1815 / 3.5 |  |  |  |  | 1 | 3.5 |
| J1652-4838 |  |  |  |  | 3124 / 4.3 | 1 | 4.3 |
| J1653-2054 |  |  |  |  | 2686 / 4.3 | 1 | 4.3 |
| J1658-5324 |  |  |  |  | 1161 / 4.1 | 1 | 4.1 |
| J1705-1903 | 9871 / 3.7 |  |  |  |  | 1 | 3.7 |
| J1708-3506 |  |  |  |  | 2492 / 3.1 | 1 | 3.1 |
| J1721-2457 |  |  |  |  | 2112 / 3.3 | 1 | 3.3 |
| J1732-5049 |  |  |  |  | 4442 / 4.5 | 1 | 4.5 |
| J1737-0811 |  |  |  |  | 3258 / 4.3 | 1 | 4.3 |
| J1745+1017 | 3017 / 4.5 |  |  |  |  | 1 | 4.5 |
| J1757-5322 |  |  |  |  | 3323 / 4.2 | 1 | 4.2 |
| J1804-2858 |  |  |  |  | 2167 / 4.0 | 1 | 4.0 |
| J1824-2452A |  |  | 1284 / 17.0 |  |  | 1 | 17.0 |
| J1825-0319 |  |  |  |  | 3044 / 4.0 | 1 | 4.0 |
| J1843-1448 |  |  |  |  | 1969 / 4.3 | 1 | 4.3 |
| J1853+1303 | 4570 / 9.1 |  |  |  |  | 1 | 9.1 |
| J1903+0327 | 6856 / 10.7 |  |  |  |  | 1 | 10.7 |
| J1903-7051 |  |  |  |  | 2913 / 4.4 | 1 | 4.4 |
| J1911-1114 |  |  |  |  | 1804 / 3.0 | 1 | 3.0 |
| J1923+2515 | 3974 / 9.0 |  |  |  |  | 1 | 9.0 |
| J1944+0907 | 5328 / 12.5 |  |  |  |  | 1 | 12.5 |
| J1946+3417 | 4743 / 5.7 |  |  |  |  | 1 | 5.7 |
| J1946-5403 |  |  |  |  | 1815 / 4.0 | 1 | 4.0 |
| J2017+0603 | 3512 / 8.3 |  |  |  |  | 1 | 8.3 |
| J2033+1734 | 3847 / 7.0 |  |  |  |  | 1 | 7.0 |
| J2039-3616 |  |  |  |  | 1917 / 4.0 | 1 | 4.0 |
| J2043+1711 | 7397 / 9.1 |  |  |  |  | 1 | 9.1 |
| J2150-0326 |  |  |  |  | 2091 / 4.0 | 1 | 4.0 |
| J2214+3000 | 7406 / 8.4 |  |  |  |  | 1 | 8.4 |
| J2222-0137 |  |  |  |  | 2993 / 4.0 | 1 | 4.0 |
| J2234+0611 | 3566 / 6.5 |  |  |  |  | 1 | 6.5 |
| J2236-5527 |  |  |  |  | 1599 / 4.3 | 1 | 4.3 |
| J2302+4442 | 10211 / 8.1 |  |  |  |  | 1 | 8.1 |
| J2322-2650 |  |  |  |  | 1967 / 3.8 | 1 | 3.8 |
