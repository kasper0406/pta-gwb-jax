# M3a validation: multi-leg container, physics-preserving ingestion, general likelihood

M3a is the first M3 implementation milestone (`docs/M3_PLAN.md` Sec. 5.1). It is CPU-only
deterministic infrastructure plus validation; there is no sampling and no evidence work. This
document gives every gate result with its numbers, its tolerance and whether it passed. Tolerances
were fixed in code before the corresponding result was seen. The one exception is the G3/G4
"likelihood impact" column, which is post hoc and labelled as such.

Measured 2026-10-08 on CPU (JAX 0.11.2, float64, PINT 1.1.7). Oracles: tempo2 2026.04.1 with
libstempo 2.5.1 (isolated conda-forge env, `scripts/setup_tempo2_env.sh`, run through
`scripts/t2py`); MetaPulsar v0.9.3 (d2067ab, from source); enterprise 3.5.0 and discovery 0.5
(oracle group).

How to reproduce:

```bash
scripts/setup_tempo2_env.sh                          # tempo2 + libstempo oracle, no root
PY=python; export JAX_PLATFORMS=cpu PYTHONPATH=src
$PY scripts/m3a_validate.py legs --pin --jobs 20     # G1/G2 for all 249 legs (first run pins the clock files)
$PY scripts/m3a_validate.py multileg                 # B and C for the three validation multi-leg pulsars
$PY scripts/m3a_validate.py multileg --only J1909-3744 --timing shared --refs PPTA EPTA --force-clock 'TT(BIPM2019)'
$PY scripts/m3a_validate.py multileg --timing shared --local-dm
$PY scripts/m3a_validate.py tempo2                   # G3/G4
$PY scripts/m3a_validate.py g5 | g6 | g7 | g8        # remaining gates
PTAGWB_REQUIRE_ORACLES=1 $PY -m pytest tests/        # strict suite
```

Results are written to `data/processed/m3a/results/*.json` (git-ignored). The tables below were
copied from those files.

---

## 1. What was built

| File | Content |
|---|---|
| `src/ptagwb/timfile.py` | tempo2-semantics tim reader (`readTim` of tempo2 2026.04.1): per-file TIME/SKIP/END state, INCLUDE, first-character comments, the whole-line flag scan (`-x` then the next token), `-to`/`-addsat`/`-padd`, fixed-column (FORMAT-less) files that contain only INCLUDEs. Unsupported commands are refused rather than guessed. Also the flat canonical tim writer: one file, no INCLUDE/TIME/END/SKIP, the effective offset written as a single `-to` |
| `src/ptagwb/legs.py` | Leg ingestion: par canonicalisation (survey rules plus new ones below), multi-valued mask flags turned into indicator flags, profiles, the PINT load with every warning captured and classified (G2), the G1 identity check TOA by TOA, the clock-coverage audit with explicit exclusion, freezing of empty mask parameters, export |
| `src/ptagwb/profiles.py` | Versioned clock/ephemeris profiles: published-analysis per PTA (and InPTA DR1), `combined-v1` (TT(BIPM2023)/DE440), `ya-v3-clocks-v1`. Clock files are pinned by sha256 (`configs/m3/clocks/*.json`), and a post-load check verifies that every file PINT used came from the pinned directory. Evaluator profiles (`ell1h_shapiro`, `allow_T2`, `allow_tcb`). Site-coordinate profile (`configs/m3/sites/tempo2-2026.04.1.json`) |
| `src/ptagwb/multileg.py` | Multi-leg container. Option B (`per_leg`, MetaPulsar "composite") and option C (`shared`, an independent re-implementation of the MetaPulsar v0.9.3 `make_parfiles_consistent` rewrite). Shared columns are merged, detector columns are per leg, systems are namespaced `<pta>:<flag>`. Reference choice (NG15 > EPTA > PPTA > MPTA). The `local_dm` variant is the MetaPulsar-main configuration. Duplicate-observation finder (G7). Save/load |
| `src/ptagwb/gp.py` | `FourierBlock`: a GP block with its own span, mode count, time origin, chromatic index (`(fref/nu)^idx`), TempoNest DM normalisation and row selection (flag or frequency band) |
| `src/ptagwb/combined.py` | The general likelihood. The column layout takes the union of block columns; equal grids merge, so M1's prefix identity is a special case. Stage 1 has fixed blocks absorbed in square-root form (augmented QR). Stage 2 uses M1's reducers and Sigma' core unchanged, with zero-information padding for variable K_a. `GeneralPTALikelihood` takes CURN/HD/any ORF and power-law or free-spectrum common processes, with the production or fast (`hh`/`levels`) kernels |
| `src/ptagwb/noise.py` | `GeneralWhiteNoise`: overlapping ECORR terms as additive epoch blocks, whitened exactly by a Cholesky factor per connected component. TN vs T2 EQUAD per system. PTA namespacing |
| `src/ptagwb/m3data.py`, `configs/m3/*.json` | Data-set discovery, the quarantine registry, the validation set |
| `scripts/m3a_validate.py`, `scripts/t2_dump.py`, `scripts/setup_tempo2_env.sh`, `scripts/t2py` | Validation driver; the tempo2 oracle |
| `tests/test_m3a_*.py`, `tests/m3a_oracles.py`, `tests/m3a_injection.py`, `tests/dense_oracle.py`, `tests/synthetic_m3.py` | Strict-suite tests and oracle glue |

The NG15 path (`likelihood.py`, `perf_likelihood.py`, `data.py`) is unchanged.

### New canonicalisation rules found during M3a

Every rule is applied explicitly and recorded in the leg's provenance (`leg_meta.json`).

* **Multi-valued mask flags (PPTA, all 32 legs).** A UWL TOA carries `-j MEDUSA_59200 -j
  MEDUSA_58925`, and the par has one `JUMP -j` line for each value. tempo2 applies every
  matching JUMP; PINT keeps one value per flag (the last), so one JUMP selected nothing. That is
  a lost ~3 us JUMP and a zero design column. The flag becomes per-value indicator flags
  (`-j__MEDUSA_59200 1`), and the par's mask lines are rewritten to select on them.
* **"NAME value X" lines (tempo2 pars, 62 legs).** tempo2 (`readParfile.C` readValue,
  nread = 2) treats X as an uncertainty and leaves the parameter frozen unless X is 0/1/2. PINT
  keeps the parameter's default fit state instead, which made InPTA DR2 `DMX_0001` free. The
  rule writes `NAME value 0 X`.
* **The InPTA DR2 `DMX_0001` template is written explicitly frozen.**
* **tempo2 flag artefacts inside the TOA fields.** EPTA archive names such as
  `20140126-09630-.cal` produce a tempo2 "flag" `-.cal`. These are dropped from the flat file;
  they are kept in the records for the libstempo comparison. 8 lines are affected.
* **Comments.** tempo2 treats a line as a comment only if its *first character* is `C` or `#`.
  InPTA DR1's 6 `CJ...` lines are commented-out TOAs (PINT would read them as TOAs), which
  resolves the survey's "+6 unexplained TOAs": our count for the YA set is **1,090,206**, YA's
  number exactly. Indented ` C ...` lines are not comments; they are unparseable and dropped,
  as tempo2 does.
* **Site coordinates.** PINT's GMRT position differs from tempo2's by **~750 m**, Effelsberg's
  by 2.8 m and Jodrell's by 0.5 m. Found by G3: InPTA residuals differed by 830 ns rms. tempo2
  PTAs now use tempo2's coordinates (site profile); NG15 keeps PINT's.
* **Clock coverage.** Some TOAs have no clock correction good to 1 ns: EPTA effix2gps ends at
  MJD 59294.5, and eff2gps.clk has no data after 57195.5 apart from a sentinel 0 at 60000
  ("No maser file found"). PINT clamps there while tempo2 interpolates to the sentinel. These
  TOAs are excluded explicitly (policy `exclude-uncovered`, default; `keep` is available):
  **65 TOAs in 7 EPTA legs** (J1600-3053: 54; J1640+2224: 3; the others 1-2 each).

## 2. Gate summary

| Gate | Scope | Oracle | Tolerance (fixed in advance) | Result | Pass |
|---|---|---|---|---|---|
| G1 TOA identity | all 249 legs (selected configuration + InPTA DR1) | tempo2-semantics records; libstempo on fixtures and real legs | count exact, dt < 2 ns, err/freq/observatory/every flag/-padd identical | 248/248 loadable legs, max dt 0.81 ns, 0 mismatches; fixtures match libstempo to < 1e-9 s | **PASS** |
| G2 warnings | all loadable legs | classification table + per-leg audits | none unexplained | 248/248 | **PASS** |
| G3 projected residuals | validation set (non-quarantined) | tempo2 2026.04.1 | rms < 1 ns and < 0.01 sigma | see Sec. 4 | **FAIL** (1/15 within tolerance) |
| G4 weighted column space | validation set | tempo2; MetaPulsar v0.9.3 | sin(max principal angle) < 1e-6, equal dimension | tempo2: 1e-5 to 7e-3; MetaPulsar: < 4e-8 | **FAIL** vs tempo2; **PASS** vs MetaPulsar |
| G5 likelihood + gradient | dense oracle (synthetic) + real multi-leg systems (B, C) | long-double dense; enterprise; discovery | 1e-9 relative (value), 1e-8 (gradient) | see Sec. 5 | **PASS** |
| G6 reference invariance | option C, J1909-3744, references NG15/PPTA/EPTA | ours | shape <= 0.1 nats; linearisation (whitened) <= 0.1; sin <= 1e-3 | see Sec. 6 | **FAIL** |
| G7 duplicates | 61 (YA: 56) multi-leg pulsars | TOA matching | none, or explicit | 0 cross-PTA; 1 LEAP/WSRT candidate in quarantined EPTA J1600 | **PASS** |
| G8 injections | J1909 + J1022 + J0437, A/B/C, signal and null | ours | score z < 3.5; Var/I within 1 +- 4.5 sqrt(2/N) | 6/6 | **PASS** |
| G9 NG15 regression | 67 NG15 pulsars, CURN + HD, production + fast | M1/M2 | bit-identical or <= 1e-9 / 1e-8 | 32/32 values bit-identical; grad <= 4.2e-14 | **PASS** |

## 3. G1 and G2 (all legs)

| data set | legs | loaded | G1 pass | TOAs (PINT) | max dt [ns] | flag/obs/padd mismatches | G2 unexplained | clock-excluded TOAs | quarantined |
|---|---|---|---|---|---|---|---|---|---|
| ng15 | 68 | 68 | 68 | 676,397 | 0.807 | 0 | 0 | 0 | 0 |
| epta_dr2new | 25 | 25 | 25 | 45,428 | 0.806 | 0 | 0 | 65 | 3 |
| ppta_dr3_gh | 32 | 32 | 32 | 113,951 | 0.614 | 0 | 0 | 0 | 4 |
| inpta_dr2 | 27 | 27 | 27 | 83,120 | 0.614 | 0 | 0 | 0 | 2 |
| mpta | 83 | 82 | 82 | 242,863 | 0.802 | 0 | 0 | 0 | 6 |
| inpta_dr1 | 14 | 14 | 14 | 8,523 | 0.704 | 0 | 0 | 0 | 0 |

The one leg that does not load is MPTA J1825-0319 (signed H3, quarantined; Sec. 8).

**G1 extended metadata.** In addition to time, count, uncertainty and frequency, every TOA's
observatory (resolved through PINT's aliases), every flag (case-insensitively, since PINT
lower-cases keys) and its `-padd` phase are compared. The flat file keeps tim order and names
(`toaNNNNNNN`), so the match is by index, not by sorting.

**G1 against tempo2 itself.**
1. Two fixture trees (`tests/test_m3a_timfile.py`) are read identically by libstempo, to 1e-9 s
   per TOA with identical flags. They cover TIME before INCLUDE, TIME in a child, SKIP around
   TIME/INCLUDE/END, END in a child, a SKIP left open in a child, `CJ` vs ` C` comments,
   valueless flags, `-to`, `-padd`, a FORMAT-less top file and a `-.cal` artefact.
2. On real legs, the SATs tempo2 reads equal our records to 0.008 ns (InPTA DR2 J1909, PPTA
   J1022) and 0.25 ns (EPTA J1744, TIME offsets).

This replaces the survey reader's incorrect parent-offset assumption (review minor #2).

**G2.** Every loguru and Python warning of every category is captured.

| warning class | legs | explanation |
|---|---|---|
| clock-override | 248 | the pinned clock directory is in use |
| tcb-to-tdb | 180 | TCB par converted by PINT (YA convert too); quantified by G3 |
| unsupported-tempo2-setting | 180 | DILATEFREQ Y, TIMEEPH IF99, T2CMETHOD. Switching them in tempo2 changes G3 by < 0.3 ns (J1802, J1744) |
| ignored-par-metadata | 180 | EPHVER, DM_SERIES, ... |
| t2-binary-resolution | 43 | T2 resolved to ELL1/DD/DDK/ELL1H (evaluator class; G3/G4) |
| clock-coverage-audited | 20 | coverage audit passed after the explicit exclusions above |
| ddk-kin / ddk-a1dot | 10 / 1 | DDK uses KIN as tempo2 does |
| dmx-overlap | 8 | DMX ranges overlap as released |
| pint-parse-overflow | 14 | over-long numeric literal; values checked by G3 |
| mask-no-toas | 2 | parameter without TOAs (frozen, as tempo2 refuses to fit it) |
| ddgr-default-init | 1 | MPTA J0955-6150: PINT's DDGR set-up divides by zero with default parameters before the par values are set (located with numpy `seterr(divide='raise')`) |
| altitude, tzr-default, pulse-numbers-partial, resource | 1-248 | benign |

Free mask parameters that select no TOA are frozen, recorded per leg. That covers 31 PPTA legs
(old-backend JUMPs absent from the pulsar), plus NG15 J1024-0719 F3 (M1's checked exception).

## 4. G3 / G4 against tempo2 (validation set)

<!-- G3G4_TABLE -->

How the comparison is done. The par file is the released one, with the leg's published clock
profile, without TRACK. The tim file is the released tree. tempo2's clock directory is overlaid
with our pinned clock files, matched by name and by `# from to` header: PINT's
`mk2utc_observatory.clk` is tempo2's (older) `mk2utc.clk`. For NG15, tempo2 reads the release's
`alternate/tempo2` par, because it cannot read the PINT par ("Date -nan out of range of TDB-TDT
table"). The difference is projected out of PINT's weighted column space (columns normalised,
twice-orthogonalised QR) and expressed in ns and in units of the TOA uncertainty. G4 compares the
weighted column spaces by principal angles.

**Diagnosis.** The initial differences were 15-1200 ns. Four causes were found and fixed:
1. Our own projection: unnormalised columns over 20 decades were truncated in the SVD.
2. GMRT, Effelsberg and Jodrell site coordinates (InPTA: 830 ns -> 4 ns).
3. Clock-file versions (overlay by header).
4. The InPTA `value uncertainty` fit-flag rule (DMX_0001 column).

The remaining differences are 1-5 ns (0.001-0.05 sigma) for most legs, and ~30 ns for PPTA J1022
and J0437. J0437 is DDK: its difference follows orbital phase (+-12 ns) with an annual modulation,
the signature of the Kopeikin terms. The G4 angles come from derivative formulae, not from
missing columns:
* astrometry: RAJ/DECJ 2e-5; ELAT/PMELAT 4e-3 for the near-ecliptic J1022;
* binary: EPS1/EPS2 and PB 1e-4.

No column is missing. **These differences fail the pre-registered 1 ns / 1e-6 tolerances.** The
post-hoc likelihood-impact column (same IRN+DM single-pulsar model on PINT's vs tempo2's
residuals and design matrix) quantifies the consequence.

**Option C and B vs MetaPulsar v0.9.3 (G4/G5 for the container).** Our independent
re-implementation of the v0.9.3 consistent rewrite and composite stacking reproduces
MetaPulsar's own objects. The inputs are the same canonical pars and flat tims, and MetaPulsar
builds every leg with PINT.

| pulsar | config | TOAs | columns ours / MetaPulsar | max abs residual difference per leg | column-space sin |
|---|---|---|---|---|---|
| J1909-3744 (5 legs) | C (ref NG15) | 58,329 | 72 / 72 | 0.0 ns (all 5 legs) | 3.9e-8 |
| J1909-3744 (5 legs) | B | 58,329 | 459 / 459 | 0.0 ns | 3.3e-8 |
| J0437-4715 (3 legs) | C (ref PPTA) and B | (strict test `test_container_vs_metapulsar`) | equal | < 1e-3 ns | < 1e-6 |

Harness adaptation: v0.9.3 copies the reference's `CLOCK` *or* `CLK` key verbatim. A `CLOCK`
reference therefore leaves a `CLK` target with both keys, and PINT refuses the par (a v0.9.3
bug). The harness spells the keyword `CLK` everywhere before calling MetaPulsar.

## 5. G5 pointwise likelihood and gradient

**Dense oracle** (`tests/test_m3a_likelihood.py`, `tests/dense_oracle.py`). Two pulsars, three
legs from two PTAs with staggered spans and equal raw backend labels (namespaced). The system has
TN and T2 EQUAD legs, overlapping ECORR (per-system plus global), IRN on each pulsar's own span,
a DM block (nu^-2) on 1.3 x span, a fixed band-noise block (absorbed in stage 1), a common
process on the array span, padding (K differs between pulsars), CURN and HD, power-law and
free-spectrum common processes. The reference is the full TOA-space covariance in long double.

| check | tolerance | result |
|---|---|---|
| value, 12 points (CURN/HD x power law / free spectrum) | 1e-9 max(1, abs lnL) | pass |
| gradient vs 4th-order FD of the long-double oracle (CURN, HD) | 1e-8 max(1, abs g) | pass |
| nearly collinear grids (IRN 19.1 yr, common 20.1 yr) | 1e-9 | pass |
| padding: pulsar alone vs padded in the array | 1e-10 value, 1e-11 gradient | pass |
| production vs fast kernels (`hh`/`levels`) | 1e-9 value, 1e-8 gradient | pass |
| `GeneralWhiteNoise` vs M1 `WhiteNoise` (disjoint ECORR) | N identical, 1e-10 | pass |

**Real multi-leg systems** (`scripts/m3a_validate.py g5`, `tests/test_m3a_multileg.py`).
J1022+1001 (5 legs) and J0437-4715 (3 legs): 45,647 TOAs, B and C. The setup is fixed white
noise per namespaced system, per-system ECORR plus a global ECORR over the PPTA leg (overlapping),
IRN on each pulsar's span, DM on 1.2 x span, and a common process on the array span. Comparisons:
* lnL differences between 6 points, ours vs enterprise (HD) and ours vs discovery (CURN);
* discovery's JAX gradients for every IRN/DM/common parameter.

discovery's ArrayLikelihood cannot combine variable per-pulsar GPs with a global GP, so discovery
checks CURN and enterprise checks HD.

<!-- G5_TABLE -->

## 6. G6 reference-model invariance (option C)

Exit bound, fixed before the run (resolves review minor #4):
* likelihood-shape difference <= **0.1 nats** over 20 random (IRN, DM, common) points;
* whitened norm of the linearisation residual (the difference of the residuals projected out of
  both column spaces) <= **0.1**;
* max principal-angle sine between the weighted column spaces <= **1e-3**.

J1909-3744 (5 legs) with reference NG15, PPTA or EPTA. CLOCK/EPHEM are held at TT(BIPM2019)/DE440
and the free-parameter sets are identical. NG15 and PPTA use the same parameterisation; EPTA is
equatorial, which spans the same space.

<!-- G6_TABLE -->

**Diagnosis.** The residual differences are linear to 0.03 sigma in total. The column spaces,
however, differ in **one** direction: SINI. Every other column agrees to < 1e-4. J1909 is nearly
edge-on (SINI = 0.998). The Shapiro derivative is proportional to 1/(1 - s sin Phi), which
changes shape with the nominal s, and the references' SINI values differ by 2.8e-4 (about
4 sigma). So option C's likelihood depends on the reference model through nonlinear Shapiro
parameters. The remedy belongs to M3b: re-linearise at a joint fit of all legs, i.e. iterate
the nominal model on the combined data, instead of copying one reference.

**Shared DM (YA-v3 / MetaPulsar 0.9.3) breaks phase connection for J1022+1001 and J0437-4715**
(the plan's R4/R5). The rewrite copies the reference DM into every leg and sets DM1 = DM2 = 0.
Each PTA's own DM offset (template/profile-evolution conventions) then becomes a nu^-2 delay of
hundreds of us that the single shared DM column cannot absorb leg by leg, and the residuals
wrap. Per-leg wrms after the rewrite:

| pulsar | config | NG15 | EPTA | PPTA | InPTA | MPTA |
|---|---|---|---|---|---|---|
| J1022+1001 | C shared DM (v0.9.3) | 34 us | 224 us | 1590 us | 3248 us | 776 us |
| J1022+1001 | C local DM | 34 us | 3.5 us | 8.4 us | 13 us | 4.6 us |
| J0437-4715 | C shared DM | - | - | 3.3 us | 1851 us | 440 us |
| J0437-4715 | C local DM | - | - | 0.39 us | 19 us | 0.29 us |

J1909-3744 stays phase-connected under shared DM: the largest leg is InPTA at 7.3 us, from DMX
removed at 400 MHz.

The likelihood code is still exact on the wrapped residuals: the C/G5 check agrees with the
oracles to 1e-12 relative of abs lnL ~ 5e10. But those residuals are not a valid linear model, so
the G5 table uses the phase-connected local-DM build of C. **The YA-v3 shared-DM
configuration is therefore not admissible for J1022 or J0437 without a nonlinear refit.** This
matters for the reproduction target (Sec. 9).

## 7. G7 duplicate observations

The check covers every pulsar with >= 2 legs: 61 in the selected configuration and 56 in the YA
set. A pair is the same physical site, overlapping observation intervals (|dt| < max(tobs)/2) and
frequencies within 1 MHz. LEAP (the coherent sum of the EPTA telescopes) is also checked against
every single-telescope EPTA TOA.

| set | multi-leg pulsars | cross-PTA same-site pairs | LEAP vs single telescope |
|---|---|---|---|
| selected | 61 | **0** | 1: EPTA J1600-3053, LEAP vs WSRT, dt = 1036 s, 1396 MHz (leg quarantined) |
| YA | 56 | **0** | same |

The five PTAs share no observatory, and DR2new does not contain InPTA DR1. The one LEAP/WSRT
coincidence is listed explicitly; its leg is quarantined.

## 8. G8 signal/null injections

The simulations use the real TOA sampling of J1909-3744 (5 legs), J1022+1001 (5 legs) and
J0437-4715 (3 legs), whose staggered spans give a 20.15-yr array. Each realisation is white
noise with ECORR, IRN (log10 A = -14.3, gamma = 3, pulsar span) and a CURN common process
(gamma = 13/3) at log10 A = -14 (signal) or 0 (null). They are recovered with:
* A: the leg with the most TOAs and its own timing model;
* B: per-leg timing;
* C: shared timing.

The statistic is the score s = dlnL/d(A^2) at the truth (null: at A = 1e-18). It needs
E[s] = 0 and Var(s) = I. R = 1000 realisations; pass when |z| < 3.5 and Var/I is within
1 +- 0.20.

<!-- G8_TABLE -->

## 9. G9 NG15 regression

The 67 NG15 pulsars go through the new path: `precompute_general` with the M1 white noise, the
IRN block on the array span and the common block on the same grid (prefix identity: K = 60,
common columns 0..27), then `GeneralPTALikelihood`. It is compared with `PTALikelihood` and
`FastPTALikelihood` at 8 points per ORF (4 chain samples, 4 prior draws).

| check | result |
|---|---|
| stage-1 terms (R_F, c, s_perp, log-determinants), all 67 pulsars | **bit-identical** |
| CURN and HD values, production and fast kernels (32 comparisons) | **32/32 bit-identical** |
| gradients | max relative difference 4.2e-14 (budget 1e-8) |
| HD free spectrum, 30 modes | <= 1e-9 / 1e-8 |
| one-leg `MultiLegPulsar` wrapper with namespaced systems | stage-1 contractions equal to 1e-10 |

## 10. Quarantine and exit conditions

**Quarantine** (`configs/m3/quarantine.json`). Quarantined legs are run and reported, and are
excluded from the validation-set pass requirement:
* **signed H3** (1): MPTA J1825-0319. There is no validated evaluator path. PINT rejects the leg
  (M2 < 0), and PINT PR #2023 is still open (checked 2026-10-08). tempo2 evaluates it, but tempo2
  is the oracle, and there is no second independent evaluator.
* **ELL1H H3+H4 without NHARMS** (10). G4 confirms it: MPTA J1327-0755 has sin = 0.96 (the
  harmonic sets differ) and G3 125 ns.
* **residual excess** (4): EPTA/PPTA J1600-3053, PPTA J1713+0747, PPTA J2241-5236. G3 now
  explains part of it. EPTA J1600 had 54 TOAs on the broken eff2gps.clk segment, now excluded.
  PPTA J2241 (PB+FB) passes G3 at 0.93 ns, but fails G4 at 3.5e-5.

**Mechanically decidable exit conditions** (replacing "all gates pass for the validation set";
review minor #4):

| # | condition | status |
|---|---|---|
| E1 | every loadable leg of the selected configuration and InPTA DR1 passes G1 and G2 | **met** (248/248) |
| E2 | G5: dense oracle and real-system oracles within tolerance | see Sec. 5 |
| E3 | G9 within the M1 budget | **met** (bit-identical) |
| E4 | B and C reproduce MetaPulsar v0.9.3 (residuals < 1e-3 ns, column space sin < 1e-6) | **met** |
| E5 | G7: no unlisted duplicate in a non-quarantined leg | **met** |
| E6 | G8 passes for A, B and C (signal and null) | see Sec. 8 |
| E7 | G6 within the numeric bound above for every reference swap of the validation pulsar | **not met** (Sec. 6) |
| E8 | G3/G4 at the fixed tolerances for every non-quarantined validation leg | **not met** (Sec. 4) |

A leg enters a likelihood that is *compared with published numbers* only if it meets E8. Today
that is no non-NG leg. Legs failing G3-G5 are not added to the quarantine wholesale: that would
make E8 pass vacuously. **M3a status: the infrastructure exit conditions E1-E6 are met; E7 and E8
are open**, with root causes identified (Secs. 4, 6).

## 11. Open issues

1. **G3/G4 engine parity (E8).** 1-5 ns residual differences remain for most legs, and ~30 ns
   for PPTA DDK/DDH legs. Next steps: DDK/DDH evaluator parity tests; a decision on whether a
   likelihood-impact tolerance should replace the 1 ns criterion (needs reviewer agreement); and
   tempo2-evaluated residuals and design matrices for non-NG legs as YA did (our container
   accepts any per-leg arrays).
2. **Option C nonlinearity (E7) and shared DM.** C needs a re-linearised nominal model: a joint
   fit before computing residuals, iterated. The v0.9.3 shared-DM rule wraps J1022 and J0437
   (Sec. 6), so YA's pipeline must have done something not captured by "copy the reference"
   (YA EM:3 mentions pulse numbers and per-PTA JUMPs). This is still an open question for the
   reproduction.
3. **Clock files.** EPTA eff2gps.clk has no corrections after MJD 57195.5, and effix2gps ends
   at 59294.5. Affected TOAs are excluded (65); this must be resolved with EPTA's own clock files.
4. **Signed H3.** Retest when PINT #2023 merges.
5. G5's discovery check covers CURN only; HD gradients are covered by the dense oracle.
