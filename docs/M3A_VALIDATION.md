# M3a validation: multi-leg container, physics-preserving ingestion, general likelihood

M3a is the first M3 implementation milestone (`docs/M3_PLAN.md` Sec. 5.1). It is CPU-only
deterministic infrastructure plus validation; there is no sampling and no evidence work. This
document gives every gate result with its numbers, its tolerance and whether it passed.

**Revision 2 (after independent review of 956661a, REQUEST_CHANGES).** The claim "E1-E6 met" of
revision 1 is withdrawn and re-established below gate by gate. Changes:
1. Option C kept PINT's forced NHARMS = 7 through the shared-model rewrite. It now carries the
   reference's harmonic count, which is tested. Quarantine status depends on the configuration.
2. The G7 duplicate search depended on leg order. It is now symmetric, with the criterion
   |dt| < (d_a + d_b)/2 and an order-invariance test. It found 7 same-channel duplicates, all now
   removed explicitly under EPTA's stated rule (Sec. 7).
3. Phase-invalid shared-DM C builds are rejected unless a diagnostic override is given. The
   configuration and admissibility are serialised with every build, and local-DM C is labelled
   as not the YA target.
4. Singleton ECORR (nmin = 1) was dropped from solve/logdet. Fixed, with a test against the
   dense operator.
5. The union projector used unpivoted QR. It is now a rank-revealing SVD with a declared
   threshold (rtol 1e-10), effective ranks are reported, and the union diagnostics were
   regenerated.
6. G5 now has an independent long-double arbiter, constant-invariant tolerances, and held-out
   points under criteria written into the code before they were run.
7. NHARMS diagnostics: the producer is committed, conventions are stated, surfaces are saved,
   and interpretations are corrected. 8. GMRT is described as a release-reproduction convention;
   both coordinate sets are kept. 9. New: posterior-level NHARMS shifts (Sec. 12b).

**Post-hoc elements** (not fixed before the result was seen): the G3/G4 "likelihood impact"
column; revision 1's G5 arbitration rule, which is replaced (Sec. 5); and the choice in G7 to
remove same-channel duplicates by EPTA's rule (Sec. 7). Every other tolerance was fixed in code
before its result was seen.

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
PTAGWB_REQUIRE_ORACLES=1 $PY -m pytest tests/ --deselect tests/test_setup.py::test_jax_sees_gpu   # strict suite, CPU
XLA_PYTHON_CLIENT_PREALLOCATE=false PTAGWB_REQUIRE_ORACLES=1 $PY -m pytest tests/test_setup.py::test_jax_sees_gpu
$PY scripts/m3a_nharms.py; $PY scripts/m3a_report.py   # Sec. 12 addendum; regenerate the tables below
```

**Strict suite** (`PTAGWB_REQUIRE_ORACLES=1`, revision 2, 2026-10-08, CPU, plus the one GPU
check run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`):
* 422 passed + 1 GPU check passed, 2 xfailed, 0 skipped.
* In the full run, `test_g6_linearisation` failed against its revision-1 bound; that bound came
  from the faulty QR projector. The test was updated to the rank-revealing numbers, and its rerun
  passed together with the GPU check.
* The 2 xfails are the open gates, kept as `xfail(strict=True)`: G6/E7
  (`test_m3a_multileg.py::test_g6_reference_swap`) and the strict G3/G4 tolerances/E8
  (`test_m3a_tempo2_parity.py::test_strict_g3_g4`).
* 59 of the tests are M3a tests (`tests/test_m3a_*.py`). In strict mode an xfail does not count
  as a skip (`tests/conftest.py`).
* The worktree needs `runs/` linked to the main checkout for the free-spectrum-gate tests merged
  from main.

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
* **Site coordinates: a release-reproduction convention.** PINT's GMRT position differs from
  tempo2's by **749.8 m**; Effelsberg's by 2.8 m and Jodrell's by 0.5 m. This was found by G3:
  InPTA residuals differed by 830 ns rms. The released tempo2 timing models were fitted with
  tempo2's coordinates, so those legs are ingested with tempo2's coordinates to reproduce the
  releases; NG15 keeps PINT's. This shows reproducibility, not physical correctness. PINT's
  GMRT coordinates have explicit provenance: they were supplied by the InPTA team on 2026-06-05
  (PINT observatories.json). Both sets are kept in `configs/m3/sites/tempo2-2026.04.1.json`.
  Before any physical refit, the reference position must be set from release or pipeline
  provenance.
* **Clock coverage.** Some TOAs have no clock correction good to 1 ns: EPTA effix2gps ends at
  MJD 59294.5, and eff2gps.clk has no data after 57195.5 apart from a sentinel 0 at 60000
  ("No maser file found"). PINT clamps there while tempo2 interpolates to the sentinel. These
  TOAs are excluded explicitly (policy `exclude-uncovered`, default; `keep` is available):
  **65 TOAs in 7 EPTA legs** (J1600-3053: 54; J1640+2224: 3; the others 1-2 each).

## 2. Gate summary

| Gate | Scope | Oracle | Tolerance | Result (revision 2) | Pass |
|---|---|---|---|---|---|
| G1 TOA identity | all 249 legs (selected configuration + InPTA DR1) | tempo2-semantics records; libstempo on fixtures and real legs | count exact (after the listed explicit removals), dt < 2 ns, err/freq/observatory/every flag/-padd identical | 248/248 loadable legs, max dt 0.81 ns, 0 mismatches | **PASS** |
| G2 warnings | all loadable legs | classification + per-leg audits | none unexplained | 248/248 | **PASS** |
| G3 projected residuals | validation set | tempo2 2026.04.1 | rms < 1 ns and < 0.01 sigma | 1/17 non-quarantined legs (NG15 J1909, 0.97 ns); others 1-5 ns, PPTA J1022/J0437 ~30 ns | **FAIL** |
| G4 column space | validation set; B/C containers | tempo2; MetaPulsar v0.9.3 | sin < 1e-6, equal dimension | tempo2: 0/17 (sin 1e-5 to 7e-3); MetaPulsar: residuals identical, sin < 4e-8 | **FAIL** vs tempo2; **PASS** vs MetaPulsar |
| G5 likelihood + gradient | dense oracle (synthetic); real systems B (J1022 B + J0437 B) and C (J1909 C YA-v3 + J0437 B), original + held-out points | long-double dense; long-double arbiter; enterprise; discovery | values: shape diff <= 1e-6 nats vs every oracle; gradients <= 1e-8 rel. vs arbiter (fixed before the held-out run) | ours vs arbiter <= 8e-9 nats, <= 4e-11 grad everywhere; B: oracles within 3e-8 nats; C (87k TOAs): discovery 8e-6 and enterprise 7.6e-5 nats from ours, and as far from the arbiter | B **PASS**; C **FAIL** under the pre-fixed all-oracle criterion (float64 oracles, not ours; ours passes vs the arbiter) |
| G6 reference invariance | C, J1909-3744, NG15 -> PPTA / EPTA | ours | shape <= 0.1 nats; linearisation <= 0.1 (rank-revealing, rtol 1e-10); sin <= 1e-3 | PPTA 0.195 / 0.140 / 0.034; EPTA 0.028 / 0.134 / 0.011 | **FAIL** |
| G7 duplicates | 61 (YA 56) multi-leg pulsars + all legs within | symmetric TOA matching | none, or explicitly removed | 0 cross-PTA; 7 within-leg same-channel pairs, all removed by EPTA's stated rule | **PASS** (revision 1's PASS was premature) |
| G8 injections | J1909-3744 (the only validation pulsar with an admissible C build), A/B/C | ours | z < 3.5; Var/I within 1 +- 0.20 | 6/6; I_C/I_B = 1.003, I_C/I_A = 1.48 | **PASS** |
| G9 NG15 regression | 67 NG15 pulsars | M1/M2 | bit-identical or <= 1e-9 / 1e-8 | 32/32 bit-identical (rerun in the strict suite) | **PASS** |
| C admissibility (new) | C builds of J1909, J1022, J0437 | own model, same clock/ephemeris | linearisation rms <= 0.1 (whitened) | J1909 YA-v3 and local-DM admissible (<= 9e-4); J1022, J0437 inadmissible in both | reported |

## 3. G1 and G2 (all legs)

| data set | legs | loaded | G1 pass | TOAs (PINT) | max dt [ns] | flag/obs/padd mismatches | G2 unexplained | clock-excluded TOAs | duplicate TOAs removed (G7) |
|---|---|---|---|---|---|---|---|---|---|
| ng15 | 68 | 68 | 68 | 676,395 | 0.807 | 0 | 0 | 0 | 2 |
| epta_dr2new | 25 | 25 | 25 | 45,426 | 0.806 | 0 | 0 | 65 | 2 |
| ppta_dr3_gh | 32 | 32 | 32 | 113,948 | 0.614 | 0 | 0 | 0 | 3 |
| inpta_dr2 | 27 | 27 | 27 | 83,120 | 0.614 | 0 | 0 | 0 | 0 |
| mpta | 83 | 82 | 82 | 242,863 | 0.802 | 0 | 0 | 0 | 0 |
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
| clock-coverage-audited | 13 | coverage audit passed after the explicit exclusions above |
| ddk-kin / ddk-a1dot | 10 / 1 | DDK uses KIN as tempo2 does |
| dmx-overlap | 4 | DMX ranges overlap as released |
| pint-parse-overflow | 14 | over-long numeric literal; values checked by G3 |
| mask-no-toas | 2 | parameter without TOAs (frozen, as tempo2 refuses to fit it) |
| ddgr-default-init | 1 | MPTA J0955-6150: PINT's DDGR set-up divides by zero with default parameters before the par values are set (located with numpy `seterr(divide='raise')`) |
| altitude, tzr-default, pulse-numbers-partial, resource | 1-248 | benign |

Free mask parameters that select no TOA are frozen, recorded per leg. That covers 31 PPTA legs
(old-backend JUMPs absent from the pulsar), plus NG15 J1024-0719 F3 (M1's checked exception).

## 4. G3 / G4 against tempo2 (validation set)

<!-- G3G4_TABLE -->
| leg | role | q | TOAs | proj. rms [ns] | proj. rms [sigma] | max proj. [ns] | G3 | cols PINT/tempo2 | G4 max sin | G4 | post-hoc dlnL shape [nats] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpta/J1825-0319 | signed H3 (DDH, negative H3 + STIG) | Q | - | - | - | - | FAIL | -/- | - | FAIL | - | no PINT leg (ingestion failed)
| mpta/J1802-2124 | ELL1H H3+STIG (absorbed Shapiro) |  | 3039 | 2.64 | 0.0012 | 7.99 | FAIL | 19/19 | 0.00067 | FAIL | 0.44 |
| mpta/J1327-0755 | ELL1H H3+H4, no NHARMS |  | 788 | 1.61 | 0.0011 | 5.01 | FAIL | 17/17 | 0.00044 | FAIL | 0.054 |
| ppta_dr3_gh/J2241-5236 | PB + FB series | Q | 6238 | 0.927 | 0.0041 | 3.05 | pass | 46/46 | 3.5e-05 | FAIL | - |
| ppta_dr3_gh/J1600-3053 | DDH | Q | 5146 | 5.27 | 0.0057 | 25.1 | FAIL | 48/48 | 0.00027 | FAIL | 1 |
| ppta_dr3_gh/J1713+0747 | DDK; multi-valued -j mask flags | Q | 5140 | 4.26 | 0.017 | 20 | FAIL | 48/48 | 0.00017 | FAIL | 1.2 |
| epta_dr2new/J1744-1134 | TIME offsets, END, -padd |  | 1541 | 1.2 | 0.0032 | 4.18 | FAIL | 26/26 | 1e-05 | FAIL | 0.35 |
| inpta_dr2/J1614-2230 | InPTA DR2: valueless flags, DMX_0001 template, INCLUDE |  | 123 | 5.53 | 0.00023 | 11.8 | FAIL | 4/4 | 0.00017 | FAIL | 0.0033 |
| ng15/J1909-3744 | ordinary NG15; multi-leg J1909 |  | 35037 | 0.971 | 0.0051 | 4.82 | pass | 345/345 | 0.00091 | FAIL | 1.4 |
| ng15/J1022+1001 | ordinary NG15 (short leg); multi-leg J1022 |  | 3978 | 3.39 | 0.0038 | 39.8 | FAIL | 76/76 | 0.0038 | FAIL | 0.61 |
| epta_dr2new/J1909-3744 | ordinary EPTA (INCLUDE tree); multi-leg J1909 |  | 2289 | 1.12 | 0.0086 | 4.01 | FAIL | 33/33 | 0.00023 | FAIL | 0.43 |
| epta_dr2new/J1022+1001 | ordinary EPTA; multi-leg J1022 |  | 1802 | 6.62e+03 | 3.6 | 6.41e+04 | FAIL | 35/35 | 1 | FAIL | 1.2e+03 |
| ppta_dr3_gh/J1909-3744 | ordinary PPTA (-j flags); multi-leg J1909 |  | 9644 | 5.67 | 0.045 | 30.4 | FAIL | 50/50 | 0.00026 | FAIL | 34 |
| ppta_dr3_gh/J1022+1001 | ordinary PPTA; multi-leg J1022 |  | 5242 | 30.8 | 0.013 | 1.02e+03 | FAIL | 46/46 | 0.0062 | FAIL | 1.9 |
| ppta_dr3_gh/J0437-4715 | PPTA DDK, large leg; multi-leg J0437 |  | 11637 | 32.2 | 0.21 | 134 | FAIL | 45/45 | 0.00024 | FAIL | 58 |
| inpta_dr2/J1909-3744 | ordinary InPTA; multi-leg J1909 |  | 4160 | 4.3 | 0.0032 | 20.5 | FAIL | 7/7 | 0.00011 | FAIL | 0.62 |
| inpta_dr2/J1022+1001 | ordinary InPTA; multi-leg J1022 |  | 3346 | 2.08 | 0.00045 | 8.63 | FAIL | 9/9 | 0.003 | FAIL | 2.1 |
| inpta_dr2/J0437-4715 | InPTA; multi-leg J0437 |  | 13283 | 1.93 | 0.0021 | 7.1 | FAIL | 7/7 | 4.5e-05 | FAIL | 2.6 |
| mpta/J1909-3744 | ordinary MPTA; multi-leg J1909 |  | 7199 | 2.51 | 0.027 | 14 | FAIL | 20/20 | 0.00023 | FAIL | 3 |
| mpta/J1022+1001 | ordinary MPTA; multi-leg J1022 |  | 2945 | 2.61 | 0.0026 | 9.79 | FAIL | 19/19 | 0.0066 | FAIL | 1.3 |
| mpta/J0437-4715 | MPTA; multi-leg J0437 |  | 3517 | 1.66 | 0.039 | 4.69 | FAIL | 20/20 | 0.00026 | FAIL | 4.8 |

Non-quarantined validation legs: G3 pass 1/17, G4 pass 0/17.
<!-- /G3G4_TABLE -->

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
post-hoc likelihood-impact column quantifies the consequence. It evaluates one single-pulsar model
(white noise from the raw errors; IRN and DM GP with 30 bins on the leg's span) on PINT's and on
tempo2's residuals and design matrix, and takes the maximum change of the lnL shape over 12 random
(IRN, DM) points:
* most legs: 0.003-3 nats;
* high-precision PPTA legs: **34 nats for J1909-3744** (0.045 sigma rms) and **58 nats for
  J0437-4715** (0.21 sigma, DDK).

These numbers come from varying **RN and DM only, with no GW process**, relative to the first
random point; for J0437 the peak-to-peak range is 83.9 nats. The independent review decomposed these numbers by holding PINT's
design matrix fixed: almost all of the change comes from the residuals (34.03 / 58.20 nats
residual-only, 0.046 / 0.019 nats matrix-only). They are diagnostics under these noise assumptions, not demonstrated GW
posterior costs. They still show that the engine differences are not negligible for those legs,
which is why E8 stays a hard condition.

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
| value, 12 points (CURN/HD x power law / free spectrum) | 1e-9 max(1, abs lnL) (absolute values with the same constant convention; synthetic, abs lnL ~ 1e3) | pass |
| gradient vs 4th-order FD of the long-double oracle (CURN, HD) | 1e-8 max(1, abs g) | pass |
| nearly collinear grids (IRN 19.1 yr, common 20.1 yr) | 1e-9 | pass |
| padding: pulsar alone vs padded in the array | 1e-10 value, 1e-11 gradient | pass |
| production vs fast kernels (`hh`/`levels`) | 1e-9 value, 1e-8 gradient | pass |
| `GeneralWhiteNoise` vs M1 `WhiteNoise` (disjoint ECORR) | N identical, 1e-10 | pass |
| `GeneralWhiteNoise` with nmin = 1 (single-TOA ECORR epochs, overlapping terms) vs dense: solve, whiten, logdet, colour | 1e-10 | pass (failed before revision 2: singleton ECORR was dropped) |

**Real multi-leg systems** (`scripts/m3a_validate.py g5`, `tests/test_m3a_multileg.py`).
J1022+1001 (5 legs) and J0437-4715 (3 legs): 45,647 TOAs, B and C. The setup is fixed white
noise per namespaced system, per-system ECORR plus a global ECORR over the PPTA leg (overlapping),
IRN on each pulsar's span, DM on 1.2 x span, and a common process on the array span. Comparisons:
* lnL shape differences, ours vs enterprise (HD), discovery (CURN) and the independent arbiter
  (CURN);
* gradients vs the arbiter (gating) and vs discovery (reported only).

discovery's ArrayLikelihood cannot combine variable per-pulsar GPs with a global GP, so discovery
checks CURN and enterprise checks HD. C is the admissible local-DM build (Sec. 6).

**Independent arbiter** (`tests/m3a_arbiter.py`). It shares no numerics with `ptagwb.combined`:
* its own Fourier columns;
* N^-1 applied with `GeneralWhiteNoise.solve` (Cholesky solves) rather than whitening;
* timing marginalisation by the normal-equation projector, with the Gram matrices accumulated in
  long double;
* the reduced covariance Sigma = Phi^-1 + A factorised by a long-double Cholesky;
* the gradient from the analytic trace derivative
  d lnL / d phi_k = 1/2 phi_k^-2 (x_k^2 + [Sigma^-1]_kk) - 1/2 phi_k^-1.

**Criteria, fixed in code (`cmd_g5` docstring) before the held-out points were run**, and
constant-invariant:
* lnL shape differences |(lnL_i - lnL_0)_ours - (...)_oracle| <= 1e-6 nats for every oracle;
* gradients |g_ours - g_arbiter| <= 1e-8 max(1, |g_arbiter|) for every component.
* Points: the 6 original points (seed 11) and 6 held-out points (seed 2026); the strict test
  uses 3 more (seed 99).

Revision 1 instead scaled the shape tolerance by abs lnL and arbitrated with our own identity-ORF
path; both are withdrawn. Reference value for the disputed point (C, seed 11, point 0,
d lnL / d log10_A): production 0.156732014257, the reviewer's independent long-double calculation
0.156732013965, discovery 0.156732041905.

<!-- G5_TABLE -->
| system | points | CURN ours-arbiter | HD ours-arbiter | grad ours-arbiter (rel.) | CURN ours-discovery | HD ours-enterprise | discovery-arbiter | enterprise-arbiter (HD) | grad ours-discovery (info) | pre-fixed criterion (all oracles) | vs arbiter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C: J1909-3744 ; J0437-4715  (86766 TOAs) | original | 7.5e-09 | 7.6e-09 | 1.6e-11 | 8.4e-06 | 7.6e-05 | 8.4e-06 | 7.6e-05 | 7.1e-09 | FAIL | pass |
| C: J1909-3744 ; J0437-4715  (86766 TOAs) | held_out | 3.9e-09 | 4.2e-09 | 4e-11 | 3.5e-06 | 3.2e-05 | 3.5e-06 | 3.2e-05 | 4.3e-09 | FAIL | pass |
| B: J1022+1001 ; J0437-4715  (45750 TOAs) | original | 3.5e-10 | 2.9e-10 | 2e-11 | 2.1e-08 | 2.6e-08 | 2.1e-08 | 2.7e-08 | 3.5e-10 | pass | pass |
| B: J1022+1001 ; J0437-4715  (45750 TOAs) | held_out | 3.7e-10 | 3.5e-10 | 4e-11 | 6.1e-09 | 1.4e-08 | 5.7e-09 | 1.5e-08 | 2.3e-10 | pass | pass |
<!-- /G5_TABLE -->

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
| swap | linearisation (whitened norm) | column-space max sin | max shape diff [nats] | pass |
|---|---|---|---|---|
| NG15 -> PPTA | 0.14 | 0.0341 | 0.195 | FAIL |
| NG15 -> EPTA | 0.134 | 0.0113 | 0.0276 | FAIL |
<!-- /G6_TABLE -->

**Diagnosis.** With the declared rank-revealing projector (rtol 1e-10, union rank 81 / 80 of 144
columns), the linearisation norm is 0.140 (PPTA) and 0.134 (EPTA). That exceeds the 0.1 bound.
The norm depends on the threshold, because the union has near-dependent directions with
singular values of 1e-11 to 1e-12: at rtol 1e-12 (rank 84) it is 0.028. Revision 1's unpivoted QR
gave 0.028 by keeping arbitrary completion vectors. The residual difference sits along nearly
degenerate directions such as SINI(s1) - SINI(s2), which is itself the nonlinearity. The column spaces,
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

**Admissibility, enforced since revision 2** (`multileg.admissibility`; threshold fixed before
the run). Every C leg is compared with the leg's own published-profile model, TOA by TOA:
e = P_perp[W M_own, W M_C] W (r_C - r_own), with a rank-revealing projector. For a linear,
phase-connected re-parameterisation, e vanishes up to clock-profile differences. A leg is
admissible iff rms(e) <= 0.1 (whitened). The table below is generated from `multileg.json`:

<!-- ADMISSIBILITY_TABLE -->
| build | configuration | leg | TOAs matched | union rank / cols | linearisation rms (whitened) | rms [ns] | admissible |
|---|---|---|---|---|---|---|---|
| J1909-3744_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | NG15 | 35037 | 348/368 | 0.000132 | 0.0422 | yes |
| J1909-3744_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | EPTA | 2289 | 41/66 | 0.000319 | 0.0637 | yes |
| J1909-3744_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | PPTA | 9644 | 59/100 | 0.000823 | 0.174 | yes |
| J1909-3744_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | InPTA | 4160 | 174/182 | 3.09e-05 | 0.0593 | yes |
| J1909-3744_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | MPTA | 7199 | 28/42 | 0.000518 | 0.0553 | yes |
| J1022+1001_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | NG15 | 3978 | 78/100 | 4.63e-06 | 0.00669 | yes |
| J1022+1001_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | EPTA | 1802 | 44/66 | 0.00205 | 1.85 | yes |
| J1022+1001_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | PPTA | 5242 | 55/91 | 0.485 | 687 | NO |
| J1022+1001_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | InPTA | 3346 | 140/144 | 902 | 3.73e+06 | NO |
| J1022+1001_C-refNG15 | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | MPTA | 2945 | 29/38 | 0.000263 | 0.423 | yes |
| J0437-4715_C-refPPTA | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | PPTA | 11637 | 45/90 | 0.000803 | 0.0983 | yes |
| J0437-4715_C-refPPTA | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | InPTA | 13283 | 129/134 | 2.67e+03 | 1.7e+06 | NO |
| J0437-4715_C-refPPTA | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | MPTA | 3517 | 28/43 | 0.173 | 7.21 | NO |
| J1909-3744_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | NG15 | 35037 | 346/365 | 0.000135 | 0.0427 | yes |
| J1909-3744_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | EPTA | 2289 | 41/66 | 0.000102 | 0.0163 | yes |
| J1909-3744_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | PPTA | 9644 | 59/100 | 0.000214 | 0.0213 | yes |
| J1909-3744_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | InPTA | 4160 | 173/179 | 2.97e-05 | 0.0589 | yes |
| J1909-3744_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | MPTA | 7199 | 28/42 | 0.00017 | 0.0146 | yes |
| J1022+1001_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | NG15 | 3978 | 76/97 | 1.65e-05 | 0.015 | yes |
| J1022+1001_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | EPTA | 1802 | 43/66 | 0.00181 | 1.6 | yes |
| J1022+1001_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | PPTA | 5242 | 54/91 | 0.486 | 685 | NO |
| J1022+1001_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | InPTA | 3346 | 137/141 | 0.0335 | 279 | yes |
| J1022+1001_C-refNG15-localDM | C/local-DM (MetaPulsar-main exclude_from | MPTA | 2945 | 27/37 | 0.000332 | 0.481 | yes |
| J0437-4715_C-refPPTA-localDM | C/local-DM (MetaPulsar-main exclude_from | PPTA | 11637 | 45/90 | 0 | 0 | yes |
| J0437-4715_C-refPPTA-localDM | C/local-DM (MetaPulsar-main exclude_from | InPTA | 13283 | 126/131 | 0.00567 | 8.31 | yes |
| J0437-4715_C-refPPTA-localDM | C/local-DM (MetaPulsar-main exclude_from | MPTA | 3517 | 28/43 | 0.17 | 7.08 | NO |
| J1909-3744_C-refPPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | NG15 | 35037 | 355/368 | 0.00015 | 0.0443 | yes |
| J1909-3744_C-refPPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | EPTA | 2289 | 42/66 | 0.000248 | 0.0475 | yes |
| J1909-3744_C-refPPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | PPTA | 9644 | 51/100 | 0.000866 | 0.204 | yes |
| J1909-3744_C-refPPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | InPTA | 4160 | 175/182 | 4.45e-05 | 0.18 | yes |
| J1909-3744_C-refPPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | MPTA | 7199 | 28/42 | 0.000902 | 0.0934 | yes |
| J1909-3744_C-refEPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | NG15 | 35037 | 355/368 | 0.000149 | 0.0465 | yes |
| J1909-3744_C-refEPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | EPTA | 2289 | 34/66 | 0.000209 | 0.0436 | yes |
| J1909-3744_C-refEPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | PPTA | 9644 | 59/100 | 0.000888 | 0.204 | yes |
| J1909-3744_C-refEPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | InPTA | 4160 | 173/182 | 4.43e-05 | 0.0901 | yes |
| J1909-3744_C-refEPTA-forced | C/YA-v3 (MetaPulsar-0.9.3 shared DM): th | MPTA | 7199 | 28/42 | 0.000872 | 0.0915 | yes |
<!-- /ADMISSIBILITY_TABLE -->

`build_multileg(timing="shared")` raises `InadmissibleBuildError` for an inadmissible build unless
`allow_inadmissible=True` is passed (diagnostics only). Every build records its configuration in
`meta["config"]`:
* "C/YA-v3 (MetaPulsar-0.9.3 shared DM): the YA reproduction target";
* "C/local-DM (...): NOT the YA target";
* "B/per-leg".

Each build also records its admissibility and the per-leg provenance (profile, evaluator,
harmonic count, consistent-rewrite info) in `LegInfo.meta`, and all of it survives
`save_multileg`/`load_multileg`. `precompute_general` refuses an inadmissible build unless
explicitly overridden. Matching MetaPulsar's own (equally wrapped) residuals does not make the
shared-DM build valid.

**Result.**
* **J1909-3744**: the YA-v3 build is admissible (every leg <= 9e-4), and so is the local-DM build.
* **J1022+1001 YA-v3** is inadmissible. Its InPTA leg wraps (902 sigma); its PPTA leg is off by
  0.49 sigma, because PPTA's DDH model with frozen H3/STIG is replaced by NG15's DD model without
  Shapiro.
* **J0437-4715 YA-v3** is inadmissible. Its InPTA leg wraps (2670 sigma); its MPTA leg is off by
  0.17 sigma (DDK with frozen KIN/KOM vs PPTA's free values).
* **Local DM** removes the wraps but not the binary-model mismatches, so J1022 (PPTA 0.49) and
  J0437 (MPTA 0.17) stay inadmissible there too.

The union test checks linearity and pulse connection. A per-leg DM offset that C's shared DM
cannot represent (the 0.2-3 ms wrms of the shared-DM legs) lies inside the union span, so it is
a model-adequacy problem that this test does not catch; it is reported separately by the wrms
table above. G5 and G8 now use only admissible builds.

## 7. G7 duplicate observations

**Criterion** (`multileg.find_duplicates`; revision 2). A pair must be:
* at the same physical site, or one of them LEAP (the coherent sum of the EPTA telescopes);
* close enough in time that the observation intervals centred on the two TOAs overlap,
  |t_a - t_b| < (d_a + d_b)/2, with d the `-tobs` duration;
* in the same channel, |f_a - f_b| < 1 MHz.

Such a pair means the same photons are counted twice under independent white noise. The
candidate window is (d_a + max_b d_b)/2, so the result does not depend on leg order (tested,
including the reviewer's EPTA J1022 pair in both orders). Revision 1 enumerated with the first
observation's duration only and missed that pair.

**Scope.** Two searches:
* across PTAs, for every pulsar with >= 2 legs (61 in the selected configuration, 56 in the YA
  set);
* within every leg, across observing systems: LEAP vs single telescopes, legacy vs new backends,
  and so on.

| set | multi-leg pulsars | cross-PTA pairs | within-leg same-channel pairs | removed explicitly | unresolved |
|---|---|---|---|---|---|
| selected | 61 | **0** | 7 | 7 | **0** |
| YA | 56 | **0** | 7 | 7 | **0** |

**Dispositions** (`configs/m3/duplicates.json`; applied in M3 leg ingestion and recorded in each
leg's provenance; the NG15 M1/M2 path is unchanged). The EPTA DR2 data paper (arXiv:2306.16224,
Sec. "Combination of the dataset") states: *"During certain observing runs, data were collected
using both legacy and new backends, or in both single-telescope and LEAP modes. As these
observations represented the same signal and noise, we eliminated the older backend and non-LEAP
data."* The same rule is applied to the 7 remnants: keep LEAP over a single telescope, the newer
backend over the older, and the longer copy of two on one backend.
* **EPTA J1022+1001** (the reviewer's case). The EFF/Asterix 150 s scan (EFF.P217.1380.tim:180,
  MJD 58810.27723) sits inside the 3720 s LEAP observation (LEAP.1396.tim:80, MJD 58810.25645),
  1796 s apart at 1404 MHz. Effelsberg is LEAP's geometric and time reference telescope (paper,
  LEAP section). Whether its baseband entered *this* combination cannot be read from the
  released TOA: the LEAP TOA lists `-fe unknown`. EPTA's rule is applied conservatively and the
  EFF TOA is removed.
* **EPTA J1600-3053**: WSRT/PuMa2 3770 s vs LEAP 1677 s, 1036 s apart; the WSRT TOA is removed.
* **PPTA J1045-4509**: CASPSR 857 s inside PDFB3 3840 s (40 cm, 709.9 vs 710.3 MHz); the CASPSR
  TOA is removed.
* **PPTA J1603-7202**: a 151 s CPSR2m copy (group 50CM_CPSR2 at 1333 MHz) inside the 1548 s
  observation, 2 channels; the copy is removed.
* **NG15 J1713+0747**: GASP 120 s inside GUPPI 1217 s, 2 channels; GASP is removed.

**Diagnostic, not gating: simultaneous recordings in overlapping bands by different backends.**
These are the same photons in different channels, so they are correlated rather than identical
TOAs. Pairs with |f_a - f_b| < (bw_a + bw_b)/2 across *backends*:
* PPTA 16,186: Medusa/UWL vs PDFB4 (20 pulsars) and CASPSR vs Medusa (13);
* EPTA 235: LEAP vs JBO/NRT/EFF in partly overlapping bands; EPTA kept data outside the LEAP band
  on purpose;
* NG15 19: the GASP/GUPPI overlap.

The PTAs released and analysed these data; PPTA models a global ECORR across UWL systems. This is
an open M3b modelling item (correlated cross-backend noise, or pruning) and does not block G7 as
defined ("the same observation in two legs").

## 8. G8 signal/null injections

The simulations use the real TOA sampling of J1909-3744 (5 legs, staggered spans 2004-2023). It is
the only validation pulsar with an admissible option-C build (Sec. 6); revision 1 also used the
inadmissible J1022 and J0437 C builds, which are no longer used. Each realisation is white
noise with ECORR, IRN (log10 A = -14.3, gamma = 3, pulsar span) and a CURN common process
(gamma = 13/3) at log10 A = -14 (signal) or 0 (null). They are recovered with:
* A: the leg with the most TOAs and its own timing model;
* B: per-leg timing;
* C: shared timing.

The statistic is the score s = dlnL/d(A^2) at the truth (null: at A = 1e-18). It needs
E[s] = 0 and Var(s) = I. R = 1000 realisations; pass when |z| < 3.5 and Var/I is within
1 +- 0.20.

<!-- G8_TABLE -->
| case | z(mean score) | Var(s)/I | tolerance | I(A^2) | pass |
|---|---|---|---|---|---|
| signal/C | 0.11 | 0.986 | 1 +- 0.20 | 9.01e+56 | pass |
| signal/B | 0.11 | 0.986 | 1 +- 0.20 | 8.98e+56 | pass |
| signal/A | 1.14 | 1.018 | 1 +- 0.20 | 6.08e+56 | pass |
| null/C | 0.64 | 1.054 | 1 +- 0.20 | 4.88e+60 | pass |
| null/B | 0.40 | 1.063 | 1 +- 0.20 | 4.6e+60 | pass |
| null/A | 0.35 | 1.018 | 1 +- 0.20 | 1.96e+60 | pass |

Information ratios at the injected amplitude: I_C/I_B = 1.003, I_C/I_A = 1.483; at the null: I_C/I_B = 1.061, I_C/I_A = 2.489.
<!-- /G8_TABLE -->

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
* **ELL1H H3+H4 without NHARMS** (10): **configuration-specific** (`m3data.quarantine(config)`).
  They are lifted under `{"ell1h_nharms": "tempo2"}` (the default, applied on every construction
  path including the option-C rewrite) and quarantined under `"pint7"`. Under PINT's 7
  harmonics, MPTA J1327-0755 has G4 sin = 0.96 and G3 125 ns; with 4 harmonics, 4.4e-4 and
  1.6 ns. At the posterior level the difference is <= 0.1 sigma (Sec. 12b).
* **residual excess** (4): EPTA/PPTA J1600-3053, PPTA J1713+0747, PPTA J2241-5236. G3 now
  explains part of it. EPTA J1600 had 54 TOAs on the broken eff2gps.clk segment, now excluded.
  PPTA J2241 (PB+FB) passes G3 at 0.93 ns, but fails G4 at 3.5e-5.

**Mechanically decidable exit conditions** (replacing "all gates pass for the validation set";
review minor #4). Revision 1's "E1-E6 met" is withdrawn; status after the revision-2 fixes:

| # | condition | status |
|---|---|---|
| E1 | every loadable leg of the selected configuration and InPTA DR1 passes G1 and G2 | **met** (248/248, re-ingested after the duplicate removals) |
| E2 | G5 within the pre-fixed criteria (shape <= 1e-6 nats vs every oracle; gradients <= 1e-8 vs the arbiter) | **met for B; not met for C**: the float64 oracles deviate by up to 7.6e-5 nats on the 87k-TOA C system, while ours agrees with the long-double arbiter to 8e-9 |
| E3 | G9 within the M1 budget | **met** (bit-identical) |
| E4 | B and C reproduce MetaPulsar v0.9.3 (residuals < 1e-3 ns, column space sin < 1e-6) | **met** (YA-v3 C builds compared as marked diagnostics where inadmissible) |
| E5 | G7: no unlisted duplicate in a non-quarantined leg | **met** (7 found, 7 explicitly removed) |
| E6 | G8 passes for A, B and C (signal and null), on admissible builds | **met** (6/6, J1909) |
| E7 | G6 within the numeric bound for every reference swap of the validation pulsar | **not met** |
| E8 | G3/G4 at the fixed tolerances for every non-quarantined validation leg | **not met** |

A leg enters a likelihood that is *compared with published numbers* only if it meets E8. Today
that is no non-NG leg. Legs failing G3-G5 are not quarantined wholesale, since that would make E8
pass vacuously. **M3a status: E1 and E3-E6 met. E2 is met for B but formally not met for C,
because of the oracles' own float64 precision. E7 and E8 are open, with root causes identified.**
Option-C builds of J1022 and J0437 are inadmissible in both DM configurations.

## 11. Open issues

1. **G3/G4 engine parity (E8).** 1-5 ns residual differences remain for most legs, and ~30 ns
   for PPTA DDK/DDH legs. Next steps: DDK/DDH evaluator parity tests; a decision on whether a
   likelihood-impact tolerance should replace the 1 ns criterion (needs reviewer agreement); and
   tempo2-evaluated residuals and design matrices for non-NG legs as YA did (our container
   accepts any per-leg arrays).
2. **Option C nonlinearity (E7), shared DM and admissibility.** C needs a re-linearised nominal
   model: a joint fit before computing residuals, iterated. J1022 and J0437 also need the
   binary-model differences between legs resolved (frozen Shapiro/Kopeikin terms) before any C
   build is admissible. The v0.9.3 shared-DM rule wraps J1022 and J0437
   (Sec. 6), so YA's pipeline must have done something not captured by "copy the reference"
   (YA EM:3 mentions pulse numbers and per-PTA JUMPs). This is still an open question for the
   reproduction.
3. **Clock files.** EPTA eff2gps.clk has no corrections after MJD 57195.5, and effix2gps ends
   at 59294.5. Affected TOAs are excluded (65); this must be resolved with EPTA's own clock files.
4. **Signed H3.** Retest when PINT #2023 merges.
5. G5 on large systems: discovery and enterprise lose about 1e-5 to 1e-4 nats of shape accuracy
   at 87k TOAs; the long-double arbiter (CURN value and gradient; HD value) is the reference there.
   A pre-registered criterion that tolerates oracle imprecision (e.g. "within 1e-6 of the arbiter,
   or closer to the arbiter than the oracle") needs reviewer agreement before it replaces the
   all-oracle criterion.
6. The engine differences that remain matter for inference on the high-precision PPTA legs: the
   post-hoc likelihood impact is 34 nats for J1909 and 58 for J0437 (Sec. 4).

## 12. Addendum: ELL1H H3+H4 harmonic count (PINT 7 vs tempo2 4) and GW inference

Scope: the 10 legs of this class:
* EPTA J0751+1807 and J1012+5307;
* InPTA DR2 J0751+1807 and J1012+5307;
* MPTA J0613-0200, J1327-0755, J1545-4550, J1804-2717 and J2145-0750;
* PPTA J0613-0200.

Each leg is loaded twice with identical inputs and profiles. One load uses NHARMS = 7 (PINT 1.1.7
forces >= 7 when H4 is given). The other uses NHARMS = 4 (tempo2's default, harmonics 3..4; set
after PINT's set-up, as nanograv/PINT#2046 makes possible). Producer: `scripts/m3a_nharms.py`.
Its output is `nharms.json`, plus the grid coordinates and both likelihood surfaces in
`nharms_surfaces.npz`; that script also writes the G3/G4 columns, so the separate
`nharms_g3.json` producer is retired. The conventions are spelled out in the script's docstring.

**Projectors.** Rank-revealing SVD complements with rtol 1e-10; effective ranks are in the table.
W = 1/sigma uses the raw uncertainties.

**Columns of the table below:**
* **Union** is the difference projected out of span[W M7, W M4]. It is a diagnostic only: a small
  union residual does *not* show that the two column spaces are equal.
* **Own** is e = P7 W r7 - P4 W r4: each analysis marginalises its own timing model.
* **GW14/GW30** is the norm of the part of e lying in the span of the timing-projected first 14 or
  30 Fourier bins of the combined span. It is given for both projectors, P7 and P4; the value
  depends on which is chosen. A whitened norm is in general not the square root of a chi^2
  change, because cross terms with the data matter.
* **D(theta) = lnL7 - lnL4** is taken on a 9 x 9 CURN grid at fixed toy noise (EFAC 1, EQUAD 1 ns,
  IRN log10_A = -14, gamma = 3). Reported: max |D - D(theta0)| with theta0 = (-14.5, 4.25), and the
  peak-to-peak range.

<!-- NHARMS_TABLE -->
| leg | TOAs | ranks M7/M4/union | Shapiro diff rms [ns] | union rms [ns] | own rms [ns] (whitened norm) | own GW14 M7 / M4 | own GW30 M7 / M4 | single-leg CURN max abs(D-D0) / peak-to-peak [nats] | vs tempo2, 7 harm.: ns / sin | 4 harm.: ns / sin |
|---|---|---|---|---|---|---|---|---|---|---|
| epta_dr2new/J0751+1807 | 2467 | 34/34/39 | 24.5 | 0.013 | 85.3 (2.9) | 0.14 / 0.56 | 0.18 / 0.73 | 0.18 / 0.3 | 18.8 / 0.68 | 1.03 / 0.00016 |
| epta_dr2new/J1012+5307 | 4187 | 40/40/45 | 23.4 | 0.013 | 28.7 (1.8) | 0.22 / 0.49 | 0.27 / 0.6 | 0.41 / 0.55 | 16.5 / 0.76 | 0.926 / 0.00011 |
| inpta_dr2/J0751+1807 | 1518 | 7/7/8 | 0.0233 | 0.023 | 0.0226 (0.00016) | 6.1e-05 / 6.1e-05 | 8.2e-05 / 8.2e-05 | 9.8e-05 / 9.9e-05 | 1.87 / 0.00012 | 1.88 / 0.00012 |
| inpta_dr2/J1012+5307 | 5952 | 7/7/8 | 18.5 | 18 | 18.1 (0.5) | 0.18 / 0.18 | 0.37 / 0.37 | 0.66 / 0.79 | 18.1 / 7.3e-05 | 1.63 / 7.3e-05 |
| mpta/J0613-0200 | 3067 | 20/20/24 | 56.3 | 0.018 | 46.2 (2.9) | 0.8 / 1.1 | 0.95 / 1.5 | 1.3 / 1.9 | 30.7 / 0.77 | 1.76 / 9.3e-05 |
| mpta/J1327-0755 | 788 | 17/17/21 | 1.63e+03 | 0.0028 | 400 (8.3) | 1.2 / 4.8 | 1.7 / 6 | 7.7 / 8.1 | 125 / 0.96 | 1.61 / 0.00044 |
| mpta/J1545-4550 | 4905 | 20/20/23 | 1.51 | 0.0024 | 4.06 (0.23) | 0.03 / 0.052 | 0.044 / 0.068 | 0.21 / 0.27 | 2.13 / 0.2 | 1.61 / 7.9e-05 |
| mpta/J1804-2717 | 1569 | 17/17/22 | 41.2 | 0.025 | 202 (2.1) | 0.25 / 0.46 | 0.26 / 0.5 | 0.77 / 0.81 | 35.7 / 0.45 | 3.38 / 0.00015 |
| mpta/J2145-0750 | 2966 | 18/18/22 | 228 | 0.0028 | 206 (11) | 3.3 / 6.8 | 3.8 / 7.7 | 17 / 19 | 87.9 / 0.96 | 4.29 / 0.00017 |
| ppta_dr3_gh/J0613-0200 | 4927 | 46/46/51 | 616 | 0.012 | 156 (11) | 2.2 / 2.4 | 4 / 3.3 | 2.6 / 3.4 | 128 / 1 | 4.98 / 8.9e-05 |
<!-- /NHARMS_TABLE -->

<!-- NHARMS_COMBINED -->
Combined system (J0613-0200 (MPTA+PPTA), J0751+1807 (EPTA+InPTA), J1012+5307 (EPTA+InPTA), J1327-0755 (MPTA), J1545-4550 (MPTA), J1804-2717 (MPTA), J2145-0750 (MPTA); legs stacked per pulsar as option B), same fixed noise: CURN max abs(D - D0) 22.8 nats, peak-to-peak 24.5; HD 23.4 / 25.2 nats. Both likelihoods peak at the scan boundary [-13.5, 6.5] (this toy noise model has no red-noise freedom), so these numbers are likelihood-shape diagnostics, not posterior shifts (Sec. 12b).
<!-- /NHARMS_COMBINED -->

**Interpretation (corrected).** The two harmonic counts give genuinely parameter-dependent
likelihood differences: the H3/H4 design columns differ (G4 sine 0.2-1.0) and so do the
marginalised data terms. But under this toy noise model both likelihoods peak at the scan boundary
(the model has no red-noise freedom). The reference box that revision 1 called the "posterior
region" came from published analyses with different noise models. Neither number is a posterior
shift. The posterior-level answer follows.

### 12b. Posterior-level effect under realistic noise

Producer: `scripts/m3a_nharms_posterior.py`. It writes `nharms_posterior.json`, plus every grid
and lnL surface in `nharms_posterior_surfaces.npz`.

**Model.** Conditional (fixed-noise) posteriors of a 14-bin power-law common process on the array
span, with uniform priors log10_A in [-18, -11] and gamma in [0, 7]. Grids: 141 x 141 in
(log10_A, gamma), and 1401 points in log10_A at gamma = 13/3. The 4- and 7-harmonic variants
differ only in the 10 affected legs.

**Noise model.** Fixed, from released values where available:
* EPTA DR2new noisefiles (EFAC/TNEQUAD; DM in TempoNest normalisation, with mode counts from
  `dm_dict`/`red_dict`; RN for J1012);
* InPTA DR2 par T2EFAC;
* MPTA `example_noise.json` (EFAC/TNEQUAD/ECORR);
* PPTA DR3 single-pulsar noise JSON (EFAC/TNEQUAD, band and global ECORR terms with the PPTA
  selections, RN, DM, band-noise-low);
* NG15 v1p1 WN dictionary plus IRN medians from the released CURN chain.

The MPTA legs of J1327, J1545, J1804 and J2145 have no machine-readable RN/DM release, so RN + DM
power laws are fitted by maximum likelihood at the released WN, on the 4-harmonic data, and the
fit is used in both variants. Several of these fits go to the lower amplitude bound, i.e. no
detectable red noise. Not modelled: PPTA chromatic GP (log10 A = -16.7), the annual DM sinusoid
and solar-wind terms.

**Arrays.**
* *affected*: the 7 pulsars, with legs stacked per pulsar as option B;
* *affected+NG15*: the same plus the NG15 legs of J1909-3744, J1713+0747, J1744-1134 and
  J0030+0451.

**Shifts are reported in units of the 4-harmonic posterior:**
* d(median) / sigma68, with sigma68 = (q84 - q16)/2;
* the shifts of the 5 % and 95 % quantiles divided by the 90 % width w90.

<!-- NHARMS_POSTERIOR -->
| array | ORF | quantity | 4 harm.: median [5%, 95%] | 7 harm.: median [5%, 95%] | dmedian / sigma68 | d5% / w90 | d95% / w90 | lnL max interior (4 / 7) |
|---|---|---|---|---|---|---|---|---|
| affected | CURN | log10_A | -11.8 [-13, -11.1] | -11.8 [-12.9, -11.1] | 0.05 | 0.019 | 0.0036 | True / True |
| affected | CURN | gamma | 3.29 [2.91, 3.78] | 3.28 [2.89, 3.76] | -0.066 | -0.013 | -0.023 | True / True |
| affected | CURN | log10_A (gamma = 13/3) | -14 [-14.2, -13.8] | -14 [-14.3, -13.8] | -0.098 | -0.038 | -0.019 | True / True |
| affected | HD | log10_A | -11.9 [-13.1, -11.1] | -11.8 [-13, -11.1] | 0.055 | 0.024 | 0.0039 | True / True |
| affected | HD | gamma | 3.31 [2.91, 3.81] | 3.3 [2.91, 3.79] | -0.065 | -0.0097 | -0.023 | True / True |
| affected | HD | log10_A (gamma = 13/3) | -14 [-14.3, -13.8] | -14 [-14.3, -13.8] | -0.082 | -0.033 | -0.015 | True / True |
| affected+NG15 | CURN | log10_A | -13.7 [-14.4, -13] | -13.7 [-14.4, -13] | 0.0041 | 0.0017 | 0.00024 | True / True |
| affected+NG15 | CURN | gamma | 3.55 [3.27, 3.81] | 3.55 [3.27, 3.81] | -0.018 | -0.0042 | -0.0076 | True / True |
| affected+NG15 | CURN | log10_A (gamma = 13/3) | -14.4 [-14.5, -14.3] | -14.4 [-14.5, -14.3] | -0.036 | -0.01 | -0.012 | True / True |
| affected+NG15 | HD | log10_A | -13.8 [-14.4, -13] | -13.8 [-14.4, -13] | -0.0017 | 0.00035 | -0.00099 | True / True |
| affected+NG15 | HD | gamma | 3.56 [3.28, 3.82] | 3.56 [3.27, 3.82] | -0.012 | -0.0028 | -0.0047 | True / True |
| affected+NG15 | HD | log10_A (gamma = 13/3) | -14.4 [-14.5, -14.3] | -14.4 [-14.5, -14.3] | -0.034 | -0.01 | -0.011 | True / True |
<!-- /NHARMS_POSTERIOR -->

Every lnL maximum is interior to the grid (last column). For the *affected* array alone, the
2D maximum is at gamma ~ 6.3, log10_A ~ -14.7. Its marginal log10_A posterior, however, extends
to the upper prior edge (95 % quantile -11.1), through the strong A-gamma degeneracy of a weak
7-pulsar array. Those quantiles are prior-bounded, and the gamma = 13/3 amplitude (interior,
-14.0 [-14.2, -13.8]) is the cleaner comparison. With the four NG15 pulsars every interval is
well inside the prior.

**Result.** At the posterior level the harmonic count is negligible for these arrays:
* *affected* alone: |d median| <= 0.07 sigma68 for log10_A and gamma, and <= 0.10 sigma68 for the
  gamma = 13/3 amplitude; 90 % bounds move by <= 0.04 w90;
* *affected+NG15*: <= 0.04 sigma68, and the bounds move by <= 0.012 w90.

The several-nat likelihood-shape changes of the toy model (Sec. 12 table) are absorbed by the
released red/DM noise and the posterior width. This holds for a *conditional* posterior with
fixed noise. A noise-marginalised analysis could differ, and is deferred to M3b.

**Quarantine decision (configuration-specific).** `quarantine({"ell1h_nharms": "tempo2"})`
lifts the 10 legs; `{"ell1h_nharms": "pint7"}` keeps them quarantined. Under the 4-harmonic
convention PINT agrees with tempo2 to 0.9-5 ns projected rms and G4 sine 7e-5 to 4e-4, the level
of the non-quarantined legs. The convention applies on every construction path: published legs
resolve NHARMS from the original par, and option C carries the reference's resolved count through
the rewrite (tested: EPTA/InPTA J0751, `nharms_used = 4` in both legs). Each leg records
`nharms_used` and `ell1h_nharms_config`. The lifted legs share the open E8 status of all non-NG
legs.
