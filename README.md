# pta-gwb-jax

Reproducing the NANOGrav 15-year gravitational-wave-background results
(Agazie et al. 2023, ApJL 951 L8, [arXiv:2306.16213](https://arxiv.org/abs/2306.16213))
with our own JAX analysis pipeline on a single GPU.

- **Front end:** [PINT](https://github.com/nanograv/PINT) turns `.par`/`.tim` files into
  residuals, TOA uncertainties, backend flags, radio frequencies and the timing-model
  design matrix.
- **Everything downstream is our own JAX code** (`src/ptagwb`): noise models, Fourier GP
  bases, the marginalised PTA likelihood (CURN / HD), samplers, Bayes factors and the
  optimal statistic.
- **Reference oracles only:** `enterprise`, `enterprise_extensions` and NANOGrav's JAX
  `discovery` sit in an optional dependency group. Tests use them to cross-check
  likelihood values. They are never imported by `src/ptagwb`.

Status: **M1 (deterministic pipeline and likelihood) done; M2 (sampling, Bayes factors,
optimal statistic) largely done; HD free spectrum: Fig. 1a partially reproduced: principal peak locations agree; tail occupancies and full posterior convergence remain unestablished** (campaign closed 2026-10-08 with
unconverged chains; see [`docs/FS_PILOT.md`](docs/FS_PILOT.md) for the pilots, the fail-closed gate, the exact hybrid
sampler and lessons for M3). M1 covers the PINT
ingestion into a hashed cache, fixed white noise, Fourier bases, ORFs, and the JAX float64 CURN / HD
likelihoods with gradients, validated against discovery, enterprise and the released chains
([`docs/M1_VALIDATION.md`](docs/M1_VALIDATION.md)). M2 adds NumPyro NUTS sampling, Bayes-factor
estimators and the optimal statistic. The power-law posteriors (HD^13/3 A = 2.45e-15, HD^gamma
A = 6.4e-15 / gamma = 3.23), the HD vs CURN Bayes factor (bridge ~180; estimators span 178-228)
and the OS (S/N 4.5 / 5.0, binned chi^2 = 8.1 at the released noise point) match the paper and
the released products, with the Bayes-factor uncertainty provisional. No run is fully
convergence-certified: several IRN nuisance parameters miss the paper's all-parameter R-hat < 1.01
criterion.
See [`docs/M2_RESULTS.md`](docs/M2_RESULTS.md). See [`docs/PLAN.md`](docs/PLAN.md) for the analysis
settings, target numbers and milestones, [`docs/SPEC_astra.md`](docs/SPEC_astra.md) for the
independent reproduction spec, and [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the
verified versions.

```python
from ptagwb.data import load_pulsars, get_tspan
from ptagwb.noise import load_noise_dict
from ptagwb.likelihood import PTALikelihood, precompute

psrs = load_pulsars()                       # 67 GWB pulsars (PINT, cached in data/cache)
T = get_tspan(psrs)                         # 505861299.1401644 s
terms = precompute(psrs, load_noise_dict(), T)
hd = PTALikelihood(terms, T, orf="hd")      # or "curn"; common="freespec" for a free spectrum
logL = hd.logL({"rn_log10_A": ..., "rn_gamma": ..., "log10_A": -14.6, "gamma": 13 / 3})

# M2: NUTS (uniform box priors via a logistic transform), BF estimators, optimal statistic
from ptagwb.sampling import ModelSpec, Posterior, RunConfig, run_nuts
post = Posterior(hd, ModelSpec(orf="hd", gamma=13 / 3))
# run_nuts(RunConfig(name="...", model={...}), post) -> runs/<name>/{samples.npz,meta.json}
from ptagwb.optstat import OptimalStatistic   # OptimalStatistic(curn_like).os(params)
from ptagwb.evidence import bridge, reweight   # ln BF from log-likelihood ratios at draws
```

M2 production runs: `scripts/m2_production.sh` (configs in `configs/m2/`), then
`scripts/m2_{compare,bayes,optstat,figures,report}.py`; see `docs/M2_RESULTS.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with a recent driver. CUDA is
pip-installed through `jax[cuda13]`.

```bash
uv sync                                   # core + dev dependencies
uv run python scripts/check_gpu.py        # JAX sees the GPU; float64 matmul + Cholesky on GPU
uv run python scripts/ingest.py           # PINT -> data/cache/pulsars (~2 min, once)
uv run pytest                             # oracle tests skip without the oracle group
```

Optional reference oracles. This builds scikit-sparse without root; see `docs/ENVIRONMENT.md`:

```bash
scripts/setup_oracle_env.sh
uv run --no-sync python scripts/oracle_sanity.py
uv run --no-sync pytest                             # incl. oracle tests (~9 min)
PTAGWB_REQUIRE_ORACLES=1 uv run --no-sync pytest    # validation mode: any skip fails
uv run --no-sync python scripts/m1_validate.py --enterprise
```

Once the oracle group is installed, use `uv run --no-sync`: a plain `uv run` re-syncs the
default groups and removes the oracle packages.

## Data

Raw data go to `data/raw/`, which is git-ignored. The download is scripted and
checksummed. `data/MANIFEST.json` is committed and records the source URL/DOI, file
names, sizes, sha256 and md5 (verified against Zenodo's published md5 where available),
plus what was not fetched and why.

```bash
uv run python scripts/fetch_data.py --list            # show sources
uv run python scripts/fetch_data.py --group ng15      # NG15 inputs only (~2.1 GB)
uv run python scripts/fetch_data.py                   # everything except "large" sources (~8.3 GB)
uv run python scripts/fetch_data.py --include-large   # + 22 GB NG15 new-physics chains
```

Groups:
- `ng15`: the NG15 data set (Zenodo v2.1.0 and v1.0.1), the official GWB companion repo
  `nanograv/15yr_stochastic_analysis` (white-noise dictionary, feathers, presampled
  chains), `nanograv/discovery` data (feathers, CURN/HD chains with `logl`), and the NG15 CW
  release (`v1p1_all_dict.json`).
- `ng15_reference`: GWB-paper figure data, including the released HD^gamma / HD^13/3
  chains, the free-spectrum and spline-ORF cores, OS products, sensitivity curves and Ceffyl
  KDEs.
- `secondary` (download only, not used yet): EPTA DR2 (Zenodo + GitLab), PPTA DR3 timing
  files (CSIRO DAP), InPTA DR2 (GitHub).

## Smoke test

```bash
uv run python scripts/smoke_load.py                  # 68 NG15 narrowband pulsars, ~2 min on 16 cores
uv run python scripts/smoke_load.py --include-split  # + 8 split-telescope (ao/gbt) files
```

The script loads every pulsar with PINT using the par-file settings (DE440, TT(BIPM2019)).
It prints the pre-fit weighted RMS, reduced chi^2, design-matrix shape, number of DMX
bins and number of backends, and computes the array span (16.030 yr, as in the paper).

## Layout

```
src/ptagwb/     our pipeline (data, noise, basis, orf, likelihood, sampling, diagnostics,
                evidence, optstat)
scripts/        fetch_data.py, smoke_load.py, check_gpu.py, ingest.py, m1_validate.py,
                oracle_sanity.py, setup_oracle_env.sh, m2_*.py, m2_production.sh
configs/m2/     committed NUTS run configurations (runs/ is git-ignored)
tests/          unit tests (synthetic PTA vs dense brute force) and oracle tests (-m oracle)
docs/           PLAN.md, ENVIRONMENT.md, SPEC_astra.md, M1_VALIDATION.md, M2_RESULTS.md,
                figures/ (M2 figures)
data/           MANIFEST.json (committed); raw/, cache/ and processed/ are git-ignored
```
