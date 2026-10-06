# Environment (verified 2026-10-06)

Machine: single NVIDIA GeForce RTX 5090 (32 GB, sm_120 / Blackwell), driver 580.126.20
(reports CUDA 13.0), Ubuntu 24.04, Python 3.12, uv 0.8.0. No root access.

## Core stack (locked in `uv.lock`)

| Package | Version |
|---|---|
| jax / jaxlib | 0.11.2 |
| jax-cuda13-plugin / jax-cuda13-pjrt | 0.11.2 (`jax[cuda13]` extra; pip-installed CUDA, nothing from the system) |
| nvidia-cuda-runtime / nvcc / nvjitlink | 13.4.92 |
| nvidia-cublas | 13.8.1.7 |
| nvidia-cusolver | 12.3.4.7 |
| nvidia-cudnn-cu13 | 9.27.0.42 |
| numpyro | 0.22.0 |
| pint-pulsar (PINT) | 1.1.7 |
| astropy | 8.0.1 |
| numpy / scipy | 2.5.3 / 1.18.1 |

`jax[cuda13]` worked on the first try, so the `jax[cuda12]` fallback was not needed.

## GPU verification

```
$ uv run python scripts/check_gpu.py
jax 0.11.2  jaxlib 0.11.2
default backend: gpu
devices: [CudaDevice(id=0)]
device kind: NVIDIA GeForce RTX 5090
matmul+cholesky n=2048: dtype=float64 on={'gpu'} logdet=16803.067593 rel.recon.err=9.31e-16
OK: float64 matmul + Cholesky ran on GPU
```

Note: the pip CUDA runtime (13.4) is newer than the driver's reported CUDA version (13.0).
It works, but if a future JAX/CUDA bump fails with a PTX/driver error, pin the
`nvidia-*` wheels back or update the driver.

## Oracle group (reference implementations, optional)

`[dependency-groups] oracle` = enterprise-pulsar 3.5.0, enterprise_extensions 3.0.3,
discovery 0.5 (git rev b26d2554df5540a7610eedf926659701daad01ad), plus scikit-sparse
0.4.16 (pulled in by enterprise).

enterprise imports `sksparse.cholmod` unconditionally. scikit-sparse is sdist-only and
needs the SuiteSparse/CHOLMOD headers, and we have no root to install
`libsuitesparse-dev`. `scripts/setup_oracle_env.sh` works around this. It runs
`apt-get download` (no root needed) for the Ubuntu SuiteSparse 7.6.1 + BLAS/LAPACK/metis
packages, unpacks them with `dpkg -x` into `~/.local/opt/suitesparse-7.6.1`, and builds
scikit-sparse against that prefix with an RPATH. Install with:

```
scripts/setup_oracle_env.sh          # == uv sync --group oracle, with the right CFLAGS/LDFLAGS
uv run --group oracle python scripts/oracle_sanity.py
```

`oracle_sanity.py` builds discovery's 67-pulsar CURN^gamma likelihood on the GPU. At
samples of the released NG15 CURN chain it reproduces the chain's stored absolute log
likelihood (~7.97e6) to within the float32 rounding of the stored column. One evaluation
takes about 2 ms after JIT.

Plain `uv sync` installs only the core and dev groups, and so does every plain `uv run`,
which re-syncs. With the oracle group installed, run everything as `uv run --no-sync ...`
so the oracle packages stay installed. Never import oracle packages
from `src/ptagwb`; `tests/test_setup.py` enforces this.
