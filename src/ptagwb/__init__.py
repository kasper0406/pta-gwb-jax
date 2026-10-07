"""ptagwb: a JAX reimplementation of the NANOGrav 15-yr GWB analysis.

PINT is used only as the timing-model front end (residuals, TOA errors, flags, radio
frequencies, design matrix). Everything downstream -- noise covariances, Fourier GP bases,
the marginalised likelihood, ORFs, samplers, optimal statistic -- is our own JAX code.

Reference packages (enterprise, enterprise_extensions, discovery) live in the optional
`oracle` dependency group and must never be imported from this package.
"""

from .config import disable_xla_cpu_ynn_fusion as _disable_xla_cpu_ynn_fusion

__version__ = "0.1.0"

# XLA:CPU (jaxlib 0.11.2) miscompiles batched dots with a broadcast operand inside its YNNPACK
# library fusion (docs/PERF.md, "XLA:CPU YNNPACK fusion bug"; bench/xla_ynn_repro.py). Turn
# that fusion off before any JAX backend is initialised. Appends to XLA_FLAGS, never clobbers it.
_disable_xla_cpu_ynn_fusion()
