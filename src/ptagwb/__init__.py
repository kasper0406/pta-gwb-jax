"""ptagwb: a JAX reimplementation of the NANOGrav 15-yr GWB analysis.

PINT is used only as the timing-model front end (residuals, TOA errors, flags, radio
frequencies, design matrix). Everything downstream -- noise covariances, Fourier GP bases,
the marginalised likelihood, ORFs, samplers, optimal statistic -- is our own JAX code.

Reference packages (enterprise, enterprise_extensions, discovery) live in the optional
`oracle` dependency group and must never be imported from this package.
"""

__version__ = "0.1.0"
