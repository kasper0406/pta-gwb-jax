"""Fourier basis, spectral normalisation, ORFs."""

from __future__ import annotations

import numpy as np

from ptagwb import orf
from ptagwb.basis import FYR, fourier_basis, free_spectrum, powerlaw


def test_fourier_basis_ordering_and_common_columns():
    T = 1e8
    t = np.linspace(4.6e9, 4.6e9 + T, 50)
    f, df, F = fourier_basis(t, 5, T)
    np.testing.assert_allclose(f, np.repeat(np.arange(1, 6) / T, 2))
    np.testing.assert_allclose(df, 1.0 / T)
    np.testing.assert_allclose(F[:, 0], np.sin(2 * np.pi * t / T))
    np.testing.assert_allclose(F[:, 1], np.cos(2 * np.pi * t / T))
    np.testing.assert_allclose(F[:, 9], np.cos(2 * np.pi * 5 * t / T))
    _, _, Fc = fourier_basis(t, 3, T)
    np.testing.assert_array_equal(Fc, F[:, :6])


def test_powerlaw_normalisation():
    T = 505861299.1401644
    f, df, _ = fourier_basis(np.zeros(1), 3, T)
    A, g = 2.4e-15, 13 / 3
    expect = A**2 / (12 * np.pi**2) * FYR ** (g - 3) * f ** (-g) / T
    np.testing.assert_allclose(powerlaw(f, df, np.log10(A), g), expect, rtol=1e-13)
    assert abs(FYR - 1 / 31557600.0) < 1e-22
    np.testing.assert_allclose(free_spectrum(np.array([-7.0, -8.0])), [1e-14, 1e-14, 1e-16, 1e-16])


def test_hd_values():
    pos = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, -1.0], [np.sin(1e-4), 0, np.cos(1e-4)]])
    G = orf.hd(pos)
    np.testing.assert_allclose(np.diag(G), 1.0)
    x = 0.5  # 90 deg
    np.testing.assert_allclose(G[0, 1], 1.5 * x * np.log(x) - x / 4 + 0.5)
    np.testing.assert_allclose(G[0, 2], 0.25)  # antipodal: x = 1
    assert abs(G[0, 3] - 0.5) < 1e-6  # nearly aligned distinct pulsars -> 1/2
    rng = np.random.default_rng(0)
    p = rng.normal(size=(67, 3))
    p /= np.linalg.norm(p, axis=1)[:, None]
    assert np.linalg.eigvalsh(orf.hd(p)).min() > 0
    np.testing.assert_array_equal(orf.curn(p), np.eye(67))
    np.testing.assert_allclose(np.diag(orf.monopole(p)), 1 + 1e-5)
    np.testing.assert_allclose(orf.dipole(p, 1e-6)[0, 1], p[0] @ p[1])
