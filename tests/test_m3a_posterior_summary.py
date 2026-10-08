"""Marginalisation of the NHARMS posterior grids (review M3a r2 #1): unequal axis lengths, known
analytic marginals."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m3a_nharms_posterior import summarize, summarize_1d


def test_marginals_unequal_axes():
    gA = np.linspace(-18.0, -11.0, 281)  # 281 amplitude points
    gg = np.linspace(0.0, 7.0, 141)  # 141 gamma points
    GG, AA = np.meshgrid(gg, gA, indexing="ij")  # shape (len(gg), len(gA)), as in the script
    lnL = -0.5 * ((AA + 14.2) / 0.3) ** 2 - 0.5 * ((GG - 4.5) / 0.6) ** 2  # independent Gaussians
    s = summarize(gA, gg, lnL)
    # the script's cumulative-sum quantile convention is accurate to about half a grid step
    assert abs(s["log10_A"]["q50"] + 14.2) < 0.02 and abs(s["gamma"]["q50"] - 4.5) < 0.03
    assert abs((s["log10_A"]["q84"] - s["log10_A"]["q16"]) / 2 - 0.3) < 0.01
    assert abs((s["gamma"]["q84"] - s["gamma"]["q16"]) / 2 - 0.6) < 0.01
    assert s["max"]["interior"] and abs(s["max"]["log10_A"] + 14.2) < 0.03 and abs(s["max"]["gamma"] - 4.5) < 0.05
    with pytest.raises(ValueError):
        summarize(gA, gg, lnL.T)
    one = summarize_1d(gA, -0.5 * ((gA + 14.0) / 0.2) ** 2)
    assert abs(one["q50"] + 14.0) < 0.02
