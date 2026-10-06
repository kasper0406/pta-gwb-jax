"""Shared fixtures. Oracle tests are skipped when the oracle group or the data are absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "oracle: cross-checks against enterprise / discovery / released products")
    config.addinivalue_line("markers", "slow: takes more than ~30 s (PINT loading, full-PTA oracles)")


@pytest.fixture(scope="session")
def noisedict():
    from ptagwb.noise import WN_DICT_PATH, load_noise_dict

    if not WN_DICT_PATH.exists():
        pytest.skip("v1p1_wn_dict.json not fetched")
    return load_noise_dict()


@pytest.fixture(scope="session")
def feathers():
    from oracle_helpers import feather_paths, feather_pulsars

    try:
        if len(feather_paths()) != 67:
            pytest.skip("discovery feathers not fetched")
    except StopIteration:
        pytest.skip("discovery repo not fetched")
    return feather_pulsars()


@pytest.fixture(scope="session")
def ours():
    """Our PINT-ingested 67 pulsars (ingests into data/cache on first use, ~2 min)."""
    from ptagwb.data import find_par_tim, load_pulsars

    try:
        find_par_tim()
    except FileNotFoundError:
        pytest.skip("NG15 release not fetched")
    return load_pulsars(verbose=False)
