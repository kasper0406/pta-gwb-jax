"""Global configuration helpers (precision, paths)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def enable_x64() -> None:
    """PTA likelihoods need float64 (residuals ~1e-7 s, covariances span ~30 decades)."""
    import jax

    jax.config.update("jax_enable_x64", True)
