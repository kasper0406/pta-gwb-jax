"""Overlap reduction functions as (P, P) matrices from pulsar unit vectors.

Auto-terms are identified by pulsar *index* (not by comparing positions).

* HD: Gamma_ab = 3/2 x ln x - x/4 + 1/2 with x = (1 - p_a.p_b)/2 for a != b, and 1 on the
  diagonal (GWB Eq. 4, enterprise ``hd_orf``: the pulsar-term auto-correlation makes the
  diagonal 1, not 1/2).
* CURN: identity.
* Monopole: all ones; dipole: p_a.p_b. enterprise adds 1e-5 to their diagonals (discovery
  1e-6) to keep the otherwise rank-1 / rank-3 matrices positive definite; ``diag_eps``
  reproduces either.
"""

from __future__ import annotations

import numpy as np


def _cos_sep(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    c = pos @ pos.T
    return np.clip(c, -1.0, 1.0)


def hd(pos: np.ndarray) -> np.ndarray:
    c = _cos_sep(pos)
    x = (1.0 - c) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        xlogx = np.where(x > 0.0, x * np.log(np.where(x > 0.0, x, 1.0)), 0.0)
    G = 1.5 * xlogx - 0.25 * x + 0.5
    np.fill_diagonal(G, 1.0)
    return G


def curn(pos: np.ndarray) -> np.ndarray:
    return np.eye(len(pos))


def monopole(pos: np.ndarray, diag_eps: float = 1e-5) -> np.ndarray:
    P = len(pos)
    return np.ones((P, P)) + diag_eps * np.eye(P)


def dipole(pos: np.ndarray, diag_eps: float = 1e-5) -> np.ndarray:
    G = _cos_sep(pos).copy()
    np.fill_diagonal(G, 1.0 + diag_eps)
    return G


ORFS = {"hd": hd, "curn": curn, "monopole": monopole, "dipole": dipole}
