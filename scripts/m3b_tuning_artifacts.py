"""Reference-derived sampler tuning artifacts for the EPTA CURN^gamma pilot v2 (review of pilot v1,
``review_epta_pilot1.out``). **A disclosed efficiency aid, not part of the target**: the metric and
the block-MH proposals leave the posterior unchanged (exact Hastings ratios, full prior support,
independent prior_central starts); they make the computation depend on the released chain, so the
pilot is no longer reference-blind. Using them is a prospective protocol revision of plan Sec. 5.2
("proposals from our own pilot draws only") that needs the user's approval.

Inputs: the released CURN^gamma chain (``epta.load_reference("crn_pl")``, sha256 verified against the
manifest), the frozen burn-in (manifest ``burn_in_rows``), every retained row, parameters matched by
name to the sampler's layout (manifest parameters of model ``crn``, free gamma).

Outputs (to ``--out``, default a staging directory outside the repo's committed configs):
* ``epta_curn_ref_z_v1.npz``: ``inverse_mass_matrix`` = 0.95 Cov(z) + 0.05 diag(Cov(z)) + 1e-6 I on
  the 66 continuous coordinates (dip t0 excluded) in sampler order, with
  z = log((x - lo) / (hi - x)) (the sampler's box logit); ``names``; a covariance, not its inverse.
* ``epta_curn_ref_v1.json``: ``BlockProposal`` specs in the driver's block layout (every
  (log10_A[, gamma]) amplitude pair, then the joint (t0, log10_tau, log10_Amp) dip block): w_prior
  0.2, and per coordinate K = 20 equal-mass bins, edges = [lo, reference quantiles k/K, hi].
* ``provenance.json``: hashes (chain, pars, outputs, this script), burn-in, rows, column mapping,
  transform, recipe, eigenvalue range, code HEAD.

Usage: PYTHONPATH=src python scripts/m3b_tuning_artifacts.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ptagwb import epta
from ptagwb.config import REPO_ROOT

STAGING = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "tuning_staging"
METRIC_NAME, PROPOSAL_NAME = "epta_curn_ref_z_v1.npz", "epta_curn_ref_v1.json"
SHRINK, JITTER, W_PRIOR, K_BINS = 0.95, 1e-6, 0.2, 20


def layout(man: dict) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    """(parameter names, lo, hi, t0 name) of the CURN^gamma sampler (= EPTAModel('crn') order)."""
    keep = [q for q in man["parameters"] if "crn" in q["models"]]
    names = [q["name"] for q in keep]
    lo = np.array([q["bounds"][0] for q in keep], np.float64)
    hi = np.array([q["bounds"][1] for q in keep], np.float64)
    return names, lo, hi, f"{man['dip']['param_prefix']}_t0"


def box_logit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if not np.all((x > lo) & (x < hi)):
        raise ValueError("reference values on or outside the prior box: the logit is undefined")
    return np.log(x - lo) - np.log(hi - x)


def regularised_metric(z: np.ndarray) -> np.ndarray:
    """0.95 Cov(z) + 0.05 diag(Cov(z)) + 1e-6 I (rows = draws)."""
    c = np.cov(z, rowvar=False, ddof=1)
    m = SHRINK * c + (1.0 - SHRINK) * np.diag(np.diag(c)) + JITTER * np.eye(c.shape[0])
    return 0.5 * (m + m.T)


def equal_mass_edges(v: np.ndarray, lo: float, hi: float, k: int = K_BINS) -> list[float]:
    """[lo, q(1/k), ..., q((k-1)/k), hi]; strictly increasing or ValueError."""
    inner = np.quantile(v, np.arange(1, k) / k)
    e = np.concatenate([[lo], inner, [hi]])
    if not np.all(np.diff(e) > 0):
        raise ValueError("equal-mass edges are not strictly increasing")
    return [float(t) for t in e]


def verified_names(pars_path: Path, expected_sha256: str) -> list[str]:
    """Parameter names of the released chain from ``pars.txt`` after verifying the file's actual
    sha256 against the manifest (review of the v2 preparation: the column mapping must not rest on
    an unverified file)."""
    raw = Path(pars_path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha256:
        raise ValueError(f"{pars_path}: sha256 {got} != manifest {expected_sha256}")
    return raw.decode().split()


def verified_reference(man: dict) -> tuple[list[str], np.ndarray, int, dict]:
    """(names, chain, burn-in, verified hashes): chain_1.txt (by the loader) and pars.txt both
    verified against the manifest."""
    rc = man["reference_chains"]["crn_pl"]
    names = verified_names(epta.extract_reference("crn_pl") / "pars.txt", rc["pars_sha256"])
    ref_names, X, burn = epta.load_reference("crn_pl", man)
    if ref_names != names:
        raise ValueError("pars.txt changed between verification and loading")
    return names, X, burn, {"chain_sha256_verified": rc["chain_sha256"], "pars_sha256_verified": rc["pars_sha256"]}


def block_layout(man: dict, names: list[str]) -> list[tuple[str, ...]]:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from m3b_run_epta import block_layout as bl  # the driver's layout, so they cannot diverge

    return bl(man, names)


def build(man: dict) -> tuple[dict, dict, dict]:
    names, lo, hi, t0 = layout(man)
    ref_names, X, burn, verified = verified_reference(man)
    col = {n: i for i, n in enumerate(ref_names)}
    missing = [n for n in names if n not in col]
    if missing:
        raise ValueError(f"sampler parameters missing from the reference: {missing}")
    R = X[burn:, [col[n] for n in names]]
    cont = [i for i, n in enumerate(names) if n != t0]
    z = box_logit(R[:, cont], lo[cont], hi[cont])
    imm = regularised_metric(z)
    ev = np.linalg.eigvalsh(imm)
    metric = {"inverse_mass_matrix": imm, "names": np.array([names[i] for i in cont])}
    blocks = []
    ix = {n: i for i, n in enumerate(names)}
    for b in block_layout(man, names):
        j = [ix[n] for n in b]
        blocks.append({"params": list(b), "lo": [float(lo[i]) for i in j], "hi": [float(hi[i]) for i in j],
                       "w_prior": W_PRIOR, "edges": [equal_mass_edges(R[:, i], lo[i], hi[i]) for i in j]})
    prov = {"purpose": "disclosed efficiency aid for the EPTA CURN^gamma pilot v2 (review_epta_pilot1.out); "
                       "requires the user's approval as a protocol revision of plan Sec. 5.2",
            "reference": {"model": "crn_pl", **verified,
                          "rows_total": int(X.shape[0]), "burn_in_rows": int(burn), "rows_used": int(R.shape[0]),
                          "column_mapping": {n: int(col[n]) for n in names}},
            "metric": {"coordinates": "66 continuous sampler coordinates, dip t0 excluded, sampler order",
                       "transform": "z = log((x - lo) / (hi - x)), lo/hi the manifest prior box",
                       "recipe": f"{SHRINK} Cov(z) + {1 - SHRINK:.2f} diag(Cov(z)) + {JITTER} I; np.cov ddof=1; "
                                 "a covariance (NumPyro inverse_mass_matrix), not its inverse",
                       "eigenvalue_min": float(ev[0]), "eigenvalue_max": float(ev[-1]),
                       "condition_number": float(ev[-1] / ev[0])},
            "proposals": {"layout": "driver block_layout: every (log10_A[, gamma]) pair in model order, then "
                                    "(t0, log10_tau, log10_Amp)", "n_blocks": len(blocks), "w_prior": W_PRIOR,
                          "bins": K_BINS, "edges": "[lo, reference quantiles k/K (numpy linear), hi]",
                          "density": "eventmh.BlockProposal: w_prior uniform box + (1 - w_prior) product of "
                                     "per-coordinate equal-mass histograms; exact Hastings ratio"},
            "numpy": np.__version__}
    return metric, {"blocks": blocks}, prov


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=STAGING)
    a = ap.parse_args()
    man = epta.load_manifest()
    metric, props, prov = build(man)
    a.out.mkdir(parents=True, exist_ok=True)
    mp, pp = a.out / METRIC_NAME, a.out / PROPOSAL_NAME
    np.savez(mp, **metric)
    pp.write_text(json.dumps({**props, "provenance": prov}, indent=1))
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout
    prov_out = {**prov, "outputs": {METRIC_NAME: _sha(mp), PROPOSAL_NAME: _sha(pp)},
                "generator": {"script": "scripts/m3b_tuning_artifacts.py", "sha256": _sha(Path(__file__)),
                              "head": head.strip()}}
    (a.out / "provenance.json").write_text(json.dumps(prov_out, indent=1))
    print(json.dumps({k: prov_out[k] for k in ("reference", "metric", "outputs")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
