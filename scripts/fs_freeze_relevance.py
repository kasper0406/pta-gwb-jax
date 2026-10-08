"""Freeze the free-spectrum gate's region-relevance declaration before a measurement.

    uv run --no-sync python scripts/fs_freeze_relevance.py --out configs/m2/relevance/hd_fs30_released_v1.json

Derives {bin: relevant regions} from the released Fig. 1(a) HD free-spectrum core (region mass >=
GATE_DEFAULTS["region_min_mass"]) and records the reference identity (path, sha256, stored la_forge
burn-in, number of retained draws) and the criteria. ``scripts/m2_freespec_diag.py --relevance-file``
reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import h5py
from m2_common import FIG1, ROOT, released

from ptagwb.diagnostics import GATE_DEFAULTS, derive_relevant_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = FIG1 / "30fCP_30fiRN_3A_freespec_chain.core"
    with h5py.File(path, "r") as h:
        burn = int(h["metadata"]["burn"][()]) if "burn" in h["metadata"] else 0
        n_total = int(h["chain"].shape[0])
    ref = {k: v for k, v in released("hd_fs30").items() if k.startswith("gw_log10_rho_")}
    rel = derive_relevant_regions(ref, 30)
    out = {
        "relevant_regions": rel,
        "reference": {"key": "hd_fs30", "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "burn_in": burn, "n_total": n_total, "n_retained": n_total - burn},
        "criteria": {k: GATE_DEFAULTS[k] for k in ("region_low", "region_high", "region_min_mass")},
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["reference"]), f"{sum(len(v) for v in rel.values())} relevant regions -> {p}")


if __name__ == "__main__":
    main()
