"""Oracle sanity check (reference packages only; NOT part of src/ptagwb).

Builds NANOGrav's `discovery` CURN-gamma likelihood (67 pulsars, 30 RN + 14 common
frequencies, fixed white noise from the feather noisedicts, SVD timing model) on the GPU and
evaluates it at samples of the released NG15 CURN chain (discovery data/NG15yr-m2a-chain.feather).
Differences of log-likelihood between samples should match the chain's own `logl` column
up to a constant offset.

Usage: uv run --group oracle python scripts/oracle_sanity.py [--n 8]
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import discovery as ds
import numpy as np
import pyarrow.feather as pf

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    ddir = next((ROOT / "data/raw/discovery_repo/extracted").glob("discovery-*")) / "data"
    files = sorted(glob.glob(str(ddir / "v1p1_de440_pint_bipm2019-*.feather")))
    psrs = [ds.Pulsar.read_feather(f) for f in files]
    T = ds.getspan(psrs)
    print(f"{len(psrs)} pulsars, T = {T / 365.25 / 86400:.3f} yr, backend {jax.default_backend()}")

    m = ds.ArrayLikelihood(
        [
            ds.PulsarLikelihood(
                [
                    p.residuals,
                    ds.makenoise_measurement(p, p.noisedict),
                    ds.makegp_ecorr(p, p.noisedict),
                    ds.makegp_timing(p, svd=True),
                ]
            )
            for p in psrs
        ],
        commongp=ds.makecommongp_fourier(
            psrs,
            ds.makepowerlaw_crn(components=14),
            components=30,
            T=T,
            name="red_noise",
            common=["crn_log10_A", "crn_gamma"],
        ),
    )
    logL = jax.jit(m.logL)

    chain = pf.read_table(ddir / "NG15yr-m2a-chain.feather").to_pandas()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(chain), size=args.n, replace=False)
    ours, ref = [], []
    for k, i in enumerate(idx):
        row = chain.iloc[i]
        p = {}
        for name in m.logL.params:
            src = {"crn_log10_A": "gw_log10_A", "crn_gamma": "gw_gamma"}.get(name, name)
            p[name] = float(row[src])
        t0 = time.time()
        v = float(logL(p))
        dt = time.time() - t0
        ours.append(v)
        ref.append(float(row["logl"]))
        print(
            f"  sample {i:6d}: discovery logL = {v:.4f}  chain logl = {ref[-1]:.4f}  "
            f"diff = {v - ref[-1]:.4f}  ({dt * 1e3:.0f} ms{' incl. jit' if k == 0 else ''})"
        )
    d = np.array(ours) - np.array(ref)
    print(f"offset mean {d.mean():.4f}, spread (max-min) {d.max() - d.min():.2e}")


if __name__ == "__main__":
    main()
