"""Summaries (same format as bench/samplers.py) of the M2 production NumPyro runs, so that their
ESS/s and ESS per gradient enter the sampler table. Sampling-phase wall time and gradient counts
from runs/<name>/meta.json (production likelihood, RTX 5090).

    JAX_PLATFORMS=cpu uv run --no-sync python bench/m2_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptagwb import diagnostics as dg
from ptagwb.sampling import load_run

OUT = Path(__file__).resolve().parent / "results" / "samplers"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in (("curn_g433_14f", "curn"), ("hd_g433_14f", "hd")):
        r = load_run(name)
        meta, X, names = r["meta"], r["x"], list(r["names"])
        D = X.shape[-1]
        bulk = np.array([dg.ess_bulk(X[:, :, j]) for j in range(D)])
        tail = np.array([dg.ess_tail(X[:, :, j]) for j in range(D)])
        rh = np.array([dg.rhat(X[:, :, j]) for j in range(D)])
        jc = names.index("gw_log10_A")
        grads = float(meta["sampling_grad_evals"])
        t = float(meta["sampling_seconds"])
        s = {
            "model": model, "sampler": "numpyro_M2_production", "seed": meta["config"]["seed"], "like": "prod",
            "opts": {"metric": meta["config"]["metric"], "adapt_mass_matrix": meta["config"]["adapt_mass_matrix"],
                     "dense_mass": meta["config"]["dense_mass"]},
            "warmup": meta["config"]["num_warmup"], "samples": meta["n_samples_per_chain"],
            "chains": X.shape[0], "draws_per_chain": X.shape[1], "grads_sampling": grads,
            "grads_warmup": meta["warmup_grad_evals"], "grads_per_draw_mean": float(np.mean(r["num_steps"])),
            "t_warm": meta["warmup_seconds"], "t_samp": t, "t_compile": float("nan"),
            "accept_mean": float(np.mean(r["accept_prob"])), "divergences": int(np.sum(r["diverging"])),
            "ess_bulk_common": float(bulk[jc]), "ess_tail_common": float(tail[jc]),
            "ess_bulk_min": float(bulk.min()), "ess_tail_min": float(tail.min()),
            "ess_bulk_min_param": names[int(bulk.argmin())], "ess_tail_min_param": names[int(tail.argmin())],
            "ess_bulk_median": float(np.median(bulk)), "rhat_max": float(rh.max()),
            "rhat_max_param": names[int(rh.argmax())], "rhat_common": float(rh[jc]), "group": "M2",
        }
        for k in ("ess_bulk_common", "ess_tail_common", "ess_bulk_min", "ess_tail_min"):
            s[k.replace("ess", "ess_per_s")] = s[k] / t
            s[k.replace("ess", "ess_per_kgrad")] = 1e3 * s[k] / grads
        (OUT / f"m2prod_{name}.json").write_text(json.dumps(s, indent=1, default=float))
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items() if "per" in k or "rhat_max" == k})


if __name__ == "__main__":
    main()
