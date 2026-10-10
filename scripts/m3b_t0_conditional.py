"""Third N13 validation item (docs/M3B_PLAN.md Sec. 5.1): the EPTA J1713+0747 conditional
t0 | all other parameters, on a fine grid, against the exact MH kernel's conditional draws.

All other parameters are fixed at released CURN draws (3 draws: the retained chain's first,
middle and last rows). In CURN the likelihood is a sum over pulsars, so the conditional of t0
only involves J1713+0747's term: a one-pulsar ``GeneralPTALikelihood`` with the dip update of
``ptagwb.epta`` (checked against the full model at the draws). Grid: 8,001 points on the window
(spacing 0.005 d) plus every TOA epoch inside the window (the likelihood is discontinuous there).
Kernel: ``eventmh.mh_t0`` (uniform 0.2 + random walk 2 d), 4 chains x 20,000 updates.

Criteria (fixed before running): occupancies of the inter-TOA intervals with >= 1 % grid mass and
of the rest, and the 5/50/95 % quantiles of t0, within 3.5 MC standard errors of the grid values.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/m3b_t0_conditional.py
Writes data/processed/m3b/epta/results/t0_conditional.json.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from ptagwb import acceptance as acc
from ptagwb import combined as C
from ptagwb import epta
from ptagwb.binding import evidence_binding
from ptagwb.config import REPO_ROOT
from ptagwb.diagnostics import mcse_quantile
from ptagwb.eventmh import T0Proposal, mh_t0

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"


def one_pulsar_logL(M: epta.EPTAModel):
    j = M.dip.index
    like1 = C.GeneralPTALikelihood([M.terms[j]], orf="curn", common="powerlaw", convention="chain")
    name = M.psrs[j].name
    ix = {n: i for i, n in enumerate(M.param_names)}
    procs = [nm for nm in epta.PROCESSES if nm in like1.processes]

    def f(x):
        c, s = M.dip_terms(x)
        p = {f"{nm}_{k}": jnp.atleast_1d(x[ix[f"{name}_{nm}_{k}"]]) for nm in procs for k in ("log10_A", "gamma")}
        p["log10_A"], p["gamma"] = x[ix["gw_crn_log10_A"]], x[ix["gw_crn_gamma"]]
        return like1._logL_cs(p, c[None, :], s[None])

    return jax.jit(f)


def main():
    man = epta.load_manifest()
    psrs = epta.load_pulsars()
    M = epta.EPTAModel(psrs, man, "crn")
    names, X, burn = epta.load_reference("crn_pl", man)
    f = one_pulsar_logL(M)
    i0 = M.t0_index
    lo, hi = M.lo[i0], M.hi[i0]
    toas = psrs[M.dip.index].toas / 86400.0
    t_in = toas[(toas > lo) & (toas < hi)]
    out = {"draw_rows": [], "results": []}
    for row in (burn, (burn + len(X)) // 2, len(X) - 1):
        x = np.array(X[row, :67])
        # consistency of the one-pulsar conditional with the full model
        x2 = x.copy()
        x2[i0] = x[i0] + 3.0 if x[i0] + 3.0 < hi else x[i0] - 3.0
        dfull = float(M.logL(x2) - M.logL(x))
        done = float(f(jnp.asarray(x2)) - f(jnp.asarray(x)))
        grid = np.unique(np.concatenate([np.linspace(lo, hi, 8001), t_in, np.nextafter(t_in, -np.inf)]))
        G = jnp.asarray(np.repeat(x[None, :], grid.size, 0)).at[:, i0].set(jnp.asarray(grid))
        lg = np.asarray(jax.lax.map(f, G, batch_size=256))
        # piecewise-linear density between grid points (exact discontinuities are grid points)
        w = np.exp(lg - lg.max())
        mass = 0.5 * (w[1:] + w[:-1]) * np.diff(grid)
        cdf = np.concatenate([[0], np.cumsum(mass)])
        cdf /= cdf[-1]
        # kernel: MH-only chains on t0
        prop = T0Proposal(lo, hi, w_uniform=0.2, rw_scale=2.0)
        nch, n = 4, 20000

        def chain(key, t0start):
            xx = jnp.asarray(x).at[i0].set(t0start)

            def body(carry, k):
                xc, ll = carry
                xc, ll, ok = mh_t0(k, xc, ll, f, i0, prop)
                return (xc, ll), (xc[i0], ok)

            (_, _), (ts, oks) = jax.lax.scan(body, (xx, f(xx)), jax.random.split(key, n))
            return ts, oks

        keys = jax.random.split(jax.random.PRNGKey(row), nch)
        starts = jnp.asarray(np.random.default_rng(row).uniform(lo, hi, nch))
        ts, oks = jax.vmap(chain)(keys, starts)
        ts = np.asarray(ts)[:, 2000:]
        rng = np.random.default_rng(1)
        tref = np.interp(rng.random(200000), cdf, grid)
        ints = acc.event_intervals(toas, (lo, hi), tref)
        rows = []
        inside_k = np.zeros(ts.shape, bool)
        for a, b in ints:
            m_grid = np.interp(b, grid, cdf) - np.interp(a, grid, cdf)
            ind = (ts >= a) & ((ts < b) if b != hi else (ts <= b))
            inside_k |= ind
            o = acc.occupancy(list(ind))
            rows.append({"what": f"interval [{a:.4f}, {b:.4f})", "ours": o.p_hat, "grid": float(m_grid),
                         "se": o.mcse, "z": abs(o.p_hat - m_grid) / o.mcse if o.mcse else None, "case": o.case})
        for q in (0.05, 0.5, 0.95):
            qo, qg, se = float(np.quantile(ts, q)), float(np.interp(q, cdf, grid)), float(mcse_quantile(ts, q))
            rows.append({"what": f"q{q}", "ours": qo, "grid": qg, "se": se, "z": abs(qo - qg) / se})
        zs = [r["z"] for r in rows if r["z"] is not None]
        unres = [r["what"] for r in rows if r.get("case") in ("all-visit", "zero-visit", "few-event")]
        out["results"].append({"row": int(row), "dlnL_full": dfull, "dlnL_one_pulsar": done,
                               "n_intervals": len(ints), "mh_accept": float(np.mean(oks)), "z_max": max(zs),
                               "unresolved_intervals": unres, "rows": rows, "pass": bool(max(zs) < 3.5 and abs(dfull - done) < 1e-6)})
        print(row, out["results"][-1]["z_max"], len(ints), unres, dfull - done, flush=True)
    out["pass"] = all(r["pass"] for r in out["results"])
    out["binding"] = evidence_binding()
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "t0_conditional.json").write_text(json.dumps(out, indent=1))
    print("PASS" if out["pass"] else "FAIL")


if __name__ == "__main__":
    main()
