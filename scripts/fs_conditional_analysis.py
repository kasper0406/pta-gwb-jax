"""Which coordinates control the conditional low-power mass of a free-spectrum bin? (docs/FS_PILOT.md Sec. 14)

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_conditional_analysis.py --validate
    JAX_PLATFORMS=cpu uv run --no-sync python scripts/fs_conditional_analysis.py --run hd_fs30_hybrid_pilot --bin 2

P(log10_rho_k < -10 | all other parameters) is computed by quadrature of the exact conditional
(``ptagwb.conditional``; uniform prior on [-15.5, -1]) on a uniform grid (``--n-grid``, checked against
4x refinement). States are saved draws of ``--run`` at fixed indices (chain c, draw n listed in the
output). Controlled coordinate changes: at the state with the largest conditional low mass S* and at
reference states T (the three states with median conditional mass), every other group (one bin, or
one pulsar's IRN (log10_A, gamma) pair) is swapped between S* and T, one at a time, and the change
in log-odds of the conditional low mass is reported; then the top groups are inserted cumulatively
into T. Writes outputs/m2/fs_conditional_<run>_bin<k>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from common import get_terms
from m2_common import ROOT, save_json

from ptagwb.conditional import make_bin_conditional
from ptagwb.sampling import ModelSpec, load_run, make_likelihood, unpack

LO, HI, LOWCUT = -15.5, -1.0, -10.0


def setup(k):
    spec = ModelSpec(orf="hd", common="freespec", n_common=30, position="enterprise")
    terms, T = get_terms()
    like = make_likelihood(terms, T, spec, "production")
    build, cond = make_bin_conditional(like, k)
    return spec, like, jax.jit(build), jax.jit(jax.vmap(cond, in_axes=(None, 0)))


def p_low(cache, condv, n_grid):
    g = np.linspace(LO, HI, n_grid)
    ll = np.asarray(condv(cache, jnp.asarray(g)))
    w = np.full(n_grid, g[1] - g[0])
    w[[0, -1]] *= 0.5
    lw = ll + np.log(w)
    lw -= lw.max()
    p = np.exp(lw)
    return float(p[g < LOWCUT].sum() / p.sum())


def logodds(p):
    p = min(max(p, 1e-300), 1 - 1e-16)
    return float(np.log(p) - np.log1p(-p))


def validate(spec, like, bins=(0, 2, 3, 7), n_states=4, seed=0):
    """Real-data check of the conditional vs the production likelihood: hybrid-pilot draws, the
    exceptional state, prior draws z ~ U(-4, 4)."""
    rng = np.random.default_rng(seed)
    r = load_run("hd_fs30_hybrid_pilot")
    X = r["x"]
    lo, hi = np.array(r["meta"]["lo"]), np.array(r["meta"]["hi"])
    states = [("hybrid c6 n99", X[6, 99])] + [(f"hybrid c{c} n{n}", X[c, n]) for c, n in ((0, 50), (3, 20), (7, 80))]
    states += [(f"prior {i}", lo + (hi - lo) / (1 + np.exp(-rng.uniform(-4, 4, len(lo))))) for i in range(n_states)]
    grid = np.linspace(-15.49, -1.01, 13)
    out = []
    for k in bins:
        build, cond = make_bin_conditional(like, k)
        build, condv = jax.jit(build), jax.jit(jax.vmap(cond, in_axes=(None, 0)))
        j = 2 * like.P + k
        for lab, x in states:
            cache = build(unpack(jnp.asarray(x), spec, like.P))
            c = np.asarray(condv(cache, jnp.asarray(grid)))
            full = []
            for gv in grid:
                x2 = np.array(x, float)
                x2[j] = gv
                full.append(float(like.logL(unpack(jnp.asarray(x2), spec, like.P))) + 0.5 * like.const_total)
            full = np.array(full)
            d = (c - c[0]) - (full - full[0])
            out.append({"bin": k, "state": lab, "max_abs_diff": float(np.abs(d).max()), "logL_range": float(np.ptp(full)),
                        "max_rho_other": float(np.max(np.delete(x[2 * like.P:], k)))})
            print(f"bin f_{k + 1} {lab:16s} max |cond - prod| {np.abs(d).max():.2e} (logL range {np.ptp(full):.0f}, "
                  f"max other log10_rho {out[-1]['max_rho_other']:.2f})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hd_fs30_hybrid_pilot")
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--n-grid", type=int, default=241)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    k = args.bin
    spec, like, build, condv = setup(k)
    if args.validate:
        save_json(validate(spec, like), ROOT / "outputs" / "m2" / "fs_conditional_validate.json")
        return
    r = load_run(args.run)
    X, names = r["x"], r["names"]
    C, N, _D = X.shape
    P = like.P
    idx = [(c, int(n)) for c in range(C) for n in np.linspace(N // 5, N - 1, 5).astype(int)]
    t0 = time.time()

    def pl(x, n_grid=args.n_grid):
        return p_low(build(unpack(jnp.asarray(x), spec, P)), condv, n_grid)

    ps = np.array([pl(X[c, n]) for c, n in idx])
    t_state = (time.time() - t0) / len(idx)
    order = np.argsort(-ps)
    share = ps[order[0]] / ps.sum() if ps.sum() > 0 else float("nan")
    print(f"{args.run} bin f_{k + 1}: {len(idx)} states (chain, draw) = {idx}")
    print(f"  mean P(low | rest) = {ps.mean():.4g}; largest {ps[order[0]]:.4g} at {idx[order[0]]} "
          f"({100 * share:.1f}% of the sum); next {ps[order[1]]:.3g}, {ps[order[2]]:.3g}; median {np.median(ps):.3g}  "
          f"[{t_state:.1f} s per state]", flush=True)
    s_idx = idx[order[0]]
    xs = X[s_idx].copy()
    quad = {"n_grid": args.n_grid, "p": pl(xs), "p_4x": pl(xs, 4 * args.n_grid - 3)}
    print(f"  quadrature check at S*: {quad['p']:.8f} vs {quad['p_4x']:.8f} (4x refined)")
    med = np.argsort(np.abs(np.log(np.maximum(ps, 1e-300)) - np.log(max(np.median(ps), 1e-300))))[:3]
    t_states = [idx[i] for i in med if idx[i] != s_idx][:3]
    groups = [(f"bin f_{i + 1}", [2 * P + i]) for i in range(30) if i != k]
    groups += [(names[a][: -len("_red_noise_log10_A")], [a, P + a]) for a in range(P)]
    res = {"run": args.run, "bin": k, "states": idx, "p_low": ps.tolist(), "s_star": s_idx, "share_of_largest": share,
           "quadrature": quad, "t_states": t_states, "swaps": {}}
    lo_s = logodds(quad["p"])
    for tix in t_states:
        xt = X[tix].copy()
        pt = pl(xt)
        lo_t = logodds(pt)
        rows = []
        for lab, js in groups:
            a = xs.copy()
            a[js] = xt[js]
            b = xt.copy()
            b[js] = xs[js]
            rows.append({"group": lab, "drop_from_S": lo_s - logodds(pl(a)), "rise_into_T": logodds(pl(b)) - lo_t})
        rows.sort(key=lambda r_: -(abs(r_["drop_from_S"]) + abs(r_["rise_into_T"])))
        # cumulative insertion of the top groups into T
        cum, b = [], xt.copy()
        for r_ in rows[: args.top]:
            js = dict(groups)[r_["group"]]
            b[js] = xs[js]
            cum.append({"group": r_["group"], "p": pl(b)})
        res["swaps"][str(tix)] = {"p_T": pt, "rows": rows, "cumulative": cum}
        print(f"\n  S* {s_idx} (P {quad['p']:.3g}, log-odds {lo_s:.2f}) vs T {tix} (P {pt:.3g}, log-odds {lo_t:.2f}); "
              f"total log-odds gap {lo_s - lo_t:.2f}")
        print("    group | log-odds drop when S* takes T's value | rise when T takes S*'s value")
        for r_ in rows[: args.top]:
            print(f"    {r_['group']:14s} | {r_['drop_from_S']:+7.2f} | {r_['rise_into_T']:+7.2f}")
        print("    cumulative insertion into T: " + ", ".join(f"{c_['group']} -> {c_['p']:.3g}" for c_ in cum), flush=True)
    # the reviewer's direct check: f_4 alone
    for v in (-14.78, -7.5):
        a = xs.copy()
        a[2 * P + 3] = v
        res.setdefault("f4_check", {})[str(v)] = pl(a)
    print(f"\n  S* with f_4 set to -14.78 / -7.5: P = {res['f4_check']['-14.78']:.4f} / {res['f4_check']['-7.5']:.4f}")
    save_json(res, ROOT / "outputs" / "m2" / f"fs_conditional_{args.run}_bin{k}.json")


if __name__ == "__main__":
    main()
