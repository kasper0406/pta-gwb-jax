"""M2 figures (ours vs released): Fig. 1a (HD free spectrum), 1b (HD gamma-A), 1c (binned
correlations), Fig. 4 (noise-marginalised OS S/N). Writes docs/figures/m2_*.png.

    uv run --no-sync python scripts/m2_figures.py

Needs outputs/m2/optstat.json + optstat_draws.npz (scripts/m2_optstat.py) and the runs.
Colours: ours = categorical slot 1 (blue), released = slot 2 (orange); ink in neutral grey.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from m2_common import ROOT, released, run_draws
from scipy.stats import gaussian_kde

from ptagwb.basis import FYR
from ptagwb.optstat import hd_curve
from ptagwb.sampling import load_run

OURS, REL, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
T_SPAN = 505861299.1401644
FIG = ROOT / "docs" / "figures"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 1.5})


def _hist_resample(x, lo=-9.0, hi=-4.0, n=400, size=20000, seed=0):
    """The Fig. 1(a) notebook's [-9, -4] histogram truncation + resampling."""
    rng = np.random.default_rng(seed)
    h, e = np.histogram(x, np.linspace(lo, hi, n), density=True)
    if h.sum() == 0:
        return np.array([])
    p = h * np.diff(e)
    p /= p.sum()
    i = rng.choice(len(h), size=size, p=p)
    return e[i] + rng.uniform(0, 1, size) * np.diff(e)[i]


def fig1a():
    try:
        run = load_run("hd_fs30")
    except FileNotFoundError:
        print("hd_fs30 missing; Fig. 1a skipped")
        return
    rel = released("hd_fs30")
    f = np.arange(1, 31) / T_SPAN
    lf = np.log10(f)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k in range(30):
        name = f"gw_log10_rho_{k}"
        for x, col, side in ((run_draws(run, name).ravel(), OURS, -1), (rel[name], REL, 1)):
            s = _hist_resample(x)
            if len(s) < 100:
                continue
            kde = gaussian_kde(s, bw_method=0.15)
            y = np.linspace(-9, -4, 300)
            d = kde(y)
            d = d / d.max() * 0.022
            ax.fill_betweenx(y, lf[k], lf[k] + side * d, color=col, alpha=0.35, lw=0)
            ax.plot(lf[k] + side * np.where(d > 0.01 * d.max(), d, np.nan), y, color=col, lw=0.8)
    # power-law (HD^gamma) posterior bands, 5-95%, as the notebook's "bayesogram"
    def band(g, a, col):
        rng = np.random.default_rng(1)
        i = rng.integers(0, len(g), 4000)
        pl = 10 ** (2 * a[i, None]) / (12 * np.pi**2) * FYR ** (g[i, None] - 3) * f[None, :] ** (-g[i, None]) / T_SPAN
        lo, hi = np.percentile(0.5 * np.log10(pl), [5, 95], axis=0)
        ax.fill_between(lf, lo, hi, color=col, alpha=0.12, lw=0)

    try:
        hv = load_run("hd_vg_14f")
        band(run_draws(hv, "gw_gamma").ravel(), run_draws(hv, "gw_log10_A").ravel(), OURS)
    except FileNotFoundError:
        pass
    rh = released("hd_vg_hm")
    band(rh["gw_gamma"], rh["gw_log10_A"], REL)
    # median HD^13/3 power law of the paper (2.4e-15), as in the notebook
    spec = 2.4e-15 * (f / FYR) ** (-2.0 / 3.0)
    spec = spec**2 / 12.0 / np.pi**2 / f**3 / T_SPAN
    ax.plot(lf, 0.5 * np.log10(spec), color=INK, ls="--", lw=1.0, label=r"$A=2.4\times10^{-15}$, $\gamma=13/3$")
    ax.plot([], [], color=OURS, lw=4, alpha=0.5, label="ours (HD free spectrum, left halves; HD$^\\gamma$ 90% band)")
    ax.plot([], [], color=REL, lw=4, alpha=0.5, label="released (right halves; 14f_PL_hd_crn HD band)")
    ax.set_xlim(-8.8, -7.55)
    ax.set_ylim(-8.9, -5.5)
    ax.set_xlabel(r"$\log_{10}$(frequency [Hz])")
    ax.set_ylabel(r"$\log_{10}\rho$ [s] (excess timing delay)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Fig. 1a: HD free spectrum, 30 modes ([-9,-4] truncation as in the release)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "m2_fig1a_freespec.png", dpi=170)
    plt.close(fig)


def _contour_levels(H, fracs=(0.393, 0.865, 0.989)):
    s = np.sort(H.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    return sorted(s[np.searchsorted(c, f)] for f in fracs)


def _contour(ax, g, a, col, ls="-", rng=((2, 4.6), (-14.8, -13.7))):
    H, xe, ye = np.histogram2d(g, a, bins=50, range=rng, density=True)
    from scipy.ndimage import gaussian_filter

    H = gaussian_filter(H, 1.2)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, H.T, levels=_contour_levels(H), colors=col, linestyles=ls, linewidths=1.3)


def fig1b():
    try:
        vg = load_run("hd_vg_14f")
        fg = load_run("hd_g433_14f")
    except FileNotFoundError:
        print("HD runs missing; Fig. 1b skipped")
        return
    rv, rf = released("hd_vg"), released("hd_g433")
    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_axes([0.12, 0.13, 0.55, 0.62])
    axt = fig.add_axes([0.12, 0.77, 0.55, 0.18], sharex=ax)
    axr = fig.add_axes([0.69, 0.13, 0.25, 0.62], sharey=ax)
    g, a = run_draws(vg, "gw_gamma").ravel(), run_draws(vg, "gw_log10_A").ravel()
    _contour(ax, rv["gw_gamma"], rv["gw_log10_A"], REL)
    _contour(ax, g, a, OURS)
    ax.axvline(13 / 3, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\log_{10}A$ ($f_\mathrm{ref}=1/\mathrm{yr}$)")
    bins_g, bins_a = np.linspace(2, 4.6, 50), np.linspace(-14.8, -13.7, 50)
    axt.hist(rv["gw_gamma"], bins_g, density=True, histtype="step", color=REL)
    axt.hist(g, bins_g, density=True, histtype="step", color=OURS)
    axr.hist(rv["gw_log10_A"], bins_a, density=True, histtype="step", color=REL, orientation="horizontal")
    axr.hist(a, bins_a, density=True, histtype="step", color=OURS, orientation="horizontal")
    axr.hist(rf["gw_log10_A"], 60, density=True, histtype="step", ls="--", color=REL, orientation="horizontal")
    axr.hist(run_draws(fg, "gw_log10_A").ravel(), 60, density=True, histtype="step", ls="--", color=OURS,
             orientation="horizontal")
    for axx in (axt, axr):
        axx.tick_params(labelbottom=False, labelleft=False)
        axx.set_yticks([]) if axx is axt else axx.set_xticks([])
    axr.set_ylim(-14.95, -13.7)
    ax.set_xlim(2, 4.6)
    ax.plot([], [], color=OURS, label=r"ours HD$^\gamma$ (solid), HD$^{13/3}$ (dashed, right)")
    ax.plot([], [], color=REL, label="released chains")
    ax.legend(loc="lower left", fontsize=7.5)
    axt.set_title(r"Fig. 1b: HD $\gamma$-$A$ (1,2,3$\sigma$ contours)", fontsize=9, color=INK)
    fig.savefig(FIG / "m2_fig1b_gamma_A.png", dpi=170)
    plt.close(fig)


def fig1c(opt):
    b = opt.get("ours_binned_map_enterprise")
    if b is None:
        print("optstat binned results missing; Fig. 1c skipped")
        return
    br = opt["ours_binned_at_released_ml_enterprise"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xx = np.linspace(1e-3, np.pi, 400)
    ax.plot(np.rad2deg(xx), hd_curve(xx), color=INK, ls="--", lw=1.0, label="Hellings-Downs")
    ax.axhline(0, color=MUTED, lw=0.6)
    xd = np.rad2deg(np.array(b["xi_mean"]))
    ax.errorbar(xd - 1.2, b["rho_bin_norm"], b["sig_bin_norm"], fmt="o", ms=4, color=OURS, capsize=3,
                label=fr"ours, MAP CURN$^{{13/3}}$ of our chain ($\chi^2$={b['chi2']:.1f})")
    ax.errorbar(np.rad2deg(np.array(br["xi_mean"])) + 1.2, br["rho_bin_norm"], br["sig_bin_norm"], fmt="s", ms=4,
                color=REL, capsize=3, label=fr"ours at the released ML noise vector ($\chi^2$={br['chi2']:.1f})")
    ax.set_xlabel(r"angular separation $\xi_{ab}$ [deg]")
    ax.set_ylabel(r"$\Gamma(\xi_{ab})$ (binned, normalised by $A^2_\mathrm{CURN}$)")
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Fig. 1c: 15-bin pair-covariance-aware correlations (paper: chi2 = 8.1)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "m2_fig1c_correlations.png", dpi=170)
    plt.close(fig)


def fig4(draws):
    if "ours_snr_vg_enterprise" not in draws:
        print("OS draws missing; Fig. 4 skipped")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bins = np.linspace(0, 10, 41)
    for key, col, lab in (("ours_snr_vg_enterprise", OURS, r"ours, varied $\gamma$"),
                          ("released_snr_vg", REL, r"released, varied $\gamma$")):
        ax.hist(draws[key], bins, density=True, histtype="step", color=col, label=lab)
        ax.axvline(np.mean(draws[key]), color=col, lw=1.0)
    for key, col, lab in (("ours_snr_g433_enterprise", OURS, r"ours, $\gamma=13/3$"),
                          ("released_snr_g433", REL, r"released, $\gamma=13/3$")):
        ax.hist(draws[key], bins, density=True, histtype="step", color=col, ls="--", label=lab)
        ax.axvline(np.mean(draws[key]), color=col, lw=1.0, ls="--")
    ax.set_xlabel("noise-marginalised HD optimal-statistic S/N")
    ax.set_ylabel("PDF")
    ax.legend(fontsize=7.5)
    ax.set_title("Fig. 4: OS S/N over CURN posterior draws (vertical lines: means)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "m2_fig4_os_snr.png", dpi=170)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    opt_path = ROOT / "outputs" / "m2" / "optstat.json"
    opt = json.loads(opt_path.read_text()) if opt_path.exists() else {}
    dpath = ROOT / "outputs" / "m2" / "optstat_draws.npz"
    draws = dict(np.load(dpath)) if dpath.exists() else {}
    fig1a()
    fig1b()
    fig1c(opt)
    fig4(draws)


if __name__ == "__main__":
    sys.exit(main())
