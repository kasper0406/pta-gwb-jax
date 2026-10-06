"""Smoke test: load every NG15 narrowband pulsar with PINT and summarise it.

For each pulsar (par + tim from the NG15 release, narrowband/par and narrowband/tim) we
  * load model + TOAs with PINT, honouring the par-file settings (EPHEM DE440,
    CLOCK TT(BIPM2019), planet Shapiro delay on),
  * compute pre-fit residuals and their weighted RMS (with raw TOA errors and with the
    EFAC/EQUAD-scaled errors from the par-file noise model),
  * build the timing-model design matrix and report its shape.

Usage:
    uv run python scripts/smoke_load.py                       # 68 primary pulsars, latest release
    uv run python scripts/smoke_load.py --release ng15_v1.0.1 # the June-2023 release
    uv run python scripts/smoke_load.py --include-split       # also *ao / *gbt split-telescope files
    uv run python scripts/smoke_load.py --clock ng15          # use the release's own clock files
    uv run python scripts/smoke_load.py --only J1909-3744 B1855+09

Writes a CSV summary to data/processed/smoke_load_<release>.csv (git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLIT_RE = re.compile(r"(ao|gbt)$")  # split-telescope variants, e.g. B1937+21ao
MIN_SPAN_YR = 3.0  # NG15 GWB analysis used pulsars with >= 3 yr baseline (67 of 68)


def release_dir(release: str) -> Path:
    base = RAW / release / "extracted"
    cands = sorted(base.glob("NANOGrav*"))
    if not cands:
        raise SystemExit(f"no extracted NG15 release under {base}; run scripts/fetch_data.py first")
    return cands[0]


def find_pairs(
    rel: Path, include_split: bool, only: list[str] | None
) -> list[tuple[str, Path, Path]]:
    pars = sorted((rel / "narrowband" / "par").glob("*.nb.par"))
    out = []
    for par in pars:
        name = par.name.split("_PINT_")[0]
        if SPLIT_RE.search(name) and not include_split:
            continue
        if only and name not in only:
            continue
        tims = sorted((rel / "narrowband" / "tim").glob(f"{name}_PINT_*.nb.tim"))
        if len(tims) != 1:
            raise SystemExit(f"{name}: expected exactly one tim file, found {tims}")
        out.append((name, par, tims[0]))
    return out


def load_one(name: str, par: Path, tim: Path) -> dict:
    import warnings

    import astropy.units as u
    import numpy as np
    import pint.logging
    from pint.models import get_model_and_toas
    from pint.residuals import Residuals

    pint.logging.setup(level="ERROR")
    warnings.filterwarnings("ignore")
    t0 = time.time()
    row: dict = {"pulsar": name}
    try:
        # get_model_and_toas reads EPHEM/CLOCK/PLANET_SHAPIRO from the par file, so the TOAs
        # are computed with DE440 and TT(BIPM2019), exactly as in the NG15 release.
        model, toas = get_model_and_toas(str(par), str(tim), planets=True)
        assert model.EPHEM.value == "DE440", model.EPHEM.value
        assert model.CLOCK.value == "TT(BIPM2019)", model.CLOCK.value
        res = Residuals(toas, model)  # pre-fit residuals (par file is already post-fit)
        r = res.time_resids.to_value(u.s)
        err_raw = toas.get_errors().to_value(u.s)
        err_scaled = model.scaled_toa_uncertainty(toas).to_value(u.s)

        def wrms(err):
            w = 1.0 / err**2
            mean = np.sum(w * r) / np.sum(w)
            return np.sqrt(np.sum(w * (r - mean) ** 2) / np.sum(w))

        M, params, _units = model.designmatrix(toas)
        mjd = toas.get_mjds().value
        span_yr = (mjd.max() - mjd.min()) / 365.25
        flags = toas.get_flag_value("f")[0]
        row.update(
            ok=True,
            ntoa=toas.ntoas,
            span_yr=round(span_yr, 3),
            mjd_min=float(mjd.min()),
            mjd_max=float(mjd.max()),
            wrms_raw_us=wrms(err_raw) * 1e6,
            wrms_scaled_us=wrms(err_scaled) * 1e6,
            chi2_red=float(res.chi2_reduced),
            M_rows=M.shape[0],
            M_cols=M.shape[1],
            n_dmx=sum(p.startswith("DMX_") for p in params),
            n_backends=len(set(flags)),
            freq_min_mhz=float(toas.get_freqs().to_value(u.MHz).min()),
            freq_max_mhz=float(toas.get_freqs().to_value(u.MHz).max()),
            binary=model.BINARY.value if "BINARY" in model.params else "",
            has_rn="PLRedNoise" in model.components,
            seconds=round(time.time() - t0, 1),
        )
    except Exception as e:  # noqa: BLE001
        row.update(
            ok=False,
            error=f"{type(e).__name__}: {e}",
            tb=traceback.format_exc(limit=3),
            seconds=round(time.time() - t0, 1),
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--release", default="ng15_v2.1.0", help="data/raw/<release> to use")
    ap.add_argument("--include-split", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument(
        "--clock",
        choices=["pint", "ng15"],
        default="pint",
        help="'pint': PINT's global clock repository (downloaded on demand; the "
        "release README recommends this); 'ng15': the clock/ dir shipped in the release",
    )
    ap.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    rel = release_dir(args.release)
    if args.clock == "ng15":
        os.environ["PINT_CLOCK_OVERRIDE"] = str(rel / "clock")
    pairs = find_pairs(rel, args.include_split, args.only)
    print(f"release: {rel.relative_to(ROOT)}  pulsars: {len(pairs)}  clock: {args.clock}")

    # Warm-up in the parent process so that ephemeris / clock / IERS downloads happen once.
    print("warm-up (downloads DE440, clock files, IERS tables on first use) ...", flush=True)
    warm = load_one(*min(pairs, key=lambda p: p[2].stat().st_size))
    if not warm.get("ok"):
        print(warm.get("tb"))
        raise SystemExit(f"warm-up failed: {warm.get('error')}")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(load_one, *p): p[0] for p in pairs}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            status = "ok " if row.get("ok") else "FAIL"
            print(
                f"  [{len(rows):2d}/{len(pairs)}] {status} {row['pulsar']:14s} {row['seconds']:6.1f}s",
                flush=True,
            )
    rows.sort(key=lambda r: r["pulsar"])

    hdr = (
        f"{'pulsar':14s} {'ntoa':>6s} {'span':>6s} {'wrms_raw':>9s} {'wrms_sc':>8s} "
        f"{'chi2r':>6s} {'M shape':>13s} {'nDMX':>5s} {'nBE':>4s} {'fmin':>6s} {'fmax':>6s} "
        f"{'binary':>6s} {'RN':>3s}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r.get("ok"):
            print(f"{r['pulsar']:14s} FAILED: {r['error']}")
            continue
        print(
            f"{r['pulsar']:14s} {r['ntoa']:6d} {r['span_yr']:6.2f} {r['wrms_raw_us']:9.3f} "
            f"{r['wrms_scaled_us']:8.3f} {r['chi2_red']:6.2f} "
            f"{'(' + str(r['M_rows']) + ',' + str(r['M_cols']) + ')':>13s} {r['n_dmx']:5d} "
            f"{r['n_backends']:4d} {r['freq_min_mhz']:6.0f} {r['freq_max_mhz']:6.0f} "
            f"{r['binary'] or '-':>6s} {'Y' if r['has_rn'] else '-':>3s}"
        )
    print("(wrms in microseconds: raw = TOA errors as in .tim; sc = EFAC/EQUAD-scaled; span in yr)")

    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    primary = [r for r in ok if not SPLIT_RE.search(r["pulsar"])]
    gwb = [r for r in primary if r["span_yr"] >= MIN_SPAN_YR]
    short = [r["pulsar"] for r in primary if r["span_yr"] < MIN_SPAN_YR]
    print(
        f"\nloaded {len(ok)}/{len(rows)} ({len(primary)} primary pulsars); "
        f"{len(gwb)} primary pulsars with span >= {MIN_SPAN_YR} yr; short: {short}"
    )
    print(f"total TOAs (primary): {sum(r['ntoa'] for r in primary)}")
    if gwb:
        t_all = (max(r["mjd_max"] for r in gwb) - min(r["mjd_min"] for r in gwb)) / 365.25
        print(f"Tspan of the >= {MIN_SPAN_YR} yr set: {t_all:.3f} yr (paper: 16.03 yr)")
    if bad:
        print("FAILURES:")
        for r in bad:
            print(f"  {r['pulsar']}: {r['error']}\n{r.get('tb', '')}")

    out = ROOT / "data" / "processed" / f"smoke_load_{args.release}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r if k != "tb"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
