"""Ingest the NG15 narrowband pulsars with PINT into the on-disk cache (data/cache/pulsars).

Usage:
    uv run --no-sync python scripts/ingest.py                    # 67 GWB pulsars, release clock files
    uv run --no-sync python scripts/ingest.py --clock pint       # PINT's global clock repository
    uv run --no-sync python scripts/ingest.py --include-split    # also the *ao / *gbt files
"""

from __future__ import annotations

import argparse
import time

from ptagwb.data import JULIAN_YEAR_S, find_par_tim, get_tspan, load_pulsars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clock", choices=["release", "pint"], default="release")
    ap.add_argument("--include-split", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    psrs = load_pulsars(clock=args.clock, jobs=args.jobs)
    T = get_tspan(psrs)
    print(f"{len(psrs)} GWB pulsars, {sum(p.ntoa for p in psrs)} TOAs, "
          f"Tspan = {T!r} s = {T / JULIAN_YEAR_S:.10f} yr  ({time.time() - t0:.0f} s)")
    if args.include_split:
        split = list(find_par_tim(split_only=True))
        sp = load_pulsars(split, clock=args.clock, jobs=args.jobs)
        for p in sp:
            print(f"  split file {p.name}: {p.ntoa} TOAs, span {p.span_s / JULIAN_YEAR_S:.2f} yr")


if __name__ == "__main__":
    main()
