"""Dump what tempo2 (via libstempo) makes of a par/tim pair, as JSON. Runs in the isolated
tempo2 oracle env: ``scripts/t2py scripts/t2_dump.py PAR TIM OUT.npz [--design] [--fit]``.

Saved arrays (npz): stoas_day / stoas_sec (site arrival times incl. TIME/-to offsets, split in
day + seconds for long-double exactness), freqs [MHz], toaerrs [us], telescope, flags (JSON per
TOA: libstempo flag values, last occurrence wins), residuals [s] (pre-fit, tempo2's own
weighted-mean subtraction), and with --design the design matrix and parameter names
(libstempo ``designmatrix(fixunits=True)``: d residual / d parameter in SI-ish units, columns
['Offset'] + fit parameters), plus tempo2 version info.
"""

import json
import sys

import numpy as np


def main():
    par, tim, out = sys.argv[1:4]
    design = "--design" in sys.argv
    import libstempo as T

    psr = T.tempopulsar(parfile=par, timfile=tim, dofit=False, maxobs=2_000_000)
    st = np.asarray(psr.stoas, dtype=np.longdouble)
    day = np.floor(st)
    sec = (st - day) * np.longdouble(86400)
    names = list(psr.flags())
    fl = {f: np.asarray(psr.flagvals(f)).astype(str).tolist() for f in names}
    res = {
        "stoas_day": day.astype(np.int64),
        "stoas_sec": sec.astype(np.float64),
        "freqs": np.asarray(psr.freqs, dtype=np.float64),
        "toaerrs": np.asarray(psr.toaerrs, dtype=np.float64),
        "residuals": np.asarray(psr.residuals(updatebats=True, formresiduals=True), dtype=np.float64),
        "deleted": np.asarray(psr.deleted, dtype=bool),
        "flags_json": np.array(json.dumps(fl)),
        "meta": np.array(json.dumps({"libstempo": T.__version__, "tempo2": str(T.libstempo.tempo2version()),
                                     "nobs": int(psr.nobs), "name": psr.name,
                                     "units": str(getattr(psr, "units", "")),
                                     "binary": str(psr.binarymodel) if hasattr(psr, "binarymodel") else ""})),
    }
    if design:
        M = np.asarray(psr.designmatrix(fixunits=True), dtype=np.float64)
        res["design"] = M
        res["design_params"] = np.array(["Offset"] + list(psr.pars(which="fit")))
    np.savez(out, **res)


if __name__ == "__main__":
    main()
