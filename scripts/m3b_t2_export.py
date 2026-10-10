"""N2: tempo2 evaluator export of every array the EPTA model consumes (docs/M3B_PLAN.md Sec. 4.1,
evaluator profile ``published-tempo2-v1``). Runs in the plain tempo2 oracle env, independent of
enterprise: ``TEMPO2_OVERRIDE=<runtime> scripts/t2py scripts/m3b_t2_export.py PAR TIM OUT.npz``.

Conventions (chosen to equal what enterprise's ``Tempo2Pulsar`` hands to the likelihood; gate T1
compares every array with the pinned enterprise built from the released and from our canonical
files):

* rows: all ``nobs`` observations libstempo returns (tempo2 deletions are *not* dropped, as in
  enterprise; the deletion mask is exported), stably sorted by barycentric TOA (mergesort);
* ``toas``: barycentric arrival times, ``float64(psr.toas()) * 86400`` [s] (TDB/TCB as the par:
  the time scale of the par's UNITS, here TCB), ``stoas`` site arrival times [s];
* ``residuals``: tempo2 pre-fit residuals at the par values, ``psr.residuals()`` (libstempo
  defaults: BATs updated, residuals formed, weighted mean removed) [s];
* ``toaerrs`` [s]; ``freqs``: **barycentric** radio frequencies ``psr.ssbfreqs() / 1e6`` [MHz]
  (enterprise's convention; site frequencies are exported as ``site_freqs``);
* ``Mmat``: ``psr.designmatrix()`` (libstempo defaults: offset column, fixed units and signs);
  ``fitpars`` = ['Offset'] + fit parameters; numerical rank from the singular values of the
  column-normalised matrix;
* ``backend_flags``: the first non-empty of -group, -g, -sys, -i, -f, then -fe_-be per TOA;
* ``pos``: unit vector from RAJ/DECJ at POSEPOCH, or for ecliptic pars from ELONG/ELAT through
  pyephem ``Equatorial(Ecliptic(elong, elat), epoch='2000')`` (enterprise's convention);
* SW / planetary geometry (not consumed by the EPTA DR2new model, exported for T1): ``pos_t`` =
  ``psr.psrPos`` (ecliptic -> equatorial rotation, obliquity 23.43704 deg, for ecliptic pars), and after setting
  DMASSPLANET1-9 = 0 and re-forming the BATs (enterprise's order), ``sunssb`` and ``planetssb``.
"""

import json
import os
import sys

import numpy as np

# ecliptic -> equatorial rotation with enterprise's obliquity convention (constants.e_ecl =
# 23.43704 deg; not the IERS 2010 value 84381.406 arcsec = 23.4392794 deg)
_EPS = 23.43704 * np.pi / 180.0
_ECL2EQ = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(_EPS), -np.sin(_EPS)], [0.0, np.sin(_EPS), np.cos(_EPS)]])


def backend_flags(fl, n):
    out = np.array([""] * n, dtype=object)
    if "fe" in fl and "be" in fl:
        out[:] = [(a + "_" + b if (a and b) else "") for a, b in zip(fl["fe"], fl["be"])]
    for f in ["f", "i", "sys", "g", "group"]:
        if f in fl:
            v = np.asarray(fl[f], dtype=object)
            out = np.where(v == "", out, v)
    return out.astype(str)


def main():
    par, tim, out = (os.path.abspath(x) for x in sys.argv[1:4])
    import ephem
    import libstempo as T
    import libstempo.libstempo as L

    with open(tim) as f:
        n_lines = sum(1 for _ in f)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(tim))
    psr = T.tempopulsar(parfile=os.path.relpath(par), timfile=os.path.basename(tim), maxobs=max(n_lines, 20000) + 100,
                        ephem="DE440", clk=None)
    os.chdir(cwd)
    toas = np.double(psr.toas()) * 86400
    stoas = np.double(psr.stoas) * 86400
    res = np.double(psr.residuals())
    err = np.double(psr.toaerrs) * 1e-6
    M = np.double(psr.designmatrix())
    ssbf = np.double(psr.ssbfreqs()) / 1e6
    sitef = np.double(psr.freqs)
    try:  # not consumed; libstempo 2.5.1's telescope() fails under numpy 2 ('a32' dtype)
        tel = np.char.decode(psr.telescope(), encoding="ascii").astype(str)
    except TypeError:
        tel = np.array([""] * len(toas))
    fit = ["Offset"] + [str(p) for p in psr.pars()]
    allpars = list(psr.pars(which="fit")) + list(psr.pars(which="set"))
    fl = {k: np.asarray(psr.flagvals(k)).astype(str) for k in psr.flags()}
    n = len(toas)
    if "RAJ" in allpars:
        raj, decj = float(psr["RAJ"].val), float(psr["DECJ"].val)
        conv = "RAJ/DECJ"
    else:
        eq = ephem.Equatorial(ephem.Ecliptic(psr["ELONG"].val, psr["ELAT"].val), epoch="2000")
        raj, decj = float(eq.ra), float(eq.dec)
        conv = "ELONG/ELAT via pyephem (epoch 2000)"
    pos = np.array([np.cos(raj) * np.cos(decj), np.sin(raj) * np.cos(decj), np.sin(decj)])
    ecl = "ELAT" in allpars
    pos_t = np.array(psr.psrPos, dtype=np.float64)
    if ecl:
        pos_t = pos_t @ _ECL2EQ.T
    for i in range(1, 10):
        psr[f"DMASSPLANET{i}"].val = 0.0
    psr.formbats()
    planets = np.zeros((n, 9, 6))
    for i, nm in enumerate(["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]):
        planets[:, i, :] = getattr(psr, f"{nm}_ssb")
    sun = np.array(psr.sun_ssb, dtype=np.float64)
    if ecl:
        for i in range(9):
            planets[:, i, :3] = planets[:, i, :3] @ _ECL2EQ.T
            planets[:, i, 3:] = planets[:, i, 3:] @ _ECL2EQ.T
        sun[:, :3] = sun[:, :3] @ _ECL2EQ.T
        sun[:, 3:] = sun[:, 3:] @ _ECL2EQ.T
    o = np.argsort(toas, kind="mergesort")
    Mn = M / np.linalg.norm(M, axis=0)
    sv = np.linalg.svd(Mn, compute_uv=False)
    np.savez(
        out, name=np.array(str(psr.name)), toas=toas[o], stoas=stoas[o], residuals=res[o], toaerrs=err[o],
        freqs=ssbf[o], site_freqs=sitef[o], Mmat=M[o], fitpars=np.array(fit), backend_flags=backend_flags(fl, n)[o],
        flags_json=np.array(json.dumps({k: v[o].tolist() for k, v in fl.items()})), telescope=tel[o],
        pos=pos, pos_t=pos_t[o], sunssb=sun[o], planetssb=planets[o], deleted=np.asarray(psr.deleted, dtype=bool)[o],
        nobs=np.array(int(psr.nobs)), order=o, mmat_sv=sv, mmat_rank=np.array(int(np.sum(sv > sv[0] * 1e-10))),
        meta=np.array(json.dumps({"libstempo": T.__version__, "tempo2": str(L.tempo2version()),
                                  "TEMPO2": os.environ.get("TEMPO2"), "position": conv, "par": par, "tim": tim,
                                  "units": str(getattr(psr, "units", ""))})))


if __name__ == "__main__":
    main()
