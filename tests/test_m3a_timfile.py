"""M3a ingestion semantics (gate G1): the tempo2-semantics tim reader against tempo2 itself
(libstempo, isolated env) on fixture trees exercising TIME scoping through INCLUDE, SKIP/NOSKIP
around TIME/INCLUDE/END, END in a child, comment forms, valueless flags and -to/-padd; the flat
canonical file read back by PINT; and the reader on the released data (counts, InPTA DR1 "CJ"
lines = Yu & Allen's 1,090,206 TOAs).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from m3a_oracles import T2PY, have_tempo2

from ptagwb.timfile import (
    TimSemanticsError,
    canonical_flags,
    read_tim,
    tempo2_flags,
    write_flat_tim,
)

ROOT = Path(__file__).resolve().parents[1]
PAR = """PSRJ J0000+0000
RAJ 06:00:00.0
DECJ 10:00:00.0
F0 100.0
PEPOCH 55000
DMEPOCH 55000
DM 10.0
EPHEM DE440
CLK TT(TAI)
UNITS TDB
"""

FIXTURES = {
    # TIME before INCLUDE: the child starts at 0, the parent's offset is restored afterwards;
    # TIME inside the child does not leak; SKIP region swallows TIME/INCLUDE/END; END in a child
    # ends only the child; 'CJ...' is a comment, ' C ...' is not (but unparseable)
    "scoping": {
        "main.tim": """FORMAT 1
toaA 1400.0 55000.1000000000000000 1.0 pks -f A
TIME 1.0
toaB 1400.0 55000.2000000000000000 1.0 pks -f A
INCLUDE child.tim
toaC 1400.0 55000.3000000000000000 1.0 pks -f A -to 0.25
CJ0437 1400.0 55000.4 1.0 pks -f A
 C commented 1400.0 55000.5 1.0 pks
TIME 0.5
toaD 1400.0 55000.6000000000000000 1.0 pks -projid -beconfig -group X -padd 0.01
SKIP
toaE 1400.0 55000.7 1.0 pks -f A
TIME 100
INCLUDE child.tim
END
NOSKIP
toaF 1400.0 55000.8000000000000000 1.0 pks -f A
""",
        "child.tim": """FORMAT 1
toaG 1400.0 55001.1000000000000000 1.0 pks -f B
TIME 5.0
toaH 1400.0 55001.2000000000000000 1.0 pks -f B
END
toaI 1400.0 55001.3 1.0 pks -f B
""",
    },
    # nested INCLUDEs from a FORMAT-less top file (EPTA layout), TIME in the grandchild, a SKIP left
    # open in a child does not leak into the parent, flag artifact '-.cal' inside an archive name
    "nested": {
        "main.tim": """INCLUDE a/child.tim
INCLUDE a/child2.tim
""",
        "a/child.tim": """FORMAT 1
TIME -0.002
x-.cal 1400.0 55100.1000000000000000 2.0 pks -f A
INCLUDE grand.tim
x 1400.0 55100.2000000000000000 2.0 pks -f A
SKIP
""",
        "a/grand.tim": """FORMAT 1
y 1400.0 55100.3000000000000000 2.0 pks -f G
TIME 3.0
y 1400.0 55100.4000000000000000 2.0 pks -f G
""",
        "a/child2.tim": """FORMAT 1
z 1400.0 55100.5000000000000000 2.0 pks -f C
""",
    },
}


def _write(tmp: Path, files: dict) -> Path:
    for name, text in files.items():
        f = tmp / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    (tmp / "fx.par").write_text(PAR)
    return tmp / "main.tim"


def _tempo2(par: Path, tim: Path) -> dict:
    out = tim.parent / "t2.npz"
    p = subprocess.run([str(T2PY), str(ROOT / "scripts" / "t2_dump.py"), str(par), str(tim), str(out)],
                       capture_output=True, text=True, cwd=str(tim.parent), check=False)
    assert p.returncode == 0, p.stderr[-2000:]
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.oracle
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_reader_matches_tempo2(tmp_path, fixture):
    if not have_tempo2():
        pytest.skip("tempo2/libstempo oracle env not installed (scripts/setup_tempo2_env.sh)")
    tim = _write(tmp_path, FIXTURES[fixture])
    recs, rep = read_tim(tim)
    t2 = _tempo2(tmp_path / "fx.par", tim)
    assert len(recs) == len(t2["freqs"])
    flags = json.loads(str(t2["flags_json"]))
    for i, r in enumerate(recs):
        day, sec = r.mjd_parts()
        dt = (int(t2["stoas_day"][i]) - day) * 86400 + (np.longdouble(t2["stoas_sec"][i]) - sec)
        assert abs(float(dt)) < 1e-9, (fixture, i, r.name, float(dt))
        assert r.freq == t2["freqs"][i] and abs(r.err - t2["toaerrs"][i]) < 1e-12
        # libstempo reports the last value of each flag; compare on the flags tempo2 knows
        mine = {k.lstrip("-"): v for k, v in r.flags + r.artifact_flags}
        for k, vals in flags.items():
            assert mine.get(k, "") == vals[i], (fixture, i, k, mine.get(k), vals[i])


def test_reader_semantics_without_tempo2(tmp_path):
    recs, rep = read_tim(_write(tmp_path, FIXTURES["scoping"]))
    assert [r.name for r in recs] == ["toaA", "toaB", "toaG", "toaH", "toaC", "toaD", "toaF"]
    assert [r.offset_s for r in recs] == [0.0, 1.0, 0.0, 5.0, 1.25, 1.5, 1.5]
    assert rep.counts["commented TOA (C, no blank)"] == 1
    assert recs[5].flags == [("-projid", "-beconfig"), ("-group", "X"), ("-padd", "0.01")]
    assert tempo2_flags("a 1 2 3 o -x 1 -y") == [("-x", "1")]


def test_unsupported_commands_rejected(tmp_path):
    (tmp_path / "t.tim").write_text("FORMAT 1\nEFAC 2\nx 1400.0 55000.1 1.0 pks\n")
    with pytest.raises(TimSemanticsError, match="EFAC"):
        read_tim(tmp_path / "t.tim")
    (tmp_path / "u.tim").write_text("FORMAT 1\nx 1400.0 55000.1 1.0 pks -to 1 -to 2\n")
    with pytest.raises(TimSemanticsError, match="duplicated physics flag"):
        read_tim(tmp_path / "u.tim")


def test_flat_file_read_by_pint(tmp_path):
    """PINT reads the flat file with exactly the tempo2 arrival times (TIME folded into -to)."""
    import warnings

    from pint.toa import get_TOAs

    recs, _ = read_tim(_write(tmp_path, FIXTURES["scoping"]))
    flat = write_flat_tim(recs, tmp_path / "flat.tim")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = get_TOAs(str(flat), ephem="DE440", planets=False, include_bipm=False)
    assert t.ntoas == len(recs)
    to = np.array([float(fl.get("to", 0.0)) for fl in t.table["flags"]])
    np.testing.assert_allclose(to, [r.offset_s for r in recs], rtol=0, atol=1e-12)
    for fl, r in zip(t.table["flags"], recs, strict=True):
        exp = {k.lstrip("-").lower(): v for k, v in canonical_flags(r) if k != "-to"}
        got = {k.lower(): v for k, v in fl.items() if k not in ("format", "clkcorr", "to", "name", "padd", "pn")}
        exp.pop("padd", None)
        assert got == exp


@pytest.mark.slow
def test_released_counts():
    from ptagwb.m3data import leg_files

    try:
        totals = {ds: sum(len(read_tim(t)[0]) for _, t in leg_files(ds).values())
                  for ds in ("ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr2", "inpta_dr1", "mpta")}
    except FileNotFoundError:
        pytest.skip("M3 data not fetched")
    assert totals == {"ng15": 676397, "epta_dr2new": 45428, "ppta_dr3_gh": 113951, "inpta_dr2": 83120,
                      "inpta_dr1": 8523, "mpta": 245907}
    # selected configuration and Yu & Allen's set (InPTA DR1's 6 'CJ...' lines are comments)
    assert sum(totals[d] for d in ("ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr2", "mpta")) == 1164803
    assert sum(totals[d] for d in ("ng15", "epta_dr2new", "ppta_dr3_gh", "inpta_dr1", "mpta")) == 1090206
