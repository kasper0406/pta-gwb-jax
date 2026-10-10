"""M3b-0E: canonical EPTA DR2new legs for the published-analysis roster, and the tempo2 runtime of
the evaluator profile ``published-tempo2-v1`` (docs/M3B_PLAN.md Sec. 4.1, 4.2).

* Published TOA set (Sec. 4.2): every TOA tempo2 reads from the released tim tree; **no** M3a
  duplicate removals and **no** clock-coverage exclusions (tempo2 interpolates as the PTA did).
* Canonical files = M3a canonicalisation (``legs.canonical_leg_texts`` with dataset=None, i.e.
  without the duplicate list) + the published clock profile, written in tempo2's spelling
  (``CLOCK`` -> ``CLK``; listed rule ``t2-clock-keyword``). The flat tim drops the tempo2 flag
  artefacts ``-.cal`` (M3a rule; flags only).
* tempo2 runtime ``epta-dr2-chain-runtime-v1`` (reproduces the stored likelihoods; chain fingerprint,
  docs/M3B0_VALIDATION.md Sec. 3): the T2runtime of conda-forge tempo2 2023.05.1 (build
  hcb8dc1c_5) with two clock files laid over its clock directory: the release's corrected Nancay
  file ``ncyobs2obspm.clk`` (profile ``epta-dr2-published-v1``) and ``gps2utc.clk`` of conda-forge
  tempo2 2022.05.1 (build h1c8e422_2; data end MJD 59149). The tempo2 *binary* is the D1 one
  (2026.04.1); only data files come from the older packages. Every runtime data file (clock/,
  earth/, observatory/, ephemeris/DE440) is hashed and checked against the committed pin
  ``configs/m3b/t2runtime_epta.json`` (written on the first run with ``--pin``).

Usage: PYTHONPATH=src python scripts/m3b_epta_prepare.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from ptagwb.config import RAW_DIR, REPO_ROOT
from ptagwb.legs import canonical_leg_texts
from ptagwb.m3data import leg_files
from ptagwb.profiles import published_profile
from ptagwb.timfile import write_flat_tim

OUT = REPO_ROOT / "data" / "processed" / "m3b" / "epta"
EF_ENV = Path(os.environ.get("EF_ENV", Path.home() / ".local" / "opt" / "epta-fork-env"))
RT_BASE_ENV = Path.home() / ".local" / "opt" / "t2rt-2023.05.1"
RT_GPS2UTC = Path.home() / ".local" / "opt" / "t2rt-2022.05.1" / "share" / "tempo2" / "clock" / "gps2utc.clk"
RUNTIME_NAME = "epta-dr2-chain-runtime-v1"
PIN = REPO_ROOT / "configs" / "m3b" / "t2runtime_epta.json"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def t2_par(text: str) -> str:
    out = []
    for ln in text.splitlines():
        tok = ln.split()
        out.append(f"CLK {' '.join(tok[1:])}" if tok[:1] == ["CLOCK"] else ln)
    return "\n".join(out) + "\n"


def runtime_hashes(rt: Path) -> dict:
    from ptagwb.binding import runtime_file_hashes

    return runtime_file_hashes(rt)


def build_runtime(tag: str = "", env: Path = RT_BASE_ENV, extra: tuple = (RT_GPS2UTC,), pin: bool = False) -> dict:
    """The profile runtime (tag ""), or a diagnostic variant ``t2runtime-<tag>`` built from another
    env's T2runtime and/or with extra clock files laid over (``extra``: paths)."""
    prof = published_profile("EPTA")
    base = Path(env) / "share" / "tempo2"
    out = OUT / ("t2runtime" + (f"-{tag}" if tag else ""))
    if out.exists():  # rebuilt from scratch: stale symlinks into another env must not survive
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for sub in base.iterdir():
        if sub.name != "clock":
            (out / sub.name).symlink_to(sub)
    shutil.copytree(base / "clock", out / "clock")
    overlay = {}
    for rel in prof.overrides:
        src = RAW_DIR / rel
        shutil.copy2(src, out / "clock" / src.name)
        overlay[src.name] = str(rel)
    for f in extra:
        shutil.copy2(f, out / "clock" / Path(f).name)
        overlay[Path(f).name] = str(f)
    m3a_pin = json.loads((REPO_ROOT / "configs" / "m3" / "clocks" / f"{prof.name}.json").read_text())
    files = {f.name: sha(f) for f in sorted((out / "clock").glob("*.clk"))}
    mism = {k: (files.get(k), v) for k, v in m3a_pin["files"].items() if files.get(k) != v}
    meta = {"runtime": RUNTIME_NAME if not tag else f"diagnostic:{tag}", "clock_profile": prof.name,
            "base": str(base), "base_conda": sorted(p.name for p in (Path(env) / "conda-meta").glob("tempo2-*.json")),
            "overlay": overlay, "clock_files": files, "m3a_pin_mismatches": mism}
    if not tag:
        hashes = runtime_hashes(out)
        if pin:
            if PIN.exists():
                raise RuntimeError(f"{PIN} exists; a re-pin is a deliberate, reviewed change (delete it first)")
            PIN.parent.mkdir(parents=True, exist_ok=True)
            PIN.write_text(json.dumps({"runtime": RUNTIME_NAME, "base_conda": meta["base_conda"],
                                       "overlay": {k: Path(v).name for k, v in overlay.items()},
                                       "files": hashes}, indent=1, sort_keys=True))
        if not PIN.exists():
            raise RuntimeError(f"no runtime pin {PIN}; run once with --pin")
        want = json.loads(PIN.read_text())["files"]
        bad = sorted(k for k in set(want) | set(hashes) if want.get(k) != hashes.get(k))
        if bad:
            raise RuntimeError(f"runtime differs from the pin {PIN.name}: {bad[:10]} ({len(bad)} files)")
        meta["pin_verified"] = len(want)
    (out / "runtime.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-tag", default=None, help="only build a diagnostic runtime t2runtime-<tag>")
    ap.add_argument("--env", default=str(RT_BASE_ENV))
    ap.add_argument("--extra-clock", nargs="*", default=None)
    ap.add_argument("--pin", action="store_true", help="write configs/m3b/t2runtime_epta.json")
    a = ap.parse_args()
    if a.runtime_tag is not None:
        meta = build_runtime(a.runtime_tag, Path(a.env), tuple(a.extra_clock or ()))
        print(json.dumps({k: meta[k] for k in ("base", "overlay", "m3a_pin_mismatches")}, indent=1))
        return
    prof = published_profile("EPTA")
    rows = {}
    for psr, (par, tim) in leg_files("epta_dr2new").items():
        fixes: Counter = Counter()
        recs, rep, ptxt, multi = canonical_leg_texts(par, tim, fixes)  # dataset=None: no removals
        ptxt, clock_changes = prof.apply_to_par(ptxt)
        d = OUT / "canonical" / psr
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{psr}.par").write_text(t2_par(ptxt))
        ctim = write_flat_tim(recs, d / f"{psr}.flat.tim")
        rows[psr] = {"par": str(par), "tim": str(tim), "par_sha256": sha(par), "tim_files": rep.files,
                     "tim_files_sha256": {Path(f).name: sha(Path(f)) for f in rep.files},
                     "canon_par": str(d / f"{psr}.par"), "canon_tim": str(ctim),
                     "canon_par_sha256": sha(d / f"{psr}.par"), "canon_tim_sha256": sha(ctim),
                     "n_records": len(recs), "fixes": dict(fixes) | {"par:t2-clock-keyword (CLOCK -> CLK)": 1},
                     "clock_changes": clock_changes, "multivalued_mask_flags": sorted(multi)}
    (OUT / "canonical" / "prepare.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    meta = build_runtime(pin=a.pin)
    print(f"{len(rows)} legs; TOA records {sum(r['n_records'] for r in rows.values())}; "
          f"runtime {meta['runtime']} overlay {sorted(meta['overlay'])}; {meta['pin_verified']} files match the pin")


if __name__ == "__main__":
    main()
