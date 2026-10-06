"""Download the PTA data sets used by this project into data/raw/ and write data/MANIFEST.json.

Usage:
    uv run python scripts/fetch_data.py                  # all default sources
    uv run python scripts/fetch_data.py --group ng15     # only NG15 primary inputs
    uv run python scripts/fetch_data.py --only ng15_v2.1.0 ng15_tutorial_repo
    uv run python scripts/fetch_data.py --include-large  # also the very large optional sets
    uv run python scripts/fetch_data.py --list           # show sources and exit

Downloads are idempotent: a file that already exists with the expected size (and the
expected md5, if the source publishes one) is not downloaded again, but its sha256 is
always recomputed and written to the manifest. Archives (.tar.gz / .zip) are unpacked
into `<dest>/extracted/` once.

Data lives under data/raw/ (git-ignored). Only data/MANIFEST.json is committed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.json"
UA = {"User-Agent": "ptagwb-fetch/0.1 (+https://github.com/kasper0406/pta-gwb-jax)"}
CHUNK = 1 << 20


# --------------------------------------------------------------------------------------
# Source definitions
# --------------------------------------------------------------------------------------


@dataclass
class Source:
    name: str
    group: str  # ng15 | ng15_reference | secondary
    kind: str  # zenodo | gdrive | github | gitlab | csiro_dap | not_fetched
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    large: bool = False  # only fetched with --include-large
    extract: bool = True


SOURCES: list[Source] = [
    # ---------------- NG15 primary inputs ----------------
    Source(
        "ng15_v2.1.0",
        "ng15",
        "zenodo",
        "NANOGrav 15-yr data set, latest version (v2.1.0) of concept record 7967584: "
        "narrowband+wideband par/tim, clock files, noise-modelling MCMC chains (added in v2), "
        "NoRedNoise/predictive par files, correlation matrices, post-fit residuals.",
        {"record": "16051178", "concept_doi": "10.5281/zenodo.7967584"},
    ),
    Source(
        "ng15_v1.0.1",
        "ng15",
        "zenodo",
        "NANOGrav 15-yr data set v1.0.1 (the June-2023 release contemporaneous with the GWB "
        "paper; v1.0.1 only fixes the tarball directory structure of v1.0.0). Kept to verify "
        "that the narrowband par/tim used for the GWB analysis are unchanged in later versions.",
        {"record": "8104459", "concept_doi": "10.5281/zenodo.7967584"},
    ),
    Source(
        "ng15_tutorial_repo",
        "ng15",
        "github",
        "nanograv/15yr_stochastic_analysis (official companion repo of arXiv:2306.16213): "
        "15yr_wn_dict.json (fixed white-noise dictionary EFAC/EQUAD/ECORR), 15yr_emp_distr.json "
        "(intrinsic-RN empirical distributions), 67 Enterprise-derived pulsar feather files "
        "(v1p1_de440_pint_bipm2019-*.feather), presampled la_forge cores (CURN/HD 14f varied-gamma, "
        "HD 30f free spectrum, spline ORF, TI), optimal-statistic results. The README warns the "
        "tutorial data are 'reduced' and may not reproduce the paper exactly.",
        {
            "repo": "nanograv/15yr_stochastic_analysis",
            "ref": "a3e8b8776a646208ca2d307f83c20bc70e029933",
        },
    ),
    Source(
        "discovery_repo",
        "ng15",
        "github",
        "nanograv/discovery source + data/: 67 NG15 feather files (v1p1_de440_pint_bipm2019-*), "
        "NG15yr-m2a-chain.feather (CURN) and NG15yr-m3a-chain.feather (HD) reference chains, "
        "EPTA DR2 feathers. Oracle/reference only.",
        {"repo": "nanograv/discovery", "ref": "b26d2554df5540a7610eedf926659701daad01ad"},
    ),
    Source(
        "ng15_cw_analysis",
        "ng15",
        "zenodo",
        "NG15 continuous-wave analysis release (arXiv:2306.16222): contains v1p1_all_dict.json "
        "(white-noise + intrinsic red-noise values for every pulsar), enterprise Pulsar pickles "
        "(jar/), pulsar distances. We need it for the noise dictionary.",
        {"record": "8067506"},
    ),
    # ---------------- NG15 reference products ----------------
    Source(
        "ng15_sensitivity_curves",
        "ng15_reference",
        "zenodo",
        "Noise spectra + GWB sensitivity curves for NG15 (detector-characterization paper, "
        "arXiv:2306.16218).",
        {"record": "8092346"},
    ),
    Source(
        "ng15_kde_freespec_v1",
        "ng15_reference",
        "zenodo",
        "Ceffyl KDE representations of the NG15 GWB free spectra, v1.0.0.",
        {"record": "8060824"},
    ),
    Source(
        "ng15_kde_freespec_v2",
        "ng15_reference",
        "zenodo",
        "Ceffyl KDE representations of the NG15 GWB free spectra, v2.0.0 (30f CURN / 30f HD "
        "free spectra).",
        {"record": "21844115"},
    ),
    Source(
        "ng15_gwb_fig1_data",
        "ng15_reference",
        "gdrive",
        "Data behind Fig. 1 of arXiv:2306.16213 (linked from 15yr_stochastic_analysis/"
        "data_release/figure_1). Google Drive, no checksum published.",
        {"id": "1zywc5zUpMSlYDrrdogPEaRd_RTr9U5J6"},
    ),
    Source(
        "ng15_gwb_fig4_correlations",
        "ng15_reference",
        "gdrive",
        "Data behind Fig. 4 (correlations_gamma4p33_nfreq14.npz, correlations_variedgamma_"
        "nfreq14.npz; data_release/figure_4).",
        {"id": "19I8qAnLCrCo7PBa5k5rXUG27U9S7LkWS"},
    ),
    Source(
        "ng15_gwb_fig5_psd_pl",
        "ng15_reference",
        "gdrive",
        "Data behind Fig. 5: psd_pl_posteriors (data_release/figure_5).",
        {"id": "1vB6U0V9E5ETyPbWvwJUe_aqNEgxfC6XU"},
    ),
    Source(
        "ng15_gwb_fig5_psd_pl_dmgp",
        "ng15_reference",
        "gdrive",
        "Data behind Fig. 5: psd_pl_posteriors_dmgp (data_release/figure_5).",
        {"id": "1VIiR4apoIWYU-sJMb9SPzsaImRruHyE-"},
    ),
    Source(
        "ng15_gwb_fig7_os",
        "ng15_reference",
        "gdrive",
        "Data behind Fig. 7 (optimal statistic / Legendre; data_release/figure_7). 2.7 GB.",
        {"id": "1T6tbrVA_hS-oUX1wiHLa2MJ4ymg0Ho1i"},
    ),
    Source(
        "ng15_gwb_fig10_telescopes",
        "ng15_reference",
        "gdrive",
        "Data behind Fig. 10: compare_telescopes (data_release/figure_10).",
        {"id": "1keklHY52MV38BXF3dIqzumjKo2mdB9I_"},
    ),
    Source(
        "ng15_gwb_fig9_variable_gamma",
        "ng15_reference",
        "gdrive",
        "variable_gamma.tar.gz linked from data_release/figure_9 notebook.",
        {"id": "1FfvQBh8Pl7FhvvMQPYQYpXEe9fuT6jDM"},
    ),
    Source(
        "ng15_newphysics_chains",
        "ng15_reference",
        "zenodo",
        "MCMC chains of the NG15 new-physics paper (arXiv:2306.16219), 22.4 GB single file. "
        "Not the baseline GWB analysis; optional.",
        {"record": "8083620"},
        large=True,
        extract=False,
    ),
    # ---------------- Secondary PTAs (download only) ----------------
    Source(
        "epta_dr2_zenodo",
        "secondary",
        "zenodo",
        "EPTA DR2 paper I data set (DR2full/DR2new, 25 MSPs, par/tim), Zenodo concept 8164424.",
        {"record": "8300645"},
    ),
    Source(
        "epta_dr2_noise_models",
        "secondary",
        "zenodo",
        "EPTA DR2 paper II: customised noise models (posterior distributions, whitened residuals).",
        {"record": "8025019"},
    ),
    Source(
        "epta_dr2_gwb_chains",
        "secondary",
        "zenodo",
        "EPTA DR2 paper III: GWB search chains (arXiv:2306.16214).",
        {"record": "8091568"},
    ),
    Source(
        "epta_dr2_gitlab",
        "secondary",
        "gitlab",
        "EPTA DR2 GitLab repository (gitlab.in2p3.fr/epta/epta-dr2), public.",
        {
            "host": "https://gitlab.in2p3.fr",
            "project": "epta/epta-dr2",
            "ref": "2911d0e52e0c8a4e528c4e3aa46b868ced1910e8",
        },
    ),
    Source(
        "ppta_dr3_timing",
        "secondary",
        "csiro_dap",
        "PPTA DR3 (doi:10.25919/j4xr-wp05, CSIRO DAP collection 59374 v2, part 1 of 2): only "
        "ppta_dr3/toas_and_parameters/ (par, tim, noise files, clock) and READMEs. The ~1.4 TB of "
        "profile/UWL data products are not fetched. DAP publishes no checksums.",
        {
            "collection_id": "59423",
            "landing": "https://data.csiro.au/collection/csiro%3A59374v2",
            "doi": "10.25919/j4xr-wp05",
            "include": [r"^ppta_dr3/toas_and_parameters/", r"^ppta_dr3/README$"],
        },
        extract=False,
    ),
    Source(
        "inpta_dr2",
        "secondary",
        "github",
        "InPTA DR2 (27 MSPs, uGMRT) official GitHub release repository.",
        {"repo": "inpta/InPTA.DR2", "ref": "HEAD"},
    ),
    Source(
        "inpta_dr2_noise",
        "secondary",
        "github",
        "InPTA DR2 noise-analysis (DM12) par files.",
        {"repo": "inpta/InPTA.DR2.NA", "ref": "HEAD"},
    ),
]


# Where the most important reference artifacts live after extraction (paths relative to
# data/raw/, glob patterns). Recorded in the manifest for discoverability.
KEY_FILES: dict[str, str] = {
    "ng15_narrowband_par_tim": "ng15_v2.1.0/extracted/NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/{par,tim}/",
    "ng15_single_pulsar_noise_chains": "ng15_v2.1.0/extracted/NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/noise/",
    "white_noise_dict_gwb": "ng15_gwb_fig1_data/extracted/figure1_data/v1p1_wn_dict.json "
    "(identical values to ng15_tutorial_repo/.../tutorials/data/15yr_wn_dict.json)",
    "wn_plus_rn_dict": "ng15_cw_analysis/extracted/15yr_cw_analysis-main/data/v1p1_all_dict.json",
    "gwb_chain_hd_varied_gamma": "ng15_gwb_fig1_data/extracted/figure1_data/nano15_hd_chain_long_050523.npy "
    "(columns: gamma, log10_A)",
    "gwb_chain_hd_gamma_13_3": "ng15_gwb_fig1_data/extracted/figure1_data/nano15_hd_chain_fg_long_050523.npy "
    "(log10_A)",
    "gwb_chain_hd_free_spectrum_30f": "ng15_gwb_fig1_data/extracted/figure1_data/30fCP_30fiRN_3A_freespec_chain.core",
    "gwb_chain_spline_orf": "ng15_gwb_fig1_data/extracted/figure1_data/SplineORF_{Varied,Fixed}Gamma_NL.core",
    "gwb_chain_hd_14f_pl": "ng15_gwb_fig1_data/extracted/figure1_data/14f_PL_hd_crn.core",
    "curn_chain_varied_gamma_with_logl": "discovery_repo/extracted/discovery-*/data/NG15yr-m2a-chain.feather",
    "hd_chain_varied_gamma_with_logl": "discovery_repo/extracted/discovery-*/data/NG15yr-m3a-chain.feather",
    "ng15_feathers_67psr": "discovery_repo/extracted/discovery-*/data/v1p1_de440_pint_bipm2019-*.feather",
    "os_pair_correlations": "ng15_gwb_fig7_os/extracted/figure7_data/correlations_*_nfreq{5,14}.npz",
    "tutorial_presampled_cores": "ng15_tutorial_repo/extracted/*/tutorials/presampled_cores/*.core",
    "curn_varied_gamma_psd_posteriors": "ng15_gwb_fig5_psd_pl/extracted/curn_variedgamma.h5",
}

NOT_AVAILABLE: dict[str, str] = {
    "ng15_gwb_official_chain_release": "No dedicated Zenodo/NANOGrav-webpage release of the full GWB-paper "
    "chains exists (checked nanograv.org/science/data and Zenodo search, 2026-10-06). The GWB "
    "chains used for the paper figures are distributed via Google Drive links in "
    "nanograv/15yr_stochastic_analysis/data_release (fetched above as ng15_gwb_fig*).",
    "ppta_dr3_profiles": "PPTA DR3 profile / UWL data products (~1.4 TB per collection) deliberately "
    "not fetched; only toas_and_parameters/.",
    "ppta_dr3_part2": "CSIRO collection 59381 (part 2 of 2) lists the same toas_and_parameters/ "
    "files as part 1; not fetched separately.",
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def http_json(url: str, retries: int = 5) -> Any:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"    retry {i + 1} for {url}: {e}")
            time.sleep(2 * (i + 1))


def hash_file(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def download(
    url: str, dest: Path, size: int | None = None, md5: str | None = None, retries: int = 5
) -> None:
    """Stream url -> dest, skipping if already present with matching size/md5."""
    if dest.exists():
        size_ok = size is None or dest.stat().st_size == size
        if size_ok and (md5 is None or hash_file(dest)[1] == md5):
            print(f"    have {dest.relative_to(ROOT)}")
            return
        print(f"    re-downloading {dest.name} (size/md5 mismatch)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0) or size
                done, t0 = 0, time.time()
                last = t0
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 15:
                        last = time.time()
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        rate = done / max(time.time() - t0, 1e-3) / 1e6
                        print(f"    {dest.name}: {done / 1e6:9.1f} MB {pct} {rate:6.1f} MB/s")
            break
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"    retry {i + 1} for {dest.name}: {e}")
            time.sleep(5 * (i + 1))
    if size is not None and tmp.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: size {tmp.stat().st_size} != expected {size}")
    if md5 is not None:
        got = hash_file(tmp)[1]
        if got != md5:
            raise RuntimeError(f"{dest.name}: md5 {got} != expected {md5}")
    tmp.replace(dest)


def extract(archive: Path, outdir: Path) -> str | None:
    marker = outdir / ".extracted_ok"
    if marker.exists():
        return str(outdir.relative_to(ROOT))
    name = archive.name.lower()
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(outdir, filter="data")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(outdir)
    else:
        outdir.rmdir()
        return None
    marker.write_text(dt.datetime.now(dt.UTC).isoformat())
    return str(outdir.relative_to(ROOT))


def file_entry(path: Path, url: str, md5_expected: str | None = None) -> dict[str, Any]:
    sha, md5 = hash_file(path)
    e: dict[str, Any] = {
        "name": path.name,
        "path": str(path.relative_to(ROOT)),
        "url": url,
        "size": path.stat().st_size,
        "sha256": sha,
        "md5": md5,
    }
    if md5_expected is not None:
        e["md5_expected"] = md5_expected
        e["checksum_verified"] = md5 == md5_expected
    else:
        e["checksum_verified"] = None  # source publishes no checksum
    return e


# --------------------------------------------------------------------------------------
# Fetchers (each returns a manifest entry dict)
# --------------------------------------------------------------------------------------


def fetch_zenodo(src: Source, dest: Path) -> dict[str, Any]:
    rec = http_json(f"https://zenodo.org/api/records/{src.params['record']}")
    md = rec["metadata"]
    files = []
    for f in rec["files"]:
        algo, _, digest = f["checksum"].partition(":")
        md5 = digest if algo == "md5" else None
        url = f["links"]["self"]
        out = dest / f["key"]
        download(url, out, size=f["size"], md5=md5)
        e = file_entry(out, url, md5)
        if src.extract:
            e["extracted_to"] = extract(out, dest / "extracted")
        files.append(e)
    return {
        "source_url": f"https://zenodo.org/records/{rec['id']}",
        "api_url": f"https://zenodo.org/api/records/{rec['id']}",
        "doi": rec.get("doi"),
        "concept_doi": rec.get("conceptdoi"),
        "title": md.get("title"),
        "version": md.get("version"),
        "publication_date": md.get("publication_date"),
        "files": files,
    }


def fetch_gdrive(src: Source, dest: Path) -> dict[str, Any]:
    fid = src.params["id"]
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        cd = r.headers.get("Content-Disposition", "")
        size = int(r.headers.get("Content-Length") or 0) or None
    m = re.search(r'filename="([^"]+)"', cd)
    fname = m.group(1) if m else f"{fid}.bin"
    out = dest / fname
    download(url, out, size=size)
    e = file_entry(out, url)
    if src.extract:
        e["extracted_to"] = extract(out, dest / "extracted")
    return {
        "source_url": f"https://drive.google.com/file/d/{fid}/view",
        "doi": None,
        "files": [e],
    }


def _github_sha(repo: str, ref: str) -> str:
    d = http_json(f"https://api.github.com/repos/{repo}/commits/{ref}")
    return d["sha"]


def fetch_github(src: Source, dest: Path) -> dict[str, Any]:
    repo, ref = src.params["repo"], src.params["ref"]
    sha = ref if re.fullmatch(r"[0-9a-f]{40}", ref) else _github_sha(repo, ref)
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    out = dest / f"{repo.split('/')[-1]}-{sha[:12]}.tar.gz"
    download(url, out)
    e = file_entry(out, url)
    if src.extract:
        e["extracted_to"] = extract(out, dest / "extracted")
    return {
        "source_url": f"https://github.com/{repo}/tree/{sha}",
        "doi": None,
        "commit": sha,
        "files": [e],
        "note": "GitHub archive tarballs are not guaranteed byte-stable; the commit sha is the "
        "authoritative identifier.",
    }


def fetch_gitlab(src: Source, dest: Path) -> dict[str, Any]:
    host, project, ref = src.params["host"], src.params["project"], src.params["ref"]
    name = project.split("/")[-1]
    url = f"{host}/{project}/-/archive/{ref}/{name}-{ref}.tar.gz"
    out = dest / f"{name}-{ref[:12]}.tar.gz"
    download(url, out)
    e = file_entry(out, url)
    if src.extract:
        e["extracted_to"] = extract(out, dest / "extracted")
    return {
        "source_url": f"{host}/{project}/-/tree/{ref}",
        "doi": None,
        "commit": ref,
        "files": [e],
    }


def fetch_csiro_dap(src: Source, dest: Path) -> dict[str, Any]:
    cid = src.params["collection_id"]
    listing = http_json(f"https://data.csiro.au/dap/ws/v2/collections/{cid}/data")["file"]
    pats = [re.compile(p) for p in src.params["include"]]
    sel = [f for f in listing if any(p.search(f["filename"]) for p in pats)]
    files = []
    for i, f in enumerate(sorted(sel, key=lambda x: x["filename"])):
        out = dest / f["filename"]
        url = f["presignedLink"]["href"]  # pre-signed S3 link, expires after 48 h
        download(url, out, size=f["fileSize"])
        e = file_entry(out, f["link"]["href"])
        files.append(e)
        if i % 50 == 0:
            print(f"    {i + 1}/{len(sel)} files")
    return {
        "source_url": src.params["landing"],
        "doi": src.params["doi"],
        "api_url": f"https://data.csiro.au/dap/ws/v2/collections/{cid}/data",
        "files": files,
    }


FETCHERS = {
    "zenodo": fetch_zenodo,
    "gdrive": fetch_gdrive,
    "github": fetch_github,
    "gitlab": fetch_gitlab,
    "csiro_dap": fetch_csiro_dap,
}


# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--group", nargs="*", help="only these groups (ng15, ng15_reference, secondary)"
    )
    ap.add_argument("--only", nargs="*", help="only these source names")
    ap.add_argument("--include-large", action="store_true", help="also fetch sources marked large")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            print(
                f"{s.name:32s} {s.group:15s} {s.kind:10s} {'LARGE ' if s.large else ''}{s.description[:80]}"
            )
        return

    manifest: dict[str, Any] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    entries: dict[str, Any] = manifest.get("datasets", {})

    failures = []
    for s in SOURCES:
        if args.group and s.group not in args.group:
            continue
        if args.only and s.name not in args.only:
            continue
        base = {"group": s.group, "kind": s.kind, "description": s.description}
        if s.large and not args.include_large:
            if entries.get(s.name, {}).get("status") == "fetched":
                continue  # keep the record of a previous --include-large run
            entries[s.name] = {
                **base,
                "status": "not_fetched",
                "reason": "large optional source; run with --include-large",
                "params": s.params,
            }
            print(f"[skip] {s.name} (large)")
            continue
        print(f"[fetch] {s.name}")
        dest = RAW / s.name
        try:
            info = FETCHERS[s.kind](s, dest)
            entries[s.name] = {
                **base,
                "status": "fetched",
                "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "dest": str(dest.relative_to(ROOT)),
                **info,
                "total_size": sum(f["size"] for f in info["files"]),
            }
            print(
                f"    ok: {len(info['files'])} file(s), "
                f"{entries[s.name]['total_size'] / 1e6:.1f} MB"
            )
        except Exception as e:  # noqa: BLE001
            failures.append(s.name)
            entries[s.name] = {
                **base,
                "status": "not_fetched",
                "reason": f"error: {e!r}",
                "params": s.params,
            }
            print(f"    FAILED: {e!r}")
        manifest = {
            "description": "PTA data sets for the NG15 GWB reproduction. Raw files live in "
            "data/raw/ (git-ignored); regenerate with `uv run python scripts/fetch_data.py`.",
            "generated_by": "scripts/fetch_data.py",
            "key_files": KEY_FILES,
            "not_available": NOT_AVAILABLE,
            "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "datasets": entries,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")

    if failures:
        print("FAILED sources:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
