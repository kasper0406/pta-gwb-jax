"""Version-pinned EPTA DR2new prior/model manifest (docs/M3B_PLAN.md Sec. 4.5; M3b-0E exit
condition) and its prior-volume check against every row of the released chains.

Every source fact is re-read from the pinned sources when the manifest is generated (fail closed):
the fork code at the chain-producing commits (``git show <commit>:<file>`` in the clones under
data/raw/epta_fork/), the release's noise files and dictionaries, the released chains' parameter
names. Provenance = (repository, commit, file, line) of the line that sets the fact.

Usage: PYTHONPATH=src python scripts/m3b_manifest_epta.py [--write]
  --write   (re)write configs/m3b/manifest_epta.json; otherwise only check the committed manifest
            against a fresh generation and run the prior-volume check.
Writes data/processed/m3b/epta/results/prior_volume.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ptagwb.binding import evidence_binding
from ptagwb.config import RAW_DIR, REPO_ROOT
from ptagwb import epta

FORK = RAW_DIR / "epta_fork"
ENT = ("gitlab.in2p3.fr/epta/enterprise", "607c28533acc18b4ed7a743ca568ffc7e41a9137", FORK / "epta_enterprise")
EXT = ("gitlab.in2p3.fr/epta/enterprise_extensions", "051173f46e64918aa4df9e9fea8fbc04a79839f9",
       FORK / "epta_enterprise_extensions")
DR2 = ("gitlab.in2p3.fr/epta/epta-dr2", "2911d0e52e0c8a4e528c4e3aa46b868ced1910e8", FORK / "epta_dr2_repo")
SCRIPT_REV = "be91c6b"  # original model_single.py (the HEAD version needs post-tag fork keywords)
REL = sorted((RAW_DIR / "epta_dr2_gitlab" / "extracted").glob("epta-dr2-*"))[0] / "EPTA-DR2"
NOISE = REL / "noisefiles" / "DR2new"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_show(repo, rev, path) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def prov(src, path, pattern, after: str | None = None) -> dict:
    """Locate ``pattern`` (exact substring) in ``path`` at the pinned commit, optionally the first
    occurrence after a line containing ``after``; fail if absent."""
    name, rev, repo = src
    lines = git_show(repo, rev, path).splitlines()
    start = 0
    if after is not None:
        start = next((i for i, ln in enumerate(lines) if after in ln), None)
        if start is None:
            sys.exit(f"provenance anchor {after!r} not found in {name}@{rev[:8]}:{path}")
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return {"repo": name, "commit": rev, "file": path, "line": i + 1, "text": lines[i].strip()}
    sys.exit(f"provenance {pattern!r} not found in {name}@{rev[:8]}:{path} (after {after!r})")


def generate() -> dict:
    B, M, C, GB, GS, U, O = ("enterprise_extensions/blocks.py", "enterprise_extensions/models.py",
                             "enterprise_extensions/chromatic/chromatic.py", "enterprise/signals/gp_bases.py",
                             "enterprise/signals/gp_signals.py", "enterprise/signals/utils.py",
                             "enterprise_extensions/model_orfs.py")
    P = {
        "red_prior": prov(EXT, B, "log10_A = parameter.Uniform(-18, -10)", after="def red_noise_block"),
        "red_gamma": prov(EXT, B, "gamma = parameter.Uniform(0, 7)", after="def red_noise_block"),
        "red_basis": prov(EXT, B, "rn = gp_signals.FourierBasisGP(pl, components=components,", after="def red_noise_block"),
        "dm_prior": prov(EXT, B, "log10_A_dm = parameter.Uniform(-18, -10)", after="def dm_noise_block"),
        "dm_gamma": prov(EXT, B, "gamma_dm = parameter.Uniform(0, 7)", after="def dm_noise_block"),
        "dm_basis": prov(EXT, B, "dm_basis = utils.createfourierdesignmatrix_dm_tn(nmodes=components,", after="def dm_noise_block"),
        "chrom_prior": prov(EXT, B, "log10_A = parameter.Uniform(-18, -10)", after="def chromatic_noise_block"),
        "chrom_gamma": prov(EXT, B, "gamma = parameter.Uniform(0, 7)", after="def chromatic_noise_block"),
        "chrom_basis": prov(EXT, B, "chm_basis = gpb.createfourierdesignmatrix_dm_tn(nmodes=components,", after="def chromatic_noise_block"),
        "gw_prior": prov(EXT, B, "log10_Agw = parameter.Uniform(-18, -10)(amp_name)", after="def common_red_noise_block"),
        "gw_gamma": prov(EXT, B, "gamma_gw = parameter.Uniform(0, 7)(gam_name)", after="def common_red_noise_block"),
        "gw_basis": prov(EXT, B, "cbasis = gpb.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan,", after="def common_red_noise_block"),
        "gw_crn": prov(EXT, B, "crn = gp_signals.BasisGP(cpl, cbasis, coefficients=coefficients, combine=combine,", after="def common_red_noise_block"),
        "gw_hd": prov(EXT, B, "crn = gp_signals.BasisCommonGP(cpl, cbasis, orfs[orf], coefficients=coefficients,", after="def common_red_noise_block"),
        "hd_orf": prov(EXT, O, "def hd_orf(pos1, pos2, diag=1.):"),
        "tspan_common": prov(EXT, M, "Tspan_common = model_utils.get_tspan(psrs)", after="def model_general"),
        "tspan_red": prov(EXT, M, "Tspan_red=None", after="def model_general"),
        "tspan_dm": prov(EXT, M, "dm_var=False, Tspan_dm=None", after="def model_general"),
        "tspan_chrom": prov(EXT, M, "dm_chrom=False, Tspan_chrom=None", after="def model_general"),
        "chrom_in_dm": prov(EXT, M, "s0 += chromatic_noise_block(gp_kernel=dmchrom_kernel,", after="def model_general"),
        "dip_call": prov(EXT, M, "name='{0}_dmexp_{1}'.format(p.name,dd+1))", after="def model_general"),
        "dip_t0": prov(EXT, C, "t0_dmexp = parameter.Uniform(tmin, tmax)", after="def dm_exponential_dip"),
        "dip_amp": prov(EXT, C, "log10_Amp_dmexp = parameter.Uniform(-10, -2)", after="def dm_exponential_dip"),
        "dip_tau": prov(EXT, C, "log10_tau_dmexp = parameter.Uniform(0, 2.5)", after="def dm_exponential_dip"),
        "dip_sign": prov(EXT, C, "sign_param = -1.0", after="def dm_exponential_dip"),
        "dip_wf": prov(EXT, C, "wf = 10**log10_Amp * np.heaviside(toas - t0, 1)", after="def chrom_exp_decay"),
        "wn_tnequad": prov(EXT, B, "efeq += white_signals.TNEquadNoise(log10_tnequad=equad,", after="def white_noise_block"),
        "tm": prov(EXT, M, "s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)", after="def model_general"),
        "tm_svd_basis": prov(ENT, U, "u, s, v = np.linalg.svd(Mmat, full_matrices=False)", after="def svd_tm_basis"),
        "tm_prior_const": prov(ENT, GS, "self.Mprior = Mmat.shape[1] * np.log(1e40)"),
        "dm_tn_factor": prov(ENT, GB, "Dm = (fref / freqs) ** idx * np.sqrt(12) * np.pi / 1400 / 1400 / 2.41e-4",
                             after="def createfourierdesignmatrix_dm_tn"),
        "fourier_grid": prov(ENT, GB, "f = 1.0 * np.arange(1, nmodes + 1) / T", after="def createfourierdesignmatrix_red"),
        "fourier_tspan": prov(ENT, GB, "T = Tspan if Tspan is not None else toas.max() - toas.min()",
                              after="def createfourierdesignmatrix_red"),
        "freqs_ssb": prov(ENT, "enterprise/pulsar.py", "self._ssbfreqs = np.double(t2pulsar.ssbfreqs()) / 1e6"),
        "script_call": prov(("gitlab.in2p3.fr/epta/epta-dr2", SCRIPT_REV, DR2[2]), "EPTA-DR2/scripts/model_single.py",
                            "dm_chrom=True, chrom_components=chrom_dict,"),
        "script_dip": prov(("gitlab.in2p3.fr/epta/epta-dr2", SCRIPT_REV, DR2[2]), "EPTA-DR2/scripts/model_single.py",
                           "dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],"),
    }
    red = json.loads((NOISE / "red_dict.json").read_text())
    dm = json.loads((NOISE / "dm_dict.json").read_text())
    chrom = json.loads((NOISE / "chrom_dict.json").read_text())
    names_crn = (epta.extract_reference("crn_pl") / "pars.txt").read_text().split()
    names_hd = (epta.extract_reference("hd_pl") / "pars.txt").read_text().split()
    pulsars = sorted(red)
    if sorted(dm) != pulsars or sorted(chrom) != pulsars or len(pulsars) != 25:
        sys.exit("noise dictionaries disagree on the roster")
    wn = {}
    for psr in pulsars:
        d = json.loads((NOISE / f"{psr}_noise.json").read_text())
        wn.update({k: v for k, v in d.items() if k.endswith(("_efac", "_log10_tnequad"))})
    blocks = {}
    for psr in pulsars:
        bl = []
        if red[psr] is not None:
            bl.append({"name": "red_noise", "modes": red[psr], "span": "pulsar", "chrom_idx": 0.0, "norm": "enterprise",
                       "prov": ["red_basis", "tspan_red", "fourier_tspan"]})
        if dm[psr] is not None:
            bl.append({"name": "dm_gp", "modes": dm[psr], "span": "pulsar", "chrom_idx": 2.0, "fref": 1400.0,
                       "norm": "temponest_dm", "prov": ["dm_basis", "dm_tn_factor", "tspan_dm"]})
        if chrom[psr] is not None:
            if dm[psr] is None:
                sys.exit(f"{psr}: chromatic GP exists only inside the DM branch of model_general")
            bl.append({"name": "chrom_gp", "modes": chrom[psr], "span": "pulsar", "chrom_idx": 4.0, "fref": 1400.0,
                       "norm": "temponest_dm", "prov": ["chrom_basis", "chrom_in_dm", "dm_tn_factor", "tspan_chrom"]})
        blocks[psr] = bl
    # parameters in chain order; bounds from the fork code (provenance keys)
    pre = "J1713+0747_J1713+0747_dmexp_1"
    rule = {"red_noise_log10_A": ((-18.0, -10.0), "red_prior"), "red_noise_gamma": ((0.0, 7.0), "red_gamma"),
            "dm_gp_log10_A": ((-18.0, -10.0), "dm_prior"), "dm_gp_gamma": ((0.0, 7.0), "dm_gamma"),
            "chrom_gp_log10_A": ((-18.0, -10.0), "chrom_prior"), "chrom_gp_gamma": ((0.0, 7.0), "chrom_gamma")}
    params = []
    union = list(names_crn) + [n for n in names_hd if n not in names_crn]
    for n in union:
        models = [m for m, nm in (("crn", names_crn), ("hd", names_hd)) if n in nm]
        if n.startswith(pre):
            k = n[len(pre) + 1:]
            b, pk = {"log10_Amp": ((-10.0, -2.0), "dip_amp"), "log10_tau": ((0.0, 2.5), "dip_tau"),
                     "t0": ((57490.0, 57530.0), "dip_t0")}[k]
            role, kern = f"dip_{k}", "mh" if k == "t0" else "nuts"
        elif n.startswith("gw_"):
            k = n.split("_", 2)[2]
            b, pk = {"log10_A": ((-18.0, -10.0), "gw_prior"), "gamma": ((0.0, 7.0), "gw_gamma")}[k]
            role, kern = f"common_{k}", "nuts"
        else:
            psr, rest = n.split("_", 1)
            b, pk = rule[rest]
            role, kern = rest, "nuts"
        params.append({"name": n, "models": models, "prior": "uniform", "bounds": list(b), "prov": pk, "role": role,
                       "kernel": kern, "shelf_prone": role.endswith("log10_A") or role == "dip_log10_Amp"})
    # spans from our export (the builder recomputes them; recorded for the record and the oracle check)
    psrs = epta.load_pulsars()
    spans = {p.name: float(p.toas.max() - p.toas.min()) for p in psrs}
    T_common = epta.array_span(psrs)
    ref = {}
    for m in ("crn_pl", "hd_pl"):
        d = epta.extract_reference(m)
        n_rows = sum(1 for _ in open(d / "chain_1.txt"))
        ref[m] = {"zenodo": "8091568", "tar": str(epta.reference_tar(m).relative_to(RAW_DIR)),
                  "tar_sha256": sha(epta.reference_tar(m)), "chain_sha256": sha(d / "chain_1.txt"),
                  "pars_sha256": sha(d / "pars.txt"), "rows": n_rows, "burn_in_rows": int(n_rows * 0.25),
                  "burn_in": "first 25 % of chain_1.txt (plan Sec. 6.5)",
                  "logging_convention": "single-model PTMCMCSampler: lnpost - lnlike = physical log prior (uniform boxes)",
                  "columns": "67 parameters (pars.txt order) + lnpost, lnlike, acceptance, PT acceptance",
                  "storage": "text; parameters ~22 significant digits; lnpost/lnlike 6 decimals (parsed to float64)"}
    rt = json.loads((REPO_ROOT / "configs" / "m3b" / "t2runtime_epta.json").read_text())
    man = {
        "manifest": "epta-dr2new-v1", "pta": "EPTA", "release": "DR2new",
        "decisions": "D1 (tempo2 evaluator), D6 (fork audit) of docs/M3B_PLAN.md; user 2026-10-09",
        "sources": {
            "data": {"repo": DR2[0], "commit": DR2[1], "dir": "EPTA-DR2/DR2new",
                     "note": "identical to the Zenodo EPTA-DR2 copy; replaced the pre-2023-08-23 files (1506123); the chain fingerprint confirms this version"},
            "noise_files": {"dir": "EPTA-DR2/noisefiles/DR2new",
                            "sha256": {f.name: sha(f) for f in sorted(NOISE.glob("*.json"))},
                            "consumed": "efac and log10_tnequad per -group backend; red/dm/chrom dict mode counts; the rn/dm_gp/cn/expd entries are not used (sampled)"},
            "script": {"repo": DR2[0], "commit": SCRIPT_REV, "file": "EPTA-DR2/scripts/model_single.py",
                       "note": "original script; HEAD (0ec5705, 2024-01-23) uses chrom_var/chrom_kernel, which exist only from fork commit f8f7ba4 (2023-07-28)",
                       "cli": {"orf": "crn | hd", "common_components": 9, "num_dmdips": 1, "gamma_common": None}},
            "enterprise": {"repo": ENT[0], "tag": "EPTADR2-v1.1", "commit": ENT[1], "upstream_base": "v3.3.1"},
            "enterprise_extensions": {"repo": EXT[0], "commit": EXT[1], "upstream_base": "v2.4.0",
                                      "identified_by": "chain parameter names J1713+0747_J1713+0747_dmexp_1_* (renamed by d3248419, 2023-03-23) and the script's orf_bins keyword (added by 23c63a17, 2023-03-14); equivalent on this path to EPTADR2-v1.1 (7619622a: rename + unused pseed only)"},
            "PTMCMCSampler": "not recoverable; proposals only (does not enter likelihood or prior)",
            "tempo2": {"binary": "2026.04.1 (conda-forge), libstempo 2.5.1 rebuilt against it: D1 profile published-tempo2-v1",
                       "runtime": rt["runtime"], "runtime_pin": "configs/m3b/t2runtime_epta.json",
                       "runtime_base_conda": rt["base_conda"], "runtime_overlay": rt["overlay"],
                       "identified_by": "chain fingerprint (docs/M3B0_VALIDATION.md Sec. 3)"},
            "ephemeris": "DE440 (Pulsar(ephem='DE440')); clock: par CLK TT(BIPM2021)",
        },
        "provenance": P,
        "pulsars": pulsars,
        "toas": {"n_total": int(sum(len(p.toas) for p in psrs)), "per_pulsar": {p.name: len(p.toas) for p in psrs},
                 "roster": "published set: every TOA tempo2 reads, no removals (plan Sec. 4.2)"},
        "conventions": {
            "timing_model": "MarginalizingTimingModel(use_svd=True): thin-SVD basis U of psr.Mmat, flat prior; logdet includes n_tm ln(1e40); no N ln(2 pi) term (fork enterprise; enterprise >= 3.5 subtracts it)",
            "white_noise": "N_ii = efac_b^2 sigma_i^2 + 10^(2 log10_tnequad_b), b = -group backend; no ECORR (all TOAs -pta EPTA)",
            "radio_frequency": "SSB frequencies (t2pulsar.ssbfreqs()/1e6)",
            "fourier": "f_k = k/T (k = 1..n), columns [sin, cos] interleaved, absolute BAT seconds (TCB)",
            "powerlaw": "phi_k = 10^(2 log10_A)/(12 pi^2) f_yr^(gamma-3) f_k^-gamma / T, f_yr = 1/(365.25 d)",
            "temponest_dm": "basis rows x (1400/nu)^idx sqrt(12) pi / (1400^2 2.41e-4), for idx 2 (DM) and 4 (chromatic)",
            "combine": "no columns merge: no pulsar spans the whole array (earliest TOA J1843-1113, latest J1640+2224)",
        },
        "white_noise": {"values": wn, "n_values": len(wn)},
        "blocks": blocks,
        "spans_s": spans,
        "common": {"modes": 9, "span": "array", "T_s": T_common, "psd": "powerlaw",
                   "orf": {"crn": "BasisGP per pulsar, shared (log10_A, gamma)", "hd": "BasisCommonGP, hd_orf (diag 1)"},
                   "prov": ["gw_basis", "gw_crn", "gw_hd", "hd_orf", "tspan_common"],
                   "modes_identified_by": "script CLI not released; freq_bins/freqs_dr2new.txt lists k/T_array; chain fingerprint discriminates (Sec. 3)"},
        "dip": {"pulsar": "J1713+0747", "param_prefix": pre, "chrom_idx": 1.0, "sign": -1.0,
                "window_mjd": [57490.0, 57530.0],
                "waveform": "d = sign 10^log10_Amp H(t - t0) exp(-(t - t0)/(10^log10_tau d)) (1400/nu)^idx, H(0) = 1",
                "prov": ["script_dip", "dip_call", "dip_t0", "dip_amp", "dip_tau", "dip_sign", "dip_wf"]},
        "parameters": params,
        "reference_chains": ref,
    }
    for m, nm in (("crn", names_crn), ("hd", names_hd)):
        if [q["name"] for q in params if m in q["models"]] != nm:
            sys.exit(f"{m}: parameter order differs from pars.txt")
    man["physical_log_prior"] = {m: float(-sum(np.log(q["bounds"][1] - q["bounds"][0]) for q in params if m in q["models"]))
                                 for m in ("crn", "hd")}
    return man


def prior_volume_check(man: dict) -> dict:
    out = {}
    for m, key in (("crn", "crn_pl"), ("hd", "hd_pl")):
        names, X, burn = epta.load_reference(key, man)
        lo = np.array([q["bounds"][0] for q in man["parameters"] if m in q["models"]])
        hi = np.array([q["bounds"][1] for q in man["parameters"] if m in q["models"]])
        th = X[:, :67]
        inside = np.all((th >= lo) & (th <= hi), axis=1)
        logged = X[:, 67] - X[:, 68]
        ours = man["physical_log_prior"][m]
        d = logged - ours
        out[key] = {"rows": int(len(X)), "rows_outside_box": int((~inside).sum()), "logged_mean": float(logged.mean()),
                    "ours": ours, "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                    "max_abs_diff": float(np.abs(d).max()),
                    "pass": bool(abs(d.mean()) <= 1e-5 and d.std(ddof=1) <= 1e-6 and inside.all())}
    out["tolerance"] = "mean |diff| <= 1e-5 nats, row sd <= 1e-6 (six-decimal storage), every row inside the box (plan Sec. 4.5 item 2)"
    out["pass"] = all(v["pass"] for k, v in out.items() if isinstance(v, dict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    man = generate()
    path = epta.MANIFEST_PATH
    if a.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(man, indent=1))
    else:
        old = json.loads(path.read_text())
        if old != json.loads(json.dumps(man)):
            sys.exit("committed manifest differs from a fresh generation")
    pv = prior_volume_check(man)
    out = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"
    out.mkdir(parents=True, exist_ok=True)
    pv["binding"] = evidence_binding()
    (out / "prior_volume.json").write_text(json.dumps(pv, indent=1))
    print(json.dumps(pv, indent=1))


if __name__ == "__main__":
    main()
