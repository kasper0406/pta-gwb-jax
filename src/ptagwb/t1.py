"""Gate T1: exactness of every array a published-model reproduction consumes (docs/M3B_PLAN.md
Sec. 4.1, decision D1).

Compares two exports of one pulsar (npz dictionaries with the keys written by
``scripts/m3b_t2_export.py`` / ``scripts/m3b_epta_oracle.py export``), e.g. our tempo2 export of
the canonical files against the pinned enterprise built from the released files. Pure numpy; no
oracle package is imported here.

Checks and tolerances (fixed before any comparison was run):

=================  ===========================================================  ==============
array              check                                                        tolerance
=================  ===========================================================  ==============
roster / rows      TOA count, deletion-mask count and indices, row order        identical
toas (TCB BAT, s)  max |diff|                                                   <= 1e-9 s
stoas (s)          max |diff|                                                   <= 1e-9 s
residuals (s)      max |diff|                                                   <= 1e-12 s
toaerrs (s)        max |diff|                                                   <= 1e-12 s
freqs (SSB, MHz)   max |rel diff| (barycentric convention, as enterprise)       <= 1e-12
backend flags      per TOA                                                      identical
flags              every flag of every TOA, except listed canonicalisation      identical
                   rules (``FLAG_RULES``)
Mmat               fit-parameter list; column-space principal sines; numerical  sin <= 1e-10;
                   rank (column-normalised singular values > 1e-10 s_max)       rank identical
pos                max |diff| of the ORF unit vector                            <= 1e-12
pos_t, sunssb,     max |diff| relative to the array's max |value| (SW geometry; <= 1e-12
planetssb          reported, consumed only by models with SW terms)
=================  ===========================================================  ==============
"""

from __future__ import annotations

import json

import numpy as np

TOL = {"toas": 1e-9, "stoas": 1e-9, "residuals": 1e-12, "toaerrs": 1e-12, "freqs_rel": 1e-12, "mmat_sin": 1e-10,
       "mmat_rank_rtol": 1e-10, "pos": 1e-12, "geom_rel": 1e-12}

# canonicalisation rules that legitimately change flags (M3A_VALIDATION.md Sec. 1): the flat tim
# drops tempo2 flag artefacts such as "-.cal" parsed from archive names inside the TOA fields, and
# writes the effective TIME offset of each TOA as an explicit "-to" flag (its effect is checked
# exactly by the stoas / toas comparison, so only the flag text is exempt)
FLAG_RULES = {"drop_artefact_flags": lambda k: k.startswith("."), "time_offset_as_to": lambda k: k == "to"}


def _principal_sines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    def basis(X):
        X = X / np.linalg.norm(X, axis=0)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        return U[:, s > s[0] * 1e-13]

    # sines from the residual of B's basis after projection on A's (accurate for small angles,
    # unlike sqrt(1 - cos^2), whose floor is ~1e-8)
    Ua, Ub = basis(A), basis(B)
    R = Ub - Ua @ (Ua.T @ Ub)
    return np.sort(np.linalg.svd(R, compute_uv=False))[::-1]


def _rank(M: np.ndarray, rtol: float) -> int:
    s = np.linalg.svd(M / np.linalg.norm(M, axis=0), compute_uv=False)
    return int(np.sum(s > s[0] * rtol))


def _flags(d) -> dict:
    return {k.lstrip("-"): v for k, v in json.loads(str(d["flags_json"])).items()}


def _deleted(d) -> np.ndarray:
    return np.asarray(d["deleted"] if "deleted" in d else d["t2_deleted"], dtype=bool)


def compare(a, b, *, geometry_consumed: bool = False,
            flag_rules: tuple = ("drop_artefact_flags", "time_offset_as_to")) -> dict:
    """T1 comparison of exports ``a`` and ``b`` (npz-like mappings). Returns the metrics, a list
    of failures and ``ok``. ``geometry_consumed``: whether pos_t / sunssb / planetssb gate (SW
    terms in the model) or are only reported."""
    out: dict = {"name": str(a["name"]), "fail": [], "notes": []}
    n_a, n_b = len(a["toas"]), len(b["toas"])
    out["n"] = (n_a, n_b)
    if n_a != n_b or str(a["name"]) != str(b["name"]):
        out["fail"].append(f"roster: {str(a['name'])}/{n_a} vs {str(b['name'])}/{n_b}")
        out["ok"] = False
        return out
    da, db = _deleted(a), _deleted(b)
    out["n_deleted"] = (int(da.sum()), int(db.sum()))
    if not np.array_equal(da, db):
        out["fail"].append("deletion mask differs")
    for k in ("toas", "stoas", "residuals", "toaerrs"):
        d = float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
        out[f"max_abs_{k}"] = d
        if not d <= TOL[k]:
            out["fail"].append(f"{k}: {d:.3e} > {TOL[k]:.0e}")
    # row order: identical sorted epochs imply identical order up to exact ties; check ties
    t = np.asarray(a["toas"])
    out["n_tied_epochs"] = int(np.sum(np.diff(t) == 0))
    fr = float(np.max(np.abs(np.asarray(a["freqs"]) / np.asarray(b["freqs"]) - 1.0)))
    out["max_rel_freqs"] = fr
    if not fr <= TOL["freqs_rel"]:
        out["fail"].append(f"freqs (SSB): rel {fr:.3e}")
    if not np.array_equal(np.asarray(a["backend_flags"]).astype(str), np.asarray(b["backend_flags"]).astype(str)):
        out["fail"].append("backend flags differ")
    fa, fb = _flags(a), _flags(b)
    rules = [FLAG_RULES[r] for r in flag_rules]
    dropped = sorted({k for k in set(fa) | set(fb) if any(r(k) for r in rules)
                      and (k not in fa or k not in fb or list(map(str, fa[k])) != list(map(str, fb[k])))})
    if dropped:
        out["notes"].append(f"flags removed by listed rules: {dropped}")
    keys = sorted(k for k in set(fa) | set(fb) if not any(r(k) for r in rules))
    bad = []
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        if va is None or vb is None:
            # a flag present on no TOA of one side but all-empty on the other is equivalent
            v = va if va is not None else vb
            if any(x != "" for x in v):
                bad.append(f"{k} (missing on one side)")
            continue
        if list(map(str, va)) != list(map(str, vb)):
            bad.append(k)
    if bad:
        out["fail"].append(f"flags differ: {bad[:8]}")
    out["n_flags_compared"] = len(keys)
    pa, pb = [str(x) for x in a["fitpars"]], [str(x) for x in b["fitpars"]]
    if pa != pb:
        out["fail"].append("fit-parameter lists differ")
    Ma, Mb = np.asarray(a["Mmat"]), np.asarray(b["Mmat"])
    if Ma.shape != Mb.shape:
        out["fail"].append(f"Mmat shapes {Ma.shape} vs {Mb.shape}")
    else:
        sines = _principal_sines(Ma, Mb)
        out["mmat_max_sin"] = float(sines[0]) if sines.size else 0.0
        out["mmat_max_abs"] = float(np.max(np.abs(Ma - Mb)))
        ra, rb = _rank(Ma, TOL["mmat_rank_rtol"]), _rank(Mb, TOL["mmat_rank_rtol"])
        out["mmat_rank"] = (ra, rb, Ma.shape[1])
        if not out["mmat_max_sin"] <= TOL["mmat_sin"]:
            out["fail"].append(f"Mmat column space: sin {out['mmat_max_sin']:.3e}")
        if ra != rb or ra != Ma.shape[1]:
            out["fail"].append(f"Mmat numerical rank {ra} vs {rb} of {Ma.shape[1]}")
    dp = float(np.max(np.abs(np.asarray(a["pos"]) - np.asarray(b["pos"]))))
    out["max_abs_pos"] = dp
    if not dp <= TOL["pos"]:
        out["fail"].append(f"pos: {dp:.3e}")
    for k in ("pos_t", "sunssb", "planetssb"):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        r = float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
        out[f"max_rel_{k}"] = r
        if not r <= TOL["geom_rel"]:
            (out["fail"] if geometry_consumed else out["notes"]).append(f"{k}: rel {r:.3e}")
    out["ok"] = not out["fail"]
    return out
