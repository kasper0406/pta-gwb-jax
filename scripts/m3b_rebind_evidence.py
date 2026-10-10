"""Verified migration of gate results to the current binding scheme, without recomputing them
(review rounds 4 and 5).

Binding schemes (``ptagwb.binding.EXCLUSIONS``): 1 binds all code; 2 excludes the control plane
(budget, supervisor, binding bookkeeping, run driver, this script); 3 also excludes the sampler-run
plane (tuning-artifact generator, pilot report, run configs, metrics, proposals). A result whose
recorded binding has scheme ``k`` (no ``scheme`` key: 1) is re-stamped from commit ``C`` only if
all of the following hold, otherwise it is left untouched (stale: recompute it):

1. its recorded binding equals the scheme-``k`` binding recomputed from the git tree of ``C``
   together with the current inputs, runtime, libraries and oracle envs; this proves the result was
   produced from code and configs equal to ``C``'s (under scheme ``k``) and from today's inputs;
2. the current-scheme binding of ``C``'s tree equals the current-scheme binding of the work tree:
   every file bound now is byte-identical to ``C``, and every file that differs from ``C`` (in the
   scheme-1 set) is excluded by the current scheme.

The re-stamped file keeps its numbers, gets ``binding`` = the current binding and appends the
migration to ``binding_migrations`` (commit, from/to scheme, from/to binding, files that differ from
``C``). A legacy single ``binding_migration`` record is moved into that list first.
Usage: PYTHONPATH=src python scripts/m3b_rebind_evidence.py COMMIT [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from ptagwb.binding import (
    EXCLUSIONS,
    SCHEME,
    code_binding,
    code_file_sets,
    evidence_binding,
    external_binding,
    worktree_listing,
)
from ptagwb.config import REPO_ROOT

RES = REPO_ROOT / "data" / "processed" / "m3b" / "epta" / "results"


def _git(*args, inp=None) -> bytes:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=True, input=inp).stdout


def tree_reader(commit: str):
    listing = _git("ls-tree", "-r", "--name-only", commit, "--", "src/ptagwb", "scripts", "configs/m3b",
                   "tests").decode().split("\n")
    listing = [x for x in listing if x]

    def read(rel: str) -> bytes:
        return _git("show", f"{commit}:{rel}")

    return listing, read


def binding_at(commit: str, external: dict, scheme: int) -> dict:
    """The scheme-``scheme`` binding of ``commit``'s tree with the given external part."""
    listing, read = tree_reader(commit)
    head = {} if scheme == 1 else {"scheme": scheme}
    return {**head, **code_binding(listing, read, REPO_ROOT, scheme), **external}


def bindings_at(commit: str, external: dict) -> tuple[dict, dict]:
    """(scheme-1 binding, current-scheme binding) of ``commit`` (kept for the strict test)."""
    return binding_at(commit, external, 1), binding_at(commit, external, SCHEME)


def changed_since(commit: str) -> list[str]:
    """Files of the scheme-1 bound set whose work-tree content differs from ``commit``."""
    listing, read = tree_reader(commit)
    sets_now, sets_then = code_file_sets(worktree_listing(), 1), code_file_sets(listing, 1)
    out = []
    for k in sets_now:
        a, b = set(sets_now[k]), set(sets_then[k])
        out += sorted(a ^ b)
        for r in sorted(a & b):
            if (REPO_ROOT / r).read_bytes() != read(r):
                out.append(r)
    return sorted(set(out))


def excluded_now(rel: str) -> bool:
    files, prefixes = EXCLUSIONS[SCHEME]
    return rel in files or rel.startswith(prefixes)


def migrate(commit: str, results_dir: Path = RES, dry_run: bool = False) -> dict:
    commit = _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    external = external_binding()
    current = evidence_binding()
    target = binding_at(commit, external, SCHEME)
    diff = changed_since(commit)
    report = {"commit": commit, "scheme": SCHEME, "changed_files": diff, "files": {}}
    bad = [r for r in diff if not excluded_now(r)]
    if target != current or bad:
        report["refused"] = f"bound files differ from the commit: {bad}; recompute the gates"
        return report
    cache: dict[int, dict] = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(p.read_text())
        b = r.get("binding")
        if b is None:
            continue
        if b == current:
            report["files"][p.name] = "already current"
            continue
        k = int(b.get("scheme", 1))
        if k not in EXCLUSIONS or k >= SCHEME:
            report["files"][p.name] = f"STALE: recorded scheme {k} cannot be migrated (recompute)"
            continue
        if k not in cache:
            cache[k] = binding_at(commit, external, k)
        if b != cache[k]:
            report["files"][p.name] = (f"STALE: recorded binding is not the scheme-{k} binding of the commit "
                                       "(recompute)")
            continue
        report["files"][p.name] = f"rebound (scheme {k} -> {SCHEME})"
        if dry_run:
            continue
        hist = list(r.pop("binding_migrations", []))
        old = r.pop("binding_migration", None)
        if old is not None:  # round-4 record (scheme 1 -> 2)
            hist.insert(0, {"from_commit": old["from_commit"], "from_scheme": 1, "to_scheme": 2,
                            "from_binding": old["legacy_binding"], "to_binding": b,
                            "files_changed_since": old["files_changed_since"], "rule": old["rule"]})
        hist.append({"from_commit": commit, "from_scheme": k, "to_scheme": SCHEME, "from_binding": b,
                     "to_binding": current, "files_changed_since": diff,
                     "rule": "scripts/m3b_rebind_evidence.py (review rounds 4-5)"})
        r["binding"] = current
        r["binding_migrations"] = hist
        fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(r, f, indent=1)
        os.replace(tmp, p)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rep = migrate(a.commit, dry_run=a.dry_run)
    print(json.dumps(rep, indent=1))
    if "refused" in rep or any(v.startswith("STALE") for v in rep["files"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
