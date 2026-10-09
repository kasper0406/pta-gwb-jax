"""Verified migration of gate results to binding scheme 2 (review round 4), without recomputing them.

Scheme 2 (``ptagwb.binding``) excludes the control plane (budget, supervisor, binding bookkeeping,
the run driver, this script) from the gate-evidence binding. A result computed under the legacy
scheme at commit ``C`` is re-stamped only if all of the following hold, otherwise it is left
untouched (stale: recompute it):

1. its recorded binding equals the **legacy** binding (all code, no exclusion) recomputed from the
   git tree of ``C`` together with the current inputs, runtime, libraries and oracle envs; this
   proves the result was produced from ``C``'s code and configs and from today's inputs;
2. the scheme-2 binding of ``C``'s tree equals the scheme-2 binding of the current work tree, i.e.
   every bound (non-control-plane) file is byte-identical to ``C``: only control-plane files differ.

The re-stamped file keeps its numbers, gets ``binding`` = the current scheme-2 binding and records
the migration (``binding_migration``: commit, legacy binding, files that differ from ``C``, all in
the control plane). Usage: PYTHONPATH=src python scripts/m3b_rebind_evidence.py COMMIT [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from ptagwb.binding import (
    CONTROL_PLANE,
    SCHEME,
    code_binding,
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


def bindings_at(commit: str, external: dict) -> tuple[dict, dict]:
    listing, read = tree_reader(commit)
    legacy = {**code_binding(listing, read, REPO_ROOT, exclude=()), **external}
    scheme2 = {"scheme": SCHEME, **code_binding(listing, read, REPO_ROOT), **external}
    return legacy, scheme2


def changed_since(commit: str) -> list[str]:
    """Files (of the legacy bound set) whose work-tree content differs from ``commit``."""
    from ptagwb.binding import code_file_sets

    listing, read = tree_reader(commit)
    now = set(worktree_listing())
    then = set(listing)
    sets_now, sets_then = code_file_sets(now, exclude=()), code_file_sets(then, exclude=())
    out = []
    for k in sets_now:
        a, b = set(sets_now[k]), set(sets_then[k])
        out += sorted(a ^ b)
        for r in sorted(a & b):
            if (REPO_ROOT / r).read_bytes() != read(r):
                out.append(r)
    return sorted(set(out))


def migrate(commit: str, results_dir: Path = RES, dry_run: bool = False) -> dict:
    commit = _git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    external = external_binding()
    legacy, scheme2_then = bindings_at(commit, external)
    current = evidence_binding()
    diff = changed_since(commit)
    report = {"commit": commit, "changed_files": diff, "files": {}}
    if scheme2_then != current or any(r not in CONTROL_PLANE for r in diff):
        report["refused"] = ("bound (non-control-plane) files differ from the commit: "
                             f"{[r for r in diff if r not in CONTROL_PLANE]}; recompute the gates")
        return report
    for p in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(p.read_text())
        b = r.get("binding")
        if b is None:
            continue
        if b == current:
            report["files"][p.name] = "already current"
            continue
        if b != legacy:
            report["files"][p.name] = "STALE: recorded binding is not the legacy binding of the commit (recompute)"
            continue
        report["files"][p.name] = "rebound"
        if dry_run:
            continue
        r["binding"] = current
        r["binding_migration"] = {"from_commit": commit, "legacy_binding": b, "files_changed_since": diff,
                                  "all_changed_files_in_control_plane": True,
                                  "rule": "scripts/m3b_rebind_evidence.py (review round 4)"}
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
