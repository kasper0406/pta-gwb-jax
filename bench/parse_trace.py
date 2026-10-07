"""Summarise a jax.profiler perfetto trace: GPU kernel time by name and by stage category.

    uv run --no-sync python bench/parse_trace.py bench/traces/jax_hd_gpu [n_calls]
"""
import collections
import glob
import gzip
import json
import re
import sys

d = sys.argv[1]
ncalls = int(sys.argv[2]) if len(sys.argv) > 2 else 1
f = max(glob.glob(f"{d}/**/perfetto_trace.json.gz", recursive=True))
ev = json.load(gzip.open(f))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
# find GPU stream threads: pid names containing "/device:GPU"
pnames = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
gpu_pids = {p for p, n in pnames.items() if "GPU" in n or "gpu" in n}
k = [e for e in ev if e.get("ph") == "X" and e.get("pid") in gpu_pids]
by = collections.defaultdict(lambda: [0.0, 0])
for e in k:
    by[e["name"]][0] += e["dur"]
    by[e["name"]][1] += 1
tot = sum(v[0] for v in by.values())
CATS = [
    ("cholesky (potrf/getrf_wo_pivot)", r"potrf|chol|getrf"),
    ("QR (geqrf/orgqr/larf/householder)", r"geqr|orgqr|orm|larf|householder|qr"),
    ("triangular solve (trsm/trsv)", r"trsm|trsv|triangular"),
    ("GEMM", r"gemm|cutlass|xmma|sm\d+_.*(nn|nt|tn|tt)|dot|matmul"),
]
cat = collections.defaultdict(float)
for n, (t, c) in by.items():
    for cn, rx in CATS:
        if re.search(rx, n, re.IGNORECASE):
            cat[cn] += t
            break
    else:
        cat["other (fusions, copies, small kernels)"] += t
print(f"trace {f}\nGPU kernel busy time per call: {tot / ncalls / 1e3:.3f} ms over {len(k) / ncalls:.0f} kernels/call")
for cn, t in sorted(cat.items(), key=lambda x: -x[1]):
    print(f"  {cn:45s} {t / ncalls / 1e3:8.3f} ms  ({100 * t / tot:4.1f}%)")
print("top kernels:")
for n, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:25]:
    print(f"  {t / ncalls / 1e3:8.3f} ms  {c / ncalls:6.1f}/call  {n[:110]}")
