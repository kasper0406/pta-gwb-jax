#!/usr/bin/env bash
# HD free-spectrum re-run (review MAJOR 1) + acceptance diagnostics. Needs the GPU for ~13-21 h
# (see docs/M2_RESULTS.md Sec. 10); NOT run yet. Usage:
#   nohup scripts/m2_rerun_fs30.sh > runs/logs/hd_fs30_v2.driver.log 2>&1 &
# CONFIG may point to another sampler-agnostic config with the same model block.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/m2/hd_fs30_v2.json}
NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$CONFIG")
mkdir -p runs/logs
if ! { [ -f "runs/$NAME/meta.json" ] && grep -q '"finished"' "runs/$NAME/meta.json"; }; then
  uv run --no-sync python scripts/m2_run.py "$CONFIG" > "runs/logs/$NAME.log" 2>&1
fi
# Acceptance gate (CPU only): exits nonzero, and so does this script, on any failure or missing
# diagnostic (all-parameter R-hat/ESS, per-bin occupancy indicators, tail stability).
set +e
JAX_PLATFORMS=cpu uv run --no-sync python scripts/m2_freespec_diag.py --run "$NAME" > "outputs/m2/freespec_gate_$NAME.txt" 2>&1
status=$?
set -e
cat "outputs/m2/freespec_gate_$NAME.txt"
if [ "$status" -ne 0 ]; then echo "hd_fs30 re-run REJECTED by the acceptance gate (exit $status)"; fi
exit "$status"
