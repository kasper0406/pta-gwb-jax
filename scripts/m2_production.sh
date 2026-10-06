#!/usr/bin/env bash
# Run the M2 production configurations sequentially (one GPU job at a time). Skips finished runs.
#   nohup scripts/m2_production.sh > runs/production.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/logs
RUNS=${RUNS:-"curn_g433_14f curn_vg_14f curn_vg_5f curn_fs30 hd_g433_14f hd_vg_14f hd_vg_5f hd_fs30 hd_g433_14f_icrs hd_vg_14f_icrs"}
for r in $RUNS; do
  if [ -f "runs/$r/meta.json" ] && grep -q '"finished"' "runs/$r/meta.json"; then
    echo "$(date -u +%FT%T) skip $r (finished)"; continue
  fi
  echo "$(date -u +%FT%T) start $r"
  uv run --no-sync python scripts/m2_run.py "configs/m2/$r.json" > "runs/logs/$r.log" 2>&1
  echo "$(date -u +%FT%T) end $r (exit $?)"
done
echo "$(date -u +%FT%T) all done"
