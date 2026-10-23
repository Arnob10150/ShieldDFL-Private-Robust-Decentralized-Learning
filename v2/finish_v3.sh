#!/usr/bin/env bash
# Catch-up pass. run_all_v3.sh fixes each grid's work list when that grid starts, so cells
# added to a grid after it has already run are never picked up. This waits for the main
# sweep to finish and then re-runs every grid once; completed cells are skipped, so it only
# executes what is genuinely missing (the adaptive_dist poisoning cells, in particular).
cd "$(dirname "$0")/.."
mkdir -p v2/results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=v2/results/progress_v3.log

until grep -q "ALL GRIDS COMPLETE" "$LOG" 2>/dev/null; do sleep 60; done

echo "CATCH-UP PASS START $(date +%H:%M:%S)" >> "$LOG"
for g in poison ablation breaking privacy scale nodes main; do
  for pass in 1 2; do
    python -I v2/run.py --grid "$g" --workers 8 --out runs_v3.jsonl >> "$LOG" 2>&1 && break
  done
done
echo "CATCH-UP COMPLETE $(date +%H:%M:%S)" >> "$LOG"
