#!/usr/bin/env bash
# Focused completion pass, ordered by what the paper still needs:
#   scale   -> heterogeneity table (the last visible placeholder)
#   nodes   -> federation-size columns of the topology table
#   privacy -> completes the Edge-IIoTset column of the privacy table
#   poison  -> ensures the adaptive_dist cells are in the main comparison tables
# Resumable: finished cells are skipped. 8 workers, threads pinned.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=v2/results/progress_v3.log
for g in scale nodes privacy poison main; do
  echo "FINISH2 $g $(date +%H:%M:%S)" >> "$LOG"
  for pass in 1 2; do
    python -I v2/run.py --grid "$g" --workers 8 --out runs_v3.jsonl >> "$LOG" 2>&1 && break
  done
done
echo "FINISH2 COMPLETE $(date +%H:%M:%S)" >> "$LOG"
