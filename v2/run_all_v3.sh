#!/usr/bin/env bash
# Full evaluation for the SN Computer Science version: five seeds, eleven attacks
# (three of them written against ShieldDFL's own gates), malicious fractions to 50%,
# a degree sweep, a federation-size sweep and ablations under the adaptive attacks.
#
# Grids run in order of importance, so an interrupted sweep still leaves the core
# tables complete. Resumable: every finished run is a line in results/runs_v3.jsonl
# and is skipped on restart. Worker count is held at 8: each PyTorch process commits
# about 1.3 GB, and 12 of them exhausted the Windows paging file.
cd "$(dirname "$0")/.."
mkdir -p v2/results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=v2/results/progress_v3.log
for g in main poison ablation breaking privacy scale nodes; do
  for pass in 1 2 3; do
    echo "GRID $g pass $pass $(date +%H:%M:%S)" >> "$LOG"
    python -I v2/run.py --grid "$g" --workers 8 --out runs_v3.jsonl >> "$LOG" 2>&1 && break
  done
done
echo "ALL GRIDS COMPLETE $(date +%H:%M:%S)" >> "$LOG"
