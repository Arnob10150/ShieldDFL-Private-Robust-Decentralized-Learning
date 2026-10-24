#!/usr/bin/env bash
# Reproduce the full evaluation. Resumable: every finished run is a line in
# results/runs.jsonl and is skipped on restart. 8 workers keeps memory bounded
# (each PyTorch process commits roughly 1.3 GB).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for g in main poison ablation breaking privacy scale nodes; do
  python -I run.py --grid "$g" --workers 8 --out runs.jsonl
done
echo "all grids complete"
