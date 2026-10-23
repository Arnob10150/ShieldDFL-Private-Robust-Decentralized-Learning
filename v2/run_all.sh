#!/usr/bin/env bash
# Resumable: every finished run is in results/runs.jsonl and is skipped on restart.
# Uses 1 worker while the AJSE training is running, 4 once it has finished.
cd "$(dirname "$0")/.."
workers() { if powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? { \$_.CommandLine -match 'Yash_AJSE|run_experiments|concat_members' }" | grep -q .; then echo 1; else echo 2; fi; }
for g in main poison privacy ablation scale; do
  for pass in 1 2 3; do
    echo "GRID $g pass $pass workers $(workers) $(date +%H:%M)" >> v2/results/progress.log
    python -I v2/run.py --grid $g --workers $(workers) >> v2/results/progress.log 2>&1 && break
  done
done
echo "ALL GRIDS COMPLETE $(date +%H:%M)" >> v2/results/progress.log
