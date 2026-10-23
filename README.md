# ShieldDFL

Reference code for *ShieldDFL: A Trust-Aware and Poisoning-Resilient Decentralized Federated Learning Framework for Privacy-Preserving DDoS Detection*.

## Contents

`v2/` contains the experiment implementation used to generate the manuscript's results (`v2/results/runs_v3.jsonl`): the ShieldDFL method, baselines, attacks, dataset preparation, and the V3 experiment launch scripts. Manuscript composition and table/figure-generation scripts are maintained separately in the paper directory and are not included here. Dataset archives and generated data/results are not included.

## Requirements

Python 3, NumPy, PyTorch, scikit-learn, and (for dataset preparation) pandas. The full evaluation launch scripts use Bash.

## Reproduction

1. Put the three source dataset archives where `v2/prepare_data.py` can find them (the script currently looks in the project directory's parent).
2. Run `python -I v2/prepare_data.py` to create `v2/data/*.npz`.
3. From the repository root, run `bash v2/run_all_v3.sh` to execute the manuscript evaluation grids. Results and logs are written under `v2/results/`.

The full sweep is compute-intensive. `v2/run_all_v3.sh` uses eight workers and can resume completed runs. `v2/finish_v3.sh` is the optional catch-up pass for grid cells added after the main sweep began.

Repository: https://github.com/Arnob10150/ShieldDFL-Private-Robust-Decentralized-Learning
