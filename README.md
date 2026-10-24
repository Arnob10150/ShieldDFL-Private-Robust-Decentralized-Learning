# ShieldDFL

Reference implementation of **ShieldDFL**, a trust-aware, poisoning-resilient decentralized
federated learning framework for privacy-preserving DDoS detection, and of the twelve poisoning
attacks it is evaluated against (including the four gate-aware attacks written against its own
screening rule).

This repository releases the framework, the preprocessing, the experimental configurations and
the evaluation code so that every table and figure in the paper can be reproduced from the raw
public datasets.

## Contents

| File | What it is |
|---|---|
| `shield.py` | the framework: the flow classifier, DP-SGD with Rényi accounting, the four-part screening rule (robust radius, validation gate, distance gate, soft reputation) and clipped trusted mixing, every baseline aggregation rule (gossip, median, trimmed mean, Krum, ClippedGossip, BALANCE), the twelve attacks, and the metrics |
| `run.py` | the experiment driver: the non-IID split, the DP calibration per node, one full decentralized run, and the grids (`main`, `poison`, `ablation`, `breaking`, `privacy`, `scale`, `nodes`) |
| `prepare_data.py` | preprocessing: drop identifier/address/port/timestamp fields, coerce to numeric, remove exact duplicate rows **before** any split, draw a class-balanced sample, write `data/<name>.npz` |
| `run_all.sh` | runs every grid in order; resumable |
| `requirements.txt` | dependencies (CPU is sufficient) |

## Datasets

Three public datasets are used, each downloaded from its official source:

- **CICDDoS2019** and **CICIDS2018** — Canadian Institute for Cybersecurity
  (<https://www.unb.ca/cic/datasets/>)
- **Edge-IIoTset** — <https://ieee-dataport.org/documents/edge-iiotset>

Place the three archives where `prepare_data.py` expects them (see the `RAW` path at the top of
that file), then build the `.npz` files:

```bash
pip install -r requirements.txt
python -I prepare_data.py      # writes data/*.npz and data/summary.json
```

## Reproducing the experiments

```bash
bash run_all.sh                # or run one grid: python -I run.py --grid poison --workers 8
```

Every finished run is appended as one JSON line to `results/runs.jsonl` and is skipped on
restart, so the grid can be interrupted and resumed. Each line records the full configuration
(dataset, method, attack, malicious fraction, DP budget, label-skew concentration, node count,
degree, seed, ablation) together with every metric (accuracy, worst-node accuracy, macro-F1,
AUC, recall, false-positive rate, recall at 1% FPR, per-node spread, communication, wall time
and the learning curve), so the results are fully auditable.

## Configuration

All settings are in `run.py`:

- model: four-layer perceptron (128–64–32) with layer normalization, ReLU, dropout 0.2
- optimizer: Adam, lr 1e-3, weight decay 1e-4, batch 64
- DP-SGD: clip `C = 1`, `delta = 1e-5`, noise multiplier solved per node by bisection to meet
  the target `epsilon` with the Opacus Rényi accountant
- federation: 20 nodes, minimum degree 6, random graph re-drawn every round, 30 rounds
- label skew: Dirichlet `alpha = 0.5` by default; swept over {0.1, 1, 100}
- seeds: 0–4
- ShieldDFL: `lambda = 2`, `kappa = 2.5`, `beta = 0.8`, `R0 = 0.5`

## Citation

If you use this code, please cite the paper (see the repository description for the reference).
