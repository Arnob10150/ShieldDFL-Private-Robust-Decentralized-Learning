# ShieldDFL: Private, Robust Decentralized Learning

This folder collects the runnable framework, experiment, configuration, and result-plotting
code. Manuscript composition and table-generation scripts remain in the paper folders.

Reference implementation for the paper *ShieldDFL: A Trust-Aware and Poisoning-Resilient
Decentralized Federated Learning Framework for Privacy-Preserving DDoS Detection*.

## Contents

- `shielddfl/`, `attacks/`, `models/`, and `src/`: framework components and data utilities.
- `experiments/`: pilot, mitigation, and tuning entry points.
- `configs/`: experiment configuration.
- `plots/`: plots generated from experiment results.
- `v2/`: the earlier standalone implementation and its launch scripts.

The data and generated result files are not included. The pilot expects the dataset archive
and results directory paths specified by its configuration.

From the `ShieldDFL` project directory, run the pilot with:

```powershell
python framework_code/experiments/run_pilot.py --config framework_code/configs/paper_pilot.yaml
```

The original project entry points remain in the parent directory; this folder is a clean
copy intended for repository organization and sharing.

Repository: https://github.com/Arnob10150/ShieldDFL-Private-Robust-Decentralized-Learning
