from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from experiments.run_pilot import load_cicddos2019_binary, local_partitions, run_gossip


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "paper_pilot.yaml").read_text(encoding="utf-8"))
    cfg["data"]["sample_size"] = 16000
    out_dir = ROOT / "results_tuning"
    bundle = load_cicddos2019_binary(
        (ROOT / cfg["data"]["zip_path"]).resolve(),
        int(cfg["data"]["sample_size"]),
        float(cfg["data"]["test_size"]),
        float(cfg["data"]["validation_size"]),
        int(cfg["project"]["seed"]),
    )

    variants = [
        {"rounds": 8, "neighbors": 9, "trust": 0.0, "self": 1.0, "peer": 1.0, "sample": False, "adaptive": True},
        {"rounds": 10, "neighbors": 9, "trust": 0.0, "self": 1.0, "peer": 1.0, "sample": False, "adaptive": True},
        {"rounds": 12, "neighbors": 9, "trust": 0.0, "self": 1.0, "peer": 1.0, "sample": False, "adaptive": True},
        {"rounds": 8, "neighbors": 9, "trust": 0.0, "self": 0.4, "peer": 0.9, "sample": False, "adaptive": True},
        {"rounds": 8, "neighbors": 9, "trust": 0.0, "self": 1.0, "peer": 1.0, "sample": True, "adaptive": True},
        {"rounds": 8, "neighbors": 9, "trust": 0.0, "self": 1.0, "peer": 1.0, "sample": False, "adaptive": False},
    ]

    for idx, variant in enumerate(variants):
        local_cfg = copy.deepcopy(cfg)
        local_cfg["training"]["rounds"] = variant["rounds"]
        local_cfg["federation"]["neighbors_per_round"] = variant["neighbors"]
        local_cfg["federation"]["trust_threshold"] = variant["trust"]
        local_cfg["federation"]["self_trust_weight"] = variant["self"]
        local_cfg["federation"]["min_peer_trust_weight"] = variant["peer"]
        local_cfg["federation"]["sample_weighted_gossip"] = variant["sample"]
        local_cfg["federation"]["adaptive_threshold"] = variant["adaptive"]
        partitions = local_partitions(bundle, local_cfg)
        row = run_gossip(bundle, partitions, local_cfg, out_dir, f"ShieldDFL-tune-{idx}", shield=True)
        print(
            idx,
            variant,
            "acc",
            round(float(row["accuracy"]), 4),
            "f1",
            round(float(row["f1_macro"]), 4),
            "auc",
            round(float(row["auc"]), 4),
            "fpr",
            round(float(row["fpr"]), 4),
            "recall",
            round(float(row["attack_recall"]), 4),
        )


if __name__ == "__main__":
    main()

