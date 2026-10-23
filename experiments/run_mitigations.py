from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from experiments.run_pilot import (
    clear_logs,
    estimated_dp_epsilon,
    local_partitions,
    load_cicddos2019_binary,
    noise_for_target_epsilon,
    run_centralized,
    run_gossip,
    run_local_only,
    run_server_fl,
    row_from_metrics,
    write_dataset_tables,
)


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def normalize_row(row: dict, experiment_id: str) -> dict:
    out = dict(row)
    out["experiment_id"] = experiment_id
    return out


def metric_row(metrics: dict, method: str, experiment_id: str, cfg: dict) -> dict:
    return row_from_metrics(
        metrics,
        method,
        experiment_id,
        int(cfg["project"]["seed"]),
        float(cfg["clients"]["dirichlet_alpha"]),
        int(cfg["clients"]["num_clients"]),
        int(cfg["training"]["rounds"]),
    )


def summarize(raw_path: Path, summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    numeric = ["accuracy", "f1_macro", "auc", "fpr", "attack_recall", "communication_mb"]
    grouped = (
        df.groupby(["experiment_id", "method"], dropna=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    grouped.to_csv(summary_path, index=False)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_pilot.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "results_mitigated"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    clear_logs(out_dir)
    for extra in ["logs/mitigation_raw.csv", "tables/mitigation_summary.csv", "logs/mitigation_manifest.json"]:
        path = out_dir / extra
        if path.exists():
            path.unlink()

    raw_rows: list[dict] = []
    manifest = {
        "seeds": args.seeds,
        "sample_size": args.sample_size,
        "rounds": args.rounds,
        "mitigations": [
            "three-seed confidence summaries",
            "8-round convergence instead of 5 rounds",
            "Opacus RDP epsilon accounting",
            "personalized classifier heads",
            "Sybil diversity filtering by update cosine similarity",
        ],
    }

    for seed in args.seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["project"]["seed"] = seed
        cfg["data"]["sample_size"] = args.sample_size
        cfg["training"]["rounds"] = args.rounds
        cfg["training"]["centralized_epochs"] = args.rounds

        bundle = load_cicddos2019_binary(
            (ROOT / cfg["data"]["zip_path"]).resolve()
            if not Path(cfg["data"]["zip_path"]).is_absolute()
            else Path(cfg["data"]["zip_path"]),
            int(cfg["data"]["sample_size"]),
            float(cfg["data"]["test_size"]),
            float(cfg["data"]["validation_size"]),
            seed,
        )
        partitions = local_partitions(bundle, cfg)
        if seed == args.seeds[0]:
            write_dataset_tables(bundle, partitions, cfg, out_dir)

        rows = [
            metric_row(run_centralized(bundle, cfg, out_dir), "Centralized MLP", "mitigated_clean", cfg),
            metric_row(run_local_only(bundle, partitions, cfg, out_dir), "Local-only", "mitigated_clean", cfg),
            normalize_row(run_server_fl(bundle, partitions, cfg, out_dir, "FedAvg"), "mitigated_clean"),
            normalize_row(run_server_fl(bundle, partitions, cfg, out_dir, "FedProx"), "mitigated_clean"),
            normalize_row(run_gossip(bundle, partitions, cfg, out_dir, "P2P Gossip FL"), "mitigated_clean"),
            normalize_row(run_gossip(bundle, partitions, cfg, out_dir, "ShieldDFL", shield=True), "mitigated_clean"),
            normalize_row(
                run_gossip(
                    bundle,
                    partitions,
                    cfg,
                    out_dir,
                    "ShieldDFL-PH+Sybil",
                    shield=True,
                    personalized_head=True,
                    sybil_similarity_threshold=0.985,
                ),
                "mitigated_clean",
            ),
        ]

        noise, accounted = noise_for_target_epsilon(
            cfg,
            int(np.mean([len(partition) for partition in partitions])),
            8.0,
        )
        rows.append(
            normalize_row(
                run_gossip(
                    bundle,
                    partitions,
                    cfg,
                    out_dir,
                    "ShieldDFL-PH+Sybil+DP",
                    shield=True,
                    personalized_head=True,
                    sybil_similarity_threshold=0.985,
                    dp_noise=noise,
                    epsilon=round(accounted, 3),
                ),
                "mitigated_privacy_e8",
            )
        )
        rows[-1]["notes"] = f"target_epsilon=8; noise_multiplier={noise:.4f}; opacus_rdp_accounted"

        rows.extend(
            [
                normalize_row(
                    run_server_fl(
                        bundle,
                        partitions,
                        cfg,
                        out_dir,
                        "FedAvg",
                        malicious_ratio=0.10,
                        attack_type="mixed",
                    ),
                    "mitigated_poison10",
                ),
                normalize_row(
                    run_gossip(
                        bundle,
                        partitions,
                        cfg,
                        out_dir,
                        "ShieldDFL-PH+Sybil",
                        shield=True,
                        malicious_ratio=0.10,
                        attack_type="mixed",
                        personalized_head=True,
                        sybil_similarity_threshold=0.985,
                    ),
                    "mitigated_poison10",
                ),
            ]
        )
        raw_rows.extend(rows)
        append_rows(out_dir / "logs" / "mitigation_raw.csv", rows)

    summary = summarize(out_dir / "logs" / "mitigation_raw.csv", out_dir / "tables" / "mitigation_summary.csv")
    manifest["summary_rows"] = int(len(summary))
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / "mitigation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
