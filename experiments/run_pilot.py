from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from opacus.accountants import RDPAccountant
except ImportError:  # pragma: no cover - optional dependency
    RDPAccountant = None

from attacks.poisoning import corrupt_labels, corrupt_update, malicious_clients
from models.mlp import FlowMLP, clone_model, state_dict_cpu
from shielddfl.aggregation import (
    State,
    add_deltas,
    apply_delta,
    clone_state,
    communication_megabytes,
    flatten_delta,
    mean_delta,
    median_delta,
    state_delta,
    trimmed_mean_delta,
)
from shielddfl.topology import random_gossip_neighbors
from shielddfl.trust_score import trust_score
from src.data import DatasetBundle, load_cicddos2019_binary
from src.metrics import binary_metrics, summarize_client_metrics
from src.seed import set_seed
from src.split_clients import dirichlet_split, iid_split, label_distribution


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def predict_probs(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in make_loader(x, y, batch_size, shuffle=False):
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            probs.append(prob.cpu().numpy())
    y_prob = np.concatenate(probs)
    return y_prob, y


def best_macro_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(y_true, (y_prob >= threshold).astype(np.int64), average="macro", zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int, threshold: float = 0.5) -> dict[str, float]:
    y_prob, _ = predict_probs(model, x, y, batch_size)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return binary_metrics(y, y_pred, y_prob)


def evaluate_with_validation_threshold(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    val_prob, _ = predict_probs(model, x_val, y_val, batch_size)
    threshold = best_macro_f1_threshold(y_val, val_prob)
    metrics = evaluate(model, x_test, y_test, batch_size, threshold=threshold)
    metrics["threshold"] = threshold
    return metrics


def train_local(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    dp_noise: float = 0.0,
    max_grad_norm: float = 1.0,
    proximal_state: State | None = None,
    fedprox_mu: float = 0.0,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for _ in range(epochs):
        for xb, yb in make_loader(x, y, batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if proximal_state is not None and fedprox_mu > 0:
                prox = torch.zeros((), dtype=loss.dtype)
                for name, param in model.named_parameters():
                    if name in proximal_state:
                        prox = prox + torch.sum((param - proximal_state[name]) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if dp_noise > 0:
                for param in model.parameters():
                    if param.grad is not None:
                        scale = dp_noise * max_grad_norm / max(1, xb.shape[0])
                        param.grad.add_(torch.randn_like(param.grad) * scale)
            optimizer.step()


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def clear_logs(out_dir: Path) -> None:
    for rel in [
        "logs/main_results.csv",
        "logs/round_metrics.csv",
        "logs/privacy_results.csv",
        "logs/poisoning_results.csv",
        "logs/communication.csv",
        "logs/ablation_results.csv",
        "tables/client_distribution.csv",
        "tables/dataset_summary.csv",
    ]:
        path = out_dir / rel
        if path.exists():
            path.unlink()


def stable_offset(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100_000


def opacus_rdp_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    if RDPAccountant is None or noise_multiplier <= 0:
        return float("inf")
    accountant = RDPAccountant()
    for _ in range(max(1, steps)):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(accountant.get_epsilon(delta=delta))


def estimated_dp_epsilon(cfg: dict, bundle: DatasetBundle, local_rows: int, noise_multiplier: float) -> float:
    batch_size = int(cfg["training"]["batch_size"])
    sample_rate = min(1.0, batch_size / max(1, local_rows))
    steps_per_epoch = max(1, math.ceil(local_rows / batch_size))
    steps = int(cfg["training"]["rounds"]) * int(cfg["training"]["local_epochs"]) * steps_per_epoch
    return opacus_rdp_epsilon(noise_multiplier, sample_rate, steps, float(cfg["privacy"]["delta"]))


def dp_steps_and_sample_rate(cfg: dict, local_rows: int) -> tuple[int, float]:
    batch_size = int(cfg["training"]["batch_size"])
    sample_rate = min(1.0, batch_size / max(1, local_rows))
    steps_per_epoch = max(1, math.ceil(local_rows / batch_size))
    steps = int(cfg["training"]["rounds"]) * int(cfg["training"]["local_epochs"]) * steps_per_epoch
    return steps, sample_rate


def noise_for_target_epsilon(cfg: dict, local_rows: int, target_epsilon: float) -> tuple[float, float]:
    if RDPAccountant is None:
        configured = float(cfg["privacy"]["noise_by_epsilon"].get(str(target_epsilon), 1.0))
        return configured, float("inf")
    steps, sample_rate = dp_steps_and_sample_rate(cfg, local_rows)
    delta = float(cfg["privacy"]["delta"])
    low, high = 0.05, 20.0
    for _ in range(40):
        mid = (low + high) / 2
        eps = opacus_rdp_epsilon(mid, sample_rate, steps, delta)
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    eps = opacus_rdp_epsilon(high, sample_rate, steps, delta)
    return float(high), float(eps)


def encoder_only_delta(delta: State) -> State:
    return {key: (value if key.startswith("encoder.") else torch.zeros_like(value)) for key, value in delta.items()}


def sybil_diverse_updates(candidates: list[tuple[State, float, int]], threshold: float) -> list[tuple[State, float, int]]:
    if threshold <= 0 or not candidates:
        return candidates
    kept: list[tuple[State, float, int]] = []
    for delta, score, source_id in sorted(candidates, key=lambda item: item[1], reverse=True):
        flat = flatten_delta(delta)
        duplicate = False
        for kept_delta, _ in kept:
            kept_flat = flatten_delta(kept_delta)
            denom = torch.norm(flat) * torch.norm(kept_flat)
            sim = float(torch.dot(flat, kept_flat) / denom) if float(denom) else 0.0
            if sim >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((delta, score, source_id))
    return kept


def gossip_candidate_weight(score: float, source_id: int, partitions: list[np.ndarray], cfg: dict) -> float:
    base = max(0.0, float(score))
    if cfg.get("federation", {}).get("sample_weighted_gossip", False):
        base *= max(1, len(partitions[source_id]))
    return base


def row_from_metrics(
    metrics: dict[str, float],
    method: str,
    experiment_id: str,
    seed: int,
    alpha: float,
    num_clients: int,
    round_id: int,
    dp_enabled: bool = False,
    epsilon: str | float = "inf",
    malicious_ratio: float = 0.0,
    attack_type: str = "none",
    communication_mb: float = 0.0,
    training_time_sec: float = 0.0,
    notes: str = "",
) -> dict:
    row = {
        "experiment_id": experiment_id,
        "seed": seed,
        "dataset": "CICDDoS2019",
        "task": "binary",
        "split_type": "dirichlet",
        "alpha": alpha,
        "num_clients": num_clients,
        "method": method,
        "dp_enabled": dp_enabled,
        "epsilon": epsilon,
        "malicious_ratio": malicious_ratio,
        "attack_type": attack_type,
        "round": round_id,
        "client_id": "mean",
        "accuracy": metrics.get("accuracy", float("nan")),
        "precision_macro": metrics.get("precision_macro", float("nan")),
        "recall_macro": metrics.get("recall_macro", float("nan")),
        "attack_recall": metrics.get("attack_recall", float("nan")),
        "f1_macro": metrics.get("f1_macro", float("nan")),
        "f1_weighted": metrics.get("f1_weighted", float("nan")),
        "auc": metrics.get("auc", float("nan")),
        "fpr": metrics.get("fpr", float("nan")),
        "training_time_sec": training_time_sec,
        "communication_mb": communication_mb,
        "notes": notes,
    }
    return row


def local_partitions(bundle: DatasetBundle, cfg: dict) -> list[np.ndarray]:
    split_type = cfg["clients"].get("split_type", "dirichlet")
    if split_type == "iid":
        return iid_split(bundle.y_train, cfg["clients"]["num_clients"], cfg["project"]["seed"])
    return dirichlet_split(
        bundle.y_train,
        cfg["clients"]["num_clients"],
        float(cfg["clients"]["dirichlet_alpha"]),
        cfg["project"]["seed"],
    )


def init_model(bundle: DatasetBundle, cfg: dict) -> FlowMLP:
    return FlowMLP(
        input_dim=bundle.x_train.shape[1],
        num_classes=2,
        hidden_dims=tuple(cfg["model"]["hidden_dims"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )


def run_centralized(bundle: DatasetBundle, cfg: dict, out_dir: Path) -> dict[str, float]:
    seed = cfg["project"]["seed"]
    set_seed(seed + stable_offset("Centralized MLP"))
    model = init_model(bundle, cfg)
    start = time.time()
    train_local(
        model,
        bundle.x_train,
        bundle.y_train,
        cfg["training"]["centralized_epochs"],
        cfg["training"]["batch_size"],
        cfg["training"]["lr"],
        cfg["training"]["weight_decay"],
        class_weights(bundle.y_train),
    )
    elapsed = time.time() - start
    metrics = evaluate(model, bundle.x_test, bundle.y_test, cfg["training"]["batch_size"])
    append_row(
        out_dir / "logs/main_results.csv",
        row_from_metrics(
            metrics,
            "Centralized MLP",
            "main",
            seed,
            cfg["clients"]["dirichlet_alpha"],
            cfg["clients"]["num_clients"],
            cfg["training"]["rounds"],
            training_time_sec=elapsed,
            notes="Pooled-data upper-bound pilot.",
        ),
    )
    return metrics


def run_local_only(bundle: DatasetBundle, partitions: list[np.ndarray], cfg: dict, out_dir: Path) -> dict[str, float]:
    seed = cfg["project"]["seed"]
    set_seed(seed + stable_offset("Local-only"))
    rows = []
    start = time.time()
    for idx in partitions:
        model = init_model(bundle, cfg)
        train_local(
            model,
            bundle.x_train[idx],
            bundle.y_train[idx],
            cfg["training"]["rounds"] * cfg["training"]["local_epochs"],
            cfg["training"]["batch_size"],
            cfg["training"]["lr"],
            cfg["training"]["weight_decay"],
            class_weights(bundle.y_train),
        )
        rows.append(evaluate(model, bundle.x_test, bundle.y_test, cfg["training"]["batch_size"]))
    elapsed = time.time() - start
    metrics = summarize_client_metrics(rows)
    append_row(
        out_dir / "logs/main_results.csv",
        row_from_metrics(
            metrics,
            "Local-only",
            "main",
            seed,
            cfg["clients"]["dirichlet_alpha"],
            cfg["clients"]["num_clients"],
            cfg["training"]["rounds"],
            communication_mb=0.0,
            training_time_sec=elapsed,
            notes="Mean of isolated client models.",
        ),
    )
    return metrics


def local_update(
    base_model: nn.Module,
    base_state: State,
    bundle: DatasetBundle,
    idx: np.ndarray,
    cfg: dict,
    malicious: bool = False,
    attack_type: str = "none",
    dp_noise: float = 0.0,
    proximal: bool = False,
    local_epochs_override: int | None = None,
) -> tuple[State, State]:
    model = clone_model(base_model)
    model.load_state_dict(base_state, strict=True)
    y_local = bundle.y_train[idx]
    if malicious and attack_type in {"label_flip", "mixed"}:
        y_local = corrupt_labels(y_local)
    prox_state = base_state if proximal else None
    train_local(
        model,
        bundle.x_train[idx],
        y_local,
        int(local_epochs_override) if local_epochs_override is not None else cfg["training"]["local_epochs"],
        cfg["training"]["batch_size"],
        cfg["training"]["lr"],
        cfg["training"]["weight_decay"],
        class_weights(bundle.y_train),
        dp_noise=dp_noise,
        max_grad_norm=float(cfg["privacy"]["max_grad_norm"]),
        proximal_state=prox_state,
        fedprox_mu=float(cfg["federation"]["fedprox_mu"]) if proximal else 0.0,
    )
    after = state_dict_cpu(model)
    delta = state_delta(base_state, after)
    return after, delta


def run_server_fl(
    bundle: DatasetBundle,
    partitions: list[np.ndarray],
    cfg: dict,
    out_dir: Path,
    method: str,
    robust: str = "mean",
    malicious_ratio: float = 0.0,
    attack_type: str = "none",
    dp_noise: float = 0.0,
    epsilon: str | float = "inf",
) -> dict[str, float]:
    seed = cfg["project"]["seed"]
    set_seed(seed + stable_offset(method, robust, malicious_ratio, attack_type, dp_noise, epsilon))
    model = init_model(bundle, cfg)
    global_state = state_dict_cpu(model)
    attackers = malicious_clients(len(partitions), malicious_ratio, seed + 31)
    total_comm = 0.0
    start = time.time()
    final_metrics: dict[str, float] = {}
    for round_id in range(1, cfg["training"]["rounds"] + 1):
        deltas = []
        weights = []
        for client_id, idx in enumerate(partitions):
            _, delta = local_update(
                model,
                global_state,
                bundle,
                idx,
                cfg,
                malicious=client_id in attackers,
                attack_type=attack_type,
                dp_noise=dp_noise,
                proximal=(method == "FedProx"),
            )
            if client_id in attackers:
                delta = corrupt_update(delta, attack_type, client_id)
            deltas.append(delta)
            weights.append(float(len(idx)))
        weights = [weight / sum(weights) for weight in weights]
        if robust == "median":
            agg = median_delta(deltas)
        elif robust == "trimmed_mean":
            agg = trimmed_mean_delta(deltas)
        else:
            agg = mean_delta(deltas, weights)
        total_comm += communication_megabytes(agg, transmissions=len(partitions) * 2)
        global_state = apply_delta(global_state, agg)
        model.load_state_dict(global_state, strict=True)
        final_metrics = evaluate(model, bundle.x_test, bundle.y_test, cfg["training"]["batch_size"])
        append_row(
            out_dir / "logs/round_metrics.csv",
            row_from_metrics(
                final_metrics,
                method,
                "rounds",
                seed,
                cfg["clients"]["dirichlet_alpha"],
                cfg["clients"]["num_clients"],
                round_id,
                dp_enabled=dp_noise > 0,
                epsilon=epsilon,
                malicious_ratio=malicious_ratio,
                attack_type=attack_type,
                communication_mb=total_comm,
            ),
        )
    elapsed = time.time() - start
    return row_from_metrics(
        final_metrics,
        method,
        "main",
        seed,
        cfg["clients"]["dirichlet_alpha"],
        cfg["clients"]["num_clients"],
        cfg["training"]["rounds"],
        dp_enabled=dp_noise > 0,
        epsilon=epsilon,
        malicious_ratio=malicious_ratio,
        attack_type=attack_type,
        communication_mb=total_comm,
        training_time_sec=elapsed,
    )


def validation_impact(
    base_model: nn.Module,
    base_state: State,
    candidate: State,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> float:
    base_model.load_state_dict(base_state, strict=True)
    before = evaluate(base_model, x_val, y_val, batch_size)["f1_macro"]
    candidate_state = apply_delta(base_state, candidate)
    base_model.load_state_dict(candidate_state, strict=True)
    after = evaluate(base_model, x_val, y_val, batch_size)["f1_macro"]
    diff = after - before
    return float(max(0.0, min(1.0, 0.5 + diff)))


def run_gossip(
    bundle: DatasetBundle,
    partitions: list[np.ndarray],
    cfg: dict,
    out_dir: Path,
    method: str,
    shield: bool = False,
    malicious_ratio: float = 0.0,
    attack_type: str = "none",
    dp_noise: float = 0.0,
    epsilon: str | float = "inf",
    no_trust: bool = False,
    personalized_head: bool = False,
    sybil_similarity_threshold: float = 0.0,
    adaptive_threshold: bool | None = None,
) -> dict:
    seed = cfg["project"]["seed"]
    set_seed(
        seed
        + stable_offset(
            method,
            shield,
            malicious_ratio,
            attack_type,
            dp_noise,
            epsilon,
            no_trust,
            personalized_head,
            sybil_similarity_threshold,
        )
    )
    template = init_model(bundle, cfg)
    init_state = state_dict_cpu(template)
    states = [clone_state(init_state) for _ in partitions]
    histories = {(i, j): 0.5 for i in range(len(partitions)) for j in range(len(partitions)) if i != j}
    attackers = malicious_clients(len(partitions), malicious_ratio, seed + 77)
    if adaptive_threshold is None:
        adaptive_threshold = bool(shield and cfg.get("federation", {}).get("adaptive_threshold", False))
    total_comm = 0.0
    start = time.time()
    final_metrics = {}

    for round_id in range(1, cfg["training"]["rounds"] + 1):
        local_deltas: list[State] = []
        local_after_states: list[State] = []
        transmitted: list[State] = []
        for client_id, idx in enumerate(partitions):
            shield_epochs = cfg.get("federation", {}).get("shield_local_epochs") if shield else None
            after_state, delta = local_update(
                template,
                states[client_id],
                bundle,
                idx,
                cfg,
                malicious=client_id in attackers,
                attack_type=attack_type,
                dp_noise=dp_noise,
                local_epochs_override=int(shield_epochs) if shield_epochs is not None else None,
            )
            if personalized_head:
                delta = encoder_only_delta(delta)
            local_deltas.append(delta)
            local_after_states.append(after_state)
            outgoing = corrupt_update(delta, attack_type, client_id) if client_id in attackers else delta
            transmitted.append(outgoing)

        neighbors = random_gossip_neighbors(
            len(partitions), cfg["federation"]["neighbors_per_round"], round_id, seed
        )
        next_states = []
        transmissions = 0
        for client_id, idx in enumerate(partitions):
            peer_ids = neighbors[client_id]
            transmissions += len(peer_ids)
            if shield and not no_trust:
                peer_norms = [float(torch.norm(flatten_delta(transmitted[j]))) for j in peer_ids]
                self_weight = float(cfg.get("federation", {}).get("self_trust_weight", 0.55))
                min_peer_weight = float(cfg.get("federation", {}).get("min_peer_trust_weight", 0.45))
                trust_threshold = float(cfg.get("federation", {}).get("trust_threshold", 0.30))
                trusted_candidates: list[tuple[State, float, int]] = (
                    [] if personalized_head else [(local_deltas[client_id], self_weight, client_id)]
                )
                for peer_id in peer_ids:
                    score = trust_score(
                        transmitted[peer_id],
                        local_deltas[client_id],
                        validation_impact(
                            template,
                            states[client_id],
                            transmitted[peer_id],
                            bundle.x_val[: min(len(bundle.y_val), 512)],
                            bundle.y_val[: min(len(bundle.y_val), 512)],
                            cfg["training"]["batch_size"],
                        ),
                        histories[(client_id, peer_id)],
                        peer_norms,
                    )
                    histories[(client_id, peer_id)] = 0.7 * histories[(client_id, peer_id)] + 0.3 * score
                    if score >= trust_threshold:
                        trusted_candidates.append((transmitted[peer_id], max(min_peer_weight, score), peer_id))
                trusted_candidates = sybil_diverse_updates(trusted_candidates, sybil_similarity_threshold)
                if trusted_candidates:
                    trusted = [candidate for candidate, _, _ in trusted_candidates]
                    trust_values = [
                        gossip_candidate_weight(score, source_id, partitions, cfg)
                        for _, score, source_id in trusted_candidates
                    ]
                    weights = [value / sum(trust_values) for value in trust_values]
                    agg = add_deltas(trusted, weights)
                else:
                    agg = {key: torch.zeros_like(value) for key, value in local_deltas[client_id].items()}
            else:
                deltas = ([] if personalized_head else [local_deltas[client_id]]) + [transmitted[j] for j in peer_ids]
                agg = mean_delta(deltas)
            base_state = local_after_states[client_id] if personalized_head else states[client_id]
            next_states.append(apply_delta(base_state, agg if not personalized_head else encoder_only_delta(agg)))
        total_comm += communication_megabytes(local_deltas[0], transmissions=transmissions)
        states = next_states

        rows = []
        for state in states:
            template.load_state_dict(state, strict=True)
            if adaptive_threshold:
                rows.append(
                    evaluate_with_validation_threshold(
                        template,
                        bundle.x_val,
                        bundle.y_val,
                        bundle.x_test,
                        bundle.y_test,
                        cfg["training"]["batch_size"],
                    )
                )
            else:
                rows.append(evaluate(template, bundle.x_test, bundle.y_test, cfg["training"]["batch_size"]))
        final_metrics = summarize_client_metrics(rows)
        append_row(
            out_dir / "logs/round_metrics.csv",
            row_from_metrics(
                final_metrics,
                method,
                "rounds",
                seed,
                cfg["clients"]["dirichlet_alpha"],
                cfg["clients"]["num_clients"],
                round_id,
                dp_enabled=dp_noise > 0,
                epsilon=epsilon,
                malicious_ratio=malicious_ratio,
                attack_type=attack_type,
                communication_mb=total_comm,
            ),
        )
    elapsed = time.time() - start
    return row_from_metrics(
        final_metrics,
        method,
        "main",
        seed,
        cfg["clients"]["dirichlet_alpha"],
        cfg["clients"]["num_clients"],
        cfg["training"]["rounds"],
        dp_enabled=dp_noise > 0,
        epsilon=epsilon,
        malicious_ratio=malicious_ratio,
        attack_type=attack_type,
        communication_mb=total_comm,
        training_time_sec=elapsed,
        notes="Mean of peer models; no central aggregation server." if shield else "Peer-to-peer gossip baseline.",
    )


def write_dataset_tables(bundle: DatasetBundle, partitions: list[np.ndarray], cfg: dict, out_dir: Path) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    label_rows = bundle.label_summary.copy().astype({"binary": str, "Label": str, "rows": int}).to_dict("records")
    with (table_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["binary", "Label", "rows"])
        writer.writeheader()
        writer.writerows(label_rows)
    split_rows = label_distribution(bundle.y_train, partitions)
    with (table_dir / "client_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_id", "samples", "benign", "ddos"])
        writer.writeheader()
        writer.writerows(split_rows)


def run_privacy_grid(bundle: DatasetBundle, partitions: list[np.ndarray], cfg: dict, out_dir: Path) -> None:
    base = run_gossip(bundle, partitions, cfg, out_dir, "ShieldDFL", shield=True)
    append_row(out_dir / "logs/privacy_results.csv", {**base, "epsilon": "inf", "dp_enabled": False})
    for epsilon in cfg["privacy"]["epsilon_grid"]:
        local_rows = int(np.mean([len(partition) for partition in partitions]))
        noise, accounted_epsilon = noise_for_target_epsilon(cfg, local_rows, float(epsilon))
        row = run_gossip(
            bundle,
            partitions,
            cfg,
            out_dir,
            "ShieldDFL+DP",
            shield=True,
            dp_noise=noise,
            epsilon=round(accounted_epsilon, 3),
        )
        row["notes"] = f"target_epsilon={epsilon}; noise_multiplier={noise:.4f}; opacus_rdp_accounted"
        append_row(out_dir / "logs/privacy_results.csv", row)


def run_poisoning_grid(bundle: DatasetBundle, partitions: list[np.ndarray], cfg: dict, out_dir: Path) -> None:
    attack_type = cfg["attacks"]["attack_type"]
    for ratio in cfg["attacks"]["malicious_ratios"]:
        fedavg = run_server_fl(
            bundle,
            partitions,
            cfg,
            out_dir,
            "FedAvg",
            malicious_ratio=float(ratio),
            attack_type=attack_type,
        )
        median = run_server_fl(
            bundle,
            partitions,
            cfg,
            out_dir,
            "Median",
            robust="median",
            malicious_ratio=float(ratio),
            attack_type=attack_type,
        )
        trimmed = run_server_fl(
            bundle,
            partitions,
            cfg,
            out_dir,
            "Trimmed Mean",
            robust="trimmed_mean",
            malicious_ratio=float(ratio),
            attack_type=attack_type,
        )
        shield = run_gossip(
            bundle,
            partitions,
            cfg,
            out_dir,
            "ShieldDFL",
            shield=True,
            malicious_ratio=float(ratio),
            attack_type=attack_type,
        )
        for row in [fedavg, median, trimmed, shield]:
            append_row(out_dir / "logs/poisoning_results.csv", row)


def run_ablation(bundle: DatasetBundle, partitions: list[np.ndarray], cfg: dict, out_dir: Path) -> None:
    rows = [
        run_gossip(
            bundle,
            partitions,
            cfg,
            out_dir,
            "Full ShieldDFL",
            shield=True,
            malicious_ratio=0.10,
            attack_type="mixed",
            dp_noise=float(cfg["privacy"]["noise_by_epsilon"]["4"]),
            epsilon=4,
        ),
        run_gossip(bundle, partitions, cfg, out_dir, "No trust filter", shield=True, no_trust=True, malicious_ratio=0.10, attack_type="mixed"),
        run_gossip(bundle, partitions, cfg, out_dir, "No DP", shield=True, malicious_ratio=0.10, attack_type="mixed"),
        run_server_fl(bundle, partitions, cfg, out_dir, "Central-server variant", malicious_ratio=0.10, attack_type="mixed"),
    ]
    for row in rows:
        append_row(out_dir / "logs/ablation_results.csv", row)


def write_run_manifest(cfg: dict, out_dir: Path, bundle: DatasetBundle) -> None:
    manifest = {
        "project": cfg["project"]["name"],
        "seed": cfg["project"]["seed"],
        "sample_size": int(len(bundle.y_train) + len(bundle.y_val) + len(bundle.y_test)),
        "train_rows": int(len(bundle.y_train)),
        "validation_rows": int(len(bundle.y_val)),
        "test_rows": int(len(bundle.y_test)),
        "features": int(bundle.x_train.shape[1]),
        "classes": ["Benign", "DDoS"],
        "notes": "Pilot sampled experiment generated from local CICDDoS2019 archive.",
    }
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs/run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_pilot.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["project"]["seed"]))
    out_dir = Path(cfg["logging"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    clear_logs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_cicddos2019_binary(
        (ROOT / cfg["data"]["zip_path"]).resolve()
        if not Path(cfg["data"]["zip_path"]).is_absolute()
        else Path(cfg["data"]["zip_path"]),
        int(cfg["data"]["sample_size"]),
        float(cfg["data"]["test_size"]),
        float(cfg["data"]["validation_size"]),
        int(cfg["project"]["seed"]),
    )
    partitions = local_partitions(bundle, cfg)
    write_dataset_tables(bundle, partitions, cfg, out_dir)
    write_run_manifest(cfg, out_dir, bundle)

    run_centralized(bundle, cfg, out_dir)
    run_local_only(bundle, partitions, cfg, out_dir)
    for method in ["FedAvg", "FedProx"]:
        append_row(out_dir / "logs/main_results.csv", run_server_fl(bundle, partitions, cfg, out_dir, method))
    append_row(out_dir / "logs/main_results.csv", run_gossip(bundle, partitions, cfg, out_dir, "P2P Gossip FL"))
    append_row(out_dir / "logs/main_results.csv", run_gossip(bundle, partitions, cfg, out_dir, "ShieldDFL", shield=True))

    run_privacy_grid(bundle, partitions, cfg, out_dir)
    run_poisoning_grid(bundle, partitions, cfg, out_dir)
    run_ablation(bundle, partitions, cfg, out_dir)


if __name__ == "__main__":
    main()
