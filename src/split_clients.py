from __future__ import annotations

import numpy as np


def dirichlet_split(y: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for client_id, part in enumerate(np.split(idx, cuts)):
            client_indices[client_id].extend(part.tolist())

    output = []
    for indices in client_indices:
        arr = np.array(indices, dtype=np.int64)
        rng.shuffle(arr)
        output.append(arr)
    return output


def iid_split(y: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]


def label_distribution(y: np.ndarray, partitions: list[np.ndarray]) -> list[dict[str, int]]:
    rows = []
    for client_id, idx in enumerate(partitions):
        values, counts = np.unique(y[idx], return_counts=True)
        row = {"client_id": client_id, "samples": int(len(idx)), "benign": 0, "ddos": 0}
        for value, count in zip(values, counts):
            row["benign" if int(value) == 0 else "ddos"] = int(count)
        rows.append(row)
    return rows

