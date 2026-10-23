from __future__ import annotations

import numpy as np


def random_gossip_neighbors(num_clients: int, degree: int, round_id: int, seed: int) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed + round_id * 997)
    neighbors: dict[int, list[int]] = {}
    for client_id in range(num_clients):
        candidates = [idx for idx in range(num_clients) if idx != client_id]
        rng.shuffle(candidates)
        neighbors[client_id] = candidates[: min(degree, len(candidates))]
    return neighbors


def ring_neighbors(num_clients: int, degree: int = 2) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = {}
    radius = max(1, degree // 2)
    for client_id in range(num_clients):
        current = []
        for shift in range(1, radius + 1):
            current.append((client_id - shift) % num_clients)
            current.append((client_id + shift) % num_clients)
        neighbors[client_id] = current[:degree]
    return neighbors

