from __future__ import annotations

from shielddfl.aggregation import State, scale_delta


def malicious_clients(num_clients: int, ratio: float, seed: int) -> set[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    count = int(round(num_clients * ratio))
    if ratio > 0 and count == 0:
        count = 1
    # Use a fixed permutation prefix so larger ratios contain smaller-ratio attackers.
    # This makes poisoning curves easier to interpret across increasing ratios.
    order = rng.permutation(num_clients)
    return set(order[: min(count, num_clients)].tolist())


def corrupt_labels(y):
    return 1 - y


def corrupt_update(delta: State, attack_type: str, client_id: int) -> State:
    if attack_type == "label_flip":
        return delta
    if attack_type == "sign_flip":
        return scale_delta(delta, -1.0)
    if attack_type == "model_replacement":
        return scale_delta(delta, 6.0)
    if attack_type == "mixed":
        return corrupt_update(delta, ["label_flip", "sign_flip", "model_replacement"][client_id % 3], client_id)
    return delta
