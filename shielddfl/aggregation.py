from __future__ import annotations

import torch


State = dict[str, torch.Tensor]


def clone_state(state: State) -> State:
    return {k: v.detach().clone() for k, v in state.items()}


def state_delta(before: State, after: State) -> State:
    return {
        k: (after[k] - before[k]) if before[k].is_floating_point() else torch.zeros_like(before[k])
        for k in before
    }


def apply_delta(state: State, delta: State, scale: float = 1.0) -> State:
    return {
        k: (state[k] + scale * delta[k]) if state[k].is_floating_point() else state[k].clone()
        for k in state
    }


def scale_delta(delta: State, scale: float) -> State:
    return {k: (v * scale if v.is_floating_point() else torch.zeros_like(v)) for k, v in delta.items()}


def add_deltas(deltas: list[State], weights: list[float] | None = None) -> State:
    if not deltas:
        raise ValueError("Cannot aggregate an empty delta list.")
    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)
    total = {k: torch.zeros_like(v) for k, v in deltas[0].items()}
    for delta, weight in zip(deltas, weights):
        for key in total:
            if total[key].is_floating_point():
                total[key] += delta[key] * float(weight)
    return total


def flatten_delta(delta: State) -> torch.Tensor:
    return torch.cat([v.float().flatten() for v in delta.values()])


def unflatten_like(vector: torch.Tensor, template: State) -> State:
    output = {}
    offset = 0
    for key, value in template.items():
        count = value.numel()
        output[key] = vector[offset : offset + count].view_as(value).type_as(value)
        offset += count
    return output


def mean_delta(deltas: list[State], weights: list[float] | None = None) -> State:
    return add_deltas(deltas, weights)


def median_delta(deltas: list[State]) -> State:
    stacked = torch.stack([flatten_delta(delta) for delta in deltas], dim=0)
    median = torch.median(stacked, dim=0).values
    return unflatten_like(median, deltas[0])


def trimmed_mean_delta(deltas: list[State], trim_ratio: float = 0.2) -> State:
    stacked = torch.stack([flatten_delta(delta) for delta in deltas], dim=0)
    trim = int(stacked.shape[0] * trim_ratio)
    sorted_vals, _ = torch.sort(stacked, dim=0)
    if trim > 0 and stacked.shape[0] > 2 * trim:
        sorted_vals = sorted_vals[trim:-trim]
    return unflatten_like(sorted_vals.mean(dim=0), deltas[0])


def communication_megabytes(delta: State, transmissions: int) -> float:
    bytes_per_update = sum(v.numel() * v.element_size() for v in delta.values())
    return float(bytes_per_update * transmissions / (1024 * 1024))
