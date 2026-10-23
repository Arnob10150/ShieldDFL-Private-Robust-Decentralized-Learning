from __future__ import annotations

import math

import torch

from shielddfl.aggregation import State, flatten_delta


def cosine_similarity(delta_a: State, delta_b: State) -> float:
    a = flatten_delta(delta_a)
    b = flatten_delta(delta_b)
    denom = torch.norm(a) * torch.norm(b)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def norm_ratio_penalty(candidate: State, peer_norms: list[float]) -> float:
    norm = float(torch.norm(flatten_delta(candidate)))
    if not peer_norms:
        return 0.0
    median = max(float(torch.median(torch.tensor(peer_norms))), 1e-8)
    ratio = norm / median
    if ratio <= 2.0:
        return 0.0
    return min(0.6, 0.12 * math.log1p(ratio))


def trust_score(
    candidate: State,
    reference: State,
    validation_score: float,
    history_score: float,
    peer_norms: list[float],
) -> float:
    cosine = (cosine_similarity(candidate, reference) + 1.0) / 2.0
    penalty = norm_ratio_penalty(candidate, peer_norms)
    score = 0.50 * cosine + 0.30 * validation_score + 0.20 * history_score - penalty
    return max(0.0, min(1.0, score))

