"""Seeded, coverage-oriented selection of trusted feedback forks.

Sampling changes only which *already valid* Stage-3 feedback windows are
forked.  It never changes the trusted prefix, source evidence, or DAG.
"""
from __future__ import annotations

import random


def select_feedback_rounds(
    rounds: list[tuple[int, dict]], *,
    limit: int | None,
    strategy: str,
    seed: int,
) -> list[tuple[int, dict]]:
    """Pick chronological or stratified rounds with a stable per-scene seed.

    Stratification splits the *ordered* round positions into `limit`
    non-empty contiguous buckets, then samples one position from each.
    It covers early/middle/late history without forcing a future outcome.
    Returned rounds are always in execution order.
    """
    if strategy not in {"chronological", "stratified"}:
        raise ValueError("invalid_feedback_round_sampling_strategy")
    if limit is not None and (
        type(limit) is not int or limit < 1
    ):
        raise ValueError("invalid_feedback_round_sampling_limit")
    if limit is None or limit >= len(rounds):
        return list(rounds)
    if strategy == "chronological":
        return list(rounds[:limit])
    rng = random.Random(seed)
    length = len(rounds)
    chosen = [
        rng.choice(rounds[(i * length) // limit:
                          ((i + 1) * length) // limit])
        for i in range(limit)
    ]
    return sorted(chosen, key=lambda item: item[0])
