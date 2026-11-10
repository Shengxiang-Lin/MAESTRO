"""Bounded accepted-scene difficulty quotas.

Candidate difficulty proportions are necessary but not sufficient after
independent Scenario/Question rejection. Selection is deterministic;
unfilled quotas remain explicitly visible and are never fabricated.
"""
from __future__ import annotations

from collections import Counter


LEVELS = ("easy", "medium", "hard")


def candidate_budget(target: int, available: int, *, enabled: bool,
                     multiplier: int) -> int:
    if not 1 <= multiplier <= 4:
        raise ValueError("quota_candidate_multiplier_must_be_1_to_4")
    if target < 1 or available < target:
        raise ValueError("insufficient_candidate_records")
    return min(available, target * multiplier) if enabled else target


def select_accepted_scenes(
    reviewed: list[dict], requested_counts: dict[str, int],
) -> tuple[list[dict], dict[str, int]]:
    remaining = Counter(requested_counts)
    selected = []
    for spec in reviewed:
        level = spec["difficulty_target"]
        if remaining[level] <= 0:
            continue
        selected.append(spec)
        remaining[level] -= 1
    return selected, {level: remaining[level] for level in LEVELS}
