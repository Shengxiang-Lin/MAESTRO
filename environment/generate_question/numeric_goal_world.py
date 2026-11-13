"""Versioned, seed-reproducible non-degenerate numeric answer witnesses.

Previous Macro-first v2 scenes keep *exactly* their original numeric world.
New scenes opt in via `numeric_goal_world_version` and select a separate
PRNG stream: entity names, source observations, sampling order, predicates,
Macro DAG and operator plan remain unchanged.

Comparison questions explicitly ask for the *higher* source; avoid ties.
For each decoy entity the derived answer is the opposite of the target,
so both independent candidate branches remain information-necessary.
"""
from __future__ import annotations

import random

from environment.schema import digest

from .goal_resolution import goal_values


NUMERIC_GOAL_WORLD_VERSION = "maestro.numeric_goal_world.v1"


def generate_numeric_goal_rows(spec: dict, *, entity_count: int) -> list[dict]:
    if spec.get("numeric_goal_world_version") != NUMERIC_GOAL_WORLD_VERSION:
        raise ValueError("numeric_goal_world_version_not_enabled")
    if entity_count < 1:
        raise ValueError("numeric_goal_world_no_entities")

    plan = spec["macro_plan"]
    kind = plan.get("goal_resolution", {}).get("kind")
    if kind not in {"comparison", "calculation"}:
        raise ValueError("numeric_goal_world_requires_numeric_goal")

    rng = random.Random(int(digest(
        spec["seed"], spec["task_id"], NUMERIC_GOAL_WORLD_VERSION
    )[:15], 16))
    rows = []
    if kind == "comparison":
        # The ground-truth winner must not be predictable from the task
        # category, the natural-language question or whether it is Hard.
        winner = rng.choice(("first", "second"))
        for index in range(entity_count):
            low = rng.randint(15, 125)
            high = low + rng.randint(8, 65)
            actual = winner if index == 0 else (
                "second" if winner == "first" else "first"
            )
            left, right = (high, low) if actual == "first" else (low, high)
            assert goal_values("comparison", left, right) == actual
            rows.append({"metric_a": left, "metric_b": right})
        return rows

    # Completion ratio is computed by the *existing* floor operator.
    # Strictly interior counts ensure both source counterfactuals change
    # the answer: setting completed=0 or total=completed must matter.
    target_ratio = None
    for index in range(entity_count):
        for attempt in range(100):
            total = rng.randint(27, 121)
            minimum = max(1, (total * 15 + 99) // 100)
            maximum = min(total - 1, (total * 92) // 100)
            completed = rng.randint(minimum, maximum)
            ratio = goal_values("calculation", completed, total)
            if index == 0 or ratio != target_ratio:
                break
        else:
            raise ValueError("numeric_goal_world_no_distinct_decoy_ratio")
        if index == 0:
            target_ratio = ratio
        rows.append({
            "completed_count": completed,
            "total_count": total,
        })
    return rows
