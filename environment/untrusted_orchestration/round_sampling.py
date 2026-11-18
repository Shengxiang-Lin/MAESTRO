"""Select existing Stage-3 feedback windows for a Stage-4 fault fork.

Offline selection can use the frozen Stage-1 graph. It must never modify a
trusted prefix, create a new verifier, or expose future evidence to Main.
"""
from __future__ import annotations

import random


def select_feedback_rounds(
    rounds: list[tuple[int, dict]], *,
    limit: int | None, strategy: str, seed: int,
    scene: dict | None = None,
) -> list[tuple[int, dict]]:
    """Cover chronological positions, optionally favoring evidence-rich ones.

    `evidence_aware` preserves the same early/middle/late buckets as
    `stratified`, but selects the strongest frozen source/verification
    affordance in each bucket. This is sampling metadata, not a Main action
    or evidence presented to Main. Unknown structural claims score zero.
    """
    if strategy not in {"chronological", "stratified", "evidence_aware"}:
        raise ValueError("invalid_feedback_round_sampling_strategy")
    if limit is not None and (
        type(limit) is not int or limit < 1
    ):
        raise ValueError("invalid_feedback_round_sampling_limit")
    if strategy == "evidence_aware" and scene is None:
        raise ValueError("evidence_aware_sampling_requires_scene")
    if limit is None or limit >= len(rounds):
        return list(rounds)
    if strategy == "chronological":
        return list(rounds[:limit])

    rng = random.Random(seed)
    graph = (scene or {}).get("macro_graph", {})
    tasks = {task["task_id"]: task for task in graph.get("tasks", [])}
    waves = graph.get("reference_waves", [])

    def score(decision_round: int) -> tuple[int, int, int]:
        wave_index = decision_round - 2
        if not 0 <= wave_index < len(waves):
            return (0, 0, 0)
        incoming = waves[wave_index]
        joined = sum(
            bool(task.get("source_ids"))
            and any(step.get("op") == "join"
                    for step in task.get("operator_chain", []))
            for tid in incoming if (task := tasks.get(tid))
        )
        multi = sum(
            len(tasks[tid].get("source_ids", [])) > 1
            for tid in incoming if tid in tasks
        )
        verify = sum(
            tasks[tid].get("macro_role") == "verification"
            for tid in incoming if tid in tasks
        )
        consumers = sum(
            bool(set(task.get("depends_on", [])) & set(incoming))
            for task in tasks.values()
        )
        # Verification and multi-source join gates are not proof of
        # corroboration. They are simply more informative fork sites.
        return (verify * 5 + joined * 3 + multi * 2,
                consumers, len(incoming))

    selected = []
    for bucket in range(limit):
        choices = rounds[(bucket * len(rounds)) // limit:
                         ((bucket + 1) * len(rounds)) // limit]
        if strategy == "stratified":
            selected.append(rng.choice(choices))
        else:
            # Stable tie-breaking varies selected sites across questions
            # with the same graph motif, but never uses future Gold rows.
            order = list(choices)
            rng.shuffle(order)
            selected.append(max(order, key=lambda row: score(row[0])))
    return sorted(selected, key=lambda item: item[0])
