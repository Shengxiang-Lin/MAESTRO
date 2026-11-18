"""Per-decision deterministic global reward for MAESTRO Stage-2 GRPO.

Scoring uses PRIVATE current-round state, not Teacher text similarity, not
a future trajectory rollout and not the predicted Risk Head scores.

Hard invalid actions: -1. For legal actions:
  0.35 evidence-goal coverage versus best feasible current round
  0.25 evidence urgency/downstream leverage versus best feasible round
  0.25 safe parallel occupancy versus maximal feasible round width
  0.15 scope focus (avoid unnecessary requested fields)

The evaluator may read the full *current* task graph, task execution
versions and available source permissions. It does not require the Main
to identify hidden fault causes or inaccessible future answers.
"""
from __future__ import annotations

from itertools import combinations
import json
import math
import re

from .action_binding import bind_tasks
from .decision_reward_state import SCHEMA


def _parse_action(candidate):
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    text = candidate.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].startswith(fence):
            text = "\n".join(lines[1:-1])
    try:
        action = json.loads(text)
    except (ValueError, TypeError):
        return None
    return action if isinstance(action, dict) else None


def _safe_subset(selected):
    ids = {x["task_id"] for x in selected}
    return all(not ids.intersection(x["descendants"])
               for x in selected)


def _legal_optima(options, max_parallel):
    """Best feasible coverage/priority and width; all three are bounded.

    Exact enumeration on the small MAESTRO Macro-DAG. An intentionally
    explicit upper bound prevents silent exponential growth when a later
    generator increases task counts.
    """
    if len(options) > 32 or max_parallel > 8:
        raise ValueError("unsupported decision width; do not approximate oracle")
    max_size = min(max_parallel, len(options))
    coverage = priority = 0.0
    parallel = 0
    for n in range(1, max_size + 1):
        for subset in combinations(options, n):
            if not _safe_subset(subset):
                continue
            coverage = max(coverage, sum(len(x["evidence_units"]) for x in subset))
            priority = max(priority, sum(x["priority"] for x in subset))
            parallel = max(parallel, n)
    return coverage, priority, parallel


def _same_completion_answer(answer: str, expected: str) -> bool:
    """Compare JSON answers structurally, scalar answers exactly."""
    try:
        return json.loads(answer) == json.loads(expected)
    except (ValueError, TypeError):
        return answer.strip() == expected.strip()


def score_decision(candidate, state) -> dict:
    """Return quantitative component scores and diagnostic failure reason.

    Invalid or unsupported states never silently become imitation rewards.
    """
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("missing/incompatible private decision reward state")
    if not state.get("scorable", False):
        raise ValueError("unobservable/private-only repair; exclude state from GRPO")
    options = state.get("actionable")
    if not isinstance(options, list):
        raise ValueError("missing actionable evidence goals")
    max_parallel = state["max_parallel_tasks"]
    if not isinstance(max_parallel, int) or max_parallel < 1:
        raise ValueError("invalid parallel cap")

    def invalid(reason):
        return {
            "reward": -1.0, "valid": False, "reason": reason,
            "coverage": 0.0, "priority": 0.0,
            "parallel": 0.0, "nonredundancy": 0.0,
            "bound_task_ids": [],
        }
    action = _parse_action(candidate)
    if action is None:
        return invalid("invalid_json")
    if action.get("action") == "complete":
        params = action.get("params")
        answer = params.get("answer") if isinstance(params, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            return invalid("missing_completion_answer")
        if not state.get("can_complete", False):
            return invalid("premature_completion")
        expected = state.get("expected_completion_answer")
        if not isinstance(expected, str) or not expected.strip():
            return invalid("missing_private_completion_reference")
        if not _same_completion_answer(answer, expected):
            return invalid("incorrect_completion_answer")
        return {
            "reward": 1.0, "valid": True, "reason": None,
            "coverage": 1.0, "priority": 1.0,
            "parallel": 1.0, "nonredundancy": 1.0,
            "bound_task_ids": [],
        }
    if action.get("action") != "delegate_task":
        return invalid("invalid_action_type")
    params = action.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("tasks"), list):
        return invalid("invalid_delegate_params")
    tasks = params["tasks"]
    if not 1 <= len(tasks) <= max_parallel:
        return invalid("parallel_limit_or_empty_batch")
    if state.get("can_complete", False):
        return invalid("unnecessary_dispatch_after_completion")
    binding = bind_tasks(tasks, state)
    if not binding["valid"]:
        return invalid(binding["reason"])
    selected = binding["selected"]
    ideal_coverage, ideal_priority, max_width = _legal_optima(
        options, max_parallel
    )
    if not ideal_coverage or not ideal_priority or not max_width:
        return invalid("no_currently_admissible_tasks")
    coverage = min(
        1.0,
        sum(len(x["evidence_units"]) for x in selected) / ideal_coverage,
    )
    # A task may target the correct evidence fields yet merely re-retrieve
    # values when the current goal actually calls for reconciliation or
    # refresh. Give only partial priority credit absent an explicit intent.
    priority = min(
        1.0,
        sum(x["priority"] * intent
            for x, intent in zip(selected, binding["operation_alignment"]))
        / ideal_priority,
    )
    parallel = min(1.0, len(selected) / max_width)
    nonredundancy = binding["scope_precision"]
    reward = (
        0.35 * coverage + 0.25 * priority
        + 0.25 * parallel + 0.15 * nonredundancy
    )
    return {
        "reward": round(reward, 8), "valid": True, "reason": None,
        "coverage": round(coverage, 8), "priority": round(priority, 8),
        "parallel": round(parallel, 8),
        "nonredundancy": round(nonredundancy, 8),
        "bound_task_ids": binding["task_ids"],
        "max_legal_parallelism": max_width,
        "ideal_coverage": ideal_coverage,
        "ideal_priority": round(ideal_priority, 6),
        "operation_alignment": round(
            sum(binding["operation_alignment"]) / len(selected), 6
        ),
    }


def orchestration_reward(candidate_text, record):
    """Drop-in reward_callable(candidate_text, record) for train_grpo.py."""
    if not isinstance(record, dict) or "decision_state" not in record:
        raise ValueError("deterministic reward requires private decision_state")
    return score_decision(candidate_text, record["decision_state"])["reward"]
