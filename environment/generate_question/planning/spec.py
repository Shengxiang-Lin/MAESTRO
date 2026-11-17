"""Validation for the only maintained Stage-1 family: output-first task_plan.

Legacy approval, macro-first and typed-knowledge construction are retired.
The serialized spec version stays v4 for the current accepted Scene schema.
"""
from __future__ import annotations
import random

VERSION = "maestro.problem_spec.v4"
DIFFICULTIES = ("easy", "medium", "hard")


def spec_issues(spec: dict) -> list[str]:
    errors = []
    if spec.get("version") != VERSION:
        errors.append("invalid_problem_version")
    if spec.get("reasoning_pattern") != "task_plan":
        errors.append("unsupported_stage1_reasoning_pattern")
    if spec.get("family") != "task_plan":
        errors.append("unsupported_stage1_family")
    if spec.get("difficulty_target") not in {"medium", "hard"}:
        errors.append("unsupported_stage1_difficulty")
    if errors:
        return sorted(set(errors))
    from .task_plan import validate_spec
    return sorted(set(validate_spec(spec)))


def difficulty_sequence(*, target: int, mix: tuple[int, int, int],
                        run_seed: int) -> list[str]:
    """Largest-remainder allocation with no silent target-count drift."""
    if target <= 0 or len(mix) != 3 or any(x < 0 for x in mix) or sum(mix) != 100:
        raise ValueError("mix must be three nonnegative percentages summing to 100")
    floats = [target * x / 100 for x in mix]
    counts = [int(n) for n in floats]
    missing = target - sum(counts)
    order = sorted(range(3), key=lambda i: (-(floats[i] - counts[i]), i))
    for index in order[:missing]:
        counts[index] += 1
    sequence = [d for d, count in zip(DIFFICULTIES, counts)
                for _ in range(count)]
    random.Random(run_seed + 881).shuffle(sequence)
    return sequence
