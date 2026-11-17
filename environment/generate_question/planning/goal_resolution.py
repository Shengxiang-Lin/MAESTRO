"""Frozen, typed Goal Resolution templates for synthetic orchestration.

The final answer is derived from independent keyed sources; none of the
world's numeric outcomes are read from MMQA. V2 snapshots lacking this
marker keep their original archive lookup terminal.
"""
from __future__ import annotations

GOAL_RESOLUTION_VERSION = "maestro.goal_resolution.v1"
KINDS = ("direct_lookup", "comparison", "calculation")
LABELS = {
    "comparison": {
        "answer_field": "higher_count_source",
        "answer_type": "str",
        "public_label": "higher count source",
        "meaning": "source with the larger count among two independent records",
    },
    "calculation": {
        "answer_field": "completion_rate_percent",
        "answer_type": "int",
        "public_label": "completion rate",
        "meaning": "completed / total * 100 rounded down to a whole percent",
    },
}


def goal_values(kind: str, a: int, b: int):
    if type(a) is not int or type(b) is not int:
        raise ValueError("goal_metrics_must_be_integers")
    if kind == "comparison":
        return "first" if a > b else "second" if b > a else "equal"
    if kind == "calculation":
        if b <= 0 or a < 0 or a > b:
            raise ValueError("invalid_completed_total_counts")
        return (100 * a) // b
    raise ValueError("unknown_goal_resolution_kind:" + kind)


def materialized_answer(plan: dict, world_sources: dict, key_field: str,
                        key_value: str):
    kind = plan.get("goal_resolution", {}).get("kind", "direct_lookup")
    if kind == "direct_lookup":
        source = world_sources["archive"]
        matches = [
            row["values"][plan["goal"]["answer_field"]]
            for row in source["observations"]
            if row["values"][key_field] == key_value
        ]
        if len(matches) != 1:
            raise ValueError("goal_archive_lookup_not_unique")
        return matches[0]
    left_name = "metric_a" if kind == "comparison" else "completed_count"
    right_name = "metric_b" if kind == "comparison" else "total_count"
    def read(sid: str, field: str):
        matches = [
            row["values"][field]
            for row in world_sources[sid]["observations"]
            if row["values"][key_field] == key_value
        ]
        if len(matches) != 1:
            raise ValueError("goal_statistic_lookup_not_unique:" + sid)
        return matches[0]
    return goal_values(kind, read("stat_a", left_name),
                       read("stat_b", right_name))
