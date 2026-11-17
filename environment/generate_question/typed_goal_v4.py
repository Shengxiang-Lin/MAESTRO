"""Versioned typed Goal execution, abstracted from the actual OmniGAIA
pretest 60 task set (data/OmniGAIA/splits/pretest_60_ids.txt).

The 60 reference questions inform OPERATION COVERAGE ONLY. Their words,
answer values, media and linked external facts never enter generated data.
V3 and frozen legacy scenes retain their exact semantic contracts.

Question resolution and evidence acquisition are intentionally distinct:
a source-conditioned entity resolver is composed first; v4 then attaches
the information-producing leaves and a real executable synthesis operation.
"""
from __future__ import annotations

import copy
import datetime as dt
import random

from environment.schema import digest

VERSION = "goal_conditioned_typed_goal_v4"
MEASUREMENTS_VERSION = "maestro.typed_goal_measurements.v2"

# Mutually exclusive *primary* resolution for a compact synthetic pilot.
# Structured output is separate from the primitive operations used to form it.
# Smoothed sampling avoids starving rare types from the 60-question reference.
KINDS = (
    "entity_resolution", "numeric_difference", "numeric_ratio",
    "temporal_gap", "geographic_distance", "comparison",
    "aggregation", "structured_answer",
)
KIND_WEIGHTS = (9, 12, 12, 12, 10, 8, 6, 16)
METRIC_KINDS = {
    "numeric_difference", "numeric_ratio", "temporal_gap",
    "geographic_distance", "comparison", "structured_answer",
}

# Operations are explicit and versioned; no private media answer is consulted.
OPERATOR_FOR_KIND = {
    "numeric_difference": "derive_numeric_difference",
    "numeric_ratio": "derive_numeric_ratio",
    "temporal_gap": "derive_temporal_gap",
    "geographic_distance": "derive_geographic_distance",
    "comparison": "compare_metrics",
    "structured_answer": "compose_structured_answer",
}


def selected_kind(seed: int, ordinal: int) -> str:
    rng = random.Random(int(digest(
        seed, ordinal, VERSION, "goal-type-distribution-v2"
    )[:15], 16))
    return rng.choices(KINDS, weights=KIND_WEIGHTS, k=1)[0]


def _metric_fields(kind: str, side: str) -> dict:
    if kind == "temporal_gap":
        return {"entity_final": "str", "date_" + side: "str"}
    if kind == "geographic_distance":
        return {
            "entity_final": "str",
            "latitude_" + side: "float", "longitude_" + side: "float",
        }
    return {"entity_final": "str", "metric_" + side: "int"}


def compose_typed_goal_plan(*, seed: int, ordinal: int, difficulty: str,
                            hops: int, goal_keys: tuple[str, ...]) -> dict:
    from .goal_composer import compose_plan, NEEDS_VERSION
    kind = selected_kind(seed, ordinal)
    # A two-branch numeric Goal needs three extra Macros, keeping a MAX-9
    # budget, and must use the single entity-reference route. This is a
    # compatibility limitation, not a claim that all producer types are valid.
    for attempt in range(80):
        probe = ordinal + attempt * 1009
        plan = compose_plan(
            seed=seed, ordinal=probe, difficulty=difficulty, hops=hops,
            goal_keys=goal_keys, policy=NEEDS_VERSION,
        )
        if kind == "entity_resolution" or (
            plan["mode"] == "trace" and len(plan["macro_tasks"]) <= (
                7 if kind == "aggregation" else 6
            )
        ):
            break
    else:
        raise ValueError("v4_no_source_feasible_macro_budget")
    plan["composer_policy"] = VERSION
    result_type = (
        "int" if kind in {"aggregation", "numeric_difference", "temporal_gap",
                          "geographic_distance"} else
        "float" if kind == "numeric_ratio" else
        "dict" if kind == "structured_answer" else "str"
    )
    plan["goal_spec"] = {
        "version": VERSION, "kind": kind,
        "answer_field": "answer", "answer_type": result_type,
        "terminal_relation_keys": list(goal_keys),
        **({"qualifying_observation_year_gte": 2025}
           if kind == "aggregation" else {}),
        **({"independent_sources": ["metric_a", "metric_b"]}
           if kind in METRIC_KINDS else {}),
        **({"round_digits": 2} if kind == "numeric_ratio" else {}),
        **({"round_unit": "nearest_km"}
           if kind == "geographic_distance" else {}),
        **({"output_fields": [
            "entity", "higher_source", "count_difference",
        ]} if kind == "structured_answer" else {}),
    }
    if kind == "entity_resolution":
        return plan

    tasks = plan["macro_tasks"]
    prior = tasks[-1]
    if prior["operation"] != "follow_yago" or prior["output_schema"] != {
        "answer": "str"
    }:
        raise ValueError("v4_missing_relation_terminal")
    # The relation label is now a dependent query key, NOT a user answer.
    prior["output_schema"] = {"entity_final": "str"}
    prior["role"] = "investigation"
    key_mid = prior["macro_id"]

    def add(operation: str, deps: list[str], output: dict,
            sources: list[str], need: str, role: str) -> str:
        earlier = {row["macro_id"]: row for row in tasks}
        mid = "M" + str(len(tasks) + 1)
        tasks.append({
            "macro_id": mid, "role": role,
            "operation": operation, "depends_on": list(deps),
            "input_contract": {
                dep: copy.deepcopy(earlier[dep]["output_schema"])
                for dep in deps
            },
            "output_schema": dict(output), "source_scope": list(sources),
            "solver_template": operation, "relation_need": need,
            "evidence_plan": {
                "description": need,
                "source_observation_targets": {
                    sid: 1 for sid in sources
                },
            },
        })
        return mid

    if kind == "aggregation":
        records = add(
            "fetch_observations", [key_mid],
            {"entity_final": "str", "observation_id": "str",
             "observation_year": "int"}, ["observation_log"],
            "Retrieve the dependent dated observation log", "investigation",
        )
        terminal = add(
            "count_qualifying_observations", [records],
            {"answer": "int"}, [],
            "Count distinct observations in or after the inclusive date",
            "synthesis",
        )
    else:
        left = add(
            "lookup_metric_a", [key_mid], _metric_fields(kind, "a"),
            ["metric_a"], "Acquire the first independent fact", "investigation",
        )
        right = add(
            "lookup_metric_b", [key_mid], _metric_fields(kind, "b"),
            ["metric_b"], "Acquire the second independent fact", "investigation",
        )
        terminal = add(
            OPERATOR_FOR_KIND[kind], [left, right],
            {"answer": result_type}, [],
            "Derive the frozen final answer using both independent facts",
            "synthesis",
        )
    if len(tasks) > 9:
        raise ValueError("v4_macro_budget_exceeded")
    plan["terminal_macro_id"] = terminal
    plan["goal"] = {
        "answer_field": "answer", "answer_type": result_type,
        "public_label": {
            "aggregation": "qualifying observation count",
            "comparison": "higher-count information source",
            "numeric_difference": "absolute numerical difference",
            "numeric_ratio": "two-source numerical ratio",
            "temporal_gap": "elapsed days between events",
            "geographic_distance": "straight-line distance in kilometres",
            "structured_answer": "structured entity and metric assessment",
        }[kind],
        "cardinality": "exactly_one",
        "meaning": "Determine the final " + kind.replace("_", " ") +
                   " from separately acquired records",
    }
    return plan


def measurement_world(*, plan: dict, binding: dict,
                      seed: int, ordinal: int) -> dict | None:
    kind = plan["goal_spec"]["kind"]
    if kind == "entity_resolution":
        return None
    paths = binding["candidate_paths"]
    target = binding["target_case"]
    rng = random.Random(int(digest(
        seed, ordinal, "v4-private-metric-world-v2"
    )[:15], 16))
    cases = [target] + sorted(c for c in paths if c != target)
    entries = {}
    for index, case in enumerate(cases):
        label = paths[case][-1]["object_label"]
        if kind == "aggregation":
            count = 2 + index
            rows = [{
                "entity_final": label,
                "observation_id": "OBS-" + digest(
                    seed, ordinal, case, j, "v4-observation"
                )[:12].upper(),
                "observation_year": (2024 if j == count else
                                     2025 + j % 2),
            } for j in range(count + 1)]
            entries[case] = {
                "entity_final": label, "observations": rows
            }
        elif kind == "temporal_gap":
            start = dt.date(2016, 1, 1) + dt.timedelta(
                days=5 * index + rng.randrange(1400)
            )
            gap = 15 + 11 * index
            entries[case] = {
                "entity_final": label,
                "date_a": start.isoformat(),
                "date_b": (start + dt.timedelta(days=gap)).isoformat(),
            }
        elif kind == "geographic_distance":
            # Real haversine inputs with synthetic, non-identifying coordinates.
            latitude_a = 22.0 + 0.4 * index
            longitude_a = -88.0 + 0.03 * index
            entries[case] = {
                "entity_final": label,
                "latitude_a": latitude_a, "longitude_a": longitude_a,
                "latitude_b": latitude_a + 0.05 + index * 0.10,
                "longitude_b": longitude_a + 0.17 + index * 0.07,
            }
        elif kind == "numeric_ratio":
            entries[case] = {
                "entity_final": label,
                "metric_a": 53 + index * 17,
                "metric_b": 19,
            }
        elif kind == "numeric_difference":
            entries[case] = {
                "entity_final": label,
                "metric_a": 51 + index * 19,
                "metric_b": 11,
            }
        else:
            # Winner alternates so relaxing any identity clue yields a
            # DIFFERENT, nonempty downstream result for comparison.
            target_first = bool(int(digest(
                seed, ordinal, "v4-comparison-winner"
            )[:8], 16) % 2)
            first = target_first if index == 0 else not target_first
            small = 12 + index * 3
            big = small + 15 + index * 5
            entries[case] = {
                "entity_final": label,
                "metric_a": big if first else small,
                "metric_b": small if first else big,
            }
    return {
        "version": MEASUREMENTS_VERSION,
        "kind": kind,
        "by_case": entries,
    }


def _final_answer_for(case_data: dict, kind: str):
    from .goal_resolution import goal_values
    if kind == "aggregation":
        return sum(
            r["observation_year"] >= 2025
            for r in case_data["observations"]
        )
    if kind == "comparison":
        return goal_values(
            "comparison", case_data["metric_a"], case_data["metric_b"],
        )
    if kind == "structured_answer":
        return {
            "entity": case_data["entity_final"],
            "higher_source": goal_values(
                "comparison", case_data["metric_a"], case_data["metric_b"]
            ),
            "count_difference": abs(
                case_data["metric_a"] - case_data["metric_b"]
            ),
        }
    if kind == "numeric_difference":
        return abs(case_data["metric_a"] - case_data["metric_b"])
    if kind == "numeric_ratio":
        return round(case_data["metric_a"] / case_data["metric_b"], 2)
    if kind == "temporal_gap":
        return abs((
            dt.date.fromisoformat(case_data["date_b"])
            - dt.date.fromisoformat(case_data["date_a"])
        ).days)
    if kind == "geographic_distance":
        from .program_ir import great_circle_km
        return round(great_circle_km(
            case_data["latitude_a"], case_data["longitude_a"],
            case_data["latitude_b"], case_data["longitude_b"],
        ))
    raise ValueError("unknown_v4_numeric_goal")


def validate_measurement_world(plan: dict, binding: dict, *,
                               seed: int, ordinal: int) -> list[str]:
    kind = plan.get("goal_spec", {}).get("kind")
    actual = binding.get("world_model", {}).get("goal_measurements")
    if kind == "entity_resolution":
        return [] if actual is None else ["v4_unexpected_metric_world"]
    expected = measurement_world(
        plan=plan, binding=binding, seed=seed, ordinal=ordinal,
    )
    if actual != expected:
        return ["v4_metric_world_not_canonical"]
    answer = _final_answer_for(
        actual["by_case"][binding["target_case"]], kind,
    )
    shadow = [
        _final_answer_for(row, kind)
        for case, row in actual["by_case"].items()
        if case != binding["target_case"]
    ]
    if any(other == answer for other in shadow):
        return ["v4_non_discriminating_shadow_answer"]
    return []


_RELATION_PHRASES = {
    "birthplace": "birthplace", "deathplace": "place of death",
    "containedinplace": "contained administrative area",
    "location": "location", "doctoraladvisor": "doctoral advisor",
    "fieldofwork": "research field", "spouse": "spouse",
    "memberof": "organization membership", "neighbor": "neighboring place",
    "twinnedwith": "twin municipality", "nationality": "nationality",
    "capital": "capital",
}


def typed_goal_question(plan: dict) -> str:
    """One final user request, no private media values or gold labels."""
    kind = plan["goal_spec"]["kind"]
    if kind == "entity_resolution":
        return plan["goal"]["semantic"]["question"]
    chain = ", followed by ".join(
        _RELATION_PHRASES[key] for key in plan["goal_recipe_keys"]
    )
    scope = (
        "the entity obtained through the documented relationship sequence "
        f"({chain})"
    )
    if kind == "aggregation":
        final = (
            "how many distinct dated observation entries from 2025 onward "
            "(2025 inclusive) does its log contain? Return only the count."
        )
    elif kind == "comparison":
        final = (
            "which of its two independently recorded observation tallies "
            "is HIGHER, the first or second? Return only 'first' or 'second'."
        )
    elif kind == "numeric_difference":
        final = (
            "what is the absolute difference between its first and second "
            "recorded counts? Return only one whole number."
        )
    elif kind == "numeric_ratio":
        final = (
            "what is the ratio of its first recorded count to its second "
            "recorded count? Round to two decimal places."
        )
    elif kind == "temporal_gap":
        final = (
            "how many full days elapsed between the two dated milestones "
            "recorded for it? Return the number of days."
        )
    elif kind == "geographic_distance":
        final = (
            "what is the straight-line great-circle distance in kilometres "
            "between the two recorded sites? Round to the nearest kilometre."
        )
    else:
        final = (
            "return a structured answer containing its entity label, "
            "the higher of the two independent tally sources (first or "
            "second), and the absolute difference of those two counts. "
            "Use fields 'entity', 'higher_source', 'count_difference'."
        )
    return f"For {scope}, {final}"
# Retain the previous public helper for callers of the in-progress branch.
numeric_goal_question = typed_goal_question
