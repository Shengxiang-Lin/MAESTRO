"""Goal-conditioned typed resolution on top of the proven Task Plan World.

V4 adds two *actual* answer operations, not alternative question phrasings:
- comparison: resolve one entity, acquire two independent numeric sources in
  parallel, then compare their reported observation counts;
- aggregation: resolve one entity, acquire its one-to-many observation log,
  apply an inclusive year criterion, then count qualifying observations.

All measurements are canonical private World facts, never copied from the
Question, public reference card, or original YAGO people/places.
Existing v2/v3 plans are not reinterpreted.
"""
from __future__ import annotations

import copy
import random

from environment.schema import digest

VERSION = "goal_conditioned_typed_goal_v4"
MEASUREMENTS_VERSION = "maestro.typed_goal_measurements.v1"
KINDS = ("entity_resolution", "comparison", "aggregation")


def selected_kind(seed: int, ordinal: int) -> str:
    return KINDS[int(digest(
        seed, ordinal, VERSION, "goal-type"
    )[:14], 16) % len(KINDS)]


def compose_typed_goal_plan(*, seed: int, ordinal: int, difficulty: str,
                            hops: int, goal_keys: tuple[str, ...]) -> dict:
    from .goal_composer import compose_plan, NEEDS_VERSION
    kind = selected_kind(seed, ordinal)
    # Start with a fully executable, provenance-sensitive entity resolver.
    # Numeric output needs a source-resolvable entity label and at most six
    # preceding Macros: two independent measurements plus a final merger
    # must remain within the established nine-Macro budget.
    for attempt in range(60):
        probe = ordinal + attempt * 1009
        plan = compose_plan(
            seed=seed, ordinal=probe, difficulty=difficulty, hops=hops,
            goal_keys=goal_keys, policy=NEEDS_VERSION,
        )
        if kind == "entity_resolution" or (
            plan["mode"] == "trace"
            and len(plan["macro_tasks"]) <= (
                6 if kind == "comparison" else 7
            )
        ):
            break
    else:
        raise ValueError("v4_no_source_feasible_macro_budget")
    plan["composer_policy"] = VERSION
    plan["goal_spec"] = {
        "version": VERSION, "kind": kind,
        "answer_field": "answer",
        "answer_type": "int" if kind == "aggregation" else "str",
        "terminal_relation_keys": list(goal_keys),
        **({"qualifying_observation_year_gte": 2025}
           if kind == "aggregation" else {}),
        **({"comparison_direction": "higher",
            "comparison_sources": ["metric_a", "metric_b"]}
           if kind == "comparison" else {}),
    }
    if kind == "entity_resolution":
        return plan

    tasks = plan["macro_tasks"]
    prior = tasks[-1]
    if prior["operation"] != "follow_yago" or prior["output_schema"] != {
        "answer": "str"
    }:
        raise ValueError("v4_missing_relation_terminal")
    # Only change the final handoff: the relationship path still returns
    # exactly the same entity; that value is now a *key*, not the answer.
    prior["output_schema"] = {"entity_final": "str"}
    prior["role"] = "investigation"
    key_mid = prior["macro_id"]

    def add(operation: str, dependencies: list[str], output: dict,
            sources: list[str], need: str, role: str):
        earlier = {row["macro_id"]: row for row in tasks}
        mid = f"M{len(tasks) + 1}"
        tasks.append({
            "macro_id": mid, "role": role,
            "operation": operation, "depends_on": list(dependencies),
            "input_contract": {
                dep: copy.deepcopy(earlier[dep]["output_schema"])
                for dep in dependencies
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

    if kind == "comparison":
        left = add(
            "lookup_metric_a", [key_mid],
            {"entity_final": "str", "metric_a": "int"}, ["metric_a"],
            "Retrieve the first independently recorded observation count",
            "investigation",
        )
        right = add(
            "lookup_metric_b", [key_mid],
            {"entity_final": "str", "metric_b": "int"}, ["metric_b"],
            "Retrieve the second independently recorded observation count",
            "investigation",
        )
        terminal = add(
            "compare_metrics", [left, right], {"answer": "str"}, [],
            "Compare the two independently retrieved counts",
            "synthesis",
        )
    else:
        rows = add(
            "fetch_observations", [key_mid],
            {"entity_final": "str", "observation_id": "str",
             "observation_year": "int"}, ["observation_log"],
            "Retrieve all log observations for the resolved entity",
            "investigation",
        )
        terminal = add(
            "count_qualifying_observations", [rows], {"answer": "int"}, [],
            "Count unique observation rows recorded in or after 2025",
            "synthesis",
        )
    if len(tasks) > 9:
        raise ValueError("v4_macro_budget_exceeded")
    plan["terminal_macro_id"] = terminal
    plan["goal"] = {
        "answer_field": "answer",
        "answer_type": plan["goal_spec"]["answer_type"],
        "public_label": (
            "higher observation count source"
            if kind == "comparison" else "qualifying observation count"
        ),
        "cardinality": "exactly_one",
        "meaning": (
            "Compare two independently sourced entity observation counts"
            if kind == "comparison" else
            "Count recorded entity observations in or after 2025"
        ),
    }
    return plan


def measurement_world(*, plan: dict, binding: dict,
                      seed: int, ordinal: int) -> dict | None:
    """Deterministic private measurement facts bound to the canonical KG.

    Every shadow case gets an alternative answer, so relaxing any clue
    reveals a *real nonempty competing result*, not a blocked execution.
    """
    kind = plan["goal_spec"]["kind"]
    if kind == "entity_resolution":
        return None
    paths = binding["candidate_paths"]
    target = binding["target_case"]
    rng = random.Random(int(digest(
        seed, ordinal, "v4-private-metric-world"
    )[:15], 16))
    target_winner = rng.choice(("first", "second"))
    ordered = [target] + sorted(c for c in paths if c != target)
    entries = {}
    for index, case in enumerate(ordered):
        label = paths[case][-1]["object_label"]
        if kind == "comparison":
            low = 12 + rng.randrange(9)
            high = low + 13 + rng.randrange(23)
            winner = target_winner if index == 0 else (
                "second" if target_winner == "first" else "first"
            )
            a, b = ((high, low) if winner == "first" else (low, high))
            entries[case] = {
                "entity_final": label, "metric_a": a, "metric_b": b,
            }
        else:
            # All candidate counts differ. An ineligible earlier observation
            # proves the year filter is computational work, not prose.
            n = 2 if index == 0 else 2 + index
            rows = [{
                "entity_final": label,
                "observation_id": "OBS-" + digest(
                    seed, ordinal, case, j, "v4-observation"
                )[:12].upper(),
                "observation_year": (2024 if j == n else
                                     2025 + j % 2),
            } for j in range(n+1)]
            entries[case] = {"entity_final": label, "observations": rows}
    return {
        "version": MEASUREMENTS_VERSION,
        "kind": kind, "by_case": entries,
    }


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
    from .goal_resolution import goal_values
    if kind == "comparison":
        target = binding["target_case"]
        row = actual["by_case"][target]
        result = goal_values("comparison", row["metric_a"], row["metric_b"])
        if result == "equal" or any(
            goal_values("comparison", entry["metric_a"], entry["metric_b"])
            == result for case, entry in actual["by_case"].items()
            if case != target
        ):
            return ["v4_comparison_shadow_not_discriminating"]
    else:
        target = binding["target_case"]
        counts = {
            case: sum(
                item["observation_year"] >= 2025
                for item in entry["observations"]
            ) for case, entry in actual["by_case"].items()
        }
        if (counts[target] < 2 or any(
                value == counts[target] for case, value in counts.items()
                if case != target)):
            return ["v4_aggregation_shadow_not_discriminating"]
    return []
