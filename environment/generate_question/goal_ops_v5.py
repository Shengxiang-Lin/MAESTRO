"""Versioned *orthogonal* Goal operation graph and output contract.

Based on the abstract operations observed in the selected OmniGAIA pretest
60; no benchmark question, answer or media enters generated sources.
v4 remains an opt-in and unchanged baseline.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import random

from environment.schema import digest
from .typed_goal_v4 import (
    compose_typed_goal_plan, measurement_world as v4_measurement_world,
    _RELATION_PHRASES,
)

VERSION = "goal_conditioned_operation_dag_v5"
# This is a source *capability*, not the name of a final Goal.
PROFILES = (
    "entity_resolution", "numeric_difference", "numeric_ratio",
    "temporal_gap", "geographic_distance", "comparison", "aggregation",
)
OUTPUTS = (
    "single_entity", "single_number", "date_time",
    "structured_multi_field", "evidence_based_explanation",
)
PROFILE_OUTPUTS = {
    "entity_resolution": ("single_entity",),
    "numeric_difference": ("single_number", "structured_multi_field",
                           "evidence_based_explanation"),
    "numeric_ratio": ("single_number", "structured_multi_field"),
    "temporal_gap": ("single_number", "date_time", "structured_multi_field"),
    "geographic_distance": ("single_number", "structured_multi_field"),
    "comparison": ("single_entity", "structured_multi_field",
                   "evidence_based_explanation"),
    "aggregation": ("single_number", "structured_multi_field"),
}


def selected_profile_output(seed: int, ordinal: int) -> tuple[str, str]:
    rng = random.Random(int(digest(
        seed, ordinal, VERSION, "orthogonal-goal-selection"
    )[:15], 16))
    profile = rng.choice(PROFILES)
    return profile, rng.choice(PROFILE_OUTPUTS[profile])


def _node(ident, op, inputs, output, typ, **kwargs):
    return {"id": ident, "op": op, "inputs": list(inputs),
            "output": output, "type": typ, **kwargs}


def _design_goal(profile: str, shape: str) -> dict:
    """A small typed DAG; output format is independent of computation kind."""
    if profile not in PROFILES or shape not in PROFILE_OUTPUTS[profile]:
        raise ValueError("v5_invalid_profile_output_pair")
    operations = [
        _node("entity", "lookup", [], "entity_final", "str",
              source="relation_chain"),
    ]
    if profile == "entity_resolution":
        return {
            "version": VERSION, "source_profile": profile,
            "operations": operations,
            "output_spec": {
                "kind": "single_entity",
                "fields": {"answer": "str"},
                "bindings": {"answer": "entity"},
            },
        }
    if profile == "aggregation":
        operations.extend([
            _node("observations", "lookup", ["entity"],
                  "observation_rows", "rows", source="observation_log"),
            _node("count", "count_aggregate", ["observations"], "count",
                  "int", year_gte=2025, unique_by="observation_id"),
        ])
        selections = (
            {"answer": "count"} if shape == "single_number"
            else {"entity": "entity", "count": "count"}
        )
    else:
        if profile == "temporal_gap":
            a, b, typ = "date_a", "date_b", "str"
        elif profile == "geographic_distance":
            a, b, typ = "coords_a", "coords_b", "coordinates"
        else:
            a, b, typ = "metric_a", "metric_b", "int"
        operations.extend([
            _node("a", "lookup", ["entity"], a, typ, source="metric_a"),
            _node("b", "lookup", ["entity"], b, typ, source="metric_b"),
        ])
        if profile in {"numeric_difference", "numeric_ratio"}:
            if shape == "single_number":
                operation = (
                    "absolute_difference" if profile == "numeric_difference"
                    else "ratio"
                )
                operations.append(_node(
                    "value", "arithmetic", ["a", "b"], "value",
                    "int" if operation == "absolute_difference" else "float",
                    function=operation, round_digits=2 if operation == "ratio" else 0,
                ))
                selections = {"answer": "value"}
            elif shape == "structured_multi_field":
                operations.extend([
                    _node("difference", "arithmetic", ["a", "b"],
                          "difference", "int", function="absolute_difference",
                          round_digits=0),
                    _node("ratio", "arithmetic", ["a", "b"],
                          "ratio", "float", function="ratio", round_digits=2),
                ])
                selections = {"entity": "entity", "difference": "difference",
                              "ratio": "ratio"}
            else:
                operations.append(_node(
                    "assessment", "verify", ["a", "b"], "assessment",
                    "str", relation="higher_count",
                ))
                selections = {
                    "conclusion": "assessment",
                    "first_count": "a", "second_count": "b",
                }
        elif profile == "temporal_gap":
            if shape == "single_number":
                operations.append(_node(
                    "days", "temporal_difference", ["a", "b"],
                    "elapsed_days", "int",
                ))
                selections = {"answer": "days"}
            elif shape == "date_time":
                operations.append(_node(
                    "midpoint", "temporal_midpoint", ["a", "b"],
                    "midpoint_date", "str",
                ))
                selections = {"answer": "midpoint"}
            else:
                operations.extend([
                    _node("days", "temporal_difference", ["a", "b"],
                          "elapsed_days", "int"),
                    _node("midpoint", "temporal_midpoint", ["a", "b"],
                          "midpoint_date", "str"),
                ])
                selections = {"entity": "entity", "elapsed_days": "days",
                              "midpoint_date": "midpoint"}
        elif profile == "geographic_distance":
            operations.append(_node(
                "distance", "geographic_distance", ["a", "b"],
                "distance_km", "int", distance_unit="km",
            ))
            selections = (
                {"answer": "distance"} if shape == "single_number"
                else {"entity": "entity", "distance_km": "distance"}
            )
        elif profile == "comparison":
            operations.append(_node(
                "winner", "compare_select", ["a", "b"],
                "higher_source", "str", direction="higher",
            ))
            if shape == "single_entity":
                selections = {"answer": "winner"}
            elif shape == "structured_multi_field":
                operations.append(_node(
                    "difference", "arithmetic", ["a", "b"],
                    "difference", "int", function="absolute_difference",
                    round_digits=0,
                ))
                selections = {"entity": "entity", "higher_source": "winner",
                              "count_difference": "difference"}
            else:
                operations.append(_node(
                    "assessment", "verify", ["a", "b"], "assessment",
                    "str", relation="higher_count",
                ))
                # Keep only answer-relevant outputs; winner is needed by
                # assessment and therefore does not become an unused branch.
                operations[-1]["inputs"] = ["winner", "a", "b"]
                selections = {"conclusion": "assessment",
                              "first_count": "a", "second_count": "b"}
        else:
            raise ValueError("unsupported_v5_source_profile")

    typed = {row["id"]: row["type"] for row in operations}
    fields = {name: typed[identifier] for name, identifier in selections.items()}
    return {
        "version": VERSION, "source_profile": profile,
        "operations": operations,
        "output_spec": {
            "kind": shape,
            "fields": fields,
            "bindings": selections,
        },
    }


def validate_operation_graph(goal: dict) -> list[str]:
    """Reject bad edges, type-incompatible operations, unsupported outputs."""
    profile = goal.get("source_profile")
    output = goal.get("output_spec")
    if profile not in PROFILES or not isinstance(output, dict):
        return ["v5_bad_goal_schema"]
    if output.get("kind") not in OUTPUTS:
        return ["v5_bad_output_kind"]
    expected = _design_goal(profile, output["kind"])
    # This grammar deliberately has a small, finite, fully executable family.
    # Refuse arbitrary synthesized IR rather than silently accepting it.
    if goal != expected:
        return ["v5_goal_not_well_typed_or_canonical"]
    nodes = goal["operations"]
    if not (1 <= len(nodes) <= 8):
        return ["v5_goal_op_budget_exceeded"]
    produced = {}
    for node in nodes:
        if any(ref not in produced for ref in node["inputs"]):
            return ["v5_goal_non_topological_dependency"]
        produced[node["id"]] = node["type"]
    if set(output["bindings"].values()) - set(produced):
        return ["v5_unbound_output"]
    return []


def compose_goal_plan(*, seed: int, ordinal: int, difficulty: str,
                      hops: int, goal_keys: tuple[str, ...]) -> dict:
    profile, shape = selected_profile_output(seed, ordinal)
    # Reuse the proven v4 bounded source planner, not its final-Goal grammar.
    plan = compose_typed_goal_plan(
        seed=seed, ordinal=ordinal, difficulty=difficulty,
        hops=hops, goal_keys=goal_keys, kind_override=profile,
    )
    goal = _design_goal(profile, shape)
    assert not validate_operation_graph(goal)
    plan["composer_policy"] = VERSION
    plan["goal_spec"] = goal
    answer_type = (
        "dict" if shape in {"structured_multi_field",
                             "evidence_based_explanation"}
        else goal["output_spec"]["fields"]["answer"]
    )
    if profile != "entity_resolution":
        last = plan["macro_tasks"][-1]
        last["operation"] = "synthesize_operation_dag"
        last["solver_template"] = last["operation"]
        last["relation_need"] = (
            "Evaluate the typed Goal operation dependencies and render "
            "the requested output contract"
        )
        last["output_schema"] = {"answer": answer_type}
    plan["goal"] = {
        "answer_field": "answer", "answer_type": answer_type,
        "public_label": goal["output_spec"]["kind"].replace("_", " "),
        "cardinality": "exactly_one",
        "meaning": (
            "Determine the entity at the end of the documented relation chain"
            if profile == "entity_resolution" else
            "Evaluate the declared operation graph and format its "
            "declared typed output"
        ),
    }
    return plan


def measurement_world(*, plan: dict, binding: dict,
                      seed: int, ordinal: int) -> dict | None:
    profile = plan["goal_spec"]["source_profile"]
    if profile == "entity_resolution":
        return None
    proxy = {"goal_spec": {"kind": profile}}
    return v4_measurement_world(
        plan=proxy, binding=binding, seed=seed, ordinal=ordinal,
    )


def evaluate_goal_value(row: dict, goal: dict):
    """Pure type-aware Goal evaluation, independent of Program IR."""
    from .program_ir import great_circle_km
    from .goal_resolution import goal_values
    profile = goal["source_profile"]
    if profile == "entity_resolution":
        return row["entity_final"]
    values = {"entity": row["entity_final"]}
    if profile == "aggregation":
        rows = row["observations"]
        values["count"] = len({
            r["observation_id"] for r in rows
            if r["observation_year"] >= 2025
        })
    else:
        values["a"] = (
            row["date_a"] if profile == "temporal_gap"
            else (row["latitude_a"], row["longitude_a"])
            if profile == "geographic_distance" else row["metric_a"]
        )
        values["b"] = (
            row["date_b"] if profile == "temporal_gap"
            else (row["latitude_b"], row["longitude_b"])
            if profile == "geographic_distance" else row["metric_b"]
        )
        for op in goal["operations"]:
            ident, kind = op["id"], op["op"]
            if kind == "lookup":
                continue
            if kind == "arithmetic":
                a, b = (values[k] for k in op["inputs"])
                values[ident] = (
                    abs(a-b) if op["function"] == "absolute_difference"
                    else round(a/b, op["round_digits"])
                )
            elif kind == "temporal_difference":
                a, b = [dt.date.fromisoformat(values[k])
                        for k in op["inputs"]]
                values[ident] = abs((b-a).days)
            elif kind == "temporal_midpoint":
                a, b = [dt.date.fromisoformat(values[k])
                        for k in op["inputs"]]
                values[ident] = (min(a, b) +
                                 dt.timedelta(days=abs((b-a).days)//2)
                                ).isoformat()
            elif kind == "geographic_distance":
                a, b = (values[k] for k in op["inputs"])
                values[ident] = round(great_circle_km(*a, *b))
            elif kind == "compare_select":
                a, b = (values[k] for k in op["inputs"])
                values[ident] = goal_values("comparison", a, b)
            elif kind == "verify":
                a, b = values["a"], values["b"]
                values[ident] = ("first" if a > b else
                                 "second" if b > a else "equal")
    bindings = goal["output_spec"]["bindings"]
    answer = {k: values[v] for k,v in bindings.items()}
    return answer["answer"] if "answer" in answer else answer


def validate_measurement_world(plan: dict, binding: dict, *,
                               seed: int, ordinal: int) -> list[str]:
    goal = plan.get("goal_spec", {})
    issues = validate_operation_graph(goal)
    if issues:
        return issues
    actual = binding.get("world_model", {}).get("goal_measurements")
    expected = measurement_world(
        plan=plan, binding=binding, seed=seed, ordinal=ordinal,
    )
    if actual != expected:
        return ["v5_noncanonical_source_evidence"]
    if expected is None:
        return []
    target = binding["target_case"]
    answers = {
        case: evaluate_goal_value(row, goal)
        for case, row in expected["by_case"].items()
    }
    reference = json.dumps(answers[target], sort_keys=True)
    if any(json.dumps(answer, sort_keys=True) == reference
           for case, answer in answers.items() if case != target):
        return ["v5_non_discriminating_goal_answer"]
    return []


def goal_question(plan: dict) -> str:
    goal = plan["goal_spec"]
    profile, output = goal["source_profile"], goal["output_spec"]
    phrase = ", followed by ".join(
        _RELATION_PHRASES[k] for k in plan["goal_recipe_keys"]
    )
    scope = "the entity identified through the documented relation sequence (" + phrase + ")"
    if plan["goal_recipe_keys"].count("containedinplace") >= 2:
        scope += " into the larger administrative area"
    if profile == "entity_resolution":
        return plan["goal"]["semantic"]["question"]
    if profile == "aggregation":
        suffix = ("how many distinct observations from 2025 onward "
                  "(inclusive) are in the log")
    elif profile == "temporal_gap":
        suffix = ("the number of days between its two recorded milestones"
                  if output["kind"] == "single_number" else
                  "the midpoint calendar date between the two recorded milestones"
                  if output["kind"] == "date_time" else
                  "both the elapsed days and midpoint calendar date")
    elif profile == "geographic_distance":
        suffix = ("the great-circle distance in kilometres between its two "
                  "recorded sites, rounded to the nearest kilometre")
    elif profile == "comparison":
        suffix = ("which of its two independent tallies is higher (first or second)"
                  if output["kind"] == "single_entity" else
                  "the higher tally source and the absolute count difference"
                  if output["kind"] == "structured_multi_field" else
                  "an evidence-based assessment of which recorded tally is higher, "
                  "including both observed counts")
    else:
        suffix = ("the absolute difference between its two recorded counts"
                  if profile == "numeric_difference" else
                  "the ratio of the first count to the second, rounded to two decimals")
        if output["kind"] == "structured_multi_field":
            suffix = ("both the absolute difference and the first-to-second "
                      "ratio of its two recorded counts")
        if output["kind"] == "evidence_based_explanation":
            suffix = ("an evidence-based assessment of which recorded count "
                      "is higher, including the two observed counts")
    if output["kind"] == "single_number":
        return f"For {scope}, calculate {suffix}. Return only one number."
    if output["kind"] == "single_entity":
        return f"For {scope}, determine {suffix}. Return only one label."
    if output["kind"] == "date_time":
        return f"For {scope}, determine {suffix}. Return the date as YYYY-MM-DD."
    fields = ", ".join(output["fields"])
    return (f"For {scope}, return a structured answer with fields "
            f"{fields}, giving {suffix}. Do not add unrelated facts.")
