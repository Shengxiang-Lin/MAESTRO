"""Output-first, typed backward Goal planning (opt-in Stage 1 v6).

Output demands are sampled first, with semantic units. Required operators,
source capabilities and the partial-order DAG are then recursively derived
from those demands. Shared operations and source evidence are deduplicated.
This module never imports benchmark questions, answers or assets.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import random

from environment.schema import digest
from .typed_goal_v4 import measurement_world as _v4_measurements, _RELATION_PHRASES

VERSION = "goal_conditioned_output_first_v6"
WORLD_VERSION = "maestro.output_first_observations.v1"

OUTPUT_KINDS = (
    "single_entity", "single_number", "date_time",
    "structured_multi_field", "evidence_based_explanation",
)
# Semantic field demand, not merely its primitive Python type.
FIELD_TYPES = {
    "entity": "str",
    "higher_source": "str",
    "difference": "int",
    "ratio": "float",
    "duration_days": "int",
    "midpoint_date": "str",
    "distance_km": "int",
    "observation_count": "int",
    "conclusion": "str",
    "first_count": "int",
    "second_count": "int",
}
PAIRED_NEEDS = {
    "difference": "numeric", "ratio": "numeric",
    "higher_source": "numeric", "conclusion": "numeric",
    "first_count": "numeric", "second_count": "numeric",
    "duration_days": "temporal", "midpoint_date": "temporal",
    "distance_km": "geographic",
}
FIELD_WORDING = {
    "entity": "the final entity label",
    "higher_source": "which source has the higher observation count (first or second)",
    "difference": "the absolute difference between the first and second counts",
    "ratio": "the ratio of the first count to the second, rounded to two decimals",
    "duration_days": "the elapsed days between two recorded dates",
    "midpoint_date": "the midpoint date between two recorded dates (YYYY-MM-DD)",
    "distance_km": "the great-circle distance between recorded sites, rounded to the nearest kilometre",
    "observation_count": "the count of distinct logged observations dated in or after 2025",
    "conclusion": "the source-supported judgment of which observed count is higher",
    "first_count": "the first independently observed count",
    "second_count": "the second independently observed count",
}


def selected_output(seed: int, ordinal: int) -> dict:
    rng = random.Random(int(digest(
        seed, ordinal, VERSION, "output-first-selection",
    )[:15], 16))
    shape = rng.choice(OUTPUT_KINDS)
    if shape == "single_entity":
        chosen = rng.choice(("entity", "higher_source"))
        bindings = {"answer": chosen}
    elif shape == "single_number":
        chosen = rng.choice((
            "difference", "ratio", "duration_days",
            "distance_km", "observation_count",
        ))
        bindings = {"answer": chosen}
    elif shape == "date_time":
        bindings = {"answer": "midpoint_date"}
    elif shape == "evidence_based_explanation":
        bindings = {
            "conclusion": "conclusion", "first_count": "first_count",
            "second_count": "second_count",
        }
    else:
        # Two or three independently meaningful output requirements;
        # not one of a handful of hard-coded Profile/Output pairs.
        palette = (
            "entity", "difference", "ratio", "duration_days",
            "midpoint_date", "distance_km", "observation_count",
            "higher_source",
        )
        demands = rng.sample(palette, k=rng.choice((2, 3)))
        # A multi-field answer must contain a derived field.
        if all(name == "entity" for name in demands):
            demands[-1] = "difference"
        bindings = {name: name for name in sorted(demands)}
    return {
        "kind": shape,
        "fields": {name: FIELD_TYPES[semantic]
                   for name, semantic in bindings.items()},
        "bindings": bindings,
    }


def _backward_goal(output: dict) -> dict:
    """Resolve requested fields against typed producer rules; memoize nodes."""
    if output["kind"] not in OUTPUT_KINDS:
        raise ValueError("v6_unsupported_output")
    if set(output["fields"]) != set(output["bindings"]):
        raise ValueError("v6_output_schema_mismatch")
    nodes = []
    seen = {}
    capabilities = set()

    def add(op: str, name: str, deps: tuple[str, ...],
            typ: str, **meta) -> str:
        if name not in seen:
            nodes.append({
                "id": name, "op": op, "inputs": list(deps),
                "output": name, "type": typ, **meta,
            })
            seen[name] = name
        return name

    def solve(field: str) -> str:
        if field in seen:
            return field
        if field == "entity":
            return add("lookup", "entity", (), "str",
                       source="relation_chain")
        if field in ("a", "b"):
            solve("entity")
            return add("lookup", field, ("entity",), "record",
                       source="metric_"+field)
        if field == "observations":
            solve("entity")
            capabilities.add("aggregate")
            return add("lookup", field, ("entity",), "rows",
                       source="observation_log")
        if field in ("first_count", "second_count"):
            capabilities.add("numeric")
            upstream = "a" if field == "first_count" else "b"
            solve(upstream)
            return add("source_value", field, (upstream,), "int",
                       source_field="metric_"+upstream)
        if field in ("difference", "ratio", "higher_source", "conclusion",
                     "duration_days", "midpoint_date", "distance_km"):
            kind = PAIRED_NEEDS[field]
            capabilities.add(kind)
            solve("a")
            solve("b")
            if field == "difference":
                return add("arithmetic", field, ("a", "b"), "int",
                           function="absolute_difference", round_digits=0)
            if field == "ratio":
                return add("arithmetic", field, ("a", "b"), "float",
                           function="ratio", round_digits=2)
            if field == "duration_days":
                return add("temporal_difference", field, ("a", "b"), "int")
            if field == "midpoint_date":
                return add("temporal_midpoint", field, ("a", "b"), "str")
            if field == "distance_km":
                return add("geographic_distance", field, ("a", "b"), "int",
                           distance_unit="km")
            if field == "higher_source":
                return add("compare_select", field, ("a", "b"), "str",
                           direction="higher")
            solve("higher_source")
            return add("verify", field, ("higher_source", "a", "b"), "str",
                       relation="higher_count")
        if field == "observation_count":
            solve("observations")
            return add("count_aggregate", field, ("observations",), "int",
                       year_gte=2025, unique_by="observation_id")
        raise ValueError("v6_unresolved_semantic_field:"+field)

    for demand in output["bindings"].values():
        solve(demand)
    required = {
        "numeric": ("metric_a", "metric_b"),
        "temporal": ("metric_a", "metric_b"),
        "geographic": ("metric_a", "metric_b"),
        "aggregate": ("observation_log",),
    }
    sources = tuple(dict.fromkeys(
        sid for category in sorted(capabilities)
        for sid in required[category]
    ))
    source_fields = {}
    for sid in sources:
        fields = {"entity_final": "str"}
        if sid == "observation_log":
            fields.update({
                "observation_id": "str", "observation_year": "int",
            })
        else:
            side = sid[-1]
            if "numeric" in capabilities:
                fields["metric_"+side] = "int"
            if "temporal" in capabilities:
                fields["date_"+side] = "str"
            if "geographic" in capabilities:
                fields["latitude_"+side] = "float"
                fields["longitude_"+side] = "float"
        source_fields[sid] = fields
    return {
        "version": VERSION,
        "source_profile": (
            "entity_resolution" if not capabilities else
            "+".join(sorted(capabilities))
        ),
        "capabilities": sorted(capabilities),
        "source_fields": source_fields,
        "operations": nodes,
        "output_spec": copy.deepcopy(output),
    }


def validate_operation_graph(goal: dict) -> list[str]:
    if not isinstance(goal, dict) or goal.get("version") != VERSION:
        return ["v6_bad_goal_version"]
    output = goal.get("output_spec")
    if not isinstance(output, dict):
        return ["v6_missing_output_spec"]
    if output.get("kind") not in OUTPUT_KINDS:
        return ["v6_bad_output_kind"]
    if not isinstance(output.get("bindings"), dict):
        return ["v6_bad_output_bindings"]
    if not isinstance(output.get("fields"), dict):
        return ["v6_bad_output_types"]
    if any(field not in FIELD_TYPES or
           output["fields"].get(name) != FIELD_TYPES[field]
           for name, field in output["bindings"].items()):
        return ["v6_semantic_type_mismatch"]
    if (output["kind"] == "structured_multi_field" and
            not 2 <= len(output["fields"]) <= 3):
        return ["v6_multi_field_budget"]
    if (output["kind"] in ("single_number", "single_entity", "date_time") and
            set(output["bindings"]) != {"answer"}):
        return ["v6_single_field_mismatch"]
    if output["kind"] == "evidence_based_explanation" and (
            set(output["bindings"]) !=
            {"conclusion", "first_count", "second_count"}):
        return ["v6_explanation_fields_mismatch"]
    try:
        expected = _backward_goal(output)
    except ValueError:
        return ["v6_unresolvable_output_demands"]
    if goal != expected:
        return ["v6_goal_graph_not_canonical"]
    if not 1 <= len(goal["operations"]) <= 13:
        return ["v6_operation_budget"]
    return []


def compose_goal_plan(*, seed: int, ordinal: int, difficulty: str,
                      hops: int, goal_keys: tuple[str, ...]) -> dict:
    from .goal_composer import compose_plan, NEEDS_VERSION
    output = selected_output(seed, ordinal)
    goal = _backward_goal(output)
    source_ids = list(goal["source_fields"])
    # One Macro per distinct evidence source, then a terminal synthesis.
    # Cross-domain outputs can require three independent source Macros.
    # This versioned bound preserves v3/v4/v5's unchanged 9-Macro limit.
    max_base = 11 - (len(source_ids)+1 if source_ids else 0)
    for attempt in range(140):
        probe = ordinal + 1009*attempt
        plan = compose_plan(
            seed=seed, ordinal=probe, difficulty=difficulty,
            hops=hops, goal_keys=goal_keys, policy=NEEDS_VERSION,
        )
        if (not source_ids or
            plan["mode"] == "trace" and
            len(plan["macro_tasks"]) <= max_base):
            break
    else:
        raise ValueError("v6_no_source_feasible_macro_budget")
    plan["composer_policy"] = VERSION
    plan["goal_spec"] = goal
    if source_ids:
        tasks = plan["macro_tasks"]
        terminal = tasks[-1]
        if (terminal["operation"] != "follow_yago" or
                terminal["output_schema"] != {"answer": "str"}):
            raise ValueError("v6_missing_typed_entity_handoff")
        terminal["output_schema"] = {"entity_final": "str"}
        terminal["role"] = "investigation"
        entity_mid = terminal["macro_id"]
        parents = []
        for sid in source_ids:
            mid = "M" + str(len(tasks)+1)
            fields = goal["source_fields"][sid]
            tasks.append({
                "macro_id": mid, "role": "investigation",
                "operation": "fetch_typed_evidence",
                "depends_on": [entity_mid],
                "input_contract": {
                    entity_mid: {"entity_final": "str"}
                },
                "output_schema": copy.deepcopy(fields),
                "source_scope": [sid],
                "solver_template": "fetch_typed_evidence",
                "relation_need": "Retrieve typed source fields from " + sid,
                "evidence_plan": {
                    "description": "Look up the independently maintained " + sid,
                    "source_observation_targets": {sid: 1},
                },
            })
            parents.append(mid)
        earlier = {t["macro_id"]: t for t in tasks}
        mid = "M" + str(len(tasks)+1)
        answer_type = (
            "dict" if output["kind"] in {
                "structured_multi_field", "evidence_based_explanation"
            } else output["fields"]["answer"]
        )
        tasks.append({
            "macro_id": mid, "role": "synthesis",
            "operation": "synthesize_operation_dag",
            "depends_on": parents,
            "input_contract": {
                dep: copy.deepcopy(earlier[dep]["output_schema"])
                for dep in parents
            },
            "output_schema": {"answer": answer_type}, "source_scope": [],
            "solver_template": "synthesize_operation_dag",
            "relation_need": "Execute necessary typed Goal operations and format output",
            "evidence_plan": {
                "description": "Derive each requested field from cited evidence",
                "source_observation_targets": {},
            },
        })
        plan["terminal_macro_id"] = mid
    result_type = (
        "dict" if output["kind"] in {
            "structured_multi_field", "evidence_based_explanation"
        } else output["fields"]["answer"]
    )
    plan["goal"] = {
        "answer_field": "answer", "answer_type": result_type,
        "public_label": output["kind"].replace("_", " "),
        "cardinality": "exactly_one",
        "meaning": (
            "Determine the entity at the end of the documented relation chain"
            if not source_ids else
            "Determine each requested output field from the minimal "
            "dependent typed evidence operation graph"
        ),
    }
    if len(plan["macro_tasks"]) > 11:
        raise ValueError("v6_macro_budget_exceeded")
    return plan


def measurement_world(*, plan: dict, binding: dict,
                      seed: int, ordinal: int) -> dict | None:
    goal = plan["goal_spec"]
    capabilities = goal["capabilities"]
    if not capabilities:
        return None
    world = {
        case: {"entity_final": path[-1]["object_label"]}
        for case, path in binding["candidate_paths"].items()
    }
    for capability in capabilities:
        v4kind = {
            "numeric": (
                "comparison" if any(
                    item["op"] in {"compare_select", "verify"}
                    for item in goal["operations"]
                ) else "numeric_ratio"
            ),
            "temporal": "temporal_gap",
            "geographic": "geographic_distance",
            "aggregate": "aggregation",
        }[capability]
        measured = _v4_measurements(
            plan={"goal_spec": {"kind": v4kind}},
            binding=binding, seed=seed, ordinal=ordinal,
        )
        for case, values in measured["by_case"].items():
            for field, value in values.items():
                if field != "entity_final":
                    if field in world[case]:
                        raise ValueError("v6_duplicate_measurement_field")
                    world[case][field] = value
    return {"version": WORLD_VERSION, "by_case": world,
            "source_fields": copy.deepcopy(goal["source_fields"])}


def evaluate_goal_value(row: dict, goal: dict):
    """Independent reference for final answer, separate from Gold IR."""
    from .program_ir import great_circle_km
    from .goal_resolution import goal_values
    values = {"entity": row["entity_final"]}
    for node in goal["operations"]:
        ident, op = node["id"], node["op"]
        if ident in values:
            continue
        if op == "lookup":
            # Raw evidence A/B is carried by record source; one table can
            # legitimately provide several typed fields.
            values[ident] = (
                row["observations"] if ident == "observations" else row
            )
        elif op == "source_value":
            values[ident] = row[node["source_field"]]
        elif op == "count_aggregate":
            values[ident] = len({
                record["observation_id"]
                for record in row["observations"]
                if record["observation_year"] >= node["year_gte"]
            })
        elif op == "arithmetic":
            a, b = row["metric_a"], row["metric_b"]
            values[ident] = (
                abs(a-b) if node["function"] == "absolute_difference"
                else round(a/b, node["round_digits"])
            )
        elif op == "temporal_difference":
            values[ident] = abs((
                dt.date.fromisoformat(row["date_b"])
                - dt.date.fromisoformat(row["date_a"])
            ).days)
        elif op == "temporal_midpoint":
            a, b = [dt.date.fromisoformat(row["date_"+side])
                    for side in ("a", "b")]
            values[ident] = (min(a,b) + dt.timedelta(
                days=abs((b-a).days)//2
            )).isoformat()
        elif op == "geographic_distance":
            values[ident] = round(great_circle_km(
                row["latitude_a"], row["longitude_a"],
                row["latitude_b"], row["longitude_b"],
            ))
        elif op in {"compare_select", "verify"}:
            values[ident] = goal_values(
                "comparison", row["metric_a"], row["metric_b"],
            )
        else:
            raise ValueError("v6_unsupported_reference_operation")
    answer = {
        name: values[producing_op] for name, producing_op
        in goal["output_spec"]["bindings"].items()
    }
    if "answer" in answer:
        return answer["answer"]
    return answer


def validate_measurement_world(plan: dict, binding: dict, *,
                               seed: int, ordinal: int) -> list[str]:
    goal = plan.get("goal_spec", {})
    issues = validate_operation_graph(goal)
    if issues:
        return issues
    expected = measurement_world(
        plan=plan, binding=binding, seed=seed, ordinal=ordinal,
    )
    actual = binding.get("world_model", {}).get("goal_measurements")
    if actual != expected:
        return ["v6_noncanonical_goal_measurements"]
    if expected is None:
        return []
    target = binding["target_case"]
    answers = {
        case: evaluate_goal_value(item, goal)
        for case, item in expected["by_case"].items()
    }
    baseline = json.dumps(answers[target], sort_keys=True)
    if any(json.dumps(value, sort_keys=True) == baseline
           for case, value in answers.items() if case != target):
        return ["v6_non_discriminating_shadow_answer"]
    return []


def goal_question(plan: dict) -> str:
    goal = plan["goal_spec"]
    output = goal["output_spec"]
    chain = ", followed by ".join(
        _RELATION_PHRASES[k] for k in plan["goal_recipe_keys"]
    )
    scope = (
        "the entity reached by the documented relationship sequence "
        "(" + chain + ")"
    )
    if plan["goal_recipe_keys"].count("containedinplace") >= 2:
        scope += " in the larger administrative area"
    if output["kind"] == "single_entity" and (
            output["bindings"]["answer"] == "entity"):
        return plan["goal"]["semantic"]["question"]
    descriptors = [
        FIELD_WORDING[semantic]
        for semantic in output["bindings"].values()
    ]
    if output["kind"] == "single_entity":
        end = "Determine " + descriptors[0] + ". Return one label."
    elif output["kind"] == "single_number":
        end = "Calculate " + descriptors[0] + ". Return only one number."
    elif output["kind"] == "date_time":
        end = "Determine " + descriptors[0] + ". Return a YYYY-MM-DD date."
    else:
        parts = ", ".join(
            "'" + name + "'" for name in output["fields"]
        )
        end = (
            "Return a " + output["kind"].replace("_", " ") +
            " with exactly these fields: " + parts +
            ". They respectively represent " + "; ".join(descriptors) +
            "."
        )
    return "For " + scope + ", " + end
