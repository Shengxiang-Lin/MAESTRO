"""Canonical Stage 1 output-first Goal planning and measurement World.

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
_RELATION_PHRASES = {
    "birthplace": "birthplace", "deathplace": "place of death",
    "containedinplace": "contained administrative area",
    "location": "location", "doctoraladvisor": "doctoral advisor",
    "fieldofwork": "research field", "spouse": "spouse",
    "memberof": "organization membership", "neighbor": "neighboring place",
    "twinnedwith": "twin municipality", "nationality": "nationality",
    "capital": "capital",
}


GOAL_SCHEMA = "maestro.goal_plan.v1"
COMPOSER = "output_first"
WORLD_SCHEMA = "maestro.goal_measurements.v1"

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


def sample_output(seed: int, ordinal: int) -> dict:
    rng = random.Random(int(digest(
        seed, ordinal, GOAL_SCHEMA, "output-first-selection",
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
        # Sample an operational family first, then combine independently
        # meaningful requirements inside the SAME investigation. Neutral
        # entity/observation fields can accompany any family; mixing a
        # geographic distance, a temporal midpoint and arbitrary numeric
        # comparison without a semantic link is deliberately disallowed.
        palette = (
            "entity", "difference", "ratio", "duration_days",
            "midpoint_date", "distance_km", "observation_count",
            "higher_source",
        )
        anchor = rng.choice((
            "difference", "ratio", "higher_source", "duration_days",
            "midpoint_date", "distance_km",
        ))
        family = PAIRED_NEEDS[anchor]
        compatible = tuple(
            field for field in palette
            if field in {"entity", "observation_count"}
            or PAIRED_NEEDS.get(field) == family
        )
        demands = rng.sample(compatible, k=min(rng.choice((2, 3)),
                                               len(compatible)))
        if anchor not in demands:
            demands[0] = anchor
        bindings = {name: name for name in sorted(demands)}
    return {
        "kind": shape,
        "fields": {name: FIELD_TYPES[semantic]
                   for name, semantic in bindings.items()},
        "bindings": bindings,
    }


def _build_operation_dag(output: dict) -> dict:
    """Resolve requested fields against typed producer rules; memoize nodes."""
    if output["kind"] not in OUTPUT_KINDS:
        raise ValueError("goal_unsupported_output")
    if set(output["fields"]) != set(output["bindings"]):
        raise ValueError("goal_output_schema_mismatch")
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
        raise ValueError("goal_unresolved_semantic_field:"+field)

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
        "version": GOAL_SCHEMA,
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
    if not isinstance(goal, dict) or goal.get("version") != GOAL_SCHEMA:
        return ["goal_bad_goal_version"]
    output = goal.get("output_spec")
    if not isinstance(output, dict):
        return ["goal_missing_output_spec"]
    if output.get("kind") not in OUTPUT_KINDS:
        return ["goal_bad_output_kind"]
    if not isinstance(output.get("bindings"), dict):
        return ["goal_bad_output_bindings"]
    if not isinstance(output.get("fields"), dict):
        return ["goal_bad_output_types"]
    if any(field not in FIELD_TYPES or
           output["fields"].get(name) != FIELD_TYPES[field]
           for name, field in output["bindings"].items()):
        return ["goal_semantic_type_mismatch"]
    if (output["kind"] == "structured_multi_field" and
            not 2 <= len(output["fields"]) <= 3):
        return ["goal_multi_field_budget"]
    if (output["kind"] in ("single_number", "single_entity", "date_time") and
            set(output["bindings"]) != {"answer"}):
        return ["goal_single_field_mismatch"]
    if output["kind"] == "evidence_based_explanation" and (
            set(output["bindings"]) !=
            {"conclusion", "first_count", "second_count"}):
        return ["goal_explanation_fields_mismatch"]
    try:
        expected = _build_operation_dag(output)
    except ValueError:
        return ["goal_unresolvable_output_demands"]
    if goal != expected:
        return ["goal_goal_graph_not_canonical"]
    if not 1 <= len(goal["operations"]) <= 13:
        return ["goal_operation_budget"]
    return []


def operation_subgraph_leq(smaller: dict, larger: dict) -> bool:
    """Typed partial order on operation DAGs, independent of output format.

    G1 <= G2 iff every typed operation (including its prerequisite edges)
    in G1 occurs in G2. Distinct output formats may have the same Goal
    operation equivalence class. This is not an arbitrary total ranking.
    """
    if validate_operation_graph(smaller) or validate_operation_graph(larger):
        return False
    expanded = {node["id"]: node for node in larger["operations"]}
    return all(expanded.get(node["id"]) == node
               for node in smaller["operations"])


def compose_goal_plan(*, seed: int, ordinal: int, difficulty: str,
                      hops: int, goal_keys: tuple[str, ...]) -> dict:
    from environment.generate_question.planning.evidence_plan import compose_evidence_plan
    output = sample_output(seed, ordinal)
    goal = _build_operation_dag(output)
    source_ids = list(goal["source_fields"])
    # One Macro per distinct evidence source, then a terminal synthesis.
    # Cross-domain outputs can require three independent source Macros.
    # This versioned bound preserves v3/v4/v5's unchanged 9-Macro limit.
    max_base = 11 - (len(source_ids)+1 if source_ids else 0)
    for attempt in range(140):
        probe = ordinal + 1009*attempt
        plan = compose_evidence_plan(
            seed=seed, ordinal=probe, difficulty=difficulty,
            hops=hops, goal_keys=goal_keys,
        )
        # Budget applies to every feasible evidence route, not just Trace.
        # Event-qualified investigations must remain available to typed Goals.
        if len(plan["macro_tasks"]) <= max_base:
            break
    else:
        raise ValueError("goal_no_source_feasible_macro_budget")
    plan["composer_policy"] = COMPOSER
    plan["goal_spec"] = goal
    if source_ids:
        tasks = plan["macro_tasks"]
        terminal = tasks[-1]
        if (terminal["operation"] != "follow_yago" or
                terminal["output_schema"] != {"answer": "str"}):
            raise ValueError("goal_missing_typed_entity_handoff")
        terminal["output_schema"] = {"entity_final": "str"}
        terminal["role"] = "investigation"
        entity_mid = terminal["macro_id"]
        parents = []
        for sid in source_ids:
            mid = "M" + str(len(tasks)+1)
            fields = goal["source_fields"][sid]
            # Browse-only observations are globally visible and do not
            # require a resolved entity. Their entity-to-fact association
            # is made *later*, in synthesis. Keyed sources still need
            # the upstream resolved entity before they may be queried.
            browse = sid in {"metric_a", "metric_b"}
            dependencies = [] if browse else [entity_mid]
            tasks.append({
                "macro_id": mid,
                "role": "discovery" if browse else "investigation",
                "operation": "fetch_typed_evidence",
                "depends_on": dependencies,
                "input_contract": (
                    {} if browse else {entity_mid: {"entity_final": "str"}}
                ),
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
            "depends_on": [entity_mid] + parents,
            "input_contract": {
                dep: copy.deepcopy(earlier[dep]["output_schema"])
                for dep in [entity_mid] + parents
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
        raise ValueError("goal_macro_budget_exceeded")
    return plan


def build_measurements(*, plan: dict, binding: dict,
                       seed: int, ordinal: int) -> dict | None:
    """Sample canonical private measurements with non-degenerate answers.

    The target and counterfactual candidate cases share the same observation
    schema, but have distinct dependent answers. Values vary with task seed
    and ordinal so none of the common numeric outputs is globally constant.
    """
    from environment.schema import digest
    goal = plan["goal_spec"]
    needs = goal["capabilities"]
    if not needs:
        return None
    rng = random.Random(int(digest(
        seed, ordinal, "stage1-world-measurements", GOAL_SCHEMA,
    )[:15], 16))
    target = binding["target_case"]
    paths = binding["candidate_paths"]
    cases = [target] + sorted(x for x in paths if x != target)
    base_delta = rng.randrange(14, 240)
    base_denominator = rng.randrange(12, 87)
    base_smaller = rng.randrange(11, 170)
    base_gap_days = rng.randrange(8, 240)
    base_first_day = rng.randrange(2300, 8500)
    base_latitude = rng.uniform(8.0, 51.0)
    base_longitude = rng.uniform(-120.0, -43.0)
    base_lat_step = rng.uniform(0.10, 1.15)
    base_long_step = rng.uniform(0.10, 1.85)
    base_qualified = rng.randrange(2, 14)
    comparison = any(
        node["op"] in {"compare_select", "verify"}
        for node in goal["operations"]
    )
    target_first = bool(rng.randrange(2))
    entries = {}
    for index, case in enumerate(cases):
        label = paths[case][-1]["object_label"]
        row = {"entity_final": label}
        if "numeric" in needs:
            if comparison:
                smaller = base_smaller + 5*index
                larger = smaller + base_delta + 19*index
                first_wins = target_first if index == 0 else not target_first
                row.update({
                    "metric_a": larger if first_wins else smaller,
                    "metric_b": smaller if first_wins else larger,
                })
            else:
                difference = base_delta + 17*index
                row.update({
                    "metric_b": base_denominator,
                    "metric_a": base_denominator + difference,
                })
        if "temporal" in needs:
            first = dt.date(2000, 1, 1) + dt.timedelta(
                days=base_first_day + index*250,
            )
            gap = base_gap_days + index*19
            row.update({
                "date_a": first.isoformat(),
                "date_b": (first + dt.timedelta(days=gap)).isoformat(),
            })
        if "geographic" in needs:
            # A task-specific pair of synthetic coordinates, NOT a claim
            # about the real-world location of the source entity.
            lat = base_latitude + index*0.03
            lon = base_longitude + index*0.04
            row.update({
                "latitude_a": round(lat, 6),
                "longitude_a": round(lon, 6),
                "latitude_b": round(
                    lat + base_lat_step + index*0.31, 6,
                ),
                "longitude_b": round(
                    lon + base_long_step + index*0.27, 6,
                ),
            })
        if "aggregate" in needs:
            n = base_qualified + index
            older = rng.randrange(1, 5)
            row["observations"] = [{
                "entity_final": label,
                "observation_id": "OBS-" + digest(
                    seed, ordinal, case, j, "observation"
                )[:12].upper(),
                "observation_year": (
                    (2025 + (j % 2)) if j < n else (2020 + j%4)
                ),
            } for j in range(n+older)]
        entries[case] = row
    return {
        "version": WORLD_SCHEMA,
        "by_case": entries,
        "source_fields": copy.deepcopy(goal["source_fields"]),
    }

def evaluate_goal_value(row: dict, goal: dict):
    """Independent reference for final answer, separate from Gold IR."""
    from environment.generate_question.execution.program_ir import great_circle_km
    from environment.generate_question.planning.goal_resolution import goal_values
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
            raise ValueError("goal_unsupported_reference_operation")
    answer = {
        name: values[producing_op] for name, producing_op
        in goal["output_spec"]["bindings"].items()
    }
    if "answer" in answer:
        return answer["answer"]
    return answer


def validate_measurements(plan: dict, binding: dict, *,
                               seed: int, ordinal: int) -> list[str]:
    goal = plan.get("goal_spec", {})
    issues = validate_operation_graph(goal)
    if issues:
        return issues
    expected = build_measurements(
        plan=plan, binding=binding, seed=seed, ordinal=ordinal,
    )
    actual = binding.get("world_model", {}).get("goal_measurements")
    if actual != expected:
        return ["goal_noncanonical_goal_measurements"]
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
        return ["goal_non_discriminating_shadow_answer"]
    return []


def goal_question(plan: dict) -> str:
    """Express the frozen semantic target as a user referent, not a DAG path.

    Gold still follows every canonical relation. Only the surface avoids
    exposing an internal 'relationship sequence' or instruction program.
    """
    from environment.generate_question.grounding.scene_semantics import focus_phrase

    goal = plan["goal_spec"]
    output = goal["output_spec"]
    scope = focus_phrase(tuple(plan["goal_recipe_keys"]))
    condition = plan.get("condition")
    if condition:
        if condition["field"] == "event_status":
            scope += ", using only the completed review event"
        elif condition["field"] == "event_year":
            scope += ", using only a review event in or after 2025"
        else:
            raise ValueError("unrecognized_event_condition_in_goal_question")
    if (output["kind"] == "single_entity"
            and output["bindings"]["answer"] == "entity"):
        return plan["goal"]["semantic"]["question"]

    descriptors = [
        FIELD_WORDING[semantic]
        for semantic in output["bindings"].values()
    ]
    prefix = "For " + scope + ", "
    if output["kind"] == "single_entity":
        return prefix + (
            "identify " + descriptors[0] + ". Return one label."
        )
    if output["kind"] == "single_number":
        return prefix + (
            "calculate " + descriptors[0] + ". Return only one number."
        )
    if output["kind"] == "date_time":
        return prefix + (
            "determine " + descriptors[0] + ". Return a YYYY-MM-DD date."
        )
    names = list(output["bindings"])
    details = "; ".join(
        "'" + name + "' means " + FIELD_WORDING[semantic]
        for name, semantic in output["bindings"].items()
    )
    if output["kind"] == "evidence_based_explanation":
        return prefix + (
            "provide an evidence-supported judgment comparing the recorded "
            "counts, including the first and second counts. Return exactly "
            "the fields 'conclusion', 'first_count' and 'second_count'. "
            + details + "."
        )
    return prefix + (
        "report the requested measurements using fields "
        + ", ".join("'" + name + "'" for name in names)
        + ". " + details + "."
    )
