"""Task-graph-first Stage-1 orchestration, lowered to ORIGINAL Gold Program IR.

Pipeline:
  Macro DAG -> information/relationship sketch -> real+fictional World
  -> evidence/source constraints -> frozen Macro solver -> old quality/replay.

No new independent execution engine; v1-v4 frozen scene semantics stay intact.
"""
from __future__ import annotations

from collections import Counter
import copy

from environment.schema import digest
from .macro_graph_sampler import (
    sample_graph, validate_graph, VERSION as GRAPH_VERSION,
)
from .relational_task_planner import plan_relations, validate_sketch
from .world_grounder import ground_world, validate_grounding
from .evidence_planner import (
    design_evidence, validate_evidence, materialize_world,
)
from . import knowledge_macro as v1

PATTERN = "task_graph_first_v1"
VERSION = "maestro.macro_plan.task_graph_first.v1"
WITNESS_VERSION = "maestro.knowledge_witness.task_graph_first.v1"
HASH_SALT = "canonical-macro-first-graph-grounding-v1"
NECESSITY_POLICY = "task_graph_first_replanning_v1"

SCHEMA_BY_TASK = {
    "identify": {"project_id": "str"},
    "event": {"project_id": "str", "event_id": "str",
              "motif_object": "str"},
    "directory_a": {"route_key": "str", "entity_0": "str"},
    "directory_b": {"route_key": "str", "entity_0": "str"},
    "associate": {"entity_0": "str"},
    "refine_b": {"entity_0": "str"},
    "crosscheck": {"entity_0": "str"},
    "answer": {"answer": "str"},
}


def lower_macro_tasks(graph: dict, clues: dict,
                      main_chain_hops: int) -> list[dict]:
    """Bind graph tasks to source scopes without changing their DAG edges."""
    validate_graph(graph)
    by_id, result = {}, []
    discovery = [node for node in graph["nodes"]
                 if node["role"] == "discovery"]
    if len(discovery) * 2 != len(clues):
        raise ValueError("graph_first_clue_count_not_supported")
    for index, node in enumerate(graph["nodes"]):
        key, mid = node["task_key"], node["macro_id"]
        if key.startswith("discover"):
            source_scope = [f"{mid}_a", f"{mid}_b"]
            schema = {"case_id": "str"}
        elif key == "identify":
            source_scope = ["entity_index"]
            schema = SCHEMA_BY_TASK[key]
        elif key == "event":
            source_scope = ["event_ledger"]
            schema = SCHEMA_BY_TASK[key]
        elif key in {"directory_a", "directory_b", "refine_b"}:
            source_scope = [
                "link_a" if key == "directory_a" else "link_b"
            ]
            schema = SCHEMA_BY_TASK[key]
        elif key == "crosscheck":
            source_scope = ["real_fork"]
            schema = SCHEMA_BY_TASK[key]
        elif key == "answer":
            source_scope = [
                f"relation_{i}" for i in range(1, main_chain_hops+1)
            ]
            schema = SCHEMA_BY_TASK[key]
        else:
            source_scope = []
            schema = SCHEMA_BY_TASK[key]
        parents = node["depends_on"]
        result.append({
            "macro_id": mid,
            "role": node["role"],
            "information_operation": key,
            "depends_on": list(parents),
            "input_contract": {
                pid: copy.deepcopy(by_id[pid]["output_schema"])
                for pid in parents
            },
            "output_schema": dict(schema),
            "source_scope": list(source_scope),
            "solver_template": key,
            "evidence_plan": {
                "description": node["information_goal"],
                "source_observation_targets": {
                    sid: 1 for sid in source_scope
                },
            },
        })
        by_id[mid] = result[-1]
    if Counter(
        sid for row in result for sid in row["source_scope"]
    ) != Counter([
        *clues, "entity_index", "event_ledger", "link_a", "link_b",
        "real_fork", *[f"relation_{i}"
                       for i in range(1, main_chain_hops+1)]
    ]):
        raise ValueError("graph_first_unbound_source_role")
    return result


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: v1.YagoKnowledgeIndex | None = None) -> dict:
    # STEP 1: no YAGO is consulted before this immutable structural choice.
    graph = sample_graph(seed=seed, ordinal=ordinal, difficulty=difficulty)
    # STEP 2: relation/claim/evidence requirements without concrete facts.
    sketch = plan_relations(
        graph, min_hops=min_hops, max_hops=max_hops,
    )
    # STEP 3: only now select real RDF witnesses and synthetic entities.
    grounded = ground_world(
        graph, sketch, kg, seed=seed, ordinal=ordinal,
        difficulty=difficulty, min_hops=min_hops, max_hops=max_hops,
        index=index,
    )
    # STEP 4: construct provenance-bearing evidence and field obligations.
    evidence = design_evidence(
        graph, sketch, grounded, seed=seed, ordinal=ordinal,
    )
    base = grounded["base_v4"]
    spec = copy.deepcopy(base)
    plan = spec["macro_plan"]
    plan.update({
        "version": VERSION, "kind": "task_graph_first_relational_investigation",
        "macro_tasks": lower_macro_tasks(
            graph, plan["clues"], len(plan["real_facts"])
        ),
        "terminal_macro_id": graph["terminal_macro_id"],
        "information_necessity_policy": NECESSITY_POLICY,
        "task_graph_first_shape": graph["shape"],
        "abstract_graph_hash": graph["graph_hash"],
        "relation_sketch_hash": sketch["sketch_hash"],
        "evidence_contract_hash": evidence["evidence_hash"],
        "required_event_status": "completed",
    })
    if graph["shape"] == "event_gated_association":
        for source in ("link_a", "link_b"):
            plan["source_schemas"][source] = {
                **plan["source_schemas"][source], "event_id": "str",
            }
    spec["reasoning_pattern"] = PATTERN
    spec["family"] = PATTERN
    spec["ordinal"] = ordinal
    spec["task_id"] = "question-gf-" + f"{ordinal:06d}-" + digest(
        seed, ordinal, graph["shape"], grounded["selected_attempt"]
    )[:8]
    spec["scene_id"] = f"task-graph-first-{ordinal:06d}"
    spec["task_first_macro_graph"] = graph
    spec["task_first_relation_sketch"] = sketch
    spec["task_first_grounding"] = grounded
    spec["task_first_evidence"] = evidence
    spec["macro_information_necessity_version"] = NECESSITY_POLICY
    spec["scenario"] = {
        **base["scenario"],
        "topic": "fictional project event investigation",
        "setting": (
            "a simulated project archive with event histories and "
            "cross-referenced real-world knowledge"
        ),
        "purpose": (
            "identify a fictional project, select its completed event, "
            "reconcile candidate references, and use a real YAGO "
            "secondary fact to establish a final linked entity"
        ),
        "motivation": (
            "information tasks were planned before world facts; the "
            "recorded events, competing directories and true YAGO "
            "relationships jointly constrain the final answer"
        ),
    }
    spec["knowledge_witness"]["version"] = WITNESS_VERSION
    spec["knowledge_witness"]["graph_first_relational_frozen"] = {
        "graph_hash": graph["graph_hash"],
        "relation_sketch_hash": sketch["sketch_hash"],
        "grounding_hash": grounded["grounding_hash"],
        "evidence_hash": evidence["evidence_hash"],
    }
    spec["knowledge_contract_hash"] = digest(
        plan, graph, sketch, grounded, evidence, spec["knowledge_witness"],
        seed, ordinal, HASH_SALT,
    )
    issues = validate_spec(spec)
    if issues:
        raise ValueError("invalid_task_graph_first_spec:" + ",".join(issues))
    return spec


def validate_spec(spec: dict) -> list[str]:
    try:
        plan = spec["macro_plan"]
        graph = spec["task_first_macro_graph"]
        sketch = spec["task_first_relation_sketch"]
        grounded = spec["task_first_grounding"]
        evidence = spec["task_first_evidence"]
        if (spec["reasoning_pattern"] != PATTERN
                or spec["family"] != PATTERN
                or plan["version"] != VERSION
                or spec["knowledge_witness"]["version"] != WITNESS_VERSION):
            return ["task_graph_first_wrong_version"]
        if graph["graph_hash"] != digest(
            {k: v for k, v in graph.items() if k != "graph_hash"},
            "graph-first-frozen"
        ):
            return ["task_graph_first_graph_tampered"]
        validate_graph(graph)
        problems = validate_sketch(sketch, graph)
        problems += validate_grounding(grounded, graph, sketch)
        problems += validate_evidence(evidence, graph, sketch, grounded)
        if problems:
            return problems
        if spec["knowledge_contract_hash"] != digest(
            plan, graph, sketch, grounded, evidence,
            spec["knowledge_witness"], spec["seed"], spec["ordinal"],
            HASH_SALT
        ):
            return ["task_graph_first_spec_hash_mismatch"]
        frozen = spec["knowledge_witness"]["graph_first_relational_frozen"]
        if frozen != {
            "graph_hash": graph["graph_hash"],
            "relation_sketch_hash": sketch["sketch_hash"],
            "grounding_hash": grounded["grounding_hash"],
            "evidence_hash": evidence["evidence_hash"],
        }:
            return ["task_graph_first_witness_disconnected"]
        base = grounded["base_v4"]["macro_plan"]
        for key in ("all_cases", "target_case", "clues",
                    "candidate_paths", "real_facts",
                    "real_fork", "synthetic_entities"):
            if plan[key] != base[key]:
                return ["task_graph_first_broken_world_binding:" + key]
        expected = lower_macro_tasks(
            graph, plan["clues"], len(plan["real_facts"])
        )
        if plan["macro_tasks"] != expected:
            return ["task_graph_first_macros_not_from_graph"]
        if plan["terminal_macro_id"] != graph["terminal_macro_id"]:
            return ["task_graph_first_terminal_changed"]
        if plan["task_graph_first_shape"] != graph["shape"]:
            return ["task_graph_first_wrong_shape"]
        original = copy.deepcopy(base["source_schemas"])
        if graph["shape"] == "event_gated_association":
            for sid in ("link_a", "link_b"):
                original[sid]["event_id"] = "str"
        if plan["source_schemas"] != original:
            return ["task_graph_first_wrong_evidence_schemas"]
        if set(plan["source_schemas"]) != {
            row["source_id"] for row in spec["sources"]
        }:
            return ["task_graph_first_source_registry_mismatch"]
        if plan["required_event_status"] != "completed":
            return ["task_graph_first_event_condition_modified"]
        return []
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return ["task_graph_first_contract_invalid:" + str(exc)[:160]]


def compile_macro_solvers(spec: dict) -> dict:
    issues = validate_spec(spec)
    if issues:
        raise ValueError("task_graph_first_bad_program:" + ",".join(issues))
    plan = spec["macro_plan"]
    program_nodes, groups, outputs = [], [], {}
    for task in plan["macro_tasks"]:
        local = []
        mid, kind = task["macro_id"], task["information_operation"]
        parents = [outputs[key] for key in task["depends_on"]]

        def add(op, *inputs, **kwargs):
            nid = f"{mid.lower()}_op{len(local)+1:02d}"
            local.append({
                "id": nid, "op": op, "inputs": list(inputs), **kwargs,
            })
            return nid

        if kind.startswith("discover"):
            matched = []
            for sid in task["source_scope"]:
                source = add("scan", source_id=sid)
                filtered = add(
                    "filter_eq", source, field="cue",
                    value=plan["clues"][sid]["value"],
                )
                matched.append(add("project", filtered, fields=["case_id"]))
            joined = add("join", matched[0], matched[1],
                         left_on="case_id", right_on="case_id")
            result = add("unique", joined)
        elif kind == "identify":
            merged = parents[0]
            for parent in parents[1:]:
                merged = add("join", merged, parent,
                             left_on="case_id", right_on="case_id")
            narrowed = add("unique", merged)
            found = add("lookup_by_key", narrowed, source_id="entity_index",
                        input_key_field="case_id",
                        source_key_field="case_id")
            result = add("project", found, fields=["project_id"])
        elif kind == "event":
            scan = add("scan", source_id="event_ledger")
            active = add("filter_eq", scan, field="event_status",
                         value=plan["required_event_status"])
            linked = add("join", parents[0], active,
                         left_on="project_id", right_on="project_id")
            result = add("project", linked,
                         fields=["project_id", "event_id", "motif_object"])
        elif kind in {"directory_a", "directory_b"}:
            sid = task["source_scope"][0]
            scanned = add("scan", source_id=sid)
            key = ("event_id" if plan["task_graph_first_shape"] ==
                   "event_gated_association" else "project_id")
            linked = add("join", parents[0], scanned,
                         left_on=key, right_on=key)
            reduced = add("project", linked,
                          fields=["route_key", "entity_0"])
            result = add("unique", reduced)
        elif kind == "associate":
            joined = add("join", parents[0], parents[1],
                         left_on="route_key", right_on="route_key")
            projected = add("project", joined, fields=["entity_0"])
            result = add("unique", projected)
        elif kind == "refine_b":
            scanned = add("scan", source_id="link_b")
            joined = add("join", parents[0], scanned,
                         left_on="route_key", right_on="route_key")
            projected = add("project", joined, fields=["entity_0"])
            result = add("unique", projected)
        elif kind == "crosscheck":
            looked_up = add(
                "lookup_by_key", parents[0], source_id="real_fork",
                input_key_field="entity_0",
                source_key_field="entity_0",
            )
            merged = add("join", looked_up, parents[1],
                         left_on="motif_object", right_on="motif_object")
            projected = add("project", merged, fields=["entity_0"])
            result = add("unique", projected)
        elif kind == "answer":
            found = parents[0]
            for sid in task["source_scope"]:
                step = int(sid.rsplit("_", 1)[-1])
                found = add(
                    "lookup_by_key", found, source_id=sid,
                    input_key_field=f"entity_{step-1}",
                    source_key_field=f"entity_{step-1}",
                )
            projected = add("project", found, fields=["answer"])
            result = add("unique", projected)
        else:
            raise ValueError("task_graph_first_unsupported_role:" + kind)
        program_nodes.extend(local)
        groups.append({
            "macro_id": mid, "node_ids": [r["id"] for r in local],
            "output_node_id": result, "external_input_nodes": parents,
        })
        outputs[mid] = result
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": PATTERN,
        "nodes": program_nodes,
        "output": outputs[plan["terminal_macro_id"]],
        "authorized_source_ids": [
            row["source_id"] for row in spec["sources"]
        ],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups,
        "macro_plan_version": VERSION,
    }


def generate_world(spec: dict) -> dict:
    issues = validate_spec(spec)
    if issues:
        raise ValueError("invalid_task_graph_first_world:" + ",".join(issues))
    from .evidence_planner import materialize_world
    return materialize_world(spec)


def mutate_witness(sources, source_id, answer_rows, spec):
    """Preserve old per-source sensitivity checks for all scoped evidence."""
    from .hybrid_world_v4 import mutate_witness
    return mutate_witness(sources, source_id, answer_rows, spec)
