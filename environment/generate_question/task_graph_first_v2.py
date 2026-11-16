"""Versioned task-graph-first v2: compositional relation goals + causal ports.

The abstract Macro DAG and local relation goal are sampled before selecting
YAGO facts; the already validated v1 YAGO/fictional World generator is only
a *witness/grounding provider*, not a source of the new task requirement.

Historical task_graph_first_v1, and typed_knowledge v1-v4 frozen scenes and
their deterministic Replay, retain their original code paths.
"""
from __future__ import annotations

import copy

from environment.schema import digest
from . import task_graph_first as old
from .macro_graph_sampler import sample_graph
from .relational_task_planner import (
    plan_relations_v2, validate_relations_v2,
)
from .evidence_planner import (
    design_evidence_v2, validate_evidence_v2,
    materialize_world_v2,
)
from . import knowledge_macro as knowledge

PATTERN = "task_graph_first_v2"
VERSION = "maestro.macro_plan.task_graph_first.v2"
WITNESS_VERSION = "maestro.knowledge_witness.task_graph_first.v2"
HASH_SALT = "canonical-macro-first-task-local-relations-v2"
NECESSITY_POLICY = "task_graph_first_port_and_relation_v2"


def make_spec(kg, *, seed: int, ordinal: int, difficulty: str,
              min_hops: int = 2, max_hops: int = 3,
              index: knowledge.YagoKnowledgeIndex | None = None) -> dict:
    # 1. Freeze source-free, entity-free Macro DAG.
    graph = sample_graph(
        seed=seed, ordinal=ordinal, difficulty=difficulty,
    )
    # 2. Pick which local event relation the task needs while still
    #    independent of concrete YAGO/synthetic entities.
    relation = plan_relations_v2(
        graph, seed=seed, ordinal=ordinal,
        min_hops=min_hops, max_hops=max_hops,
    )
    # 3. Ground into genuine YAGO witnesses + fictional entities. This
    #    reuses the validated legacy fact constructor, but the v2 Macro
    #    DAG/relationship constraints were chosen BEFORE it was called.
    backbone = old.make_spec(
        kg, seed=seed, ordinal=ordinal, difficulty=difficulty,
        min_hops=min_hops, max_hops=max_hops, index=index,
    )
    if backbone["task_first_macro_graph"]["graph_hash"] != graph["graph_hash"]:
        raise ValueError("v2_grounder_changed_preplanned_macro_graph")
    # 4. Freeze alternative, event-specific simulated observations.
    evidence = design_evidence_v2(graph, relation, backbone)
    spec = copy.deepcopy(backbone)
    spec["task_first_v1_backbone"] = copy.deepcopy(backbone)
    spec["task_first_v2_relation_sketch"] = relation
    spec["task_first_v2_evidence"] = evidence
    plan = spec["macro_plan"]
    plan.update({
        "version": VERSION,
        "kind": "graph_first_composed_event_relations",
        "information_necessity_policy": NECESSITY_POLICY,
        "task_local_event_relation": copy.deepcopy(
            relation["event_relation"]
        ),
        "event_id_port_necessity": evidence["causal_port"]["required"],
        "relation_sketch_v2_hash": relation["sketch_hash"],
        "evidence_v2_hash": evidence["evidence_hash"],
    })
    spec.update({
        "reasoning_pattern": PATTERN, "family": PATTERN,
        "task_id": (
            f"question-gf2-{ordinal:06d}-"
            + digest(
                seed, ordinal, relation["event_goal"], graph["shape"]
            )[:8]
        ),
        "scene_id": f"task-graph-first-v2-{ordinal:06d}",
        "macro_information_necessity_version": NECESSITY_POLICY,
    })
    spec["scenario"]["purpose"] = (
        "identify a fictional project, select an eligible event based "
        "on the specified status or year criterion, reconcile different "
        "event-specific candidate sources and follow genuine YAGO "
        "relationships to the final entity"
    )
    spec["scenario"]["motivation"] = (
        "the abstract Macro task graph and local relation requirement "
        "were fixed before grounding actual YAGO relations and simulated "
        "project/event records"
    )
    spec["knowledge_witness"]["version"] = WITNESS_VERSION
    spec["knowledge_witness"]["task_graph_first_v2"] = {
        "base_v1_task": backbone["task_id"],
        "macro_graph_hash": graph["graph_hash"],
        "composed_relation_hash": relation["sketch_hash"],
        "event_specific_evidence_hash": evidence["evidence_hash"],
    }
    spec["knowledge_contract_hash"] = digest(
        plan, backbone, relation, evidence, spec["knowledge_witness"],
        seed, ordinal, HASH_SALT,
    )
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_graph_first_v2_spec:" + ",".join(problems))
    return spec


def validate_spec(spec: dict) -> list[str]:
    try:
        plan = spec["macro_plan"]
        base = spec["task_first_v1_backbone"]
        relation = spec["task_first_v2_relation_sketch"]
        evidence = spec["task_first_v2_evidence"]
        if (spec["reasoning_pattern"] != PATTERN
                or spec["family"] != PATTERN
                or plan["version"] != VERSION
                or spec["knowledge_witness"]["version"] != WITNESS_VERSION
                or plan["information_necessity_policy"] !=
                NECESSITY_POLICY):
            return ["graph_first_v2_version_mismatch"]
        if spec["knowledge_contract_hash"] != digest(
            plan, base, relation, evidence, spec["knowledge_witness"],
            spec["seed"], spec["ordinal"], HASH_SALT,
        ):
            return ["graph_first_v2_frozen_hash_mismatch"]
        problems = old.validate_spec(base)
        problems += validate_relations_v2(
            relation, base["task_first_macro_graph"]
        )
        problems += validate_evidence_v2(evidence, relation, base)
        if problems:
            return ["graph_first_v2_invalid_backbone_or_local_sketch:" +
                    ",".join(problems)]
        expected = copy.deepcopy(base["macro_plan"])
        expected.update({
            "version": VERSION,
            "kind": "graph_first_composed_event_relations",
            "information_necessity_policy": NECESSITY_POLICY,
            "task_local_event_relation": copy.deepcopy(
                relation["event_relation"]
            ),
            "event_id_port_necessity": evidence["causal_port"]["required"],
            "relation_sketch_v2_hash": relation["sketch_hash"],
            "evidence_v2_hash": evidence["evidence_hash"],
        })
        if plan != expected:
            return ["graph_first_v2_macro_plan_not_from_frozen_graph"]
        if any(spec[k] != base[k] for k in (
            "task_first_macro_graph", "task_first_relation_sketch",
            "task_first_grounding", "task_first_evidence", "sources",
        )):
            return ["graph_first_v2_changed_grounded_source_or_graph"]
        if spec["knowledge_witness"].get("task_graph_first_v2") != {
            "base_v1_task": base["task_id"],
            "macro_graph_hash": base["task_first_macro_graph"]["graph_hash"],
            "composed_relation_hash": relation["sketch_hash"],
            "event_specific_evidence_hash": evidence["evidence_hash"],
        }:
            return ["graph_first_v2_wrong_knowledge_witness"]
        if evidence["event_selection"] != plan["task_local_event_relation"]:
            return ["graph_first_v2_event_filter_not_from_relation"]
        return []
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return ["invalid_graph_first_v2:" + str(exc)[:150]]


def compile_macro_solvers(spec: dict) -> dict:
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_graph_first_v2_program:" +
                         ",".join(problems))
    base = spec["task_first_v1_backbone"]
    program = old.compile_macro_solvers(base)
    chosen = spec["task_first_v2_relation_sketch"]["event_relation"]
    selectors = [
        node for node in program["nodes"]
        if node["op"] == "filter_eq"
        and node.get("field") == "event_status"
    ]
    if len(selectors) != 1:
        raise ValueError("graph_first_v2_no_unique_event_field_filter")
    selectors[0].update({
        "op": chosen["operator"],
        "field": chosen["field"],
        "value": chosen["value"],
    })
    program["semantic_operator_pattern"] = PATTERN
    program["macro_plan_version"] = VERSION
    program["frozen_problem_spec_id"] = spec["task_id"]
    return program


def generate_world(spec: dict) -> dict:
    problems = validate_spec(spec)
    if problems:
        raise ValueError("invalid_graph_first_v2_world:" +
                         ",".join(problems))
    return materialize_world_v2(spec)


def mutate_witness(sources: dict, source_id: str,
                   gold_rows: list[dict], spec: dict) -> dict:
    # Legacy source and handoff counterfactuals retain one-observation
    # provenance sensitivity; the stronger alternative-plan check is
    # independently run by audit_information_v2.
    return old.mutate_witness(
        sources, source_id, gold_rows,
        spec["task_first_v1_backbone"],
    )
