"""Information-relationship sketch between Macro DAG and grounded World.

This module is deliberately *world-free*. It turns information-work nodes
into typed relation requirements, causal evidence obligations and source
contracts, but it never chooses entities, triples, names or concrete data.
"""
from __future__ import annotations

from environment.schema import digest
from .macro_graph_sampler import validate_graph, VERSION as GRAPH_VERSION

VERSION = "maestro.relational_task_sketch.v1"

RELATION_REQUIREMENTS = (
    {
        "relation": "fictional_project_has_event",
        "domain": "FictionalProject", "range": "FictionalEvent",
        "origin": "synthetic", "causal_role": "event_discovery",
    },
    {
        "relation": "event_has_organizer_and_venue",
        "domain": "FictionalEvent", "range": "FictionalContext",
        "origin": "synthetic", "causal_role": "context_only",
    },
    {
        "relation": "event_status",
        "domain": "FictionalEvent", "range": "EventStatus",
        "origin": "synthetic", "causal_role": "required_completed_event_filter",
    },
    {
        "relation": "fictional_project_candidate_yago_reference",
        "domain": "FictionalProject", "range": "RealEntityRef",
        "origin": "synthetic_claim", "causal_role": "two_source_competing_candidates",
    },
    {
        "relation": "real_secondary_yago_fork",
        "domain": "RealEntityRef", "range": "RealMotifObject",
        "origin": "yago", "causal_role": "required_motif_crosscheck",
    },
    {
        "relation": "real_yago_multihop_answer",
        "domain": "RealEntityRef", "range": "RealEntityRef",
        "origin": "yago", "causal_role": "terminal_goal",
    },
)


def plan_relations(graph: dict, *, min_hops: int = 2,
                   max_hops: int = 3) -> dict:
    validate_graph(graph)
    if not 2 <= min_hops <= max_hops <= 4:
        raise ValueError("graph_first_relation_depth_invalid")
    shape = graph["shape"]
    requirements = [dict(item) for item in RELATION_REQUIREMENTS]
    sketch = {
        "version": VERSION,
        "graph_version": GRAPH_VERSION,
        "graph_hash": graph["graph_hash"],
        "requirements": requirements,
        "knowledge": {
            "origin": "yago",
            "main_chain_hops": [min_hops, max_hops],
            "secondary_fork_for_all_candidates": True,
            "same_real_fork_predicate_for_competitors": True,
            "complete_distinct_real_final_paths": True,
        },
        "synthetic": {
            "entity_kinds": [
                "fictional_project", "fictional_event",
                "fictional_organizer", "fictional_venue",
            ],
            "required_event_status": "completed",
            "disambiguating_sources": ["link_a", "link_b", "event_ledger"],
            "other_event_status": "scheduled",
            "context_relations_not_claimed_causal": [
                "event_has_organizer_and_venue",
            ],
        },
        "evidence": {
            "clues_per_discovery": 2,
            "unverified_candidate_directories": 2,
            "event_record_status_must_be_used": True,
            "real_fork_is_required_source": True,
            "source_selection_mode": (
                "event_id_keyed" if shape == "event_gated_association"
                else "project_id_keyed"
            ),
            "source_provenance_separated": True,
        },
        "terminal": {
            "answer_field": "answer", "cardinality": "exactly_one",
            "output_type": "str",
        },
        "sampling_policy": "abstract_graph_before_relations_before_world_v1",
    }
    sketch["sketch_hash"] = digest(sketch, "relation-sketch-frozen")
    return sketch


def validate_sketch(sketch: dict, graph: dict) -> list[str]:
    try:
        validate_graph(graph)
        raw = {key: value for key, value in sketch.items()
               if key != "sketch_hash"}
        if (sketch["version"] != VERSION
                or sketch["graph_hash"] != graph["graph_hash"]
                or sketch["graph_version"] != GRAPH_VERSION
                or sketch["sketch_hash"] != digest(
                    raw, "relation-sketch-frozen"
                )):
            return ["relation_sketch_hash_or_graph_mismatch"]
        known = {
            item["relation"]: item["origin"] for item in RELATION_REQUIREMENTS
        }
        if {item["relation"]: item["origin"] for item in
                sketch["requirements"]} != known:
            return ["relation_sketch_changed_requirements"]
        if sketch["synthetic"]["required_event_status"] != "completed":
            return ["relation_sketch_wrong_event_filter"]
        expected = ("event_id_keyed" if graph["shape"] ==
                    "event_gated_association" else "project_id_keyed")
        if sketch["evidence"]["source_selection_mode"] != expected:
            return ["relation_sketch_invalid_source_handoff"]
        if not sketch["knowledge"]["secondary_fork_for_all_candidates"]:
            return ["relation_sketch_missing_real_fork"]
        return []
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return ["invalid_relation_sketch:" + str(exc)[:140]]


# Versioned extension: role-local relationship composition, sampled BEFORE
# World grounding. The v1 relation sketch and frozen replay are unmodified.
COMPOSITION_VERSION = "maestro.relational_task_sketch.v2"
EVENT_PREDICATES = ("status_completed", "year_at_least_2025")


def plan_relations_v2(graph: dict, *, seed: int, ordinal: int,
                      min_hops: int = 2, max_hops: int = 3) -> dict:
    """Select a typed *local* information relation, not a fixed world fact.

    Two different event goals use different source fields and executable
    comparisons. The graph is frozen before this operation and the
    chosen event relation is frozen before any YAGO grounding.
    """
    from .macro_graph_sampler import validate_graph
    validate_graph(graph)
    predicate = EVENT_PREDICATES[(seed + ordinal) % len(EVENT_PREDICATES)]
    relation = ({
        "relation": "fictional_event_status",
        "field": "event_status", "operator": "filter_eq",
        "value": "completed", "input_type": "EventRecord",
        "output_type": "EligibleEventSet",
    } if predicate == "status_completed" else {
        "relation": "fictional_event_year",
        "field": "event_year", "operator": "filter_gte",
        "value": 2025, "input_type": "EventRecord",
        "output_type": "EligibleEventSet",
    })
    base = plan_relations(
        graph, min_hops=min_hops, max_hops=max_hops,
    )
    # v1 is retained solely as a compatibility bridge to the existing
    # grounded RDF witness sampler; v2's task-local requirements own
    # the materialized event evidence and compiled predicate.
    sketch = {
        "version": COMPOSITION_VERSION,
        "graph_hash": graph["graph_hash"],
        "legacy_grounding_sketch": base,
        "event_goal": predicate,
        "event_relation": relation,
        "task_local_relations": {
            "event": relation,
            "directory_a": {
                "relation": "references_for_event" if graph["shape"] ==
                    "event_gated_association" else "references_for_project",
                "key_type": "EventId" if graph["shape"] ==
                    "event_gated_association" else "ProjectId",
                "evidence_kind": "unverified_simulated_claims",
            },
            "directory_b": {
                "relation": "references_for_event" if graph["shape"] ==
                    "event_gated_association" else "references_for_project",
                "key_type": "EventId" if graph["shape"] ==
                    "event_gated_association" else "ProjectId",
                "evidence_kind": "unverified_simulated_claims",
            },
            "crosscheck": {
                "relation": "real_yago_secondary_fork",
                "subject_type": "RealYagoEntityRef",
                "object_type": "RealYagoMotifObject",
            },
            "answer": {
                "relation": "real_yago_chain_answer",
                "answer_type": "RealYagoEntityLabel",
            },
        },
        "provenance_contract": {
            "real": "frozen_yago_triples",
            "synthetic": "fictional_event_and_project_truth",
            "claims": "unverified_source_hypotheses",
        },
        "sampling_order": (
            "abstract_macro_graph_then_task_local_relation_sketch_"
            "then_yago_and_fictional_world"
        ),
    }
    sketch["sketch_hash"] = digest(sketch, "relation-composition-v2")
    return sketch


def validate_relations_v2(sketch: dict, graph: dict) -> list[str]:
    try:
        validate_graph(graph)
        raw = {k: v for k, v in sketch.items() if k != "sketch_hash"}
        if (sketch["version"] != COMPOSITION_VERSION
                or sketch["graph_hash"] != graph["graph_hash"]
                or sketch["sketch_hash"] != digest(
                    raw, "relation-composition-v2"
                )):
            return ["invalid_composed_relation_sketch_hash"]
        old = validate_sketch(sketch["legacy_grounding_sketch"], graph)
        if old:
            return old
        expected = plan_relations_v2(
            graph, seed=0, ordinal=(
                0 if sketch["event_goal"] == EVENT_PREDICATES[0] else 1
            ), min_hops=sketch["legacy_grounding_sketch"][
                "knowledge"]["main_chain_hops"][0],
            max_hops=sketch["legacy_grounding_sketch"][
                "knowledge"]["main_chain_hops"][1],
        )
        if (sketch["event_goal"] not in EVENT_PREDICATES
                or sketch["event_relation"] != expected["event_relation"]
                or sketch["task_local_relations"] !=
                expected["task_local_relations"]):
            return ["invalid_composed_relation_task_ports"]
        return []
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return ["invalid_relation_composition:" + str(exc)[:120]]
