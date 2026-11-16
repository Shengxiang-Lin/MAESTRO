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
