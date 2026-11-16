"""Evidence-source planner: relation requirements -> observable source records.

The planner does not decide the Macro DAG or real RDF paths. It designs
typed fictional source records, necessary event-status discrimination,
provenance and semantic source scopes AFTER World grounding.
"""
from __future__ import annotations

import copy
from collections import Counter

from environment.schema import digest
from . import hybrid_world_v4 as v4
from .world_grounder import validate_grounding
from .relational_task_planner import validate_sketch
from .macro_graph_sampler import validate_graph

VERSION = "maestro.task_first_evidence_plan.v1"
STATUS = "completed"


def _secondary_event(seed: int, ordinal: int, case: str,
                     original: dict, fork_object: str) -> dict:
    return {
        **original,
        "event_id": "urn:maestro:fictional:scheduled-event:" + digest(
            seed, ordinal, case, "additional-event-v1"
        )[:20],
        "event_name": original["event_name"] + " (scheduled follow-up)",
        "event_status": "scheduled",
        "event_year": original["event_year"] + 1,
        "motif_object": fork_object,
    }


def design_evidence(graph: dict, sketch: dict, grounded: dict,
                    *, seed: int, ordinal: int) -> dict:
    validate_graph(graph)
    if validate_sketch(sketch, graph):
        raise ValueError("invalid_relation_sketch_for_evidence")
    if validate_grounding(grounded, graph, sketch):
        raise ValueError("invalid_world_grounding_for_evidence")
    base = grounded["base_v4"]
    plan = base["macro_plan"]
    cases = list(plan["all_cases"])
    event_rows = plan["synthetic_graph"]["event_rows"]
    main = {}
    extra = {}
    for case in cases:
        project = plan["synthetic_entities"][case]["entity_id"]
        candidates = [r for r in event_rows if r["project_id"] == project]
        if len(candidates) != 3:
            raise ValueError("unexpected_world_event_eligibility_cardinality")
        main[case] = [
            {**copy.deepcopy(row), "event_status": STATUS}
            for row in candidates
        ]
        i = cases.index(case)
        u = cases[(i+1) % len(cases)]
        alternate_fork = plan["real_fork"]["candidate_forks"][u]["object"]
        extra[case] = _secondary_event(
            seed, ordinal, case, candidates[0], alternate_fork
        )
    required = {
        "cue_bundle": ["cue"],
        "project_registry": ["case_id", "project_id"],
        "event_records": ["project_id", "event_status",
                          "motif_object", "event_id"],
        "simulated_directory_a": [
            "event_id" if graph["shape"] == "event_gated_association"
            else "project_id", "route_key", "entity_0"
        ],
        "simulated_directory_b": [
            "event_id" if graph["shape"] == "event_gated_association"
            else "project_id", "route_key", "entity_0"
        ],
        "real_secondary_fork": ["entity_0", "motif_object"],
        "real_relation_chain": ["entity_0", "answer"],
    }
    evidence = {
        "version": VERSION,
        "graph_hash": graph["graph_hash"],
        "relation_sketch_hash": sketch["sketch_hash"],
        "grounding_hash": grounded["grounding_hash"],
        "main_event_records": main,
        "scheduled_event_records": extra,
        "required_fields_by_source_role": required,
        "context_fields": {
            "event_records": [
                "organizer_id", "organizer_name", "venue_id",
                "venue_name", "event_year", "event_name", "project_name",
            ],
            "project_registry": ["project_name"],
        },
        "source_evidence_policy": {
            "truth_vs_claims": "frozen_synthetic_truth_and_unverified_claims",
            "real_rdf": "frozen_yago_witness",
            "event_filter": "completed",
            "event_id_gates_directory": graph["shape"] ==
                "event_gated_association",
            "minimum_valid_alternative_answers_when_filter_removed": 2,
        },
        "causal_claim_scope": (
            "Only observed required fields and selected task handoffs "
            "count toward information-complexity metrics. Other fields "
            "remain explicitly context."
        ),
    }
    evidence["evidence_hash"] = digest(evidence, "task_first_evidence_frozen")
    return evidence


def validate_evidence(evidence: dict, graph: dict, sketch: dict,
                      grounded: dict) -> list[str]:
    try:
        issues = validate_grounding(grounded, graph, sketch)
        if issues:
            return issues
        raw = {k: v for k, v in evidence.items() if k != "evidence_hash"}
        if (evidence["version"] != VERSION
                or evidence["evidence_hash"] != digest(
                    raw, "task_first_evidence_frozen")
                or evidence["graph_hash"] != graph["graph_hash"]
                or evidence["grounding_hash"] != grounded["grounding_hash"]
                or evidence["relation_sketch_hash"] !=
                sketch["sketch_hash"]):
            return ["evidence_contract_hash_mismatch"]
        cases = grounded["base_v4"]["macro_plan"]["all_cases"]
        if (set(evidence["main_event_records"]) != set(cases)
                or set(evidence["scheduled_event_records"]) != set(cases)):
            return ["evidence_missing_per_case_event"]
        real = grounded["base_v4"]["macro_plan"]["real_fork"][
            "candidate_forks"]
        for i, case in enumerate(cases):
            good = evidence["main_event_records"][case]
            decoy = evidence["scheduled_event_records"][case]
            if len(good) != 3 or any(
                    row["event_status"] != STATUS for row in good
            ):
                return ["evidence_required_event_status_changed"]
            u = cases[(i+1) % len(cases)]
            if (decoy["event_status"] != "scheduled"
                    or decoy["motif_object"] != real[u]["object"]
                    or decoy["event_id"] == good[0]["event_id"]):
                return ["evidence_scheduled_event_not_a_real_competitor"]
        if evidence["source_evidence_policy"]["event_id_gates_directory"] != (
                graph["shape"] == "event_gated_association"):
            return ["evidence_false_gated_source"]
        if evidence["source_evidence_policy"]["event_filter"] != STATUS:
            return ["evidence_missing_compiled_field_gate"]
        return []
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        return ["invalid_evidence_contract:" + str(exc)[:140]]


def materialize_world(spec: dict) -> dict:
    """Regenerate source observations deterministically from frozen spec.

    No knowledge-graph access happens on replay. Concrete source records
    use the existing _rows/FactRefs and a legacy source schema.
    """
    from .world import _rows
    evidence = spec["task_first_evidence"]
    grounded = spec["task_first_grounding"]
    graph = spec["task_first_macro_graph"]
    sketch = spec["task_first_relation_sketch"]
    issues = validate_evidence(evidence, graph, sketch, grounded)
    if issues:
        raise ValueError("invalid_task_first_evidence:" + ",".join(issues))
    base = grounded["base_v4"]
    original = v4.generate_world(base)
    bank = original["sources"]
    items = evidence["main_event_records"]
    other = evidence["scheduled_event_records"]
    cases = base["macro_plan"]["all_cases"]
    records = []
    for case in cases:
        records.extend(copy.deepcopy(items[case]))
        records.append(copy.deepcopy(other[case]))
    meta = next(row for row in spec["sources"]
                if row["source_id"] == "event_ledger")
    src = _rows("event_ledger", meta["modality"], meta["title"],
                meta["purpose"], records)
    src["access_policy"] = {"mode": "browse"}
    src["evidence_kind"] = "fictional_completed_and_scheduled_event_records"
    for obs in src["observations"]:
        obs["evidence_origin_id"] = "synthetic:" + digest(
            obs["values"]["event_id"], obs["values"]["motif_object"],
        )[:20]
        obs["claim_status"] = "fictional_event_fact"
    bank["event_ledger"] = src

    if evidence["source_evidence_policy"]["event_id_gates_directory"]:
        for sid in ("link_a", "link_b"):
            values = []
            for row in bank[sid]["observations"]:
                record = row["values"]
                for case in cases:
                    pid = base["macro_plan"]["synthetic_entities"][
                        case]["entity_id"]
                    if pid != record["project_id"]:
                        continue
                    for eid in (items[case][0]["event_id"],
                                other[case]["event_id"]):
                        values.append({**copy.deepcopy(record), "event_id": eid})
                    break
            source_meta = next(s for s in spec["sources"]
                               if s["source_id"] == sid)
            bank[sid] = _rows(
                sid, source_meta["modality"], source_meta["title"],
                source_meta["purpose"], values,
            )
            bank[sid]["access_policy"] = {"mode": "browse"}
            bank[sid]["evidence_kind"] = "unverified_event_keyed_reference_claims"
            for obs in bank[sid]["observations"]:
                obs["claim_status"] = "unverified_candidate"
                obs["evidence_origin_id"] = "simulated:" + digest(
                    sid, obs["values"]["route_key"],
                    obs["values"]["event_id"],
                )[:20]
    synthetic = original["synthetic_world"]
    scheduled_entities = {
        case: {
            "entity_id": other[case]["event_id"],
            "display_name": other[case]["event_name"],
            "entity_type": "fictional_event", "status": "scheduled",
            "origin": "synthetic", "fictional": True,
        } for case in cases
    }
    scheduled_truth = []
    for case in cases:
        pid = base["macro_plan"]["synthetic_entities"][case]["entity_id"]
        scheduled_truth.extend([
            {
                "subject": pid,
                "predicate": "urn:maestro:synthetic:hasEvent",
                "object": other[case]["event_id"],
                "origin": "synthetic", "truth_scope": "fictional_event_world",
                "record_type": "synthetic_world_fact",
            },
            {
                "subject": other[case]["event_id"],
                "predicate": "urn:maestro:synthetic:permitsYagoForkObject",
                "object": other[case]["motif_object"],
                "origin": "synthetic", "truth_scope": "fictional_event_world",
                "record_type": "synthetic_world_fact",
            },
        ])
    synthetic["entities"]["scheduled_events"] = scheduled_entities
    for case in cases:
        synthetic["entities"]["events"][case]["status"] = STATUS
    synthetic["true_relations"].extend(scheduled_truth)
    synthetic["version"] = "maestro.synthetic_event_graph.task_first_v1"
    original["synthetic_world"] = synthetic
    original["task_first_evidence_contract"] = {
        "version": VERSION,
        "required_fields_by_source_role":
            copy.deepcopy(evidence["required_fields_by_source_role"]),
        "context_fields": copy.deepcopy(evidence["context_fields"]),
    }
    original["value_sampling_contract"] = (
        "macro_first_world_grounding_and_evidence_dependency_v1"
    )
    original["knowledge_witness"] = copy.deepcopy(
        spec["knowledge_witness"]
    )
    return original
