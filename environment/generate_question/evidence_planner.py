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


def audit_information_necessity(scene: dict) -> dict:
    """Bounded executable necessity audit over a task-FIRST evidence graph.

    Test removal of any clue, whole Discovery, either directory constraint,
    real RDF fork sensitivity, and the REQUIRED completed-event status
    field. Counterfactuals must yield >1 distinct *executable* real
    answers, never merely a blocked/empty program.
    """
    from .program_ir import evaluate_program, answer_values

    spec, program, sources = (
        scene["problem_spec"], scene["gold_program"],
        scene["world"]["sources"],
    )
    plan = spec["macro_plan"]
    evidence = spec["task_first_evidence"]
    target = plan["target_case"]
    answer = scene["gold_answer"][0]["answer"]
    parent = next(row for row in plan["macro_tasks"]
                  if row["information_operation"] == "identify")
    actual = next(row for row in scene["task_feedback"]
                  if row["task_id"] == "work_" + parent["macro_id"])
    unique_identity = (
        actual["result_rows"] == 1
        and actual["workproduct"][0]["values"]["project_id"] ==
        plan["synthetic_entities"][target]["entity_id"]
    )
    by_id = {node["id"]: node for node in program["nodes"]}
    cue_ids, status_id = {}, None
    for node in program["nodes"]:
        if node["op"] != "filter_eq":
            continue
        if node.get("field") == "cue":
            upstream = by_id[node["inputs"][0]]
            cue_ids[upstream["source_id"]] = node["id"]
        elif node.get("field") == "event_status":
            if status_id is not None:
                raise ValueError("multiple_event_status_filters")
            status_id = node["id"]
    if set(cue_ids) != set(plan["clues"]) or status_id is None:
        raise ValueError("task_graph_first_missing_required_primitive_filters")
    def alternative(rows):
        distinct = {record["answer"] for record in rows}
        return len(distinct) >= 2 and answer in distinct

    def execute(*, changed_sources=None, loosen_cues=(), drop_status=False):
        derived = copy.deepcopy(program)
        loose = {cue_ids[sid] for sid in loosen_cues}
        for node in derived["nodes"]:
            if node["id"] in loose:
                oid, parents = node["id"], list(node["inputs"])
                node.clear()
                node.update({
                    "id": oid, "op": "project",
                    "inputs": parents, "fields": ["case_id", "cue"],
                })
            elif drop_status and node["id"] == status_id:
                oid, parents = node["id"], list(node["inputs"])
                node.clear()
                node.update({
                    "id": oid, "op": "project",
                    "inputs": parents,
                    "fields": list(sources["event_ledger"]["schema"]),
                })
        return answer_values(evaluate_program(
            derived, sources=changed_sources
                if changed_sources is not None else sources,
        ))

    clue_results = {
        sid: alternative(execute(loosen_cues=(sid,)))
        for sid in cue_ids
    }
    discovery_results = {
        node["macro_id"]: alternative(execute(
            loosen_cues=node["source_scope"]
        ))
        for node in plan["macro_tasks"]
        if node["role"] == "discovery"
    }
    fully_relaxed = execute(loosen_cues=cue_ids)
    status_without_filter = execute(drop_status=True)

    cases = plan["all_cases"]
    target_index = cases.index(target)
    project = plan["synthetic_entities"][target]["entity_id"]
    anchors = {
        case: plan["candidate_paths"][case][0]["subject"]
        for case in cases
    }
    main_event = evidence["main_event_records"][target][0]["event_id"]
    broken_directory_checks = {}
    for sid, shift in (("link_a", 3), ("link_b", 2)):
        changed = copy.deepcopy(sources)
        real_uri = anchors[cases[(target_index+shift) % len(cases)]]
        from .hybrid_world import _route_key
        new_candidate = {
            "project_id": project, "entity_0": real_uri,
            "route_key": _route_key(project, real_uri),
        }
        if evidence["source_evidence_policy"]["event_id_gates_directory"]:
            new_candidate["event_id"] = main_event
        changed[sid]["observations"].append({
            "observation_id": sid + ".RELAXED",
            "fact_refs": [sid + ":F-RELAXED"],
            "values": new_candidate,
            "claim_status": "hypothetical_extra_candidate",
        })
        broken_directory_checks[sid] = alternative(
            execute(changed_sources=changed)
        )

    u = cases[(target_index+1) % len(cases)]
    u_uri = anchors[u]
    fork = plan["real_fork"]["candidate_forks"]
    changed_fork = copy.deepcopy(sources)
    entry = next(
        row for row in changed_fork["real_fork"]["observations"]
        if row["values"]["entity_0"] == u_uri
    )
    # Hypothetical falsification of a source observation; NOT real RDF.
    entry["values"]["motif_object"] = fork[target]["object"]
    fork_result = alternative(execute(changed_sources=changed_fork))
    # The field, not just its containing source, must be necessary:
    # relaxing "completed" now admits an entirely different scheduled
    # event and its genuine complete downstream YAGO answer path.
    field_causality = {
        "event_status": alternative(status_without_filter),
        "real_fork_motif_object": fork_result,
    }
    # Context fields are NOT credited as additional required complexity.
    context_fields = evidence["context_fields"]
    gates = [
        unique_identity, all(clue_results.values()),
        all(discovery_results.values()), alternative(fully_relaxed),
        all(broken_directory_checks.values()),
        all(field_causality.values()),
    ]
    return {
        "policy": "task_graph_first_replanning_v1",
        "passed": all(gates),
        "identification_unique": unique_identity,
        "each_clue_necessary": clue_results,
        "each_discovery_necessary": discovery_results,
        "all_discovery_bypass_ambiguous": alternative(fully_relaxed),
        "simulated_directory_necessity": broken_directory_checks,
        "required_field_necessity": field_causality,
        "event_id_gates_source_access": (
            evidence["source_evidence_policy"]["event_id_gates_directory"]
        ),
        "context_fields_not_credited": context_fields,
        "macro_shape": spec["task_first_macro_graph"]["shape"],
        "required_real_fork": True,
        "limitation": (
            "Bounded counterfactual re-planning within the frozen World; "
            "not proof of global solution minimality or factual claims "
            "about simulated historical media."
        ),
    }


# --- Graph-first v2: field-selective evidence and causal event-key handoffs ---
EVIDENCE_V2_VERSION = "maestro.task_first_evidence_plan.v2"


def design_evidence_v2(graph: dict, relation_v2: dict, base: dict) -> dict:
    """Freeze event-specific source claims after graph and sketch selection.

    For an event-gated Macro, completed-event directories have the
    original sets A={T,U,V}, B={T,U,W}; the scheduled event adds W only
    to directory A and V only to directory B. Collapsing the event_id
    handoff to project_id therefore reopens *real* alternative YAGO
    answers. The original v1 source rows were identical by event_id.
    """
    from .relational_task_planner import validate_relations_v2
    from .hybrid_world import _route_key
    problems = validate_relations_v2(relation_v2, graph)
    if problems:
        raise ValueError("invalid_event_relation_for_evidence:" +
                         ",".join(problems))
    cases = base["macro_plan"]["all_cases"]
    paths = base["macro_plan"]["candidate_paths"]
    projects = base["macro_plan"]["synthetic_entities"]
    extras = {"link_a": {}, "link_b": {}}
    event_records = base["task_first_evidence"]["scheduled_event_records"]
    for i, case in enumerate(cases):
        pid = projects[case]["entity_id"]
        eid = event_records[case]["event_id"]
        for sid, shift in (("link_a", 3), ("link_b", 2)):
            uri = paths[cases[(i+shift) % len(cases)]][0]["subject"]
            extras[sid][case] = {
                "project_id": pid, "event_id": eid, "entity_0": uri,
                "route_key": _route_key(pid, uri),
            }
    evidence = {
        "version": EVIDENCE_V2_VERSION,
        "graph_hash": graph["graph_hash"],
        "relation_sketch_hash": relation_v2["sketch_hash"],
        "frozen_v1_evidence_hash": base["task_first_evidence"][
            "evidence_hash"],
        "event_selection": relation_v2["event_relation"],
        "event_facts_policy": {
            "status_completed": {
                "selected": {"event_status": "completed", "event_year": 2025},
                "other": {"event_status": "scheduled", "event_year": 2025},
            },
            "year_at_least_2025": {
                "selected": {"event_status": "recorded", "event_year": 2025},
                "other": {"event_status": "recorded", "event_year": 2024},
            },
        }[relation_v2["event_goal"]],
        "scheduled_only_source_claims": extras,
        "causal_port": {
            "node_key": "event", "field": "event_id",
            "downstream": ["directory_a", "directory_b"],
            "required": graph["shape"] == "event_gated_association",
            "bypass_plan": "relink_both_directory_inputs_by_project_id",
        },
        "context_not_credited": [
            "event_name", "project_name", "organizer_name",
            "venue_name", "organizer_id", "venue_id",
        ],
        "minimum_alternative_answers": 2,
    }
    evidence["evidence_hash"] = digest(
        evidence, "task-first-evidence-v2"
    )
    return evidence


def validate_evidence_v2(evidence: dict, relation_v2: dict,
                         base: dict) -> list[str]:
    try:
        from .relational_task_planner import validate_relations_v2
        graph = base["task_first_macro_graph"]
        issues = validate_relations_v2(relation_v2, graph)
        if issues:
            return issues
        data = {k: v for k, v in evidence.items()
                if k != "evidence_hash"}
        if (evidence["version"] != EVIDENCE_V2_VERSION
                or evidence["evidence_hash"] != digest(
                    data, "task-first-evidence-v2"
                ) or evidence["graph_hash"] != graph["graph_hash"]
                or evidence["relation_sketch_hash"] !=
                relation_v2["sketch_hash"]
                or evidence["frozen_v1_evidence_hash"] !=
                base["task_first_evidence"]["evidence_hash"]):
            return ["task_first_v2_evidence_hash_mismatch"]
        if evidence["event_selection"] != relation_v2["event_relation"]:
            return ["task_first_v2_event_selection_not_from_sketch"]
        goal = relation_v2["event_goal"]
        policy = evidence["event_facts_policy"]
        if goal == "status_completed":
            if (policy["selected"] != {
                    "event_status": "completed", "event_year": 2025
            } or policy["other"] != {
                    "event_status": "scheduled", "event_year": 2025
            }):
                return ["task_first_v2_bad_status_evidence"]
        elif goal == "year_at_least_2025":
            if (policy["selected"] != {
                    "event_status": "recorded", "event_year": 2025
            } or policy["other"] != {
                    "event_status": "recorded", "event_year": 2024
            }):
                return ["task_first_v2_bad_temporal_evidence"]
        else:
            return ["task_first_v2_unsupported_event_relation"]
        cases = base["macro_plan"]["all_cases"]
        plans = base["macro_plan"]
        from .hybrid_world import _route_key
        extra = evidence["scheduled_only_source_claims"]
        if set(extra) != {"link_a", "link_b"}:
            return ["task_first_v2_missing_scheduled_competitors"]
        for sid, shift in (("link_a", 3), ("link_b", 2)):
            if set(extra[sid]) != set(cases):
                return ["task_first_v2_missing_per_case_claim"]
            for i, case in enumerate(cases):
                record = extra[sid][case]
                pid = plans["synthetic_entities"][case]["entity_id"]
                expected_uri = plans["candidate_paths"][
                    cases[(i+shift) % len(cases)]
                ][0]["subject"]
                expected_eid = base["task_first_evidence"][
                    "scheduled_event_records"][case]["event_id"]
                if record != {
                    "project_id": pid, "event_id": expected_eid,
                    "entity_0": expected_uri,
                    "route_key": _route_key(pid, expected_uri),
                }:
                    return ["task_first_v2_noncausal_scheduled_candidate"]
        if (evidence["causal_port"]["required"] != (
                graph["shape"] == "event_gated_association")
                or evidence["causal_port"]["field"] != "event_id"):
            return ["task_first_v2_wrong_port_contract"]
        return []
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return ["task_first_v2_evidence_invalid:" + str(exc)[:150]]


def materialize_world_v2(spec: dict) -> dict:
    """Keep frozen v1 replay untouched; overlay v2 evidence deterministically."""
    from .world import _rows
    import copy as _copy

    base = spec["task_first_v1_backbone"]
    relation = spec["task_first_v2_relation_sketch"]
    evidence = spec["task_first_v2_evidence"]
    issues = validate_evidence_v2(evidence, relation, base)
    if issues:
        raise ValueError("task_first_v2_world_contract:" + ",".join(issues))
    world = materialize_world(base)
    bank = world["sources"]
    cases = base["macro_plan"]["all_cases"]
    selected = base["task_first_evidence"]["main_event_records"]
    other = base["task_first_evidence"]["scheduled_event_records"]
    policy = evidence["event_facts_policy"]
    by_eid = {}
    for case in cases:
        by_eid[selected[case][0]["event_id"]] = policy["selected"]
        by_eid[other[case]["event_id"]] = policy["other"]
    for row in bank["event_ledger"]["observations"]:
        data = row["values"]
        label = by_eid[data["event_id"]]
        data.update(label)
        if "(scheduled follow-up)" in data["event_name"]:
            data["event_name"] = data["event_name"].replace(
                "(scheduled follow-up)", "(supplementary session)"
            )
    # Make event records correspond to their materialized synthetic
    # entities; do not leave a stale status/year in the truth ledger.
    entities = world["synthetic_world"]["entities"]
    for case in cases:
        entities["events"][case]["status"] = policy["selected"]["event_status"]
        entities["events"][case]["year"] = policy["selected"]["event_year"]
        entities["scheduled_events"][case]["status"] = policy["other"][
            "event_status"]
        entities["scheduled_events"][case]["display_name"] = other[case][
            "event_name"].replace(
                "(scheduled follow-up)", "(supplementary session)"
            )
        entities["scheduled_events"][case]["year"] = policy["other"][
            "event_year"]

    if evidence["causal_port"]["required"]:
        for sid in ("link_a", "link_b"):
            values = [dict(row["values"])
                      for row in bank[sid]["observations"]]
            values += [
                dict(evidence["scheduled_only_source_claims"][sid][case])
                for case in cases
            ]
            # Deterministic ordering, no source row loses provenance.
            values.sort(key=lambda r: (
                r["project_id"], r["event_id"], r["entity_0"]
            ))
            entry = next(s for s in spec["sources"]
                         if s["source_id"] == sid)
            source = _rows(sid, entry["modality"], entry["title"],
                           entry["purpose"], values)
            source["access_policy"] = {"mode": "browse"}
            source["evidence_kind"] = (
                "event_specific_unverified_reference_hypotheses"
            )
            for record in source["observations"]:
                record["claim_status"] = "unverified_candidate"
                record["evidence_origin_id"] = "simulated:" + digest(
                    sid, record["values"]["event_id"],
                    record["values"]["route_key"],
                )[:20]
            bank[sid] = source

    world["task_first_evidence_contract"] = {
        "version": EVIDENCE_V2_VERSION,
        "task_local_relation": _copy.deepcopy(
            relation["event_relation"]
        ),
        "required_port": _copy.deepcopy(evidence["causal_port"]),
        "context_fields": _copy.deepcopy(
            evidence["context_not_credited"]
        ),
    }
    world["value_sampling_contract"] = (
        "graph_first_local_relation_and_causal_event_port_v2"
    )
    world["knowledge_witness"] = _copy.deepcopy(
        spec["knowledge_witness"]
    )
    return world
