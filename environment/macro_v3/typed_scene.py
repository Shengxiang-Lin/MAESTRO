"""Execute, author and audit YAGO-grounded typed Macro DAG pilot scenes (v3.2).

The abstract graph is frozen BEFORE World binding and question realization.
A Macro sees only its local sources and the explicitly declared input ports.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import random

from .composer import _digest
from .graph_generator import generate_graph
from .graph_validator import validate_graph
from .typed_world import ground_graph
from .yago import Edge, KnowledgeGraph

VERSION = "maestro.graph_grounded_scene.v3.2"
PROGRAM_VERSION = "maestro.port_based_macro_program.v1"


def _single(source: dict, field: str, value: str) -> dict:
    matched = [obs for obs in source["observations"]
               if obs["values"].get(field) == value]
    if len(matched) != 1:
        raise ValueError("nonunique_or_missing_local_observation:" + source["source_id"])
    return matched[0]


def execute(scene: dict) -> tuple[list[dict], dict]:
    """No global World scan inside an individual Macro; all inputs are explicit."""
    graph = scene["macro_plan"]
    bank = scene["world"]["sources"]
    binding = scene["oracle_binding"]
    products = {}
    reports = []
    for node in graph["nodes"]:
        mid, role = node["macro_id"], node["role"]
        local = {sid: bank[sid] for sid in node["source_scope"]}
        inputs = {
            port: products[edge["from_macro"]]["ports"][edge["from_port"]]
            for port, edge in node["input_ports"].items()
        }
        input_refs = {
            port: set(products[edge["from_macro"]]["refs"])
            for port, edge in node["input_ports"].items()
        }
        refs: set[str] = set()
        steps = []
        if role == "discovery":
            criteria = binding["criteria"][mid]
            allowed_cases = (set(inputs["prior_candidates"])
                             if "prior_candidates" in inputs else None)
            candidate_sets = []
            provenance = defaultdict(set)
            for sid in node["source_scope"]:
                source = local[sid]
                matches = [
                    o for o in source["observations"]
                    if o["values"]["cue"] == criteria[sid]
                    and (allowed_cases is None or
                         o["values"]["case_id"] in allowed_cases)
                ]
                candidate_sets.append({o["values"]["case_id"] for o in matches})
                for o in matches:
                    provenance[o["values"]["case_id"]].update(o["fact_refs"])
                steps.append({
                    "tool": ("observe_simulated_media" if source["modality"] in
                             {"image", "video", "audio"} else "search_and_read_records"),
                    "source_id": sid,
                    "criterion": criteria[sid],
                    "returned_candidates": len(matches),
                    "supporting_fact_refs": sorted(
                        ref for o in matches for ref in o["fact_refs"]
                    ),
                })
            if not candidate_sets:
                raise ValueError("discovery_has_no_sources")
            common = set.intersection(*candidate_sets)
            if allowed_cases is not None:
                common &= allowed_cases
            if not common:
                raise ValueError("discovery_empty_candidate_set")
            if "prior_candidates" in inputs:
                # Only accepted upstream candidate evidence can enter this
                # Macro's accumulated provenance.
                previous_edge = node["input_ports"]["prior_candidates"]
                previous = products[previous_edge["from_macro"]]["ports"]["evidence"]
                for case in common:
                    provenance[case].update(previous.get(case, []))
            witness = {case: sorted(provenance[case]) for case in sorted(common)}
            ports = {"candidates": sorted(common), "evidence": witness}
            refs = {r for values in witness.values() for r in values}
            steps.append({
                "tool": "join_candidates_from_independent_sources",
                "source_id": None,
                "candidate_count": len(common),
                "supporting_fact_refs": sorted(refs),
            })
        elif role == "identification":
            sets = [set(v) for k, v in inputs.items()
                    if k.startswith("candidate_")]
            if not sets or len(set.intersection(*sets)) != 1:
                raise ValueError("identity_not_unique")
            case = next(iter(set.intersection(*sets)))
            obs = _single(local["entity_crosswalk"], "case_id", case)
            ports = {"entity": {"case_id": case,
                                "entity_uri": obs["values"]["entity_uri"]}}
            refs.update(obs["fact_refs"])
            steps.extend([
                {"tool": "reconcile_candidate_handoffs", "source_id": None,
                 "candidate_count": 1, "supporting_fact_refs": []},
                {"tool": "lookup_dossier_entity", "source_id": "entity_crosswalk",
                 "supporting_fact_refs": list(obs["fact_refs"])},
            ])
        elif role == "investigation":
            subject = inputs["subject"]["entity_uri"]
            paths = binding["relation_paths"][mid]
            for step in paths:
                sid = step["source_id"]
                matches = [
                    o for o in local[sid]["observations"]
                    if o["values"]["subject_uri"] == subject
                    and o["values"]["predicate"] == step["predicate"]
                ]
                if len(matches) != 1:
                    raise ValueError("investigation_nonunique_relation:" + mid)
                o = matches[0]
                if o["values"]["object_uri"] != step["object"]:
                    raise ValueError("grounded_yago_path_disagrees")
                subject = o["values"]["object_uri"]
                refs.update(o["fact_refs"])
                label = o["values"]["object_label"]
                steps.append({
                    "tool": "search_and_query_yago_evidence",
                    "source_id": sid,
                    "predicate": step["predicate"],
                    "observed_object_uri": subject,
                    "supporting_fact_refs": list(o["fact_refs"]),
                })
            finding = {
                "linked_entity_uri": subject, "linked_entity": label,
                "predicate_path": [step["predicate"] for step in paths],
            }
            if "verified_context" in inputs:
                context = inputs["verified_context"]
                finding["verified_context"] = context["review_status"]
                refs.update(input_refs["verified_context"])
                steps.append({
                    "tool": "incorporate_verified_prior_claim",
                    "source_id": None, "status": context["review_status"],
                    "supporting_fact_refs": sorted(input_refs["verified_context"]),
                })
            ports = {"finding": finding}
        elif role == "verification":
            observed = inputs["finding"]["linked_entity_uri"]
            statuses = []
            for sid in binding["review_sources"][mid]:
                o = _single(local[sid], "object_uri", observed)
                statuses.append(o["values"]["assessment"])
                refs.update(o["fact_refs"])
                steps.append({
                    "tool": "read_independent_case_assessment",
                    "source_id": sid, "reported_status": o["values"]["assessment"],
                    "supporting_fact_refs": list(o["fact_refs"]),
                })
            disagreement = len(set(statuses)) > 1
            ports = {"verified": {
                "object_uri": observed,
                "review_status": statuses[-1],
                "agreement": "discrepancy_resolved" if disagreement else "corroborated",
            }}
            refs.update(input_refs["finding"])
            steps.append({
                "tool": "compare_claims_and_adjudicate", "source_id": None,
                "supporting_fact_refs": sorted(refs),
            })
        elif role == "comparison":
            left, right = inputs["left"], inputs["right"]
            ports = {"comparison": {
                "left_entity": left["linked_entity_uri"],
                "right_entity": right["linked_entity_uri"],
                "same_entity": left["linked_entity_uri"] == right["linked_entity_uri"],
            }}
            refs = set().union(*input_refs.values())
            steps.append({
                "tool": "compare_independent_relation_results",
                "source_id": None,
                "supporting_fact_refs": sorted(refs),
            })
        elif role == "synthesis":
            subject = inputs["entity"]
            results = [
                {"port": port, "type": node["input_ports"][port]["type"],
                 "result": deepcopy(value)}
                for port, value in sorted(inputs.items())
                if port.startswith("result_")
            ]
            if not results:
                raise ValueError("synthesis_without_evidence_product")
            original = []
            if "original_evidence" in inputs:
                original = list(inputs["original_evidence"].get(
                    subject["case_id"], []
                ))
                if not original:
                    raise ValueError("synthesis_missing_original_witness")
            ports = {"result": {
                "identified_case": subject["case_id"],
                "identified_entity_uri": subject["entity_uri"],
                "findings": results,
                "original_witness_refs": sorted(original),
            }}
            refs = set().union(*input_refs.values())
            refs.update(original)
            steps.append({
                "tool": "synthesize_typed_workproducts",
                "source_id": None, "input_count": len(inputs),
                "supporting_fact_refs": sorted(refs),
            })
        else:
            raise ValueError("unknown_macro_role_execution:" + role)
        if set(ports) != set(node["output_ports"]):
            raise ValueError("macro_output_ports_missing:" + mid)
        products[mid] = {"ports": ports, "refs": sorted(refs)}
        reports.append({
            "task_id": "work_" + mid, "macro_id": mid,
            "macro_role": role, "objective": node["objective"],
            "used_source_ids": list(node["source_scope"]),
            "used_upstream_handoffs": list(node["depends_on"]),
            "input_port_contract": deepcopy(node["input_ports"]),
            "output_ports": deepcopy(node["output_ports"]),
            "workproduct": deepcopy(ports),
            "fact_refs": sorted(refs),
            "internal_trace": steps,
            "grounding": "deterministic_scoped_synthetic_and_yago_evidence",
        })
    terminal = graph["terminal_macro_id"]
    return reports, products[terminal]["ports"]["result"]


def _question_draft(scene: dict, kg: KnowledgeGraph, rng: random.Random) -> tuple[str, dict]:
    binding = scene["oracle_binding"]
    nodes = scene["macro_plan"]["nodes"]
    clues = binding["question_hints"]
    descriptions = [
        f"{x['title']} gives {x['attribute']} '{x['cue']}'"
        for x in clues
    ]
    routes = [
        " then ".join(kg.label(step["predicate"])
                      for step in binding["relation_paths"][n["macro_id"]])
        for n in nodes if n["role"] == "investigation"
    ]
    flags = {n["role"] for n in nodes}
    request = "Identify the referenced case and report the documented linked entities"
    if "comparison" in flags:
        request += ", including whether the compared results refer to the same entity"
    if "verification" in flags:
        request += ", along with the separately adjudicated assessment"
    if any(n["role"] == "discovery" and
           n["macro_id"] in {x["from_macro"] for x in
                           nodes[-1]["input_ports"].values()}
           for n in nodes):
        request += ", citing the original source evidence for the dossier"
    samples = (
        "Several simulated records in a digital reference archive point to a "
        "single unidentified dossier. {clues}. {request}; the research "
        "directions concern {routes}.",
        "An archive analyst needs to reconcile a set of differently indexed "
        "records. Available evidence says: {clues}. {request}, drawing on "
        "the documented relation trails {routes}.",
        "A researcher is tracing a recovered reference across archives. "
        "The available source observations are: {clues}. {request}; consult "
        "the recorded YAGO relations {routes}.",
        "The reference desk received incomplete descriptions from independent "
        "collections: {clues}. {request}. The relevant relationship paths are "
        "{routes}.",
    )
    question = rng.choice(samples).format(
        clues="; ".join(descriptions), routes="; ".join(routes), request=request,
    )
    contract = {
        "question_status": "deterministic_draft_not_accepted",
        "required_clues": {x["source_id"]: x["cue"] for x in clues},
        "relation_routes": {
            k: [v["predicate"] for v in route]
            for k, route in binding["relation_paths"].items()
        },
        "required_roles": [n["role"] for n in nodes],
        "terminal_port": "result",
        "result_slots": sorted(k for k in nodes[-1]["input_ports"]
                               if k.startswith("result_")),
        "original_witness_required": "original_evidence" in nodes[-1]["input_ports"],
    }
    return question, contract


def validate_scene(scene: dict, kg: KnowledgeGraph | None = None) -> dict:
    """Replay, provenance, port correctness and source-level counterfactual audit."""
    if scene.get("version") != VERSION:
        raise ValueError("unsupported_typed_scene_version")
    graph = scene["macro_plan"]
    audit = validate_graph(graph)
    sources = scene["world"]["sources"]
    fact_source = {}
    for sid, source in sources.items():
        for row in source["observations"]:
            for fact in row["fact_refs"]:
                if fact in fact_source:
                    raise ValueError("reused_fact_ref_between_sources")
                fact_source[fact] = sid

    for n in graph["nodes"]:
        if not set(n["source_scope"]) <= set(sources):
            raise ValueError("macro_read_scope_invalid")
        if not n["objective"]:
            raise ValueError("macro_missing_objective")
    for fact in scene["world"]["real_fact_path"]:
        if fact["origin"] != "yago":
            raise ValueError("forged_yago_provenance")
        if kg is not None and Edge(fact["subject"], fact["predicate"],
                                   fact["object"]) not in kg.edges:
            raise ValueError("unknown_fact_labeled_as_real_yago")
        if not any(
            row["origin"] == "yago"
            and row["values"]["subject_uri"] == fact["subject"]
            and row["values"]["predicate"] == fact["predicate"]
            and row["values"]["object_uri"] == fact["object"]
            for row in sources[fact["source_id"]]["observations"]
        ):
            raise ValueError("missing_yago_source_witness")

    discovery = [n for n in graph["nodes"] if n["role"] == "discovery"]
    all_matches = []
    for n in discovery:
        for sid in n["source_scope"]:
            criterion = scene["oracle_binding"]["criteria"][n["macro_id"]][sid]
            selected = {
                r["values"]["case_id"]
                for r in sources[sid]["observations"]
                if r["values"]["cue"] == criterion
            }
            all_matches.append(selected)
    if len(set.intersection(*all_matches)) != 1:
        raise ValueError("candidate_identity_not_unique_from_all_sources")
    for i in range(len(all_matches)):
        if len(set.intersection(*[s for j, s in enumerate(all_matches) if j != i])) <= 1:
            raise ValueError("one_discovery_source_is_redundant")
    if scene["question_contract"]["required_clues"] != {
        hint["source_id"]: hint["cue"]
        for hint in scene["oracle_binding"]["question_hints"]
    }:
        raise ValueError("question_clue_contract_mismatch")
    if scene["question_contract"]["result_slots"] != sorted(
        k for k in graph["nodes"][-1]["input_ports"] if k.startswith("result_")
    ):
        raise ValueError("question_result_ports_mismatch")
    reports, answer = execute(scene)
    if reports != scene["task_feedback"] or scene["gold_answer"] != [answer]:
        raise ValueError("typed_scene_replay_mismatch")
    for report in reports:
        node = next(n for n in graph["nodes"] if n["macro_id"] == report["macro_id"])
        legal = set(node["source_scope"])
        legal.update(
            fact_source[f] for parent in node["depends_on"]
            for f in next(row["fact_refs"] for row in reports
                          if row["macro_id"] == parent)
        )
        if any(fact_source.get(f) not in legal for f in report["fact_refs"]):
            raise ValueError("macro_uses_nonlocal_unreported_evidence")
        for step in report["internal_trace"]:
            if step["source_id"] is not None and step["source_id"] not in node["source_scope"]:
                raise ValueError("unscoped_trace_source")
            if not set(step["supporting_fact_refs"]) <= set(report["fact_refs"]):
                raise ValueError("unreported_trace_fact_ref")

    expected_hash = _digest({
        "macro_plan": graph, "world": scene["world"],
        "oracle_binding": scene["oracle_binding"],
        "question_contract": scene["question_contract"],
        "question": scene["question"],
        "answer": scene["gold_answer"],
    })
    if scene["replay_hash"] != expected_hash:
        raise ValueError("typed_scene_frozen_content_changed")
    return {
        **audit, "source_necessary": True,
        "yago_facts_used": len(scene["world"]["real_fact_path"]),
        "independent_investigations": len(discovery),
        "question_status": "draft",
    }


def compose_scene(kg: KnowledgeGraph, *, seed: int = 17, ordinal: int = 0,
                  min_hops: int = 1, max_hops: int = 3,
                  max_branches: int = 3, max_macros: int = 12,
                  max_parallel: int = 3, max_waves: int = 7) -> dict:
    abstract = generate_graph(
        seed=seed, ordinal=ordinal, max_branches=max_branches,
        max_macros=max_macros, max_parallel=max_parallel, max_waves=max_waves,
    )
    rng = random.Random(f"typed-macro-world-v1:{seed}:{ordinal}")
    world, binding = ground_graph(
        kg, abstract, rng, min_hops=min_hops, max_hops=max_hops,
    )
    id_fragment = _digest([seed, ordinal, abstract["terminal_macro_id"],
                           world["private_witness"]])
    task_id = f"question-v32-{ordinal:06d}-{id_fragment[:8]}"
    scene = {
        "version": VERSION, "task_id": task_id,
        "macro_plan": abstract,
        "world": world, "oracle_binding": binding,
        "question": "", "question_contract": {},
        "question_authoring": {"status": "draft", "model": None},
        "problem_spec": {"version": PROGRAM_VERSION,
                         "seed": seed, "ordinal": ordinal},
        "gold_program": {"version": PROGRAM_VERSION,
                         "steps": [dict(macro_id=n["macro_id"],
                                        operation=n["operation"])
                                   for n in abstract["nodes"]]},
        "gold_answer": [], "task_feedback": [], "gold_trace": [],
        "execution_assumptions": {
            "simulated_media": True, "full_world_hidden_from_main": True,
            "scope_limited_macro_execution": True,
        },
    }
    question, contract = _question_draft(scene, kg, rng)
    scene["question"] = question
    scene["question_contract"] = contract
    reports, answer = execute(scene)
    scene["gold_answer"] = [answer]
    scene["task_feedback"] = reports
    scene["gold_trace"] = deepcopy(reports)
    scene["macro_graph"] = {
        "version": graph_version(abstract),
        "tasks": [
            {"task_id": "work_" + n["macro_id"],
             "depends_on": ["work_" + p for p in n["depends_on"]],
             "source_ids": list(n["source_scope"]),
             "macro_role": n["role"], "objective": n["objective"],
             "input_ports": n["input_ports"], "outputs": n["output_ports"]}
            for n in abstract["nodes"]
        ],
        "reference_waves": [["work_" + mid for mid in wave]
                            for wave in abstract["reference_waves"]],
        "terminal_task_id": "work_" + abstract["terminal_macro_id"],
        "macro_task_count": len(abstract["nodes"]),
    }
    scene["public"] = {
        "task_id": task_id, "question": question,
        "available_source_descriptions": [
            {"source_id": sid, "title": source["title"],
             "modality_hint": source["modality"]}
            for sid, source in world["sources"].items()
        ],
        "status": "draft_not_llm_reviewed",
        "max_concurrent_subagents": max_parallel,
    }
    scene["replay_hash"] = _digest({
        "macro_plan": scene["macro_plan"], "world": world,
        "oracle_binding": binding, "question_contract": contract,
        "question": question, "answer": scene["gold_answer"],
    })
    scene["quality_report"] = validate_scene(scene, kg)
    return scene


def graph_version(graph: dict) -> str:
    return graph["version"]
