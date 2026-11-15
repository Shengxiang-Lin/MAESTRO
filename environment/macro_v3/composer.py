"""Macro-first v3: typed DAG -> synthetic source binding -> executable evidence traces.

This is intentionally a NEW, versioned scene contract. It never routes new
graphs through the hard-coded v2 macro solver or claims Stage-2 compatibility.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import random

from .yago import Edge, KnowledgeGraph

VERSION = "maestro.macro_graph.v3"
CUES = (
    ("image", "visual observation", "color", ("blue", "amber", "green", "silver", "red")),
    ("text", "registry entry", "registry status", ("active", "reviewed", "flagged", "pending")),
    ("table", "inspection sheet", "inspection code", ("north", "east", "priority", "routine")),
    ("text", "field notes", "marking", ("triangle", "spiral", "crescent", "star")),
)
QUESTION_STYLES = (
    "For a simulated investigation, use {clues} to identify the same item. "
    "Then follow its documented {relations}. {request}",
    "An analyst is reconciling several independent source records. Locate the "
    "item satisfying {clues}; examine its {relations}. {request}",
    "The available materials do not identify their subject directly. Determine "
    "which item meets {clues}, then trace {relations}. {request}",
    "Complete this source-grounded inquiry: first establish the item meeting "
    "{clues}; next investigate {relations}. {request}",
)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _source(sid: str, modality: str, title: str, fields: list[str],
            records: list[dict], *, mode: str = "browse", origin: str = "synthetic") -> dict:
    assert all(set(row) == set(fields) for row in records)
    observations = [
        {"observation_id": f"{sid}.OBS{i:03d}", "fact_refs": [f"{sid}:F{i:03d}"],
         "origin_fact_id": ("yago:" + _digest(values)
                            if origin == "yago" else f"synthetic:{sid}:{i}"),
         "values": dict(values), "origin": origin}
        for i, values in enumerate(records, 1)
    ]
    return {
        "source_id": sid, "modality": modality, "title": title,
        "schema": list(fields),
        "representation": "synthetic_structured_observations_no_media_decoding",
        "access_policy": {"mode": mode, **({"key_field": fields[0]}
                                          if mode == "keyed_lookup" else {})},
        "observations": observations,
    }


def _macro(mid: str, role: str, objective: str, parents: list[str],
           sources: list[str], output: dict[str, str], strategy: str, **params) -> dict:
    return {
        "macro_id": mid, "role": role, "objective": objective,
        "depends_on": list(parents), "source_scope": list(sources),
        "output_schema": output, "solver_strategy": strategy,
        "solver_params": params,
        "evidence_requirements": ["source_fact_refs"],
    }


def _structural_plan(rng: random.Random, *, n_branches: int, n_hops: int,
                     verify: bool) -> list[dict]:
    """Sample typed partial-order graph BEFORE binding YAGO entities/sources."""
    tasks = []
    for i in range(n_branches):
        mid = f"M{i+1}"
        role = ("identification", "discovery", "investigation", "verification")[i]
        # A later investigation can narrow an earlier candidate handoff.
        # The dependency is meaningful: execution restricts inspected records.
        parent = rng.choice(tasks[:i])["macro_id"] if i and rng.random() < 0.4 else None
        tasks.append(_macro(
            mid, role,
            ("Refine prior candidates with " if parent else "Identify candidates from ")
            + f"source group {i+1}",
            [parent] if parent else [], [f"clue_{i+1}"],
            {"candidate_ids": "list[str]"},
            "filter_candidates", cue_source=i,
        ))
    merge = f"M{len(tasks)+1}"
    tasks.append(_macro(
        merge, "identification", "Reconcile all candidate investigations "
        "and recover the referenced real-world entity",
        [t["macro_id"] for t in tasks], ["entity_index"],
        {"candidate_id": "str", "entity_uri": "str"}, "resolve_identity",
    ))
    previous = merge
    for i in range(n_hops):
        mid = f"M{len(tasks)+1}"
        is_last = i == n_hops - 1 and not verify
        tasks.append(_macro(
            mid, "investigation",
            f"Investigate relationship hop {i+1} using its scoped document",
            [previous], [f"relation_{i+1}"],
            {"answer": "str"} if is_last else {"entity_uri": "str"},
            "follow_relation", step=i, terminal=is_last,
        ))
        previous = mid
    if verify:
        check = f"M{len(tasks)+1}"
        tasks.append(_macro(
            check, "verification",
            "Independently check the identified reference's simulated review status",
            [merge], ["verification_log"], {"status": "str"}, "check_status",
        ))
        final = f"M{len(tasks)+1}"
        tasks.append(_macro(
            final, "synthesis",
            "Combine the relationship investigation and the review verdict",
            [previous, check], [], {"answer": "str", "verification_status": "str"},
            "synthesize",
        ))
    schemas = {t["macro_id"]: t["output_schema"] for t in tasks}
    for task in tasks:
        task["input_contract"] = {
            parent: dict(schemas[parent]) for parent in task["depends_on"]
        }
    return tasks


def _build_world(graph: KnowledgeGraph, path: list[Edge], tasks: list[dict],
                 rng: random.Random) -> tuple[dict, list[dict]]:
    branch_count = sum(t["solver_strategy"] == "filter_candidates" for t in tasks)
    count = branch_count + rng.randint(3, 6)
    labels = [f"Item-{x+1:03d}" for x in range(count)]
    rng.shuffle(labels)
    target = labels[0]
    # Each clue eliminates one unique shadow distractor. Removing ANY clue
    # leaves at least one alternative candidate, not merely a broken program.
    shadows = {labels[i+1]: i for i in range(branch_count)}
    sources = {}
    hints = []
    for branch in range(branch_count):
        modality, title, attribute, pool = CUES[branch]
        expected = rng.choice(pool)
        alternate = rng.choice([x for x in pool if x != expected])
        rows = []
        for item in labels:
            matches = (item == target or
                       (item in shadows and shadows[item] != branch))
            rows.append({"candidate_id": item,
                         "cue": expected if matches else alternate})
        rng.shuffle(rows)
        sid = f"clue_{branch+1}"
        sources[sid] = _source(sid, modality, title,
                               ["candidate_id", "cue"], rows)
        tasks[branch]["solver_params"].update(
            {"cue": expected, "attribute": attribute})
        hints.append({"source_id": sid, "attribute": attribute,
                      "expected_value": expected, "title": title})
    subjects = list(dict.fromkeys(edge.subject for edge in graph.edges))
    rng.shuffle(subjects)
    index_rows = []
    for item in labels:
        uri = path[0].subject if item == target else (
            subjects.pop() if subjects else "urn:maestro:decoy:" + item
        )
        if item != target and uri == path[0].subject:
            uri = "urn:maestro:decoy:" + item
        index_rows.append({"candidate_id": item, "entity_uri": uri})
    rng.shuffle(index_rows)
    sources["entity_index"] = _source(
        "entity_index", "table", "entity cross-reference directory",
        ["candidate_id", "entity_uri"], index_rows, mode="keyed_lookup",
    )
    for i, fact in enumerate(path, 1):
        rows = [{"subject_uri": fact.subject, "predicate": fact.predicate,
                 "object_uri": fact.object}]
        # Include unrelated REAL YAGO facts as distractors. No invented YAGO facts.
        pool = [e for e in graph.edges if e.subject != fact.subject]
        for other in rng.sample(pool, min(len(pool), rng.randint(1, 5))):
            rows.append({"subject_uri": other.subject, "predicate": other.predicate,
                         "object_uri": other.object})
        rng.shuffle(rows)
        sid = f"relation_{i}"
        sources[sid] = _source(
            sid, "text", f"knowledge-base relation report {i}",
            ["subject_uri", "predicate", "object_uri"], rows,
            mode="keyed_lookup", origin="yago",
        )
    if any(t["solver_strategy"] == "check_status" for t in tasks):
        statuses = ("approved", "under review", "revised", "confirmed")
        verdict = rng.choice(statuses)
        rows = [{"entity_uri": row["entity_uri"],
                 "status": verdict if row["candidate_id"] == target
                           else rng.choice(statuses)}
                for row in index_rows]
        rng.shuffle(rows)
        sources["verification_log"] = _source(
            "verification_log", "table", "simulated independent review ledger",
            ["entity_uri", "status"], rows, mode="keyed_lookup",
        )
    world = {
        "version": "maestro.graph_world.v3", "simulated_only": True,
        "sources": sources,
        "witness": {"candidate_id": target, "entity_uri": path[0].subject},
        "real_fact_path": [dict(subject=x.subject, predicate=x.predicate,
                                object=x.object, origin="yago") for x in path],
    }
    return world, hints


def _run(tasks: list[dict], sources: dict, graph: KnowledgeGraph) -> tuple[list[dict], dict]:
    products = {}
    feedback = []
    for task in tasks:
        strategy = task["solver_strategy"]
        mid = task["macro_id"]
        parents = [products[p] for p in task["depends_on"]]
        scoped = {sid: sources[sid] for sid in task["source_scope"]}
        if strategy == "filter_candidates":
            source = scoped[task["source_scope"][0]]
            cue = task["solver_params"]["cue"]
            upstream = (set(parents[0]["values"]["candidate_ids"])
                        if parents else None)
            matches = {row["values"]["candidate_id"]: row["fact_refs"]
                       for row in source["observations"]
                       if row["values"]["cue"] == cue
                       and (upstream is None or row["values"]["candidate_id"] in upstream)}
            if not matches or (not parents and len(matches) < 2):
                raise ValueError("unreasonably_easy_candidate_filter")
            value = {"candidate_ids": sorted(matches)}
            refs = sorted({x for items in matches.values() for x in items})
            witness = {
                key: sorted(set(row_refs) | set(parents[0]["witness"][key]))
                if parents else row_refs
                for key, row_refs in matches.items()
            }
            refs = sorted({x for items in witness.values() for x in items})
            steps = (["reuse_upstream_candidates"] if parents else []) + [
                "observe_or_search_source", "extract_attributes",
                "filter_candidates", "record_evidence",
            ]
        elif strategy == "resolve_identity":
            if not parents:
                raise ValueError("identity_missing_branches")
            common = set(parents[0]["witness"])
            for p in parents[1:]:
                common.intersection_update(p["witness"])
            if len(common) != 1:
                raise ValueError("candidate_resolution_not_unique")
            candidate = next(iter(common))
            source = scoped["entity_index"]
            selected = [row for row in source["observations"]
                        if row["values"]["candidate_id"] == candidate]
            if len(selected) != 1:
                raise ValueError("entity_index_not_unique")
            row = selected[0]
            value = {"candidate_id": candidate,
                     "entity_uri": row["values"]["entity_uri"]}
            refs = sorted({ref for parent in parents
                           for ref in parent["witness"][candidate]}
                          | set(row["fact_refs"]))
            witness = {}
            steps = ["compare_independent_candidate_sets",
                     "intersect_candidates", "lookup_entity_identity"]
        elif strategy == "follow_relation":
            incoming = parents[0]["values"]["entity_uri"]
            step = task["solver_params"]["step"]
            # Predicate is frozen in the task, not inferred from a private answer.
            predicate = task["solver_params"]["predicate"]
            source = scoped[f"relation_{step+1}"]
            found = [row for row in source["observations"]
                     if row["values"]["subject_uri"] == incoming
                     and row["values"]["predicate"] == predicate]
            if len(found) != 1:
                raise ValueError("relation_query_not_unique")
            row = found[0]
            uri = row["values"]["object_uri"]
            value = ({"answer": graph.label(uri)}
                     if task["solver_params"]["terminal"] else
                     {"entity_uri": uri})
            refs = sorted(set(parents[0]["refs"]) | set(row["fact_refs"]))
            witness = {}
            steps = ["search_relation_source", "lookup_using_upstream_entity",
                     "compare_relation_candidates", "extract_result"]
        elif strategy == "check_status":
            entity_uri = parents[0]["values"]["entity_uri"]
            found = [row for row in scoped["verification_log"]["observations"]
                     if row["values"]["entity_uri"] == entity_uri]
            if len(found) != 1:
                raise ValueError("verification_record_not_unique")
            row = found[0]
            value = {"status": row["values"]["status"]}
            refs = sorted(set(parents[0]["refs"]) | set(row["fact_refs"]))
            witness = {}
            steps = ["search_review_ledger", "inspect_matching_record",
                     "verify_entity_reference", "return_status"]
        elif strategy == "synthesize":
            value = {"answer": graph.label(parents[0]["values"]["entity_uri"]),
                     "verification_status": parents[1]["values"]["status"]}
            refs = sorted(set(parents[0]["refs"]) | set(parents[1]["refs"]))
            witness = {}
            steps = ["compare_investigation_and_review",
                     "resolve_final_response", "retain_provenance"]
        else:
            raise ValueError("unknown_macro_strategy:" + strategy)
        if set(value) != set(task["output_schema"]):
            raise ValueError("macro_output_schema_mismatch:" + mid)
        products[mid] = {"values": value, "refs": refs, "witness": witness}
        feedback.append({
            "task_id": "work_" + mid, "macro_id": mid,
            "macro_role": task["role"], "objective": task["objective"],
            "used_source_ids": list(scoped),
            "used_upstream_handoffs": ["work_" + p for p in task["depends_on"]],
            "output_schema": dict(task["output_schema"]),
            "result_rows": (len(witness) if strategy == "filter_candidates" else 1),
            "fact_refs": refs,
            "workproduct": (
                [{"values": {"candidate_id": key}, "source_fact_refs": list(support)}
                 for key, support in sorted(witness.items())]
                if strategy == "filter_candidates" else
                [{"values": deepcopy(value), "source_fact_refs": refs}]
            ),
            "internal_trace": [{"tool": op} for op in steps],
        })
    return feedback, products[tasks[-1]["macro_id"]]["values"]


def validate_scene(scene: dict) -> dict:
    """Independently check topology, provenance and counterfactual necessity."""
    tasks = scene["macro_plan"]["macro_tasks"]
    if not tasks:
        raise ValueError("empty_macro_dag")
    known = set()
    for task in tasks:
        if task["macro_id"] in known or any(p not in known for p in task["depends_on"]):
            raise ValueError("duplicate_macro_or_non_topological_dependency")
        known.add(task["macro_id"])
        expected_inputs = {m["macro_id"]: m["output_schema"]
                           for m in tasks if m["macro_id"] in task["depends_on"]}
        if task.get("input_contract") != expected_inputs:
            raise ValueError("macro_typed_input_contract_mismatch")
        if not set(task["source_scope"]) <= set(scene["world"]["sources"]):
            raise ValueError("missing_source_scope")
        if not task["objective"] or not task["output_schema"]:
            raise ValueError("invalid_macro_contract")
    branches = [t for t in tasks if t["solver_strategy"] == "filter_candidates"]
    observed = []
    for task in branches:
        sid = task["source_scope"][0]
        rows = scene["world"]["sources"][sid]["observations"]
        observed.append({
            r["values"]["candidate_id"] for r in rows
            if r["values"]["cue"] == task["solver_params"]["cue"]
        })
    if len(observed) < 2 or len(set.intersection(*observed)) != 1:
        raise ValueError("full_branch_intersection_not_unique")
    necessary = []
    for idx in range(len(observed)):
        remaining = [s for j, s in enumerate(observed) if j != idx]
        # For 2 branches, the remaining branch must still be ambiguous.
        count = len(set.intersection(*remaining))
        necessary.append(count > 1)
    if not all(necessary):
        raise ValueError("redundant_candidate_evidence_branch")
    if any(not t["depends_on"] for t in tasks[len(branches):]):
        raise ValueError("disconnected_downstream_macro")
    if not scene.get("gold_answer") or len(scene["gold_answer"]) != 1:
        raise ValueError("invalid_gold_answer")
    for i, edge in enumerate(scene["world"]["real_fact_path"], 1):
        if edge["origin"] != "yago":
            raise ValueError("false_real_fact_origin")
        rows = [row["values"] for row in
                scene["world"]["sources"][f"relation_{i}"]["observations"]]
        required = {"subject_uri": edge["subject"], "predicate": edge["predicate"],
                    "object_uri": edge["object"]}
        if required not in rows:
            raise ValueError("yago_fact_missing_from_evidence")
    return {
        "passed": True, "macro_count": len(tasks), "branch_count": len(branches),
        "all_branch_sources_necessary": True,
        "real_yago_hops": len(scene["world"]["real_fact_path"]),
        "role_distribution": dict(Counter(t["role"] for t in tasks)),
        "progressive_clue_tasks": sum(
            bool(t["depends_on"]) for t in branches
        ),
    }


def compose_scene(graph: KnowledgeGraph, *, seed: int = 17,
                  ordinal: int = 0, min_hops: int = 1,
                  max_hops: int = 3, max_branches: int = 4) -> dict:
    """Produce one Macro-first v3 pilot scene; no LLM and no original benchmark QA."""
    if not 1 <= min_hops <= max_hops <= 5:
        raise ValueError("invalid_hop_range")
    if not 2 <= max_branches <= len(CUES):
        raise ValueError("invalid_branch_range")
    rng = random.Random(int(_digest([seed, ordinal, "macro-v3"]), 16))
    path = graph.choose_walk(rng, min_hops=min_hops, max_hops=max_hops)
    branches = rng.randint(2, max_branches)
    verify = rng.random() < 0.5
    tasks = _structural_plan(rng, n_branches=branches,
                             n_hops=len(path), verify=verify)
    for task in tasks:
        if task["solver_strategy"] == "follow_relation":
            edge = path[task["solver_params"]["step"]]
            task["solver_params"]["predicate"] = edge.predicate
            task["objective"] = (
                "Investigate the documented " + graph.label(edge.predicate)
                + " relation for the upstream entity, using scoped records"
            )
    world, hints = _build_world(graph, path, tasks, rng)
    feedback, result = _run(tasks, world["sources"], graph)
    labels = [
        f"{hint['title']} ({hint['attribute']} = {hint['expected_value']})"
        for hint in hints
    ]
    relations = " followed by ".join(graph.label(e.predicate) for e in path)
    request = ("Return the final linked entity and the independent review status."
               if verify else "Which entity is reached at the end of this chain?")
    question = rng.choice(QUESTION_STYLES).format(
        clues=", ".join(labels), relations=relations, request=request
    )
    waves = []
    pending = {t["macro_id"] for t in tasks}
    finished = set()
    while pending:
        wave = [t["macro_id"] for t in tasks
                if t["macro_id"] in pending and set(t["depends_on"]) <= finished]
        if not wave:
            raise ValueError("macro_graph_cycle")
        waves.append(["work_" + mid for mid in wave])
        finished.update(wave)
        pending.difference_update(wave)
    task_id = f"question-v3-{ordinal:06d}-{_digest([seed, ordinal, path[0].subject])[:8]}"
    scene = {
        "version": "maestro.graph_grounded_scene.v3", "task_id": task_id,
        "question": question, "question_origin": "deterministic_contract_draft",
        "question_authoring": {"status": "draft", "model": None},
        "problem_spec": {"version": VERSION, "seed": seed, "ordinal": ordinal,
                         "goal": {"output_schema": {k: "str" for k in result}}},
        "macro_plan": {"version": VERSION, "macro_tasks": tasks,
                       "terminal_macro_id": tasks[-1]["macro_id"]},
        "macro_graph": {
            "version": VERSION,
            "tasks": [{"task_id": "work_" + t["macro_id"],
                       "depends_on": ["work_" + p for p in t["depends_on"]],
                       "source_ids": t["source_scope"],
                       "macro_role": t["role"], "objective": t["objective"],
                       "outputs": t["output_schema"]} for t in tasks],
            "reference_waves": waves,
            "terminal_task_id": "work_" + tasks[-1]["macro_id"],
            "macro_task_count": len(tasks),
        },
        "world": world,
        "gold_program": {"version": VERSION,
                         "macro_ops": [{"macro_id": t["macro_id"],
                                        "strategy": t["solver_strategy"],
                                        "params": t["solver_params"]}
                                       for t in tasks]},
        "gold_answer": [result],
        "gold_trace": feedback,
        "task_feedback": feedback,
        "public": {
            "task_id": task_id, "question": question,
            "available_source_descriptions": [
                {"source_id": sid, "title": source["title"],
                 "modality_hint": source["modality"]}
                for sid, source in world["sources"].items()
            ],
            "status": "ready_for_investigation",
            "max_concurrent_subagents": 10,
        },
        "execution_assumptions": {
            "simulated_media": True, "full_world_hidden_from_main": True,
            "scope_limited_macro_execution": True,
        },
    }
    scene["quality_report"] = validate_scene(scene)
    scene["replay_hash"] = _digest({
        "plan": scene["macro_plan"], "world": scene["world"],
        "answer": scene["gold_answer"],
    })
    return scene


def replay_scene(scene: dict, graph: KnowledgeGraph) -> dict:
    """Replay frozen scoped evidence; detect tampering or solver drift."""
    audit = validate_scene(scene)
    for row in scene["world"]["real_fact_path"]:
        if Edge(row["subject"], row["predicate"], row["object"]) not in graph.edges:
            raise ValueError("frozen_yago_fact_not_in_source_graph")
    tasks = scene["macro_plan"]["macro_tasks"]
    feedback, result = _run(tasks, scene["world"]["sources"], graph)
    if [result] != scene["gold_answer"] or feedback != scene["task_feedback"]:
        raise ValueError("macro_v3_replay_mismatch")
    expected_hash = _digest({
        "plan": scene["macro_plan"], "world": scene["world"],
        "answer": scene["gold_answer"],
    })
    if expected_hash != scene["replay_hash"]:
        raise ValueError("macro_v3_frozen_content_changed")
    return audit
