"""MMQA-anchored Macro TaskGraph compiler for the V4.5 Task-First environment.

The native MMQA composition and supporting context determine the work
dependencies. Each Macro Task is a substantial evidence work package, never
one KoPL operator. Output handoffs are typed. KQA stays an INTERNAL prior.
"""
from __future__ import annotations

import random

from envir.schema import digest
from .graph import (build_atomic_draft, coarsen, compile_contracts,
                    compile_waves, complexity)
from .priors import choose_kqa
from collections import Counter


def _topology_metrics(tasks: list[dict], edges: list[dict],
                      waves: list[list[str]]) -> dict:
    outdegree = Counter(e["from"] for e in edges)
    roots = sum(not task["depends_on"] for task in tasks)
    joins = sum(len(task["depends_on"]) > 1 for task in tasks)
    fanouts = sum(outdegree[task["task_id"]] > 1 for task in tasks)
    return {
        "roots": roots, "joins": joins, "fanouts": fanouts,
        "width": max(map(len, waves), default=0),
    }


def _classify_topology(stats: dict, depth: int) -> str:
    if stats["roots"] == 1 and stats["joins"] == 0:
        return "chain"
    if depth == 2 and stats["roots"] >= 2:
        return "parallel_join"
    if stats["fanouts"] and stats["joins"]:
        return "fork_join"
    if stats["joins"] and depth >= 4:
        return "staged_join"
    if stats["joins"]:
        return "mixed_join"
    return "progressive"


LEAF_MODALITIES = {
    "table": "table", "text": "text", "image": "image",
}
OPERATION_NAMES = {
    "compose": ("cross_source_composition", "ComposedFinding",
                "Connect evidence and intermediate references across sources"),
    "compare": ("cross_source_comparison", "ComparativeFinding",
                "Compare the relevant claims and values across source findings"),
    "intersect": ("evidence_intersection", "MatchedEvidence",
                  "Identify common entities or facts supported by both findings"),
    "intersection": ("evidence_intersection", "MatchedEvidence",
                     "Identify common entities or facts supported by both findings"),
    "count": ("evidence_aggregation", "AggregateFinding",
              "Count the supported matching entities in the combined findings"),
    "aggregate": ("evidence_aggregation", "AggregateFinding",
                  "Aggregate the relevant facts from the upstream findings"),
    "union": ("evidence_union", "CombinedEvidence",
              "Combine the relevant supported source findings"),
    "and": ("evidence_intersection", "MatchedEvidence",
            "Identify common supported facts across inputs"),
    "or": ("evidence_union", "CombinedEvidence",
           "Combine supported facts from the upstream inputs"),
}
FINAL_PREFIX = "Resolve the following source-scoped information need: "


class UnsuitableSeed(ValueError):
    """This seed cannot produce a legitimate multi-work-package DAG."""


def _modality_for_leaf(name: str) -> str:
    lower = name.lower()
    for token in ("table", "image", "text"):
        if token in lower:
            return token
    return "text"


def _work_source(source: dict, seed: dict, fallback: str) -> dict:
    modality = source["modality"]
    doc_id = source.get("doc_id")
    descriptor = (f"MultiModalQA {modality} reference {doc_id}"
                  if doc_id else f"simulated {modality} evidence role for {seed['topic']}")
    role = {"table": "records", "image": "material", "text": "context"}[modality]
    return {
        "role": role,
        "description": descriptor,
        "access": "synthetic_observation_only",
        "candidate_modalities": [modality],
        "source_origin": source["provenance"],
        "original_doc_id": doc_id,
        "source_leaf": fallback,
        "synthetic": True,
    }


def compose_seed_taskfirst(*, seed_record: dict, ordinal: int, run_seed: int,
                           kqa: list[dict]) -> dict:
    """Construct the Macro DAG before deriving its initial user question.

    MMQA types are the preferred graph prior. For a simple source type, real
    MMQA supporting-context roles provide a two-source integration task.
    No artificial extra tasks are introduced just to inflate graph depth.
    """
    rng = random.Random(run_seed * 1000003 + ordinal * 99017)
    question = seed_record["question"].strip()
    if not question:
        raise UnsuitableSeed("empty source question")
    available = list(seed_record["supporting_sources"])
    for modality in seed_record["source_modalities"]:
        if modality in LEAF_MODALITIES and not any(s["modality"] == modality for s in available):
            available.append({"modality": modality, "doc_id": None,
                              "provenance": "mmqa_modality_role_only"})
    if not available:
        raise UnsuitableSeed("no declared MMQA supporting source")
    tasks, edges, used_source_keys = [], [], set()
    leaf_cache = {}
    source_cursor = 0
    recipes = {}

    def source_task(modality: str, leaf_label: str) -> tuple[str, str, str]:
        nonlocal source_cursor
        # Prefer an unconsumed supporting source in the required modality.
        pick = next((s for s in available
                     if s["modality"] == modality
                     and (s["modality"], s.get("doc_id")) not in used_source_keys), None)
        if pick is None:
            # Never silently substitute a table for a requested image (etc.).
            # The original MMQA type must have a corresponding source role.
            pick = next((s for s in available if s["modality"] == modality), None)
        if pick is None:
            raise UnsuitableSeed("missing required source modality: " + modality)
        key = (pick["modality"], pick.get("doc_id"))
        if key in leaf_cache:
            return leaf_cache[key]
        used_source_keys.add(key)
        source_cursor += 1
        tid = "S" + str(len(tasks) + 1)
        role = _work_source(pick, seed_record, leaf_label)
        artifact = {
            "table": "RecordFindings", "image": "VisualSourceFindings",
            "text": "TextSourceFindings",
        }[pick["modality"]]
        task = {
            "task_id": tid,
            "work_type": "source_investigation_" + pick["modality"],
            "recipe_id": "mmqa_source_" + pick["modality"],
            "goal": (
                f"Investigate the {pick['modality']} reference material concerning "
                f"{seed_record['topic']}; establish source-scoped findings "
                f"relevant to the requested information."
            ),
            "deliverable": (
                f"A source-referenced {pick['modality']} evidence brief "
                "with identifiable claims and scope."
            ),
            "outputs": {"source_findings": artifact},
            "input_schema": {},
            "capability_hints": [{
                "table": "TableAnalysis", "image": "ImageAnalysis",
                "text": "DocumentAnalysis",
            }[pick["modality"]], "Research"],
            "source_requirements": [role],
            "internal_actions": [
                "identify the relevant source and question scope",
                "inspect the available simulated source observations",
                "extract and link supported attributes or entities",
                "prepare a source-scoped evidence brief",
            ],
            "grounding_status": "pending_synthetic_evidence",
        }
        tasks.append(task)
        recipes[artifact + ":" + tid] = task["recipe_id"]
        item = (tid, "source_findings", artifact)
        leaf_cache[key] = item
        return item

    def combine(op: str, inputs: list[tuple[str, str, str]],
                *, terminal: bool = False) -> tuple[str, str, str]:
        key = op.lower()
        work_type, output, action = OPERATION_NAMES.get(key, (
            "cross_source_integration", "IntegratedFinding",
            "Integrate source-scoped evidence while preserving reference identity",
        ))
        if terminal:
            output = "AnswerReport"
            work_type = "information_need_resolution"
        if len(inputs) < 1:
            raise UnsuitableSeed("operation without inputs")
        if len(inputs) == 1 and not terminal:
            return inputs[0]
        tid = "S" + str(len(tasks) + 1)
        input_schema = {}
        for i, (parent, out_slot, dtype) in enumerate(inputs, 1):
            slot = "evidence_" + str(i)
            input_schema[slot] = dtype
            edges.append({
                "from": parent, "to": tid,
                "input_slot": slot, "output_slot": out_slot, "type": dtype,
                "kind": "required_data",
                "activation": "producer_completed",
                "handoff_reason": (
                    f"{action}: required upstream {dtype} for the source question."
                ),
            })
        task = {
            "task_id": tid,
            "work_type": work_type,
            "recipe_id": "mmqa_" + ("final_" if terminal else "") + key,
            "goal": (
                FINAL_PREFIX + question if terminal
                else f"{action} to support the information need about {seed_record['topic']}."
            ),
            "deliverable": (
                "A sourced resolution of the original information need."
                if terminal else f"A typed {output} with cross-source references."
            ),
            "outputs": {"result": output},
            "input_schema": input_schema,
            "capability_hints": ["Research", "Analysis"],
            "source_requirements": [],
            "internal_actions": [
                "review completed upstream evidence briefs",
                "bind shared entities and source references",
                action.lower(),
                "produce the typed evidence-backed work product",
            ],
            "grounding_status": "pending_synthetic_evidence",
        }
        tasks.append(task)
        recipes[output + ":" + tid] = task["recipe_id"]
        return tid, "result", output

    def walk(node: dict, depth: int = 0) -> tuple[str, str, str]:
        if depth > 12:
            raise UnsuitableSeed("composition nesting too deep")
        children = node.get("children") or []
        if not children:
            return source_task(_modality_for_leaf(node.get("op", "")),
                               node.get("op", "Source"))
        resolved = [walk(child, depth + 1) for child in children]
        return combine(node.get("op", "Integrate"), resolved)

    tree = seed_record["composition"]
    if tree.get("children"):
        output = walk(tree)
        # A composed MMQA type already defines a joint work product.
        # Convert the terminal work to the explicit answer-oriented contract.
        last = next(t for t in tasks if t["task_id"] == output[0])
        last["work_type"] = "information_need_resolution"
        last["recipe_id"] = "mmqa_final_" + tree["op"].lower()
        last["goal"] = FINAL_PREFIX + question
        last["deliverable"] = "A sourced resolution of the original information need."
        last["outputs"] = {"result": "AnswerReport"}
        # Its consumers do not exist because it is the root of the expression.
        final_id, final_slot, final_type = last["task_id"], "result", "AnswerReport"
    else:
        # Non-composite MMQA questions may still require multiple independent
        # supporting documents; do not invent a second modality.
        for source in available[:4]:
            source_task(source["modality"], "supporting_context")
        if len(tasks) < 2:
            raise UnsuitableSeed("single-source lookup is not a multi-task composition")
        output = combine("integrate", [
            (t["task_id"], "source_findings", t["outputs"]["source_findings"])
            for t in list(tasks)
        ], terminal=True)
        final_id, final_slot, final_type = output
    if not 3 <= len(tasks) <= 9:
        raise UnsuitableSeed("macro task count outside 3..9")
    draft = build_atomic_draft(tasks, edges)
    macro_tasks, coarsening = coarsen(draft)
    compile_contracts(macro_tasks, edges)
    waves = compile_waves(macro_tasks)
    comp = complexity(macro_tasks, waves, coarsening)
    if comp["macro_dependency_depth"] < 2:
        raise UnsuitableSeed("insufficient meaningful dependency depth")
    stats = _topology_metrics(macro_tasks, edges, waves)
    motif = _classify_topology(stats, comp["macro_dependency_depth"])
    task_id = "macro-v45-data-" + digest(run_seed, ordinal, seed_record["record_id"], question)
    kqa_prior, kqa_policy = choose_kqa(
        kqa, "mixed" if stats["joins"] else "chain", rng, ordinal)
    plan = {
        "schema": "maestro.clean_taskfirst.v4",
        "task_id": task_id,
        "scenario": {
            "domain": seed_record["domain_hint"],
            "setting": "reference collection about " + seed_record["topic"],
            "issue": "the information need concerning " + seed_record["topic"],
            "user_role": "researcher",
            "source_kind": "MultiModalQA",
            "topic": seed_record["topic"],
            "source_record_id": seed_record["record_id"],
        },
        "goal": {"intent": seed_record["operation"],
                 "desired_artifact_type": "AnswerReport"},
        "macro_graph": {
            "motif": motif,
            "tasks": macro_tasks,
            "edges": edges,
            "terminal_handoffs": [{
                "from_task": final_id,
                "output_slot": final_slot,
                "type": final_type,
            }],
            "reference_waves": waves,
            "provenance_recipe_routes": recipes,
            "complexity": {**comp, **stats},
            "coarsening": coarsening,
        },
        "source_priors": {
            "KQA": {
                "policy": kqa_policy,
                "signature": kqa_prior["signature"],
                "program": kqa_prior["program"],
                "usage": "internal_reasoning_program_for_future_subagent_trace",
            },
            "MultiModalQA": {
                "policy": "question_and_composition_grounded_in_original_record",
                "record_id": seed_record["record_id"],
                "official_question_type": seed_record["question_type"],
                "composition_family": seed_record["logical_pattern"],
                "composition_tree": seed_record["composition"],
                "source_modalities": seed_record["source_modalities"],
                "supporting_sources": seed_record["supporting_sources"],
                "original_question": question,
                "usage": "scene_need_and_macro_evidence_dependency_authority",
            },
            "YAGO": {
                "matches": seed_record.get("yago_matches", []),
                "usage": "optional_exact_label_context_not_claimed_source_truth",
            },
        },
        "task_plan_provenance": {
            "parent": "mmqa_need_to_macro_workproduct_grammar",
            "ordinal": ordinal,
            "seed": run_seed,
            "question_generated_after_macro_graph": True,
            "question_inputs": "terminal_goal_and_task_contracts",
            "v3_final_gold_answer_not_used": True,
        },
        "scheduler": {
            "max_concurrent_subagents": 10,
            "runtime_variants_generated": False,
        },
    }
    from .v4_taskfirst import task_level_brief, author_question
    plan["question_brief"] = task_level_brief(plan)
    plan["question"] = author_question(plan["question_brief"], ordinal=ordinal, seed=run_seed)
    return plan
