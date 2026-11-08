"""Task-First Work-Product Grammar grounded in MultiModalQA source compositions.

A source leaf is one delegable evidence-investigation package (which may
review multiple source documents). Macro work is separate from micro KoPL
operations. Cross-source binding, comparison/intersection and independent
claim verification are typed handoffs, rather than filler DAG nodes.
"""
from __future__ import annotations

import random
from collections import Counter

from envir.schema import digest
from .graph import build_atomic_draft, coarsen, compile_contracts, compile_waves, complexity
from .priors import choose_kqa

FINAL_PREFIX = "Resolve the following source-scoped information need: "
MODALITY_TYPES = {
    "table": ("TableEvidence", "TabularEvidenceInvestigation", "TableAnalysis"),
    "image": ("VisualEvidence", "VisualEvidenceInvestigation", "ImageAnalysis"),
    "text": ("TextEvidence", "DocumentEvidenceInvestigation", "DocumentAnalysis"),
}
OPERATORS = {
    "compose": ("reference_chain_resolution", "LinkedEvidence",
                "Resolve entity/reference links so results from one source select the relevant content in another."),
    "compare": ("comparative_assessment", "ComparativeFinding",
                "Compare scoped candidate attributes under the question's comparison criterion."),
    "intersect": ("record_matching", "MatchedCandidates",
                  "Match candidate entities shared by both evidence branches."),
    "intersection": ("record_matching", "MatchedCandidates",
                     "Match candidate entities shared by both evidence branches."),
    "union": ("candidate_set_union", "CombinedCandidates",
              "Combine candidate sets while recording the source of each member."),
}
ASSESSMENTS = {
    "lookup": ("attribute_verification", "VerifiedAttributes",
               "Check the requested attributes of source-linked candidates."),
    "composition": ("reference_chain_verification", "VerifiedReferences",
                    "Verify the entity/reference chain used by the composed question."),
    "comparison": ("comparative_assessment", "ComparativeFinding",
                   "Compare the normalized attributes of the supported candidates."),
    "intersection": ("record_matching", "MatchedCandidates",
                     "Find common candidates satisfying both source conditions."),
    "aggregation": ("candidate_aggregation", "AggregateFinding",
                    "Count or aggregate candidates satisfying the question criteria."),
    "temporal": ("timeline_reconstruction", "ChronologicalFinding",
                 "Order the source-backed candidate dates or events."),
}


class UnsuitableSeed(ValueError):
    """Cannot create a well-supported multi-work-package task graph."""


def _modality(op: str) -> str:
    label = str(op).lower()
    for modality in ("table", "image", "text"):
        if modality in label:
            return modality
    return "text"


def _leaves(tree: dict) -> list[str]:
    children = tree.get("children") or []
    if not children:
        return [str(tree.get("op", "TextQ"))]
    result = []
    for child in children:
        result.extend(_leaves(child))
    return result


def _suboperations(tree: dict) -> int:
    return int(bool(tree.get("children"))) + sum(
        _suboperations(x) for x in tree.get("children", []))


def _source_key(source: dict) -> tuple:
    return (source["modality"], source.get("doc_id"))


def _unique_sources(seed: dict) -> list[dict]:
    unique, seen = [], set()
    for item in seed.get("supporting_sources", []):
        modality = item.get("modality")
        if modality not in MODALITY_TYPES:
            continue
        key = _source_key(item)
        if key not in seen:
            unique.append(dict(item))
            seen.add(key)
    for modality in seed.get("source_modalities", []):
        if modality not in MODALITY_TYPES:
            continue
        if any(row["modality"] == modality for row in unique):
            continue
        unique.append({
            "modality": modality, "doc_id": None,
            "provenance": "mmqa_modality_role_only",
        })
    return unique


def _allocate_sources(seed: dict) -> tuple[list[list[dict]], dict]:
    """Every MMQA supporting source goes to exactly one root investigation.

    For composite types each logical leaf receives the appropriate modality;
    additional documents are bundled into the related leaf's work package.
    For a noncomposite multi-source type, independent document packages are
    used (up to four), with remaining documents fully preserved in a package.
    """
    available = _unique_sources(seed)
    if len(available) < 2:
        raise UnsuitableSeed("at least two distinct supporting sources required")
    tree = seed["composition"]
    leaf_names = _leaves(tree)
    if len(leaf_names) > 5 or _suboperations(tree) > 4:
        raise UnsuitableSeed("source composition exceeds macro-work budget")
    buckets = [[] for _ in leaf_names] if tree.get("children") else []
    consumed = set()
    if buckets:
        for i, label in enumerate(leaf_names):
            modality = _modality(label)
            possible = next((source for source in available
                             if source["modality"] == modality
                             and _source_key(source) not in consumed), None)
            if possible is None:
                possible = next((source for source in available
                                 if source["modality"] == modality), None)
            if possible is None:
                raise UnsuitableSeed("MMQA type has no source role: " + modality)
            buckets[i].append(possible)
            consumed.add(_source_key(possible))
    else:
        max_roots = min(4, len(available))
        buckets = [[item] for item in available[:max_roots]]
        consumed.update(_source_key(item) for item in available[:max_roots])
    for item in available:
        if _source_key(item) in consumed:
            continue
        compatible = [i for i, bucket in enumerate(buckets)
                      if bucket[0]["modality"] == item["modality"]]
        if not compatible:
            # A missing modality leaf must remain visible as a distinct root,
            # not silently dropped or mislabelled.
            if len(buckets) >= 5:
                raise UnsuitableSeed("too many distinct source-role groups")
            buckets.append([item])
        else:
            idx = min(compatible, key=lambda i: (len(buckets[i]), i))
            buckets[idx].append(item)
        consumed.add(_source_key(item))
    observed = {_source_key(source) for bucket in buckets for source in bucket}
    expected = {_source_key(source) for source in available}
    if observed != expected:
        raise UnsuitableSeed("source allocation does not cover MMQA support")
    return buckets, {"expected": len(expected), "assigned": len(observed),
                     "source_keys": [
                         {"modality": key[0], "doc_id": key[1]} for key in sorted(
                             expected, key=lambda item: (item[0], str(item[1])))
                     ]}


def _requirement(source: dict, seed: dict) -> dict:
    modality, doc_id = _source_key(source)
    role = {"table": "records", "image": "material", "text": "context"}[modality]
    return {
        "role": role,
        "description": (f"MultiModalQA {modality} reference {doc_id}" if doc_id
                        else f"simulated {modality} role for {seed['topic']}"),
        "access": "synthetic_observation_only",
        "candidate_modalities": [modality],
        "source_origin": source["provenance"],
        "original_doc_id": doc_id,
        "source_leaf": "supporting_context",
        "synthetic": True,
    }


def _topology_metrics(tasks: list[dict], edges: list[dict],
                      waves: list[list[str]]) -> dict:
    degree = Counter(edge["from"] for edge in edges)
    return {
        "roots": sum(not task["depends_on"] for task in tasks),
        "joins": sum(len(task["depends_on"]) > 1 for task in tasks),
        "fanouts": sum(degree[task["task_id"]] > 1 for task in tasks),
        "width": max(map(len, waves), default=0),
    }


def _topology(stats: dict, depth: int) -> str:
    if stats["fanouts"] and stats["joins"]:
        return "fork_join"
    if stats["joins"] and depth >= 4:
        return "staged_join"
    if stats["joins"]:
        return "mixed_join"
    return "chain"


def compose_seed_taskfirst(*, seed_record: dict, ordinal: int, run_seed: int,
                           kqa: list[dict]) -> dict:
    rng = random.Random(run_seed * 1000003 + ordinal * 99017)
    question = seed_record["question"].strip()
    if not question:
        raise UnsuitableSeed("empty original information need")
    buckets, source_coverage = _allocate_sources(seed_record)
    tree = seed_record["composition"]
    tasks, edges, recipes = [], [], {}
    root_outputs = []

    def add(work_type: str, artifact: str, goal: str, deliverable: str,
            inputs: list[tuple] | None = None,
            requirements: list[dict] | None = None,
            actions: list[str] | None = None,
            op: str | None = None) -> tuple[str, str, str]:
        deps = inputs or []
        tid = "S" + str(len(tasks) + 1)
        schema = {}
        for i, (parent, out_slot, kind) in enumerate(deps, 1):
            slot = "evidence_" + str(i)
            schema[slot] = kind
            edges.append({
                "from": parent, "to": tid, "input_slot": slot,
                "output_slot": out_slot, "type": kind,
                "kind": "required_data", "activation": "producer_completed",
                "handoff_reason": (
                    f"{work_type} requires the {kind} produced by {parent}."
                ),
            })
        output_slot = "source_findings" if not deps else "result"
        result = (tid, output_slot, artifact)
        task = {
            "task_id": tid,
            "work_type": work_type,
            "recipe_id": "mmqa_work_" + work_type,
            "goal": goal,
            "deliverable": deliverable,
            "outputs": {output_slot: artifact},
            "input_schema": schema,
            "capability_hints": ["Research", "EvidenceAnalysis"],
            "source_requirements": requirements or [],
            "internal_actions": actions or [
                "review source-scoped input contracts",
                "align the requested target and its evidence",
                "perform the named work-product operation",
                "deliver the typed result with references",
            ],
            "operation_spec": {"op": op or work_type,
                               "question_operation": seed_record["operation"],
                               "question": question,
                               "topic": seed_record["topic"]},
            "grounding_status": "pending_synthetic_evidence",
        }
        tasks.append(task)
        recipes[tid] = task["recipe_id"]
        return result

    for bucket in buckets:
        mod = bucket[0]["modality"]
        kind, name, capability = MODALITY_TYPES[mod]
        requirements = [_requirement(source, seed_record) for source in bucket]
        root_outputs.append(add(
            "source_investigation_" + mod, kind,
            f"Inspect the {len(bucket)} {mod} source reference(s) for relevant "
            f"entities, attributes and candidate records concerning {seed_record['topic']}.",
            f"A source-linked {mod} findings package with observed candidate fields.",
            requirements=requirements,
            actions=[
                "identify the source documents and access scope",
                "extract the candidate entities and requested attributes",
                "record each original reference and its simulated observation",
                "compile the typed source evidence package",
            ],
            op="source_extract_" + mod,
        ))

    leaf_index = 0

    def walk(node: dict, level: int = 0) -> tuple[str, str, str]:
        nonlocal leaf_index
        if level > 12:
            raise UnsuitableSeed("composition nesting too deep")
        children = node.get("children") or []
        if not children:
            idx = leaf_index
            leaf_index += 1
            return root_outputs[idx]
        upstream = [walk(child, level + 1) for child in children]
        op = str(node.get("op", "Compose")).lower()
        recipe, result_type, description = OPERATORS.get(op, (
            "cross_source_evidence_alignment", "AlignedEvidence",
            "Align the entities and requested attributes across the source branches."
        ))
        return add(
            recipe, result_type,
            description + f" Target information need: {question}",
            f"{result_type} with explicit upstream and original-source lineage.",
            inputs=upstream, op=op,
        )

    if tree.get("children"):
        main = walk(tree)
        # If the composition metadata lists a supporting document not
        # referenced by a leaf, all such records are covered by a root bucket
        # and explicitly handed into the verification work product.
        verification_inputs = [main] + root_outputs[len(_leaves(tree)):]
        check = add(
            "evidence_consistency_review", "VerifiedEvidence",
            "Verify the source bindings and candidate relationships supporting " + question,
            "A checked evidence chain with explicit limitations and provenance.",
            inputs=verification_inputs, op="verify_evidence",
        )
        final_inputs = [main, check]
    else:
        # The work types correspond to distinct useful operations: map source
        # candidates across records; independently evaluate the asked property
        # or relation; finally reconcile them for the answer request.
        align = add(
            "entity_reference_alignment", "AlignedCandidates",
            "Resolve equivalent candidate references across the supporting sources "
            "without assuming unrelated names are identical.",
            "A candidate alignment with a source-to-entity lookup index.",
            inputs=root_outputs, op="align_candidates",
        )
        family, assessment_type, operation = ASSESSMENTS.get(
            seed_record["operation"], ASSESSMENTS["lookup"])
        assessment = add(
            family, assessment_type,
            operation + f" Target information need: {question}",
            f"{assessment_type} recording evaluated candidates and evidence links.",
            inputs=root_outputs, op=seed_record["operation"],
        )
        final_inputs = [align, assessment]
    final = add(
        "answer_evidence_synthesis", "AnswerReport",
        FINAL_PREFIX + question,
        "An evidence-backed response to the original information need, "
        "explicitly scoped to the available synthetic observations.",
        inputs=final_inputs, op="answer_synthesis",
    )
    if not 3 <= len(tasks) <= 9:
        raise UnsuitableSeed("macro task count outside 3..9")
    draft = build_atomic_draft(tasks, edges)
    macro_tasks, coarsening = coarsen(draft)
    compile_contracts(macro_tasks, edges)
    waves = compile_waves(macro_tasks)
    comp = complexity(macro_tasks, waves, coarsening)
    if comp["macro_dependency_depth"] < 2:
        raise UnsuitableSeed("insufficient multi-stage dependency depth")
    metrics = _topology_metrics(macro_tasks, edges, waves)
    task_uid = "macro-v46-data-" + digest(
        run_seed, ordinal, seed_record["record_id"], question)
    kqa_prior, kqa_policy = choose_kqa(
        kqa, "mixed" if metrics["joins"] else "chain", rng, ordinal)
    plan = {
        "schema": "maestro.clean_taskfirst.v4",
        "task_id": task_uid,
        "scenario": {
            "domain": seed_record["domain_hint"],
            "setting": "MultiModalQA reference collection",
            "issue": "source-grounded information need",
            "user_role": "researcher",
            "source_kind": "MultiModalQA",
            "topic": seed_record["topic"],
            "source_record_id": seed_record["record_id"],
        },
        "goal": {
            "intent": seed_record["operation"],
            "desired_artifact_type": "AnswerReport",
        },
        "macro_graph": {
            "motif": _topology(metrics, comp["macro_dependency_depth"]),
            "tasks": macro_tasks,
            "edges": edges,
            "terminal_handoffs": [{
                "from_task": final[0], "output_slot": final[1], "type": final[2],
            }],
            "reference_waves": waves,
            "provenance_recipe_routes": recipes,
            "complexity": {**comp, **metrics},
            "coarsening": coarsening,
            "source_coverage": source_coverage,
        },
        "source_priors": {
            "KQA": {
                "policy": kqa_policy, "signature": kqa_prior["signature"],
                "program": kqa_prior["program"],
                "usage": "unexecuted_internal_reasoning_prior",
            },
            "MultiModalQA": {
                "policy": "original_question_and_all_supporting_sources",
                "record_id": seed_record["record_id"],
                "official_question_type": seed_record["question_type"],
                "composition_family": seed_record["logical_pattern"],
                "composition_tree": seed_record["composition"],
                "source_modalities": seed_record["source_modalities"],
                "supporting_sources": seed_record["supporting_sources"],
                "original_question": question,
                "usage": "scenario_need_and_macro_evidence_dependency_authority",
            },
            "YAGO": {
                "matches": seed_record.get("yago_matches", []),
                "usage": "optional_exact_label_context_not_claimed_source_truth",
            },
        },
        "task_plan_provenance": {
            "parent": "mmqa_data_driven_typed_workproduct_grammar_v46",
            "ordinal": ordinal, "seed": run_seed,
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
    plan["question"] = author_question(
        plan["question_brief"], ordinal=ordinal, seed=run_seed)
    return plan
