"""Macro-first question planning from KQA/MMQA priors and coherent work goals.

Output is a STRUCTURAL plan. Grounded facts, actual Sub-Agent answers and
summary_traces are built later from per-work evidence; never fabricate them
at the macro-planning stage.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from typing import Any

from envir.schema import digest
from .graph import (MAX_CONCURRENCY, build_atomic_draft, coarsen,
                    compile_contracts, compile_waves, complexity)
from .patterns import (
    CASES, CASE_VARIANTS, DEPENDENCIES, GRAPH_MOTIFS, INTENTS, NODE_SETS,
    WORK_DEFINITIONS, ASSESSMENT_VARIANTS, RESPONSE_VARIANTS, question_brief,
)
from .priors import choose_kqa, choose_mmqa, relation_hints


def _format(text: str, case) -> str:
    return text.format(
        issue=case.issue, material=case.material, records=case.records,
        rule_source=case.rule_source, affected=case.affected,
        focus=case.focus, decision_owner=case.decision_owner,
        setting=case.setting,
    )


def build_plan(*, ordinal: int, seed: int, kqa: list[dict], mmqa: list[dict],
               store=None) -> dict[str, Any]:
    if ordinal < 0:
        raise ValueError("ordinal must be nonnegative")
    rng = random.Random(seed * 100003 + ordinal * 9176)
    # Crossed round-robin: every domain cycles across all macro DAG motifs.
    # Unlike the old pilot, domain does NOT determine the graph or intent.
    index = ordinal % len(CASES)
    iteration = ordinal // len(CASES)
    base = CASES[index]
    motif = GRAPH_MOTIFS[(iteration + 2 * index + seed) % len(GRAPH_MOTIFS)]
    intent = INTENTS[(3 * iteration + index + seed) % len(INTENTS)]
    variants = CASE_VARIANTS[base.domain]
    issue = variants[(iteration + seed // 3) % len(variants)]
    case = replace(base, issue=issue)
    kp, kqa_policy = choose_kqa(kqa, motif, rng, ordinal)
    mp, mmqa_policy = choose_mmqa(mmqa, domain=case.domain,
                                 motif=motif, rng=rng, ordinal=ordinal)
    plan_id = "macro-" + digest(seed, ordinal, case.domain, issue,
                                motif, intent, kp["signature"], mp["motif_id"])
    roles = NODE_SETS[motif]
    names = {role: f"S{i+1}" for i, role in enumerate(roles)}
    public = question_brief(case, intent)
    if "E" in roles:
        public["overall_goal"] = (
            public["overall_goal"].rstrip(".")
            + " Recommend feasible next steps based on the findings."
        )
        public["expected_deliverable"] = "assessment_with_response_options"
    packages = []
    for role in roles:
        spec = WORK_DEFINITIONS[role]
        desc = spec["description"]
        if role == "D":
            desc = ASSESSMENT_VARIANTS[intent]
        elif role == "E":
            desc = RESPONSE_VARIANTS[intent]
        capability_hints = list(spec["capabilities"])
        # MMQA shapes are evidence-role priors: include appropriate modality
        # hints, but never claim that original media files are available.
        if role == "A":
            modality = case.entry_modalities[0]
            if modality in mp.get("source_modalities", ()):
                capability_hints.insert(0, {
                    "image": "ImageAnalysis", "video": "VideoAnalysis",
                    "audio": "AudioAnalysis",
                }.get(modality, "MediaAnalysis"))
            else:
                capability_hints.insert(0, {
                    "image": "ImageAnalysis", "video": "VideoAnalysis",
                    "audio": "AudioAnalysis",
                }.get(modality, "MediaAnalysis"))
        if role == "C" and "table" in mp.get("source_modalities", ()):
            capability_hints.insert(0, "TableAnalysis")
        # Deduplicate without implying required agent/tool identity.
        capability_hints = list(dict.fromkeys(capability_hints))
        packages.append({
            "task_id": names[role],
            "package_role": role,
            "work_type": spec["work_type"],
            "goal": _format(desc, case),
            "deliverable": _format(spec["deliverable"], case),
            "outputs": dict(spec["produces"]),
            "input_schema": {},
            "capability_hints": capability_hints,
            "internal_actions": list(spec["internal_actions"]),
            "grounding_status": "not_grounded",
        })
    by_id = {p["task_id"]: p for p in packages}
    edges = []
    for source, target, input_slot, output_slot in DEPENDENCIES[motif]:
        expected = WORK_DEFINITIONS[source]["produces"][output_slot]
        by_id[names[target]]["input_schema"][input_slot] = expected
        edges.append({
            "from": names[source], "to": names[target],
            "input_slot": input_slot, "output_slot": output_slot,
            "type": expected, "kind": "required_data",
            "activation": "producer_completed",
        })
    atomic = build_atomic_draft(packages, edges)
    tasks, coarsening = coarsen(atomic)
    compile_contracts(tasks, edges)
    waves = compile_waves(tasks, max_concurrency=MAX_CONCURRENCY)
    last = tasks[-1]
    question = public["overall_goal"]
    semantic_hints = relation_hints(
        store, context=f"{case.issue} {case.focus} {case.affected}"
    )
    plan = {
        "schema": "maestro.macro_taskplan.v1",
        "task_id": plan_id,
        "construction_status": "structural_candidate_not_fact_grounded",
        "scenario": {
            "domain": case.domain, "setting": case.setting,
            "issue": case.issue, "user_role": case.decision_owner,
            "case_context": public["situation"],
            "material_roles": public["public_resources"],
            "simulated_media_allowed": True,
        },
        "goal": {
            "intent": intent, "user_request": question,
            "result_type": public["expected_deliverable"],
            "required_context": public["required_context"],
        },
        "macro_graph": {
            "motif": motif, "tasks": tasks,
            "edges": edges,
            "terminal_handoffs": [{
                "from_task": last["task_id"],
                "output_slot": next(iter(last["outputs"])),
                "type": next(iter(last["outputs"].values())),
            }],
            "reference_waves": waves,
        },
        "complexity": complexity(tasks, waves, coarsening),
        "coarsening": coarsening,
        "source_priors": {
            "KQA": {
                "policy": kqa_policy,
                "signature": kp["signature"],
                "example_question": kp["source_question"],
                "full_internal_logical_program": kp["program"],
            },
            "MultiModalQA": {
                "policy": mmqa_policy,
                "motif_id": mp["motif_id"], "record_id": mp.get("record_id"),
                "original_question_hint": mp["question_hint"],
                "official_question_type": mp.get("question_type", ""),
                "composition_family": mp["logical_pattern"],
                "source_modalities": mp["source_modalities"],
                "synthetic_extension": True,
            },
            "YAGO": {
                "policy": "observed_relation_schema_priors_only",
                "matching_relation_hints": semantic_hints,
                "fact_grounding_applied": False,
            },
        },
        "scheduler": {
            "max_concurrent_subagents": MAX_CONCURRENCY,
            "environment_profile": "fixed_default",
            "runtime_variants_generated": False,
        },
        "question_brief": {
            "situation": public["situation"],
            "user_role": case.decision_owner,
            "final_goal": question,
            "result_type": public["expected_deliverable"],
            "public_resources": public["public_resources"],
            "policy": "public_user_need_only_no_internal_subtask_steps",
        },
        "question": question,
        "public": {
            "task_id": plan_id, "question": question,
            "resource_descriptions": public["public_resources"],
            "notice": "Resource descriptions are design placeholders, not executable files.",
            "construction_status": "unrealized",
        },
    }
    from .validation import validate_plan
    issues = validate_plan(plan)
    if issues:
        raise ValueError(f"macro-plan certification failed: {issues}")
    return plan


def build_batch(*, kqa: list[dict], mmqa: list[dict], target: int = 100,
                seed: int = 17, store=None) -> tuple[list[dict], dict]:
    if target < 1 or not kqa or not mmqa:
        raise ValueError("positive target and nonempty KQA/MMQA prior banks required")
    plans = [build_plan(ordinal=i, seed=seed, kqa=kqa, mmqa=mmqa,
                        store=store) for i in range(target)]
    domain = Counter(p["scenario"]["domain"] for p in plans)
    motif = Counter(p["macro_graph"]["motif"] for p in plans)
    intent = Counter(p["goal"]["intent"] for p in plans)
    pairs = Counter((p["scenario"]["domain"], p["macro_graph"]["motif"])
                    for p in plans)
    patterns_per_domain = {
        d: len({m for (dd, m), n in pairs.items() if dd == d and n})
        for d in domain
    }
    uniq = len({p["question"] for p in plans})
    stats = {
        "generated": len(plans),
        "unique_question_texts": uniq,
        "question_duplicate_count": len(plans) - uniq,
        "domains": dict(domain),
        "macro_graph_patterns": dict(motif),
        "intents": dict(intent),
        "domain_graph_pairs": len(pairs),
        "patterns_per_domain": patterns_per_domain,
        "macro_dependency_depth": dict(Counter(
            p["complexity"]["macro_dependency_depth"] for p in plans)),
        "fixed_concurrency_limit": MAX_CONCURRENCY,
        "teacher_feedback_generated": False,
        "world_facts_generated": False,
        "certification": "macro_topology_and_typed_dependencies_only",
    }
    return plans, stats
