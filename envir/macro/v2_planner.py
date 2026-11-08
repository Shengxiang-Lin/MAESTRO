"""Semantic macro work composition. The DAG is an OUTPUT of task design.

Unlike v1: neither goal intent nor scenario domain selects a fixed topology.
A backward planner recursively chooses alternative providers of typed
deliverables; shared prerequisite artifacts are memoized. Internal KoPL/media
operations never turn into separately delegated macro tasks.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import replace
from typing import Any

from envir.schema import digest
from .graph import build_atomic_draft, coarsen, compile_contracts, compile_waves, complexity
from .patterns import CASES, CASE_VARIANTS
from .priors import choose_kqa, choose_mmqa, relation_hints
from .v2_workbank import (
    BY_OUTPUT, GOAL_TYPES, OUTPUT_FIELDS, ROOT_SOURCE_DESCRIPTIONS, Recipe,
)

STYLE_FAMILIES = (
    "concise_compositional",
    "contextual_investigation",
    "explicit_result_constraints",
    "cross_source_explanation",
)

# These are compositional question grammars, not copies of OmniGAIA questions.
# OmniGAIA-style user context, relational constraints and final deliverable
# are retained without revealing a tool route or fake observations.
INTENT_REQUESTS = {
    "impact": (
        "which {affected} are actually affected by {issue}, and what consequences can be supported?",
        "separate the affected from unaffected {affected}, and explain the practical implications of {issue}.",
        "identify the affected {affected} and explain which observations make those impacts credible.",
    ),
    "compliance": (
        "does the situation surrounding {issue} meet the relevant requirements, including scope and timing?",
        "which aspects of {issue} comply with the applicable requirements and which do not?",
        "give a requirements-based determination for {issue}, distinguishing applicable conditions and exceptions.",
    ),
    "reconciliation": (
        "which claims about {issue} are corroborated, and which discrepancies remain?",
        "establish the most defensible account of {issue}, distinguishing agreement from unresolved differences.",
        "identify the facts about {issue} that can be jointly supported and isolate the disputed details.",
    ),
    "mitigation": (
        "which practical responses to {issue} should be considered, and what limits each option?",
        "develop feasible options to address {issue}, explaining their main trade-offs.",
        "recommend practical responses to {issue} while distinguishing immediate measures from longer-term choices.",
    ),
    "prioritization": (
        "which {affected} or corrective actions should be handled first following {issue}, and on what basis?",
        "rank the cases requiring attention after {issue}, accounting for urgency and operational constraints.",
        "explain the ordering of the most urgent {affected} in the response to {issue}.",
    ),
    "comparison": (
        "how do the competing interpretations or responses to {issue} compare, and which is better supported?",
        "compare the relevant findings about {issue} and explain the decisive differences.",
        "explain which interpretation of {issue} is strongest after comparing the relevant evidence and constraints.",
    ),
    "feasibility": (
        "which proposed actions for {issue} can realistically be implemented under applicable constraints?",
        "evaluate whether the potential responses to {issue} are feasible and what restrictions matter.",
        "identify feasible next actions for {issue}, and state which constraints rule out other options.",
    ),
    "timeline": (
        "what happened in what order during {issue}, and where do the available timelines differ?",
        "reconstruct the chronology of {issue} and identify any timing discrepancies.",
        "establish the order of events during {issue}, including dates that materially change the conclusion.",
    ),
}

FINAL_RECIPE_QUESTIONS = {
    "compare_responses": "compare the possible responses and identify which options best fit the available resources.",
    "compare_timeline_rules": "compare the event chronology against the requirements in force at the relevant times.",
    "compare_impacts_compliance": "compare the operational impact with the applicable-rule findings and explain where their conclusions differ.",
    "feasibility_options_capacity": "determine which proposed responses are feasible within the available operational capacity.",
    "feasibility_options_rules": "determine which proposed responses comply with the applicable requirements and therefore remain feasible.",
    "feasibility_priorities_rules": "determine whether the proposed priority order can be executed under the applicable requirements.",
    "comparison_from_timeline": "compare the recorded sequence with the time-based rules and explain any important differences.",
    "feasibility_from_options": "evaluate which proposed responses could be implemented under the documented resource constraints.",
}


def _render(template: str, case) -> str:
    return template.format(
        issue=case.issue, setting=case.setting, material=case.material,
        records=case.records, rule_source=case.rule_source,
        affected=case.affected, decision_owner=case.decision_owner,
        focus=case.focus,
    )


def _weighted_order(recipes: tuple[Recipe, ...], rng: random.Random,
                    mmqa_pattern: str, attempt: int,
                    structural_preference: str) -> list[Recipe]:
    candidates = []
    for r in recipes:
        weight = 1.0
        if mmqa_pattern == "intersection" and (
            "reconcil" in r.name or "match" in r.name or "compare" in r.name
        ):
            weight *= 2.2
        if mmqa_pattern == "composition" and (
            "timeline" in r.name or "profile" in r.name
        ):
            weight *= 1.5
        if mmqa_pattern == "comparison" and (
            "compare" in r.name or "compliance" in r.name
        ):
            weight *= 1.6
        # A soft structural preference varies the macro-scheduling
        # opportunities independently of domain and answer intent.
        # Actual graph edges ALWAYS come from required work products.
        if structural_preference == "serial" and len(r.inputs) == 1:
            weight *= 18.0
        elif structural_preference == "branch" and len(r.inputs) >= 2:
            weight *= 5.0
        # Draw a random priority; sorting yields a weighted permutation,
        # not a deterministic mapping from MMQA label to a macro graph.
        key = rng.random() ** (1.0 / weight)
        candidates.append((key, r.name, r))
    candidates.sort(reverse=True)
    ordered = [item[-1] for item in candidates]
    # Retry attempts explore alternative producers even with the same motif.
    if attempt and len(ordered) > 1:
        shift = attempt % len(ordered)
        ordered = ordered[shift:] + ordered[:shift]
    return ordered


def _build_work_graph(case, intent: str, mmqa_pattern: str,
                      rng: random.Random, attempt: int,
                      structural_preference: str) -> tuple[list[dict], list[dict], dict, dict]:
    selected: dict[str, dict] = {}
    by_type: dict[str, str] = {}
    edges: list[dict] = []
    selected_recipe: dict[str, str] = {}

    def produce(result_type: str) -> str:
        if result_type in by_type:
            return by_type[result_type]
        options = BY_OUTPUT[result_type]
        recipe = _weighted_order(options, rng, mmqa_pattern, attempt, structural_preference)[0]
        links = []
        for slot, required_type, rationale in recipe.inputs:
            parent_id = produce(required_type)
            links.append((slot, required_type, rationale, parent_id))
        task_id = f"S{len(selected) + 1}"
        # IDs are allocated AFTER prerequisites, so they always form a
        # topological order. This is a teacher reference, not a Main mandate.
        by_type[result_type] = task_id
        selected_recipe[result_type] = recipe.name
        own_output = OUTPUT_FIELDS[result_type]
        modalities = case.entry_modalities
        cap = list(recipe.capabilities)
        if recipe.source_role == "material":
            primary = modalities[0]
            cap.insert(0, {
                "image": "ImageAnalysis",
                "video": "VideoAnalysis",
                "audio": "AudioAnalysis",
                "text": "DocumentAnalysis",
            }.get(primary, "MediaAnalysis"))
        if recipe.source_role == "records":
            cap.insert(0, "TableAnalysis")
        cap = list(dict.fromkeys(cap))
        if recipe.source_role:
            desc = _render(ROOT_SOURCE_DESCRIPTIONS[recipe.source_role], case)
            sources = [{
                "role": recipe.source_role,
                "description": desc,
                "access": "not_instantiated",
                "candidate_modalities": list(modalities if recipe.source_role == "material"
                                              else ("table", "text", "search")),
                "synthetic": True,
            }]
        else:
            sources = []
        selected[task_id] = {
            "task_id": task_id,
            "work_type": recipe.work_type,
            "recipe_id": recipe.name,
            "goal": _render(recipe.goal, case),
            "deliverable": _render(recipe.deliverable, case),
            "outputs": {own_output: result_type},
            "input_schema": {slot: required_type for slot, required_type, _, _ in links},
            "capability_hints": cap,
            "source_requirements": sources,
            "internal_actions": list(recipe.internal_actions),
            "grounding_status": "not_grounded",
        }
        for slot, required_type, rationale, parent_id in links:
            output_slot = OUTPUT_FIELDS[required_type]
            edges.append({
                "from": parent_id, "to": task_id,
                "input_slot": slot,
                "output_slot": output_slot,
                "type": required_type,
                "kind": "required_data",
                "activation": "producer_completed",
                "handoff_reason": rationale,
            })
        return task_id

    final_type = GOAL_TYPES[intent]
    final_id = produce(final_type)
    packages = list(selected.values())
    # Every selected macro task must have a meaningful contribution to the
    # final artifact; root tasks are NOT randomly attached.
    final_task = selected[final_id]
    return packages, edges, {
        "from_task": final_id,
        "output_slot": OUTPUT_FIELDS[final_type],
        "type": final_type,
    }, selected_recipe


def _topology_metrics(tasks: list[dict], edges: list[dict], waves: list[list[str]]):
    outdegree = Counter(e["from"] for e in edges)
    roots = sum(not t["depends_on"] for t in tasks)
    joins = sum(len(t["depends_on"]) > 1 for t in tasks)
    fanouts = sum(outdegree[t["task_id"]] > 1 for t in tasks)
    return {"roots": roots, "joins": joins, "fanouts": fanouts,
            "width": max(map(len, waves), default=0)}


def _classify_topology(metrics, depth):
    if metrics["roots"] == 1 and metrics["joins"] == 0:
        return "chain"
    if depth == 2 and metrics["roots"] >= 2:
        return "parallel_join"
    if metrics["fanouts"] and metrics["joins"]:
        return "fork_join"
    if metrics["joins"] and depth >= 4:
        return "staged_join"
    if metrics["joins"]:
        return "mixed_join"
    return "progressive"


def _question(case, intent, recipe_name, source_desc, style, variant):
    ask = FINAL_RECIPE_QUESTIONS.get(recipe_name)
    if ask is None:
        ask = INTENT_REQUESTS[intent][variant % len(INTENT_REQUESTS[intent])]
    ask = _render(ask, case)
    ask = ask.replace(case.issue, "this situation")
    resources = list(dict.fromkeys(source_desc))
    if len(resources) > 3:
        evidence = ", ".join(resources[:2]) + ", and other relevant supporting records"
    elif len(resources) == 3:
        evidence = ", ".join(resources[:2]) + ", and " + resources[2]
    elif len(resources) == 2:
        evidence = " and ".join(resources)
    elif resources:
        evidence = resources[0]
    else:
        evidence = "the available case documentation"
    if style == "concise_compositional":
        question = (
            f"For {case.issue} at the {case.setting}, using {evidence}, "
            f"{ask}"
        )
    elif style == "contextual_investigation":
        question = (
            f"I'm reviewing {case.issue} at the {case.setting}. "
            f"Using {evidence}, {ask}"
        )
    elif style == "explicit_result_constraints":
        question = (
            f"The {case.decision_owner} needs a defensible assessment of "
            f"{case.issue} at the {case.setting}. Based on {evidence}, "
            f"{ask} Include the key evidence and any relevant limitations."
        )
    elif style == "cross_source_explanation":
        question = (
            f"Information about {case.issue} at the {case.setting} comes from "
            f"{evidence}. {ask.capitalize()} Make clear how the sources support the conclusion."
        )
    else:
        raise ValueError("unsupported question style")
    return question[0].upper() + question[1:]


def build_v2_plan(*, ordinal: int, seed: int, kqa: list[dict],
                  mmqa: list[dict], store=None, omni_styles: list[dict] | None = None) -> dict:
    if ordinal < 0:
        raise ValueError("ordinal must be nonnegative")
    rng = random.Random(ordinal * 99017 + seed * 1000003)
    index = ordinal % len(CASES)
    iteration = ordinal // len(CASES)
    original = CASES[index]
    issues = CASE_VARIANTS[original.domain]
    case = replace(original, issue=issues[(iteration * 3 + seed) % len(issues)])
    intents = tuple(GOAL_TYPES)
    intent = intents[(iteration + index * 3 + seed) % len(intents)]
    source_motif_hint = rng.choice(("chain", "parallel_join", "fork_join", "mixed", "staged"))
    mmqa_prior, mmqa_alignment = choose_mmqa(
        mmqa, domain=case.domain, motif=source_motif_hint, rng=rng,
        ordinal=ordinal,
    )
    mmqa_pattern = mmqa_prior["logical_pattern"]
    # Preference controls the availability of sequential vs join work
    # routes, but never injects dependency edges or fixes a DAG template.
    structural_preference = ("serial", "branch", "balanced")[
        (iteration + index * 5 + seed) % 3
    ]
    candidate = None
    for attempt in range(60):
        # Fresh local randomness explores alternate producers and shared
        # prerequisite structure; no fixed DAG is selected or forced.
        options_rng = random.Random(rng.randrange(1 << 40) + attempt)
        packages, edges, terminal, provenance = _build_work_graph(
            case, intent, mmqa_pattern, options_rng, attempt,
            structural_preference)
        count = len(packages)
        if not 3 <= count <= 9:
            continue
        atomic = build_atomic_draft(packages, edges)
        tasks, coarsening = coarsen(atomic)
        compile_contracts(tasks, edges)
        waves = compile_waves(tasks)
        cm = complexity(tasks, waves, coarsening)
        metrics = _topology_metrics(tasks, edges, waves)
        # Real one-root serial chains are valuable orchestration cases.
        # Unlike the v1 fork-join bias, do not reject them merely because
        # they lack parallelism; include a diversified mixture.
        if cm["macro_dependency_depth"] < 2:
            continue
        candidate = (tasks, edges, terminal, provenance, coarsening, waves, cm, metrics)
        break
    if candidate is None:
        raise ValueError("cannot construct a complete nontrivial macro plan")
    tasks, edges, terminal, recipe_choices, coarsening, waves, cm, met = candidate
    topology = _classify_topology(met, cm["macro_dependency_depth"])
    kqa_prior, kqa_policy = choose_kqa(
        kqa, source_motif_hint, rng, ordinal)
    # Genuine KQA fine-grained programs remain priors for execution inside
    # a macro task, not one-node-per-agent delegation boundaries.
    thinking_tasks = [t["task_id"] for t in tasks if t["input_schema"]]
    kqa_application = thinking_tasks[-1] if thinking_tasks else tasks[-1]["task_id"]
    styles = STYLE_FAMILIES
    chosen_style = styles[(iteration * 3 + index + seed) % len(styles)]
    omni_provenance = {"policy": "published_question_styles_not_specific_examples"}
    if omni_styles:
        selected = omni_styles[rng.randrange(len(omni_styles))]
        chosen_style = selected["style"]
        omni_provenance = {
            "policy": "real_dataset_question_style_only",
            "source_record_id": selected["record_id"],
            "source_difficulty": selected.get("difficulty"),
        }
    root_sources = [
        s for task in tasks for s in task.get("source_requirements", ())
    ]
    descriptions = [s["description"] for s in root_sources]
    # Case issues repeat across an 11-cycle pilot; avoid repeating the
    # exact same wording for the same issue/intent by rotating the speech
    # act across repeated occurrences, not by appending meaningless IDs.
    wording_variant = (iteration // len(issues) + iteration + index) % 3
    question = _question(case, intent, recipe_choices[terminal["type"]],
                         descriptions, chosen_style, wording_variant)
    plan_id = "macro-v2-" + digest(
        seed, ordinal, case.domain, intent, case.issue,
        sorted(recipe_choices.items()), mmqa_prior["motif_id"])
    world_inputs = [
        {"role": s["role"], "description": s["description"],
         "candidate_modalities": s["candidate_modalities"]}
        for s in root_sources
    ]
    plan = {
        "schema": "maestro.macro_taskplan.v2",
        "task_id": plan_id,
        "construction_status": "structural_candidate_not_fact_grounded",
        "scenario": {
            "domain": case.domain, "setting": case.setting,
            "issue": case.issue, "user_role": case.decision_owner,
            "simulated_media_allowed": True,
            "evidence_requirements": world_inputs,
        },
        "goal": {
            "intent": intent, "user_request": question,
            "result_type": terminal["type"],
            "required_context": case.focus,
            "target_deliverable": tasks[-1]["deliverable"],
            "question_style": chosen_style,
            "structural_preference": structural_preference,
        },
        "macro_graph": {
            "motif": topology,
            "tasks": tasks,
            "edges": edges,
            "terminal_handoffs": [terminal],
            "reference_waves": waves,
            "provenance_recipe_routes": recipe_choices,
        },
        "complexity": {**cm, **met},
        "coarsening": coarsening,
        "source_priors": {
            "KQA": {
                "policy": kqa_policy,
                "signature": kqa_prior["signature"],
                "internal_reasoning_task_id": kqa_application,
                "full_internal_logical_program": kqa_prior["program"],
            },
            "MultiModalQA": {
                "policy": mmqa_alignment,
                "motif_id": mmqa_prior["motif_id"],
                "record_id": mmqa_prior["record_id"],
                "original_question_hint": mmqa_prior["question_hint"],
                "official_question_type": mmqa_prior["question_type"],
                "composition_family": mmqa_pattern,
                "source_modalities": mmqa_prior["source_modalities"],
                "no_media_truth_claim": True,
            },
            "OmniGAIA": omni_provenance,
            "YAGO": {
                "policy": "observed_relation_schema_priors_only",
                "matching_relation_hints": relation_hints(
                    store, context=f"{case.issue} {case.focus} {case.affected}"),
                "fact_grounding_applied": False,
            },
        },
        "scheduler": {
            "max_concurrent_subagents": 10,
            "environment_profile": "fixed_default",
            "runtime_variants_generated": False,
        },
        "question_brief": {
            "situation": f"{case.decision_owner} at {case.setting} is reviewing {case.issue}.",
            "user_role": case.decision_owner,
            "final_goal": question,
            "result_type": terminal["type"],
            "evidence_requirement_descriptions": descriptions,
            "source_requirement_types": [s["role"] for s in root_sources],
            "policy": "user_need_and_resource_context_only_no_internal_steps",
        },
        "question": question,
        "public": {
            "task_id": plan_id,
            "question": question,
            "resource_descriptions": world_inputs,
            "notice": "Resources have not yet been realized into simulated documents or media.",
            "construction_status": "unrealized",
        },
    }
    from .v2_validation import validate_v2
    issues_found = validate_v2(plan)
    if issues_found:
        raise ValueError("invalid generated v2 TaskPlan: " + str(issues_found))
    return plan


def build_v2_batch(*, kqa: list[dict], mmqa: list[dict], target: int = 100,
                   seed: int = 17, store=None, omni_styles: list[dict] | None = None) -> tuple[list[dict], dict]:
    if target <= 0 or not kqa or not mmqa:
        raise ValueError("need target>0, nonempty original KQA/MMQA priors")
    plans = [build_v2_plan(ordinal=i, seed=seed, kqa=kqa, mmqa=mmqa,
                           store=store, omni_styles=omni_styles)
             for i in range(target)]
    intents = Counter(p["goal"]["intent"] for p in plans)
    topology = Counter(p["macro_graph"]["motif"] for p in plans)
    pairs = Counter((p["goal"]["intent"], p["macro_graph"]["motif"]) for p in plans)
    work_types = Counter(
        task["work_type"] for p in plans for task in p["macro_graph"]["tasks"]
    )
    signature = lambda p: (
        tuple(t["recipe_id"] for t in p["macro_graph"]["tasks"]),
        tuple(sorted((e["from"], e["to"], e["type"]) for e in p["macro_graph"]["edges"])),
    )
    domains = Counter(p["scenario"]["domain"] for p in plans)
    mmqa_families = Counter(
        p["source_priors"]["MultiModalQA"]["composition_family"] for p in plans)
    return plans, {
        "generated": len(plans),
        "domains": dict(domains),
        "intents": dict(intents),
        "macro_graph_patterns": dict(topology),
        "unique_intent_topology_pairs": len(pairs),
        "intent_topology_diversity": {
            key: len({shape for (i, shape) in pairs if i == key}) for key in intents
        },
        "unique_macro_work_types": len(work_types),
        "macro_work_types": dict(work_types),
        "unique_macro_recipe_graphs": len({signature(p) for p in plans}),
        "unique_question_texts": len({p["question"] for p in plans}),
        "duplicate_question_texts": len(plans) - len({p["question"] for p in plans}),
        "macro_task_count_distribution": dict(Counter(
            p["complexity"]["macro_task_count"] for p in plans)),
        "macro_depth_distribution": dict(Counter(
            p["complexity"]["macro_dependency_depth"] for p in plans)),
        "mmqa_composition_families": dict(mmqa_families),
        "question_styles": dict(Counter(p["goal"]["question_style"] for p in plans)),
        "structural_preferences": dict(Counter(
            p["goal"]["structural_preference"] for p in plans)),
        "fixed_max_concurrency": 10,
        "runtime_variants_generated": False,
        "world_and_feedback_generated": False,
        "certification_scope": "typed macro contracts and structural consistency only",
    }
