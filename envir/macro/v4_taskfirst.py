"""Task-first CLEAN orchestration design.

A concrete scene and desired macro work deliverable guide a legitimate DAG of
substantial work packages. Only AFTER this graph is compiled do we author a
public user question from the *work goals and deliverables*. The question
author never sees task-internal source facts or summary traces.

This intentionally reuses the tested V2 macro work grammar for the pilot,
not V3's fixed numeric-answer templates or Gold Answer compiler.
"""
from __future__ import annotations

import random
from dataclasses import replace

from envir.schema import digest
from .patterns import CASES, CASE_VARIANTS
from .priors import choose_kqa, choose_mmqa
from .v2_workbank import GOAL_TYPES
from .graph import (build_atomic_draft, coarsen, compile_contracts,
                    compile_waves, complexity)
from .v2_planner import _build_work_graph, _topology_metrics, _classify_topology


# Non-agent-friendly public resource labels (not instructions or planned steps).
SOURCE_LABELS = {
    "material": "primary event material",
    "records": "relevant operational records",
    "rule_source": "applicable guidance",
    "capacity": "capacity information",
    "stakeholders": "stakeholder requirements",
    "independent_reports": "independent reports",
    "context": "background documents",
}


def task_level_brief(plan: dict) -> dict:
    """No evidence values, internal steps, teacher Gold or hidden traces."""
    tasks = plan["macro_graph"]["tasks"]
    terminal = plan["macro_graph"]["terminal_handoffs"][0]
    by_id = {t["task_id"]: t for t in tasks}
    final = by_id[terminal["from_task"]]
    roots = [t for t in tasks if not t["depends_on"]]
    sources = []
    for task in tasks:
        for source in task.get("source_requirements", ()):
            role = source["role"]
            label = SOURCE_LABELS.get(role, "supporting documentation")
            if label not in sources:
                sources.append(label)
    result = {
        "domain": plan["scenario"]["domain"],
        "setting": plan["scenario"]["setting"],
        "issue": plan["scenario"]["issue"],
        "user_role": plan["scenario"]["user_role"],
        "intent": plan["goal"]["intent"],
        "final_work_goal": final["goal"],
        "final_deliverable": final["deliverable"],
        "root_work_goals": [t["goal"] for t in roots],
        "supporting_work_types": sorted({t["work_type"] for t in tasks}),
        "public_resource_categories": sources,
    }
    if plan["scenario"].get("source_kind") == "MultiModalQA":
        result["source_kind"] = "MultiModalQA"
    return result


def author_question(brief: dict, *, ordinal: int, seed: int) -> str:
    """Public wording is a function of macro work contracts, never Evidence."""
    # Data-derived reference wording follows the immutable terminal work goal.
    if brief.get("source_kind") == "MultiModalQA":
        from .data_taskfirst import FINAL_PREFIX
        goal = brief["final_work_goal"]
        if not goal.startswith(FINAL_PREFIX):
            raise ValueError("MMQA question requires a terminal question goal")
        return goal[len(FINAL_PREFIX):].strip()
    intent = brief["intent"]
    rng = random.Random(seed * 314159 + ordinal * 271828)
    # The requested action comes from the TERMINAL MACRO WORK PRODUCT,
    # not from a separate question-intent template. Different legitimate
    # work routes therefore yield genuinely different final asks.
    task_goal = brief["final_work_goal"].rstrip(". ")
    task_goal = task_goal.replace(brief["issue"], "the reported situation")
    ask = task_goal[0].lower() + task_goal[1:]
    # The intent-specific clause is only an optional delivery constraint,
    # never the main source of user-question semantics.
    if rng.randrange(3) == 0:
        ask += "; explain which documented findings support the conclusion"
    resources = brief["public_resource_categories"]
    if len(resources) >= 3:
        described = ", ".join(resources[:-1]) + " and " + resources[-1]
    elif len(resources) == 2:
        described = " and ".join(resources)
    elif resources:
        described = resources[0]
    else:
        described = "the relevant documentation"
    if ordinal % 3 == 0:
        first = (
            f"The {brief['user_role']} is reviewing {brief['issue']} "
            f"at the {brief['setting']}."
        )
        second = f"Using {described}, please {ask}."
    elif ordinal % 3 == 1:
        first = f"At the {brief['setting']}, {brief['issue']} has been reported."
        second = f"Based on {described}, please {ask}."
    else:
        first = f"We need to address {brief['issue']} at the {brief['setting']}."
        second = f"Considering {described}, please {ask}."
    # Scope is defined by the reported incident, not an invented record ID.
    # The original scenario explicitly remains an ungrounded synthetic case.
    return f"{first} {second[0].upper() + second[1:]} Explain which documented findings support your assessment."


def compose_taskfirst(*, ordinal: int, seed: int,
                      kqa: list[dict], mmqa: list[dict]) -> dict:
    """The Question is authored ONLY after all macro work packages exist.

    The goal/result-type is a design constraint on task composition. It is
    not a prewritten user Question or a final-answer oracle.
    """
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
    source_hint = rng.choice(
        ("chain", "parallel_join", "fork_join", "mixed", "staged")
    )
    mmqa_prior, mmqa_policy = choose_mmqa(
        mmqa, domain=case.domain, motif=source_hint, rng=rng,
        ordinal=ordinal,
    )
    structural_preference = ("serial", "branch", "balanced")[
        (iteration + index * 5 + seed) % 3
    ]
    candidate = None
    for attempt in range(60):
        work_rng = random.Random(rng.randrange(1 << 40) + attempt)
        packages, edges, terminal, recipes = _build_work_graph(
            case, intent, mmqa_prior["logical_pattern"], work_rng,
            attempt, structural_preference,
        )
        if not 3 <= len(packages) <= 9:
            continue
        draft = build_atomic_draft(packages, edges)
        tasks, coarsening = coarsen(draft)
        compile_contracts(tasks, edges)
        waves = compile_waves(tasks)
        comp = complexity(tasks, waves, coarsening)
        if comp["macro_dependency_depth"] < 2:
            continue
        stats = _topology_metrics(tasks, edges, waves)
        candidate = (tasks, coarsening, edges, terminal, recipes, waves,
                     comp, stats)
        break
    if candidate is None:
        raise ValueError("could not compose a complete macro work graph")
    tasks, coarsening, edges, terminal, recipes, waves, comp, stats = candidate
    kqa_prior, kqa_policy = choose_kqa(kqa, source_hint, rng, ordinal)
    graph = {
        "motif": _classify_topology(stats, comp["macro_dependency_depth"]),
        "tasks": tasks,
        "edges": edges,
        "terminal_handoffs": [terminal],
        "reference_waves": waves,
        "provenance_recipe_routes": recipes,
        "complexity": {**comp, **stats},
        "coarsening": coarsening,
    }
    task_uid = "macro-v4-" + digest(
        seed, ordinal, case.domain, intent, case.issue, recipes,
        mmqa_prior["motif_id"],
    )
    plan = {
        "schema": "maestro.clean_taskfirst.v4",
        "task_id": task_uid,
        "scenario": {
            "domain": case.domain,
            "setting": case.setting,
            "issue": case.issue,
            "user_role": case.decision_owner,
        },
        "goal": {
            "intent": intent,
            "desired_artifact_type": terminal["type"],
        },
        "macro_graph": graph,
        "source_priors": {
            "KQA": {
                "policy": kqa_policy,
                "signature": kqa_prior["signature"],
                "program": kqa_prior["program"],
                "usage": "internal_reasoning_prior_not_one_subtask_per_operator",
            },
            "MultiModalQA": {
                "policy": mmqa_policy,
                "motif_id": mmqa_prior["motif_id"],
                "official_question_type": mmqa_prior["question_type"],
                "composition_family": mmqa_prior["logical_pattern"],
                "source_modalities": mmqa_prior["source_modalities"],
                "usage": "macro_work_source_pattern_prior_not_actual_media",
            },
        },
        "task_plan_provenance": {
            "parent": "v2_macro_work_grammar",
            "ordinal": ordinal,
            "seed": seed,
            "question_generated_after_macro_graph": True,
            "question_inputs": "task_goals_deliverables_source_roles_only",
            "v3_final_gold_answer_not_used": True,
        },
        "scheduler": {
            "max_concurrent_subagents": 10,
            "runtime_variants_generated": False,
        },
    }
    brief = task_level_brief(plan)
    plan["question_brief"] = brief
    plan["question"] = author_question(brief, ordinal=ordinal, seed=seed)
    return plan
