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
from collections import Counter

from envir.schema import digest
from .patterns import BY_DOMAIN
from .v2_planner import build_v2_plan


QUESTION_REQUESTS = {
    "impact": (
        "identify which {affected} are materially affected, and explain what "
        "the available findings imply for their handling",
        "establish the scope and consequences for {affected}, distinguishing "
        "supported effects from matters outside the reported case",
    ),
    "compliance": (
        "assess whether the relevant conduct and records satisfy the applicable "
        "requirements, stating the basis for each conclusion",
        "determine which aspects of the documented situation meet the applicable "
        "rules and which warrant attention",
    ),
    "reconciliation": (
        "reconcile the different accounts of what occurred and explain which "
        "claims are supported by the available records",
        "establish a coherent account of the reported situation, noting any "
        "material differences between the available sources",
    ),
    "mitigation": (
        "identify practical responses and compare their feasibility and likely "
        "consequences for the responsible coordinator",
        "recommend a workable response, considering both documented impacts "
        "and the available operational constraints",
    ),
    "prioritization": (
        "determine which affected cases or follow-up activities deserve priority "
        "and justify the proposed order",
        "recommend a defensible order of action, considering the relevant "
        "evidence and operational obligations",
    ),
    "comparison": (
        "compare the available findings and explain which interpretation or "
        "response is better supported",
        "evaluate the differences between the relevant findings or proposed "
        "responses and state what they mean for the decision",
    ),
    "feasibility": (
        "evaluate which available response options can be implemented under "
        "the relevant practical and policy constraints",
        "assess the feasibility of possible next steps, including the conditions "
        "that limit their implementation",
    ),
    "timeline": (
        "reconstruct the relevant sequence of events and explain which timing "
        "details matter for resolving the reported situation",
        "establish the event chronology using the available documentation and "
        "identify any material difference in the recorded sequence",
    ),
}

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
    return {
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


def author_question(brief: dict, *, ordinal: int, seed: int) -> str:
    """Public wording is a function of macro work contracts, never Evidence."""
    intent = brief["intent"]
    case = BY_DOMAIN[brief["domain"]]
    rng = random.Random(seed * 314159 + ordinal * 271828)
    ask = rng.choice(QUESTION_REQUESTS[intent]).format(affected=case.affected)
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
    """Compose Task plan FIRST. Explicitly discard V2's preauthored question."""
    base = build_v2_plan(ordinal=ordinal, seed=seed, kqa=kqa, mmqa=mmqa)
    graph = base["macro_graph"]
    brief = task_level_brief(base)
    question = author_question(brief, ordinal=ordinal, seed=seed)
    uid = "macro-v4-" + digest(seed, ordinal, base["task_id"])
    return {
        "schema": "maestro.clean_taskfirst.v4",
        "task_id": uid,
        "scenario": dict(base["scenario"]),
        "macro_graph": graph,
        "question_brief": brief,
        "question": question,
        "source_priors": base["source_priors"],
        "task_plan_provenance": {
            "parent": "v2_macro_work_grammar",
            "question_generated_after_macro_graph": True,
            "question_inputs": "task_goals_deliverables_source_roles_only",
            "v3_final_gold_answer_not_used": True,
        },
        "scheduler": {
            "max_concurrent_subagents": 10,
            "runtime_variants_generated": False,
        },
    }
