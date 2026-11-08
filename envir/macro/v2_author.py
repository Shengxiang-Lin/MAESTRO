"""Optional natural question author with hard private/public prompt boundary.

LLM gets ONLY a public QuestionBrief and resource roles. It cannot see macro
DAG, typed edges, reference waves, KQA programs, future feedback or answers.
Rules provide only a *heuristic* semantic check, not a certification.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .author import OpenAICompatibleQuestionAuthor, QuestionAuthorError
from .v2_validation import validate_v2


INTENT_WORDS = {
    "impact": ("affected", "impact", "consequence", "implication"),
    "compliance": ("compliance", "comply", "requirement", "rule"),
    "reconciliation": ("reconcile", "corroborated", "discrepancy", "discrepancies"),
    "mitigation": ("response", "option", "action", "mitigation", "respond"),
    "prioritization": ("rank", "priority", "prioritize", "first"),
    "comparison": ("compare", "comparison", "difference", "trade-off"),
    "feasibility": ("feasible", "feasibility", "realistic", "implement"),
    "timeline": ("chronology", "sequence", "timeline", "order", "when"),
}


def check_question(plan: dict, proposed: str) -> list[str]:
    text = " ".join(str(proposed or "").split())
    problems = []
    if not 75 <= len(text) <= 650:
        problems.append("question_length")
    issue = plan["scenario"]["issue"]
    if issue.casefold() not in text.casefold():
        problems.append("scenario_subject_changed")
    if re.search(r"\b(?:S\d+|subtask|koPL|delegate to|first agent|second agent)\b",
                 text, re.I):
        problems.append("leaks_orchestration_details")
    key = plan["goal"]["intent"]
    if not any(w in text.lower() for w in INTENT_WORDS[key]):
        problems.append("missing_requested_result")
    # A generated number or ID is risky before evidence is grounded.
    original = plan["question"]
    invented_numbers = set(re.findall(r"\b\d{2,}\b", text)) - set(
        re.findall(r"\b\d{2,}\b", original))
    if invented_numbers:
        problems.append("invented_unverified_number")
    return problems


def naturalize_v2(plans: list[dict], *, author: OpenAICompatibleQuestionAuthor,
                  workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("question-workers must be >= 1")

    def propose(plan):
        brief = plan["question_brief"]
        packet = {
            "situation": brief["situation"],
            "user_role": brief["user_role"],
            "reference_question": brief["final_goal"],
            "expected_deliverable": brief["result_type"],
            "resource_descriptions": brief["evidence_requirement_descriptions"],
        }
        try:
            question = " ".join(author.complete(packet).split())
        except QuestionAuthorError as e:
            return ("backend_error", str(e)[:180], None)
        issues = check_question(plan, question)
        if issues:
            return ("rejected", ",".join(issues), None)
        return ("accepted", "heuristically_checked", question)

    counts = {"accepted": 0, "rejected": 0, "backend_error": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(propose, plans))
    for plan, (status, detail, question) in zip(plans, results):
        if question:
            before = (
                plan["question"], plan["goal"]["user_request"],
                plan["question_brief"]["final_goal"], plan["public"]["question"]
            )
            plan["question"] = question
            plan["goal"]["user_request"] = question
            plan["question_brief"]["final_goal"] = question
            plan["public"]["question"] = question
            faults = validate_v2(plan)
            if faults:
                (plan["question"], plan["goal"]["user_request"],
                 plan["question_brief"]["final_goal"],
                 plan["public"]["question"]) = before
                status, detail = "rejected", "structural_contract:" + ",".join(faults)
        counts[status] += 1
        plan["question_authoring"] = {
            "mode": "local_llm" if status == "accepted" else "deterministic_fallback",
            "status": status,
            "details": detail,
            "semantic_equivalence_certified": False,
        }
    return counts
