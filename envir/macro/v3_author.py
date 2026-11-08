"""Optional local Question Author; strict public-only prompt and safe fallback.

The LLM receives the user-facing question and public resource descriptions,
NOT the macro graph, hidden evidence, verified feedback, or Gold answer.
Surface checks are conservative and cannot replace a semantic judge.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from .author import OpenAICompatibleQuestionAuthor, QuestionAuthorError
from .v3_runtime import validate_grounded


INTENT_TERMS = {
    "compliance": ("registration", "status", "time", "requirements", "rule"),
    "impact": ("affected", "threshold", "total", "units"),
    "comparison": ("larger", "higher", "difference", "workload"),
    "timeline": ("difference", "gap", "time", "largest"),
    "reconciliation": ("disagree", "difference", "minute", "timestamp"),
    "feasibility": ("capacity", "valid", "time", "eligible"),
    "prioritization": ("priority", "urgency", "highest", "workload"),
    "mitigation": ("capacity", "shortfall", "priority", "mitigation"),
}


def assess_surface(scene: dict, candidate: str) -> list[str]:
    question = " ".join(str(candidate or "").split())
    baseline = scene["question"]
    need = scene["information_need"]
    issues = []
    if len(question) < 100 or len(question) > 1050:
        issues.append("length")
    if any(cid not in question for cid in need["case_ids"]):
        issues.append("case_scope_changed")
    if need["as_of"] not in question:
        issues.append("date_changed")
    if need["setting"].casefold() not in question.casefold():
        issues.append("setting_changed")
    if need["issue"].casefold() not in question.casefold():
        issues.append("case_context_changed")
    if any(name.casefold() not in question.casefold()
           for name in (need["source_names"][role]
                        for role in need["required_source_roles"])):
        issues.append("evidence_roles_changed")
    required_words = INTENT_TERMS[need["intent"]]
    if sum(any(term in word.lower() for word in question.split())
           for term in required_words) < 2:
        issues.append("result_or_conditions_changed")
    baseline_digits = set(re.findall(r"\b\d+\b", baseline))
    new_digits = set(re.findall(r"\b\d+\b", question))
    if new_digits - baseline_digits:
        issues.append("invented_numeric_constraint")
    if need["intent"] == "prioritization" and not all(
        marker in question.lower() for marker in ("workload", "urgency", "available")
    ):
        issues.append("priority_formula_changed")
    if re.search(r"\b(?:subtask|KoPL|first agent|second agent|delegate|S\d+)\b",
                 question, re.I):
        issues.append("orchestration_leak")
    return sorted(set(issues))


def rewrite_public_questions(scenes: list[dict], *,
                             author: OpenAICompatibleQuestionAuthor,
                             workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    def one(scene):
        pub = scene["public"]
        # ONLY public data, never a private gold/problem graph, trace or source
        # observations. Full source descriptions appear in the question.
        packet = {
            "situation": scene["information_need"]["issue"],
            "reference_question": pub["question"],
            "resource_descriptions": [
                {"title": s["title"], "modality": s["modality"]}
                for s in pub["provided_sources"]
            ],
            "task": (
                "Rewrite as one natural, concise, answer-equivalent user request. "
                "Keep ALL case IDs, date, requested result, formulas and source "
                "names exactly; introduce no new facts, numbers or agent steps."
            ),
        }
        try:
            rewritten = " ".join(author.complete(packet).split())
        except QuestionAuthorError as exc:
            return ("backend_error", str(exc)[:150], None)
        issues = assess_surface(scene, rewritten)
        if issues:
            return ("rejected", ",".join(issues), None)
        return ("accepted", "surface_checked_not_semantically_certified", rewritten)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        proposals = list(pool.map(one, scenes))
    counts = {"accepted": 0, "rejected": 0, "backend_error": 0}
    for scene, (status, reason, proposal) in zip(scenes, proposals):
        if proposal is not None:
            old = scene["question"]
            scene["question"] = proposal
            scene["public"]["question"] = proposal
            if validate_grounded(scene):
                scene["question"] = old
                scene["public"]["question"] = old
                status, reason = "rejected", "structural_or_replay_validation_failed"
        counts[status] += 1
        scene["question_authoring"] = {
            "mode": "local_llm" if status == "accepted" else "deterministic_fallback",
            "status": status,
            "reason": reason,
            "semantic_equivalence_certified": False,
        }
    return counts
