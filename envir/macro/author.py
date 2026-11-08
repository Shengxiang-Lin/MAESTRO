"""Optional OpenAI-compatible natural language author for public user questions.

The author sees a scoped QuestionBrief ONLY, never the hidden macro DAG, KQA
operations, teacher-private records, or future Sub-Agent feedback. It may
rewrite the question surface but cannot modify an executable contract.
Semantic equivalence is NOT formally certified by these heuristic gates.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class QuestionAuthorError(ValueError):
    pass


_WORDS = {"after", "there", "their", "about", "which", "those", "following",
          "during", "current", "event", "scheduled", "situation", "recent",
          "relevant", "updated", "following", "newly", "reported", "with",
          "under", "before", "another", "from", "this", "that"}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", str(text).lower())
            if w not in _WORDS}


def check_rewrite(plan: dict, question: str) -> list[str]:
    original = plan["question_brief"]["final_goal"]
    issue = plan["scenario"]["issue"]
    candidate = " ".join(question.split())
    if not 40 <= len(candidate) <= 600:
        return ["length"]
    if re.search(r"\b(?:S\d+|subtask|agent|step\s*\d+|first\s+step)\b",
                 candidate, re.I):
        return ["orchestration_instructions"]
    if re.search(r"\b(?:AX|MX|EQ)-\d{3,}\b", candidate):
        return ["invented_private_identifier"]
    issue_tokens = _keywords(issue)
    matched = issue_tokens & _keywords(candidate)
    if len(matched) < min(2, len(issue_tokens)):
        return ["case_subject_not_preserved"]
    # Keep prompt's central question speech act and output obligation.
    intent = plan["goal"]["intent"]
    required = {
        "impact": ("affect", "impact", "consequence", "implicat"),
        "compliance": ("requirement", "compliance", "satisf", "rule", "policy"),
        "reconciliation": ("discrepan", "reconcil", "inconsisten", "conflict"),
        "mitigation": ("respond", "mitigat", "action", "step", "recommend"),
        "prioritization": ("priorit", "urgent", "rank", "first"),
    }[intent]
    if not any(any(w.startswith(stem) for w in _keywords(candidate))
               for stem in required):
        return ["requested_output_not_preserved"]
    if "Recommend feasible next steps" in original and not any(
        "recommend" in w or "next" in w or "action" in w
        for w in _keywords(candidate)
    ):
        return ["response_recommendation_missing"]
    return []


class OpenAICompatibleQuestionAuthor:
    def __init__(self, *, api_base: str, model: str, timeout: int = 90,
                 api_key: str | None = None):
        if not model.strip() or not api_base.strip():
            raise ValueError("question model and API base are required")
        self.url = api_base.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            if self.url.endswith("/v1"):
                self.url += "/chat/completions"
            else:
                self.url += "/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "EMPTY")

    def complete(self, payload: dict) -> str:
        body = {
            "model": self.model,
            "temperature": 0.55,
            "max_tokens": 420,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": (
                    "You write one natural, realistic user question for a "
                    "multi-agent work scenario. Preserve the reference question's "
                    "topic, constraints and requested deliverable. Never invent "
                    "new case facts, evidence, answers, tool actions or identities. "
                    "Do not tell the user which subtask, agent, or execution order "
                    "to use. Return JSON {\"question\":\"...\"}, nothing else."
                )},
                {"role": "user", "content": json.dumps(
                    payload, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            item = data["choices"][0]["message"]["content"]
            result = json.loads(item)
            return str(result["question"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise QuestionAuthorError(
                f"LLM question author failed: {type(exc).__name__}: {str(exc)[:180]}"
            ) from exc


def naturalize_questions(plans: list[dict], *,
                         author: OpenAICompatibleQuestionAuthor,
                         workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("workers must be >=1")

    def one(plan):
        # ONLY public situation/need. No macro-task IDs, DAG, KoPL trace,
        # source-prior metadata or private world/answer.
        brief = plan["question_brief"]
        packet = {
            "situation": brief["situation"],
            "user_role": brief["user_role"],
            "reference_question": brief["final_goal"],
            "expected_deliverable": brief["result_type"],
            "resource_descriptions": brief["public_resources"],
        }
        try:
            suggestion = author.complete(packet)
            issues = check_rewrite(plan, suggestion)
            if issues:
                return False, "rejected:" + ",".join(issues), None
            return True, "surface_heuristically_checked", " ".join(suggestion.split())
        except QuestionAuthorError as exc:
            return False, "backend_error:" + str(exc)[:150], None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(one, plans))
    counts = {"accepted": 0, "rejected": 0, "errors": 0}
    for plan, (accepted, status, question) in zip(plans, outputs):
        if accepted:
            plan["question"] = question
            plan["public"]["question"] = question
            counts["accepted"] += 1
        else:
            counts["errors" if status.startswith("backend_error") else "rejected"] += 1
        plan["question_authoring"] = {
            "source": "local_llm" if accepted else "deterministic_fallback",
            "status": status,
            "semantic_equivalence_certified": False,
        }
    return counts
