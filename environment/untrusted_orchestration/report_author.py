"""Constrained naturalization of an *executed* Sub-Agent attempt.

LLMs may improve the summary/trace prose, never mutate delivered values,
completion status, provenance, tool selection, or execution metadata.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from .state_machine import deterministic_report, Attempt


_SYSTEM = """You are writing a realistic Orchestra-o1 Sub-Agent summary from
the EXACT result of an already executed investigation in a controlled
workspace. Never guess missing results or replace fields. The immutable
result, field values, status and evidence references are determined by
the execution engine. Output JSON with ONLY:
{"summary": "concise grounded summary",
 "trace_summary": "concise description of the observed work"}
Do NOT mention private labels, fault injection, Oracle, Gold, simulation,
Teacher or generator revisions. If no result exists, acknowledge that no
usable result was delivered. No invented source, number, entity or tool.
"""

_REVIEW = """Assess only groundedness and realism of the proposed Sub-Agent
summary/trace relative to the exact public execution record. The result
string/status/provenance are immutable and have already been validated by
code. Do not review source-access permissions. Return JSON ONLY:
{"verdict":"pass"|"fail","grounded":true|false,"o1_style":true|false,
"reason":"brief reason"}.
"""

_TOKEN = re.compile(r"(?:Case-\d+|MK-\d+|RK-\d+|Entry-\d+|\b\d+(?:\.\d+)?\b)")
_PRIVATE = re.compile(
    r"\b(?:oracle|gold\s+(?:answer|program)|teacher|simulat\w*|"
    r"fault\s+inject\w*|previous\s+draft|critic\s+(?:said|noted))\b",
    re.I,
)


def naturalize_report(task: dict, attempt: Attempt, *,
                      instruction: str, model_alias: str,
                      reviewed: dict | None,
                      author=None, critic=None,
                      max_revisions: int = 1) -> tuple[dict, dict]:
    base = deterministic_report(
        task, attempt, instruction=instruction,
        model_alias=model_alias, reviewed=reviewed,
    )
    provenance = base.pop("_report_mode")
    if author is None or provenance == "reviewed_clean":
        return base, {"mode": provenance, "attempts": 0}
    if critic is None:
        raise ValueError("subagent_naturalization_requires_critic")
    fixed = base.get("finish_result")
    public = {
        "assigned_task_instruction": instruction,
        "immutable_finish_result": fixed,
        "immutable_trace_summary": base["trace_summary"],
    }
    allowed = set(_TOKEN.findall(json.dumps(public, ensure_ascii=False)))
    failures = []
    for retry in range(max_revisions + 1):
        try:
            draft = author.complete_json(
                _SYSTEM, json.dumps({
                    **public, "previous_prose_feedback": failures[-1:],
                }, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "author", "reason": str(exc)[:200]})
            continue
        if not isinstance(draft, dict):
            failures.append({"stage": "invalid_json"})
            continue
        summary, trace = draft.get("summary"), draft.get("trace_summary")
        if not (isinstance(summary, str) and isinstance(trace, str)
                and 15 <= len(summary) <= 600
                and 15 <= len(trace) <= 800):
            failures.append({"stage": "invalid_report_length"})
            continue
        authored_text = summary + " " + trace
        if (_PRIVATE.search(authored_text)
                or not set(_TOKEN.findall(authored_text)) <= allowed):
            failures.append({"stage": "new_unverified_claim_or_private_text"})
            continue
        try:
            verdict = critic.complete_json(
                _REVIEW, json.dumps({
                    "source_execution": public,
                    "candidate_summary": summary,
                    "candidate_trace": trace,
                }, ensure_ascii=False),
            )
        except Exception as exc:
            failures.append({"stage": "critic_backend", "reason": str(exc)[:200]})
            continue
        if (isinstance(verdict, dict) and verdict.get("verdict") == "pass"
                and verdict.get("grounded") is True
                and verdict.get("o1_style") is True):
            final = deepcopy(base)
            if fixed is not None:
                final["finish_result"]["summary"] = summary
            final["trace_summary"] = trace
            return final, {
                "mode": "llm_reviewed", "attempts": retry + 1,
            }
        failures.append({
            "stage": "critic",
            "reason": str(verdict.get("reason", ""))[:240]
                      if isinstance(verdict, dict) else "invalid_verdict",
        })
    return base, {
        "mode": "grounded_template_fallback",
        "attempts": max_revisions + 1,
        "failures": failures,
    }
