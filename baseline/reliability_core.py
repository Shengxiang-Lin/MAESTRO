"""Shared *online adaptation* contract for evidence-reliability baselines.

Only delivered O1 Main-Agent task_entries are passed into reviewers.
No future Sub-Agent reports, private simulation labels, or oracle source facts.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any
from pydantic import Field, PrivateAttr
from orchestra.base_main_agent import BaseMainAgent

HERE = Path(__file__).resolve().parent
METHODS = ("a-trust", "omni-decision", "propuq-mas", "hassum", "verifymas")


def public_reports(task_entries: list[dict]) -> list[dict]:
    """Extract visible reports with stable IDs; never infer hidden DAG edges."""
    out = []
    for position, entry in enumerate(task_entries, 1):
        attempt = entry.get("attempt", position)
        members = (entry.get("subtask_results", [])
                   if entry.get("is_parallel_batch") else [entry])
        for index, report in enumerate(members, 1):
            if not isinstance(report, dict):
                continue
            out.append({
                "id": f"r{attempt}.{index}", "attempt": attempt,
                "agent": str(report.get("model", "unknown")),
                "task": str(report.get("instruction", "")),
                "status": str(report.get("status", "unknown")),
                "result": str(report.get("result", "")),
                "summary": str(report.get("summary", "")),
                "trace_summary": str(report.get("trace_summary", "")),
                "depends_on": list(report.get("depends_on") or []),
                "confidence": report.get("confidence"),
            })
    return out


def strict_json(text: str) -> dict:
    value = str(text).strip()
    fence = chr(96) * 3
    if value.startswith(fence):
        value = value[len(fence):].strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
        if not value.endswith(fence):
            raise ValueError("unclosed JSON fence")
        value = value[:-len(fence)].strip()
    if value.startswith("{") and value.endswith("}"):
        obj = json.loads(value)
    else:
        raise ValueError("reviewer must return one JSON object")
    if not isinstance(obj, dict):
        raise ValueError("review must be JSON object")
    return obj


async def ask_json(llm: Any, prompt: str) -> dict:
    """Use the existing cost-metered O1 LLM, with no silent fallback."""
    return strict_json(await llm(prompt))


def clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_review(review: dict, reports: list[dict]) -> dict:
    if not isinstance(review, dict):
        raise TypeError("review must be an object")
    ids = {r["id"] for r in reports}
    evidence = []
    for item in review.get("evidence", []):
        if not isinstance(item, dict) or item.get("report_id") not in ids:
            continue
        try:
            risk = clip(item["risk"])
        except (KeyError, TypeError, ValueError):
            continue
        evidence.append({"report_id": item["report_id"], "risk": risk,
                         "reason": str(item.get("reason", ""))[:500]})
    suggestions = []
    allowed = {"verify", "retry", "switch", "hold", "accept",
               "check_dependency", "isolate"}
    for item in review.get("suggestions", []):
        if not isinstance(item, dict) or item.get("report_id") not in ids:
            continue
        if item.get("action") not in allowed:
            continue
        suggestions.append({"report_id": item["report_id"],
                            "action": item["action"],
                            "reason": str(item.get("reason", ""))[:400]})
    issues = review.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    return {"risk": max([x["risk"] for x in evidence], default=0.0),
            "evidence": evidence, "suggestions": suggestions,
            "issues": [str(v)[:400] for v in issues[:12]],
            "details": review.get("details", {})}


def review_to_note(review: dict, method: str, limit: int = 3500) -> str:
    safe = {k: review[k] for k in ("risk", "evidence", "issues", "suggestions")}
    return ("\n\n==== EXTERNAL RELIABILITY ASSESSMENT (ADVISORY) ====\n"
            "Method: " + method +
            ". This may be wrong. It is not new evidence or an execution result. "
            "Use the original O1 schema and decide independently.\n"
            + json.dumps(safe, ensure_ascii=False)[:limit]
            + "\n==== END ASSESSMENT ====\n")


def load_reviewer(method: str):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    path = HERE / method / "method.py"
    spec = importlib.util.spec_from_file_location(
        "maestro_method_" + method.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.review


class ReliabilityMainAgent(BaseMainAgent):
    """Thin plugin hook preserving original Main-Agent decision and tools."""

    baseline_method: str = Field(default="propuq-mas")
    baseline_options: dict = Field(default_factory=dict)
    review_audit_dir: str = Field(default="")
    _review_note: str = PrivateAttr(default="")
    _review_record: dict = PrivateAttr(default_factory=dict)

    def reset(self, env_info) -> None:
        super().reset(env_info)
        self._review_note = ""
        self._review_record = {}

    async def _refresh_review(self) -> None:
        self._review_note = ""
        reports = public_reports(self.task_entries)
        if not reports:
            return
        reviewer = load_reviewer(self.baseline_method)
        output = await reviewer(reports=reports, llm=self.llm,
                                options=dict(self.baseline_options))
        result = normalize_review(output, reports)
        self._review_record = {"method": self.baseline_method,
                               "attempt": self.attempt + 1,
                               "report_ids": [x["id"] for x in reports],
                               "review": result}
        self._review_note = review_to_note(result, self.baseline_method)
        if self.review_audit_dir:
            path = Path(self.review_audit_dir)
            path.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(
                self.meta.get("task_id", id(self))))
            with (path / f"{safe}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._review_record, ensure_ascii=False) + "\n")

    def _build_prompt(self, *, context: str, subtask_history: str,
                      attempt_index: int) -> str:
        prompt = super()._build_prompt(
            context=context, subtask_history=subtask_history,
            attempt_index=attempt_index)
        return prompt + self._review_note

    async def step(self, observation, history, **kwargs):
        await self._refresh_review()
        return await super().step(observation, history, **kwargs)

    async def generate_next_action(self, **kwargs):
        await self._refresh_review()
        return await super().generate_next_action(**kwargs)
