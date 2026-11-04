"""Only two simulation roles: O1 Main action and fragment-scoped Sub-Agent trace.

Fixed symbolic answers and graph bindings are provided by the controller.
Use JSON-object mode, not a dynamic enum/regex schema containing private facts.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from environment.question_generation.backend import OpenAICompatibleBackend
from .contract import answer_string


class PassthroughPlanner:
    """Offline reference only. Not a substitute for LLM-authored gold examples."""
    offline = True

    def decide(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        guidance = payload["gold_guidance"]
        if guidance["may_complete"]:
            return {
                "action": "complete", "reasoning": "The official subgoals are complete.",
                "params": {"answer": answer_string(payload["offline_oracle_answer"])},
            }
        return {
            "action": "delegate_task",
            "reasoning": "Delegate the independent current gold subgoals in parallel.",
            "params": {
                "tasks": [{
                    "task_instruction": (
                        "Use the available clues to perform " +
                        ", ".join(op["op"] for op in task["operations"]) +
                        " and return the resulting intermediate answer."
                    ),
                    "context": payload["public_state"]["question"],
                    "model": "model_1",
                    "tools": task["suggested_tools"],
                } for task in guidance["ready"]
                ],
            },
        }


class LLMPlanner:
    offline = False

    def __init__(self, backend: OpenAICompatibleBackend):
        self.backend = backend
        self.review_backend = OpenAICompatibleBackend(**{
            **asdict(backend), "temperature": 0.0, "max_tokens": 1024})

    def review(self, prompt: str) -> dict[str, Any]:
        return self.review_backend.complete_json(
            "Check teacher language against the supplied fixed contract, not a new solution. "
            "Return only a short accept/issues JSON verdict with exact quotes.", prompt)

    def decide(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You generate one grounded, baseline-compatible Orchestra-o1 "
            "Main action from public history and PRIVATE fragment-level "
            "teacher guidance. Never copy private hints into public output. "
            "Return a JSON object, not an entire trajectory.", prompt,
        )


class PassthroughExecutor:
    offline = True

    def respond(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        fragment = payload["fragment"]
        return {
            "summary": (
                "The assigned composed subtask has the verified result "
                + answer_string(fragment["expected_answer"]) + "."
            ),
            "trace_summary": (
                "Used the assigned source observations and predecessor results "
                "to perform the specified inference steps, without speculative evidence."
            ),
        }


class LLMExecutor:
    offline = False

    def __init__(self, backend: OpenAICompatibleBackend):
        self.backend = backend
        self.review_backend = OpenAICompatibleBackend(**{
            **asdict(backend), "temperature": 0.0, "max_tokens": 1024})

    def review(self, prompt: str) -> dict[str, Any]:
        return self.review_backend.complete_json(
            "Check teacher language against the supplied fixed contract, not a new solution. "
            "Return only a short accept/issues JSON verdict with exact quotes.", prompt)

    def respond(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You simulate one successful Sub-Agent's compressed trace for a "
            "PREDETERMINED reasoning subtask. The controller supplies and "
            "checks the exact answer; write only summary and trace_summary "
            "using the assigned evidence. Return JSON only.", prompt,
        )


__all__ = [
    "PassthroughPlanner", "LLMPlanner", "PassthroughExecutor", "LLMExecutor",
]
