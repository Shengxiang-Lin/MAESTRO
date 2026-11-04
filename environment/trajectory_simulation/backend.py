"""Two independent LLM roles for gold orchestration trajectory simulation.

Planner: assign current parallel subtasks, integrate replies, finalize.
Executor: solve one assigned subtask using its frozen evidence, returning an
exact structured gold answer and a concise execution trace.
"""
from __future__ import annotations

from typing import Any, Mapping

from environment.question_generation.backend import OpenAICompatibleBackend


class PassthroughPlanner:
    """Deterministic offline teacher for regression tests, not benchmark data."""

    def plan(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return {
            "parallel_subtasks": [
                {
                    "subtask_id": task["subtask_id"],
                    "agent": task["agent"],
                    "resource_id": task["resource_id"],
                    "instruction": f"Inspect {task['resource_id']} for evidence relevant to the question.",
                }
                for task in payload["ready_tasks"]
            ]
        }

    def summarize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        ids = [result["subtask_id"] for result in payload["current_results"]]
        return {"main_summary": "Completed and incorporated evidence from " + ", ".join(ids) + "."}

    def finalize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return {
            "action": "complete",
            "answer": payload["expected_answer"],
            "summary": "All necessary evidence has been processed; the official answer is confirmed.",
        }


class LLMPlanner:
    def __init__(self, backend: OpenAICompatibleBackend):
        self.backend = backend

    def plan(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a gold-trajectory Main Orchestrator. "
            "Do not modify the frozen solution plan or access future evidence. "
            "Return only the requested JSON.", prompt
        )

    def summarize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a gold-trajectory Main Orchestrator. Summarize only "
            "already-observed subagent results, not unseen evidence. Return JSON.", prompt
        )

    def finalize(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a gold-trajectory Main Orchestrator. Conclude the task "
            "from completed subagent evidence; return only JSON.", prompt
        )


class PassthroughExecutor:
    """Offline deterministic gold-copying reference (no LLM)."""

    def execute(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        task = payload["task"]
        return {
            "subtask_id": task["subtask_id"],
            "status": "success",
            "answer": task["gold_answer_claims"],
            "answer_summary": "Grounded facts have been obtained from the assigned resource.",
            "trajectory_summary": "Inspected the supplied evidence and extracted supported facts.",
            "evidence_source_ids": [task["resource_id"]],
        }


class LLMExecutor:
    def __init__(self, backend: OpenAICompatibleBackend):
        self.backend = backend

    def execute(self, payload: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return self.backend.complete_json(
            "You are a clean Sub-Agent teacher. Use only the assigned source. "
            "Copy the authoritative structured facts exactly; do not invent "
            "additional evidence, conclusions, or unseen results. Return JSON.", prompt
        )


__all__ = [
    "LLMPlanner", "LLMExecutor", "PassthroughPlanner", "PassthroughExecutor",
]
