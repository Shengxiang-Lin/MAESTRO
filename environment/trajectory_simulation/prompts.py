"""Separated Main planning, Sub-Agent execution, and Main update prompts.

Main sees the public frozen task graph and past observations only.
The Sub-Agent receives just its assigned evidence and its scoped gold facts.
The oracle final answer is used for validation, not revealed to the LLM Main.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


def _dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def planner_prompt(payload: Mapping[str, Any]) -> str:
    public = {
        "question": payload["question"],
        "overall_solution_architecture": payload["public_plan"],
        "completed_subtask_ids": payload["completed_subtask_ids"],
        "history_summaries": payload["history_summaries"],
        "ready_tasks_this_round": payload["ready_tasks"],
        "pending_task_ids": payload["pending_task_ids"],
    }
    return (
        "Generate only the NEXT Main Agent task allocation, not the entire solution. "
        "All ready tasks must be scheduled NOW and in parallel; they are independent "
        "according to the frozen architecture. Keep every subtask_id, agent and "
        "resource_id exactly as given, with one specific instruction each. "
        "Do not schedule unready tasks, hallucinate tools, or claim future results. "
        "Return JSON: {\"parallel_subtasks\":[{\"subtask_id\":str,"
        "\"agent\":str,\"resource_id\":str,\"instruction\":str}]}.\\n\\n"
        + _dump(public)
    )


def executor_prompt(payload: Mapping[str, Any]) -> str:
    task = payload["task"]
    public = {
        "question": payload["question"],
        "assigned_subtask": payload["assigned_subtask"],
        "dependency_results": payload["dependency_results"],
        "source_observation": task["observation"],
        "authoritative_answer_claims": task["gold_answer_claims"],
    }
    return (
        "You are executing ONE assigned subtask in a clean environment. "
        "Read its provided source observation and completed prerequisite results. "
        "The authoritative_answer_claims are already verified against the frozen "
        "symbolic world. Output that list EXACTLY in the JSON field 'answer' "
        "(same fact IDs, values, ordering; do not add or omit facts). "
        "Write a concise natural-language answer_summary explaining what was found "
        "and a compressed trajectory_summary describing the work performed. "
        "Do not claim to have observed other sources or future results. "
        "Return JSON: {\"subtask_id\":str,\"status\":\"success\","
        "\"answer\":[...],\"answer_summary\":str,\"trajectory_summary\":str,"
        "\"evidence_source_ids\":[str]}. "
        "evidence_source_ids must be the single resource_id assigned to you.\\n\\n"
        + _dump(public)
    )


def summary_prompt(payload: Mapping[str, Any]) -> str:
    public = {
        "question": payload["question"],
        "completed_history": payload["history_summaries"],
        "current_parallel_subtask_results": payload["current_results"],
        "pending_task_ids": payload["pending_task_ids"],
    }
    return (
        "Write the Main Agent's concise round summary AFTER receiving all "
        "parallel Sub-Agent answers. State what has now been learned and which "
        "work remains, but do not predict unknown future observations or invent "
        "missing facts. Return exactly {\"main_summary\":str}.\\n\\n"
        + _dump(public)
    )


def final_prompt(payload: Mapping[str, Any]) -> str:
    public = {
        "question": payload["question"],
        "overall_solution_architecture": payload["public_plan"],
        "all_completed_subagent_results": payload["all_results"],
        "history_summaries": payload["history_summaries"],
    }
    return (
        "Every required gold subtask is completed. Return a final answer to "
        "the original question, supported by the accumulated evidence. "
        "Use the same answer type/format as the question requires: JSON string, "
        "number, boolean, or list. Do not invent facts. Return JSON: "
        "{\"action\":\"complete\",\"answer\":<answer>,\"summary\":str}.\\n\\n"
        + _dump(public)
    )


__all__ = [
    "planner_prompt", "executor_prompt", "summary_prompt", "final_prompt",
]
