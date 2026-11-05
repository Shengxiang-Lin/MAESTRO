"""Round-level, teacher-forced Main SFT examples with native O1 prompt inputs.

Do not use this path to synthesize tool executions or incorrect Sub-Agent
feedback. The full frozen ProgramIR and public-witness preflight certify the
possible clean continuation; a decision round is exported independently after
its history prefix, action contract, and natural-language requirements pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from train_maestro_8b.schema import validate_stage1_record
from .eligibility import digest
from .exporter import training_target, validate_public_episode


# These are public tool names/descriptions, not fabricated tool executions.
# The canonical prompt builder supplies formatting and native O1 instructions.
_TOOL_DESCRIPTIONS = {
    "GoogleSearchAction": "Search public web pages for relevant information.",
    "ExtractUrlAction": "Extract the relevant text of a public web URL.",
    "ExtractUrlContentAction": "Extract the relevant text of a public web URL.",
    "VideoAnalysisAction": "Analyze supplied video content, including frames and audio.",
    "ImageAnalysisAction": "Analyze a task-provided image.",
    "ParseAudioAction": "Transcribe and analyze task-provided audio.",
    "ExecuteCodeAction": "Run Python code to calculate or process evidence.",
}


def _prior_context(history: list[Mapping[str, Any]]) -> str:
    """Produce Main-visible context from verified past replies, never gold truth."""
    sections: list[str] = []
    for step in reversed(history):
        if step["action"] != "delegate_task":
            continue
        results = step["result"]["subtask_results"]
        lines = [f"[{step['attempt']}] delegate_task",
                 f"  Parallel batch: {len(results)} subtask(s)"]
        for item in results:
            result = item["finish_result"]
            lines.append(
                f"  - Subtask: {item['task_instruction'][:180]}"
                f"\n    Status: {result['status']}, Result: {result['result']}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def o1_input(
    state: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    *,
    question_record: Mapping[str, Any],
    round_index: int,
    max_attempts: int = 10,
) -> str:
    """Use the REAL Orchestra-o1 Main prompt builder, not Stage-3's short prompt.

    Synthetic source descriptors are already appended to state.question.
    Actual media paths are included only if they truly occur in the task
    record. No future subtask observations or private supervision are added.
    """
    from orchestra.prompts.omnigaia import OmniGAIAMainAgentPrompt
    tools = [
        SimpleNamespace(
            name=str(name),
            description=_TOOL_DESCRIPTIONS.get(str(name), "Task-specific public action."),
            parameters={},
        )
        for name in state["available_tools"]
    ]
    meta: dict[str, Any] = {}
    for key in ("task_type", "category"):
        if isinstance(question_record.get(key), str):
            meta[key] = question_record[key]
    modal = question_record.get("omni_modal_input", question_record.get("modal_inputs"))
    if isinstance(modal, list):
        meta["modal_inputs"] = [{
            key: item[key] for key in ("type", "id", "path") if key in item
        } for item in modal if isinstance(item, dict)]
    prompt = OmniGAIAMainAgentPrompt.build_prompt(
        instruction=state["question"],
        meta=meta,
        prior_context=_prior_context(history),
        attempt_index=round_index,
        max_attempts=max(max_attempts, round_index),
        sub_models=["model_1"],
        subtask_history=state["subtask_history"],
        tools=tools,
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("empty_real_o1_prompt")
    return prompt


def make_round_record(
    *,
    task_id: str,
    round_index: int,
    question: str,
    eligibility: Mapping[str, Any],
    state: Mapping[str, Any],
    prefix: list[Mapping[str, Any]],
    action: Mapping[str, Any],
    question_record: Mapping[str, Any],
    natural_task_count: int,
    assigned_task_count: int,
    history_origin: str,
    max_o1_attempts: int = 10,
    require_review: bool = True,
) -> dict[str, Any]:
    """Validate the previous public rounds and export exactly one Main decision."""
    if history_origin not in {"natural", "teacher_reconstructed"}:
        raise ValueError("invalid_history_origin")
    if require_review and not (eligibility.get("eligible") and eligibility.get("reviewed")):
        raise ValueError("unreviewed_question_cannot_export")
    if eligibility.get("question_sha256") != digest(question):
        raise ValueError("question_changed_after_review")
    validate_public_episode({
        "question": question, "task_id": task_id,
        "input_eligibility": eligibility,
        "validation": {"public_evidence_replay_verified": True},
        "trajectory": prefix,
    }, require_training=False, require_terminal=False)
    if action["action"] not in {"complete", "delegate_task"}:
        raise ValueError("not_a_main_decision")
    if action["action"] == "delegate_task":
        if len(action["params"]["tasks"]) != assigned_task_count or assigned_task_count < 1:
            raise ValueError("round_task_count_mismatch")
    else:
        if assigned_task_count != 0:
            raise ValueError("complete_with_delegation")
    target = training_target(action)
    sample = {
        "id": f"{task_id}:round-{round_index}",
        "task_id": task_id,
        "step": round_index,
        "round_index": round_index,
        "input": o1_input(
            state, prefix, question_record=question_record,
            round_index=round_index, max_attempts=max_o1_attempts,
        ),
        "target_json": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "macro_label": "complete" if action["action"] == "complete" else "advance",
        "macro_label_source": "gold_constrained_clean_round",
        "history_origin": history_origin,
        "history_valid": True,
        "current_action_valid": True,
        "natural_tasks_rewritten": natural_task_count,
        "assigned_task_count": assigned_task_count,
        "question_reviewed": bool(eligibility.get("reviewed")),
        "clean_feedback_assumption": True,
    }
    return validate_stage1_record(sample)


class RoundSink:
    """Synchronous per-round durable writing; output atomicity is per line."""

    def __init__(
        self, path: str | Path, *, question: Mapping[str, Any],
        include_fallback_rounds: bool = False,
        include_complete: bool = True,
        max_o1_attempts: int = 10,
    ):
        self.output = Path(path)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.question = question
        self.include_fallback_rounds = include_fallback_rounds
        self.include_complete = include_complete
        self.max_o1_attempts = max_o1_attempts
        self.written = 0
        self.skipped = 0
        self.skip_reasons: dict[str, int] = {}
        self._writer = self.output.open("a", encoding="utf-8")

    def close(self):
        self._writer.close()

    def __call__(self, event: Mapping[str, Any]):
        action = event["action"]
        if action["action"] == "complete" and not self.include_complete:
            self._skip("completion_not_selected")
            return
        count = int(event["task_count"])
        natural = int(event["rewritten_tasks"])
        if (event["natural_required"] and not self.include_fallback_rounds
                and count and natural != count):
            self._skip("incomplete_naturalization")
            return
        sample = make_round_record(
            task_id=str(event["task_id"]),
            round_index=int(event["round_index"]),
            question=str(event["question"]),
            eligibility=event["eligibility"],
            state=event["state"],
            prefix=event["history"],
            action=action,
            question_record=self.question,
            natural_task_count=natural,
            assigned_task_count=count,
            history_origin=event["history_origin"],
            max_o1_attempts=self.max_o1_attempts,
        )
        self._writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._writer.flush()
        self.written += 1

    def _skip(self, reason: str):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


__all__ = ["o1_input", "make_round_record", "RoundSink"]
