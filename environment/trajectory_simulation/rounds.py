"""Round-level, teacher-forced Main SFT examples with native O1 prompt inputs.

Do not use this path to synthesize tool executions or incorrect Sub-Agent
feedback. The full frozen ProgramIR and public-witness preflight certify the
possible clean continuation; a decision round is exported independently after
its history prefix, action contract, and natural-language requirements pass.
"""
from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from train_maestro_8b.schema import validate_stage1_record
from .eligibility import digest
from .exporter import training_target, validate_public_episode


# Derive real OmniGAIA tools' static public descriptions/schemas from their
# class declarations. Importing live tool implementations would load providers,
# models and optional GPU packages during offline SFT export.
_NATIVE_TOOL_MODULES = {
    "GoogleSearchAction": "google_search.py",
    "ExtractUrlContentAction": "extract_url_jina.py",
    "ExecuteCodeAction": "execute_code.py",
    "VideoAnalysisAction": "video_analysis.py",
    "ImageAnalysisAction": "multimodal_toolkit.py",
    "ParseAudioAction": "audio_analysis.py",
}


@lru_cache(maxsize=16)
def native_tool_spec(name: str) -> SimpleNamespace:
    module = _NATIVE_TOOL_MODULES.get(name)
    if module is None:
        raise ValueError(f"unknown_native_o1_tool:{name}")
    path = Path(__file__).resolve().parents[2] / "benchmark" / "omnigaia" / "tools" / module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != name:
            continue
        values = {}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                key, value = stmt.target.id, stmt.value
            elif isinstance(stmt, ast.Assign) and len(stmt.targets)==1 and isinstance(stmt.targets[0], ast.Name):
                key, value = stmt.targets[0].id, stmt.value
            else:
                continue
            if key in {"name","description","parameters"} and value is not None:
                values[key] = ast.literal_eval(value)
        if all(k in values for k in ("name","description","parameters")):
            if values["name"] != name:
                raise ValueError("native_tool_name_disagrees_with_class")
            return SimpleNamespace(**values)
    raise ValueError(f"native_o1_tool_schema_unavailable:{name}")


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
    tools = [native_tool_spec(str(name)) for name in state["available_tools"]]
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
    from .history import format_o1_subtask_history
    if set(state) != {"question","subtask_history","available_tools","evidence_store"}:
        raise ValueError("unexpected_main_state_fields")
    if state["subtask_history"] != format_o1_subtask_history(prefix):
        raise ValueError("round_history_not_equal_to_verified_prefix")
    if not str(state["question"]).startswith(question):
        raise ValueError("round_question_not_equal_to_frozen_question")
    seen = {}
    for step in prefix:
        for reply in step["result"]["subtask_results"]:
            for observation in reply["observed_evidence"]:
                source_id = observation["resource_id"]
                if source_id in seen and seen[source_id] != observation:
                    raise ValueError("inconsistent_round_history_sources")
                seen[source_id] = observation
    if state["evidence_store"] != seen:
        raise ValueError("round_evidence_store_not_equal_to_verified_prefix")
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
        self.considered = 0
        self.natural_task_total = 0
        self.natural_task_rewritten = 0
        self.history_origin_counts: dict[str,int] = {}
        self.skipped = 0
        self.skip_reasons: dict[str, int] = {}
        self._writer = self.output.open("a", encoding="utf-8")

    def close(self):
        self._writer.close()

    def __call__(self, event: Mapping[str, Any]):
        action = event["action"]
        self.considered += 1
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
        self.natural_task_total += count
        self.natural_task_rewritten += natural
        origin = event["history_origin"]
        self.history_origin_counts[origin] = self.history_origin_counts.get(origin,0)+1

    def _skip(self, reason: str):
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def synthesize_rounds_jsonl(
    worlds_path: str | Path, realized_path: str | Path, output_path: str | Path,
    *, planner, executor, max_records: int | None = None,
    round_indices: set[int] | None = None, history_style: str = "natural",
    include_fallback_rounds: bool = False, include_complete: bool = True,
    max_o1_attempts: int = 10, max_attempts: int = 3,
    subagent_workers: int = 4, seed: int = 17, scene: str = "omnigaia",
    schedule: str = "compact", question_review: bool = True,
    quarantine_output_path: str | Path | None = None,
    rejected_output_path: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Stream independent clean Main decisions, not all-or-nothing episodes.

    Previously validated rounds remain on disk if a later round fails.
    Previous unselected decisions and correct Sub-Agent feedback can be
    reconstructed from the teacher's source-backed gold witness. This does
    NOT train on an erroneous state or copy future gold into the Main prompt.
    """
    from collections import Counter
    from traceback import format_exc
    from tqdm.auto import tqdm
    from .pipeline import (
        _rows, _world_map, protect_paths, synthesize_one, GoldTrajectoryError,
    )
    if round_indices is not None and (
        not round_indices or any(not isinstance(n,int) or n < 1 for n in round_indices)
    ):
        raise ValueError("round_indices must be positive integers")
    if history_style not in {"teacher", "natural"}:
        raise ValueError("history_style must be teacher or natural")
    if max_o1_attempts < 1:
        raise ValueError("max_o1_attempts must be positive")
    rows=_rows(realized_path)
    if max_records is not None:
        rows=rows[:max_records]
    worlds=_world_map(worlds_path,seed=seed)
    output=Path(output_path)
    bad=Path(rejected_output_path) if rejected_output_path else output.with_suffix(
        output.suffix+".rejected.jsonl"
    )
    quarantine=Path(quarantine_output_path) if quarantine_output_path else output.with_suffix(
        output.suffix+".quarantine.jsonl"
    )
    protect_paths([worlds_path,realized_path],[output,bad,quarantine])
    if len({output.resolve(),bad.resolve(),quarantine.resolve()})!=3:
        raise ValueError("round output and diagnostics must use distinct paths")
    for path in (output,bad,quarantine):
        path.parent.mkdir(parents=True,exist_ok=True)
    # Truncate once. Each successful round then becomes a durable JSONL line.
    output.write_text("",encoding="utf-8")
    requests=Counter()
    stats=Counter()
    skip_counts=Counter()
    origin_counts=Counter()
    attempted=written=skipped=natural_total=natural_written=0
    with bad.open("w",encoding="utf-8") as bad_writer, quarantine.open(
        "w",encoding="utf-8"
    ) as quarantine_writer:
        for item in tqdm(rows,desc="Clean Main decision rounds",unit="task",disable=not progress):
            tid=str(item.get("task_id",""))
            sink=RoundSink(
                output,question=item,
                include_fallback_rounds=include_fallback_rounds,
                include_complete=include_complete,
                max_o1_attempts=max_o1_attempts,
            )
            try:
                if tid not in worlds:
                    raise GoldTrajectoryError("input", ["missing_matching_world"])
                synthesize_one(
                    item,*worlds[tid],planner,executor,
                    max_attempts=max_attempts,subagent_workers=subagent_workers,
                    scene=scene,question_review=question_review,schedule=schedule,
                    round_sink=sink,selected_rounds=round_indices,
                    history_style=history_style,
                    stop_after_round=max(round_indices) if round_indices else None,
                )
                stats["tasks_processed"]+=1
            except (GoldTrajectoryError,ValueError,KeyError,TypeError) as exc:
                stage=exc.stage if isinstance(exc,GoldTrajectoryError) else "unexpected_error"
                reasons=exc.issues if isinstance(exc,GoldTrajectoryError) else [
                    f"{type(exc).__name__}: {exc}"
                ]
                isolated=stage in {
                    "question_eligibility","public_replay","public_executability",
                    "round_validation",
                }
                writer=quarantine_writer if isolated else bad_writer
                stats["tasks_quarantined" if isolated else "tasks_rejected"]+=1
                writer.write(json.dumps({
                    "task_id":tid,"reason":stage,"issues":reasons,
                    "rounds_already_written":sink.written,
                    "audit":exc.audit if isinstance(exc,GoldTrajectoryError) else {
                        "exception_type":type(exc).__name__,
                        "traceback":format_exc(limit=8),
                    },
                },ensure_ascii=False)+"\n")
                writer.flush()
                requests[stage]+=1
            finally:
                sink.close()
                attempted+=sink.considered
                written+=sink.written
                skipped+=sink.skipped
                natural_total+=sink.natural_task_total
                natural_written+=sink.natural_task_rewritten
                skip_counts.update(sink.skip_reasons)
                origin_counts.update(sink.history_origin_counts)
    return {
        "requested_tasks":len(rows),
        "tasks_processed":stats["tasks_processed"],
        "tasks_quarantined":stats["tasks_quarantined"],
        "tasks_rejected":stats["tasks_rejected"],
        "rounds_considered":attempted,
        "eligible_rounds":written,
        "rounds_skipped":skipped,
        "round_yield":written/attempted if attempted else 0.0,
        "natural_main_tasks_rewritten":natural_written,
        "natural_main_tasks_total":natural_total,
        "natural_action_rate":natural_written/natural_total if natural_total else 0.0,
        "history_origins":dict(origin_counts),
        "round_skip_reasons":dict(skip_counts),
        "task_failure_reasons":dict(requests),
        "request_counts":{
            "main":dict(getattr(planner,"request_counts",{})),
            "subagent":dict(getattr(executor,"request_counts",{})),
        },
        "output":str(output),"rejected_output":str(bad),
        "quarantine_output":str(quarantine),
        "policy":"clean_gold_teacher_forced_round_sft",
        "history_style":history_style,
        "selected_rounds":sorted(round_indices) if round_indices else "all",
    }


__all__ = [
    "native_tool_spec","o1_input","make_round_record","RoundSink",
    "synthesize_rounds_jsonl",
]

