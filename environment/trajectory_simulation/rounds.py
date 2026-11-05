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
    if round_index != len(prefix) + 1:
        raise ValueError("round_index_does_not_follow_history_prefix")
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
        if not prefix:
            raise ValueError("completion_without_previous_result")
        from .contract import answers_equal
        last=prefix[-1]["result"]["subtask_results"][-1]
        contract={
            "collection_mode": "set" if last["result_kind"]=="entity_set" else "multiset"
        }
        if not answers_equal(
            action["params"]["answer"],last["result_value"],contract
        ):
            raise ValueError("complete_answer_not_supported_by_public_history")
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



def synthesize_rounds_jsonl(
    worlds_path: str | Path, realized_path: str | Path, output_path: str | Path,
    *, planner, executor, max_records: int | None = None,
    round_indices: set[int] | None = None,
    include_fallback_rounds: bool = False, include_complete: bool = True,
    max_o1_attempts: int = 10, max_attempts: int = 3,
    subagent_workers: int = 4, seed: int = 17, scene: str = "omnigaia",
    schedule: str = "compact", question_review: bool = True,
    quarantine_output_path: str | Path | None = None,
    rejected_output_path: str | Path | None = None,
    bank_output_path: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Precompute correct subtasks, then emit each independent O1 Main round.

    Failed natural rewrites never invalidate a correct subtask result. A valid
    teacher-substituted earlier round may remain in the current Main prompt.
    Main state composition releases sources and results in DAG topological order.
    """
    from collections import Counter
    import traceback
    from tqdm.auto import tqdm
    from .pipeline import _rows, _world_map, protect_paths, GoldTrajectoryError
    from .subtask_bank import build_subtask_bank

    if round_indices is not None and (
        not round_indices or any(
            not isinstance(index,int) or index<1 for index in round_indices
        )
    ):
        raise ValueError("round_indices must contain positive integers")
    if max_o1_attempts<1:
        raise ValueError("max_o1_attempts must be positive")
    if max_attempts<1 or subagent_workers<1:
        raise ValueError("max_attempts and subagent_workers must be positive")

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
    bank_file=Path(bank_output_path) if bank_output_path else None
    outs=[output,bad,quarantine]+([bank_file] if bank_file else [])
    protect_paths([worlds_path,realized_path],outs)
    if len({file.resolve() for file in outs}) != len(outs):
        raise ValueError("round and bank outputs must be distinct")
    for file in outs:
        file.parent.mkdir(parents=True,exist_ok=True)

    tasks=Counter()
    skips=Counter()
    history_origins=Counter()
    rewritten_tasks=0
    assigned_tasks=0
    attempted_rewrites=0
    attempted_assignments=0
    bank_fragments=0
    bank_natural_feedback=0
    rounds_considered=0
    eligible_rounds=0
    with output.open("w",encoding="utf-8") as good, bad.open(
        "w",encoding="utf-8"
    ) as reject, quarantine.open("w",encoding="utf-8") as isolate:
        # A bank sidecar is teacher-private; never use it as Main SFT input.
        from contextlib import nullcontext
        ctx=bank_file.open("w",encoding="utf-8") if bank_file else nullcontext()
        with ctx as bank_writer:
            for question in tqdm(
                rows,desc="Teacher-forced Main rounds",unit="task",
                disable=not progress,
            ):
                tid=str(question.get("task_id",""))
                try:
                    if tid not in worlds:
                        raise GoldTrajectoryError("input",["missing_matching_world"])
                    record,rendered=worlds[tid]
                    bank=build_subtask_bank(
                        question,record,rendered,planner=planner,executor=executor,
                        scene=scene,schedule=schedule,question_review=question_review,
                    )
                    bank_fragments+=len(bank["fragments"])
                    attempted_rewrites+=sum(
                        wave["naturalization"]["rewritten_tasks"]
                        for wave in bank["waves"]
                    )
                    attempted_assignments+=sum(
                        len(wave["fragment_ids"]) for wave in bank["waves"]
                    )
                    bank_natural_feedback+=sum(
                        bool(f["natural_feedback"]) for f in bank["fragments"].values()
                    )
                    if bank_writer is not None:
                        bank_writer.write(json.dumps({
                            "private_teacher_bank":True,
                            "task_id":tid,
                            "question":bank["question"],
                            "fragments":bank["fragments"],
                            "ready_order":bank["ready_order"],
                            "complete":bank["complete"],
                            "waves":bank["waves"],
                            "input_eligibility":bank["eligibility"],
                        },ensure_ascii=False)+"\n")
                        bank_writer.flush()

                    for wave_idx,wave in enumerate(bank["waves"]):
                        n=wave["round_index"]
                        if round_indices is not None and n not in round_indices:
                            continue
                        rounds_considered+=1
                        total=len(wave["fragment_ids"])
                        natural=wave["naturalization"]["rewritten_tasks"]
                        if not include_fallback_rounds and natural<total and (
                            getattr(planner,"style_provider",None)=="natural"
                            and not getattr(planner,"offline",False)
                        ):
                            skips["incomplete_naturalization"]+=1
                            continue
                        prior=bank["waves"][:wave_idx]
                        reconstructed=any(
                            item["naturalization"]["rewritten_tasks"] != len(item["fragment_ids"])
                            or not all(item["feedback_natural"])
                            for item in prior
                        )
                        origin=(
                            "teacher_reconstructed" if reconstructed else "natural"
                        )
                        row=make_round_record(
                            task_id=tid,
                            round_index=n,
                            question=bank["question"],
                            eligibility=bank["eligibility"],
                            state=wave["state"],
                            prefix=[item["step"] for item in prior],
                            action=wave["action"],
                            question_record=question,
                            natural_task_count=natural,
                            assigned_task_count=total,
                            history_origin=origin,
                            max_o1_attempts=max_o1_attempts,
                        )
                        good.write(json.dumps(row,ensure_ascii=False)+"\n")
                        good.flush()
                        eligible_rounds+=1
                        assigned_tasks+=total
                        rewritten_tasks+=natural
                        history_origins[origin]+=1

                    final=bank["complete"]
                    if include_complete and (
                        round_indices is None or final["round_index"] in round_indices
                    ):
                        rounds_considered+=1
                        prior=bank["waves"]
                        reconstructed=any(
                            wave["naturalization"]["rewritten_tasks"]!=len(wave["fragment_ids"])
                            or not all(wave["feedback_natural"])
                            for wave in prior
                        )
                        origin="teacher_reconstructed" if reconstructed else "natural"
                        row=make_round_record(
                            task_id=tid,
                            round_index=final["round_index"],
                            question=bank["question"],
                            eligibility=bank["eligibility"],
                            state=final["state"],
                            prefix=[wave["step"] for wave in prior],
                            action=final["action"],
                            question_record=question,
                            natural_task_count=0,
                            assigned_task_count=0,
                            history_origin=origin,
                            max_o1_attempts=max_o1_attempts,
                        )
                        good.write(json.dumps(row,ensure_ascii=False)+"\n")
                        good.flush()
                        eligible_rounds+=1
                        history_origins[origin]+=1
                    tasks["processed"]+=1
                except (GoldTrajectoryError,ValueError,KeyError,TypeError) as exc:
                    stage=exc.stage if isinstance(exc,GoldTrajectoryError) else "unexpected_error"
                    issues=exc.issues if isinstance(exc,GoldTrajectoryError) else [
                        f"{type(exc).__name__}: {exc}"
                    ]
                    isolated=stage in {
                        "question_eligibility","public_replay",
                        "public_executability","round_validation",
                    }
                    writer=isolate if isolated else reject
                    tasks["quarantined" if isolated else "rejected"]+=1
                    tasks["reason:"+stage]+=1
                    writer.write(json.dumps({
                        "task_id":tid,"reason":stage,"issues":issues,
                        "audit":exc.audit if isinstance(exc,GoldTrajectoryError)
                                else {"exception_type":type(exc).__name__,
                                      "traceback":traceback.format_exc(limit=8)},
                    },ensure_ascii=False)+"\n")
                    writer.flush()
    return {
        "requested_tasks":len(rows),
        "tasks_processed":tasks["processed"],
        "tasks_quarantined":tasks["quarantined"],
        "tasks_rejected":tasks["rejected"],
        "subtask_bank_fragments":bank_fragments,
        "subtask_bank_natural_feedback":bank_natural_feedback,
        "subtask_bank_feedback_natural_rate":(
            bank_natural_feedback/bank_fragments if bank_fragments else 0.0
        ),
        "rounds_considered":rounds_considered,
        "eligible_rounds":eligible_rounds,
        "rounds_skipped":sum(skips.values()),
        "round_yield":eligible_rounds/rounds_considered if rounds_considered else 0.0,
        "natural_main_tasks_rewritten":attempted_rewrites,
        "natural_main_tasks_total":attempted_assignments,
        "natural_action_rate":(
            attempted_rewrites/attempted_assignments
            if attempted_assignments else 0.0
        ),
        "exported_main_tasks_rewritten":rewritten_tasks,
        "exported_main_tasks_total":assigned_tasks,
        "history_origins":dict(history_origins),
        "round_skip_reasons":dict(skips),
        "task_failure_reasons":{
            key.removeprefix("reason:"):val for key,val in tasks.items()
            if key.startswith("reason:")
        },
        "request_counts":{
            "main":dict(getattr(planner,"request_counts",{})),
            "subagent":dict(getattr(executor,"request_counts",{})),
        },
        "output":str(output),
        "bank_output":str(bank_file) if bank_file else None,
        "rejected_output":str(bad),
        "quarantine_output":str(quarantine),
        "policy":"precomputed_clean_subtask_bank_then_main_round_sft",
        "selected_rounds":sorted(round_indices) if round_indices else "all",
    }


__all__=[
    "native_tool_spec","o1_input","make_round_record","synthesize_rounds_jsonl",
]
