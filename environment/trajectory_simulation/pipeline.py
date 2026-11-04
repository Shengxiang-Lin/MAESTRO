"""Stream native O1 episodes with immutable, private world supervision."""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from tqdm.auto import tqdm

from environment.question_generation.backend import BackendError
from environment.task_construction.rendering.pipeline import render_task_record

from .compiler import (
    GoldCompilationError, compile_gold_solution, public_plan,
    ready_subtasks, json_value,
)
from .prompts import (
    executor_prompt, final_prompt, planner_prompt, summary_prompt,
)
from .validator import (
    answers_equal,
    validate_final, validate_gold_trajectory, validate_main_summary,
    validate_parallel_plan, validate_subagent_result,
)
from .grounding import grouped_evidence


class GoldTrajectoryError(RuntimeError):
    def __init__(self, stage: str, issues: list[str], audit=None) -> None:
        self.stage = stage
        self.issues = issues
        self.audit = audit or {}
        super().__init__(f"{stage}: {', '.join(issues)}")


def _rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _world_map(path: str | Path, seed: int = 17) -> dict[str, tuple[dict[str, Any], Any]]:
    result = {}
    for index, record in enumerate(_rows(path)):
        rendered = render_task_record(record, seed=seed + index)
        if rendered.task_id in result:
            raise GoldTrajectoryError("world_index", [f"duplicate world {rendered.task_id}"])
        result[rendered.task_id] = (record, rendered)
    return result


def _with_feedback(prompt: str, issues: list[str]) -> str:
    """Repair hint only; validation and gold evidence stay authoritative."""
    if not issues:
        return prompt
    return prompt + "\n\nPrevious output was rejected for: " + json.dumps(issues) + (
        "\nRegenerate the requested JSON without changing any frozen task IDs, "
        "resource IDs, authoritative source facts or dependency constraints. "
        "Rewrite rejected summary statements using the specific reasons above; "
        "do not repeat or preserve a contradicted claim from your previous response. "
        "Return only the JSON fields requested by this stage."
    )


def _stage(call,validate,*,stage,max_attempts,diagnostics=None):
    last=[];attempts=[]
    for attempt in range(1,max_attempts+1):
        value=None
        try:
            value=call(last)
            if not isinstance(value,Mapping):last=["non_object_model_response"]
            else:
                okay,issues=validate(value)
                if okay:return dict(value)
                last=issues
        except (BackendError,ValueError,TypeError,KeyError) as exc:
            last=[f"{type(exc).__name__}:{str(exc)[:180]}"]
        audit={"attempt":attempt,"response":json_value(value),"issues":last}
        if stage=="main_final_answer" and isinstance(value,Mapping) and diagnostics:
            audit["answer_validation"]={
                "accepted":answers_equal(value.get("answer"),diagnostics["expected_answer"],
                                         diagnostics["answer_contract"]),
            }
        attempts.append(audit)
    raise GoldTrajectoryError(stage,last,{"attempts":attempts,**(diagnostics or {})})


def _reviewed(role,value,validate,claims,fields,*,question,pending=(),allowed=(),context=None):
    okay,issues=validate(value)
    if not okay:return okay,issues
    from .grounding import semantic_review
    text="\n".join(str(value.get(field,"")) for field in fields)
    return semantic_review(role,text,claims,question=question,pending_ids=pending,allowed_ids=allowed,context=context)


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    result={
        k: task[k] for k in (
            "subtask_id", "kind", "resource_id", "source_type", "role", "agent",
            "available_tool_families", "depends_on",
            "depends_on_any",
        )
    }
    if task.get("kind")=="discover":result["discovery_terms"]=task.get("discovery_terms",[])
    return result


def _synthesize_reference_one(
    realized: Mapping[str, Any], world_record: Mapping[str, Any],
    rendered: Any, planner: Any, executor: Any, *,
    max_attempts: int = 3, subagent_workers: int = 4,
) -> dict[str, Any]:
    """Generate one official trajectory by repeated Main/Sub-Agent calls."""
    if max_attempts < 1 or subagent_workers < 1:
        raise ValueError("max_attempts and subagent_workers must be positive")
    try:
        gold = compile_gold_solution(realized, world_record, rendered)
    except (ValueError, KeyError, TypeError) as exc:
        raise GoldTrajectoryError("gold_compilation", [str(exc)]) from exc
    graph = public_plan(gold)
    tasks_by_id = {item["subtask_id"]: item for item in gold["tasks"]}
    completed: set[str] = set()
    previous_results: dict[str, dict[str, Any]] = {}
    history_summaries: list[str] = []
    rounds: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=subagent_workers,
                            thread_name_prefix="gold-subagent") as pool:
        while len(completed) < len(tasks_by_id):
            ready = ready_subtasks(gold, completed)
            if not ready:
                raise GoldTrajectoryError("scheduling", ["no_ready_tasks_or_dependency_cycle"])
            ready_ids = [item["subtask_id"] for item in ready]
            pending_ids = [item["subtask_id"] for item in gold["tasks"]
                           if item["subtask_id"] not in completed]
            round_index = len(rounds) + 1
            plan_payload = {
                "question": gold["question"],
                "public_plan": graph,
                "completed_subtask_ids": sorted(completed),
                "completed_subtask_results": [
                    previous_results[item["subtask_id"]] for item in gold["tasks"]
                    if item["subtask_id"] in completed
                ],
                "history_summaries": history_summaries,
                "ready_tasks": [_public_task(task) for task in ready],
                "pending_task_ids": pending_ids,
            }
            allocation = _stage(
                lambda feedback: planner.plan(
                    plan_payload, _with_feedback(planner_prompt(plan_payload), feedback)),
                lambda value: validate_parallel_plan(value, ready),
                stage=f"round_{round_index}:main_plan", max_attempts=max_attempts,
            )
            allocated = {item["subtask_id"]: item
                         for item in allocation["parallel_subtasks"]}

            def execute_task(task: Mapping[str, Any]) -> dict[str, Any]:
                task_id = task["subtask_id"]
                dependency_ids=set(task["depends_on"])
                for group in task.get("depends_on_any",()):
                    dependency_ids.update(set(group) & completed)
                dependency_results = [
                    previous_results[dep] for dep in sorted(dependency_ids)
                ]
                if task.get("kind")=="discover":
                    descriptor=task["gold_answer_claims"]
                    result={"subtask_id":task_id,"status":"success","answer":descriptor,
                        "evidence_source_ids":[task["resource_id"]],
                        "answer_summary":f"Located {descriptor[0]['title']} ({task['resource_id']}); only its resource descriptor is available.",
                        "trajectory_summary":"Used already available clues to retrieve the resource descriptor; its contents remain unread."}
                    okay,issues=validate_subagent_result(result,task)
                    if not okay:raise GoldTrajectoryError("discovery_execution",issues)
                    return result
                execution_payload = {
                    "question": gold["question"],
                    "assigned_subtask": allocated[task_id],
                    "dependency_results": dependency_results,
                    "task": task,
                }
                inspection_claims,inspection_state=grouped_evidence(tasks_by_id,[
                    {"subtask_id":task_id,"answer":task["gold_answer_claims"]}])
                inspection_claims[0]["source_observation"]=task["observation"]
                return _stage(
                    lambda feedback: executor.execute(
                        execution_payload,
                        _with_feedback(executor_prompt(execution_payload), feedback)),
                    lambda value: _reviewed(executor,value,
                        lambda v:validate_subagent_result(v,task,question=gold["question"]),
                        inspection_claims,
                        ("answer_summary","trajectory_summary"),question=gold["question"],
                        allowed=[task["resource_id"]],
                        context={"kind":"inspect","resource_id":task["resource_id"],
                                 "source_type":task["source_type"],"agent":task["agent"],
                                 **inspection_state}),
                    stage=f"round_{round_index}:subagent:{task_id}",
                    max_attempts=max_attempts,
                )

            # Each independent subtask has its own Sub-Agent LLM request.
            # Collect in fixed Gold DAG order although requests run concurrently.
            futures = [pool.submit(execute_task, item) for item in ready]
            responses = []
            for future in futures:
                responses.append(future.result())
            known=[*previous_results.values(),*responses]
            known_claims,review_state=grouped_evidence(tasks_by_id,known)
            summary_payload = {
                "question": gold["question"],
                "history_summaries": history_summaries,
                "current_results": responses,
                "observed_evidence":known_claims,
                **review_state,
                "pending_task_ids": [
                    item for item in pending_ids if item not in set(ready_ids)
                ],
            }
            known_ids={sid for result in known for sid in result["evidence_source_ids"]}
            summary = _stage(
                lambda feedback: planner.summarize(
                    summary_payload, _with_feedback(summary_prompt(summary_payload), feedback)),
                lambda value:_reviewed(planner,value,
                    lambda v:validate_main_summary(v,claims=known_claims,
                        pending_ids=summary_payload["pending_task_ids"],allowed_ids=known_ids,question=gold["question"]),
                    known_claims,("main_summary",),question=gold["question"],
                    pending=summary_payload["pending_task_ids"],allowed=known_ids,
                    context={"kind":"main_round_summary",**review_state}),
                stage=f"round_{round_index}:main_summary",
                max_attempts=max_attempts,
            )
            before = sorted(completed)
            for item in responses:
                completed.add(item["subtask_id"])
                previous_results[item["subtask_id"]] = item
            history_summaries.append(summary["main_summary"])
            rounds.append({
                "round": round_index,
                "completed_before": before,
                "current_parallel_task_ids": ready_ids,
                "main_action": allocation,
                "subagent_results": responses,
                "main_summary": summary["main_summary"],
                "completed_after": sorted(completed),
                "pending_after": [
                    item["subtask_id"] for item in gold["tasks"]
                    if item["subtask_id"] not in completed
                ],
            })

    final_payload = {
        "question": gold["question"],
        "public_plan": graph,
        "all_results": [previous_results[t["subtask_id"]] for t in gold["tasks"]],
        "history_summaries": history_summaries,
        "observed_evidence":known_claims,
        **review_state,
        # Used by the offline passthrough role only; deliberately NOT included
        # in the real LLM Main prompt.
        "expected_answer": gold["final_answer"],
    }
    final = _stage(
        lambda feedback: planner.finalize(
            final_payload, _with_feedback(final_prompt(final_payload), feedback)),
        lambda value:_reviewed(planner,value,
            lambda v:validate_final(v,gold["final_answer"],gold["answer_contract"]),
            [*known_claims,{"verified_final_answer":value.get("answer")}],
            ("summary",),question=gold["question"],
            allowed={sid for result in previous_results.values() for sid in result["evidence_source_ids"]},
            context={"kind":"main_final_summary",**review_state,
                     "verified_final_answer":value.get("answer")}),
        stage="main_final_answer", max_attempts=max_attempts,
        diagnostics={"expected_answer":gold["final_answer"],"answer_contract":gold["answer_contract"],
                     "completed_rounds":rounds,"public_plan":graph},
    )
    validation = validate_gold_trajectory(gold, rounds, final)
    if not validation["accepted"]:
        raise GoldTrajectoryError("gold_trajectory_validation", validation["issues"])
    return {
        "task_id": gold["task_id"],
        "question": gold["question"],
        "plan": graph,
        "steps": rounds,
        "final": final,
        "validation": validation,
        "gold_supervision": {
            "proof_source_ids": gold["proof_source_ids"],
            "required_fact_ids":gold["required_fact_ids"],
            "answer_contract":gold["answer_contract"],
            "acquisition_validation":gold["acquisition_validation"],
            "acquisition_alternatives":gold["acquisition_alternatives"],
            "required_resource_ids": gold["required_resource_ids"],
            "gold_parallel_waves": gold["gold_parallel_waves"],
            "symbolic_trace": gold["symbolic_trace"],
            "symbolic_answer": gold["symbolic_answer"],
            "final_answer": gold["final_answer"],
        },
        "metadata": {
            "clean_trajectory": True,
            "discovery_responses_environment_authoritative":True,
            "text_grounding_review":{
                "planner":"offline_deterministic" if getattr(planner,"offline",False) else "llm_review",
                "executor":"offline_deterministic" if getattr(executor,"offline",False) else "llm_review"},
            "two_role_teacher": True,
            "subtasks_compiled_from_frozen_gold": True,
            "oracle_answer_not_shown_to_main_llm": True,
            "resource_observations_from_frozen_world": True,
            "no_adversarial_feedback": True,
        },
    }


def synthesize_one(
    realized, world_record, rendered, planner, executor, *,
    max_attempts=3, subagent_workers=4,
):
    from .o1_pipeline import synthesize_one as synthesize_o1
    return synthesize_o1(
        realized, world_record, rendered, planner, executor,
        max_attempts=max_attempts, subagent_workers=subagent_workers,
    )


def plan_trajectories_jsonl(
    worlds_path: str | Path, realized_path: str | Path,
    output_path: str | Path, *, max_records: int | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Inspect gold graphs without LLM calls."""
    realized = _rows(realized_path)
    if max_records is not None:
        realized = realized[:max_records]
    worlds = _world_map(worlds_path, seed=seed)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    rejection_reasons = Counter()
    with destination.open("w", encoding="utf-8") as writer:
        for item in realized:
            task_id = str(item.get("task_id"))
            try:
                if task_id not in worlds:
                    raise GoldCompilationError("no matching world")
                gold = compile_gold_solution(item, *worlds[task_id], full_environment=True)
                output = {
                    "status": "ready", "task_id": task_id,
                    "gold_plan": public_plan(gold),
                    "gold_parallel_waves": gold["gold_parallel_waves"],
                    "proof_source_ids": gold["proof_source_ids"],
                    "final_answer": gold["final_answer"],
                }
                passed += 1
            except (GoldCompilationError, ValueError, TypeError, KeyError) as exc:
                rejection_reasons[type(exc).__name__] += 1
                output = {
                    "status": "rejected", "task_id": task_id, "reason": str(exc),
                }
            writer.write(json.dumps(output, ensure_ascii=False) + "\n")
    return {
        "planned": len(realized),
        "ready": passed,
        "rejected": len(realized) - passed,
        "rejection_reasons": dict(rejection_reasons),
        "output": str(destination),
    }


def synthesize_trajectories_jsonl(
    worlds_path: str | Path, realized_path: str | Path,
    output_path: str | Path, *, planner: Any, executor: Any,
    max_records: int | None = None, max_attempts: int = 3,
    subagent_workers: int = 4, rejected_output_path: str | Path | None = None,
    progress: bool = True, flush_every: int = 5, seed: int = 17,
    model_roles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if max_attempts < 1 or subagent_workers < 1 or flush_every < 1:
        raise ValueError("max_attempts, subagent_workers and flush_every must be >= 1")
    realized = _rows(realized_path)
    if max_records is not None:
        realized = realized[:max_records]
    worlds = _world_map(worlds_path, seed=seed)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = (Path(rejected_output_path) if rejected_output_path
                     else target.with_suffix(target.suffix + ".rejected.jsonl"))
    if rejected_path.resolve() == target.resolve():
        raise ValueError("rejected output must differ from accepted trajectory output")
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0
    reasons = Counter()
    # One whole trajectory is the atomic output unit; each internal round
    # is causally sequential, while its independent Sub-Agent calls are parallel.
    with (
        target.open("w", encoding="utf-8") as good,
        rejected_path.open("w", encoding="utf-8") as bad,
    ):
        bar = tqdm(realized, desc="O1 trajectory simulation",
                   unit="task", disable=not progress)
        for item in bar:
            task_id = str(item.get("task_id", ""))
            try:
                if task_id not in worlds:
                    raise GoldTrajectoryError("input", ["missing_matching_world"])
                record, rendered = worlds[task_id]
                payload = synthesize_one(
                    item, record, rendered, planner, executor,
                    max_attempts=max_attempts, subagent_workers=subagent_workers,
                )
            except (GoldTrajectoryError, ValueError, KeyError, TypeError) as exc:
                rejected += 1
                stage = exc.stage if isinstance(exc, GoldTrajectoryError) else "unexpected_error"
                issues = exc.issues if isinstance(exc, GoldTrajectoryError) else [str(exc)]
                reasons[stage] += 1
                bad.write(json.dumps({
                    "task_id": task_id,
                    "reason": stage,
                    "issues": issues,
                    "audit":json_value(exc.audit) if isinstance(exc,GoldTrajectoryError) else {},
                }, ensure_ascii=False) + "\n")
            else:
                accepted += 1
                good.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if (accepted + rejected) % flush_every == 0:
                good.flush()
                bad.flush()
            if progress:
                bar.set_postfix(accepted=accepted, rejected=rejected)
        good.flush()
        bad.flush()
    return {
        "requested": len(realized),
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / len(realized) if realized else 0.,
        "rejection_reasons": dict(reasons),
        "output": str(target),
        "rejected_output": str(rejected_path),
        "model_roles": dict(model_roles or {}),
        "subagent_workers": subagent_workers,
        "policy": "orchestra_o1_main_controlled_two_role_simulation",
    }


__all__ = [
    "GoldTrajectoryError", "synthesize_one",
    "plan_trajectories_jsonl", "synthesize_trajectories_jsonl",
]
