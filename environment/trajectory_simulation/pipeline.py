"""Fragment-guided gold O1 trajectories over existing frozen task solutions.

Main sees an O1 public state plus *private generation guidance*. ProgramIR
fragments are assigned in dependency order, with independent ones in parallel.
Sub-Agent LLMs write only compressed language; the program supplies the exact
intermediate answers and their private evidence bindings.
"""
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
    GoldCompilationError, compile_gold_solution, json_value,
    private_guidance, ready_fragments,
)
from .contract import (
    TOOLS, answer_string, feedback_prompt, main_prompt, public_action,
    validate_action, validate_feedback,
)
from .history import format_o1_subtask_history
from .realization import validate_bound_action


class GoldTrajectoryError(RuntimeError):
    def __init__(self, stage: str, issues: list[str], audit: Any = None):
        self.stage = stage
        self.issues = list(issues)
        self.audit = audit or {}
        super().__init__(f"{stage}: {', '.join(self.issues)}")


def _rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _world_map(path: str | Path, seed: int) -> dict[str, tuple[dict[str, Any], Any]]:
    result = {}
    for index, record in enumerate(_rows(path)):
        rendered = render_task_record(record, seed=seed + index)
        if rendered.task_id in result:
            raise GoldTrajectoryError("world_index", [f"duplicate task_id {rendered.task_id}"])
        result[rendered.task_id] = (record, rendered)
    return result


def _stage(call, validate, *, stage: str, max_attempts: int,
           diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    attempts = []
    issues = []
    for index in range(max_attempts):
        value: Any = None
        try:
            value = call(issues)
            if not isinstance(value, Mapping):
                issues = ["response_not_json_object"]
            else:
                okay, issues = validate(value)
                if okay:
                    return dict(value)
        except (BackendError, ValueError, TypeError, KeyError) as exc:
            issues = [f"{type(exc).__name__}:{str(exc)[:220]}"]
        attempts.append({
            "attempt": index + 1, "issues": list(issues),
            "response": json_value(value),
        })
    raise GoldTrajectoryError(stage, issues, {"attempts": attempts, **(diagnostics or {})})


def _with_feedback(prompt: str, issues: list[str]) -> str:
    if not issues:
        return prompt
    return (
        prompt + "\n\nYour previous JSON was rejected for " +
        json.dumps(issues, ensure_ascii=False) +
        ". Regenerate only this action/summary; preserve the frozen fragment "
        "order, assigned evidence and private/public information boundaries."
    )


def _public_state(question: str, trajectory: list[dict[str, Any]], *, tools=TOOLS, resources=()) -> dict[str, Any]:
    """O1-compatible observation: question + cumulative formatted subtask history.

    Gold fragment graph/answers remain exclusively in the teacher guidance and
    private supervision, never in the public Main state.
    """
    return {
        "question": question + ("\n\n[Provided reference records]\n" + "\n".join(f"{r['title']} ({r['resource_id']}, {r['source_type']})" for r in resources) if resources else ""),
        "subtask_history": format_o1_subtask_history(trajectory),
        "available_tools": list(tools),
    }


def _model(role: Any) -> str:
    return str(getattr(getattr(role, "backend", None), "model", "passthrough"))


def synthesize_one(
    realized: Mapping[str, Any], world_record: Mapping[str, Any], rendered: Any,
    planner: Any, executor: Any, *, max_attempts: int = 3,
    subagent_workers: int = 4, scene: str = "omnigaia",
) -> dict[str, Any]:
    if max_attempts < 1 or subagent_workers < 1:
        raise ValueError("max_attempts and subagent_workers must be >= 1")
    try:
        gold = compile_gold_solution(realized, world_record, rendered, scene=scene)
    except (GoldCompilationError, ValueError, TypeError, KeyError) as exc:
        raise GoldTrajectoryError("gold_compilation", [str(exc)]) from exc

    trajectory: list[dict[str, Any]] = []
    private_rounds = []
    completed: set[str] = set()
    known_results: dict[str, dict[str, Any]] = {}
    fragment_by_id = {f["fragment_id"]: f for f in gold["fragments"]}

    with ThreadPoolExecutor(max_workers=subagent_workers,
                            thread_name_prefix="gold-fragment") as pool:
        # Each delegation must complete a full, nonempty ready wave. The
        # last extra iteration is the Main's final complete action.
        for round_index in range(1, len(gold["fragments"]) + 2):
            guidance = private_guidance(gold, completed)
            ready = ready_fragments(gold, completed)
            public_state = _public_state(gold["question"], trajectory, tools=gold["available_tools"], resources=gold["provided_resources"])
            payload = {
                "public_state": public_state,
                "gold_guidance": guidance,
                "completed_subtask_contracts": list(known_results.values()),
                "visible_resources": list({r["resource_id"]: r for r in [
                    *gold["provided_resources"],
                    *({"resource_id": o["resource_id"], "source_type": o["source_type"], "title": o["title"]}
                      for f in gold["fragments"] if f["fragment_id"] in completed for o in f["observations"])
                ]}.values()),
                # Offline passthrough test mode only. NOT included in main_prompt.
                "offline_oracle_answer": gold["final_answer"],
            }

            def check_action(value):
                okay, issues = validate_action(
                    value, guidance, gold["final_answer"], gold["answer_contract"],
                    public_state=public_state,
                )
                if not okay:
                    return okay, issues
                return validate_bound_action(value, payload)

            action = _stage(
                lambda issues: planner.decide(
                    payload, _with_feedback(main_prompt(payload), issues)
                ),
                check_action,
                stage=f"round_{round_index}:main_action",
                max_attempts=max_attempts,
                diagnostics={"completed_trajectory": json_value(trajectory)},
            )
            wire = public_action(action)
            if guidance["may_complete"]:
                trajectory.append({
                    "attempt": round_index, **wire,
                    "state_before": public_state,
                    "result": {"done": True, "success": True,
                               "answer": wire["params"]["answer"]},
                })
                if completed != set(fragment_by_id):
                    raise GoldTrajectoryError("validation", ["gold_graph_incomplete"])
                return {
                    "task_id": gold["task_id"],
                    "question": gold["question"],
                    "instruction": gold["question"],
                    "main_model": _model(planner),
                    "sub_models": [_model(executor)],
                    "attempts": len(trajectory),
                    "success": True,
                    "trajectory": trajectory,
                    "validation": {
                        "accepted": True, "final_answer_verified": True,
                        "language_contract_verified": True,
                        "upstream_question_semantics_revalidated": False,
                        "completed_gold_fragments": len(completed),
                        "expected_gold_fragments": len(fragment_by_id),
                        "fragment_dependencies_respected": True,
                    },
                    "gold_supervision": {
                        "policy": gold["policy"],
                        "composition_fragments": [{
                            "fragment_id": f["fragment_id"],
                            "output_node_id": f["output_node_id"],
                            "depends_on": f["depends_on"],
                            "resource_ids": f["resource_ids"],
                            "support_resource_ids": f["support_resource_ids"],
                            "operations": f["operations"],
                            "expected_answer": f["expected_answer"],
                        } for f in gold["fragments"]],
                        "parallel_waves": gold["waves"],
                        "symbolic_trace": gold["symbolic_trace"],
                        "final_proof_source_ids": gold["final_proof_source_ids"],
                        "final_answer": gold["final_answer"],
                        "round_bindings": private_rounds,
                    },
                    "metadata": {
                        "format": "orchestra_o1",
                        "gold_policy": "first_stage_fragment_guided",
                        "clean_trajectory": True,
                        "two_role_teacher": True,
                        "scene": scene,
                        "language_generation": "LLM_style_selection_with_verified_semantic_projection",
                        "main_public_state_matches_o1": True,
                        "gold_guidance_private": True,
                        "subagent_answers_programmatic": True,
                        "raw_tool_traces_generated": False,
                        "attacks_simulated": False,
                        "language_validation": (
                            "deterministic_offline_reference"
                            if getattr(planner, "offline", False) and getattr(executor, "offline", False)
                            else "bounded_language_realization_of_verified_execution"
                        ),
                    },
                }

            assignments = wire["params"]["tasks"]
            if len(assignments) != len(ready):
                raise GoldTrajectoryError("validation", ["unexpected_task_count"])

            def generate_subagent(index: int, fragment: dict[str, Any]) -> dict[str, Any]:
                assigned = assignments[index]
                predecessor_results = [
                    known_results[dep] for dep in fragment["depends_on"]
                ]
                sub_payload = {
                    "question": gold["question"],
                    "task_instruction": assigned["task_instruction"],
                    "assigned_task": assigned,
                    "dependency_results": predecessor_results,
                    "fragment": fragment,
                }

                def check_feedback(value):
                    return validate_feedback(value, fragment, strict=True)

                language = _stage(
                    lambda issues: executor.respond(
                        sub_payload,
                        _with_feedback(feedback_prompt(sub_payload), issues),
                    ),
                    check_feedback,
                    stage=f"round_{round_index}:subagent:{index}",
                    max_attempts=max_attempts,
                    diagnostics={
                        "completed_trajectory": json_value(trajectory),
                        "fragment_id": fragment["fragment_id"],
                    },
                )
                # The exact answer is NOT generated or copied by the LLM.
                fixed_answer = answer_string(fragment["expected_answer"])
                return {
                    "subtask_index": index,
                    "task_instruction": assigned["task_instruction"],
                    "model": _model(executor),
                    "tools_assigned": assigned["tools"],
                    "done": True,
                    "finish_result": {
                        "status": "done",
                        "result": fixed_answer,
                        "summary": language["summary"],
                    },
                    "trace_summary": language["trace_summary"],
                }

            futures = [
                pool.submit(generate_subagent, index, fragment)
                for index, fragment in enumerate(ready)
            ]
            # Run in parallel but publish all results in deterministic fragment order.
            replies = [future.result() for future in futures]
            for reply_index, (fragment, reply) in enumerate(zip(ready, replies)):
                known_results[fragment["fragment_id"]] = {
                    "fragment_id": fragment["fragment_id"],
                    "public_ref": f"Attempt {round_index}, subtask {reply_index + 1}",
                    "answer": reply["finish_result"]["result"],
                    "answer_kind": fragment["answer_kind"],
                    "summary": reply["finish_result"]["summary"],
                }
                completed.add(fragment["fragment_id"])
            private_rounds.append({
                "round": round_index,
                "task_bindings": [{
                    "task_index": index, "fragment_id": f["fragment_id"],
                    "output_node_id": f["output_node_id"],
                    "resource_ids": f["resource_ids"],
                    "support_resource_ids": f["support_resource_ids"],
                } for index, f in enumerate(ready)],
            })
            trajectory.append({
                "attempt": round_index, **wire,
                "state_before": public_state,
                "result": {
                    "total_subtasks": len(replies),
                    "subtask_results": replies,
                },
            })
    raise GoldTrajectoryError("round_budget", ["gold_fragments_not_completed"])


def plan_trajectories_jsonl(
    worlds_path: str | Path, realized_path: str | Path,
    output_path: str | Path, *, max_records: int | None = None, seed: int = 17, scene: str = "omnigaia",
) -> dict[str, Any]:
    realized = _rows(realized_path)
    if max_records is not None:
        realized = realized[:max_records]
    worlds = _world_map(worlds_path, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    reasons: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as writer:
        for item in realized:
            task_id = str(item.get("task_id", ""))
            try:
                if task_id not in worlds:
                    raise GoldCompilationError("world_not_found")
                plan = compile_gold_solution(item, *worlds[task_id], scene=scene)
                row = {
                    "task_id": task_id, "status": "ready",
                    "fragment_graph": [{
                        "fragment_id": f["fragment_id"],
                        "output_node_id": f["output_node_id"],
                        "depends_on": f["depends_on"],
                        "resource_ids": f["resource_ids"],
                        "expected_answer": f["expected_answer"],
                    } for f in plan["fragments"]],
                    "gold_waves": plan["waves"],
                    "final_answer": plan["final_answer"],
                }
                accepted += 1
            except (GoldCompilationError, ValueError, TypeError, KeyError) as exc:
                reasons[type(exc).__name__] += 1
                row = {"task_id": task_id, "status": "rejected", "reason": str(exc)}
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "planned": len(realized), "ready": accepted,
        "rejected": len(realized) - accepted,
        "rejection_reasons": dict(reasons), "output": str(output),
    }


def synthesize_trajectories_jsonl(
    worlds_path: str | Path, realized_path: str | Path,
    output_path: str | Path, *, planner: Any, executor: Any,
    max_records: int | None = None, max_attempts: int = 3,
    subagent_workers: int = 4, rejected_output_path: str | Path | None = None,
    progress: bool = True, flush_every: int = 5, seed: int = 17,
    model_roles: Mapping[str, Any] | None = None, scene: str = "omnigaia",
) -> dict[str, Any]:
    if min(max_attempts, subagent_workers, flush_every) < 1:
        raise ValueError("attempts, workers, flush_every must be >= 1")
    realized = _rows(realized_path)
    if max_records is not None:
        realized = realized[:max_records]
    worlds = _world_map(worlds_path, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rejected_output = (Path(rejected_output_path) if rejected_output_path
                       else output.with_suffix(output.suffix + ".rejected.jsonl"))
    rejected_output.parent.mkdir(parents=True, exist_ok=True)
    if rejected_output.resolve() == output.resolve():
        raise ValueError("accepted and rejected output paths must differ")
    accepted = rejected = 0
    reasons: Counter[str] = Counter()
    # Atomic unit = one complete trajectory. Rounds remain causally ordered.
    with output.open("w", encoding="utf-8") as good, rejected_output.open("w", encoding="utf-8") as bad:
        bar = tqdm(realized, desc="Fragment gold trajectories",
                   unit="task", disable=not progress)
        for row in bar:
            task_id = str(row.get("task_id", ""))
            try:
                if task_id not in worlds:
                    raise GoldTrajectoryError("input", ["missing_matching_world"])
                record, rendered = worlds[task_id]
                episode = synthesize_one(
                    row, record, rendered, planner, executor,
                    max_attempts=max_attempts, subagent_workers=subagent_workers, scene=scene,
                )
            except (GoldTrajectoryError, ValueError, KeyError, TypeError) as exc:
                rejected += 1
                stage = exc.stage if isinstance(exc, GoldTrajectoryError) else "unexpected_error"
                reasons[stage] += 1
                bad.write(json.dumps({
                    "task_id": task_id, "reason": stage,
                    "issues": exc.issues if isinstance(exc, GoldTrajectoryError) else [str(exc)],
                    "audit": exc.audit if isinstance(exc, GoldTrajectoryError) else {},
                }, ensure_ascii=False) + "\n")
            else:
                accepted += 1
                good.write(json.dumps(episode, ensure_ascii=False) + "\n")
            if (accepted + rejected) % flush_every == 0:
                good.flush()
                bad.flush()
            if progress:
                bar.set_postfix(accepted=accepted, rejected=rejected)
        good.flush()
        bad.flush()
    return {
        "requested": len(realized), "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / len(realized) if realized else 0.0,
        "rejection_reasons": dict(reasons), "output": str(output),
        "rejected_output": str(rejected_output),
        "model_roles": dict(model_roles or {}),
        "subagent_workers": subagent_workers,
        "policy": "first_stage_fragment_guided_o1_gold_trajectory",
    }


__all__ = [
    "GoldTrajectoryError", "synthesize_one",
    "plan_trajectories_jsonl", "synthesize_trajectories_jsonl",
]
