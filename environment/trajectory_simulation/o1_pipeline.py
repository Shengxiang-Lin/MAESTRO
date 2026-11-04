"""Main-controlled O1 rounds with one simulated reply per delegated task."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
from environment.task_construction.executor import SymbolicExecutor
from environment.task_construction.validation import normalize_answer
from .acquisition import decision_complete_evidence
from .compiler import compile_gold_solution, json_value, ready_subtasks
from .o1_contract import (
    feedback_prompt, main_prompt, public_action, validate_action,
    validate_feedback, work_hint,
)
from .pipeline import GoldTrajectoryError, _stage, _with_feedback


def _model(role: Any) -> str:
    return str(getattr(getattr(role, "backend", None), "model", "passthrough"))


def _history(trajectory):
    return [{
        **{key: step[key] for key in ("action", "reasoning", "params")},
        "result": {key: value for key, value in step["result"].items()
                   if not key.startswith("_")},
    } for step in trajectory]


def synthesize_one(
    realized: Mapping[str, Any], world_record: Mapping[str, Any],
    rendered: Any, planner: Any, executor: Any, *,
    max_attempts: int = 3, subagent_workers: int = 4,
) -> dict[str, Any]:
    if min(max_attempts, subagent_workers) < 1:
        raise ValueError("attempts and workers must be positive")
    gold = compile_gold_solution(realized, world_record, rendered, full_environment=True)
    tasks = {task["subtask_id"]: task for task in gold["tasks"]}
    completed: set[str] = set()
    observed_facts: set[str] = set()
    inspected_sources: set[str] = set()
    required_facts = set(gold["required_fact_ids"])
    program = program_from_dict(world_record["program"])
    world = world_from_dict(world_record["world"])
    expected = normalize_answer(SymbolicExecutor().execute(program, world))
    known_resources = {}
    for task in tasks.values():
        if task["kind"] == "inspect" and task["visibility"] in {
            "provided", "attached", "visible", "initial", "provided_initial",
        }:
            known_resources[task["resource_id"]] = {
                key: task["observation"][key] for key in ("resource_id", "source_type", "title")
            }
    trajectory = []
    private_rounds = []
    past_evidence = []
    world_sources = world_record["world"]["sources"]
    source_fact_ids = {sid: set(source["fact_ids"]) for sid, source in world_sources.items()}

    with ThreadPoolExecutor(max_workers=subagent_workers, thread_name_prefix="o1-simulator") as pool:
        # Every delegated round consumes >=1 new feasible scope; unused alternatives
        # need not be executed before Main can finish a decision-complete proof.
        for round_index in range(1, len(tasks) + 2):
            ready = ready_subtasks(gold, completed)
            sufficient = decision_complete_evidence(
                program, world, inspected_sources, required_facts, expected,
            )
            public_state = {
                "question": gold["question"],
                "known_resources": list(known_resources.values()),
                "history": _history(trajectory),
            }
            payload = {
                "public_state": public_state,
                "feasible_work": [work_hint(task) for task in ready],
                "evidence_sufficient": sufficient,
                "answer_contract": gold["answer_contract"],
                # Offline reference only. main_prompt deliberately excludes it.
                "offline_answer": gold["final_answer"],
            }
            action = _stage(
                lambda errors: planner.decide(payload, _with_feedback(main_prompt(payload), errors)),
                lambda value: validate_action(value, ready, sufficient,
                                              gold["final_answer"], gold["answer_contract"]),
                stage=f"round_{round_index}:main_action", max_attempts=max_attempts,
                diagnostics={"completed_trajectory": json_value(trajectory),
                             "evidence_sufficient": sufficient},
            )
            wire = public_action(action)
            if wire["action"] == "complete":
                trajectory.append({
                    "attempt": round_index, **wire, "state_before": public_state,
                    "result": {"done": True, "success": True, "answer": wire["params"]["answer"]},
                })
                return {
                    "task_id": gold["task_id"], "question": gold["question"],
                    "instruction": gold["question"],
                    "main_model": _model(planner), "sub_models": [_model(executor)],
                    "attempts": len(trajectory), "success": True,
                    "trajectory": trajectory,
                    "validation": {
                        "accepted": True, "final_answer_verified": True,
                        "decision_information_complete": True,
                        "observed_fact_ids": sorted(observed_facts),
                        "inspected_resource_ids": sorted({
                            tasks[sid]["resource_id"] for sid in completed if tasks[sid]["kind"] == "inspect"
                        }),
                    },
                    "gold_supervision": {
                        "final_answer": gold["final_answer"],
                        "answer_contract": gold["answer_contract"],
                        "required_fact_ids": gold["required_fact_ids"],
                        "minimum_proof_source_ids": gold["proof_source_ids"],
                        "acquisition_alternatives": gold["acquisition_alternatives"],
                        "available_resource_ids": sorted(world_sources),
                        "round_bindings": private_rounds,
                    },
                    "metadata": {
                        "format": "orchestra_o1", "clean_trajectory": True,
                        "two_role_teacher": True, "feedback_simulated": True,
                        "main_controls_task_grouping_and_parallel_width": True,
                        "raw_tool_traces_generated": False,
                        "oracle_answer_not_shown_to_main_llm": True,
                        "final_answer_validation": "deterministic_terminal_contract",
                        "natural_feedback_validation": "scoped_claims_and_deterministic_text_checks",
                        "feedback_evidence_prefix": (
                            "schema_constrained_assigned_facts"
                            if getattr(getattr(executor, "backend", None), "json_mode", False)
                            else "validated_fact_surface_coverage"
                        ),
                    },
                }

            assignments = wire["params"]["tasks"]
            bindings = sorted(action["task_bindings"], key=lambda item: item["task_index"])

            def simulate(binding):
                index = binding["task_index"]
                scopes = [tasks[sid] for sid in binding["subtask_ids"]]
                execution = {
                    "question": gold["question"], "assigned_task": assignments[index],
                    "scopes": scopes, "past_evidence": past_evidence,
                    "known_resource_ids": sorted(known_resources),
                }
                response = _stage(
                    lambda errors: executor.respond(
                        execution, _with_feedback(feedback_prompt(execution), errors)),
                    lambda value: validate_feedback(value, execution),
                    stage=f"round_{round_index}:subagent:{index}",
                    max_attempts=max_attempts,
                    diagnostics={"assigned_task": assignments[index],
                                 "scope_ids": binding["subtask_ids"],
                                 "completed_trajectory": json_value(trajectory)},
                )
                public_reply = {
                    "subtask_index": index, "task_instruction": assignments[index]["task_instruction"],
                    "model": _model(executor), "tools_assigned": assignments[index]["tools"],
                    "done": True, "finish_result": response["finish_result"],
                    "trace_summary": response["trace_summary"],
                }
                return public_reply, response["grounded_claims"]

            # One request per Main task, even when that task binds several sources.
            futures = [pool.submit(simulate, binding) for binding in bindings]
            replies = [future.result() for future in futures]
            trajectory.append({
                "attempt": round_index, **wire, "state_before": public_state,
                "result": {
                    "subtask_results": [reply for reply, _ in replies],
                    "total_subtasks": len(replies), "all_done": True,
                    "_internal_metadata": {"simulated_feedback": True, "raw_tool_traces_available": False},
                },
            })
            private_rounds.append({
                "attempt": round_index, "task_bindings": bindings,
                "grounded_claims": [claims for _, claims in replies],
            })
            for _, claims in replies:
                for sid, values in claims.items():
                    task = tasks[sid]
                    completed.add(sid)
                    known_resources[task["resource_id"]] = {
                        key: task["observation"][key] for key in ("resource_id", "source_type", "title")
                    }
                    if task["kind"] == "inspect":
                        inspected_sources.add(task["resource_id"])
                        observed_facts.update(source_fact_ids[task["resource_id"]])
                    past_evidence.append({
                        "kind": task["kind"], "resource_id": task["resource_id"],
                        "title": task["observation"]["title"], "claims": values,
                    })
    raise GoldTrajectoryError("scheduling", ["main_round_budget_exhausted"],
                              {"completed_trajectory": trajectory})
