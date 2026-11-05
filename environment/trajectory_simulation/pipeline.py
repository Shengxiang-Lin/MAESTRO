"""Fragment-guided gold O1 trajectories over existing frozen task solutions.

Main sees an O1 public state plus *private generation guidance*. ProgramIR
fragments are assigned in dependency order, with independent ones in parallel.
Sub-Agent LLMs write only compressed language; the program supplies the exact
intermediate answers and their private evidence bindings.
"""
from __future__ import annotations

import json
import time
import traceback
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
from .eligibility import check_question
from .evidence import bind_ready,public_record,verify_known_packet
from .replay import replay_task
from .public_preflight import audit_public_executability
from .contract import answers_equal


class GoldTrajectoryError(RuntimeError):
    def __init__(self, stage: str, issues: list[str], audit: Any = None):
        self.stage = stage
        self.issues = list(issues)
        self.audit = audit or {}
        super().__init__(f"{stage}: {', '.join(self.issues)}")


def protect_paths(inputs,outputs):
    protected={Path(p).resolve() for p in inputs}
    targets=[Path(p).resolve() for p in outputs]
    if len(set(targets))!=len(targets) or protected.intersection(targets):
        raise ValueError('output paths must be distinct and must not overwrite frozen input files')

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
           diagnostics: Mapping[str, Any] | None = None, attempt_audit=None) -> dict[str, Any]:
    attempts = []
    issues = []
    started=time.monotonic()
    for index in range(max_attempts):
        value: Any = None
        try:
            value = call(issues)
            if not isinstance(value, Mapping):
                issues = ["response_not_json_object"]
            else:
                okay, issues = validate(value)
                if okay:
                    if attempt_audit is not None:attempt_audit.append({'stage':stage,'attempts_used':index+1,'elapsed_seconds':round(time.monotonic()-started,4)})
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


def _public_state(question: str, trajectory: list[dict[str, Any]], *, tools=TOOLS, resources=(), evidence_store=None) -> dict[str, Any]:
    """O1-compatible observation: question + cumulative formatted subtask history.

    Gold fragment graph/answers remain exclusively in the teacher guidance and
    private supervision, never in the public Main state.
    """
    return {
        "question": question + ("\n\n[Provided reference records]\n" + "\n".join(f"{r['title']} ({r['resource_id']}, {r['source_type']})" for r in resources) if resources else ""),
        "subtask_history": format_o1_subtask_history(trajectory),
        "available_tools": list(tools),
        "evidence_store": json_value(evidence_store or {}),
    }


def _model(role: Any) -> str:
    return str(getattr(getattr(role, "backend", None), "model", "passthrough"))


def synthesize_one(
    realized: Mapping[str, Any], world_record: Mapping[str, Any], rendered: Any,
    planner: Any, executor: Any, *, max_attempts: int = 3,
    subagent_workers: int = 4, scene: str = "omnigaia", question_review: bool = True, schedule: str = "baseline",
) -> dict[str, Any]:
    if max_attempts < 1 or subagent_workers < 1:
        raise ValueError("max_attempts and subagent_workers must be >= 1")
    try:
        gold = compile_gold_solution(realized, world_record, rendered, scene=scene, schedule=schedule)
    except GoldCompilationError as exc:
        raise GoldTrajectoryError(
            "gold_compilation", [str(exc)],
            audit={"witness_diagnostics": exc.diagnostics},
        ) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise GoldTrajectoryError("gold_compilation", [str(exc)]) from exc

    public_preflight = audit_public_executability(gold)
    if not public_preflight["publicly_executable"]:
        raise GoldTrajectoryError(
            "public_executability",
            [str(public_preflight["reason"])],
            {"public_preflight": public_preflight},
        )

    eligibility=check_question(gold,world_record,planner,review=question_review)
    if not eligibility['eligible']:
        raise GoldTrajectoryError('question_eligibility',['frozen_question_not_eligible'],{'eligibility':eligibility})
    evidence_cache={}
    generation_audit=[]
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
            ready=bind_ready(ready_fragments(gold,completed),evidence_cache)
            runtime={f['fragment_id']:f for f in ready}
            runtime_gold={**gold,'fragments':[runtime.get(f['fragment_id'],f) for f in gold['fragments']]}
            guidance=private_guidance(runtime_gold,completed)
            public_state = _public_state(gold["question"], trajectory, tools=gold["available_tools"], resources=gold["provided_resources"], evidence_store=evidence_cache)
            payload = {
                "public_state": public_state,
                "evidence_cache": evidence_cache,
                "gold_guidance": guidance,
                "completed_subtask_contracts": list(known_results.values()),
                "visible_resources": list({r["resource_id"]: r for r in [
                    *gold["provided_resources"],
                    *({"resource_id": o["resource_id"], "source_type": o["source_type"], "title": o["title"]}
                      for f in gold["fragments"] if f["fragment_id"] in completed for o in f["observations"])
                ]}.values()),
                # Offline passthrough test mode only. NOT included in main_prompt.
                "offline_oracle_answer": gold["final_answer"],
                "answer_contract": gold["answer_contract"],
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
                stage=f"round_{round_index}:main_action", attempt_audit=generation_audit,
                max_attempts=max_attempts,
                diagnostics={"completed_trajectory": json_value(trajectory)},
            )
            # Exact, executable O1 tasks are kept for source-only replay.
            # Natural wording changes the emitted public wire, not the
            # task graph, available tools, dependencies or gold answer.
            wire = public_action(action)
            if (
                getattr(planner, "style_provider", None) == "natural"
                and callable(getattr(planner, "naturalize_action", None))
            ):
                try:
                    candidate = planner.naturalize_action(payload, action)
                    valid, issues = validate_action(
                        candidate, guidance, gold["final_answer"],
                        gold["answer_contract"], public_state=public_state,
                    )
                    if not valid:
                        raise ValueError("natural_main_action:" + ",".join(issues))
                    # Immutable task transport must never change.
                    if candidate["action"] != action["action"]:
                        raise ValueError("natural_action_type_changed")
                    if candidate["action"] == "delegate_task":
                        for original, rewritten in zip(
                            action["params"]["tasks"], candidate["params"]["tasks"]
                        ):
                            from .natural_language import natural_public_context
                            if (rewritten["tools"] != original["tools"]
                                    or rewritten["model"] != original["model"]
                                    or rewritten["context"] != natural_public_context(
                                        original["context"], original["task_instruction"]
                                    )):
                                raise ValueError("natural_task_transport_changed")
                    elif candidate["params"] != action["params"]:
                        raise ValueError("natural_complete_answer_changed")
                    wire = public_action(candidate)
                    generation_audit.append({
                        "stage": f"round_{round_index}:natural_main",
                        "accepted": True, "fallback": False,
                    })
                except (BackendError, ValueError, TypeError, KeyError) as exc:
                    generation_audit.append({
                        "stage": f"round_{round_index}:natural_main",
                        "accepted": False, "fallback": True,
                        "reason": f"{type(exc).__name__}:{str(exc)[:200]}",
                    })
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
                        "upstream_question_semantics_revalidated": eligibility['reviewed'],
                        "public_evidence_replay_verified": True,
                        "public_plan_preflight_verified": True,
                        "public_projection_count": len(gold.get("public_projections", [])),
                        "training_eligible": eligibility['reviewed'] and not (getattr(planner,'offline',False) or getattr(executor,'offline',False)),
                        "completed_gold_fragments": len(completed),
                        "expected_gold_fragments": len(fragment_by_id),
                        "fragment_dependencies_respected": True,
                    },
                    "input_eligibility": eligibility,
                    "generation_audit": generation_audit,
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
                        "public_projections": gold.get("public_projections", []),
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
                        "scene": scene, "schedule":schedule,
                        "language_generation": (
                            "offline_reference" if getattr(planner,"offline",False) else
                            "gold_constrained_natural_generation_with_public_replay"
                            if getattr(planner,"style_provider",None) == "natural" else
                            getattr(planner,"style_provider","llm")+"_selection_with_public_evidence_replay"
                        ),
                        "natural_language_attempted": any(
                            entry.get("stage", "").endswith(":natural_main")
                            for entry in generation_audit
                        ),
                        "natural_language_fallbacks": sum(
                            bool(entry.get("fallback")) for entry in generation_audit
                        ),
                        "main_public_state_matches_o1": False,
                        "public_state_contract":"o1_with_explicit_public_evidence_store",
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

            assignments = action["params"]["tasks"]
            public_assignments = wire["params"]["tasks"]
            if len(assignments) != len(ready) or len(public_assignments) != len(ready):
                raise GoldTrajectoryError("validation", ["unexpected_task_count"])

            def generate_subagent(index: int, fragment: dict[str, Any]) -> dict[str, Any]:
                assigned = assignments[index]
                public_assigned = public_assignments[index]
                try:
                    verify_known_packet(assigned,evidence_cache)
                    observed=[public_record(o) for o in fragment['observations']]
                    replay=replay_task(assigned,observed,public_cache=evidence_cache)
                    contract={'collection_mode':'multiset' if fragment['operations'][-1]['op']=='QUERY_ATTR' else 'set'}
                    if not answers_equal(json_value(replay['result']),fragment['expected_answer'],contract):
                        raise ValueError('public task result does not reproduce the frozen subtask answer')
                    for i,(op,result) in enumerate(zip(fragment['operations'],fragment['operation_results'])):
                        if not answers_equal(json_value(replay['step_values'][i]),result['answer'],{'collection_mode':'multiset' if op['op']=='QUERY_ATTR' else 'set'}):
                            raise ValueError('public evidence cannot reproduce instruction '+str(i+1))
                        result['answer']=json_value(replay['step_values'][i])
                        result['input_values']=json_value(replay['step_inputs'][i])
                except (ValueError,KeyError,TypeError) as exc:
                    raise GoldTrajectoryError('public_replay',[str(exc)],{'public_task':assigned,'expected_answer':fragment['expected_answer'],'public_result':json_value(replay['result']) if 'replay' in locals() else None}) from exc
                predecessor_results = [
                    known_results[dep] for dep in fragment["depends_on"]
                ]
                sub_payload = {
                    "question": gold["question"],
                    "task_instruction": public_assigned["task_instruction"],
                    "assigned_task": public_assigned,
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
                    stage=f"round_{round_index}:subagent:{index}", attempt_audit=generation_audit,
                    max_attempts=max_attempts,
                    diagnostics={
                        "completed_trajectory": json_value(trajectory),
                        "fragment_id": fragment["fragment_id"],
                    },
                )
                # A verified source-derived result stays immutable. The
                # Sub-Agent LLM may only rewrite report prose.
                if (
                    getattr(executor, "style_provider", None) == "natural"
                    and callable(getattr(executor, "naturalize_feedback", None))
                ):
                    try:
                        rewritten = executor.naturalize_feedback(sub_payload, language)
                        valid, problems = validate_feedback(rewritten, fragment, strict=False)
                        if not valid:
                            raise ValueError("natural_report:" + ",".join(problems))
                        language = rewritten
                        language_status = {"accepted": True, "fallback": False}
                    except (BackendError, ValueError, TypeError, KeyError) as exc:
                        language_status = {
                            "accepted": False, "fallback": True,
                            "reason": f"{type(exc).__name__}:{str(exc)[:200]}",
                        }
                else:
                    language_status = None
                fixed_answer = answer_string(fragment["expected_answer"])
                return {
                    "subtask_index": index,
                    "task_instruction": public_assigned["task_instruction"],
                    "model": _model(executor),
                    "tools_assigned": assigned["tools"],
                    "done": True,
                    "finish_result": {
                        "status": "done",
                        "result": fixed_answer,
                        "summary": language["summary"],
                    },
                    "trace_summary": language["trace_summary"],
                    "observed_evidence": observed,
                    "result_value": json_value(replay['result']),
                    "result_kind": fragment['answer_kind'],
                    "public_replay_verified": True,
                    "_natural_language_audit": language_status,
                }

            futures = [
                pool.submit(generate_subagent, index, fragment)
                for index, fragment in enumerate(ready)
            ]
            # Run in parallel but publish all results in deterministic fragment order.
            replies = [future.result() for future in futures]
            for reply_index, (fragment, reply) in enumerate(zip(ready, replies)):
                report_audit = reply.pop("_natural_language_audit")
                if report_audit is not None:
                    generation_audit.append({
                        "stage": f"round_{round_index}:natural_subagent:{reply_index}",
                        **report_audit,
                    })
                known_results[fragment["fragment_id"]] = {
                    "fragment_id": fragment["fragment_id"],
                    "public_ref": f"Attempt {round_index}, subtask {reply_index + 1}",
                    "answer": reply["finish_result"]["result"],
                    "answer_kind": fragment["answer_kind"],
                    "value": reply["result_value"],
                    "summary": reply["finish_result"]["summary"],
                }
                for observation in reply['observed_evidence']:
                    sid=observation['resource_id']
                    if sid in evidence_cache and evidence_cache[sid]!=observation:
                        raise GoldTrajectoryError('public_replay',['inconsistent source observations'])
                    evidence_cache[sid]=observation
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
    output_path: str | Path, *, max_records: int | None = None, seed: int = 17, scene: str = "omnigaia", schedule: str = "baseline",
) -> dict[str, Any]:
    realized = _rows(realized_path)
    if max_records is not None:
        realized = realized[:max_records]
    worlds = _world_map(worlds_path, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    protect_paths([worlds_path,realized_path],[output])
    accepted = rejected = quarantined = compiled = 0
    reasons: Counter[str] = Counter()
    with output.open("w", encoding="utf-8") as writer:
        for item in realized:
            task_id = str(item.get("task_id", ""))
            try:
                if task_id not in worlds:
                    raise GoldCompilationError("world_not_found")
                plan = compile_gold_solution(item, *worlds[task_id], scene=scene, schedule=schedule)
                compiled += 1
                public_preflight = audit_public_executability(plan)
                row = {
                    "task_id": task_id,
                    "status": "ready" if public_preflight["publicly_executable"] else "quarantined",
                    "public_executability": public_preflight,
                    "fragment_graph": [{
                        "fragment_id": f["fragment_id"],
                        "output_node_id": f["output_node_id"],
                        "depends_on": f["depends_on"],
                        "resource_ids": f["resource_ids"],
                        "expected_answer": f["expected_answer"],
                    } for f in plan["fragments"]],
                    "gold_waves": plan["waves"],
                    "public_projections": plan.get("public_projections", []),
                    "public_projection_diagnostics": plan.get("public_projection_diagnostics", []),
                    "public_intermediate_projections": plan.get("public_intermediate_projections", []),
                    "final_answer": plan["final_answer"],
                }
                if public_preflight["publicly_executable"]:
                    accepted += 1
                else:
                    quarantined += 1
                    reasons["public_executability:" + str(public_preflight["reason"])] += 1
            except (GoldCompilationError, ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
                rejected += 1
                reasons[reason] += 1
                row = {
                    "task_id": task_id, "status": "rejected", "reason": reason,
                }
                if isinstance(exc, GoldCompilationError) and exc.diagnostics:
                    row["witness_diagnostics"] = exc.diagnostics
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "planned": len(realized), "compiled": compiled, "ready": accepted,
        "quarantined": quarantined, "rejected": rejected,
        "rejection_reasons": dict(reasons), "output": str(output),
    }


def synthesize_trajectories_jsonl(
    worlds_path: str | Path, realized_path: str | Path,
    output_path: str | Path, *, planner: Any, executor: Any,
    max_records: int | None = None, max_attempts: int = 3,
    subagent_workers: int = 4, rejected_output_path: str | Path | None = None,
    progress: bool = True, flush_every: int = 5, seed: int = 17,
    model_roles: Mapping[str, Any] | None = None, scene: str = "omnigaia",
    quarantine_output_path: str | Path | None = None, question_review: bool = True, schedule: str = "baseline",
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
    quarantine_output=Path(quarantine_output_path) if quarantine_output_path else output.with_suffix(output.suffix+'.quarantine.jsonl')
    quarantine_output.parent.mkdir(parents=True,exist_ok=True)
    protect_paths([worlds_path,realized_path],[output,rejected_output,quarantine_output])
    accepted = rejected = quarantined = 0
    reasons: Counter[str] = Counter()
    # Atomic unit = one complete trajectory. Rounds remain causally ordered.
    with output.open("w", encoding="utf-8") as good, rejected_output.open("w", encoding="utf-8") as bad, quarantine_output.open("w",encoding="utf-8") as quarantine:
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
                    max_attempts=max_attempts, subagent_workers=subagent_workers, scene=scene, question_review=question_review, schedule=schedule,
                )
            except (GoldTrajectoryError, ValueError, KeyError, TypeError) as exc:
                stage = exc.stage if isinstance(exc, GoldTrajectoryError) else "unexpected_error"
                reasons[stage] += 1
                isolated=stage in {'question_eligibility','public_replay','public_executability'}
                if isolated:quarantined+=1
                else:rejected+=1
                (quarantine if isolated else bad).write(json.dumps({
                    "task_id": task_id, "reason": stage,
                    "issues": (
                        exc.issues if isinstance(exc, GoldTrajectoryError)
                        else [f"{type(exc).__name__}: {exc}"]
                    ),
                    "audit": (
                        exc.audit if isinstance(exc, GoldTrajectoryError)
                        else {
                            "exception_type": type(exc).__name__,
                            "traceback": traceback.format_exc(limit=10),
                        }
                    ),
                }, ensure_ascii=False) + "\n")
            else:
                accepted += 1
                good.write(json.dumps(episode, ensure_ascii=False) + "\n")
            if (accepted + rejected + quarantined) % flush_every == 0:
                good.flush()
                bad.flush()
                quarantine.flush()
            if progress:
                bar.set_postfix(accepted=accepted, rejected=rejected)
        good.flush()
        bad.flush()
        quarantine.flush()
    return {
        "requested": len(realized), "accepted": accepted,
        "rejected": rejected, "quarantined": quarantined,
        "quarantine_output":str(quarantine_output),
        "request_counts":{"main":dict(getattr(planner,"request_counts",{})),"subagent":dict(getattr(executor,"request_counts",{}))},
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
