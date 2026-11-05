"""Precompute a clean, source-replayable Subtask Bank before Main SFT states.

The Gold fragment DAG determines the subtask boundaries and partial order.
All expected results come from the frozen symbolic witness, not from an LLM.
Sub-Agent language is a natural, evidence-grounded *description* of those
verified results. No simulated error, retry, or unobserved future result is
ever placed in a Main-visible state.

A bank is private teacher data. Main SFT examples are constructed separately
from each wave's frozen public state and public action; a full-episode success
is not required to emit an individually valid decision sample.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .backend import PassthroughExecutor, PassthroughPlanner
from .compiler import (
    GoldCompilationError, compile_gold_solution, json_value,
    private_guidance, ready_fragments,
)
from .contract import (
    answer_string, answers_equal, main_prompt, public_action,
    validate_action, validate_feedback,
)
from .eligibility import check_question
from .evidence import bind_ready, public_record, verify_known_packet
from .pipeline import GoldTrajectoryError, _public_state
from .public_preflight import audit_public_executability
from .replay import replay_task


def _report_natural(executor, payload, template, fragment):
    if not (
        getattr(executor, "style_provider", None) == "natural"
        and callable(getattr(executor, "naturalize_feedback", None))
    ):
        return template, {"natural":False, "reason":"template_or_offline_mode"}
    try:
        report = executor.naturalize_feedback(payload, template)
        okay, issues = validate_feedback(report, fragment, strict=False)
        if not okay:
            raise ValueError("unsupported_report:" + ",".join(issues))
        return report, {"natural":True, "reason":None}
    except (ValueError, KeyError, TypeError) as exc:
        return template, {
            "natural":False, "reason":f"{type(exc).__name__}:{str(exc)[:160]}",
        }
    except Exception as exc:
        # A backend can become unavailable. Never use its unchecked prose.
        from environment.question_generation.backend import BackendError
        if not isinstance(exc, BackendError):
            raise
        return template, {
            "natural":False, "reason":f"{type(exc).__name__}:{str(exc)[:160]}",
        }


def _natural_action(planner, payload, canonical):
    if not (
        getattr(planner, "style_provider", None) == "natural"
        and callable(getattr(planner, "naturalize_action", None))
    ):
        return canonical, {"rewritten_tasks":0, "errors":[], "natural":False}
    try:
        proposed = planner.naturalize_action(payload, canonical)
        ok, issues = validate_action(
            proposed, payload["gold_guidance"], payload["offline_oracle_answer"],
            payload["answer_contract"], public_state=payload["public_state"],
        )
        if not ok or proposed["action"] != canonical["action"]:
            raise ValueError("invalid_natural_action:" + ",".join(issues))
        old_tasks=canonical["params"].get("tasks",[])
        new_tasks=proposed["params"].get("tasks",[])
        if len(new_tasks)!=len(old_tasks):
            raise ValueError("natural_action_task_count_changed")
        from .natural_language import natural_public_context
        for original, new in zip(old_tasks,new_tasks):
            expected = (
                original["context"]
                if new["task_instruction"] == original["task_instruction"]
                else natural_public_context(
                    original["context"],original["task_instruction"]
                )
            )
            if (new["context"]!=expected or new["model"]!=original["model"]
                    or new["tools"]!=original["tools"]):
                raise ValueError("natural_action_transport_changed")
        if not old_tasks and proposed["params"]!=canonical["params"]:
            raise ValueError("natural_complete_result_changed")
        changed=sum(
            a["task_instruction"]!=b["task_instruction"]
            for a,b in zip(old_tasks,new_tasks)
        )
        return proposed, {
            "natural":changed>0 or proposed["action"]=="complete",
            "rewritten_tasks":changed,
            "errors":getattr(planner,"last_natural_audit",{}).get("errors",[]),
        }
    except (ValueError,KeyError,TypeError) as exc:
        return canonical,{
            "natural":False,"rewritten_tasks":0,
            "errors":[{"reason":f"{type(exc).__name__}:{str(exc)[:160]}"}],
        }
    except Exception as exc:
        from environment.question_generation.backend import BackendError
        if not isinstance(exc,BackendError):
            raise
        return canonical,{
            "natural":False,"rewritten_tasks":0,
            "errors":[{"reason":f"{type(exc).__name__}:{str(exc)[:160]}"}],
        }


def _verify_subtask(canonical_task, natural_task, fragment, evidence_cache):
    # Native O1 params are not the execution oracle. Both target variants must
    # replay the exact same public observations/previously published values.
    observations = [public_record(row) for row in fragment["observations"]]
    expected = fragment["expected_answer"]
    answer_mode = {
        "collection_mode": "multiset"
        if fragment["operations"][-1]["op"] == "QUERY_ATTR" else "set"
    }
    for task in (canonical_task,natural_task):
        verify_known_packet(task,evidence_cache)
        outcome = replay_task(task,observations,public_cache=evidence_cache)
        if not answers_equal(json_value(outcome["result"]),expected,answer_mode):
            raise GoldTrajectoryError(
                "public_replay", ["subtask_bank_answer_mismatch"]
            )
        for i,(op,result) in enumerate(zip(
            fragment["operations"],fragment["operation_results"]
        )):
            contract={"collection_mode":"multiset"
                      if op["op"]=="QUERY_ATTR" else "set"}
            if not answers_equal(
                json_value(outcome["step_values"][i]),result["answer"],contract
            ):
                raise GoldTrajectoryError(
                    "public_replay", ["subtask_bank_intermediate_mismatch"]
                )
    return observations


def build_subtask_bank(
    realized: Mapping[str,Any], world_record: Mapping[str,Any], rendered: Any,
    *, planner, executor, scene="omnigaia", schedule="compact",
    question_review=True,
) -> dict[str,Any]:
    """Generate ALL clean subtasks first, in topological waves.

    Every wave is prepared from only earlier waves' public feedback and
    evidence; siblings cannot see each other's results. The returned private
    bank contains immutable final answers but each saved state_before is an
    exact snapshot from BEFORE its delegated wave.
    """
    try:
        gold=compile_gold_solution(
            realized,world_record,rendered,scene=scene,schedule=schedule
        )
    except GoldCompilationError as exc:
        raise GoldTrajectoryError(
            "gold_compilation",[str(exc)],
            {"witness_diagnostics":exc.diagnostics},
        ) from exc
    except (ValueError,KeyError,TypeError) as exc:
        raise GoldTrajectoryError("gold_compilation",[str(exc)]) from exc
    preflight=audit_public_executability(gold)
    if not preflight["publicly_executable"]:
        raise GoldTrajectoryError(
            "public_executability",[str(preflight["reason"])],
            {"public_preflight":preflight},
        )
    eligibility=check_question(gold,world_record,planner,review=question_review)
    if not eligibility["eligible"]:
        raise GoldTrajectoryError(
            "question_eligibility",["frozen_question_not_eligible"],
            {"eligibility":eligibility},
        )

    teacher_main=PassthroughPlanner()
    teacher_sub=PassthroughExecutor()
    history=[]
    evidence_cache={}
    known_results={}
    completed=set()
    waves=[]
    records={}
    while len(completed)<len(gold["fragments"]):
        index=len(waves)+1
        ready=bind_ready(ready_fragments(gold,completed),evidence_cache)
        if not ready:
            raise GoldTrajectoryError("subtask_bank",["unresolved_dependencies"])
        bound={f["fragment_id"]:f for f in ready}
        runtime_gold={
            **gold, "fragments":[bound.get(f["fragment_id"],f)
                                 for f in gold["fragments"]],
        }
        guidance=private_guidance(runtime_gold,completed)
        state=_public_state(
            gold["question"],history,tools=gold["available_tools"],
            resources=gold["provided_resources"],evidence_store=evidence_cache,
        )
        visible=list({
            r["resource_id"]:r for r in [
                *gold["provided_resources"],
                *(
                    {"resource_id":o["resource_id"],
                     "source_type":o["source_type"],"title":o["title"]}
                    for fragment in gold["fragments"]
                    if fragment["fragment_id"] in completed
                    for o in fragment["observations"]
                ),
            ]
        }.values())
        payload={
            "public_state":state,"evidence_cache":evidence_cache,
            "gold_guidance":guidance,
            "completed_subtask_contracts":list(known_results.values()),
            "visible_resources":visible,
            "offline_oracle_answer":gold["final_answer"],
            "answer_contract":gold["answer_contract"],
        }
        canonical=teacher_main.decide(payload,main_prompt(payload))
        ok,issues=validate_action(
            canonical,guidance,gold["final_answer"],gold["answer_contract"],
            public_state=state,
        )
        if not ok:
            raise GoldTrajectoryError("subtask_bank",issues)
        if canonical["action"]!="delegate_task":
            raise GoldTrajectoryError("subtask_bank",["unexpected_complete"])
        natural,audit=_natural_action(planner,payload,canonical)
        original_tasks=canonical["params"]["tasks"]
        public_tasks=natural["params"]["tasks"]
        if len(original_tasks)!=len(ready):
            raise GoldTrajectoryError("subtask_bank",["incorrect_ready_wave"])
        reply_list=[]
        pending=[]
        for sub_index,(fragment,original,public) in enumerate(
            zip(ready,original_tasks,public_tasks),1
        ):
            observations=_verify_subtask(
                original,public,fragment,evidence_cache
            )
            report_payload={
                "question":gold["question"],
                "task_instruction":public["task_instruction"],
                "assigned_task":public,
                "dependency_results":[known_results[dep]
                                      for dep in fragment["depends_on"]],
                "fragment":fragment,
            }
            base=teacher_sub.respond(report_payload,"")
            ok,issues=validate_feedback(base,fragment,strict=True)
            if not ok:
                raise GoldTrajectoryError("subtask_bank",issues)
            report,language=_report_natural(
                executor,report_payload,base,fragment
            )
            # Semantic reviewers may be LLMs, but exact result is immutable.
            fixed=answer_string(fragment["expected_answer"])
            reply={
                "subtask_index":sub_index-1,
                "task_instruction":public["task_instruction"],
                "model":str(getattr(getattr(executor,"backend",None),
                                      "model","passthrough")),
                "tools_assigned":original["tools"],
                "done":True,
                "finish_result":{
                    "status":"done","result":fixed,"summary":report["summary"],
                },
                "trace_summary":report["trace_summary"],
                "observed_evidence":observations,
                "result_value":json_value(fragment["expected_answer"]),
                "result_kind":fragment["answer_kind"],
                "public_replay_verified":True,
            }
            records[fragment["fragment_id"]]={
                "fragment_id":fragment["fragment_id"],
                "depends_on":list(fragment["depends_on"]),
                "wave_index":index,"task_index":sub_index,
                "task_instruction":public["task_instruction"],
                "canonical_task_instruction":original["task_instruction"],
                "correct_answer":fixed,
                "summary":report["summary"],
                "trace_summary":report["trace_summary"],
                "public_evidence":observations,
                "natural_task":public["task_instruction"]!=original["task_instruction"],
                "natural_feedback":language["natural"],
                "feedback_fallback_reason":language["reason"],
            }
            pending.append((fragment,reply))
            reply_list.append(reply)
        # Only after ALL sibling answers are verified are they made visible.
        for sub_index,(fragment,reply) in enumerate(pending,1):
            known_results[fragment["fragment_id"]]={
                "fragment_id":fragment["fragment_id"],
                "public_ref":f"Attempt {index}, subtask {sub_index}",
                "answer":reply["finish_result"]["result"],
                "answer_kind":fragment["answer_kind"],
                "value":reply["result_value"],
                "summary":reply["finish_result"]["summary"],
            }
            for observation in reply["observed_evidence"]:
                sid=observation["resource_id"]
                if sid in evidence_cache and evidence_cache[sid]!=observation:
                    raise GoldTrajectoryError(
                        "public_replay",["inconsistent_subtask_bank_sources"]
                    )
                evidence_cache[sid]=observation
            completed.add(fragment["fragment_id"])
        wave_step={
            "attempt":index,**public_action(natural),
            "state_before":state,
            "result":{"total_subtasks":len(reply_list),
                      "subtask_results":reply_list},
        }
        waves.append({
            "round_index":index,"state":state,
            "action":public_action(natural),
            "canonical_action":public_action(canonical),
            "fragment_ids":[f["fragment_id"] for f in ready],
            "naturalization":audit,
            "feedback_natural":[
                records[f["fragment_id"]]["natural_feedback"] for f in ready
            ],
            "step":wave_step,
        })
        history.append(wave_step)
    index=len(waves)+1
    state=_public_state(
        gold["question"],history,tools=gold["available_tools"],
        resources=gold["provided_resources"],evidence_store=evidence_cache,
    )
    final_payload={
        "public_state":state,"gold_guidance":private_guidance(gold,completed),
        "offline_oracle_answer":gold["final_answer"],
        "answer_contract":gold["answer_contract"],
    }
    complete=teacher_main.decide(final_payload,main_prompt(final_payload))
    natural_complete,_=_natural_action(planner,final_payload,complete)
    if natural_complete["action"]!="complete":
        raise GoldTrajectoryError("subtask_bank",["not_a_completion"])
    ok,issues=validate_action(
        natural_complete,final_payload["gold_guidance"],
        gold["final_answer"],gold["answer_contract"],public_state=state,
    )
    if not ok:
        raise GoldTrajectoryError("subtask_bank",issues)
    return {
        "task_id":gold["task_id"],"question":gold["question"],
        "eligibility":eligibility,"preflight":preflight,
        "fragments":records,"waves":waves,
        "complete":{"round_index":index,"state":state,
                    "action":public_action(natural_complete)},
        "schedule":schedule,
        "full_public_history":history,
        "final_answer":gold["final_answer"],  # PRIVATE: not Main-visible
        "ready_order":[list(wave["fragment_ids"]) for wave in waves],
    }


__all__=["build_subtask_bank"]
