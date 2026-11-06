"""Phase 1: precompute clean, source-replayable Sub-Agent feedback before Main SFT.

The Gold fragment DAG determines the subtask boundaries and partial order.
All expected results come from the frozen symbolic witness, not from an LLM.
Only Sub-Agent report language is generated in this phase. Main delegation
language and Main training decisions belong exclusively to phase 2. No simulated error, retry, or unobserved future result is
ever placed in a Main-visible state.

A bank is private teacher data. Main SFT examples are constructed separately
from each wave's frozen public state and public action; a full-episode success
is not required to emit an individually valid decision sample.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .backend import LLMExecutor, PassthroughExecutor, PassthroughPlanner
from .compiler import (
    GoldCompilationError, compile_gold_solution, json_value,
    private_guidance, ready_fragments,
)
from .contract import (
    answer_string, answers_equal, main_prompt, public_action,
    validate_action, validate_feedback,
)
from .eligibility import controlled_question_reference, digest
from .evidence import bind_ready, public_record, verify_known_packet
from .pipeline import GoldTrajectoryError, _public_state
from .public_preflight import audit_public_executability
from .replay import replay_task


def _report_natural(executor, payload, template, fragment):
    audit=[]
    if not (
        getattr(executor, "style_provider", None) == "natural"
        and callable(getattr(executor, "naturalize_feedback", None))
    ):
        return template, {"natural":False, "reason":"template_or_offline_mode"}
    try:
        report = (executor.naturalize_feedback(payload, template, audit=audit)
                  if isinstance(executor,LLMExecutor)
                  else executor.naturalize_feedback(payload, template))
        okay, issues = validate_feedback(
            report,fragment,strict=False,
            public_refs=[r.get("public_ref") for r in payload.get("dependency_results",[])],
        )
        if not okay:
            raise ValueError("unsupported_report:" + ",".join(issues))
        return report, {"natural":True, "reason":None, "audit":audit}
    except (ValueError, KeyError, TypeError) as exc:
        return template, {
            "natural":False, "reason":f"{type(exc).__name__}:{str(exc)[:160]}",
            "audit":audit,
        }
    except Exception as exc:
        # A backend can become unavailable. Never use its unchecked prose.
        from environment.question_generation.backend import BackendError
        if not isinstance(exc, BackendError):
            raise
        return template, {
            "natural":False, "reason":f"{type(exc).__name__}:{str(exc)[:160]}",
            "audit":audit,
        }


def _verify_subtask(task, fragment, evidence_cache):
    """Replay the single immutable canonical assignment, without a duplicate pass."""
    observations = [public_record(row) for row in fragment["observations"]]
    verify_known_packet(task,evidence_cache)
    outcome = replay_task(task,observations,public_cache=evidence_cache)
    answer_mode = {
        "collection_mode": "multiset"
        if fragment["operations"][-1]["op"] == "QUERY_ATTR" else "set"
    }
    if not answers_equal(json_value(outcome["result"]),fragment["expected_answer"],answer_mode):
        raise GoldTrajectoryError("public_replay", ["subtask_bank_answer_mismatch"])
    for i,(op,result) in enumerate(zip(fragment["operations"],fragment["operation_results"])):
        contract={"collection_mode":"multiset" if op["op"]=="QUERY_ATTR" else "set"}
        if not answers_equal(json_value(outcome["step_values"][i]),result["answer"],contract):
            raise GoldTrajectoryError("public_replay", ["subtask_bank_intermediate_mismatch"])
        # The report writer sees only the intermediate public replay inputs.
        result["input_values"]=json_value(outcome["step_inputs"][i])
        result["answer"]=json_value(outcome["step_values"][i])
    return observations


def build_subtask_bank(
    realized: Mapping[str,Any], world_record: Mapping[str,Any], rendered: Any,
    *, executor, scene="omnigaia", schedule="compact",
    feedback_workers: int = 4,
) -> dict[str,Any]:
    """Generate canonical clean subtasks independently of Stage 2 surface text.

    Every wave is prepared from only earlier waves' public feedback and
    evidence; siblings cannot see each other's results. The returned private
    bank contains immutable final answers but each saved state_before is an
    exact snapshot from BEFORE its delegated wave.
    """
    if feedback_workers < 1:
        raise ValueError("feedback_workers must be positive")
    # The frozen renderer question is the ONLY compiler input. Stage 2's
    # language rewrite is never allowed to change hidden-identity bindings,
    # fragment cuts, gold answers, or source-replayable task instructions.
    canonical_question=getattr(rendered,"question",None)
    if not isinstance(canonical_question,str) or not canonical_question.strip():
        raise GoldTrajectoryError("canonical_question",["renderer_question_unavailable"])
    surface_question=realized.get("question")
    if not isinstance(surface_question,str) or not surface_question.strip():
        raise GoldTrajectoryError("question_eligibility",["empty_stage2_surface_question"])
    canonical_record={**realized,"question":canonical_question}
    try:
        gold=compile_gold_solution(
            canonical_record,world_record,rendered,scene=scene,schedule=schedule
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
    # Canonical is verified deterministically, without an LLM role.
    reference=controlled_question_reference(gold,world_record,rendered)
    if reference is None or reference["question"]!=canonical_question:
        raise GoldTrajectoryError(
            "canonical_question",["canonical_renderer_not_certified"],
        )
    canonical_eligibility={
        "eligible":True,"reviewed":True,"decision":"eligible",
        "method":"verified_controlled_renderer","issues":[],
        "renderer_certificate":reference,
        "question_sha256":digest(canonical_question),
        "program_sha256":digest(world_record["program"]),
    }
    # Stage 2 has already completed the expensive full-ProgramIR LLM review.
    # Stage 3.1 performs only certificate / frozen-input integrity checks.
    # Exact controlled-renderer text is also safe without an old Stage-2
    # certificate, because the renderer itself has just been re-certified.
    from environment.question_generation.verification import validate_surface_certificate
    try:
        if (surface_question==canonical_question
                and not realized.get("surface_certificate")):
            eligibility=dict(canonical_eligibility)
        else:
            eligibility=validate_surface_certificate(
                realized,world_record,canonical_question,
            )
    except ValueError as exc:
        eligibility={
            "eligible":False,"reviewed":False,"decision":"review_error",
            "method":"stage2_certificate_validation",
            "review_failure_kind":str(exc),
            "issues":[str(exc)],
            "question_sha256":digest(surface_question),
            "program_sha256":digest(world_record["program"]),
        }

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
        # Phase 1 generates Sub-Agent evidence reports ONLY. Main task
        # language is a phase-2 decision target, never generated here.
        original_tasks=canonical["params"]["tasks"]
        if len(original_tasks)!=len(ready):
            raise GoldTrajectoryError("subtask_bank",["incorrect_ready_wave"])
        # The parallel wave shares only the PREVIOUS wave's immutable public
        # history/cache. All natural feedback jobs may run concurrently, but
        # results are published together after their public replay succeeds.
        def prepare_subtask(item):
            sub_index,(fragment,original)=item
            observations=_verify_subtask(
                original,fragment,evidence_cache
            )
            from .tool_provenance import reconstruct_tool_observations
            # The preflight's independent search is a TEACHER reachability
            # test, never a Sub-Agent tool execution. Only simulate and
            # publish search calls when the Main delegation actually assigned
            # GoogleSearchAction AND this fragment acquires a new source.
            search_audit = None
            fresh_search_sources = [
                obs for obs in fragment["observations"]
                if obs.get("visibility") == "discoverable"
            ]
            if set(original["tools"]) != set(fragment["suggested_tools"]):
                raise GoldTrajectoryError(
                    "tool_assignment", ["delegation_tool_mismatch"]
                )
            if fresh_search_sources and "GoogleSearchAction" not in original["tools"]:
                raise GoldTrajectoryError(
                    "tool_assignment", ["search_tool_not_assigned_for_new_source"]
                )
            if (fresh_search_sources and gold.get("retrieval_policy")
                    == "task_local_ranked_public_index_v1"):
                from .search_simulator import (
                    simulate_search, verify_retrieved_witnesses,
                )
                search_audit = simulate_search(
                    gold["search_catalog"],
                    public_question=gold["question"],
                    public_task=original["task_instruction"],
                    already_known_ids=list(evidence_cache)+[
                        x["resource_id"]
                        for x in gold.get("provided_resources", [])
                    ],
                )
                verify_retrieved_witnesses(
                    search_audit, fresh_search_sources
                )
            tool_provenance = reconstruct_tool_observations(
                fragment,search_audit
            )
            report_payload={
                "question":gold["question"],
                "task_instruction":original["task_instruction"],
                "assigned_task":original,
                "dependency_results":[known_results[dep]
                                      for dep in fragment["depends_on"]],
                "fragment":fragment,
                "tool_provenance":tool_provenance,
            }
            base=teacher_sub.respond(report_payload,"")
            ok,issues=validate_feedback(base,fragment,strict=True)
            if not ok:
                raise GoldTrajectoryError("subtask_bank",issues)
            report,language=_report_natural(
                executor,report_payload,base,fragment
            )
            fixed=answer_string(fragment["expected_answer"])
            reply={
                "subtask_index":sub_index-1,
                "task_instruction":original["task_instruction"],
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
            bank_item={
                "fragment_id":fragment["fragment_id"],
                "depends_on":list(fragment["depends_on"]),
                "wave_index":index,"task_index":sub_index,
                "task_instruction":original["task_instruction"],
                "correct_answer":fixed,
                "summary":report["summary"],
                "trace_summary":report["trace_summary"],
                "public_evidence":observations,
                "natural_feedback":language["natural"],
                "feedback_fallback_reason":language["reason"],
                "feedback_private_audit":language.get("audit",[]),
                "simulated_tool_provenance":tool_provenance,
                "search_audit":search_audit,
            }
            return fragment,reply,bank_item

        jobs=list(enumerate(zip(ready,original_tasks),1))
        with ThreadPoolExecutor(max_workers=min(feedback_workers,len(jobs))) as pool:
            prepared=list(pool.map(prepare_subtask,jobs))
        pending=[]
        reply_list=[]
        for fragment,reply,bank_item in prepared:
            records[fragment["fragment_id"]]=bank_item
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
            "attempt":index,**public_action(canonical),
            "state_before":state,
            "result":{"total_subtasks":len(reply_list),
                      "subtask_results":reply_list},
        }
        waves.append({
            "round_index":index,"state":state,
            "canonical_action":public_action(canonical),
            "bound_fragments":ready,
            "fragment_ids":[f["fragment_id"] for f in ready],
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
    if complete["action"]!="complete":
        raise GoldTrajectoryError("subtask_bank",["not_a_completion"])
    ok,issues=validate_action(
        complete,final_payload["gold_guidance"],
        gold["final_answer"],gold["answer_contract"],public_state=state,
    )
    if not ok:
        raise GoldTrajectoryError("subtask_bank",issues)
    # A valid paraphrase must be able to discover the same public evidence
    # from its actual Main-visible question. Canonical discovery logs remain
    # private teacher evidence; a different surface query must never inherit
    # their success automatically.
    surface_preflight={"publicly_executable":True,"checked_fragments":0}
    if (surface_question!=canonical_question
            and eligibility.get("eligible") and eligibility.get("reviewed")
            and gold.get("retrieval_policy")=="task_local_ranked_public_index_v1"):
        from .search_simulator import simulate_search, verify_retrieved_witnesses
        for wave in waves:
            known=list(wave["state"]["evidence_store"])+[
                x["resource_id"] for x in gold["provided_resources"]
            ]
            for fragment in wave["bound_fragments"]:
                fid=fragment["fragment_id"]
                if records[fid]["search_audit"] is None:
                    continue
                surface_preflight["checked_fragments"]+=1
                new_audit=simulate_search(
                    gold["search_catalog"],
                    public_question=surface_question,
                    public_task=records[fid]["task_instruction"],
                    already_known_ids=known,
                )
                required=[
                    obs for obs in fragment["observations"]
                    if obs.get("visibility")=="discoverable"
                ]
                try:
                    verify_retrieved_witnesses(new_audit,required)
                except ValueError as exc:
                    surface_preflight={
                        **surface_preflight,"publicly_executable":False,
                        "reason":str(exc)[:240],"failed_fragment":fid,
                    }
                    break
            if not surface_preflight["publicly_executable"]:
                break

    return {
        "task_id":gold["task_id"],"question":gold["question"],
        "surface_question":surface_question,
        "canonical_eligibility":canonical_eligibility,
        "eligibility":eligibility,"preflight":preflight,
        "surface_preflight":surface_preflight,
        "fragments":records,"waves":waves,
        "complete":{"round_index":index,"state":state,
                    "action":public_action(complete)},
        "schedule":schedule,
        "full_public_history":history,
        "final_answer":gold["final_answer"],  # PRIVATE: not Main-visible
        "gold_plan":gold,  # Teacher-private; never fed to Main
        "question_record":dict(realized),
        "schema_version":2,
        "ready_order":[list(wave["fragment_ids"]) for wave in waves],
    }



def generate_subtasks_jsonl(
    worlds_path, realized_path, output_path, *,
    executor, max_records=None, seed=17, scene="omnigaia",
    schedule="compact", feedback_workers=4,
    rejected_output_path=None, quarantine_output_path=None, progress=True,
):
    """Complete Phase 1 and persist reusable teacher-private Subtask Banks.

    This phase checks Stage-2 certificates and calls only the Sub-Agent language model.
    It does NOT generate any natural Main delegation or training target.
    Phase 2 reads the output independently, without worlds or Stage-2 files.
    """
    import json
    import traceback
    from collections import Counter
    from pathlib import Path
    from tqdm.auto import tqdm
    from .eligibility import digest
    from .pipeline import _rows, _world_map, protect_paths

    rows=_rows(realized_path)
    if max_records is not None:
        rows=rows[:max_records]
    worlds=_world_map(worlds_path,seed=seed)
    output=Path(output_path)
    rejected=Path(rejected_output_path) if rejected_output_path else Path(str(output)+".rejected.jsonl")
    quarantined=Path(quarantine_output_path) if quarantine_output_path else Path(str(output)+".quarantine.jsonl")
    protect_paths([worlds_path,realized_path],[output,rejected,quarantined])
    if len({p.resolve() for p in (output,rejected,quarantined)})!=3:
        raise ValueError("Phase-1 output paths must be distinct")
    for p in (output,rejected,quarantined):
        p.parent.mkdir(parents=True,exist_ok=True)
    stats=Counter()
    reasons=Counter()
    review_decisions=Counter()
    review_failure_kinds=Counter()
    fallback_reasons=Counter()
    with output.open("w",encoding="utf-8") as good, rejected.open(
        "w",encoding="utf-8"
    ) as bad, quarantined.open("w",encoding="utf-8") as isolate:
        for row in tqdm(rows,desc="Phase 1: correct Sub-Agent reports",unit="task",disable=not progress):
            task_id=str(row.get("task_id",""))
            try:
                if task_id not in worlds:
                    raise GoldTrajectoryError("input",["missing_matching_world"])
                world,rendered=worlds[task_id]
                bank=build_subtask_bank(
                    row,world,rendered,executor=executor,
                    scene=scene,schedule=schedule,feedback_workers=feedback_workers,
                )
                stats["canonical_banks"]+=1
                if not bank["surface_preflight"]["publicly_executable"]:
                    raise GoldTrajectoryError(
                        "surface_public_executability",
                        [bank["surface_preflight"].get("reason","surface_sources_unreachable")],
                        {"surface_preflight":bank["surface_preflight"],
                         "canonical_bank":json_value(bank)},
                    )
                if not (bank["eligibility"].get("eligible") and bank["eligibility"].get("reviewed")):
                    # Canonical Bank is valid diagnostic evidence but is NOT
                    # eligible for Main-Agent SFT with this unverified surface.
                    raise GoldTrajectoryError(
                        "question_eligibility",["surface_semantics_not_confirmed"],
                        {"eligibility":bank["eligibility"],
                         "canonical_bank":json_value(bank)},
                    )
                payload=json_value(bank)
                record={"format":"maestro.clean_subtasks.v2","bank":payload}
                record["sha256"]=digest(record)
                good.write(json.dumps(record,ensure_ascii=False)+"\n")
                good.flush()
                stats["accepted"]+=1
                review_decisions["eligible"]+=1
                fallback_reasons.update(
                    item["feedback_fallback_reason"].split(":",1)[-1].split(":",1)[0].split(";",1)[0]
                    for item in payload["fragments"].values()
                    if item["feedback_fallback_reason"]
                )
                stats["subtasks"]+=len(payload["fragments"])
                stats["natural_feedback"]+=sum(
                    bool(item["natural_feedback"])
                    for item in payload["fragments"].values()
                )
                stats["waves"]+=len(payload["waves"])
            except (GoldTrajectoryError,ValueError,KeyError,TypeError) as exc:
                stage=exc.stage if isinstance(exc,GoldTrajectoryError) else "unexpected_error"
                quarantine=stage in {
                    "question_eligibility","public_replay","public_executability",
                    "surface_public_executability",
                }
                dest=isolate if quarantine else bad
                stats["quarantined" if quarantine else "rejected"]+=1
                reasons[stage]+=1
                if stage=="question_eligibility":
                    eligibility=exc.audit.get("eligibility",{})
                    review_decisions[eligibility.get("decision","review_error")]+=1
                    review_failure_kinds[eligibility.get(
                        "review_failure_kind", "unclassified_review_failure"
                    )]+=1
                dest.write(json.dumps({
                    "task_id":task_id,"reason":stage,
                    "issues":exc.issues if isinstance(exc,GoldTrajectoryError)
                             else [f"{type(exc).__name__}: {exc}"],
                    "audit":exc.audit if isinstance(exc,GoldTrajectoryError)
                            else {"traceback":traceback.format_exc(limit=8)},
                },ensure_ascii=False)+"\n")
                dest.flush()
    return {
        "phase":"subtasks","requested":len(rows),
        "canonical_banks_generated":stats["canonical_banks"],
        "accepted_banks":stats["accepted"],
        "quarantined":stats["quarantined"],"rejected":stats["rejected"],
        "subtasks":stats["subtasks"],"gold_waves":stats["waves"],
        "natural_feedback_count":stats["natural_feedback"],
        "natural_feedback_rate":stats["natural_feedback"]/stats["subtasks"]
                                  if stats["subtasks"] else 0.0,
        "reasons":dict(reasons),
        "question_review_decisions":dict(review_decisions),
        "question_review_summary":{
            "eligible":review_decisions["eligible"],
            "semantic_mismatch":review_decisions["mismatch"],
            "semantic_uncertain":review_decisions["uncertain"],
            "review_errors":review_decisions["review_error"],
        },
        "question_review_failure_kinds":dict(review_failure_kinds),
        "feedback_fallback_reasons":dict(fallback_reasons),
        "request_counts":{
            "subagent":dict(getattr(executor,"request_counts",{})),
        },
        "output":str(output),
        "rejected_output":str(rejected),
        "quarantine_output":str(quarantined),
    }


__all__=["build_subtask_bank","generate_subtasks_jsonl"]
