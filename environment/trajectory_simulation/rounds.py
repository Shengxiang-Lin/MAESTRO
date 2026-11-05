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



def _natural_main(planner, payload, canonical):
    """Only Phase 2 may generate Main wording; immutable Gold action survives."""
    from .contract import validate_action
    from .natural_language import natural_public_context
    if not (
        getattr(planner,"style_provider",None)=="natural"
        and callable(getattr(planner,"naturalize_action",None))
    ):
        return canonical,0,[]
    try:
        new=planner.naturalize_action(payload,canonical)
        okay,issues=validate_action(
            new,payload["gold_guidance"],payload["offline_oracle_answer"],
            payload["answer_contract"],public_state=payload["public_state"],
        )
        if not okay or new["action"]!=canonical["action"]:
            raise ValueError("invalid_main_naturalization:"+",".join(issues))
        original=canonical["params"].get("tasks",[])
        changed=new["params"].get("tasks",[])
        if len(original)!=len(changed):
            raise ValueError("Main rewrote the number of scheduled tasks")
        for old,task in zip(original,changed):
            ctx=old["context"] if task["task_instruction"]==old["task_instruction"] else natural_public_context(
                old["context"],old["task_instruction"]
            )
            if task["model"]!=old["model"] or task["tools"]!=old["tools"] or task["context"]!=ctx:
                raise ValueError("Main changed immutable assignment transport")
        if not original and new["params"]!=canonical["params"]:
            raise ValueError("Main modified fixed completion answer")
        n=sum(a["task_instruction"]!=b["task_instruction"] for a,b in zip(original,changed))
        return new,n,getattr(planner,"last_natural_audit",{}).get("errors",[])
    except (ValueError,KeyError,TypeError) as exc:
        return canonical,0,[{"reason":f"{type(exc).__name__}: {exc}"}]
    except Exception as exc:
        from environment.question_generation.backend import BackendError
        if not isinstance(exc,BackendError):
            raise
        return canonical,0,[{"reason":f"{type(exc).__name__}: {exc}"}]


def _read_verified_bank(record):
    """Load only a complete independently generated, source-backed Phase-1 bank."""
    from .exporter import validate_public_episode
    from .contract import answers_equal
    if not isinstance(record,dict) or set(record)!={"format","bank","sha256"}:
        raise ValueError("invalid_phase1_bank_envelope")
    if record["format"]!="maestro.clean_subtasks.v1":
        raise ValueError("incompatible_phase1_bank_version")
    if digest({key:record[key] for key in ("format","bank")})!=record["sha256"]:
        raise ValueError("phase1_bank_integrity_mismatch")
    bank=record["bank"]
    if bank.get("schema_version")!=1:
        raise ValueError("incompatible_subtask_schema")
    if not (bank["eligibility"].get("reviewed") and bank["eligibility"].get("eligible")):
        raise ValueError("phase1_question_not_reviewed_and_eligible")
    gold=bank["gold_plan"]
    if gold["task_id"]!=bank["task_id"] or gold["question"]!=bank["question"]:
        raise ValueError("phase1_gold_task_question_mismatch")
    if bank["question_record"]["task_id"]!=bank["task_id"]:
        raise ValueError("phase1_question_record_task_mismatch")
    if len(bank["waves"])!=len(bank["full_public_history"]):
        raise ValueError("phase1_wave_history_size_mismatch")
    if sum(len(w["fragment_ids"]) for w in bank["waves"])!=len(gold["fragments"]):
        raise ValueError("phase1_fragment_bank_not_complete")
    steps=bank["full_public_history"]
    for w,step in zip(bank["waves"],steps):
        if w["round_index"]!=step["attempt"] or w["canonical_action"]["params"]!=step["params"]:
            raise ValueError("phase1_stored_wave_mismatch")
    complete=bank["complete"]
    final={
        "attempt":complete["round_index"],**complete["action"],
        "state_before":complete["state"],
        "result":{"done":True,"success":True,"answer":complete["action"]["params"]["answer"]},
    }
    validate_public_episode({
        "task_id":bank["task_id"],"question":bank["question"],
        "input_eligibility":bank["eligibility"],
        "validation":{"public_evidence_replay_verified":True},
        "trajectory":[*steps,final],
    },require_training=False)
    if not answers_equal(complete["action"]["params"]["answer"],bank["final_answer"],
                         gold["answer_contract"]):
        raise ValueError("phase1_final_answer_mismatch")
    return bank


def compose_main_rounds(bank, *, planner, round_indices=None,
                        include_fallback_rounds=False, include_complete=True,
                        max_o1_attempts=10, emit=None):
    """Phase 2: consume a FULL Subtask Bank, return validated Main SFT rows.

    History prefixes contain exactly the previous waves, never any future
    subtask report. No Sub-Agent executor or World data is accessed here.
    """
    import copy
    from .compiler import private_guidance
    from .contract import main_prompt, validate_action
    from .backend import PassthroughPlanner
    from .pipeline import _public_state
    from .evidence import verify_known_packet
    from .replay import replay_task
    from .contract import answers_equal
    from .compiler import json_value

    gold=bank["gold_plan"]
    question=bank["question"]
    history=[]
    cache={}
    known={}
    completed=set()
    rows=[]
    stats={"considered":0,"skipped":0,"rewritten":0,"assigned":0,
           "history_origins":{},"skip_reasons":{}}
    teacher=PassthroughPlanner()
    prior_fallback=False
    for wave in bank["waves"]:
        idx=wave["round_index"]
        if idx!=len(history)+1:
            raise ValueError("out_of_order_subtask_bank_wave")
        ready=copy.deepcopy(wave["bound_fragments"])
        gids=[item["fragment_id"] for item in ready]
        if gids!=wave["fragment_ids"] or len(gids)!=len(set(gids)):
            raise ValueError("phase1_wave_fragment_order_mismatch")
        if any(not set(item["depends_on"])<=completed for item in ready):
            raise ValueError("phase1_wave_has_future_dependency")
        if any(fid in completed for fid in gids):
            raise ValueError("phase1_wave_duplicate_fragment")
        bound={item["fragment_id"]:item for item in ready}
        runtime={**gold,"fragments":[bound.get(f["fragment_id"],f)
                                   for f in gold["fragments"]]}
        guidance=private_guidance(runtime,completed)
        public_state=_public_state(
            question,history,tools=gold["available_tools"],
            resources=gold["provided_resources"],evidence_store=cache,
        )
        if public_state!=wave["state"]:
            # A Main-visible source/summary was changed after the Phase-1 bank
            # was produced. Exact state equality is expected before natural
            # Main wording is substituted in history (hence check all fields
            # except previously delegated task text).
            if public_state["evidence_store"]!=wave["state"]["evidence_store"]:
                raise ValueError("bank_prior_evidence_store_mismatch")
        visible=list({
            item["resource_id"]:item
            for item in [
                *gold["provided_resources"],
                *(
                    {"resource_id":o["resource_id"],
                     "source_type":o["source_type"],"title":o["title"]}
                    for previous in history
                    for reply in previous["result"]["subtask_results"]
                    for o in reply["observed_evidence"]
                ),
            ]
        }.values())
        payload={
            "public_state":public_state,"evidence_cache":cache,
            "gold_guidance":guidance,
            "completed_subtask_contracts":list(known.values()),
            "visible_resources":visible,
            "offline_oracle_answer":gold["final_answer"],
            "answer_contract":gold["answer_contract"],
        }
        canonical=teacher.decide(payload,main_prompt(payload))
        if canonical!=wave["canonical_action"]:
            raise ValueError("phase1_canonical_action_mismatch")
        # Historical waves that are not SFT targets need no Main LLM
        # generation. The immutable Gold action + correct Sub-Agent replies
        # suffice to reconstruct all later states.
        selected=round_indices is None or idx in round_indices
        if selected:
            output,changed,errors=_natural_main(planner,payload,canonical)
        else:
            output,changed,errors=canonical,0,[]
        okay,issues=validate_action(
            output,guidance,gold["final_answer"],gold["answer_contract"],
            public_state=public_state,
        )
        if not okay:
            raise ValueError("phase2_main_action_invalid:"+",".join(issues))
        count=len(ready)
        pending=[]
        for num,(fragment,task,reply) in enumerate(zip(
            ready,output["params"]["tasks"],
            wave["step"]["result"]["subtask_results"]
        ),1):
            # Recheck public execution for the newly naturalized target.
            verify_known_packet(task,cache)
            new=copy.deepcopy(reply)
            new["task_instruction"]=task["task_instruction"]
            replay=replay_task(task,new["observed_evidence"],public_cache=cache)
            kind={"collection_mode":"set" if new["result_kind"]=="entity_set" else "multiset"}
            if not answers_equal(json_value(replay["result"]),new["result_value"],kind):
                raise ValueError("phase2_action_public_replay_mismatch")
            if not answers_equal(new["result_value"],fragment["expected_answer"],kind):
                raise ValueError("phase2_subtask_answer_mismatch")
            pending.append((fragment,new))
        if selected:
            stats["considered"]+=1
            stats["rewritten"]+=changed
            stats["assigned"]+=count
            if not include_fallback_rounds and changed<count and (
                getattr(planner,"style_provider",None)=="natural"
                and not getattr(planner,"offline",False)
            ):
                stats["skipped"]+=1
                stats["skip_reasons"]["incomplete_naturalization"] = (
                    stats["skip_reasons"].get("incomplete_naturalization",0)+1
                )
            else:
                origin="teacher_reconstructed" if prior_fallback else "natural"
                row=make_round_record(
                    task_id=bank["task_id"],round_index=idx,
                    question=question,eligibility=bank["eligibility"],
                    state=public_state,prefix=history,action=output,
                    question_record=bank["question_record"],
                    natural_task_count=changed,assigned_task_count=count,
                    history_origin=origin,max_o1_attempts=max_o1_attempts,
                )
                rows.append(row)
                if emit is not None:
                    emit(row)
                stats["history_origins"][origin]=stats["history_origins"].get(origin,0)+1
        results=[]
        # Only after the entire wave succeeds do sibling observations become
        # visible in subsequent Main states.
        for num,(fragment,reply) in enumerate(pending,1):
            results.append(reply)
            fid=fragment["fragment_id"]
            known[fid]={
                "fragment_id":fid,
                "public_ref":f"Attempt {idx}, subtask {num}",
                "answer":reply["finish_result"]["result"],
                "answer_kind":fragment["answer_kind"],
                "value":reply["result_value"],
                "summary":reply["finish_result"]["summary"],
            }
            for observation in reply["observed_evidence"]:
                sid=observation["resource_id"]
                if sid in cache and cache[sid]!=observation:
                    raise ValueError("phase2_conflicting_public_evidence")
                cache[sid]=observation
            completed.add(fid)
        history.append({
            "attempt":idx,**output,
            "state_before":public_state,
            "result":{"total_subtasks":len(results),"subtask_results":results},
        })
        if changed<count or not all(wave["feedback_natural"]):
            prior_fallback=True
    if completed!={f["fragment_id"] for f in gold["fragments"]}:
        raise ValueError("phase2_incomplete_gold_fragment_graph")
    complete=copy.deepcopy(bank["complete"]["action"])
    complete["reasoning"]=""
    idx=len(history)+1
    state=_public_state(
        question,history,tools=gold["available_tools"],
        resources=gold["provided_resources"],evidence_store=cache,
    )
    guidance=private_guidance(gold,completed)
    okay,issues=validate_action(
        complete,guidance,gold["final_answer"],gold["answer_contract"],
        public_state=state,
    )
    if not okay:
        raise ValueError("phase2_complete_invalid:"+",".join(issues))
    if include_complete and (round_indices is None or idx in round_indices):
        stats["considered"]+=1
        origin="teacher_reconstructed" if prior_fallback else "natural"
        final_row=make_round_record(
            task_id=bank["task_id"],round_index=idx,question=question,
            eligibility=bank["eligibility"],state=state,prefix=history,
            action=complete,question_record=bank["question_record"],
            natural_task_count=0,assigned_task_count=0,
            history_origin=origin,max_o1_attempts=max_o1_attempts,
        )
        rows.append(final_row)
        if emit is not None:
            emit(final_row)
        stats["history_origins"][origin]=stats["history_origins"].get(origin,0)+1
    return rows,stats


def generate_main_rounds_jsonl(bank_path, output_path, *, planner,
                               max_records=None, round_indices=None,
                               include_fallback_rounds=False,
                               include_complete=True, max_o1_attempts=10,
                               rejected_output_path=None, progress=True):
    """Independent Phase 2: bank JSONL in; Main SFT JSONL out."""
    import traceback
    from collections import Counter
    from tqdm.auto import tqdm
    path=Path(bank_path)
    output=Path(output_path)
    rejected=Path(rejected_output_path) if rejected_output_path else Path(str(output)+".rejected.jsonl")
    if output.resolve()==path.resolve() or rejected.resolve() in {path.resolve(),output.resolve()}:
        raise ValueError("Phase-2 output cannot overwrite the Subtask Bank")
    if round_indices is not None and (
        not round_indices or any(not isinstance(n,int) or n<1 for n in round_indices)
    ):
        raise ValueError("round_indices must be positive integers")
    for f in (output,rejected):
        f.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    reasons=Counter()
    history_counts=Counter()
    with path.open(encoding="utf-8") as src,output.open("w",encoding="utf-8") as good,rejected.open(
        "w",encoding="utf-8"
    ) as bad:
        for row_index,line in enumerate(tqdm(src,desc="Phase 2: Main decisions",unit="bank",disable=not progress)):
            if max_records is not None and row_index>=max_records:
                break
            if not line.strip():
                continue
            try:
                bank=_read_verified_bank(json.loads(line))
                def emit_round(row):
                    # A verified Main decision is durable immediately, even if
                    # a later decision in this bank fails language realization.
                    good.write(json.dumps(row,ensure_ascii=False)+"\n")
                    good.flush()
                    counts["rounds"]+=1
                rows,stats=compose_main_rounds(
                    bank,planner=planner,round_indices=round_indices,
                    include_fallback_rounds=include_fallback_rounds,
                    include_complete=include_complete,max_o1_attempts=max_o1_attempts,
                    emit=emit_round,
                )
                counts["banks"]+=1
                counts["considered"]+=stats["considered"]
                counts["skipped"]+=stats["skipped"]
                counts["assigned"]+=stats["assigned"]
                counts["rewritten"]+=stats["rewritten"]
                reasons.update(stats["skip_reasons"])
                history_counts.update(stats["history_origins"])
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                counts["rejected"]+=1
                bad.write(json.dumps({
                    "bank_index":row_index+1,
                    "reason":f"{type(exc).__name__}: {exc}",
                    "traceback":traceback.format_exc(limit=8),
                },ensure_ascii=False)+"\n")
                bad.flush()
    return {
        "phase":"main_rounds",
        "banks_processed":counts["banks"],
        "banks_rejected":counts["rejected"],
        "rounds_considered":counts["considered"],
        "eligible_rounds":counts["rounds"],
        "rounds_skipped":counts["skipped"],
        "round_yield":counts["rounds"]/counts["considered"] if counts["considered"] else 0.0,
        "natural_main_tasks_rewritten":counts["rewritten"],
        "natural_main_tasks_total":counts["assigned"],
        "natural_action_rate":counts["rewritten"]/counts["assigned"] if counts["assigned"] else 0.0,
        "history_origins":dict(history_counts),
        "round_skip_reasons":dict(reasons),
        "request_counts":{"main":dict(getattr(planner,"request_counts",{}))},
        "bank_input":str(path),"output":str(output),"rejected_output":str(rejected),
    }


__all__ = [
    "native_tool_spec","o1_input","make_round_record",
    "compose_main_rounds","generate_main_rounds_jsonl",
]
