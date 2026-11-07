"""Frozen-boundary LLM question generation with deterministic and semantic review."""
from __future__ import annotations
import argparse,json
from time import monotonic
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from environment.config import EnvironmentConfigError,parse_with_config,resolve_model
from .backend import OpenAICompatibleBackend
from .boundary import BoundaryError,validate_frozen_inputs
from .realization import _realize_one
from .streaming import (QuestionCheckpointError,QuestionProgress,
    question_output_lock,signature_for_question_run)

def _rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:return [json.loads(line) for line in handle if line.strip()]
def _world_by_task(path,seed=17):
    from environment.task_construction.rendering.pipeline import render_task_record
    result={}
    for index,row in enumerate(_rows(path)):
        task=render_task_record(row,seed=seed+index)
        if task.task_id in result:raise BoundaryError(f"duplicate world task_id: {task.task_id}")
        result[task.task_id]={**row,"_display_answer":task.answer,"_rendered":task}
    return result
def _capability_by_task(path):return {str(row.get("task_id")):row for row in _rows(path) if row.get("task_id")}
def _backends(args):
    author_spec = resolve_model(args, "question_author")
    critic_spec = resolve_model(args, "question_critic")
    if (author_spec["provider"] == "passthrough"
            or critic_spec["provider"] == "passthrough"):
        raise EnvironmentConfigError(
            "Stage 2 naturalization requires both Author and Critic LLM backends"
        )
    args._resolved_models = {
        "question_author": {k: v for k, v in author_spec.items() if k != "api_key"},
        "question_critic": {k: v for k, v in critic_spec.items() if k != "api_key"},
    }
    def backend(spec, temperature, max_tokens):
        return OpenAICompatibleBackend(
            spec["base_url"], spec["model"], spec["api_key"],
            temperature, max_tokens, args.timeout,
            spec["json_mode"] and not args.no_json_mode,
            enable_thinking=spec.get("enable_thinking"),
        )
    return (
        backend(author_spec, args.question_temperature, args.max_tokens),
        backend(critic_spec, args.critic_temperature, args.critic_max_tokens),
    )


def _safe_future_result(future,index,authoring):
    """Keep failures isolated to the corresponding frozen input task."""
    try:
        return future.result()
    except Exception as exc:
        return False,{
            "task_id":authoring[index].get("task_id"),
            "reason":"unhandled_exception",
            "error":str(exc),
        },[]


def _stream_parallel(authoring,worlds,capabilities,author,critic,args,progress):
    """Bound in-flight requests and append completed results in input order.

    The Author/Critic calls within an individual task remain sequential.
    Scheduling and completion may be out of order; writing never is.
    """
    workers=args.question_workers
    total=len(authoring)
    next_submit=progress.completed
    pending={}
    completed_this_run=0
    bar=tqdm(total=total,initial=progress.completed,desc="Question generation",
             unit="task",disable=args.no_progress,dynamic_ncols=True)
    try:
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="question") as pool:
            active={}
            def submit_ready():
                nonlocal next_submit
                while len(active)<2*workers and next_submit<total:
                    index=next_submit
                    payload=authoring[index]
                    future=pool.submit(
                        _realize_one,payload,
                        worlds.get(str(payload.get("task_id")),{}),
                        capabilities.get(str(payload.get("task_id")),{}),
                        author,critic,
                        max_revisions=args.max_revisions,
                        blueprint_workers=args.blueprint_workers)
                    active[future]=index
                    next_submit+=1

            submit_ready()
            while active:
                ready,_=wait(tuple(active),return_when=FIRST_COMPLETED)
                for future in ready:
                    index=active.pop(future)
                    pending[index]=_safe_future_result(future,index,authoring)
                    completed_this_run+=1
                    bar.update(1)
                # Write only the contiguous completed prefix. A slow early
                # task cannot reorder the questions or their critic audits.
                while progress.completed in pending:
                    index=progress.completed
                    ok,row,audit=pending.pop(index)
                    progress.append(index,ok,row,audit)
                bar.set_postfix({
                    "written":progress.completed,
                    "accepted":progress.accepted,
                    "workers":workers,
                })
                submit_ready()
    finally:
        bar.close()
    return completed_this_run


def run(args):
    started = monotonic()
    try:
        frozen=validate_frozen_inputs(
            args.manifest,args.authoring_inputs,args.capabilities,worlds_path=args.worlds)
    except (BoundaryError,FileNotFoundError,json.JSONDecodeError) as exc:
        raise SystemExit(f"Frozen-input validation failed: {exc}") from exc
    authoring=_rows(args.authoring_inputs)
    start = getattr(args, "start_record", 0)
    if start < 0:
        raise SystemExit("--start-record must be >= 0")
    authoring=authoring[start:]
    if args.max_records is not None:
        authoring=authoring[:args.max_records]
    worlds=_world_by_task(args.worlds,seed=args.seed)
    capabilities=_capability_by_task(args.capabilities)
    if any(str(row.get("task_id")) not in worlds for row in authoring):
        raise SystemExit("authoring tasks are missing corresponding executable worlds")
    if args.max_revisions<0 or args.max_revisions>8:
        raise SystemExit("--max-revisions must be in [0, 8]")
    if args.flush_every<1:
        raise SystemExit("--flush-every must be >= 1")
    if args.question_workers<1:
        raise SystemExit("--question-workers must be >= 1")
    if args.blueprint_workers<1:
        raise SystemExit("--blueprint-workers must be >= 1")
    author,critic=_backends(args)
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    task_ids=[str(row.get("task_id","")) for row in authoring]
    source_files = [
        Path(args.manifest), Path(args.authoring_inputs),
        Path(args.worlds), Path(args.capabilities),
        *(
            Path(__file__).with_name(name) for name in (
                "pipeline.py", "realization.py", "question_review.py",
                "verification.py", "semantic_checks.py", "scoped_surface.py",
                "public_semantics.py",
                "contracts.py", "validators.py", "backend.py", "boundary.py",
                "streaming.py",
            )
        ),
        Path(__file__).parents[1] / "delegation_blueprint.py",
        Path("environment/trajectory_simulation/eligibility.py"),
        Path("environment/trajectory_simulation/compiler.py"),
        Path("environment/trajectory_simulation/skeleton.py"),
    ]
    signature=signature_for_question_run(
        frozen,args,task_ids,source_files=source_files)
    with question_output_lock(output_dir):
        with QuestionProgress(
            output_dir,signature,task_ids,
            flush_every=args.flush_every,restart=args.restart_questions
        ) as progress:
            resumed_count=progress.completed
            _stream_parallel(authoring,worlds,capabilities,author,critic,args,progress)
            accepted=progress.accepted
            rejected=progress.rejected
            if progress.completed!=len(authoring):
                raise QuestionCheckpointError(
                    f"question stream incomplete: {progress.completed}/{len(authoring)}")
        from collections import Counter
        reasons = Counter()
        with (output_dir / "question_rejected.jsonl").open(
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if line.strip():
                    reasons[json.loads(line).get("reason", "unknown")] += 1
        author_calls = critic_calls = 0
        blueprint_calls = blueprint_fragments = blueprint_retries = 0
        critic_protocol_failures = 0
        review_outcomes = Counter()
        accepted_after_revision = 0
        regression_10, expansion, holdout, later = (
            Counter(), Counter(), Counter(), Counter()
        )
        with (output_dir / "question_review_audit.jsonl").open(
            encoding="utf-8",
        ) as handle:
            for event_index, line in enumerate(handle):
                if not line.strip():
                    continue
                event = json.loads(line)
                original_index = start + event_index
                slice_stats = (
                    regression_10 if original_index < 10
                    else expansion if original_index < 50
                    else holdout if original_index < 100 else later
                )
                slice_stats["tasks"] += 1
                slice_stats["accepted" if event.get("accepted") else "rejected"] += 1
                for attempt in event.get("audit", ()):
                    author_calls += int(attempt.get("author_calls", 0))
                    critic_calls += int(attempt.get("critic_calls", 0))
                    if attempt.get("stage") == "delegation_blueprint":
                        blueprint_calls += int(attempt.get("author_calls", 0))
                        fragments = attempt.get("fragments", [])
                        blueprint_fragments += len(fragments)
                        blueprint_retries += sum(
                            max(0, int(x.get("calls", 0)) - 1)
                            for x in fragments
                        )
                    critic_protocol_failures += len(
                        attempt.get("critic_protocol_failures", ())
                    )
                    if attempt.get("stage") == "candidate":
                        review_outcomes[attempt.get(
                            "failure", "accepted" if attempt.get("accepted")
                            else "unclassified",
                        )] += 1
                if event.get("accepted") and sum(
                    row.get("stage") == "candidate"
                    for row in event.get("audit", ())
                ) > 1:
                    accepted_after_revision += 1
        summary = {
            "pipeline": "per_fragment_blueprint_dag_assembly_question_subtask",
            "frozen_batch_validation": frozen,
            "model_roles": args._resolved_models,
            "requested": len(authoring),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": (
                accepted / len(authoring) if authoring else 0.0
            ),
            "publication_status_counts": {
                "naturalized_certified": accepted,
                "rejected": rejected,
            },
            "rejection_reasons": dict(reasons),
            "evaluation_slices": {
                "previously_debugged_first_10": dict(regression_10),
                "diagnostic_tasks_11_to_50": dict(expansion),
                "holdout_tasks_51_to_100": dict(holdout),
                "later_tasks": dict(later),
            },
            "frozen_task_range": {"start_record": start, "end_record_exclusive": start + len(authoring)},
            "current_invocation_elapsed_seconds": round(monotonic() - started, 3),
            "review_statistics": {
                "max_revisions": args.max_revisions,
                "author_calls": author_calls,
                "blueprint_goal_calls": blueprint_calls,
                "blueprint_fragments_attempted": blueprint_fragments,
                "blueprint_goal_retries": blueprint_retries,
                "critic_calls": critic_calls,
                "total_model_calls": author_calls + critic_calls,
                "critic_protocol_failures": critic_protocol_failures,
                "accepted_after_revision": accepted_after_revision,
                "candidate_decisions": dict(review_outcomes),
            },
            "streaming": {
                "enabled": True,
                "ordered_writes": True,
                "flush_every": args.flush_every,
                "question_workers": args.question_workers,
                "blueprint_workers_per_task": args.blueprint_workers,
                "resumed_records": resumed_count,
                "checkpoint": str(output_dir / "question_progress_state.json"),
            },
            "artifacts": {
                "realized": str(
                    output_dir / "realized_capability_worlds.jsonl"
                ),
                "rejected": str(output_dir / "question_rejected.jsonl"),
                "review_audit": str(
                    output_dir / "question_review_audit.jsonl"
                ),
                "summary": str(
                    output_dir / "question_generation_summary.json"
                ),
            },
        }
        # Write completion summary only when all ordered records are committed.
        summary_path=output_dir/"question_generation_summary.json"
        summary_tmp=summary_path.with_suffix(".json.tmp")
        summary_tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        summary_tmp.replace(summary_path)
    return summary

def build_parser():
    p = argparse.ArgumentParser(
        description="Generate naturally written and semantically reviewed MAESTRO questions"
    )
    for name in ("manifest", "authoring-inputs", "worlds", "capabilities",
                 "output-dir", "base-url", "model", "critic-model"):
        p.add_argument("--" + name)
    p.add_argument("--max-records", type=int)
    p.add_argument("--start-record", type=int, default=0,
                   help="Zero-based offset in the validated frozen authoring pool")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--provider", choices=("passthrough", "openai-compatible"),
                   default="passthrough")
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    p.add_argument("--question-temperature", type=float, default=0.4)
    p.add_argument("--critic-temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--critic-max-tokens", type=int, default=3072)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-revisions", type=int, default=2)
    p.add_argument("--question-workers", type=int, default=4)
    p.add_argument("--blueprint-workers", type=int, default=2,
                   help="Concurrent independent Fragment-Goal requests per task")
    p.add_argument("--flush-every", type=int, default=5)
    p.add_argument("--restart-questions", action="store_true")
    p.add_argument("--no-json-mode", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p

def main():
    try:
        args=parse_with_config(build_parser(),"question_generation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    missing=[key for key in ("manifest","authoring_inputs","worlds","capabilities","output_dir") if not getattr(args,key,None)]
    if missing:raise SystemExit("Missing question-generation inputs (CLI or YAML): "+", ".join(missing))
    if args.question_workers<1:raise SystemExit("--question-workers must be >= 1")
    if args.blueprint_workers<1:raise SystemExit("--blueprint-workers must be >= 1")
    if args.flush_every<1:raise SystemExit("--flush-every must be >= 1")
    try:
        result=run(args)
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment model error: {exc}") from exc
    except QuestionCheckpointError as exc:
        raise SystemExit(f"Question-generation checkpoint error: {exc}") from exc
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
