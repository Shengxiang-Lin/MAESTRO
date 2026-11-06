"""Stage 3 has two explicit independent jobs: Sub-Agent reports, then Main SFT."""
from __future__ import annotations

import argparse
import json

from environment.config import (
    EnvironmentConfigError, parse_with_config, resolve_model,
)
from environment.question_generation.backend import OpenAICompatibleBackend
from .backend import LLMExecutor, LLMPlanner, PassthroughExecutor, PassthroughPlanner
from .tools import PROFILES


def _print(value):
    print(json.dumps(value,ensure_ascii=False,indent=2))


def _add_llm_options(parser):
    parser.add_argument("--provider",choices=("passthrough","openai-compatible"),
                        default="passthrough")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--planner-model")
    parser.add_argument("--executor-model")
    parser.add_argument("--api-key-env",default="DASHSCOPE_API_KEY")
    parser.add_argument("--planner-temperature",type=float,default=0.2)
    parser.add_argument("--executor-temperature",type=float,default=0.2)
    parser.add_argument("--max-tokens",type=int,default=2048)
    parser.add_argument("--timeout",type=float,default=90.0)
    parser.add_argument("--no-json-mode",action="store_true")
    parser.add_argument("--no-progress",action="store_true")
    parser.add_argument("--max-records",type=int)
    parser.add_argument("--rejected-output")


def _role(args, role):
    spec=resolve_model(args,"trajectory_planner" if role=="main" else "trajectory_executor")
    if spec["provider"]=="passthrough":
        obj=PassthroughPlanner() if role=="main" else PassthroughExecutor()
    else:
        backend=OpenAICompatibleBackend(
            spec["base_url"],spec["model"],spec["api_key"],
            args.planner_temperature if role=="main" else args.executor_temperature,
            args.max_tokens,args.timeout,
            spec["json_mode"] and not args.no_json_mode,
        )
        obj=LLMPlanner(backend) if role=="main" else LLMExecutor(backend)
    obj.style_provider="natural"
    return obj


def command_subtasks(args):
    from .subtask_bank import generate_subtasks_jsonl
    executor=_role(args,"subagent")
    output=args.output if "--output" in args._explicit_options else args.bank
    _print(generate_subtasks_jsonl(
        args.worlds,args.realized,output,
        executor=executor,
        max_records=args.max_records,seed=args.seed,scene=args.scene,
        schedule=args.schedule,feedback_workers=args.feedback_workers,
        rejected_output_path=args.rejected_output,
        quarantine_output_path=args.quarantine_output,
        progress=not args.no_progress,
    ))


def command_rounds(args):
    from .rounds import generate_main_rounds_jsonl
    selected=None
    if args.round_indices:
        try:
            selected={int(piece.strip()) for piece in args.round_indices.split(",")}
        except ValueError as exc:
            raise SystemExit("--round-indices must contain positive integers") from exc
        if not selected or min(selected)<1:
            raise SystemExit("--round-indices must contain positive integers")
    _print(generate_main_rounds_jsonl(
        args.bank,args.output,planner=_role(args,"main"),
        max_records=args.max_records,round_indices=selected,
        include_fallback_rounds=args.include_fallback_rounds,
        include_complete=not args.exclude_complete,
        max_o1_attempts=args.max_o1_attempts,
        rejected_output_path=args.rejected_output,
        progress=not args.no_progress,
    ))


def build_parser():
    parser=argparse.ArgumentParser(
        description="Stage 3: (1) validated Sub-Agent feedback bank; "
                    "(2) independent Main scheduling SFT from that bank"
    )
    sub=parser.add_subparsers(dest="command",required=True)
    subtasks=sub.add_parser(
        "subtasks",help="Phase 1: verify Stage-2 surface certificates and generate source-replayable Sub-Agent reports"
    )
    subtasks.add_argument("--worlds")
    subtasks.add_argument("--realized")
    subtasks.add_argument("--bank",help="default Phase-1 output bank JSONL")
    subtasks.add_argument("--output")
    subtasks.add_argument("--seed",type=int,default=17)
    subtasks.add_argument("--scene",choices=sorted(PROFILES),default="omnigaia")
    subtasks.add_argument("--schedule",choices=("compact","baseline"),default="compact")
    subtasks.add_argument("--feedback-workers",type=int,default=4)
    subtasks.add_argument("--quarantine-output")
    _add_llm_options(subtasks)
    subtasks.set_defaults(func=command_subtasks)

    rounds=sub.add_parser(
        "rounds",help="Phase 2: only read completed bank, generate Main scheduling SFT rounds"
    )
    rounds.add_argument("--bank",help="completed Phase-1 Subtask Bank JSONL")
    rounds.add_argument("--output")
    rounds.add_argument("--round-indices",help="e.g. 2,3; unselected rounds still supply history")
    rounds.add_argument("--include-fallback-rounds",action="store_true")
    rounds.add_argument("--exclude-complete",action="store_true")
    rounds.add_argument("--max-o1-attempts",type=int,default=10)
    _add_llm_options(rounds)
    rounds.set_defaults(func=command_rounds)
    return parser


def main():
    try:
        args=parse_with_config(build_parser(),"trajectory_simulation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    required=("worlds","realized","bank") if args.command=="subtasks" else ("bank","output")
    missing=[name for name in required if not getattr(args,name,None)]
    if missing:
        raise SystemExit("Missing Phase-"+("1" if args.command=="subtasks" else "2")
                         +" input(s): "+", ".join(missing))
    if args.max_records is not None and args.max_records<0:
        raise SystemExit("--max-records must be nonnegative")
    if args.command=="subtasks" and args.feedback_workers<1:
        raise SystemExit("--feedback-workers must be positive")
    if args.command=="rounds" and args.max_o1_attempts<1:
        raise SystemExit("--max-o1-attempts must be positive")
    try:
        args.func(args)
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment model error: {exc}") from exc


if __name__=="__main__":
    main()
