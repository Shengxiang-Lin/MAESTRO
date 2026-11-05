"""CLI for Stage-1-fragment-guided O1 gold trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from environment.config import (
    EnvironmentConfigError, parse_with_config, resolve_model,
)
from environment.question_generation.backend import OpenAICompatibleBackend

from .backend import (
    LLMExecutor, LLMPlanner, PassthroughExecutor, PassthroughPlanner,
)
from .tools import PROFILES
from .pipeline import plan_trajectories_jsonl, synthesize_trajectories_jsonl


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--worlds", help="typed_worlds_*.jsonl")
    parser.add_argument("--realized", help="accepted question-generation JSONL")
    parser.add_argument("--output")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--schedule",choices=("baseline","compact"),default="compact")
    parser.add_argument("--scene", choices=sorted(PROFILES), default="omnigaia")


def command_plan(args: argparse.Namespace) -> None:
    # Do not overwrite the configured synthesis output during plan-only runs.
    output = args.output
    if "--output" not in args._explicit_options:
        output = str(Path(output).with_suffix(".gold_plan.jsonl"))
    _print(plan_trajectories_jsonl(
        args.worlds, args.realized, output,
        max_records=args.max_records, seed=args.seed, scene=args.scene, schedule=args.schedule,
    ))


def _role(spec: dict, args: argparse.Namespace, *, planner: bool):
    if spec["provider"] == "passthrough":
        return PassthroughPlanner() if planner else PassthroughExecutor()
    backend = OpenAICompatibleBackend(
        spec["base_url"], spec["model"], spec["api_key"],
        args.planner_temperature if planner else args.executor_temperature,
        args.max_tokens, args.timeout,
        spec["json_mode"] and not args.no_json_mode,
    )
    return LLMPlanner(backend) if planner else LLMExecutor(backend)


def _configured_roles(args: argparse.Namespace):
    planner_spec = resolve_model(args, "trajectory_planner")
    executor_spec = resolve_model(args, "trajectory_executor")
    planner = _role(planner_spec, args, planner=True)
    executor = _role(executor_spec, args, planner=False)
    for role in (planner, executor):
        role.style_provider = args.style_provider
        role.style_seed = args.style_seed
    if args.eligibility_cache:
        planner.eligibility_cache = args.eligibility_cache
    return planner, executor, planner_spec, executor_spec


def command_synthesize(args: argparse.Namespace) -> None:
    planner, executor, planner_spec, executor_spec = _configured_roles(args)
    if args.eligibility_cache:
        from .pipeline import protect_paths
        protect_paths([args.worlds, args.realized], [args.eligibility_cache, args.output])
    clean = lambda spec: {key: value for key, value in spec.items() if key != "api_key"}
    result = synthesize_trajectories_jsonl(
        args.worlds, args.realized, args.output,
        planner=planner, executor=executor,
        max_records=args.max_records,
        max_attempts=args.max_attempts,
        subagent_workers=args.subagent_workers,
        rejected_output_path=args.rejected_output,
        quarantine_output_path=args.quarantine_output,
        progress=not args.no_progress,
        flush_every=args.flush_every,
        seed=args.seed, scene=args.scene, schedule=args.schedule,
        model_roles={
            "main_planner": clean(planner_spec),
            "subagent_executor": clean(executor_spec),
        },
    )
    _print(result)


def command_rounds(args: argparse.Namespace) -> None:
    from .rounds import synthesize_rounds_jsonl
    planner, executor, _, _ = _configured_roles(args)
    indices = None
    if args.round_indices:
        try:
            indices = {int(piece.strip()) for piece in args.round_indices.split(",")}
        except ValueError as exc:
            raise SystemExit("--round-indices must be comma-separated positive integers") from exc
        if not indices or min(indices) < 1:
            raise SystemExit("--round-indices must be positive")
    output = args.output
    if "--output" not in args._explicit_options:
        path = Path(output)
        output = str(path.with_name(path.stem + ".rounds.jsonl"))
    if args.eligibility_cache:
        from .pipeline import protect_paths
        protect_paths([args.worlds, args.realized], [args.eligibility_cache, output])
    _print(synthesize_rounds_jsonl(
        args.worlds, args.realized, output,
        planner=planner, executor=executor,
        max_records=args.max_records,
        round_indices=indices,
        include_fallback_rounds=args.include_fallback_rounds,
        include_complete=not args.exclude_complete,
        max_o1_attempts=args.max_o1_attempts,
        feedback_workers=args.feedback_workers,
        seed=args.seed, scene=args.scene, schedule=args.schedule,
        rejected_output_path=args.rejected_output,
        quarantine_output_path=args.quarantine_output,
        bank_output_path=args.bank_output,
        progress=not args.no_progress,
    ))


def _add_synthesis_options(synth: argparse.ArgumentParser) -> None:
    synth.add_argument(
        "--provider", choices=("passthrough", "openai-compatible"),
        default="passthrough",
    )
    synth.add_argument(
        "--style-provider", choices=("natural","llm","template"), default="natural",
        help="natural: guarded fluent O1 prose; llm: fixed variant choice; template: deterministic",
    )
    synth.add_argument("--style-seed", type=int, default=17)
    synth.add_argument("--base-url")
    synth.add_argument("--model", help="override both role model names")
    synth.add_argument("--planner-model", help="override only Main Planner model")
    synth.add_argument("--executor-model", help="override only Sub-Agent model")
    synth.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    synth.add_argument("--planner-temperature", type=float, default=0.2)
    synth.add_argument("--executor-temperature", type=float, default=0.2)
    synth.add_argument("--max-tokens", type=int, default=2048)
    synth.add_argument("--timeout", type=float, default=90.)
    synth.add_argument("--rejected-output")
    synth.add_argument("--quarantine-output")
    synth.add_argument("--eligibility-cache")
    synth.add_argument("--no-json-mode", action="store_true")
    synth.add_argument("--no-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fragment-guided official O1 trajectories from frozen tasks"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="compile first-stage answered fragment DAGs without an LLM")
    _add_common(plan)
    plan.set_defaults(func=command_plan)

    synth = sub.add_parser(
        "synthesize",
        help="legacy full-episode gold Main and Sub-Agent trajectories",
    )
    _add_common(synth)
    _add_synthesis_options(synth)
    synth.add_argument("--max-attempts", type=int, default=3)
    synth.add_argument("--subagent-workers", type=int, default=4)
    synth.add_argument("--flush-every", type=int, default=5)
    synth.set_defaults(func=command_synthesize)

    rounds = sub.add_parser(
        "rounds",
        help="precompute clean Subtask Bank, then compose and export O1 Main decisions",
    )
    _add_common(rounds)
    _add_synthesis_options(rounds)
    rounds.add_argument(
        "--round-indices", help="only target these decision rounds, e.g. 2,3; "
        "unselected predecessor rounds are reconstructed as valid history",
    )
    rounds.add_argument(
        "--include-fallback-rounds", action="store_true",
        help="include canonical-template targets when some Main natural rewrites fail",
    )
    rounds.add_argument(
        "--exclude-complete", action="store_true",
        help="omit Gold complete decisions from the SFT training examples",
    )
    rounds.add_argument(
        "--bank-output",
        help="optional teacher-private JSONL with precomputed source-backed "
             "subtask instructions, feedback and answers; never use as SFT input",
    )
    rounds.add_argument("--max-o1-attempts", type=int, default=10)
    rounds.add_argument(
        "--feedback-workers",type=int,default=4,
        help="parallel report language workers for sibling Subtasks within a Gold wave",
    )
    rounds.set_defaults(func=command_rounds)
    return parser


def main() -> None:
    try:
        args = parse_with_config(build_parser(), "trajectory_simulation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    missing = [
        key for key in ("worlds", "realized", "output")
        if not getattr(args, key, None)
    ]
    if missing:
        raise SystemExit("Missing trajectory inputs: " + ", ".join(missing))
    if args.max_records is not None and args.max_records < 0:
        raise SystemExit("--max-records must be nonnegative")
    if args.command == "synthesize":
        if min(args.max_attempts, args.subagent_workers, args.flush_every) < 1:
            raise SystemExit("attempts, workers and flush_every must be >= 1")
    try:
        args.func(args)
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment model error: {exc}") from exc


if __name__ == "__main__":
    main()
