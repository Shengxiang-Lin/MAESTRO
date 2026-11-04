"""CLI for clean trajectory planning and LLM trajectory synthesis."""
from __future__ import annotations

import argparse
import json
import os

from .pipeline import plan_trajectories_jsonl, synthesize_trajectories_jsonl
from environment.config import EnvironmentConfigError, parse_with_config, resolve_model


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--worlds", help="typed_worlds_*.jsonl")
    parser.add_argument(
        "--realized",
        help="validated question-generation realized_capability_worlds.jsonl",
    )
    parser.add_argument("--output")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=17)


def command_plan(args: argparse.Namespace) -> None:
    _print(
        plan_trajectories_jsonl(
            args.worlds,
            args.realized,
            args.output,
            max_records=args.max_records,
        )
    )


def command_synthesize(args: argparse.Namespace) -> None:
    spec=resolve_model(args,"trajectory_teacher")
    result=synthesize_trajectories_jsonl(
        args.worlds,
        args.realized,
        args.output,
        provider=spec["provider"],
        base_url=spec["base_url"],
        model=spec["model"],
        api_key=spec["api_key"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_records=args.max_records,
        max_attempts=args.max_attempts,
        rejected_output_path=args.rejected_output,
        progress=not args.no_progress,
        flush_every=args.flush_every,
        json_mode=spec["json_mode"] and not args.no_json_mode,
        seed=args.seed,
    )
    result["model_backend"]=spec["backend"]
    _print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dependency-aware clean orchestration trajectory simulation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="compile frozen acquisition frontiers without calling a Teacher LLM",
    )
    _add_common(plan)
    plan.set_defaults(func=command_plan)

    synthesize = subparsers.add_parser(
        "synthesize",
        help="generate validated clean trajectories round by round",
    )
    _add_common(synthesize)
    synthesize.add_argument(
        "--provider",
        choices=("passthrough", "openai-compatible"),
        default="passthrough",
    )
    synthesize.add_argument("--base-url")
    synthesize.add_argument("--model")
    synthesize.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    synthesize.add_argument("--temperature", type=float, default=0.2)
    synthesize.add_argument("--max-tokens", type=int, default=2048)
    synthesize.add_argument("--timeout", type=float, default=90.0)
    synthesize.add_argument("--max-attempts", type=int, default=3)
    synthesize.add_argument("--flush-every", type=int, default=5)
    synthesize.add_argument("--rejected-output")
    synthesize.add_argument("--no-json-mode", action="store_true")
    synthesize.add_argument("--no-progress", action="store_true")
    synthesize.set_defaults(func=command_synthesize)
    return parser


def main() -> None:
    try:
        args = parse_with_config(build_parser(),"trajectory_simulation")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    missing=[key for key in ("worlds","realized","output") if not getattr(args,key,None)]
    if missing:raise SystemExit("Missing trajectory inputs (CLI or YAML): "+", ".join(missing))
    try:
        args.func(args)
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment model error: {exc}") from exc


if __name__ == "__main__":
    main()
