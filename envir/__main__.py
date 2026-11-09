"""MAESTRO Stage 1: generate and validate MacroGraph-grounded Questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, args_from_config, load_config
from .generate_question.pipeline import generate_questions, audit_questions
from .implement_subtask.config import (
    DEFAULT_CONFIG as STAGE2_CONFIG,
    load_config as load_stage2_config,
)
from .implement_subtask.pipeline import materialize_subtasks


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m envir",
        description="Stage 1: construct Macro TaskGraphs and generate aligned Questions",
    )
    sub = root.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-questions",
                              help="Run the complete config-driven generation pipeline")
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate.add_argument("--output-dir", type=Path,
                          help="Override generation.output_dir from YAML")
    generate.add_argument("--target", type=int,
                          help="Override generation.target from YAML")
    generate.add_argument("--no-progress", action="store_true")
    audit = sub.add_parser("quality",
                           help="Read-only strict replay and MacroGraph/Question audit")
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--scenes", type=Path,
                       help="Default: <generation.output_dir>/questions/scenes.jsonl")
    audit.add_argument("--output", type=Path,
                       help="Default: <generation.output_dir>/quality_recheck.json")
    audit.add_argument("--no-progress", action="store_true")
    subtask = sub.add_parser(
        "implement-subtasks",
        help="Stage 2: exact per-Macro answer and evidence-grounded trace_summary",
    )
    subtask.add_argument("--config", type=Path, default=STAGE2_CONFIG)
    subtask.add_argument("--scenes", type=Path,
                         help="Override the accepted frozen Stage-1 scene bank")
    subtask.add_argument("--output-dir", type=Path,
                         help="Override Stage-2 output directory")
    subtask.add_argument("--max-records", type=int)
    subtask.add_argument("--no-progress", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    p = parser()
    cli = p.parse_args(argv)
    if cli.command == "implement-subtasks":
        options = load_stage2_config(cli.config)
        if cli.scenes is not None:
            options["scenes_path"] = cli.scenes
        if cli.output_dir is not None:
            options["output_dir"] = cli.output_dir
        if cli.max_records is not None:
            options["max_records"] = cli.max_records
        if cli.no_progress:
            options["no_progress"] = True
        result = materialize_subtasks(**options)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"done", "partial"} else 1
    config = load_config(cli.config)
    if cli.command == "generate-questions":
        options = args_from_config(
            config, output_dir=cli.output_dir,
            target=cli.target, no_progress=cli.no_progress,
        )
        result = generate_questions(options)
    else:
        config_args = args_from_config(config)
        directory = config_args.output_dir
        scenes = cli.scenes or directory / "questions" / "scenes.jsonl"
        output = cli.output or directory / "quality_recheck.json"
        result = audit_questions(argparse.Namespace(
            scenes=scenes, max_records=None, output=output,
            no_progress=cli.no_progress,
        ))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"done", "partial", "valid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
