"""MAESTRO Stage 1: generate and validate MacroGraph-grounded Questions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, args_from_config, load_config
from .generate_question.pipeline import generate_questions, audit_questions


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
    return root


def main(argv: list[str] | None = None) -> int:
    p = parser()
    cli = p.parse_args(argv)
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
