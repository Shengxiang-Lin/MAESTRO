"""MAESTRO configuration-driven evidence and orchestration workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, args_from_config, load_config
from .generate_question.pipeline import generate_questions, audit_questions
from .implement_subtask.config import (
    DEFAULT_CONFIG as STAGE2_CONFIG,
    load_config as load_stage2_config,
    load_reporting_config,
)
from .implement_subtask.pipeline import materialize_subtasks
from .implement_subtask.naturalize import naturalize_subtasks
from .implement_subtask.combined import generate_subagent_reports
from .workflow import run_pipeline
from .base_orchestration.config import (
    DEFAULT_CONFIG as MAIN_CONFIG,
    generate_from_config as generate_main_from_config,
    load_config as load_main_config,
)
from .untrusted_orchestration.config import (
    DEFAULT_CONFIG as UNTRUSTED_CONFIG,
    load_config as load_untrusted_config,
    generate_from_config as generate_untrusted_from_config,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m environment",
        description="MAESTRO Stage 1 -> 2 -> 4 evidence and decision workflow",
    )
    sub = root.add_subparsers(dest="command", required=True)
    workflow = sub.add_parser(
        "run", help="Run Stage 1 -> Stage 2 -> Stage 4 end to end"
    )
    workflow.add_argument("--questions-config", type=Path,
                          default=DEFAULT_CONFIG)
    workflow.add_argument("--reports-config", type=Path,
                          default=STAGE2_CONFIG)
    workflow.add_argument("--untrusted-config", type=Path,
                          default=UNTRUSTED_CONFIG)
    workflow.add_argument("--mode", choices=("template", "llm"),
                          default="llm")
    workflow.add_argument("--max-scenes", type=int)
    workflow.add_argument("--no-progress", action="store_true")
    generate = sub.add_parser("generate-questions",
                              help="Run the complete config-driven generation pipeline")
    generate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate.add_argument("--output-dir", type=Path,
                          help="Override generation.output_dir from YAML")
    generate.add_argument("--target", type=int,
                          help="Override generation.target from YAML")
    generate.add_argument("--no-progress", action="store_true")
    generate.add_argument("--verbose-progress", action="store_true",
                          help="Also show auxiliary validation and Replay audit progress")
    generate.add_argument("--flush-every", type=int,
                          help="Commit a durable checkpoint every N LLM reports")
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
    naturalize = sub.add_parser(
        "naturalize-subtasks",
        help="Stage 2: frozen Oracle Macro -> realistic Sub-Agent finish via LLM Author/Critic",
    )
    naturalize.add_argument("--config", type=Path, default=STAGE2_CONFIG)
    naturalize.add_argument(
        "--scenes", type=Path,
        help="One-command Stage 2: first build Oracle from accepted Stage-1 scenes, then Author/Critic reports",
    )
    naturalize.add_argument("--source-subtasks", type=Path,
                            help="Override frozen Oracle subtasks.jsonl")
    naturalize.add_argument("--source-scene-index", type=Path,
                            help="Override frozen Oracle scene_index.jsonl")
    naturalize.add_argument("--output-dir", type=Path,
                            help="Override output for reviewed Sub-Agent responses")
    naturalize.add_argument("--max-scenes", type=int,
                            help="Process a small scene subset for testing")
    naturalize.add_argument("--no-progress", action="store_true")
    naturalize.add_argument("--flush-every", type=int,
                            help="Commit a durable checkpoint every N Macro reports")
    naturalize.add_argument(
        "--reuse-from", type=Path,
        help="Completed prior reporting directory: reuse accepted Macros, retry only rejected ones in a NEW output folder",
    )
    trusted = sub.add_parser(
        "generate-main-turns",
        help="Stage 3: reviewed, dependency-safe Main decisions",
    )
    trusted.add_argument("--config", type=Path, default=MAIN_CONFIG)
    trusted.add_argument("--scenes", type=Path)
    trusted.add_argument("--reports", type=Path)
    trusted.add_argument("--output-dir", type=Path)
    trusted.add_argument("--max-scenes", type=int)
    trusted.add_argument("--workers", type=int)
    trusted.add_argument("--mode", choices=("template", "llm"))
    trusted.add_argument("--no-progress", action="store_true")
    untrusted = sub.add_parser(
        "generate-untrusted-turns",
        help="Branch trusted Main history at a returned Sub-Agent batch and simulate evidence failures",
    )
    untrusted.add_argument("--config", type=Path, default=UNTRUSTED_CONFIG)
    untrusted.add_argument("--scenes", type=Path)
    untrusted.add_argument("--reports", type=Path)
    untrusted.add_argument("--output-dir", type=Path)
    untrusted.add_argument("--max-scenes", type=int)
    untrusted.add_argument("--branches-per-scene", type=int)
    untrusted.add_argument("--workers", type=int)
    untrusted.add_argument("--mode", choices=("template", "llm"))
    untrusted.add_argument("--overwrite", action="store_true")
    untrusted.add_argument("--no-progress", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    p = parser()
    cli = p.parse_args(argv)
    if cli.command == "run":
        summary = run_pipeline(
            questions_config=cli.questions_config,
            reports_config=cli.reports_config,
            orchestration_config=cli.untrusted_config,
            mode=cli.mode, max_scenes=cli.max_scenes,
            no_progress=cli.no_progress,
        )
        print(
            "MAESTRO Stage 1->2->4: " + summary["status"]
            + " | published Main turns: "
            + str(summary["published_main_turns"])
        )
        return 0
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
        print(f"Stage 2: {result['status']} | summary: "
              f"{options['output_dir']}/summary.json")
        return 0 if result["status"] in {"done", "partial"} else 1
    if cli.command == "naturalize-subtasks":
        options = load_reporting_config(cli.config)
        for override in (
            "source_subtasks", "source_scene_index", "output_dir", "max_scenes"
        ):
            value = getattr(cli, override)
            if value is not None:
                options[override] = value
        if cli.no_progress:
            options["no_progress"] = True
        if cli.flush_every is not None:
            options["flush_every"] = cli.flush_every
        if cli.reuse_from is not None:
            options["reuse_from"] = cli.reuse_from
        if (cli.scenes is not None or
                (cli.source_subtasks is None and cli.source_scene_index is None)):
            if (cli.source_subtasks is not None
                    or cli.source_scene_index is not None):
                p.error(
                    "--scenes (combined Stage 2) cannot be used together with "
                    "--source-subtasks or --source-scene-index"
                )
            impl = load_stage2_config(cli.config)
            result = generate_subagent_reports(
                scenes_path=cli.scenes or impl["scenes_path"],
                output_dir=options["output_dir"],
                reporting_options=options,
                evidence_limit=impl["evidence_limit"],
                strict_replay=impl["strict_replay"],
                max_scenes=options.get("max_scenes"),
                no_progress=options["no_progress"],
            )
        else:
            result = naturalize_subtasks(**options)
        print(f"subagent reports: {result['status']} | details: "
              f"{options['output_dir']}/summary.json")
        return 0 if result["status"] in {"done", "partial"} else 1
    if cli.command == "generate-main-turns":
        overrides = {}
        for cli_name, option in (
            ("scenes", "scenes_path"), ("reports", "reports_path"),
            ("output_dir", "output_dir"), ("max_scenes", "max_scenes"),
            ("workers", "workers"), ("mode", "mode"),
        ):
            value = getattr(cli, cli_name)
            if value is not None:
                overrides[option] = value
        if cli.no_progress:
            overrides["no_progress"] = True
        summary = generate_main_from_config(cli.config, **overrides)
        out = overrides.get("output_dir") or load_main_config(
            cli.config
        )["output_dir"]
        print(f"main orchestration: {summary['status']} | details: "
              f"{out}/summary.json")
        return 0 if summary["status"] in {"done", "partial"} else 1
    if cli.command == "generate-untrusted-turns":
        overrides = {}
        for cli_name, option in (
            ("scenes", "scenes_path"), ("reports", "reports_path"),
            ("output_dir", "output_dir"), ("max_scenes", "max_scenes"),
            ("branches_per_scene", "branches_per_scene"),
            ("workers", "workers"), ("mode", "mode"),
        ):
            value = getattr(cli, cli_name)
            if value is not None:
                overrides[option] = value
        if cli.overwrite:
            overrides["overwrite"] = True
        if cli.no_progress:
            overrides["no_progress"] = True
        summary = generate_untrusted_from_config(cli.config, **overrides)
        out = overrides.get("output_dir") or load_untrusted_config(
            cli.config
        )["output_dir"]
        print(f"untrusted orchestration: {summary['status']} | details: "
              f"{out}/summary.json")
        return 0 if summary["status"] in {"done", "partial"} else 1
    config = load_config(cli.config)
    if cli.command == "generate-questions":
        options = args_from_config(
            config, output_dir=cli.output_dir,
            target=cli.target, no_progress=cli.no_progress,
        )
        if cli.flush_every is not None:
            options.flush_every = cli.flush_every
        options.verbose_progress = cli.verbose_progress
        result = generate_questions(options)
        report_location = options.output_dir / "summary.json"
    else:
        config_args = args_from_config(config)
        directory = config_args.output_dir
        # A deterministic Stage-1 run publishes constructed/scenes.jsonl.
        # Optional LLM Question authoring publishes questions/scenes.jsonl.
        # Prefer the frozen Stage-1 summary over either guessed path.
        if cli.scenes is not None:
            scenes = cli.scenes
        elif (directory / "summary.json").is_file():
            manifest = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            final_scenes = manifest.get("final_scenes")
            if not final_scenes:
                raise ValueError("stage1_summary_missing_final_scenes")
            scenes = Path(final_scenes)
        else:
            scenes = directory / (
                "questions" if config_args.question_model else "constructed"
            ) / "scenes.jsonl"
        output = cli.output or directory / "quality_recheck.json"
        result = audit_questions(argparse.Namespace(
            scenes=scenes, max_records=None, output=output,
            no_progress=cli.no_progress,
        ))
        report_location = output
    print(f"{cli.command}: {result['status']} | details: {report_location}")
    return 0 if result["status"] in {"done", "partial", "valid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
