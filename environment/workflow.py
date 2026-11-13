"""Single public workflow: Stage 1 -> Stage 2 -> Stage 4.

Stage 4 compiles its clean O1 prefix deterministically from the two
upstream artifacts; there is no standalone Stage-3 generation call.
"""
from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_CONFIG as QUESTIONS_CONFIG
from .config import args_from_config, load_config as load_questions_config
from .generate_question.pipeline import generate_questions
from .implement_subtask.config import (
    DEFAULT_CONFIG as REPORTS_CONFIG,
    load_config as load_subtask_config, load_reporting_config,
)
from .implement_subtask.combined import generate_subagent_reports
from .untrusted_orchestration.config import (
    DEFAULT_CONFIG as ORCHESTRATION_CONFIG,
    generate_from_config as generate_untrusted_from_config,
)


def run_pipeline(
    *, questions_config: Path = QUESTIONS_CONFIG,
    reports_config: Path = REPORTS_CONFIG,
    orchestration_config: Path = ORCHESTRATION_CONFIG,
    mode: str = "llm",
    max_scenes: int | None = None,
    no_progress: bool = False,
) -> dict:
    """Run accepted scenes, reviewed Sub-Agent reports, then Stage 4.

    Each stage's checkpoint, retry and validation rules remain intact.
    A failed upstream stage stops the pipeline; no fabricated placeholder
    data is sent to the next stage.
    """
    if mode not in {"llm", "template"}:
        raise ValueError("invalid_pipeline_mode")
    if max_scenes is not None and (
        type(max_scenes) is not int or max_scenes < 1
    ):
        raise ValueError("invalid_pipeline_max_scenes")

    questions_args = args_from_config(
        load_questions_config(questions_config), no_progress=no_progress,
    )
    questions = generate_questions(questions_args)
    if questions.get("status") not in {"done", "partial"}:
        raise RuntimeError("stage1_not_accepted:" + str(questions.get("status")))
    scenes_path = questions_args.output_dir / "questions" / "scenes.jsonl"
    if not scenes_path.is_file():
        raise FileNotFoundError("stage1_did_not_publish_scenes")

    subtask_options = load_subtask_config(reports_config)
    report_options = load_reporting_config(reports_config)
    reports = generate_subagent_reports(
        scenes_path=scenes_path,
        output_dir=report_options["output_dir"],
        reporting_options=report_options,
        evidence_limit=subtask_options["evidence_limit"],
        strict_replay=subtask_options["strict_replay"],
        max_scenes=max_scenes,
        no_progress=no_progress,
    )
    if reports.get("status") not in {"done", "partial"}:
        raise RuntimeError("stage2_not_accepted:" + str(reports.get("status")))
    reports_path = report_options["output_dir"] / "subtasks.jsonl"
    if not reports_path.is_file():
        raise FileNotFoundError("stage2_did_not_publish_reports")

    untrusted = generate_untrusted_from_config(
        orchestration_config,
        scenes_path=scenes_path,
        reports_path=reports_path,
        mode=mode, max_scenes=max_scenes,
        no_progress=no_progress,
    )
    if untrusted.get("status") not in {"done", "partial"}:
        raise RuntimeError("stage4_not_accepted:" + str(untrusted.get("status")))
    return {
        "status": "done",
        "stages": {"1": questions["status"], "2": reports["status"],
                   "4": untrusted["status"]},
        "scenes_path": str(scenes_path),
        "reports_path": str(reports_path),
        "stage4_output": str(
            load_untrusted_output(orchestration_config)
        ),
        "published_main_turns": untrusted.get("main_training_turns", 0),
    }


def load_untrusted_output(path: Path) -> Path:
    # Keep the stage4 config parser as the sole owner of path resolution.
    from .untrusted_orchestration.config import load_config
    return load_config(path)["output_dir"]
