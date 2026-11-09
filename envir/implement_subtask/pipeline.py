"""Stage-2 Subtask-bank export from accepted Stage-1 frozen scenes."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from envir.generate_question.stages import read_jsonl
from .realize import realize_scene


def _json(path: Path, value: dict) -> None:
    target = path.with_suffix(path.suffix + ".tmp")
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    target.replace(path)


def materialize_subtasks(*, scenes_path: Path, output_dir: Path,
                         max_records: int | None = None,
                         evidence_limit: int = 12,
                         strict_replay: bool = True,
                         no_progress: bool = False,
                         overwrite: bool = False) -> dict:
    """Export one accepted, evidence-grounded feedback row per Macro Task.

    Whole scenes are the atomic units of success: a scene with any invalid
    Macro contributes NO entries to the accepted Subtask output.
    """
    scenes_path = Path(scenes_path)
    output_dir = Path(output_dir)
    if not scenes_path.is_file():
        raise FileNotFoundError(scenes_path)
    if max_records is not None and (type(max_records) is not int or max_records < 1):
        raise ValueError("max_records must be null or a positive integer")
    if type(evidence_limit) is not int or evidence_limit < 0:
        raise ValueError("evidence_limit must be a nonnegative integer")
    if scenes_path.parent.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from source_scene_directory")
    filenames = {
        "subtasks": output_dir / "subtasks.jsonl",
        "scene_index": output_dir / "scene_index.jsonl",
        "failed": output_dir / "failed.jsonl",
        "summary": output_dir / "summary.json",
    }
    if not overwrite and any(path.exists() for path in filenames.values()):
        raise FileExistsError("Stage-2 output exists; choose a new directory")
    rows = read_jsonl(scenes_path, limit=max_records)
    if not rows:
        raise ValueError("No accepted Stage-1 scenes found")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {
        key: path.with_suffix(path.suffix + ".tmp")
        for key, path in filenames.items() if key != "summary"
    }
    completed_scenes = 0
    completed_macros = 0
    failures = Counter()
    macro_sizes = Counter()
    completed_items = 0
    with (temp_paths["subtasks"].open("w", encoding="utf-8") as tasks_file,
          temp_paths["scene_index"].open("w", encoding="utf-8") as index_file,
          temp_paths["failed"].open("w", encoding="utf-8") as failures_file):
        with tqdm(rows, desc="Stage 2: Macro answers & traces",
                  unit="scene", disable=no_progress,
                  dynamic_ncols=True) as bar:
            for scene in bar:
                completed_items += 1
                scene_id = scene.get("task_id")
                try:
                    status = scene.get("question_authoring", {}).get("status")
                    if status != "accepted":
                        raise ValueError("stage1_question_not_accepted")
                    result = realize_scene(
                        scene, evidence_limit=evidence_limit,
                        strict=strict_replay,
                    )
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    reason = type(exc).__name__ + ":" + str(exc)[:400]
                    failures[reason] += 1
                    failures_file.write(json.dumps({
                        "scene_task_id": scene_id, "stage": "implement_subtask",
                        "status": "failed", "reason": reason,
                    }, ensure_ascii=False) + "\n")
                else:
                    # Publish only the entire scene's validated Macro rows.
                    for item in result["macro_tasks"]:
                        tasks_file.write(json.dumps(
                            item, ensure_ascii=False
                        ) + "\n")
                    index_file.write(json.dumps({
                        key: value for key, value in result.items()
                        if key != "macro_tasks"
                    }, ensure_ascii=False) + "\n")
                    completed_scenes += 1
                    completed_macros += result["macro_task_count"]
                    macro_sizes[result["macro_task_count"]] += 1
                for stream in (tasks_file, index_file, failures_file):
                    stream.flush()
                bar.set_postfix({
                    "accepted": completed_scenes,
                    "rejected": completed_items - completed_scenes,
                    "macros": completed_macros,
                }, refresh=False)
    for key, path in temp_paths.items():
        path.replace(filenames[key])
    status = (
        "done" if completed_scenes == len(rows)
        else "partial" if completed_scenes else "failed"
    )
    summary = {
        "status": status,
        "pipeline": "maestro_implement_subtask_oracle_v1",
        "source_scenes": str(scenes_path),
        "attempted_scenes": len(rows),
        "accepted_scenes": completed_scenes,
        "failed_scenes": len(rows) - completed_scenes,
        "materialized_macro_tasks": completed_macros,
        "macro_task_count_histogram": dict(sorted(macro_sizes.items())),
        "failure_reasons": dict(failures),
        "evidence_atoms_per_step_limit": evidence_limit,
        "strict_replay": strict_replay,
        "trace_contract": (
            "Program-executed operator actions plus exact source evidence; "
            "not an LLM Sub-Agent reasoning transcript"
        ),
        "files": {name: str(path) for name, path in filenames.items()},
    }
    _json(filenames["summary"], summary)
    return summary
