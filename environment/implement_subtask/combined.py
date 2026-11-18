"""Stage 2 single-entry generation: deterministic Oracle -> LLM finish reports.

The naturalize-subtasks --scenes mode exports verified Macro Oracle
records under <report-dir>/oracle/, then runs the same independent
Author/Critic pipeline over them. Existing Oracle-only and pre-materialized
report APIs remain available.

The Oracle is inexpensive and deterministic. Always rebuild it from the
current accepted Scenes instead of maintaining a separate content/code/
parameter-signed Oracle cache. LLM report checkpoints remain independent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .pipeline import materialize_subtasks
from .naturalize import review_subagent_reports

def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def generate_subagent_reports(
    *, scenes_path: Path, output_dir: Path,
    reporting_options: dict, evidence_limit: int = 12,
    strict_replay: bool = True, max_scenes: int | None = None,
    no_progress: bool = False, clients: tuple | None = None,
) -> dict:
    """Build Oracle + Sub-Agent reports in one call.

    Oracle teacher data live in <output-dir>/oracle and stay private.
    They are rebuilt on every invocation from the current Scenes, without
    manifest/signature validation or stale-cache reuse. The reviewed reports
    are streamed separately to <output-dir> and retain their resume checks.
    """
    scenes_path = Path(scenes_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not scenes_path.is_file():
        raise FileNotFoundError(scenes_path)
    if not isinstance(reporting_options, dict):
        raise TypeError("reporting_options must be a dict")
    if max_scenes is not None and (
        type(max_scenes) is not int or max_scenes < 1
    ):
        raise ValueError("max_scenes must be positive or null")
    if type(evidence_limit) is not int or evidence_limit < 0:
        raise ValueError("evidence_limit must be nonnegative")
    if type(strict_replay) is not bool:
        raise ValueError("strict_replay must be boolean")
    if scenes_path.parent == output_dir or scenes_path.parent == output_dir / "oracle":
        raise ValueError("Stage-2 output must not be the input Scene directory")
    oracle_dir = output_dir / "oracle"
    oracle_files = {
        name: oracle_dir / name
        for name in ("subtasks.jsonl", "scene_index.jsonl",
                     "failed.jsonl", "summary.json")
    }
    # No Oracle cache fingerprints: rebuild the cheap deterministic
    # WorkProducts with the current code, Scene bank and parameters.
    result = materialize_subtasks(
        scenes_path=scenes_path,
        output_dir=oracle_dir,
        max_records=max_scenes,
        evidence_limit=evidence_limit,
        strict_replay=strict_replay,
        no_progress=no_progress,
        overwrite=True,
    )
    # Drop any pre-existing legacy cache manifest. It is no longer read.
    (oracle_dir / "manifest.json").unlink(missing_ok=True)

    if result["accepted_scenes"] < 1:
        # Even an early Oracle failure must publish the promised Stage-2
        # summary. The CLI should point to an existing explanation, not to a
        # non-existent summary.json after rejecting every Scene.
        failure = {
            "status": "failed",
            "pipeline": "maestro_stage2_oracle_and_subagent_reports_v1",
            "oracle_status": result["status"],
            "oracle_accepted_scenes": 0,
            "oracle_failed_scenes": result["failed_scenes"],
            "oracle_dir": str(oracle_dir),
            "failure_reason": "no_accepted_oracle_scenes",
            "oracle_failure_reasons": result.get("failure_reasons", {}),
        }
        _atomic_json(output_dir / "summary.json", failure)
        return failure

    naturalize_options = dict(reporting_options)
    naturalize_options.update({
        "source_subtasks": oracle_files["subtasks.jsonl"],
        "source_scene_index": oracle_files["scene_index.jsonl"],
        "output_dir": output_dir,
        # Oracle stage already applied this limit.
        "max_scenes": None,
        "no_progress": no_progress or naturalize_options.get(
            "no_progress", False
        ),
    })
    if clients is not None:
        naturalize_options["clients"] = clients
    reviewed = review_subagent_reports(**naturalize_options)
    return {
        **reviewed,
        "pipeline": "maestro_stage2_oracle_and_subagent_reports_v1",
        "oracle_status": result["status"],
        "oracle_accepted_scenes": result["accepted_scenes"],
        "oracle_failed_scenes": result["failed_scenes"],
        "oracle_macro_tasks": result["materialized_macro_tasks"],
        "oracle_dir": str(oracle_dir),
    }
