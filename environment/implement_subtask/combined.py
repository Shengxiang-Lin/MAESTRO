"""Stage 2 single-entry generation: deterministic Oracle -> LLM finish reports.

The naturalize-subtasks --scenes mode exports verified Macro Oracle
records under <report-dir>/oracle/, then runs the same independent
Author/Critic pipeline over them. Existing Oracle-only and pre-materialized
report APIs remain available.

Oracle caching is fail-closed: its manifest binds exact Stage-1 scene
bytes, materializer code, settings, and export hashes. A completed Oracle
stage can be reused after a crash but never silently switched to a
different scene bank or executor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from environment.streaming import StreamingError, file_digest, stable_signature
from . import pipeline as oracle_pipeline
from . import realize as oracle_realize
from .pipeline import materialize_subtasks
from .naturalize import naturalize_subtasks

MANIFEST_VERSION = "maestro.stage2.oracle_cache.v1"


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def _oracle_signature(*, scenes: Path, max_scenes: int | None,
                      evidence_limit: int, strict_replay: bool) -> str:
    return stable_signature({
        "version": MANIFEST_VERSION,
        "scenes_path": str(scenes.resolve()),
        "scenes_sha256": file_digest(scenes),
        "max_scenes": max_scenes,
        "evidence_limit": evidence_limit,
        "strict_replay": strict_replay,
        "oracle_pipeline_sha256": file_digest(Path(oracle_pipeline.__file__)),
        "oracle_realize_sha256": file_digest(Path(oracle_realize.__file__)),
    })


def generate_subagent_reports(
    *, scenes_path: Path, output_dir: Path,
    reporting_options: dict, evidence_limit: int = 12,
    strict_replay: bool = True, max_scenes: int | None = None,
    no_progress: bool = False, clients: tuple | None = None,
) -> dict:
    """Build Oracle + Sub-Agent reports in one call.

    Oracle teacher data live in <output-dir>/oracle and stay private.
    The reviewed reports are streamed to <output-dir>. A finished valid
    Oracle is reused after a crash without recomputing Stage 2A.
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
    manifest_path = oracle_dir / "manifest.json"
    oracle_files = {
        name: oracle_dir / name
        for name in ("subtasks.jsonl", "scene_index.jsonl",
                     "failed.jsonl", "summary.json")
    }
    desired_signature = _oracle_signature(
        scenes=scenes_path, max_scenes=max_scenes,
        evidence_limit=evidence_limit, strict_replay=strict_replay,
    )
    regenerate = (
        bool(reporting_options.get("overwrite"))
        or not manifest_path.is_file()
    )
    if regenerate:
        if not reporting_options.get("overwrite") and any(
            path.exists() for path in oracle_files.values()
        ):
            raise StreamingError(
                "Partial or legacy Oracle folder exists without a manifest; "
                "use a fresh output directory to avoid mixing inputs"
            )
        result = materialize_subtasks(
            scenes_path=scenes_path,
            output_dir=oracle_dir,
            max_records=max_scenes,
            evidence_limit=evidence_limit,
            strict_replay=strict_replay,
            no_progress=no_progress,
            overwrite=bool(reporting_options.get("overwrite", False)),
        )
        hashes = {
            name: file_digest(path)
            for name, path in oracle_files.items()
        }
        _atomic_json(manifest_path, {
            "version": MANIFEST_VERSION,
            "signature": desired_signature,
            "oracle_status": result["status"],
            "accepted_scenes": result["accepted_scenes"],
            "failed_scenes": result["failed_scenes"],
            "files_sha256": hashes,
        })
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StreamingError("invalid_oracle_manifest") from exc
        if (manifest.get("version") != MANIFEST_VERSION
                or manifest.get("signature") != desired_signature):
            raise StreamingError(
                "Oracle cache does not match the current scenes or implementation; "
                "use a new output directory"
            )
        if set(manifest.get("files_sha256", {})) != set(oracle_files):
            raise StreamingError("oracle_cache_manifest_incomplete")
        for name, expected_hash in manifest["files_sha256"].items():
            path = oracle_files[name]
            if (not path.is_file() or file_digest(path) != expected_hash):
                raise StreamingError("oracle_cache_file_changed:" + name)
        result = json.loads(
            oracle_files["summary.json"].read_text(encoding="utf-8")
        )
        if (result.get("accepted_scenes") != manifest["accepted_scenes"]
                or result.get("failed_scenes") != manifest["failed_scenes"]
                or result.get("status") != manifest["oracle_status"]):
            raise StreamingError("oracle_cache_summary_mismatch")

    if result["accepted_scenes"] < 1:
        return {
            "status": "failed",
            "pipeline": "maestro_stage2_oracle_and_subagent_reports_v1",
            "oracle_status": result["status"],
            "oracle_accepted_scenes": 0,
            "oracle_failed_scenes": result["failed_scenes"],
            "oracle_dir": str(oracle_dir),
            "failure_reason": "no_accepted_oracle_scenes",
        }

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
    reviewed = naturalize_subtasks(**naturalize_options)
    return {
        **reviewed,
        "pipeline": "maestro_stage2_oracle_and_subagent_reports_v1",
        "oracle_status": result["status"],
        "oracle_accepted_scenes": result["accepted_scenes"],
        "oracle_failed_scenes": result["failed_scenes"],
        "oracle_macro_tasks": result["materialized_macro_tasks"],
        "oracle_dir": str(oracle_dir),
    }
