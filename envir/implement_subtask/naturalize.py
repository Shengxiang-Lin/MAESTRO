"""LLM Author + independent Critic naturalization of frozen Oracle Subtasks."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from tqdm.auto import tqdm

from envir.generate_question.backend import OpenAICompatibleBackend
from envir.generate_question.stages import read_jsonl
from .reporting import author_critic_report


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _clients(*, api_base: str, model: str,
             critic_api_base: str | None, critic_model: str | None,
             api_key_env: str | None, critic_api_key_env: str | None,
             timeout: int) -> tuple:
    if not api_base or not model:
        raise ValueError("reporting API base and model are required")
    key = os.environ.get(api_key_env) if api_key_env else None
    critic_key = (
        os.environ.get(critic_api_key_env)
        if critic_api_key_env else key
    )
    return (
        OpenAICompatibleBackend(
            base_url=api_base, model=model, api_key=key,
            temperature=0.55, max_tokens=1100, timeout=timeout,
        ),
        OpenAICompatibleBackend(
            base_url=critic_api_base or api_base,
            model=critic_model or model, api_key=critic_key,
            temperature=0.0, max_tokens=800, timeout=timeout,
        ),
    )


def naturalize_subtasks(
    *, source_subtasks: Path, source_scene_index: Path, output_dir: Path,
    api_base: str, model: str,
    critic_api_base: str | None = None, critic_model: str | None = None,
    api_key_env: str | None = None, critic_api_key_env: str | None = None,
    workers: int = 8, max_revisions: int = 2, timeout: int = 180,
    max_scenes: int | None = None, no_progress: bool = False,
    overwrite: bool = False, clients: tuple | None = None,
) -> dict:
    """Produce only COMPLETE-scene accepted Sub-Agent finish messages.

    No Stage-1 World re-execution or model-based answer generation is needed.
    Teacher rows are frozen and are not rewritten by Author/Critic. A single
    rejected Macro rejects the whole scene, preserving downstream DAG use.
    """
    source_subtasks = Path(source_subtasks)
    source_scene_index = Path(source_scene_index)
    output_dir = Path(output_dir)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    if type(timeout) is not int or timeout < 1:
        raise ValueError("timeout must be positive")
    if type(max_revisions) is not int or not 0 <= max_revisions <= 8:
        raise ValueError("max_revisions must be between 0 and 8")
    if max_scenes is not None and (
        type(max_scenes) is not int or max_scenes < 1
    ):
        raise ValueError("max_scenes must be null or positive")
    if not source_subtasks.is_file() or not source_scene_index.is_file():
        raise FileNotFoundError("Stage-2 oracle subtasks and scene_index required")
    if output_dir.resolve() in {
        source_subtasks.parent.resolve(), source_scene_index.parent.resolve()
    }:
        raise ValueError("reporting output must not overwrite frozen oracle input")
    names = {
        key: output_dir / value
        for key, value in {
            "subtasks": "subtasks.jsonl",
            "scene_index": "scene_index.jsonl",
            "failed": "failed.jsonl",
            "summary": "summary.json",
        }.items()
    }
    if not overwrite and any(path.exists() for path in names.values()):
        raise FileExistsError("reporting output exists; choose a new directory")
    indexes = read_jsonl(source_scene_index, limit=max_scenes)
    if not indexes:
        raise ValueError("empty_scene_index")
    index = {r["scene_task_id"]: r for r in indexes}
    if len(index) != len(indexes):
        raise ValueError("duplicate_scene_index_id")
    all_rows = read_jsonl(source_subtasks)
    groups = defaultdict(dict)
    for row in all_rows:
        sid = row["scene_task_id"]
        if sid not in index:
            continue
        tid = row["macro_task_id"]
        if tid in groups[sid]:
            raise ValueError("duplicate_macro_task:" + sid + ":" + tid)
        groups[sid][tid] = row
    ordered = []
    for scene in indexes:
        sid = scene["scene_task_id"]
        expected = [task for wave in scene["reference_waves"] for task in wave]
        if len(expected) != scene["macro_task_count"]:
            raise ValueError("scene_index_macro_count_mismatch:" + sid)
        if set(groups[sid]) != set(expected):
            raise ValueError("incomplete_oracle_macro_graph:" + sid)
        for task_id in expected:
            row = groups[sid][task_id]
            if any(parent not in set(expected) for parent in row["depends_on"]):
                raise ValueError("unknown_upstream_handoff:" + sid)
            ordered.append((sid, row, scene["question"]))
    if clients is None:
        clients = _clients(
            api_base=api_base, model=model,
            critic_api_base=critic_api_base, critic_model=critic_model,
            api_key_env=api_key_env, critic_api_key_env=critic_api_key_env,
            timeout=timeout,
        )
    author, critic = clients

    def review(entry):
        sid, row, question = entry
        accepted, feedback = author_critic_report(
            row, question=question, author=author, critic=critic,
            max_revisions=max_revisions,
        )
        return sid, row["macro_task_id"], accepted, feedback

    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(tqdm(
            pool.map(review, ordered), total=len(ordered),
            desc="Sub-Agent report Author/Critic", unit="macro",
            disable=no_progress,
        ))
    by_scene = defaultdict(list)
    for sid, task_id, accepted, feedback in completed:
        by_scene[sid].append((task_id, accepted, feedback))
    publish, keep_indices, failures = [], [], []
    feedback_counts = Counter()
    for scene in indexes:
        sid = scene["scene_task_id"]
        reports = by_scene[sid]
        failed = [(task_id, fb) for task_id, row, fb in reports
                  if row is None]
        feedback_counts.update(fb["status"] for _, _, fb in reports)
        if failed:
            failures.append({
                "scene_task_id": sid,
                "stage": "subagent_report_author_critic",
                "status": "failed",
                "failed_macro_tasks": [
                    {"macro_task_id": task_id, "review": report}
                    for task_id, report in failed
                ],
            })
            continue
        publish.extend(row for _, row, _ in reports)
        keep_indices.append(scene)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(names["subtasks"], publish)
    _write_jsonl(names["scene_index"], keep_indices)
    _write_jsonl(names["failed"], failures)
    summary = {
        "status": "done" if len(keep_indices) == len(indexes)
                  else "partial" if keep_indices else "failed",
        "pipeline": "maestro_subagent_finish_author_critic_v1",
        "source_oracle_subtasks": str(source_subtasks),
        "source_scene_index": str(source_scene_index),
        "attempted_scenes": len(indexes),
        "accepted_scenes": len(keep_indices),
        "failed_scenes": len(failures),
        "attempted_macro_tasks": len(ordered),
        "author_critic_passed_macros": feedback_counts["accepted"],
        "author_critic_failed_macros": feedback_counts["failed"],
        "published_macro_tasks": len(publish),
        "review_counts": dict(feedback_counts),
        "fallback_counted_as_success": False,
        "teacher_answers_and_evidence_unchanged": True,
        "subagent_reports_are_oracle_narratives_not_real_llm_tool_runs": True,
        "files": {k: str(p) for k, p in names.items()},
    }
    _write_json(names["summary"], summary)
    return summary
