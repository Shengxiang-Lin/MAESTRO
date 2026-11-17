"""LLM Author + independent Critic naturalization of frozen Oracle Subtasks."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from tqdm.auto import tqdm

from environment.generate_question.authoring.backend import OpenAICompatibleBackend
from environment.generate_question.stages import read_jsonl
from .reporting import author_critic_report
from .reuse import load_completed_accepted
from environment.streaming import (
    OrderedJournal, StreamingError, file_digest, output_lock,
    run_ordered, stable_signature,
)


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


class _ScenePublisher:
    """Single-writer public outputs; only fully reviewed DAGs get published.

    Public outputs are reconstructed from the durable Macro journal on resume,
    so a partial scene or an uncommitted tail can never be published by
    accident after a crash. Macro review/audit files grow continuously.
    """

    def __init__(self, names: dict, indexes: list[dict],
                 ordered: list[tuple], *,
                 source_subtasks: Path, source_scene_index: Path,
                 journal: OrderedJournal, flush_every: int):
        self.names = names
        self.indexes = indexes
        self.ordered = ordered
        self.source_subtasks = source_subtasks
        self.source_scene_index = source_scene_index
        self.journal = journal
        self.handles = {
            key: names[key].open("w", encoding="utf-8")
            for key in ("subtasks", "scene_index", "failed", "review_audit")
        }
        self.completed_scenes = 0
        self.accepted_scenes = 0
        self.failed_scenes = 0
        self.published_macro_tasks = 0
        self.reused_accepted_macros = 0
        self._pending_scene = []
        self.rejection_causes = Counter()
        self.flush_every = flush_every

    def _append(self, key: str, row: dict) -> None:
        self.handles[key].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handles[key].flush()

    def commit(self, index: int, accepted: bool,
               payload: dict, meta: dict, *, snapshot: bool = True) -> None:
        expected = self.ordered[index][0]
        macro_id = self.ordered[index][1]["macro_task_id"]
        self._append("review_audit", {
            "scene_task_id": expected,
            "macro_task_id": macro_id,
            "status": meta["status"],
            "attempts": meta.get("attempts", 0),
            "reused_previous_review": meta.get(
                "reused_previous_review", False
            ),
            "failures": meta.get("failures", []),
            "accepted_preview": meta.get("accepted_preview"),
        })
        for attempt in meta.get("failures", []):
            if attempt.get("stage") == "guard":
                self.rejection_causes.update(attempt.get("errors", []))
            elif attempt.get("stage") == "critic":
                self.rejection_causes.update([
                    "critic:" + str(attempt.get("reason", "unspecified"))[:120]
                ])
            elif attempt.get("stage") in {"backend", "oracle_contract"}:
                self.rejection_causes.update([attempt["stage"]])
        self._pending_scene.append((macro_id, accepted, payload, meta))
        if accepted and meta.get("reused_previous_review") is True:
            self.reused_accepted_macros += 1
        end_of_scene = (
            index == len(self.ordered) - 1
            or expected != self.ordered[index + 1][0]
        )
        if not end_of_scene:
            return
        scene = self.indexes[self.completed_scenes]
        if scene["scene_task_id"] != expected:
            raise StreamingError("scene_publisher_input_order_mismatch")
        bad = [
            {"macro_task_id": tid, "review": meta}
            for tid, accepted, _, meta in self._pending_scene if not accepted
        ]
        if bad:
            self._append("failed", {
                "scene_task_id": expected,
                "stage": "subagent_report_author_critic",
                "status": "failed",
                "failed_macro_tasks": bad,
            })
            self.failed_scenes += 1
        else:
            for _, _, row, _ in self._pending_scene:
                self._append("subtasks", row)
            self._append("scene_index", scene)
            self.accepted_scenes += 1
            self.published_macro_tasks += len(self._pending_scene)
        self.completed_scenes += 1
        self._pending_scene = []
        for key in ("subtasks", "scene_index", "failed"):
            os.fsync(self.handles[key].fileno())
        if snapshot:
            _write_json(self.names["summary"], self.summary())

    def summary(self) -> dict:
        completed = self.journal.completed
        all_done = completed == len(self.ordered)
        return {
            "status": (
                "running" if not all_done
                else "done" if self.accepted_scenes == len(self.indexes)
                else "partial" if self.accepted_scenes else "failed"
            ),
            "pipeline": "maestro_subagent_finish_author_critic_v1",
            "source_oracle_subtasks": str(self.source_subtasks),
            "source_scene_index": str(self.source_scene_index),
            "attempted_scenes": len(self.indexes),
            "completed_scenes": self.completed_scenes,
            "accepted_scenes": self.accepted_scenes,
            "failed_scenes": self.failed_scenes,
            "attempted_macro_tasks": len(self.ordered),
            "completed_macro_tasks": completed,
            "author_critic_passed_macros": self.journal.accepted,
            "author_critic_failed_macros": self.journal.rejected,
            "reused_accepted_macros": self.reused_accepted_macros,
            "freshly_reviewed_macro_tasks": (
                completed - self.reused_accepted_macros
            ),
            "published_macro_tasks": self.published_macro_tasks,
            "review_counts": {
                "accepted": self.journal.accepted,
                "failed": self.journal.rejected,
            },
            "rejection_causes_across_attempts": dict(self.rejection_causes),
            "review_audit_is_private": True,
            "fallback_counted_as_success": False,
            "teacher_answers_and_evidence_unchanged": True,
            "subagent_reports_are_oracle_narratives_not_real_llm_tool_runs": True,
            "streaming": {
                "ordered_writes": True,
                "flush_every": self.flush_every,
                "resumable": True,
                "checkpoint": str(self.journal.state_path),
                "resume_at_macro": completed,
                "reviewed_accepted_provisional_file": str(
                    self.journal.paths["accepted"]),
                "reviewed_rejected_provisional_file": str(
                    self.journal.paths["rejected"]),
                "publication_policy": "complete_scene_only",
            },
            "files": {k: str(p) for k, p in self.names.items()},
        }

    def close(self) -> None:
        try:
            for file in self.handles.values():
                file.flush()
                os.fsync(file.fileno())
            _write_json(self.names["summary"], self.summary())
        finally:
            for file in self.handles.values():
                file.close()


def naturalize_subtasks(
    *, source_subtasks: Path, source_scene_index: Path, output_dir: Path,
    api_base: str, model: str,
    critic_api_base: str | None = None, critic_model: str | None = None,
    api_key_env: str | None = None, critic_api_key_env: str | None = None,
    workers: int = 8, max_revisions: int = 2, timeout: int = 180,
    max_scenes: int | None = None, no_progress: bool = False,
    overwrite: bool = False, flush_every: int = 1,
    reuse_from: Path | None = None,
    result_policy: str = "llm_authored",
    clients: tuple | None = None,
) -> dict:
    """Stream per-Macro reviews, publish complete scenes, and resume safely.

    Independent Macro LLM calls can finish out of order. Journal records
    persist as soon as the contiguous prefix completes; accepted/rejected
    counters update when EACH call completes, not when its whole scene
    is eventually publishable.
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
    if type(flush_every) is not int or flush_every < 1:
        raise ValueError("flush_every must be a positive integer")
    if result_policy not in {"llm_authored", "code_owned"}:
        raise ValueError("unknown_subagent_result_policy")
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
            "review_audit": "review_audit.jsonl",
            "summary": "summary.json",
        }.items()
    }
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
            ordered.append((
                sid, row, scene["question"],
                task_id == scene["terminal_macro_task_id"],
            ))
    previous = {}
    previous_fingerprint = None
    if reuse_from is not None:
        reuse_from = Path(reuse_from).expanduser().resolve()
        if reuse_from == output_dir.resolve():
            raise ValueError("reuse_from and output_dir must be different")
        previous, previous_fingerprint = load_completed_accepted(
            reuse_from, ordered,
        )
    if clients is None:
        clients = _clients(
            api_base=api_base, model=model,
            critic_api_base=critic_api_base, critic_model=critic_model,
            api_key_env=api_key_env, critic_api_key_env=critic_api_key_env,
            timeout=timeout,
        )
    author, critic = clients

    def review(entry):
        sid, row, question, is_terminal = entry
        key = sid + "::" + row["macro_task_id"]
        if key in previous:
            # The old report passed the independent Critic and has been
            # checked against this EXACT Oracle. No LLM call is needed.
            return sid, row["macro_task_id"], previous[key], {
                "status": "accepted", "attempts": 0, "failures": [],
                "reused_previous_review": True,
            }
        try:
            accepted, feedback = author_critic_report(
                row, question=question, author=author, critic=critic,
                max_revisions=max_revisions, is_terminal=is_terminal,
                # The current Task Plan carries typed evidence rather than
                # the retired evidence_trust_version marker. Honor the
                # configured policy directly instead of downgrading it.
                result_policy=result_policy,
            )
        except (KeyError, TypeError, ValueError) as exc:
            accepted, feedback = None, {
                "status": "failed", "attempts": 0,
                "failures": [{
                    "stage": "oracle_contract",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:240],
                }],
            }
        return sid, row["macro_task_id"], accepted, feedback

    def classify(result):
        sid, task_id, accepted, feedback = result
        ok = accepted is not None
        data = (accepted if ok else {
            "scene_task_id": sid, "macro_task_id": task_id, "review": feedback
        })
        return ok, data, {
            "status": feedback["status"],
            "attempts": feedback.get("attempts", 0),
            "reused_previous_review": feedback.get(
                "reused_previous_review", False
            ),
            "failures": feedback.get("failures", []),
            "accepted_preview": (
                accepted["finish_result"] if ok else None
            ),
        }

    task_ids = [sid + "::" + row["macro_task_id"] for sid, row, _, _ in ordered]
    # Changing models, revisions, the frozen input bytes or the author/critic
    # implementation invalidates a prior checkpoint. workers and display
    # options do not influence semantics, so may change on resume.
    code_root = Path(__file__).resolve().parent
    signature = stable_signature({
        "schema": "maestro.subagent_reports.stream.v1",
        "task_ids": task_ids,
        "oracle_subtasks_hash": file_digest(source_subtasks),
        "scene_index_hash": file_digest(source_scene_index),
        "author_model": model, "author_endpoint": api_base,
        "critic_model": critic_model or model,
        "critic_endpoint": critic_api_base or api_base,
        "max_revisions": max_revisions, "timeout": timeout,
        "result_policy": result_policy,
        "reuse_source_fingerprint": previous_fingerprint,
        "prompt_code_hash": file_digest(code_root / "reporting.py"),
        "visible_evidence_code_hash": file_digest(
            code_root / "visible_evidence.py"
        ),
        "stage_local_claim_code_hash": file_digest(
            code_root / "stage_local_claims.py"
        ),
        "pipeline_code_hash": file_digest(Path(__file__)),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    if names["summary"].is_file() and not overwrite:
        previous = json.loads(names["summary"].read_text(encoding="utf-8"))
        if previous.get("status") in {"done", "partial", "failed"}:
            raise FileExistsError(
                "Completed Sub-Agent reporting output already exists; "
                "use a new output folder or enable overwrite"
            )
    checkpoint = output_dir / "subagent_reports_progress.json"
    if not overwrite and not checkpoint.exists() and any(
        path.exists() for path in names.values()
    ):
        raise StreamingError(
            "Existing legacy report output has no resumable checkpoint; "
            "choose a new --output-dir or enable overwrite"
        )
    with output_lock(output_dir, "subagent_reports"):
        with OrderedJournal(
            output_dir, "subagent_reports", signature, task_ids,
            flush_every=flush_every, restart=overwrite,
        ) as journal:
            publisher = _ScenePublisher(
                names, indexes, ordered,
                source_subtasks=source_subtasks,
                source_scene_index=source_scene_index,
                journal=journal, flush_every=flush_every,
            )
            try:
                # Rebuild independently from committed checkpoints. Public
                # output files never contain a partially accepted DAG.
                for ix, (good, data, meta) in enumerate(journal.replay()):
                    publisher.commit(ix, good, data, meta, snapshot=False)
                _write_json(names["summary"], publisher.summary())
                run_ordered(
                    ordered, review, classify, journal=journal,
                    workers=workers, desc="Sub-Agent report Author/Critic",
                    unit="macro", no_progress=no_progress,
                    on_commit=publisher.commit,
                    on_exception=lambda ix, exc: (
                        False,
                        {
                            "scene_task_id": ordered[ix][0],
                            "macro_task_id": ordered[ix][1]["macro_task_id"],
                            "review": {
                                "status": "failed", "attempts": 0,
                                "failures": [{
                                    "stage": "backend",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc)[:240],
                                }],
                            },
                        },
                        {
                            "status": "failed", "attempts": 0,
                            "failures": [{
                                "stage": "backend",
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:240],
                            }],
                        },
                    ),
                )
                result = publisher.summary()
            finally:
                publisher.close()
            return result
