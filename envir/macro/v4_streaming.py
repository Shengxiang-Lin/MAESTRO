"""Incremental, crash-recoverable V4 Question naturalization.

Bounded Author/Critic futures run concurrently, but private/public/audit
JSONL files are appended IN ORIGINAL TASK ORDER by a single writer. Every
flush fsyncs all three outputs before atomically advancing the checkpoint
(committed byte offsets and SHA-256 hashes). On resume any uncommitted tails
are discarded, committed records are integrity checked and never re-called.

This only changes Question text. Macro TaskGraphs, EvidenceGraphs, Feedback
and source worlds remain immutable. The progress file is not exposed to
the Main Agent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm.auto import tqdm

from environment.question_generation.streaming import question_output_lock
from .v4_question_review import (
    _fingerprint, naturalize_one, macro_task_view, reference_question,
    polished_reference_question,
)
from .v4_validation import validate_clean_scene


SCHEMA = "maestro.v4.question_stream.v1"
STATE_FILE = "question_progress_state_v4.json"
OUTPUT_FILES = {
    "private": "private_clean_scenes_v4.jsonl",
    "public": "public_questions_v4.jsonl",
    "audit": "question_review_audit_v4.jsonl",
}


class V4QuestionCheckpointError(RuntimeError):
    """Existing output cannot safely be continued."""


def _encoded(obj: dict) -> bytes:
    # Preserve the insertion order of typed input slots. The clean replay
    # compiler respects those slots when constructing its internal evidence
    # sequence, so a JSON writer MUST NOT alphabetically reorder dict keys.
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _durable_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signature_for_v4_scenes(scenes: list[dict], args) -> str:
    """Hash all immutable inputs, model settings and executable review code.

    API KEY VALUES are intentionally excluded. Workers/flush/progress style
    do not influence logical output and may change when resuming.
    """
    h = hashlib.sha256()
    h.update(SCHEMA.encode("utf-8"))
    for scene in scenes:
        h.update(_encoded(scene))
    source_files = (
        Path(__file__),
        Path(__file__).with_name("v4_question_review.py"),
        Path(__file__).with_name("v4_taskfirst.py"),
        Path(__file__).with_name("v4_validation.py"),
        Path(__file__).with_name("v4_cli.py"),
        Path(__file__).parents[2] / "environment/question_generation/backend.py",
    )
    payload = {
        "base_scene_sha256": h.hexdigest(),
        "task_ids": [scene["task_id"] for scene in scenes],
        "question_api_base": args.question_api_base,
        "question_model": args.question_model,
        "critic_api_base": args.question_critic_api_base or args.question_api_base,
        "critic_model": args.question_critic_model or args.question_model,
        "question_temperature": args.question_temperature,
        "critic_temperature": args.question_critic_temperature,
        "question_max_tokens": args.question_max_tokens,
        "critic_max_tokens": args.question_critic_max_tokens,
        "max_revisions": args.max_revisions,
        "timeout": args.question_timeout,
        "json_mode": not args.question_no_json_mode,
        "source_code_sha256": {
            file.name: _sha_file(file) for file in source_files
        },
    }
    return _fingerprint(payload)


class V4QuestionProgress:
    """Durable three-file transaction journal with a committed prefix."""

    def __init__(self, directory: Path, signature: str, task_ids: list[str],
                 *, flush_every: int = 1, overwrite: bool = False):
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = {
            name: self.directory / value for name, value in OUTPUT_FILES.items()
        }
        self.checkpoint_path = self.directory / STATE_FILE
        self.summary_path = self.directory / "macro_v4_summary.json"
        self.signature = signature
        self.task_ids = task_ids
        self.flush_every = flush_every
        self.handles = {}
        self.offsets = {}
        self.digests = {}
        self._uncommitted = 0
        if overwrite:
            for path in (
                *self.paths.values(), self.checkpoint_path, self.summary_path,
                self.checkpoint_path.with_suffix(".json.tmp"),
            ):
                path.unlink(missing_ok=True)
        if self.checkpoint_path.exists():
            self.state = json.loads(
                self.checkpoint_path.read_text(encoding="utf-8")
            )
            if (self.state.get("schema") != SCHEMA or
                    self.state.get("signature") != signature or
                    self.state.get("task_ids") != task_ids):
                raise V4QuestionCheckpointError(
                    "V4 checkpoint input/model/code signature differs. "
                    "Use a different --output-dir, or explicitly --overwrite "
                    "to restart from scratch."
                )
            if not 0 <= self.completed <= len(task_ids):
                raise V4QuestionCheckpointError("invalid completed count")
            self._recover_committed()
        else:
            if (self.summary_path.exists()
                    or any(path.exists() for path in self.paths.values())):
                raise V4QuestionCheckpointError(
                    "V4 output exists without checkpoint; refusing to "
                    "overwrite it. Use a fresh --output-dir or --overwrite."
                )
            self.state = {
                "schema": SCHEMA,
                "signature": signature,
                "task_ids": task_ids,
                "completed": 0,
                "accepted": 0,
                "fallback": 0,
                "accepted_after_revision": 0,
                "author_calls": 0,
                "critic_calls": 0,
                "failure_reasons": {},
                "offsets": {name: 0 for name in self.paths},
                "sha256": {
                    name: hashlib.sha256(b"").hexdigest()
                    for name in self.paths
                },
            }
            for key, path in self.paths.items():
                path.touch()
                self.offsets[key] = 0
                self.digests[key] = hashlib.sha256()
            _durable_json(self.checkpoint_path, self.state)
        self.handles = {
            name: path.open("ab") for name, path in self.paths.items()
        }

    @property
    def completed(self) -> int:
        return int(self.state["completed"])

    def _recover_committed(self):
        for key, path in self.paths.items():
            committed = self.state.get("offsets", {}).get(key)
            if not isinstance(committed, int) or committed < 0 or not path.is_file():
                raise V4QuestionCheckpointError("missing output: " + str(path))
            if path.stat().st_size < committed:
                raise V4QuestionCheckpointError(
                    "previously committed output was truncated: " + str(path)
                )
            digest = hashlib.sha256()
            with path.open("rb") as reader:
                remaining = committed
                while remaining:
                    chunk = reader.read(min(remaining, 1 << 20))
                    if not chunk:
                        raise V4QuestionCheckpointError("truncated output")
                    digest.update(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != self.state.get("sha256", {}).get(key):
                raise V4QuestionCheckpointError(
                    "previously committed output checksum mismatch: " + str(path)
                )
            self.offsets[key] = committed
            self.digests[key] = digest
        self._verify_alignment()
        for key, path in self.paths.items():
            # Discard an unfinished multi-file row from the previous process,
            # never a committed row.
            with path.open("r+b") as writer:
                writer.truncate(self.offsets[key])

    def _read_committed(self, key: str):
        with self.paths[key].open("rb") as reader:
            text = reader.read(self.offsets[key])
        try:
            return [json.loads(line) for line in text.splitlines()]
        except (UnicodeError, ValueError) as exc:
            raise V4QuestionCheckpointError(
                "invalid committed JSONL in " + key
            ) from exc

    def _verify_alignment(self):
        rows = {key: self._read_committed(key) for key in self.paths}
        if any(len(row) != self.completed for row in rows.values()):
            raise V4QuestionCheckpointError(
                "committed private/public/audit JSONL counts disagree"
            )
        accepted = fallback = author_calls = critic_calls = revised = 0
        reasons = Counter()
        for index in range(self.completed):
            private = rows["private"][index]
            public = rows["public"][index]
            audit = rows["audit"][index]
            expected = self.task_ids[index]
            meta = private.get("question_authoring", {})
            if (audit.get("index") != index or
                    any(record.get("task_id") != expected
                        for record in (private, public, audit)) or
                    private.get("question") != public.get("question") or
                    audit.get("question") != private.get("question") or
                    audit.get("review") != meta):
                raise V4QuestionCheckpointError(
                    "committed outputs do not match task order or review audit"
                )
            status = meta.get("status")
            if status == "accepted":
                accepted += 1
                revised += meta.get("revision_rounds", 0) > 0
            elif status == "fallback":
                fallback += 1
                reasons[meta.get("reason", "unknown")] += 1
            else:
                raise V4QuestionCheckpointError("invalid committed review status")
            author_calls += meta.get("author_calls", 0)
            critic_calls += meta.get("critic_calls", 0)
        if any((
            self.state["accepted"] != accepted,
            self.state["fallback"] != fallback,
            self.state["author_calls"] != author_calls,
            self.state["critic_calls"] != critic_calls,
            self.state["accepted_after_revision"] != revised,
            self.state["failure_reasons"] != dict(reasons),
        )):
            raise V4QuestionCheckpointError("checkpoint counters disagree with audit")

    def committed_questions(self) -> set[str]:
        return {
            _normalize_question(row["question"])
            for row in self._read_committed("private")
        }

    def append(self, index: int, scene: dict):
        if index != self.completed or scene["task_id"] != self.task_ids[index]:
            raise V4QuestionCheckpointError(
                "out-of-order task write or mismatched task ID"
            )
        issues = validate_clean_scene(scene)
        if issues:
            raise ValueError(
                "cannot persist scene with invalid replay: " + str(issues)
            )
        review = scene["question_authoring"]
        if review.get("status") not in {"accepted", "fallback"}:
            raise V4QuestionCheckpointError("review must accept or fall back")
        audit = {
            "index": index,
            "task_id": scene["task_id"],
            "reference_question": review["reference_question"],
            "question": scene["question"],
            "review": review,
        }
        rows = {
            "private": scene,
            "public": scene["public"],
            "audit": audit,
        }
        for key, row in rows.items():
            data = _encoded(row)
            self.handles[key].write(data)
            self.offsets[key] += len(data)
            self.digests[key].update(data)
        self.state["completed"] += 1
        self.state[review["status"]] += 1
        self.state["author_calls"] += review["author_calls"]
        self.state["critic_calls"] += review["critic_calls"]
        if review["status"] == "accepted" and review["revision_rounds"] > 0:
            self.state["accepted_after_revision"] += 1
        if review["status"] == "fallback":
            reason = review["reason"]
            reasons = self.state["failure_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
        self._uncommitted += 1
        if self._uncommitted >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._uncommitted:
            return
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        self.state["offsets"] = dict(self.offsets)
        self.state["sha256"] = {
            key: digest.hexdigest() for key, digest in self.digests.items()
        }
        _durable_json(self.checkpoint_path, self.state)
        self._uncommitted = 0

    def close(self):
        try:
            self.flush()
        finally:
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def stats(self) -> dict:
        return {
            "mode": "optional_llm_author_critic",
            "accepted": self.state["accepted"],
            "fallback": self.state["fallback"],
            "accepted_after_revision": self.state["accepted_after_revision"],
            "author_calls": self.state["author_calls"],
            "critic_calls": self.state["critic_calls"],
            "failure_reasons": dict(self.state["failure_reasons"]),
            "max_revisions": None,  # Filled by the calling CLI.
            "model_review_not_formal_proof": True,
            "streaming": {
                "enabled": True,
                "ordered_writes": True,
                "completed": self.completed,
                "total": len(self.task_ids),
                "flush_every": self.flush_every,
                "checkpoint": str(self.checkpoint_path),
            },
        }


def _normalize_question(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _fallback(scene, reason: str, *, detail: str = "") -> tuple[str, dict]:
    original = reference_question(scene)
    question = polished_reference_question(original)
    return question, {
        "pipeline": "v4_benchmark_style_graph_primary_author_critic_repair",
        "review_policy_version": "v4.5_graph_primary_benchmark_style",
        "safe_reference_polish_applied": question != original,
        "status": "fallback",
        "reason": reason,
        "reference_question": original,
        "final_question": question,
        "macro_task_view_sha256": _fingerprint(macro_task_view(scene)),
        "macro_graph_sha256": _fingerprint(scene["macro_graph"]),
        "reference_sha256": _fingerprint(original),
        "accepted_review": None,
        "audit": [{"stage": "unhandled_exception", "detail": detail[:180]}],
        "revision_rounds": 0,
        "author_calls": 0,
        "critic_calls": 0,
        "semantic_equivalence_certified": False,
    }


def _complete(scene, question: str, meta: dict, seen: set[str]):
    """Commit-time uniqueness + full clean replay before writing any bytes."""
    normalized = _normalize_question(question)
    if meta["status"] == "accepted" and normalized in seen:
        question = polished_reference_question(meta["reference_question"])
        meta["safe_reference_polish_applied"] = (
            question != meta["reference_question"]
        )
        meta["status"] = "fallback"
        meta["reason"] = "duplicate_naturalized_question"
        meta["final_question"] = question
        meta["accepted_review"] = None
        for row in meta["audit"]:
            if row.get("accepted"):
                row["accepted"] = False
                row["failure"] = "duplicate_naturalized_question"
    scene["question"] = question
    scene["public"]["question"] = question
    scene["question_authoring"] = meta
    issues = validate_clean_scene(scene)
    if issues and meta["status"] == "accepted":
        # An accepted LLM rewrite is never allowed to corrupt the evidence
        # replay or the frozen macro-task semantic contract.
        question = polished_reference_question(meta["reference_question"])
        meta["safe_reference_polish_applied"] = (
            question != meta["reference_question"]
        )
        meta["status"] = "fallback"
        meta["reason"] = "postflight_replay_failed"
        meta["final_question"] = question
        meta["accepted_review"] = None
        for row in meta["audit"]:
            if row.get("accepted"):
                row["accepted"] = False
                row["failure"] = "postflight_replay_failed"
        scene["question"] = question
        scene["public"]["question"] = question
        issues = validate_clean_scene(scene)
    if issues:
        raise ValueError(
            f"V4 input or fallback scene failed replay ({scene['task_id']}): {issues}"
        )
    seen.add(_normalize_question(scene["question"]))
    return scene


def _attempt(scene, *, author, critic, max_revisions):
    try:
        return naturalize_one(
            scene, author=author, critic=critic, max_revisions=max_revisions
        )
    except Exception as exc:
        # Preserve the frozen TaskGraph even on one misbehaving model request.
        return _fallback(scene, "unhandled_author_critic_error", detail=str(exc))


def stream_naturalize_scenes(
    scenes: list[dict], *, args, output_dir: Path,
    author, critic, overwrite: bool = False,
) -> tuple[dict, int]:
    """Incremental ordered writer, resume, and real-time tqdm counters."""
    if not scenes:
        raise ValueError("need at least one V4 scene")
    if args.question_workers < 1 or args.flush_every < 1:
        raise ValueError("question-workers and flush-every must be positive")
    signature = signature_for_v4_scenes(scenes, args)
    task_ids = [item["task_id"] for item in scenes]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate Task IDs in V4 input")
    with question_output_lock(output_dir):
        with V4QuestionProgress(
            output_dir, signature, task_ids, flush_every=args.flush_every,
            overwrite=overwrite,
        ) as progress:
            completed_at_start = progress.completed
            seen = progress.committed_questions()
            next_submit = progress.completed
            active = {}
            pending = {}
            progressbar = tqdm(
                total=len(scenes), initial=progress.completed,
                desc="V4 Question naturalization", unit="question",
                dynamic_ncols=True, disable=args.no_progress,
            )
            try:
                with ThreadPoolExecutor(
                    max_workers=args.question_workers,
                    thread_name_prefix="v4-question",
                ) as pool:
                    def fill():
                        nonlocal next_submit
                        while (
                            (len(active) + len(pending))
                            < max(1, 2 * args.question_workers)
                            and next_submit < len(scenes)
                        ):
                            index = next_submit
                            future = pool.submit(
                                _attempt, scenes[index], author=author,
                                critic=critic, max_revisions=args.max_revisions,
                            )
                            active[future] = index
                            next_submit += 1

                    fill()
                    while active:
                        ready, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                        for future in ready:
                            index = active.pop(future)
                            pending[index] = future.result()
                        while progress.completed in pending:
                            index = progress.completed
                            question, meta = pending.pop(index)
                            scene = _complete(scenes[index], question, meta, seen)
                            progress.append(index, scene)
                            progressbar.update(1)
                            progressbar.set_postfix(
                                accepted=progress.state["accepted"],
                                fallback=progress.state["fallback"],
                                revised=progress.state["accepted_after_revision"],
                                refresh=False,
                            )
                        fill()
            finally:
                progressbar.close()
            progress.flush()
            stats = progress.stats()
            stats["max_revisions"] = args.max_revisions
            stats["workers"] = args.question_workers
            stats["resumed_records"] = completed_at_start
            stats["author_model"] = args.question_model
            stats["critic_model"] = (
                args.question_critic_model or args.question_model
            )
            # This is exact even when all records were previously completed.
            unique_questions = len(progress.committed_questions())
            return stats, unique_questions
