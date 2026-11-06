"""Ordered streaming question output with crash-safe committed progress.

The accepted/rejected JSONLs and private review audit are append-only. A checkpoint records their
committed byte lengths, SHA-256 digests and counts. On resume, uncommitted tails
are discarded, while every committed question and rejection is reused. Model
settings and frozen inputs are validated by the caller's signature.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "maestro.question_generation.progress.v1"
_OUTPUTS = {
    "realized": "realized_capability_worlds.jsonl",
    "rejected": "question_rejected.jsonl",
    "audit": "question_review_audit.jsonl",
}


class QuestionCheckpointError(RuntimeError):
    """A prior run cannot safely be resumed."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _persist(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # Ensure the rename is durable as well as the data.
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def question_output_lock(directory: Path):
    """Prevent overlapping generators from mutating the same outputs."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "question_generation.lock"
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QuestionCheckpointError(
                f"another question-generation process owns {path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class QuestionProgress:
    """Single-writer journal and recoverable, ordered JSONL sidecars."""

    def __init__(
        self, directory: Path, signature: str, task_ids: list[str],
        *, flush_every: int = 5, restart: bool = False,
    ) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = {key: self.directory / name for key, name in _OUTPUTS.items()}
        self.state_path = self.directory / "question_progress_state.json"
        self.summary_path = self.directory / "question_generation_summary.json"
        self.signature = signature
        self.task_ids = task_ids
        self.flush_every = flush_every
        self.state: dict[str, Any]
        self.handles: dict[str, Any] = {}
        self.digests: dict[str, Any] = {}
        self.offsets: dict[str, int] = {}

        if restart:
            for path in (*self.paths.values(), self.state_path, self.summary_path):
                path.unlink(missing_ok=True)

        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("schema") != SCHEMA or self.state.get("signature") != signature:
                raise QuestionCheckpointError(
                    "question-generation checkpoint has different frozen inputs, "
                    "code or model settings; use --restart-questions to start over"
                )
            completed = self.completed
            if not 0 <= completed <= len(task_ids):
                raise QuestionCheckpointError("checkpoint has invalid completed count")
            if self.accepted + self.rejected != completed:
                raise QuestionCheckpointError("checkpoint counts are inconsistent")
            expected_offsets = self.state.get("offsets", {})
            expected_digests = self.state.get("sha256", {})
            for key, path in self.paths.items():
                committed = expected_offsets.get(key)
                if not isinstance(committed, int) or committed < 0 or not path.is_file():
                    raise QuestionCheckpointError(f"missing committed output: {path}")
                if path.stat().st_size < committed:
                    raise QuestionCheckpointError(f"committed output was truncated: {path}")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    remaining = committed
                    while remaining:
                        chunk = handle.read(min(1 << 20, remaining))
                        if not chunk:
                            raise QuestionCheckpointError(f"truncated committed data: {path}")
                        digest.update(chunk)
                        remaining -= len(chunk)
                if digest.hexdigest() != expected_digests.get(key):
                    raise QuestionCheckpointError(f"committed data hash mismatch: {path}")
                self.offsets[key] = committed
                self.digests[key] = digest
            self._verify_order()
            # Previous process may have written a tail but not committed state.
            for key, path in self.paths.items():
                with path.open("r+b") as handle:
                    handle.truncate(self.offsets[key])
        else:
            if self.summary_path.is_file() or any(p.exists() for p in self.paths.values()):
                raise QuestionCheckpointError(
                    "question output already exists without a checkpoint; "
                    "use --restart-questions to discard it explicitly"
                )
            self.state = {
                "schema": SCHEMA, "signature": signature,
                "completed": 0, "accepted": 0, "rejected": 0,
                "offsets": {key: 0 for key in self.paths},
                "sha256": {key: hashlib.sha256(b"").hexdigest() for key in self.paths},
            }
            for key, path in self.paths.items():
                path.touch()
                self.offsets[key] = 0
                self.digests[key] = hashlib.sha256()
            _persist(self.state_path, self.state)

        self.handles = {key: path.open("ab") for key, path in self.paths.items()}
        self._uncommitted = 0

    @property
    def completed(self) -> int:
        return int(self.state["completed"])

    @property
    def accepted(self) -> int:
        return int(self.state["accepted"])

    @property
    def rejected(self) -> int:
        return int(self.state["rejected"])

    def _verify_order(self) -> None:
        """Ensure the committed audit journal aligns with the input order."""
        with self.paths["audit"].open("rb") as handle:
            entries = handle.read(self.offsets["audit"]).splitlines()
        if len(entries) != self.completed:
            raise QuestionCheckpointError("committed audit record count mismatch")
        accepted = rejected = 0
        for index, entry in enumerate(entries):
            try:
                record = json.loads(entry)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QuestionCheckpointError("invalid committed audit JSONL") from exc
            if record.get("index") != index or record.get("task_id") != self.task_ids[index]:
                raise QuestionCheckpointError(
                    f"committed audit record {index} does not match frozen task order"
                )
            accepted += record.get("accepted") is True
            rejected += record.get("accepted") is False
        if (accepted, rejected) != (self.accepted, self.rejected):
            raise QuestionCheckpointError("committed audit counts do not match checkpoint")
        for key, count in (("realized", accepted), ("rejected", rejected)):
            with self.paths[key].open("rb") as handle:
                lines = handle.read(self.offsets[key]).splitlines()
            if len(lines) != count:
                raise QuestionCheckpointError(f"committed {key} record count mismatch")

    def append(self, index: int, ok: bool, row: dict[str, Any], audit: Any) -> None:
        if index != self.completed:
            raise QuestionCheckpointError(
                f"out-of-order write index={index}, expected={self.completed}"
            )
        if str(row.get("task_id", "")) != self.task_ids[index]:
            raise QuestionCheckpointError(f"task ID mismatch at input index {index}")
        key = "realized" if ok else "rejected"
        audit_row = {
            "index": index, "task_id": self.task_ids[index],
            "accepted": bool(ok), "audit": audit,
        }
        for name, payload in ((key, row), ("audit", audit_row)):
            data = _json_bytes(payload)
            self.handles[name].write(data)
            self.offsets[name] += len(data)
            self.digests[name].update(data)
        self.state["completed"] = index + 1
        self.state["accepted" if ok else "rejected"] += 1
        self._uncommitted += 1
        if self._uncommitted >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._uncommitted:
            return
        # fsync output data before advancing the committed offsets in state.
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        self.state["offsets"] = dict(self.offsets)
        self.state["sha256"] = {
            key: digest.hexdigest() for key, digest in self.digests.items()
        }
        _persist(self.state_path, self.state)
        self._uncommitted = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()

    def __enter__(self) -> "QuestionProgress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def signature_for_question_run(
    frozen: Mapping[str, Any], args: Any, task_ids: list[str],
    *, source_files: list[Path],
) -> str:
    """Hash all inputs and contract settings that affect question outcomes."""
    sources = {}
    for path in source_files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        sources[str(path)] = digest.hexdigest()
    payload = {
        "schema": SCHEMA, "frozen": dict(frozen), "task_ids": task_ids,
        "seed": args.seed, "max_records": args.max_records,
        "author_model": args._resolved_models["question_author"],
        "critic_model": args._resolved_models["question_critic"],
        "question_temperature": args.question_temperature,
        "critic_temperature": args.critic_temperature,
        "max_tokens": args.max_tokens,
        "critic_max_tokens": args.critic_max_tokens,
        "timeout": args.timeout,
        "max_revisions": args.max_revisions,
        "no_json_mode": args.no_json_mode,
        "code_sha256": sources,
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "QuestionCheckpointError", "QuestionProgress", "question_output_lock",
    "signature_for_question_run",
]
