"""Shared ordered, recoverable streaming for envir LLM generator stages.

The worker pool may complete out of order, but ONE thread appends outcomes in
frozen input order. Each accepted/rejected record is paired with an audit
record. A committed checkpoint stores the exact byte offsets and hashes.
On resume, uncommitted tails are truncated and the original task order
and model/config/code signature must match before any new calls are made.

Ported to the independent envir package from the tested
environment.question_generation.streaming design.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

from tqdm.auto import tqdm

SCHEMA = "maestro.envir.ordered_stream.v1"


class StreamingError(RuntimeError):
    pass


def stable_signature(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _durable_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2).encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _line(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


@contextmanager
def output_lock(folder: Path, prefix: str):
    """Avoid concurrent jobs writing to the same checkpoint/output directory."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / (prefix + ".lock")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StreamingError("Another process is writing " + str(folder)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class OrderedJournal:
    """Single-writer, source-signed, crash-recoverable record journal.

    Accepted/rejected are provisional per-item outcomes, NOT necessarily
    published scenes. Each accepted/rejected line has {task_id, payload}.
    audit.jsonl has one entry for EVERY completed item.
    """

    def __init__(self, directory: Path, prefix: str,
                 signature: str, task_ids: list[str], *,
                 flush_every: int = 1, restart: bool = False):
        if type(flush_every) is not int or flush_every < 1:
            raise ValueError("flush_every must be a positive integer")
        if len(set(task_ids)) != len(task_ids):
            raise StreamingError("duplicate_task_ids_in_ordered_stream")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix, self.signature = prefix, signature
        self.task_ids = list(task_ids)
        self.flush_every = flush_every
        self.paths = {
            "accepted": self.directory / (prefix + "_accepted.jsonl"),
            "rejected": self.directory / (prefix + "_rejected.jsonl"),
            "audit": self.directory / (prefix + "_audit.jsonl"),
        }
        self.state_path = self.directory / (prefix + "_progress.json")
        self.handles: dict = {}
        self.hashes: dict = {}
        self.offsets: dict = {}
        self._uncommitted = 0

        if restart:
            for path in (*self.paths.values(), self.state_path):
                path.unlink(missing_ok=True)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (self.state.get("schema") != SCHEMA
                    or self.state.get("signature") != signature
                    or self.state.get("task_ids") != task_ids):
                raise StreamingError(
                    "Checkpoint signature, task order, or inputs changed; "
                    "choose a new output directory or explicitly overwrite"
                )
            if not 0 <= self.completed <= len(task_ids):
                raise StreamingError("invalid_checkpoint_completed")
            if self.accepted + self.rejected != self.completed:
                raise StreamingError("inconsistent_checkpoint_counts")
            for key, path in self.paths.items():
                end = self.state.get("offsets", {}).get(key)
                recorded_hash = self.state.get("sha256", {}).get(key)
                if type(end) is not int or end < 0 or not path.is_file():
                    raise StreamingError("missing_committed_stream:" + str(path))
                if path.stat().st_size < end:
                    raise StreamingError("truncated_committed_stream:" + str(path))
                digest = hashlib.sha256()
                with path.open("rb") as file:
                    remaining = end
                    while remaining:
                        piece = file.read(min(remaining, 1 << 20))
                        if not piece:
                            raise StreamingError("committed_stream_short_read")
                        digest.update(piece)
                        remaining -= len(piece)
                if digest.hexdigest() != recorded_hash:
                    raise StreamingError("committed_stream_hash_mismatch:" + str(path))
                self.offsets[key] = end
                self.hashes[key] = digest
            self._validate_audit()
            # Ignore partial writes following the last durable checkpoint.
            for key, path in self.paths.items():
                with path.open("r+b") as file:
                    file.truncate(self.offsets[key])
        else:
            if any(p.exists() for p in (*self.paths.values(), self.state_path)):
                raise StreamingError(
                    "Existing stream files without a matching checkpoint; "
                    "choose a new folder or explicitly overwrite"
                )
            self.state = {
                "schema": SCHEMA, "signature": signature,
                "task_ids": task_ids,
                "completed": 0, "accepted": 0, "rejected": 0,
                "offsets": {key: 0 for key in self.paths},
                "sha256": {
                    key: hashlib.sha256(b"").hexdigest()
                    for key in self.paths
                },
            }
            for key, path in self.paths.items():
                path.touch()
                self.offsets[key] = 0
                self.hashes[key] = hashlib.sha256()
            _durable_json(self.state_path, self.state)
        self.handles = {key: path.open("ab") for key, path in self.paths.items()}

    @property
    def completed(self) -> int:
        return int(self.state["completed"])

    @property
    def accepted(self) -> int:
        return int(self.state["accepted"])

    @property
    def rejected(self) -> int:
        return int(self.state["rejected"])

    def _read(self, key: str) -> list[dict]:
        with self.paths[key].open("rb") as file:
            data = file.read(self.offsets[key])
        try:
            return [json.loads(row) for row in data.splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StreamingError("invalid_committed_JSONL:" + key) from exc

    def _validate_audit(self) -> None:
        audit = self._read("audit")
        if len(audit) != self.completed:
            raise StreamingError("committed_audit_length_mismatch")
        accepted = rejected = 0
        for index, record in enumerate(audit):
            if (record.get("index") != index
                    or record.get("task_id") != self.task_ids[index]
                    or type(record.get("accepted")) is not bool):
                raise StreamingError("committed_audit_order_mismatch")
            accepted += record["accepted"]
            rejected += not record["accepted"]
        if (accepted, rejected) != (self.accepted, self.rejected):
            raise StreamingError("committed_audit_counts_mismatch")
        for key, count in (("accepted", accepted), ("rejected", rejected)):
            if len(self._read(key)) != count:
                raise StreamingError("committed_outcome_counts_mismatch:" + key)

    def replay(self) -> list[tuple[bool, dict, dict]]:
        """Recover committed outcomes in exact original input order."""
        self.flush()
        accepted = iter(self._read("accepted"))
        rejected = iter(self._read("rejected"))
        replayed = []
        for entry in self._read("audit"):
            ok = entry["accepted"]
            item = next(accepted if ok else rejected)
            if item.get("task_id") != entry["task_id"]:
                raise StreamingError("outcome_does_not_match_audit")
            replayed.append((ok, item["payload"], entry.get("meta", {})))
        return replayed

    def append(self, index: int, ok: bool, payload: dict,
               meta: dict | None = None) -> None:
        if index != self.completed:
            raise StreamingError("out_of_order_append")
        ident = self.task_ids[index]
        rows = {
            "accepted" if ok else "rejected": {
                "task_id": ident, "payload": payload,
            },
            "audit": {
                "index": index, "task_id": ident,
                "accepted": bool(ok), "meta": meta or {},
            },
        }
        for key, obj in rows.items():
            data = _line(obj)
            self.handles[key].write(data)
            self.offsets[key] += len(data)
            self.hashes[key].update(data)
        self.state["completed"] = index + 1
        self.state["accepted" if ok else "rejected"] += 1
        self._uncommitted += 1
        if self._uncommitted >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._uncommitted:
            return
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        self.state["offsets"] = dict(self.offsets)
        self.state["sha256"] = {
            key: digest.hexdigest() for key, digest in self.hashes.items()
        }
        _durable_json(self.state_path, self.state)
        self._uncommitted = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def run_ordered(items: list, job, classify, *, journal: OrderedJournal,
                workers: int, desc: str = "Processing", unit: str = "task",
                no_progress: bool = False, on_commit=None,
                on_exception=None) -> None:
    """Bounded parallel execution; completed-future progress; ordered writes.

    All writes, callbacks, and tqdm updates run on the caller's thread.
    A slow earlier item can delay ordered disk publication, but not the
    visible completed/accepted/rejected counters.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    if len(items) != len(journal.task_ids):
        raise StreamingError("items_and_frozen_journal_length_mismatch")
    next_submit = journal.completed
    futures = {}
    pending = {}
    done_accepted = journal.accepted
    done_rejected = journal.rejected
    with tqdm(total=len(items), initial=journal.completed,
              desc=desc, unit=unit, disable=no_progress,
              dynamic_ncols=True) as bar:
        def postfix():
            bar.set_postfix({
                "accepted": done_accepted,
                "rejected": done_rejected,
                "written": journal.completed,
            }, refresh=False)
        postfix()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def submit():
                nonlocal next_submit
                while next_submit < len(items) and len(futures) < 2 * workers:
                    index = next_submit
                    futures[pool.submit(job, items[index])] = index
                    next_submit += 1
            submit()
            while futures:
                ready, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in ready:
                    index = futures.pop(future)
                    try:
                        ok, payload, meta = classify(future.result())
                    except Exception as exc:
                        if on_exception is None:
                            raise
                        ok, payload, meta = on_exception(index, exc)
                    pending[index] = (bool(ok), payload, meta)
                    done_accepted += bool(ok)
                    done_rejected += not bool(ok)
                    bar.update(1)
                while journal.completed in pending:
                    index = journal.completed
                    good, payload, meta = pending.pop(index)
                    journal.append(index, good, payload, meta)
                    if on_commit is not None:
                        on_commit(index, good, payload, meta)
                postfix()
                submit()
        journal.flush()
        postfix()
