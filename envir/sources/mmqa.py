"""MultiModalQA: official QA records and actual context references.

A dataset answer is labeled, not automatically certified by replaying raw
vision. Missing image pixels remain unavailable; a caption is a PROXY.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from envir.schema import Subtask, digest
from envir.store import EvidenceStore

_BANK_FILES = {
    "text": ("texts.jsonl.gz", "texts.jsonl", "MMQA_texts.jsonl.gz", "MMQA_texts.jsonl"),
    "table": ("tables.jsonl.gz", "tables.jsonl", "MMQA_tables.jsonl.gz", "MMQA_tables.jsonl"),
    "image": ("images.jsonl.gz", "images.jsonl", "MMQA_images.jsonl.gz", "MMQA_images.jsonl"),
}


def _rows(path: Path) -> Iterator[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    yield data


def _contexts(record: dict) -> list[dict]:
    for owner in (record.get("metadata", {}), record):
        if not isinstance(owner, dict):
            continue
        for key in ("supporting_context", "supporting_contexts", "context"):
            values = owner.get(key)
            if isinstance(values, list) and values and isinstance(values[0], dict):
                return values
    return []


def _modality(value: object) -> str:
    t = str(value or "text").strip().lower().replace("-", "_")
    if t in {"image", "img", "photo", "image_report", "visual"}:
        return "image"
    if t in {"table", "tabular"}:
        return "table"
    return "text"


def _answer(value: object) -> str | None:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value).strip() or None
    if isinstance(value, dict):
        for key in ("answer", "value", "text"):
            if key in value:
                return _answer(value[key])
    if isinstance(value, list) and len(value) == 1:
        return _answer(value[0])
    return None


def _bank_indexes(parent: Path, needed: dict[str, set[str]]) -> dict[tuple[str, str], dict]:
    index = {}
    for mod, names in _BANK_FILES.items():
        path = next((parent / name for name in names if (parent / name).exists()), None)
        if not path:
            continue
        wanted = needed[mod]
        for row in _rows(path):
            key = row.get("id") or row.get("doc_id") or row.get("source_id")
            if key is not None and str(key) in wanted:
                index[(mod, str(key))] = row
                if len([x for x in index if x[0] == mod]) >= len(wanted):
                    break
    return index


def _body(row: dict, modality: str) -> str:
    if modality == "text":
        return str(row.get("text") or row.get("passage") or row.get("content") or "")
    if modality == "image":
        return str(row.get("caption") or row.get("image_caption") or
                   row.get("description") or row.get("alt_text") or "")
    table = row.get("table") or {}
    if isinstance(table, dict):
        values = [str(table.get("table_name", ""))]
        values += [str(x.get("column_name", x.get("text", "")))
                   if isinstance(x, dict) else str(x)
                   for x in table.get("header", [])]
        for r in (table.get("table_rows") or table.get("rows") or [])[:30]:
            if isinstance(r, list):
                values.append(" | ".join(str(c.get("text", c.get("value", "")))
                                         if isinstance(c, dict) else str(c)
                                         for c in r))
        return "\n".join(values)
    return ""


def ingest_mmqa(store: EvidenceStore, path: str | Path, *,
                max_records: int = 2000) -> dict:
    path = Path(path)
    records = []
    needed: dict[str, set[str]] = defaultdict(set)
    for ix, record in enumerate(_rows(path)):
        if ix >= max_records:
            break
        question = str(record.get("question") or "").strip()
        answer = _answer(record.get("answer"))
        context = _contexts(record)
        if not question or not answer or not context:
            continue
        supports = []
        for item in context:
            doc_id = item.get("doc_id") or item.get("source_id") or item.get("id")
            mod = _modality(item.get("doc_part") or item.get("part") or
                            item.get("modality") or item.get("source_type"))
            if doc_id is not None:
                supports.append((mod, str(doc_id), item))
                needed[mod].add(str(doc_id))
        if supports:
            records.append((ix, record, question, answer, supports))
    index = _bank_indexes(path.parent, needed)
    counters = Counter()
    for ix, record, question, answer, supports in records:
        sources = []
        media = []
        for mod, doc_id, stub in supports:
            raw = dict(index.get((mod, doc_id), {}))
            raw.update(stub)
            body = _body(raw, mod)
            media_path = None
            if mod == "image":
                for field in ("image_path", "path", "file", "filename"):
                    value = raw.get(field)
                    if isinstance(value, str):
                        candidate = (path.parent / value).resolve()
                        if candidate.is_file() and candidate.is_relative_to(path.parent.resolve()):
                            media_path = str(candidate)
                            break
            if not body.strip() and media_path is None:
                counters["unavailable_context"] += 1
                continue
            sid = "mmqa-source-" + digest(mod, doc_id)
            title = str(raw.get("title") or raw.get("name") or f"{mod} {doc_id}")
            store.add_source(
                sid, "MultiModalQA", mod, title, body[:10000],
                asset_path=media_path,
                metadata={"original_doc_id": doc_id,
                          "media_status": "file_available_unverified" if media_path
                          else "caption_proxy" if mod == "image" else "text_accessible"},
            )
            sources.append(sid)
            if mod == "image":
                media.append("file_available_unverified" if media_path else "caption_proxy")
        # Only label QA supported by sources that were actually indexed.
        if not sources:
            counters["no_accessible_evidence"] += 1
            continue
        task_id = "mmqa-" + digest(record.get("qid") or record.get("id") or ix, question)
        task = Subtask(
            subtask_id=task_id,
            template_id="mmqa:" + str((record.get("metadata") or {}).get("type") or "QA"),
            kind="mmqa_qa",
            goal=question,
            inputs={"source_ids": sources},
            output_type="DatasetLabeledAnswer",
            answer={"value": answer, "type": "dataset_label", "entity_id": None},
            evidence_ids=tuple(sources),
            tool_events=(),
            summary=f"MultiModalQA records the answer as: {answer}",
            summary_trace=(
                "Referenced the official MultiModalQA QA annotation and its supporting context IDs.",
                "No independent raw-image perception or answer replay is claimed.",
            ),
            verification="dataset_labeled",
            provenance={
                "dataset": "MultiModalQA",
                "record_id": str(record.get("qid") or record.get("id") or ix),
                "question_type": str((record.get("metadata") or {}).get("type") or ""),
                "media_status": media,
            },
        )
        store.cache_subtask(task.to_dict(), answer_key=answer.casefold())
        counters["labeled_subtasks"] += 1
        if media:
            counters["containing_images"] += 1
    store.db.commit()
    return {"dataset": "MultiModalQA", "records_sampled": max_records,
            "context_documents_found": len(index), **dict(counters)}
