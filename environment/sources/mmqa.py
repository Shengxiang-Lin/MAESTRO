"""Small, read-only MultiModalQA annotation reader.

Parses the original QA and its supporting source handles. No answer labeling,
fake image decoding, legacy SQLite source bank, or native media execution.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def _rows(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _contexts(record: dict) -> list[dict]:
    for owner in (record.get("metadata", {}), record):
        if not isinstance(owner, dict):
            continue
        for key in ("supporting_context", "supporting_contexts", "context"):
            value = owner.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    return []


def _modality(value: object) -> str:
    name = str(value or "text").strip().lower().replace("-", "_")
    if name in {"image", "img", "photo", "image_report", "visual"}:
        return "image"
    if name in {"table", "tabular"}:
        return "table"
    return "text"
