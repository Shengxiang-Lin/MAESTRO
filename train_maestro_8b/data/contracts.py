"""Schema for evidence-window risk pretraining; labels are never part of the prompt."""
from __future__ import annotations
import json
from pathlib import Path

VERSION = "maestro.window_risk.v1"

def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{number}: invalid JSON") from e
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: not an object")
                yield row

def validate(row):
    if row.get("schema") != VERSION:
        raise ValueError("wrong window-risk schema")
    text = row.get("prompt")
    reports = row.get("reports")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing Main prompt")
    if not isinstance(reports, list) or not reports:
        raise ValueError("missing reports")
    before = 0
    seen = set()
    for r in reports:
        if not isinstance(r, dict):
            raise ValueError("invalid report")
        rid = r.get("report_id")
        start, end = r.get("start"), r.get("end")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("invalid/duplicate report ID")
        seen.add(rid)
        if (type(start) is not int or type(end) is not int
                or not before <= start < end <= len(text)
                or not text[start:end].strip()):
            raise ValueError("invalid/overlapping report offset")
        before = end
        if type(r.get("risk")) is not int or r["risk"] not in (0, 1):
            raise ValueError("risk must be zero or one")
        if type(r.get("mask")) is not int or r["mask"] not in (0, 1):
            raise ValueError("mask must be zero or one")
    if not any(r["mask"] for r in reports):
        raise ValueError("no supervised reports")
    return row

# Token windowing and byte/character span alignment.
def windows(start: int, end: int, size: int = 128, stride: int = 96):
    if not (0 <= start < end and size >= 1 and 1 <= stride <= size):
        raise ValueError("invalid token window")
    length = end - start
    if length <= size:
        return [(start, end)]
    offsets = list(range(0, length - size + 1, stride))
    if offsets[-1] != length - size:
        offsets.append(length - size)
    return [(start + o, start + o + size) for o in offsets]

def span_to_tokens(offsets, start: int, end: int):
    ids = [i for i, (left, right) in enumerate(offsets)
           if right > left and right > start and left < end]
    if not ids:
        raise ValueError("no token overlaps report span")
    return ids[0], ids[-1] + 1
