"""Optional original MultiModalQA text, table and caption context lookup.

Reads *existing* MMQA context-bank files in the question dataset directory.
Never uses answer labels and never claims that caption/metadata is raw image
perception. Missing bank files are normal; callers fall back to explicitly
synthetic evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from envir.sources.mmqa import _rows

BANK_FILES = {
    "text": ("MMQA_texts.jsonl.gz", "MMQA_texts.jsonl", "texts.jsonl.gz", "texts.jsonl"),
    "table": ("MMQA_tables.jsonl.gz", "MMQA_tables.jsonl", "tables.jsonl.gz", "tables.jsonl"),
    "image": ("MMQA_images.jsonl.gz", "MMQA_images.jsonl", "images.jsonl.gz", "images.jsonl"),
}


def _body(record: dict, modality: str, limit: int) -> str:
    if modality == "text":
        value = record.get("text") or record.get("passage") or record.get("content") or ""
        return " ".join(str(value).split())[:limit]
    if modality == "image":
        # Official image *metadata* captions, never inferred pixels.
        value = (record.get("caption") or record.get("image_caption")
                 or record.get("description") or record.get("alt_text") or "")
        return " ".join(str(value).split())[:limit]
    table = record.get("table") or record
    if not isinstance(table, dict):
        return ""
    parts = [str(table.get("table_name") or record.get("title") or "")]
    headers = table.get("header") or table.get("headers") or []
    for item in headers[:32] if isinstance(headers, list) else []:
        parts.append(str(item.get("column_name") or item.get("text") or "")
                     if isinstance(item, dict) else str(item))
    rows = table.get("table_rows") or table.get("rows") or []
    for row in rows[:24] if isinstance(rows, list) else []:
        if isinstance(row, list):
            parts.append(" | ".join(
                str(cell.get("text") or cell.get("value") or "")
                if isinstance(cell, dict) else str(cell)
                for cell in row[:32]
            ))
        elif isinstance(row, dict):
            parts.append(" | ".join(str(x) for x in list(row.values())[:32]))
    return " ".join(" ".join(parts).split())[:limit]


def load_supporting_contexts(seeds: list[dict], directory: str | Path, *,
                             excerpt_chars: int = 1200) -> tuple[dict, dict]:
    """Return an index keyed by (modality, original doc_id).

    A context is accessible only if a supplied bank file has the exact
    supporting doc ID. Never match document titles by fuzzy strings.
    """
    if excerpt_chars < 100:
        raise ValueError("excerpt_chars must be >=100")
    root = Path(directory)
    needed: dict[str, set[str]] = defaultdict(set)
    for seed in seeds:
        for source in seed.get("supporting_sources", []):
            if source.get("doc_id") is not None and source["modality"] in BANK_FILES:
                needed[source["modality"]].add(str(source["doc_id"]))
    result = {}
    stats = Counter()
    present = {}
    for modality, filenames in BANK_FILES.items():
        bank = next((root / name for name in filenames if (root / name).is_file()), None)
        present[modality] = bank.name if bank else None
        if bank is None or not needed[modality]:
            continue
        for row in _rows(bank):
            raw_key = row.get("id") or row.get("doc_id") or row.get("source_id")
            if raw_key is None:
                continue
            key = str(raw_key)
            if key not in needed[modality] or (modality, key) in result:
                continue
            body = _body(row, modality, excerpt_chars)
            if not body.strip():
                stats["empty_bank_content"] += 1
                continue
            result[(modality, key)] = {
                "text": body,
                "provenance": ("original_mmqa_caption_proxy" if modality == "image"
                               else f"original_mmqa_{modality}_text"),
                "modality": modality,
                "original_doc_id": key,
            }
            stats["found:" + modality] += 1
            if stats["found:" + modality] >= len(needed[modality]):
                break
    stats["requested_original_handles"] = sum(map(len, needed.values()))
    stats["available_context_excerpts"] = len(result)
    stats["unavailable_context_handles"] = (
        stats["requested_original_handles"] - len(result))
    return result, {"policy": "original_source_text_or_caption_only_not_pixels",
                    "bank_files": present, **dict(stats)}
