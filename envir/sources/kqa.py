"""KQA-Pro: abstract logical motifs, NOT invented concrete questions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from envir.schema import digest
from envir.store import EvidenceStore


def ingest_kqa(store: EvidenceStore, path: str | Path, *,
               max_records: int = 10000) -> dict:
    # KQA-Pro stores train/val as JSON arrays. Keep only compact structure,
    # never use a KQA answer as if it were grounded in YAGO.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("KQA-Pro expects a JSON list")
    counts = Counter()
    recognized = 0
    for record in data[:max_records]:
        program = record.get("program") if isinstance(record, dict) else None
        if not isinstance(program, list) or not program:
            continue
        operators = [str(node.get("function", "")) for node in program]
        if any(op in {"And", "Or", "SelectBetween"} for op in operators):
            pattern = "parallel"
        elif operators.count("Relate") >= 1:
            pattern = "chain"
        elif "QueryAttr" in operators or "What" in operators:
            pattern = "lookup"
        else:
            pattern = "other"
        counts[pattern] += 1
        signature = [
            [node.get("function"), node.get("dependencies", [])]
            for node in program
        ]
        store.add_skeleton({
            "skeleton_id": "kqa-" + digest(signature),
            "pattern": pattern,
            "functions": operators,
            "edges": [node.get("dependencies", []) for node in program],
            "example_question": str(record.get("question") or "")[:300],
            "source": "KQA-Pro",
        })
        recognized += 1
    store.db.commit()
    return {"dataset": "KQA-Pro", "records_scanned": min(len(data), max_records),
            "records_with_program": recognized, "pattern_counts": dict(counts),
            "unique_skeletons": len(store.skeletons())}
