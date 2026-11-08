"""Optional QUESTION-STYLE priors from local official OmniGAIA metadata.

Only question wording/shape and the category/difficulty labels are used.
No benchmark answer, annotated solution, identity, image, audio, video or
source evidence is imported into generated synthetic truth.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _records(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "questions", "test"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError("OmniGAIA JSON must be a list or a dictionary with list data")
        yield from payload


def detect_style(question: str) -> str:
    q = " ".join(question.lower().split())
    if not q:
        raise ValueError("empty question")
    if re.search(r"\b(i'm|i am|as part of|you are|imagine)\b", q):
        return "contextual_investigation"
    if re.search(r"\b(calculate|rounded|nearest|number of months|number of years|distance|first|then)\b", q):
        return "explicit_result_constraints"
    if re.search(r"\b(compare|two (?:videos|images|sources)|both|difference|reconcile)\b", q):
        return "cross_source_explanation"
    return "concise_compositional"


def load_omni_styles(path: str | Path, *, limit: int = 360):
    path = Path(path)
    if limit < 1:
        raise ValueError("OmniGAIA limit must be positive")
    result, counts = [], Counter()
    for i, row in enumerate(_records(path)):
        if i >= limit:
            break
        if not isinstance(row, dict):
            continue
        q = str(row.get("question") or "").strip()
        if not q:
            continue
        style = detect_style(q)
        counts[style] += 1
        result.append({
            "record_id": str(row.get("id", i)),
            "style": style,
            "difficulty": row.get("level"),
            "original_length": len(q),
            "policy": "question_style_only_no_original_content_reused",
        })
    if not result:
        raise ValueError("no OmniGAIA question style records")
    return result, {
        "records": len(result),
        "question_style_counts": dict(counts),
        "policy": "styles_only_no_media_answers_or_annotated_solutions",
    }
