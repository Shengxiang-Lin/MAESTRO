"""Extract MMQA evidence/goal motifs without requiring the original media files.

The dataset grounds the *patterns*, not the truth of our generated scenarios.
All generated subjects, events, readings and trace observations are synthetic.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from environment.schema import digest
from environment.sources.mmqa import _contexts, _modality, _rows


DOMAIN_CUES = {
    "logistics": ("vehicle", "truck", "transport", "ship", "road", "station", "airport", "train"),
    "culture": ("film", "movie", "painting", "artist", "museum", "album", "music", "book"),
    "sports": ("team", "sport", "player", "league", "football", "match", "tennis", "olympic"),
    "education": ("university", "school", "student", "degree", "teacher", "college", "campus"),
    "science": ("research", "laboratory", "science", "discovery", "device", "experiment", "satellite"),
    "commerce": ("company", "business", "product", "brand", "store", "sales", "manufacturer"),
    "environment": ("bird", "animal", "river", "plant", "forest", "species", "mountain", "lake"),
    "civic": ("city", "country", "government", "building", "street", "location", "district"),
}
_MODALITY_RE = re.compile(r"(image|table|text|video|audio)", re.I)


def classify_domain(text: str) -> str:
    lower = " " + text.lower() + " "
    scores = {
        category: sum(bool(re.search(r"\b" + re.escape(cue) + r"\b", lower))
                      for cue in cues)
        for category, cues in DOMAIN_CUES.items()
    }
    best = max(scores, key=lambda category: scores[category])
    return best if scores[best] else "general"


def _text(value) -> str:
    if isinstance(value, str):
        return " ".join(value.split())[:400]
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text(v) for v in value[:5])[:400]
    if isinstance(value, dict):
        return " ".join(_text(v) for v in list(value.values())[:5])[:400]
    return ""


def infer_pattern(type_text: str, question: str) -> str:
    combined = (type_text + " " + question).lower()
    if "intersect" in combined or "both" in combined or "common" in combined:
        return "intersection"
    if "compare" in combined or "between" in combined or "difference" in combined:
        return "comparison"
    if "compose" in combined or "then" in combined:
        return "composition"
    if "count" in combined or "how many" in combined:
        return "aggregation"
    return "lookup"


def extract_motif(record: dict, index: int) -> dict | None:
    question = _text(record.get("question"))
    if not question:
        return None
    meta = record.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    type_text = _text(meta.get("type") or record.get("type"))
    contexts = _contexts(record)
    modalities = []
    for ctx in contexts:
        mod = _modality(
            ctx.get("doc_part") or ctx.get("part") or ctx.get("modality")
            or ctx.get("source_type") or ctx.get("type")
        )
        if mod not in modalities:
            modalities.append(mod)
    # The task's formal MMQA type encodes modality roles even when the
    # source document files are absent or contextual metadata is incomplete.
    for token in _MODALITY_RE.findall(type_text):
        mod = token.lower()
        if mod not in modalities:
            modalities.append(mod)
    if not modalities:
        modalities = ["text"]
    entity_hint = _text(
        record.get("wiki_entities_in_question")
        or meta.get("wiki_entities_in_question") or []
    )
    pseudolang = _text(
        record.get("pseudo_language_question")
        or meta.get("pseudo_language_question")
    )
    intermediate = _text(
        record.get("intermediate_answers")
        or meta.get("intermediate_answers") or []
    )
    basis = [record.get("qid") or record.get("id") or index, question]
    return {
        "motif_id": "mmqa-motif-" + digest(basis),
        "source": "MultiModalQA",
        "record_id": str(record.get("qid") or record.get("id") or index),
        "question_hint": question,
        "pseudo_language_question": pseudolang,
        "question_type": type_text,
        "intermediate_pattern": intermediate[:250],
        "entity_topic_hints": entity_hint[:250],
        "source_modalities": modalities,
        "supporting_source_count": len(contexts),
        "logical_pattern": infer_pattern(type_text, question),
        "domain_hint": classify_domain(" ".join(
            (question, pseudolang, entity_hint, type_text)
        )),
        "policy": "motif_only_no_original_media_or_answer_reused",
    }


def load_motifs(path: str | Path, *, limit: int = 3000) -> tuple[list[dict], dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    motifs = []
    counters = Counter()
    for index, record in enumerate(_rows(Path(path))):
        if index >= limit:
            break
        counters["scanned"] += 1
        motif = extract_motif(record, index)
        if motif is None:
            counters["missing_question"] += 1
            continue
        motifs.append(motif)
        counters["valid"] += 1
        counters["pattern:" + motif["logical_pattern"]] += 1
        counters["domain:" + motif["domain_hint"]] += 1
        for mod in motif["source_modalities"]:
            counters["modality:" + mod] += 1
    return motifs, dict(counters)
