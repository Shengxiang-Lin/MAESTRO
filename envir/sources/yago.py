"""YAGO-4.6: original observed triples -> searchable, replayable pages.

Uses only the legacy line-based Turtle reader, not the old task pipeline.
No LLM-synthesized facts or predicate-profile substitutions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from envir.schema import SubtaskTemplate, display
from envir.store import EvidenceStore

# Reuse the already exercised low-level YAGO streaming reader only.
from environment.task_construction.data_sources.yago import (
    METADATA_PREDICATES, TYPE_PREDICATES, iter_turtle_triples,
)

_LITERAL = re.compile(r'^"((?:[^"\\]|\\.)*)"|^\'((?:[^\'\\]|\\.)*)\'')


def literal(raw: str) -> str:
    match = _LITERAL.match(raw.strip())
    if match:
        value = (match.group(1) if match.group(1) is not None else match.group(2))
        return value.replace('\\"', '"').replace("\\n", " ").strip()
    return raw.strip()


def ingest_yago(store: EvidenceStore, path: str | Path, *,
                max_triples: int = 300000, max_subjects: int = 100000) -> dict:
    seen = added = 0
    for triple in iter_turtle_triples(path):
        seen += 1
        if seen > max_triples:
            break
        if triple.predicate in TYPE_PREDICATES | METADATA_PREDICATES:
            continue
        subject = triple.subject
        if not subject or not triple.predicate:
            continue
        obj = literal(triple.object) if triple.object_is_literal else triple.object
        if len(obj) > 400 or not obj:
            continue
        if store.add_fact(subject, triple.predicate, obj,
                          "literal" if triple.object_is_literal else "entity"):
            added += 1
        if added % 5000 == 0 and added:
            store.db.commit()
    pages = store.create_yago_pages(max_subjects=max_subjects)
    templates = []
    for row in store.db.execute(
        "SELECT predicate,object_kind, count(*) AS n FROM facts "
        "GROUP BY predicate,object_kind ORDER BY n DESC"
    ):
        template = SubtaskTemplate(
            template_id="yago:" + row["predicate"] + ":" + row["object_kind"],
            kind="search_relation" if row["object_kind"] == "entity"
                 else "search_attribute",
            input_type="YAGOEntity",
            output_type="YAGOEntity" if row["object_kind"] == "entity" else "Value",
            goal_pattern="Find the " + display(row["predicate"]) + " of {subject}",
            predicate=row["predicate"],
            dataset="YAGO-4.6",
        )
        templates.append(template.to_dict())
    store.db.commit()
    return {"dataset": "YAGO-4.6", "triples_scanned": min(seen, max_triples),
            "facts_indexed": added, "searchable_subject_pages": pages,
            "templates": templates}
