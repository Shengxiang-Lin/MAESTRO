"""Lazy, evidence-replay-verified YAGO subtasks.

MMQA questions are ingested independently with the weaker
'dataset_labeled' status; no external modality inference is fabricated.
"""
from __future__ import annotations

from envir.schema import Subtask, digest, display
from envir.store import EvidenceStore


class SubtaskFactory:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def yago(self, subject: str, predicate: str) -> dict | None:
        task_id = "yago-task-" + digest(subject, predicate)
        cached = self.store.task(task_id)
        if cached is not None:
            return cached
        rows = self.store.get_facts(subject, predicate)
        # No arbitrary choice among multiple facts: ambiguous output is
        # excluded until a set-valued executor is implemented.
        outputs = {(r["object"], r["object_kind"]) for r in rows}
        if len(outputs) != 1:
            return None
        row = rows[0]
        sid = row["source_id"]
        source = self.store.source(sid) if sid else None
        obj, kind = next(iter(outputs))
        value = display(obj) if kind == "entity" else obj
        claim = f"{display(subject)} — {display(predicate)}: {value}"
        if source is None or claim not in source["body"]:
            return None
        # Search knows NOTHING about the expected source or answer. The
        # target must be discoverable from the public goal's subject terms.
        query = f"{display(subject)} {display(predicate)}"
        hits = self.store.search(query, limit=30)
        if sid not in {h["source_id"] for h in hits}:
            return None
        goal = f"Find the {display(predicate)} of {display(subject)}."
        task = Subtask(
            subtask_id=task_id,
            template_id=f"yago:{predicate}:{kind}",
            kind="search_relation" if kind == "entity" else "search_attribute",
            goal=goal,
            inputs={"subject_id": subject, "subject_name": display(subject)},
            output_type="YAGOEntity" if kind == "entity" else "Value",
            answer={"value": value, "type": kind,
                    "entity_id": obj if kind == "entity" else None},
            evidence_ids=(sid,),
            tool_events=(
                {"action": "LocalSearch", "query": query,
                 "returned_ids": [h["source_id"] for h in hits]},
                {"action": "OpenPage", "source_id": sid,
                 "observed_claim": claim},
            ),
            summary=f"{display(subject)} has {display(predicate)}: {value}.",
            summary_trace=(
                f"Searched the public index for '{query}'.",
                f"Opened {source['title']} and observed: {claim}",
            ),
            verification="source_replay_verified",
            provenance={"dataset": "YAGO-4.6", "fact_ids": sorted(
                r["fact_id"] for r in rows),
                "replay_policy": "observed_fact_and_public_search_v1"},
        )
        # Normalize tuple fields through JSON once so a newly generated
        # row and a subsequently cached row have identical public contracts.
        import json
        item = json.loads(json.dumps(task.to_dict(), ensure_ascii=False))
        self.store.cache_subtask(
            item, subject_key=subject,
            answer_key=obj if kind == "entity" else value,
        )
        return item

    def replay(self, task: dict) -> bool:
        """Independent check from public search/doc and the frozen YAGO fact."""
        if task.get("kind") not in ("search_relation", "search_attribute"):
            return False
        subject = task["inputs"]["subject_id"]
        predicate = task["template_id"].split(":", 1)[1].rsplit(":", 1)[0]
        rows = self.store.get_facts(subject, predicate)
        outputs = {(r["object"], r["object_kind"]) for r in rows}
        if len(outputs) != 1:
            return False
        raw, kind = next(iter(outputs))
        answer = task["answer"]
        if answer["value"] != (display(raw) if kind == "entity" else raw):
            return False
        if answer["entity_id"] != (raw if kind == "entity" else None):
            return False
        if len(task["evidence_ids"]) != 1 or not rows:
            return False
        sid = task["evidence_ids"][0]
        if any(row["source_id"] != sid for row in rows):
            return False
        source = self.store.source(sid)
        if source is None:
            return False
        claim = f"{display(subject)} — {display(predicate)}: {answer['value']}"
        if claim not in source["body"]:
            return False
        query = f"{display(subject)} {display(predicate)}"
        return sid in {
            r["source_id"] for r in self.store.search(query, limit=30)
        }


def unique_entity_labels(store: EvidenceStore, *, max_subjects: int = 100000
                         ) -> dict[str, str]:
    """Only return normalized labels mapping to exactly one YAGO entity ID."""
    from envir.schema import normalize_label
    lookup = {}
    duplicates = set()
    for subject in store.subjects(max_subjects):
        key = normalize_label(display(subject))
        if not key:
            continue
        if key in lookup and lookup[key] != subject:
            duplicates.add(key)
        else:
            lookup[key] = subject
    return {name: subject for name, subject in lookup.items()
            if name not in duplicates}
