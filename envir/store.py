"""SQLite evidence / fact index and lazy verified-subtask cache.

No answers are supplied to public search. Search reads only the indexed
source records; private fact and subtask tables are teacher-only.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from .schema import digest, display


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY, dataset TEXT NOT NULL,
                modality TEXT NOT NULL, title TEXT NOT NULL,
                body TEXT NOT NULL, asset_path TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                predicate TEXT NOT NULL, object TEXT NOT NULL,
                object_kind TEXT NOT NULL, source_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
            CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object);
            CREATE INDEX IF NOT EXISTS idx_facts_subject_predicate
                ON facts(subject,predicate);
            CREATE TABLE IF NOT EXISTS subtasks (
                subtask_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                subject_key TEXT, answer_key TEXT, data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_subtasks_subject ON subtasks(subject_key);
            CREATE INDEX IF NOT EXISTS idx_subtasks_answer ON subtasks(answer_key);
            CREATE TABLE IF NOT EXISTS skeletons (
                skeleton_id TEXT PRIMARY KEY, pattern TEXT NOT NULL,
                data TEXT NOT NULL
            );
        """)
        try:
            self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS source_search
                USING fts5(source_id UNINDEXED, title, body)""")
            self._fts = True
        except sqlite3.OperationalError:
            self._fts = False

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def add_fact(self, subject: str, predicate: str, obj: str, kind: str) -> bool:
        key = "fact-" + digest(subject, predicate, obj, kind)
        cur = self.db.execute(
            "INSERT OR IGNORE INTO facts "
            "(fact_id,subject,predicate,object,object_kind) VALUES (?,?,?,?,?)",
            (key, subject, predicate, obj, kind),
        )
        return cur.rowcount > 0

    def get_facts(self, subject: str, predicate: str | None = None):
        if predicate is None:
            return list(self.db.execute(
                "SELECT * FROM facts WHERE subject=? ORDER BY predicate,object",
                (subject,),
            ))
        return list(self.db.execute(
            "SELECT * FROM facts WHERE subject=? AND predicate=? ORDER BY object",
            (subject, predicate),
        ))

    def subjects(self, limit: int = 100000) -> Iterator[str]:
        for row in self.db.execute(
            "SELECT DISTINCT subject FROM facts ORDER BY subject LIMIT ?", (limit,)
        ):
            yield str(row[0])

    def add_source(self, source_id: str, dataset: str, modality: str,
                   title: str, body: str, asset_path: str | None = None,
                   metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        self.db.execute(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?)",
            (source_id, dataset, modality, title, body, asset_path,
             json.dumps(metadata, ensure_ascii=False)),
        )
        if self._fts:
            self.db.execute("DELETE FROM source_search WHERE source_id=?", (source_id,))
            self.db.execute(
                "INSERT INTO source_search(source_id,title,body) VALUES (?,?,?)",
                (source_id, title, body),
            )

    def source(self, source_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM sources WHERE source_id=?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"])
        return data

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search public title and body only, without expected answers/Gold IDs."""
        words = re.findall(r"[^\W_]+", str(query).lower(), re.UNICODE)
        if not words:
            return []
        if self._fts:
            fts = " OR ".join('"' + w.replace('"', '""') + '"' for w in words[:10])
            try:
                rows = self.db.execute(
                    "SELECT source_id FROM source_search WHERE source_search MATCH ?"
                    " ORDER BY bm25(source_search) LIMIT ?",
                    (fts, max(50, limit * 8)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        else:
            # FTS5 is present on usual Python distributions. Fallback is
            # deliberately bounded and ranked only on public text.
            rows = self.db.execute(
                "SELECT source_id FROM sources WHERE lower(title) LIKE ? LIMIT ?",
                ("%" + words[0] + "%", max(50, limit * 8)),
            ).fetchall()
        scored = []
        terms = set(words)
        for row in rows:
            source = self.source(str(row["source_id"]))
            if source is None:
                continue
            title = source["title"].lower()
            text = (title + " " + source["body"]).lower()
            overlap = len([w for w in terms if w in text])
            title_overlap = len([w for w in terms if w in title])
            score = 3 * title_overlap + overlap
            scored.append((-score, source["source_id"], source))
        scored.sort()
        return [
            {"source_id": s["source_id"], "title": s["title"],
             "modality": s["modality"], "snippet": s["body"][:220]}
            for _, _, s in scored[:limit]
        ]

    def add_skeleton(self, record: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO skeletons VALUES (?,?,?)",
            (record["skeleton_id"], record["pattern"],
             json.dumps(record, ensure_ascii=False)),
        )

    def skeletons(self) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT data FROM skeletons ORDER BY skeleton_id"
        )]

    def cache_subtask(self, row: dict[str, Any], *, subject_key: str = "",
                      answer_key: str = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO subtasks VALUES (?,?,?,?,?)",
            (row["subtask_id"], row["kind"], subject_key, answer_key,
             json.dumps(row, ensure_ascii=False)),
        )

    def task(self, subtask_id: str) -> dict[str, Any] | None:
        r = self.db.execute(
            "SELECT data FROM subtasks WHERE subtask_id=?", (subtask_id,)
        ).fetchone()
        return json.loads(r[0]) if r else None

    def cached_count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM subtasks").fetchone()[0])

    def stats(self) -> dict[str, int]:
        return {
            "sources": int(self.db.execute("SELECT count(*) FROM sources").fetchone()[0]),
            "facts": int(self.db.execute("SELECT count(*) FROM facts").fetchone()[0]),
            "templates": int(self.db.execute(
                "SELECT count(DISTINCT predicate) FROM facts").fetchone()[0]),
            "skeletons": int(self.db.execute(
                "SELECT count(*) FROM skeletons").fetchone()[0]),
            "cached_subtasks": self.cached_count(),
        }

    def create_yago_pages(self, *, max_subjects: int = 100000) -> int:
        """Publish one searchable page per real YAGO subject."""
        created = 0
        # snapshot subjects because individual writes to sources also occur.
        for subject in self.subjects(max_subjects):
            facts = self.get_facts(subject)
            if not facts:
                continue
            sid = "yago-page-" + digest(subject)
            lines = [
                f"{display(subject)} — {display(f['predicate'])}: "
                f"{display(f['object']) if f['object_kind']=='entity' else f['object']}"
                for f in facts[:80]
            ]
            self.add_source(
                sid, "YAGO-4.6", "text", display(subject),
                "\n".join(lines),
                metadata={"subject_id": subject, "fact_count": len(facts),
                          "truncated": len(facts) > 80},
            )
            self.db.execute(
                "UPDATE facts SET source_id=? WHERE subject=?",
                (sid, subject),
            )
            created += 1
            if created % 1000 == 0:
                self.db.commit()
        self.db.commit()
        return created
