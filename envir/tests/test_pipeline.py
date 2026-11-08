"""Offline integration checks for the independent envir pipeline."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from envir.cli import main
from envir.composition import TaskGraphComposer
from envir.store import EvidenceStore
from envir.subtasks import SubtaskFactory, unique_entity_labels
from envir.validation import check_graph


class EnvirIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.kqa = self.root / "kqa.json"
        self.yago = self.root / "yago-facts.ttl"
        self.mmqa = self.root / "MMQA_train.jsonl"
        self.bank = self.root / "output" / "bank.sqlite"
        self.kqa.write_text(json.dumps([
            {"question": "Who is the director?",
             "program": [
                 {"function": "Find", "inputs": ["Film A"], "dependencies": []},
                 {"function": "Relate", "inputs": ["director", "forward"], "dependencies": [0]},
                 {"function": "QueryAttr", "inputs": ["birthDate"], "dependencies": [1]},
             ]},
            {"question": "What are two attributes?",
             "program": [
                 {"function": "Find", "inputs": ["Film A"], "dependencies": []},
                 {"function": "Find", "inputs": ["Director B"], "dependencies": []},
                 {"function": "And", "inputs": [], "dependencies": [0, 1]},
                 {"function": "Count", "inputs": [], "dependencies": [2]},
             ]},
        ]), encoding="utf-8")
        self.yago.write_text(
            'yago:Film_A schema:director yago:Director_B .\n'
            'yago:Film_A schema:datePublished "2000" .\n'
            'yago:Film_A schema:producer yago:Producer_Q .\n'
            'yago:Director_B schema:birthDate "1970" .\n'
            'yago:Director_B schema:birthPlace yago:City_C .\n'
            'yago:City_C schema:population "3000" .\n'
            'yago:Producer_Q schema:birthDate "1980" .\n',
            encoding="utf-8",
        )
        self.mmqa.write_text(json.dumps({
            "qid": "img-q-1", "question": "Which film is pictured?",
            "answer": "Film A",
            "metadata": {"type": "ImageQ", "supporting_context": [
                {"doc_id": "image-1", "doc_part": "image"},
            ]},
        }) + "\n", encoding="utf-8")
        (self.root / "images.jsonl").write_text(json.dumps({
            "id": "image-1", "title": "Film poster", "caption": "Film A poster",
        }) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([
                "prepare", "--kqa", str(self.kqa), "--yago", str(self.yago),
                "--mmqa", str(self.mmqa), "--output-dir", str(self.bank.parent),
                "--max-yago-triples", "100", "--max-kqa-records", "100",
                "--max-mmqa-records", "100",
            ])
        self.assertEqual(code, 0)

    def test_prepare_has_sources_skeletons_templates_and_labeled_mmqa(self):
        with EvidenceStore(self.bank) as db:
            stats = db.stats()
            self.assertGreaterEqual(stats["facts"], 7)
            self.assertGreaterEqual(stats["sources"], 5)
            self.assertGreaterEqual(stats["skeletons"], 2)
            self.assertIn("chain", {r["pattern"] for r in db.skeletons()})
            self.assertIn("parallel", {r["pattern"] for r in db.skeletons()})
            mmqa = db.db.execute("SELECT data FROM subtasks WHERE kind='mmqa_qa'").fetchone()
            self.assertIsNotNone(mmqa)
            task = json.loads(mmqa["data"])
            self.assertEqual(task["verification"], "dataset_labeled")
            self.assertEqual(task["provenance"]["media_status"], ["caption_proxy"])
            self.assertEqual(task["tool_events"], [])
        templates = json.loads((self.bank.parent / "subtask_templates.json").read_text())
        self.assertTrue(any(t["predicate"] == "schema:director" for t in templates))

    def test_official_mmqa_answers_plural_and_supporting_context(self):
        """The released dataset uses answers=[{answer:...}], not answer."""
        from envir.sources.mmqa import ingest_mmqa
        official = self.root / "MMQA_official.jsonl"
        valid = {
            "qid": "official-qa", "question": "Which film is described?",
            "answers": [{"answer": "Film A", "type": "string",
                         "modality": "text"}],
            "supporting_context": [{"doc_id": "text-7",
                                    "doc_part": "text"}],
            "metadata": {"type": "TextQ"},
        }
        ambiguous = {
            **valid, "qid": "conflicting-answers",
            "answers": [{"answer": "Film A"}, {"answer": "Film B"}],
        }
        official.write_text(
            json.dumps(valid) + "\n" + json.dumps(ambiguous) + "\n",
            encoding="utf-8",
        )
        (self.root / "MMQA_texts.jsonl").write_text(json.dumps({
            "id": "text-7", "title": "Film archive",
            "text": "Film A is described in this paragraph.",
        }) + "\n", encoding="utf-8")
        with EvidenceStore(self.bank) as db:
            stats = ingest_mmqa(db, official, max_records=10)
            self.assertEqual(stats["context_documents_found"], 1)
            self.assertEqual(stats["eligible_labeled_records"], 1)
            self.assertEqual(stats["missing_or_ambiguous_answers"], 1)
            self.assertEqual(stats["labeled_subtasks"], 1)
            row = db.db.execute(
                "SELECT data FROM subtasks WHERE kind='mmqa_qa' "
                "AND answer_key='film a' ORDER BY subtask_id LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(row)

    def test_lazy_verification_and_tamper_rejection(self):
        with EvidenceStore(self.bank) as db:
            factory = SubtaskFactory(db)
            before = db.cached_count()
            task = factory.yago("yago:Film_A", "schema:director")
            self.assertIsNotNone(task)
            self.assertEqual(task["answer"]["entity_id"], "yago:Director_B")
            self.assertTrue(factory.replay(task))
            self.assertEqual(factory.yago("yago:Film_A", "schema:director"), task)
            self.assertEqual(db.cached_count(), before + 1)
            bad = {**task, "answer": {**task["answer"], "value": "Fake Director"}}
            self.assertFalse(factory.replay(bad))
            invalid_trace = {**task, "summary_trace": ["Invented a web page."]}
            self.assertFalse(factory.replay(invalid_trace))
            db.cache_subtask(bad, subject_key="yago:Film_A")
            self.assertIsNone(factory.yago("yago:Film_A", "schema:director"))
            self.assertIsNone(factory.yago("yago:Film_A", "schema:unknown"))

    def test_chain_parallel_cross_dataset_bridge(self):
        with EvidenceStore(self.bank) as db:
            composer = TaskGraphComposer(db)
            for pattern in ("chain", "parallel", "media_to_search"):
                graph = getattr(composer, pattern)()
                self.assertIsNotNone(graph, f"missing {pattern}: {composer.rejections}")
                self.assertEqual(check_graph(graph, db), [])
                required = {node["subtask_id"] for node in graph["nodes"]}
                self.assertFalse(set(graph["optional_candidate_ids"]) & required)
                self.assertEqual(
                    graph["validation"]["status"],
                    "mixed_provenance" if pattern == "media_to_search" else "verified",
                )
            self.assertEqual(graph["nodes"][0]["verification"], "dataset_labeled")
            self.assertEqual(graph["dependencies"][0]["binding"], "unique_exact_label")

    def test_broken_dependency_detected(self):
        with EvidenceStore(self.bank) as db:
            graph = TaskGraphComposer(db).chain()
            self.assertIsNotNone(graph)
            graph["dependencies"][0]["binding"] = "answer"
            self.assertIn("unexpected_answer_binding", check_graph(graph, db))
            graph["dependencies"][0]["binding"] = "entity_id"
            graph["nodes"][1]["inputs"]["subject_id"] = "yago:Another"
            self.assertIn("entity_binding_mismatch", check_graph(graph, db))

    def test_public_export_does_not_contain_private_gold(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([
                "compose", "--bank", str(self.bank),
                "--output-dir", str(self.bank.parent),
                "--target", "3", "--max-attempts", "30",
            ])
        self.assertEqual(code, 0)
        public = [
            json.loads(line) for line in
            (self.bank.parent / "public_questions.jsonl").read_text().splitlines()
        ]
        private = [
            json.loads(line) for line in
            (self.bank.parent / "gold_graphs.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(public), 3)
        self.assertEqual(len(private), 3)
        self.assertTrue(all("gold_answer" not in q and "nodes" not in q
                            and "dependencies" not in q for q in public))
        self.assertEqual({q["task_id"] for q in public},
                         {g["task_id"] for g in private})

    def test_explicit_lazy_instantiation_and_no_silent_bank_overwrite(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "instantiate", "--bank", str(self.bank),
                "--subject", "yago:Film_A", "--predicate", "schema:director",
            ]), 0)
            self.assertEqual(main([
                "instantiate", "--bank", str(self.bank),
                "--subject", "yago:Film_A", "--predicate", "schema:unknown",
            ]), 2)
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main([
                    "prepare", "--kqa", str(self.kqa), "--yago", str(self.yago),
                    "--mmqa", str(self.mmqa),
                    "--output-dir", str(self.bank.parent),
                ])

    def test_type_only_mmqa_yago_joins_not_accepted(self):
        with EvidenceStore(self.bank) as db:
            self.assertEqual(unique_entity_labels(db)["film a"], "yago:Film_A")
            row = db.db.execute(
                "SELECT subtask_id,data FROM subtasks WHERE kind='mmqa_qa'"
            ).fetchone()
            payload = json.loads(row["data"])
            payload["answer"]["value"] = "Unknown film"
            db.cache_subtask(payload, answer_key="unknown film")
            composer = TaskGraphComposer(db)
            self.assertIsNone(composer.media_to_search())
            self.assertIn("no_exact_mmqa_yago_bridge", composer.rejections)


if __name__ == "__main__":
    unittest.main()
