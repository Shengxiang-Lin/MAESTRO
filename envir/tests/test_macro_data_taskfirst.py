"""Focused tests for the dataset-driven Task-First question generator."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from envir.macro.scene_catalog import load_scenario_seeds, parse_composition
from envir.macro.data_taskfirst import UnsuitableSeed, compose_seed_taskfirst
from envir.macro.v4_builder import build_one
from envir.macro.v4_validation import validate_clean_scene


KQA = [{
    "signature": "fixture-kopl",
    "program": {"nodes": []},
    "logical_depth": 4,
    "branch_nodes": 1,
    "merge_nodes": 1,
    "operator_count": 4,
}]


def seed(tree="Compare(Compose(TableQ,ImageQ),TextQ)"):
    return {
        "record_id": "mmqa-fixture-1",
        "question": "Which artist has the earlier exhibition date?",
        "topic": "two artists",
        "entities": ["Artist A", "Artist B"],
        "domain_hint": "culture",
        "operation": "comparison",
        "question_type": tree,
        "logical_pattern": "comparison",
        "composition": parse_composition(tree),
        "source_modalities": ["table", "image", "text"],
        "supporting_sources": [
            {"modality": "table", "doc_id": "t1",
             "provenance": "mmqa_supporting_context"},
            {"modality": "image", "doc_id": "i1",
             "provenance": "mmqa_supporting_context"},
            {"modality": "text", "doc_id": "x1",
             "provenance": "mmqa_supporting_context"},
        ],
        "yago_matches": [],
    }


class MMQATaskFirstTests(unittest.TestCase):
    def test_nested_composition_parser_preserves_dependencies(self):
        tree = parse_composition("Compare(Compose(TableQ,ImageQ),TextQ)")
        self.assertEqual(tree["op"], "Compare")
        self.assertEqual(tree["children"][0]["op"], "Compose")
        self.assertEqual(tree["children"][0]["children"][1]["op"], "ImageQ")
        self.assertEqual(tree["children"][1]["op"], "TextQ")

    def test_catalog_preserves_source_ids_and_does_not_copy_answers(self):
        record = {
            "qid": "mmqa-001",
            "question": "Which artist has the earlier exhibition date?",
            "answers": [{"answer": "SECRET_ANSWER"}],
            "wiki_entities_in_question": ["Artist A", "Artist B"],
            "metadata": {
                "type": "Compare(TableQ,ImageQ)",
                "supporting_context": [
                    {"doc_id": "table-101", "doc_part": "table"},
                    {"doc_id": "image-202", "doc_part": "image"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "MMQA_train.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            seeds, stats = load_scenario_seeds(path)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(stats["unique_scenario_seeds"], 1)
        self.assertEqual([s["doc_id"] for s in seeds[0]["supporting_sources"]],
                         ["table-101", "image-202"])
        self.assertNotIn("SECRET_ANSWER", json.dumps(seeds))

    def test_macro_dag_compiled_before_question(self):
        record = seed()
        plan = compose_seed_taskfirst(
            seed_record=record, ordinal=0, run_seed=17, kqa=KQA)
        tasks = plan["macro_graph"]["tasks"]
        self.assertEqual(len(tasks), 5)
        self.assertEqual(len(plan["macro_graph"]["edges"]), 4)
        self.assertEqual(plan["question"], record["question"])
        self.assertEqual(plan["goal"]["intent"], "comparison")
        self.assertEqual(plan["macro_graph"]["terminal_handoffs"][0]["type"],
                         "AnswerReport")
        self.assertEqual(plan["source_priors"]["MultiModalQA"]["record_id"],
                         record["record_id"])
        self.assertEqual(plan["source_priors"]["KQA"]["signature"], "fixture-kopl")
        self.assertTrue(all(edge["type"] for edge in plan["macro_graph"]["edges"]))

    def test_complete_clean_replay_and_question_provenance(self):
        scene = build_one(
            ordinal=0, seed=17, kqa=KQA, seed_record=seed())
        self.assertEqual([], validate_clean_scene(scene))
        self.assertEqual(scene["public"]["question"], scene["question"])
        self.assertEqual(scene["question"], seed()["question"])
        self.assertEqual(scene["goal_evidence_coverage"]["can_finish_cleanly"], True)
        self.assertEqual(len(scene["macro_graph"]["tasks"]), 5)
        self.assertEqual(scene["world"]["real_multimodal_files"], False)

    def test_two_independent_sources_yield_three_tasks(self):
        record = seed("TextQ")
        record["supporting_sources"] = record["supporting_sources"][:2]
        record["source_modalities"] = ["table", "image"]
        plan = compose_seed_taskfirst(
            seed_record=record, ordinal=1, run_seed=17, kqa=KQA)
        self.assertEqual(len(plan["macro_graph"]["tasks"]), 3)
        self.assertEqual(len(plan["macro_graph"]["edges"]), 2)

    def test_single_source_is_not_inflated_into_fake_macro_dag(self):
        record = seed("TextQ")
        record["source_modalities"] = ["text"]
        record["supporting_sources"] = [record["supporting_sources"][-1]]
        with self.assertRaises(UnsuitableSeed):
            compose_seed_taskfirst(
                seed_record=record, ordinal=1, run_seed=17, kqa=KQA)


if __name__ == "__main__":
    unittest.main()
