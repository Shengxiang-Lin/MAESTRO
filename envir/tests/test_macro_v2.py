"""Tests macro v2 against representative original-dataset schemas, no model API."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from envir.macro.v2_cli import main
from envir.macro.v2_omni import load_omni_styles
from envir.macro.v2_planner import build_v2_batch, build_v2_plan
from envir.macro.v2_validation import validate_v2
from envir.macro.priors import load_kqa, load_mmqa


class MacroV2Tests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.kqa_path = self.tmp / "kqa.json"
        self.mmqa_path = self.tmp / "mmqa.jsonl"
        kqa = [
            {"question": "Which actors are related to the film?",
             "program": [
                 {"function": "Find", "inputs": ["Film A"], "dependencies": []},
                 {"function": "Relate", "inputs": ["cast", "forward"], "dependencies": [0]},
                 {"function": "FilterConcept", "inputs": ["Actor"], "dependencies": [1]},
                 {"function": "Count", "inputs": [], "dependencies": [2]}]},
            {"question": "What do the two groups have in common?",
             "program": [
                 {"function": "Find", "inputs": ["A"], "dependencies": []},
                 {"function": "Relate", "inputs": ["related", "forward"], "dependencies": [0]},
                 {"function": "Find", "inputs": ["B"], "dependencies": []},
                 {"function": "Relate", "inputs": ["related", "forward"], "dependencies": [2]},
                 {"function": "And", "inputs": [], "dependencies": [1, 3]},
                 {"function": "Count", "inputs": [], "dependencies": [4]}]},
        ]
        self.kqa_path.write_text(json.dumps(kqa), encoding="utf-8")
        mmqa = [
            {"qid": "r1", "question": "Which painting shown is on the museum list?",
             "metadata": {"type": "Compose(TableQ,ImageListQ)",
                          "supporting_context": [
                              {"doc_id": "a", "doc_part": "image"},
                              {"doc_id": "b", "doc_part": "table"}]}},
            {"qid": "r2", "question": "Which player is also in the table?",
             "metadata": {"type": "Intersect(ImageListQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "c", "doc_part": "image"},
                              {"doc_id": "d", "doc_part": "table"}]}},
            {"qid": "r3", "question": "Which year is earlier?",
             "metadata": {"type": "Compare(TableQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "e", "doc_part": "text"},
                              {"doc_id": "f", "doc_part": "table"}]}},
        ]
        self.mmqa_path.write_text(
            "\n".join(json.dumps(row) for row in mmqa) + "\n",
            encoding="utf-8")
        self.kqa, _ = load_kqa(self.kqa_path)
        self.mmqa, _ = load_mmqa(self.mmqa_path)

    def test_100_real_macro_combinations(self):
        plans, stats = build_v2_batch(
            kqa=self.kqa, mmqa=self.mmqa, target=100, seed=17)
        self.assertEqual(len(plans), 100)
        self.assertGreaterEqual(stats["unique_macro_work_types"], 13)
        self.assertGreaterEqual(stats["unique_macro_recipe_graphs"], 15)
        self.assertGreaterEqual(stats["unique_intent_topology_pairs"], 12)
        self.assertGreaterEqual(stats["unique_question_texts"], 90)
        self.assertEqual(stats["fixed_max_concurrency"], 10)
        self.assertEqual(len(stats["domains"]), 9)
        self.assertEqual(len(stats["intents"]), 8)
        print("MACRO_V2_STATS", json.dumps({
            k: stats[k] for k in ("unique_macro_work_types",
                                 "unique_macro_recipe_graphs",
                                 "unique_intent_topology_pairs",
                                 "macro_graph_patterns",
                                 "unique_question_texts",
                                 "intent_topology_diversity",
                                 "macro_depth_distribution")
        }, sort_keys=True))
        for plan in plans:
            self.assertEqual(validate_v2(plan), [])
            self.assertEqual(plan["scheduler"]["max_concurrent_subagents"], 10)
            self.assertFalse(plan["scheduler"]["runtime_variants_generated"])
            self.assertNotIn("macro_graph", plan["public"])
            self.assertNotIn("source_priors", plan["public"])
            self.assertNotIn("summary_trace", json.dumps(plan["public"]))
            self.assertNotIn("gold_answer", plan)
            self.assertGreater(plan["complexity"]["internal_operation_count"],
                               plan["complexity"]["macro_task_count"])

    def test_malicious_contract_replacement_detected(self):
        p = build_v2_plan(ordinal=9, seed=17, kqa=self.kqa, mmqa=self.mmqa)
        broken = copy.deepcopy(p)
        edge = broken["macro_graph"]["edges"][0]
        edge["handoff_reason"] = "a plausible but unsupported justification"
        self.assertIn("missing_semantic_handoff_rationale",
                      validate_v2(broken))
        broken = copy.deepcopy(p)
        dependent = next(t for t in broken["macro_graph"]["tasks"]
                         if t["inputs"])
        slot = next(iter(dependent["inputs"]))
        dependent["inputs"][slot]["type"] = "WrongType"
        self.assertIn("incorrect_typed_dependency", validate_v2(broken))
        broken = copy.deepcopy(p)
        broken["public"]["macro_graph"] = broken["macro_graph"]
        self.assertIn("public_contains_private_keys", validate_v2(broken))

    def test_optional_omnigaia_question_styles_no_answers(self):
        path = self.tmp / "omni.json"
        data = [
            {"id": 1, "question": "I'm studying two images. Please compare their dates.",
             "level": "Hard", "answer": "SECRET ANSWER",
             "annotated_solution": ["A private reference step"]},
            {"id": 2, "question": "Calculate the number of months between the events.",
             "level": "Medium", "answer": "PRIVATE"},
        ]
        path.write_text(json.dumps(data), encoding="utf-8")
        styles, stats = load_omni_styles(path)
        self.assertEqual(stats["records"], 2)
        self.assertEqual(styles[0]["style"], "contextual_investigation")
        p = build_v2_plan(ordinal=1, seed=17, kqa=self.kqa,
                          mmqa=self.mmqa, omni_styles=styles)
        self.assertEqual(validate_v2(p), [])
        self.assertNotIn("SECRET ANSWER", json.dumps(p))
        self.assertNotIn("private reference step", json.dumps(p))
        self.assertEqual(p["source_priors"]["OmniGAIA"]["policy"],
                         "real_dataset_question_style_only")

    def test_cli_build_replay_without_touching_v1(self):
        out = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "build", "--kqa", str(self.kqa_path),
                "--mmqa", str(self.mmqa_path),
                "--output-dir", str(out), "--target", "25",
            ]), 0)
            self.assertEqual(main([
                "replay", "--plans", str(out / "private_taskplans_v2.jsonl")
            ]), 0)
        public = [json.loads(x) for x in
                  (out / "public_questions_v2.jsonl").read_text().splitlines()]
        self.assertEqual(len(public), 25)
        self.assertNotIn("internal_execution_outline", json.dumps(public))
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main([
                    "build", "--kqa", str(self.kqa_path),
                    "--mmqa", str(self.mmqa_path),
                    "--output-dir", str(out), "--target", "25"
                ])


if __name__ == "__main__":
    unittest.main()
