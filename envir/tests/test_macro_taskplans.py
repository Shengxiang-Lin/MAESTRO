"""Macro TaskPlan tests: no API, actual media, or real-world YAGO needed."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from envir.macro.cli import main
from envir.macro.graph import (MacroPlanError, build_atomic_draft,
                               coarsen, compile_contracts, compile_waves)
from envir.macro.planner import build_batch, build_plan
from envir.macro.priors import load_kqa, load_mmqa
from envir.macro.validation import validate_plan


class TestMacroTaskPlans(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.base = Path(td.name)
        self.kqa_path = self.base / "train.json"
        self.mmqa_path = self.base / "mmqa.jsonl"
        kqa = [
            {
                "question": "Which company operates the relevant vehicles?",
                "program": [
                    {"function": "Find", "inputs": ["Vehicles"], "dependencies": []},
                    {"function": "Relate", "inputs": ["operator", "forward"],
                     "dependencies": [0]},
                    {"function": "FilterConcept", "inputs": ["Company"],
                     "dependencies": [1]},
                    {"function": "Count", "inputs": [], "dependencies": [2]},
                ],
            },
            {
                "question": "Which objects appear in both groups?",
                "program": [
                    {"function": "Find", "inputs": ["A"], "dependencies": []},
                    {"function": "Relate", "inputs": ["linked", "forward"],
                     "dependencies": [0]},
                    {"function": "Find", "inputs": ["B"], "dependencies": []},
                    {"function": "Relate", "inputs": ["linked", "forward"],
                     "dependencies": [2]},
                    {"function": "And", "inputs": [], "dependencies": [1, 3]},
                    {"function": "Count", "inputs": [], "dependencies": [4]},
                ],
            },
        ]
        self.kqa_path.write_text(json.dumps(kqa), encoding="utf-8")
        records = [
            {"qid": "r1", "question": "Which painting matches the museum program?",
             "metadata": {"type": "Compose(ImageQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "im1", "doc_part": "image"},
                              {"doc_id": "tb1", "doc_part": "table"},
                          ]}},
            {"qid": "r2", "question": "Which participants appear in both sources?",
             "metadata": {"type": "Intersect(ImageListQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "im2", "doc_part": "image"},
                              {"doc_id": "tx2", "doc_part": "text"},
                          ]}},
            {"qid": "r3", "question": "Is the description consistent with the table?",
             "metadata": {"type": "Compare(TableQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "tb3", "doc_part": "table"},
                              {"doc_id": "tx3", "doc_part": "text"},
                          ]}},
        ]
        self.mmqa_path.write_text(
            "\n".join(json.dumps(x) for x in records) + "\n",
            encoding="utf-8",
        )
        self.kqa, _ = load_kqa(self.kqa_path)
        self.mmqa, _ = load_mmqa(self.mmqa_path)

    def test_100_plans_cross_domains_and_shapes(self):
        plans, stats = build_batch(kqa=self.kqa, mmqa=self.mmqa,
                                   target=100, seed=17)
        self.assertEqual(len(plans), 100)
        self.assertEqual(len(stats["macro_graph_patterns"]), 5)
        self.assertEqual(len(stats["domains"]), 9)
        self.assertGreaterEqual(min(stats["patterns_per_domain"].values()), 5)
        self.assertGreaterEqual(stats["unique_question_texts"], 90)
        self.assertEqual(stats["fixed_concurrency_limit"], 10)
        self.assertEqual(len({p["task_id"] for p in plans}), 100)
        for plan in plans:
            self.assertEqual(validate_plan(plan), [])
            self.assertEqual(plan["scheduler"]["max_concurrent_subagents"], 10)
            self.assertEqual(plan["scheduler"]["runtime_variants_generated"], False)
            self.assertNotIn("macro_graph", plan["public"])
            self.assertNotIn("source_priors", plan["public"])
            self.assertGreater(plan["complexity"]["collapsed_internal_steps"], 0)
            self.assertEqual(plan["construction_status"],
                             "structural_candidate_not_fact_grounded")
            self.assertGreaterEqual(plan["complexity"]["macro_dependency_depth"], 2)

    def test_atomic_coarsening_rejects_cross_package_micro_leak(self):
        packages = [
            {"task_id": "A", "internal_actions": ["read media", "deliver result"],
             "outputs": {"case": "CaseProfile"}, "input_schema": {}},
            {"task_id": "B", "internal_actions": ["audit records", "deliver audit"],
             "outputs": {"audit": "RecordSet"}, "input_schema": {"upstream": "CaseProfile"}},
        ]
        edges = [{"from": "A", "to": "B", "input_slot": "upstream",
                  "output_slot": "case"}]
        draft = build_atomic_draft(packages, edges)
        coarse, stats = coarsen(draft)
        self.assertEqual(len(coarse), 2)
        self.assertEqual(stats["collapsed_internal_steps"], 2)
        compile_contracts(coarse, edges)
        self.assertEqual(coarse[1]["inputs"]["upstream"]["from_task"], "A")
        altered = copy.deepcopy(draft)
        altered["steps"][2]["depends_on"].append("A.internal.1")
        with self.assertRaises(MacroPlanError):
            coarsen(altered)

    def test_schema_damage_is_detected(self):
        plan = build_plan(ordinal=3, seed=17, kqa=self.kqa, mmqa=self.mmqa)
        broken = copy.deepcopy(plan)
        dependent = next(t for t in broken["macro_graph"]["tasks"]
                         if t["inputs"])
        first = next(iter(dependent["inputs"]))
        dependent["inputs"][first]["type"] = "Wrong"
        self.assertIn("invalid_typed_handoff", validate_plan(broken))
        broken = copy.deepcopy(plan)
        broken["macro_graph"]["tasks"][0]["depends_on"] = ["S99"]
        self.assertIn("invalid_or_cyclic_dag", validate_plan(broken))

    def test_public_projection_has_no_gold_or_micro_ops(self):
        plan = build_plan(ordinal=4, seed=17, kqa=self.kqa, mmqa=self.mmqa)
        public_str = json.dumps(plan["public"])
        self.assertNotIn("internal_execution_outline", public_str)
        self.assertNotIn("full_internal_logical_program", public_str)
        self.assertNotIn("reference_waves", public_str)
        self.assertNotIn("answer", plan["public"])

    def test_optional_question_author_sees_only_public_question_brief(self):
        from envir.macro.author import naturalize_questions, check_rewrite
        plans, _ = build_batch(kqa=self.kqa, mmqa=self.mmqa,
                               target=6, seed=17)

        class StubAuthor:
            packets = []
            def complete(self, packet):
                self.packets.append(packet)
                return packet["reference_question"]

        author = StubAuthor()
        stats = naturalize_questions(plans, author=author, workers=2)
        self.assertEqual(stats["accepted"], 6, [p["question_authoring"] for p in plans])
        for packet in author.packets:
            self.assertEqual(set(packet), {
                "situation", "user_role", "reference_question",
                "expected_deliverable", "resource_descriptions",
            })
            self.assertNotIn("source_priors", json.dumps(packet))
            self.assertNotIn("internal_execution_outline", json.dumps(packet))
        for plan in plans:
            self.assertEqual(validate_plan(plan), [])
            self.assertFalse(plan["question_authoring"]["semantic_equivalence_certified"])
            self.assertIn("orchestration_instructions",
                          check_rewrite(plan, "First subtask S1: inspect the media."))

    def test_cli_build_replay_and_nonoverwrite(self):
        out = self.base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "build", "--kqa", str(self.kqa_path),
                "--mmqa", str(self.mmqa_path), "--target", "18",
                "--output-dir", str(out),
            ]), 0)
            self.assertEqual(main([
                "replay", "--plans", str(out / "private_taskplans.jsonl")
            ]), 0)
        self.assertEqual(len((out / "public_questions.jsonl").read_text().splitlines()), 18)
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["build", "--kqa", str(self.kqa_path),
                      "--mmqa", str(self.mmqa_path), "--target", "18",
                      "--output-dir", str(out)])


if __name__ == "__main__":
    unittest.main()
