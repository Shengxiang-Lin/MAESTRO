"""True source-grounded macro task plans with reliable replayable feedback.

Fixtures follow released KQA-Pro KoPL and MultiModalQA annotation shapes.
They never require model APIs or real image/video/audio perception.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from envir.macro.priors import load_kqa, load_mmqa
from envir.macro.v3_builder import build_many, build_one, counterfactual_sensitivity
from envir.macro.v3_cli import main
from envir.macro.v3_runtime import (MissingEvidence, evaluate, validate_grounded)


class TestMacroV3(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.kqa_path = self.tmp / "kqa.json"
        self.mmqa_path = self.tmp / "mmqa.jsonl"
        kqa = [
            {
                "question": "Which groups intersect after the filter?",
                "program": [
                    {"function": "Find", "inputs": ["A"], "dependencies": []},
                    {"function": "Relate", "inputs": ["member", "forward"], "dependencies": [0]},
                    {"function": "Find", "inputs": ["B"], "dependencies": []},
                    {"function": "Relate", "inputs": ["member", "forward"], "dependencies": [2]},
                    {"function": "And", "inputs": [], "dependencies": [1, 3]},
                    {"function": "Count", "inputs": [], "dependencies": [4]},
                ],
            },
            {
                "question": "Which objects satisfy the condition?",
                "program": [
                    {"function": "FindAll", "inputs": [], "dependencies": []},
                    {"function": "FilterConcept", "inputs": ["Object"], "dependencies": [0]},
                    {"function": "FilterNum", "inputs": ["value", ">", "5"], "dependencies": [1]},
                    {"function": "Count", "inputs": [], "dependencies": [2]},
                ],
            },
            {
                "question": "Which is earlier?",
                "program": [
                    {"function": "Find", "inputs": ["A"], "dependencies": []},
                    {"function": "Find", "inputs": ["B"], "dependencies": []},
                    {"function": "SelectBetween", "inputs": ["date", "less"], "dependencies": [0, 1]},
                ],
            },
        ]
        self.kqa_path.write_text(json.dumps(kqa), encoding="utf-8")
        mmqa = [
            {"qid": "r1", "question": "Which painting is also in the table?",
             "metadata": {"type": "Intersect(ImageListQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "im", "doc_part": "image"},
                              {"doc_id": "tb", "doc_part": "table"}]}},
            {"qid": "r2", "question": "Which of the two matches has a higher score?",
             "metadata": {"type": "Compare(TextQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "tx", "doc_part": "text"},
                              {"doc_id": "tbl", "doc_part": "table"}]}},
            {"qid": "r3", "question": "What result comes after looking up the entity?",
             "metadata": {"type": "Compose(ImageQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "im2", "doc_part": "image"},
                              {"doc_id": "tx2", "doc_part": "text"}]}},
            {"qid": "r4", "question": "How many works satisfy the conditions?",
             "metadata": {"type": "Count(TableQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "tx3", "doc_part": "text"},
                              {"doc_id": "tb3", "doc_part": "table"}]}},
        ]
        self.mmqa_path.write_text(
            "\n".join(json.dumps(row) for row in mmqa) + "\n",
            encoding="utf-8",
        )
        self.kqa, _ = load_kqa(self.kqa_path)
        self.mmqa, _ = load_mmqa(self.mmqa_path)

    def test_100_grounded_scenarios_have_verified_feedback(self):
        scenes, stats = build_many(kqa=self.kqa, mmqa=self.mmqa,
                                   target=100, seed=17)
        self.assertEqual(len(scenes), 100)
        self.assertEqual(stats["value_counterfactuals_changed_answer"], 100)
        self.assertEqual(len(stats["intents"]), 8)
        self.assertGreaterEqual(len(stats["domains"]), 9)
        self.assertEqual(stats["unique_question_texts"], 100)
        self.assertEqual(stats["max_concurrent_subagents"], 10)
        self.assertGreater(stats["total_reliable_feedback_packets"], 300)
        print("MACRO_V3_STATS", json.dumps({
            k: stats[k] for k in ("intents", "macro_node_distribution",
                                 "macro_depth_distribution",
                                 "coarsening_actual_merges",
                                 "unique_macro_work_graphs",
                                 "total_reliable_feedback_packets")
        }, sort_keys=True))
        for scene in scenes:
            self.assertEqual(validate_grounded(scene), [])
            self.assertTrue(
                scene["verification"]["counterfactual_sensitivity"]["answer_changed"])
            self.assertNotIn("gold_answer", json.dumps(scene["public"]))
            self.assertNotIn("feedback_bank", json.dumps(scene["public"]))
            self.assertTrue(scene["world"]["real_media_available"] is False)
            self.assertTrue(all(
                len(packet["summary_trace"]) == len(packet["tool_events"])
                for packet in scene["feedback_bank"].values()))
            for packet in scene["feedback_bank"].values():
                self.assertEqual(packet["verification"],
                                 "deterministic_simulated_source_replay")

    def test_scoped_parallel_compared_work_products_are_distinct(self):
        scene = next(
            build_one(ordinal=i, seed=17, kqa=self.kqa, mmqa=self.mmqa)
            for i in range(80)
            if build_one(ordinal=i, seed=17, kqa=self.kqa,
                         mmqa=self.mmqa)["information_need"]["intent"] == "comparison"
        )
        tasks = scene["macro_graph"]["tasks"]
        scoped = [t for t in tasks if t["outputs"] == {"matched_cases": "CaseSet"}]
        self.assertEqual(len(scoped), 2)
        self.assertNotEqual(scoped[0]["scope_key"], scoped[1]["scope_key"])
        terminal = next(t for t in tasks if t["work_type"] == "final_assessment")
        self.assertEqual(set(terminal["inputs"]), {"case_a", "case_b"})
        self.assertNotEqual(terminal["inputs"]["case_a"]["from_task"],
                            terminal["inputs"]["case_b"]["from_task"])

    def test_tampered_feedback_and_grounding_rejected(self):
        scene = build_one(ordinal=1, seed=17, kqa=self.kqa, mmqa=self.mmqa)
        corrupt = copy.deepcopy(scene)
        first_id = next(iter(corrupt["feedback_bank"]))
        corrupt["feedback_bank"][first_id]["summary_trace"].append("Invented evidence.")
        self.assertIn("feedback_trace_or_binding_drift", validate_grounded(corrupt))
        corrupt = copy.deepcopy(scene)
        corrupt["gold_answer"]["made_up"] = True
        self.assertIn("gold_answer_replay_mismatch", validate_grounded(corrupt))
        corrupt = copy.deepcopy(scene)
        sid = next(iter(corrupt["world"]["sources"]))
        del corrupt["world"]["sources"][sid]
        self.assertTrue(any(x.startswith("runtime_failed:")
                            for x in validate_grounded(corrupt)))
        corrupt = copy.deepcopy(scene)
        corrupt["public"]["gold_answer"] = scene["gold_answer"]
        self.assertIn("public_projection_keys", validate_grounded(corrupt))

    def test_visibility_and_source_ablation(self):
        scene = build_one(ordinal=3, seed=17, kqa=self.kqa, mmqa=self.mmqa)
        world = scene["world"]
        self.assertEqual(evaluate(scene, world)["gold_answer"],
                         scene["gold_answer"])
        for sid in world["sources"]:
            with self.assertRaises(MissingEvidence):
                evaluate(scene, world, hidden_sources=frozenset((sid,)))
        witness = counterfactual_sensitivity(scene)
        self.assertTrue(witness["answer_changed"])
        public = scene["public"]
        self.assertTrue(all("observations" not in source
                            for source in public["provided_sources"]))

    def test_optional_question_styles_and_scoped_local_author(self):
        from envir.macro.v3_author import rewrite_public_questions, assess_surface
        omni = self.tmp / "omnigaia.json"
        omni.write_text(json.dumps([
            {"id": "o1", "question": "I'm checking two research reports. Compare their results.",
             "answer": "DO-NOT-COPY", "level": "Hard"},
            {"id": "o2", "question": "Calculate the number of months from the video.",
             "answer": "PRIVATE", "annotated_solution": "SECRET"},
        ]), encoding="utf-8")
        from envir.macro.v2_omni import load_omni_styles
        styles, _ = load_omni_styles(omni)
        scenes, _ = build_many(kqa=self.kqa, mmqa=self.mmqa,
                               target=8, seed=17, omni_styles=styles)

        class Stub:
            prompts = []
            def complete(self, packet):
                self.prompts.append(packet)
                return packet["reference_question"]

        agent = Stub()
        stats = rewrite_public_questions(scenes, author=agent, workers=2)
        self.assertEqual(stats["accepted"], 8)
        self.assertEqual(len(agent.prompts), 8)
        for prompt in agent.prompts:
            self.assertEqual(set(prompt), {
                "situation", "reference_question", "resource_descriptions", "task",
            })
            self.assertNotIn("macro_graph", json.dumps(prompt))
            self.assertNotIn("gold_answer", json.dumps(prompt))
            self.assertNotIn("DO-NOT-COPY", json.dumps(prompt))
            self.assertNotIn("SECRET", json.dumps(prompt))
        for scene in scenes:
            self.assertEqual(validate_grounded(scene), [])
            self.assertFalse(scene["question_authoring"]["semantic_equivalence_certified"])
            bad = scene["question"] + " Next, delegate to S1."
            self.assertIn("orchestration_leak", assess_surface(scene, bad))

    def test_cli_build_replay_no_overwrite(self):
        out = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["build", "--kqa", str(self.kqa_path),
                                  "--mmqa", str(self.mmqa_path),
                                  "--output-dir", str(out), "--target", "24"]), 0)
            self.assertEqual(main(["replay", "--scenes",
                                   str(out / "private_grounded_scenes_v3.jsonl")]), 0)
        self.assertEqual(len((out / "public_questions_v3.jsonl").read_text().splitlines()), 24)
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["build", "--kqa", str(self.kqa_path),
                      "--mmqa", str(self.mmqa_path),
                      "--output-dir", str(out), "--target", "24"])


if __name__ == "__main__":
    unittest.main()
