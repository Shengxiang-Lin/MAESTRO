"""V4: Task-first public Question and complete per-Task evidence (clean only).

Small fixtures mirror KQA-Pro program and MMQA official question-type schema,
not private real-world data and not actual multimedia files.
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
from envir.macro.v4_builder import build_batch, build_one
from envir.macro.v4_cli import main
from envir.macro.v4_evidence import run_clean_tasks, EvidenceExecutionError
from envir.macro.v4_taskfirst import author_question, task_level_brief
from envir.macro.v4_validation import validate_clean_scene


class CleanTaskFirstTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.kqa_file = self.temp / "train.json"
        self.mmqa_file = self.temp / "MMQA_train.jsonl"
        kqa = [
            {
                "question": "Which groups share a member?",
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
                "question": "How many things match?",
                "program": [
                    {"function": "FindAll", "inputs": [], "dependencies": []},
                    {"function": "FilterConcept", "inputs": ["Item"], "dependencies": [0]},
                    {"function": "FilterNum", "inputs": ["count", ">", "5"], "dependencies": [1]},
                    {"function": "Count", "inputs": [], "dependencies": [2]},
                ],
            },
            {
                "question": "Which is earlier?",
                "program": [
                    {"function": "Find", "inputs": ["A"], "dependencies": []},
                    {"function": "Find", "inputs": ["B"], "dependencies": []},
                    {"function": "SelectBetween", "inputs": ["date", "less"],
                     "dependencies": [0, 1]},
                ],
            },
        ]
        mmqa = [
            {"qid": "a", "question": "Which photograph matches the record?",
             "metadata": {"type": "Compose(ImageQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "i", "doc_part": "image"},
                              {"doc_id": "t", "doc_part": "table"}]}},
            {"qid": "b", "question": "Which person is in both sources?",
             "metadata": {"type": "Intersect(ImageListQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "j", "doc_part": "image"},
                              {"doc_id": "x", "doc_part": "text"}]}},
            {"qid": "c", "question": "Which has the larger amount?",
             "metadata": {"type": "Compare(TableQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "k", "doc_part": "table"},
                              {"doc_id": "y", "doc_part": "text"}]}},
        ]
        self.kqa_file.write_text(json.dumps(kqa), encoding="utf-8")
        self.mmqa_file.write_text(
            "\n".join(json.dumps(r) for r in mmqa) + "\n",
            encoding="utf-8",
        )
        self.kqa, _ = load_kqa(self.kqa_file)
        self.mmqa, _ = load_mmqa(self.mmqa_file)

    def build(self, ordinal):
        return build_one(
            ordinal=ordinal, seed=17, kqa=self.kqa, mmqa=self.mmqa)

    def test_100_scenes_have_clean_multi_evidence_feedback(self):
        scenes, stats = build_batch(
            kqa=self.kqa, mmqa=self.mmqa, target=100, seed=17)
        self.assertEqual(stats["generated"], 100)
        self.assertEqual(stats["clean_evidence_coverage"], 100)
        self.assertEqual(stats["fixed_max_concurrency"], 10)
        self.assertFalse(stats["final_gold_answer_generated"])
        self.assertFalse(stats["untrusted_scenarios_generated"])
        self.assertGreater(stats["macro_task_count"], 300)
        self.assertGreater(stats["evidence_node_count"],
                           3 * stats["macro_task_count"] - 1)
        self.assertGreaterEqual(stats["unique_macro_work_types"], 12)
        self.assertGreaterEqual(stats["macro_graph_signatures"], 15)
        self.assertGreaterEqual(stats["unique_question_texts"], 90)
        print("MACRO_V4_STATS", json.dumps({
            k: stats[k] for k in (
                "macro_task_count", "evidence_node_count",
                "mean_evidence_nodes_per_task", "macro_graph_signatures",
                "unique_macro_work_types", "clean_evidence_coverage",
            )
        }, sort_keys=True))
        for s in scenes:
            self.assertEqual(validate_clean_scene(s), [])
            self.assertNotIn("gold_answer", s)
            self.assertNotIn("task_evidence_graphs", s["public"])
            self.assertNotIn("macro_graph", s["public"])
            self.assertTrue(s["goal_evidence_coverage"]["can_finish_cleanly"])
            self.assertTrue(s["goal_evidence_coverage"]["evidence_level"])
            for t in s["macro_graph"]["tasks"]:
                graph = s["task_evidence_graphs"][t["task_id"]]
                feedback = s["feedback_bank"][t["task_id"]]
                self.assertGreaterEqual(len(graph["nodes"]), 3)
                self.assertTrue(any(
                    n["kind"] == "work_synthesis" for n in graph["nodes"]))
                self.assertEqual(len(feedback["summary_trace"]),
                                 len(graph["nodes"]))
                self.assertEqual(feedback["status"], "completed")
                self.assertTrue(feedback["summary"])
                self.assertTrue(feedback["answer"])
                self.assertTrue(all(
                    trace["evidence_id"] in feedback["evidence_refs"]
                    for trace in feedback["summary_trace"]
                ))

    def test_question_determined_by_tasks_not_private_evidence(self):
        scene = self.build(3)
        brief = task_level_brief(scene)
        meta = scene["task_plan_provenance"]
        self.assertEqual(scene["question"], author_question(
            brief, ordinal=meta["ordinal"], seed=meta["seed"]))
        # Private facts can vary while the public Question remains identical.
        changed = copy.deepcopy(scene)
        sid = next(iter(changed["world"]["sources"]))
        changed["world"]["sources"][sid]["facts"][2]["value"] = "updated source fact"
        self.assertEqual(scene["question"], author_question(
            task_level_brief(changed), ordinal=meta["ordinal"],
            seed=meta["seed"]))
        # Changing the actual terminal work goal MUST affect the Question.
        modified_brief = dict(brief)
        modified_brief["final_work_goal"] = "Assess the alternative proposals"
        self.assertNotEqual(scene["question"], author_question(
            modified_brief, ordinal=meta["ordinal"], seed=meta["seed"]))

    def test_evidence_graphs_only_consume_authorized_and_completed_work(self):
        scene = self.build(7)
        for t in scene["macro_graph"]["tasks"]:
            graph = scene["task_evidence_graphs"][t["task_id"]]
            for n in graph["nodes"]:
                if n["kind"] == "source_observation":
                    self.assertIn(
                        n["source_id"],
                        scene["world"]["source_grants"][t["task_id"]],
                    )
                elif n["kind"].startswith("upstream_"):
                    self.assertIn(n["from_task"], t["depends_on"])
        self.assertEqual(
            run_clean_tasks(scene, scene["world"])["feedback_bank"],
            scene["feedback_bank"],
        )

    def test_tampering_and_missing_evidence_detected(self):
        scene = self.build(9)
        modified = copy.deepcopy(scene)
        tid = next(iter(modified["task_evidence_graphs"]))
        modified["task_evidence_graphs"][tid]["nodes"][0]["statement"] = "INVENTED"
        problems = validate_clean_scene(modified)
        self.assertIn(
            "clean_evidence_or_feedback_replay_mismatch:task_evidence_graphs",
            problems,
        )
        modified = copy.deepcopy(scene)
        modified["feedback_bank"][tid]["summary_trace"].append(
            {"evidence_id": "FAKE"}
        )
        self.assertIn(
            "clean_evidence_or_feedback_replay_mismatch:feedback_bank",
            validate_clean_scene(modified),
        )
        modified = copy.deepcopy(scene)
        sid = next(iter(modified["world"]["sources"]))
        del modified["world"]["sources"][sid]
        self.assertTrue(any(
            x.startswith("clean_replay_failed:")
            for x in validate_clean_scene(modified)
        ))
        modified = copy.deepcopy(scene)
        modified["goal_evidence_coverage"]["can_finish_cleanly"] = False
        self.assertIn("evidence_sufficiency_not_proven",
                      validate_clean_scene(modified))
        modified = copy.deepcopy(scene)
        modified["public"]["macro_graph"] = scene["macro_graph"]
        self.assertIn("public_projection_keys",
                      validate_clean_scene(modified))

    def test_cli_build_replay_and_no_overwrite(self):
        path = self.temp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "build", "--kqa", str(self.kqa_file),
                "--mmqa", str(self.mmqa_file),
                "--target", "24", "--output-dir", str(path),
            ]), 0)
            self.assertEqual(main([
                "replay", "--scenes", str(
                    path / "private_clean_scenes_v4.jsonl"),
            ]), 0)
        self.assertEqual(len(
            (path / "public_questions_v4.jsonl").read_text().splitlines()), 24)
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main([
                    "build", "--kqa", str(self.kqa_file),
                    "--mmqa", str(self.mmqa_file),
                    "--target", "24", "--output-dir", str(path),
                ])


if __name__ == "__main__":
    unittest.main()
