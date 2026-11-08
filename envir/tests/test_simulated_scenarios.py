"""Dataset-schema and simulator tests; no real media files or LLM needed."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from envir.scenarios.cli import main
from envir.scenarios.generator import build_scene, generate_scenes
from envir.scenarios.mmqa_motifs import extract_motif, load_motifs
from envir.scenarios.runtime import SimulatedRuntime
from envir.scenarios.validation import derive_answer, validate_scene


class TestMMQASyntheticScenes(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mmqa = self.root / "MMQA_train.jsonl"
        records = [
            {"qid": "r1", "question": "Which painting is in the museum?",
             "metadata": {"type": "Compose(ImageQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "im1", "doc_part": "image"},
                              {"doc_id": "tab1", "doc_part": "table"},
                          ]},
             "pseudo_language_question": "find image then lookup table",
             "intermediate_answers": ["Painting A"],
             "answers": [{"answer": "Painting B"}]},
            {"qid": "r2", "question": "What team is shown in the match photo?",
             "metadata": {"type": "Intersect(TextQ,ImageQ)",
                          "supporting_context": [
                              {"doc_id": "im2", "doc_part": "image"},
                              {"doc_id": "tx2", "doc_part": "text"},
                          ]},
             "answers": [{"answer": "Team Alpha"}]},
            {"qid": "r3", "question": "How many buses stopped at the station?",
             "metadata": {"type": "Compose(TableQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "tab3", "doc_part": "table"},
                              {"doc_id": "tx3", "doc_part": "text"},
                          ]},
             "wiki_entities_in_question": ["bus station"],
             "answers": [{"answer": "3"}]},
        ]
        self.mmqa.write_text("\n".join(json.dumps(x) for x in records) + "\n",
                             encoding="utf-8")

    def test_mmqa_extracts_reusable_motifs_without_media_files(self):
        motifs, stats = load_motifs(self.mmqa, limit=10)
        self.assertEqual(stats["valid"], 3)
        self.assertIn("image", motifs[0]["source_modalities"])
        self.assertIn("table", motifs[0]["source_modalities"])
        self.assertEqual(motifs[0]["logical_pattern"], "composition")
        self.assertEqual(motifs[0]["supporting_source_count"], 2)
        self.assertEqual(motifs[0]["domain_hint"], "culture")
        self.assertNotIn("answer", motifs[0])
        self.assertIn("intermediate_pattern", motifs[0])

    def test_36_scenes_are_diverse_and_replayable(self):
        motifs, _ = load_motifs(self.mmqa)
        scenes, stats = generate_scenes(motifs, target=36, seed=17)
        self.assertEqual(len(scenes), 36)
        self.assertGreaterEqual(len(stats["domains"]), 8)
        self.assertGreaterEqual(len(stats["scene_types"]), 10)
        self.assertEqual(set(stats["media"]), {"image","video","audio"})
        self.assertEqual(set(stats["graph_patterns"]),
                         {"chain","fork_join","cross_check"})
        self.assertEqual(len({s["scene_id"] for s in scenes}), len(scenes))
        for s in scenes:
            self.assertEqual(validate_scene(s), [])
            self.assertEqual(derive_answer(s), s["gold_answer"])
            self.assertEqual(s["verification"]["real_media_perception"], False)
            for source_id in s["required_source_ids"]:
                self.assertIsNone(derive_answer(s, hidden_sources={source_id}))
            self.assertNotIn(s["world"]["target_reference"], s["question"])

    def test_tampered_feedback_and_wrong_gold_are_rejected(self):
        motifs, _ = load_motifs(self.mmqa)
        scene = build_scene(motifs[0], 11, seed=17, domain="culture",
                            modality="video", pattern="fork_join")
        tampered = copy.deepcopy(scene)
        tampered["task_graph"]["nodes"][0]["summary_trace"] = ["Invented observation."]
        self.assertIn("subtask_feedback_or_binding_drift",
                      validate_scene(tampered))
        tampered = copy.deepcopy(scene)
        tampered["gold_answer"]["registration"] = "Invented"
        self.assertIn("answer_source_replay_mismatch",
                      validate_scene(tampered))
        tampered = copy.deepcopy(scene)
        tampered["sources"] = [
            s for s in tampered["sources"]
            if not s["source_id"].endswith(":registry")
        ]
        self.assertIn("answer_source_replay_mismatch",
                      validate_scene(tampered))

    def test_runtime_privacy_tool_access_and_feedback_variants(self):
        motifs, _ = load_motifs(self.mmqa)
        scene = build_scene(motifs[1], 7, seed=22, domain="sports",
                            modality="audio", pattern="cross_check")
        runtime = SimulatedRuntime(scene)
        public = runtime.public_task()
        self.assertNotIn("gold_answer", public)
        self.assertNotIn(scene["world"]["target_reference"], str(public))
        media_id = scene["scene_id"] + ":media"
        observed = runtime.call("InspectSimulatedMedia", source_id=media_id)
        self.assertEqual(observed["modality"], "audio")
        self.assertGreaterEqual(len(observed["observations"]), 4)
        results = runtime.call("SimulatedSearch",
                               query=scene["world"]["target_reference"])
        self.assertIn(scene["scene_id"] + ":registry",
                      [r["source_id"] for r in results["results"]])
        wrong = runtime.subagent_feedback(variant="misleading")
        self.assertNotEqual(wrong["answer"], scene["world"]["target_reference"])
        partial = runtime.subagent_feedback(variant="partial")
        self.assertIsNone(partial["answer"])
        self.assertEqual(runtime.total_cost, 3)

    def test_cli_roundtrip_and_explicit_overwrite(self):
        out = self.root / "generated"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "build", "--mmqa", str(self.mmqa),
                "--output-dir", str(out), "--target", "12"
            ]), 0)
            self.assertEqual(main([
                "replay", "--gold", str(out / "scenes_gold.jsonl")
            ]), 0)
        public = [json.loads(line) for line in
                  (out / "public_tasks.jsonl").read_text().splitlines()]
        gold = [json.loads(line) for line in
                (out / "scenes_gold.jsonl").read_text().splitlines()]
        self.assertEqual(len(public), 12)
        self.assertEqual(len(gold), 12)
        self.assertTrue(all("world" not in p and "gold_answer" not in p
                            for p in public))
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["build", "--mmqa", str(self.mmqa),
                      "--output-dir", str(out), "--target", "12"])


if __name__ == "__main__":
    unittest.main()
