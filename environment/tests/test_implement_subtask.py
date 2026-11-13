"""Stage-2 oracle Subtask answer + evidence-trace regression tests."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from environment.__main__ import parser
from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate
from environment.implement_subtask.config import (
    DEFAULT_CONFIG, load_config,
)
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.pipeline import materialize_subtasks


def _scene(index: int, difficulty: str = "medium") -> dict:
    spec = make_spec(
        seed_record={
            "record_id": f"stage2-test-{index}",
            "topic": f"Fictional Study {index}",
            "domain_hint": "science",
            "source_modalities": ["table", "image", "audio", "text"],
        }, ordinal=index, run_seed=17, difficulty=difficulty,
    )
    scene = compile_and_validate(spec, reference_question(spec))
    scene["question_authoring"] = {"status": "accepted", "attempts": 1}
    scene["question_origin"] = "llm_accepted_frozen_spec"
    return scene


def _write(path: Path, scenes: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(sc, ensure_ascii=False) + "\n" for sc in scenes),
        encoding="utf-8",
    )


class Stage2ConfigTests(unittest.TestCase):
    def test_stage2_config_and_cli_are_separate_from_generation(self):
        cfg = load_config(DEFAULT_CONFIG)
        self.assertTrue(str(cfg["scenes_path"]).endswith("questions/scenes.jsonl"))
        self.assertTrue(cfg["strict_replay"])
        self.assertEqual(cfg["evidence_limit"], 12)
        cmds = parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(cmds), {"generate-questions", "quality", "implement-subtasks", "naturalize-subtasks", "generate-main-turns", "generate-untrusted-turns", "run"}
        )

    def test_invalid_config_fails(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "bad.yaml"
            file.write_text(
                "version: 1\nimplementation:\n  scenes: path.jsonl\n"
                "  output_dir: output\n  evidence_limit: -1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                load_config(file)


class Stage2TraceTests(unittest.TestCase):
    def test_every_macro_has_exact_grounded_answer_and_trace(self):
        for index in range(12):
            difficulty = ("easy", "medium", "hard")[index % 3]
            with self.subTest(index=index, difficulty=difficulty):
                scene = _scene(index, difficulty)
                original = copy.deepcopy(scene)
                bank = realize_scene(scene)
                self.assertEqual(scene, original, "Stage 1 must be immutable")
                tasks = scene["macro_graph"]["tasks"]
                self.assertEqual(len(bank["macro_tasks"]), len(tasks))
                feedback = {row["task_id"]: row for row in scene["task_feedback"]}
                for row in bank["macro_tasks"]:
                    original_fb = feedback[row["macro_task_id"]]
                    self.assertEqual(
                        row["answer"], [
                            entry["values"] for entry in original_fb["workproduct"]
                        ]
                    )
                    self.assertEqual(
                        row["answer_with_provenance"],
                        original_fb["workproduct"],
                    )
                    self.assertEqual(
                        row["trace_summary"]["answer_supporting_fact_refs"],
                        sorted(original_fb["fact_refs"]),
                    )
                    self.assertGreater(len(row["trace_summary"]["steps"]), 0)
                    for step in row["trace_summary"]["steps"]:
                        self.assertIn("action", step)
                        self.assertIn("observed_result", step)
                        for atom in step["evidence_atoms"]:
                            self.assertIn(
                                atom["source_id"],
                                row["trace_summary"]["available_evidence_source_ids"],
                            )
                            source = scene["world"]["sources"][atom["source_id"]]
                            actual = next(
                                obs for obs in source["observations"]
                                if obs["observation_id"] == atom["observation_id"]
                            )
                            self.assertEqual(atom["values"], actual["values"])
                            self.assertIn(atom["fact_ref"], actual["fact_refs"])
                terminal = next(
                    row for row in bank["macro_tasks"]
                    if row["macro_task_id"] == bank["terminal_macro_task_id"]
                )
                self.assertCountEqual(
                    [json.dumps(x, sort_keys=True) for x in terminal["answer"]],
                    [json.dumps(x, sort_keys=True) for x in scene["gold_answer"]],
                )

    def test_evidence_limit_does_not_lose_provenance(self):
        scene = _scene(19, "hard")
        full = realize_scene(scene, evidence_limit=100)
        compressed = realize_scene(scene, evidence_limit=0)
        for a, b in zip(full["macro_tasks"], compressed["macro_tasks"]):
            self.assertEqual(a["answer"], b["answer"])
            self.assertEqual(
                a["trace_summary"]["answer_supporting_fact_refs"],
                b["trace_summary"]["answer_supporting_fact_refs"],
            )
            for step in b["trace_summary"]["steps"]:
                self.assertEqual(step["evidence_atoms"], [])
                if step["evidence_atoms_total"] > 0:
                    self.assertTrue(step["evidence_atoms_truncated"])

    def test_modified_source_scene_is_refused_before_export(self):
        invalid = _scene(13, "easy")
        invalid["world"]["sources"]["register"]["observations"][0][
            "values"]["active"] = "tampered"
        with self.assertRaises(ValueError):
            realize_scene(invalid, strict=True)


class Stage2OutputTests(unittest.TestCase):
    def test_rejected_stage1_scene_never_enters_macro_bank(self):
        valid = _scene(31, "medium")
        rejected = _scene(34, "easy")
        rejected["question_authoring"]["status"] = "fallback"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "accepted_and_failed.jsonl"
            _write(input_path, [valid, rejected])
            output = root / "stage2"
            result = materialize_subtasks(
                scenes_path=input_path, output_dir=output,
                no_progress=True,
            )
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["attempted_scenes"], 2)
            self.assertEqual(result["accepted_scenes"], 1)
            self.assertEqual(result["failed_scenes"], 1)
            accepted = [
                json.loads(row) for row in
                (output / "subtasks.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(accepted), valid["macro_graph"]["macro_task_count"])
            self.assertEqual(
                {row["scene_task_id"] for row in accepted},
                {valid["task_id"]},
            )
            failed = [
                json.loads(row) for row in
                (output / "failed.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["scene_task_id"], rejected["task_id"])
            with self.assertRaises(FileExistsError):
                materialize_subtasks(
                    scenes_path=input_path, output_dir=output,
                    no_progress=True,
                )

    def test_all_valid_scenes_publish_all_macro_answers(self):
        rows = [_scene(40, "hard"), _scene(41, "medium")]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "scenes.jsonl"
            _write(source, rows)
            result = materialize_subtasks(
                scenes_path=source, output_dir=root / "stage2",
                no_progress=True,
            )
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["failed_scenes"], 0)
            self.assertEqual(
                result["materialized_macro_tasks"],
                sum(x["macro_graph"]["macro_task_count"] for x in rows),
            )
            self.assertEqual(
                len((root / "stage2/scene_index.jsonl").read_text().splitlines()),
                len(rows),
            )


if __name__ == "__main__":
    unittest.main()
