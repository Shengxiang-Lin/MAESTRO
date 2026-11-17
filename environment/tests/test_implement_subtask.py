"""Stage-2 oracle Subtask answer + evidence-trace regression tests."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from environment.__main__ import parser
from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.config import (
    DEFAULT_CONFIG, load_config,
)
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.pipeline import materialize_subtasks


def _scene(index: int, difficulty: str = "medium") -> dict:
    scene = current_scene(index, difficulty=difficulty)
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

    def test_unreviewed_deterministic_stage1_question_is_publishable(self):
        # Stage 1's default question.model=null produces this exact shape:
        # deterministic_frozen_problem_spec with no question_authoring.
        valid = _scene(50, "medium")
        valid.pop("question_authoring")
        valid["question_origin"] = "deterministic_frozen_problem_spec"

        # An LLM-authored question without a review is NOT allowed merely
        # because its question_authoring field is missing.
        unreviewed_llm = _scene(51, "medium")
        unreviewed_llm.pop("question_authoring")

        # Explicit negative review decisions must still fail closed.
        failed = _scene(52, "easy")
        failed["question_authoring"]["status"] = "failed"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "scenes.jsonl"
            _write(source, [valid, unreviewed_llm, failed])
            result = materialize_subtasks(
                scenes_path=source, output_dir=root / "stage2",
                no_progress=True, strict_replay=True,
            )
            self.assertEqual(result["status"], "partial", result)
            self.assertEqual(result["accepted_scenes"], 1)
            self.assertEqual(result["failed_scenes"], 2)
            self.assertEqual(
                result["materialized_macro_tasks"],
                valid["macro_graph"]["macro_task_count"],
            )
            failures = [
                json.loads(row)
                for row in (root / "stage2/failed.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["scene_task_id"] for row in failures},
                {unreviewed_llm["task_id"], failed["task_id"]},
            )
            self.assertTrue(all(
                row["reason"] == "ValueError:stage1_question_not_accepted"
                for row in failures
            ))

    def test_empty_oracle_writes_real_summary_with_failure_causes(self):
        from environment.implement_subtask.combined import (
            generate_subagent_reports,
        )

        rejected = _scene(54, "medium")
        rejected["question_authoring"]["status"] = "failed"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenes = root / "scenes.jsonl"
            _write(scenes, [rejected])
            output = root / "stage2"
            result = generate_subagent_reports(
                scenes_path=scenes,
                output_dir=output,
                reporting_options={"overwrite": False},
                no_progress=True,
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["oracle_accepted_scenes"], 0)
            self.assertEqual(result["oracle_failed_scenes"], 1)
            self.assertEqual(result["oracle_failure_reasons"], {
                "ValueError:stage1_question_not_accepted": 1
            })
            self.assertEqual(
                json.loads((output / "summary.json").read_text()),
                result,
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
