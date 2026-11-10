"""Regression tests for round-local faulty evidence and dynamic Main recovery."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from environment.__main__ import main
from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.faults import Fault, FaultUnsupported
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.generate_turns import generate_turns
from environment.untrusted_orchestration.config import DEFAULT_CONFIG, load_config


def sample(i=0):
    spec = make_spec(
        seed_record={
            "record_id": "untrusted-" + str(i), "topic": "Pomona, California",
            "domain_hint": "geography",
        },
        ordinal=i, run_seed=97, difficulty="hard",
        macro_first_v2_share=100,
    )
    scene = compile_and_validate(spec, reference_question(spec))
    reports = [{
        **deepcopy(row),
        "finish_result": {
            "status": "done",
            "result": "Observed: " + json.dumps(row["answer"]),
            "summary": "Grounded observations " + json.dumps(row["answer"]),
        },
        "report_authoring": {"status": "accepted"},
    } for row in realize_scene(scene)["macro_tasks"]]
    episode, _, _, status = build_episode(scene, reports)
    if status["status"] != "accepted":
        raise AssertionError("invalid_clean_fixture")
    return scene, reports, episode


class RoundLocalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.episode = sample(0)

    def test_config_only_entrypoint_and_model_paths(self):
        config = load_config()
        self.assertEqual(DEFAULT_CONFIG.name, "untrusted_orchestration.yaml")
        self.assertEqual(config["mode"], "template")
        self.assertTrue(config["scenes_path"].is_absolute())
        self.assertEqual(config["fault_modes"], ["missing", "partial", "wrong"])
        with patch("environment.__main__.generate_untrusted_from_config",
                   return_value={"status": "partial"}) as runner:
            self.assertEqual(main(["generate-untrusted-turns",
                                   "--config", str(DEFAULT_CONFIG),
                                   "--mode", "template", "--max-scenes", "5"]), 0)
            self.assertEqual(runner.call_args.kwargs["max_scenes"], 5)

    def test_missing_report_has_outer_done_and_recovers_by_targeted_retry(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        ep, turns, audit = build_branch(
            self.scene, self.reports, self.episode,
            branch_round=2, faults=[Fault(tid, "missing")],
        )
        self.assertEqual(audit["outcome"], "recovered")
        self.assertTrue(turns)
        self.assertEqual(turns[0]["round"], 2)
        self.assertIn("no result", turns[0]["input"]["subtask_history"])
        original = ep["trajectory"][0]["result"]["subtask_results"][0]
        self.assertTrue(original["done"])
        self.assertIsNone(original["finish_result"])
        self.assertEqual(len(audit["runtime"]["attempt_versions"][tid]), 2)
        self.assertEqual(audit["counterfactual_unrepaired"]["status"], "blocked")
        self.assertEqual(ep["trajectory"][-1]["action"], "complete")
        self.assertNotIn("private", json.dumps(turns))
        self.assertNotIn("fault", json.dumps(turns))

    def test_persistent_missing_is_not_magically_repaired(self):
        tid = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.episode, branch_round=2,
            faults=[Fault(tid, "missing", persistence="same_path")],
            max_retries=1,
        )
        self.assertEqual(turns, [])
        self.assertEqual(audit["outcome"], "unresolved_dependencies")
        self.assertNotEqual(episode["trajectory"][-1]["action"], "complete")

    def test_partial_preserves_fields_and_restores_upstream(self):
        tasks = {t["task_id"]: t for t in self.scene["macro_graph"]["tasks"]}
        tid = next(tid for tid in self.scene["macro_graph"]["reference_waves"][0]
                   if len(tasks[tid]["outputs"]["typed_relation"]) > 1)
        ep, turns, audit = build_branch(
            self.scene, self.reports, self.episode, branch_round=2,
            faults=[Fault(tid, "partial")],
        )
        self.assertEqual(audit["outcome"], "recovered")
        versions = audit["runtime"]["attempt_versions"][tid]
        self.assertEqual(versions[0]["status"], "partial")
        self.assertTrue(versions[0]["runtime"]["available_fields"])
        self.assertTrue(versions[0]["runtime"]["missing_fields"])
        self.assertEqual(versions[1]["status"], "done")
        self.assertTrue(turns)
        self.assertEqual(ep["trajectory"][-1]["action"], "complete")

    def test_wrong_join_uses_real_distractor_and_exposes_propagation(self):
        found = False
        for i in range(12):
            scene, rows, episode = (
                (self.scene, self.reports, self.episode)
                if i == 0 else sample(i)
            )
            targets = [task["task_id"] for task in scene["macro_graph"]["tasks"]
                       if any(x["op"] == "join"
                              for x in task["operator_chain"])]
            if not targets:
                continue
            tid = targets[0]
            wave = next(i for i, w in enumerate(
                scene["macro_graph"]["reference_waves"]) if tid in w)
            if wave + 2 > len([
                row for row in episode["trajectory"]
                if row["action"] == "delegate_task"
            ]) + 1:
                continue
            try:
                ep, turns, audit = build_branch(
                    scene, rows, episode, branch_round=wave + 2,
                    faults=[Fault(tid, "wrong")],
                )
            except FaultUnsupported:
                continue
            if audit["faults"][0]["execution"].get("kind") != (
                    "misjoined_real_candidates"):
                continue
            found = True
            self.assertEqual(audit["counterfactual_unrepaired"]["status"],
                             "completed")
            self.assertFalse(audit["counterfactual_unrepaired"]["correct"])
            self.assertEqual(audit["outcome"], "recovered")
            self.assertIn("Recheck only", json.dumps(turns[0]["output"]))
            versions = audit["runtime"]["attempt_versions"][tid]
            self.assertEqual(len(versions), 2)
            self.assertNotEqual(versions[0]["rows"], versions[1]["rows"])
            self.assertEqual(ep["trajectory"][-1]["action"], "complete")
            break
        self.assertTrue(found, "no valid wrong-join fixture found")

    def test_journal_resume_positive_publication_and_manifest(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes, reports, episodes = (
                root / "scenes.jsonl", root / "subtasks.jsonl",
                root / "episodes.jsonl",
            )
            scenes.write_text(json.dumps(self.scene) + "\n")
            reports.write_text("".join(json.dumps(x) + "\n"
                                       for x in self.reports))
            episodes.write_text(json.dumps(self.episode) + "\n")
            args = dict(
                scenes_path=scenes, reports_path=reports,
                base_episodes_path=episodes, output_dir=root / "output",
                max_scenes=1, branches_per_scene=1,
                seed=17, fault_modes=["missing"], workers=2,
                no_progress=True,
            )
            summary = generate_turns(**args)
            self.assertEqual(summary["requested_branches"], 1)
            self.assertEqual(summary["recovered"], 1)
            self.assertGreater(summary["main_training_turns"], 0)
            output = root / "output"
            lines = [
                json.loads(s) for s in
                (output / "main_turns.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(
                set(turn["input"]) == {"subtask_history"} for turn in lines
            ))
            self.assertNotIn("faults", json.dumps(lines))
            self.assertTrue((output / "private" /
                             "fault_alignment.jsonl").exists())
            self.assertEqual(generate_turns(**args), summary)
            self.assertEqual(len(lines),
                             len((output / "main_turns.jsonl")
                                 .read_text().splitlines()))
            with self.assertRaisesRegex(ValueError, "signature_mismatch"):
                generate_turns(**{**args, "seed": 99})


if __name__ == "__main__":
    unittest.main()
