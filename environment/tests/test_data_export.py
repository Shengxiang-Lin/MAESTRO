"""Canonical paired data + immutable Scene split regression tests."""
from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from environment.data_export.source_io import digest, read_jsonl
from environment.base_orchestration.trajectory_builder import build_episode
from environment.data_export.export import export_dataset
from environment.data_export.splits import (
    load_splits, make_splits, verify_splits,
)
from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.runtime.faults import Fault
from environment.untrusted_orchestration.trajectory_builder import build_branch


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ), encoding="utf-8")


class FrozenSplitTests(unittest.TestCase):
    def test_exact_scene_group_counts_and_fixed_assignment(self):
        scene_ids = ["scene-" + str(i) for i in range(10)]
        result = make_splits(scene_ids, seed=17)
        self.assertEqual(result["counts"], {
            "train": 8, "val": 1, "test": 1,
        })
        self.assertEqual(
            result, make_splits(list(reversed(scene_ids)), seed=17),
        )
        self.assertEqual(verify_splits(result, scene_ids),
                         result["scene_to_split"])
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "splits.json"
            file.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(load_splits(file, scene_ids), result)
            with self.assertRaisesRegex(ValueError, "coverage"):
                load_splits(file, scene_ids[:-1])

    def test_no_independent_split_of_fault_candidates(self):
        scenes = ["scene-A", "scene-B", "scene-C", "scene-D"]
        assignments = make_splits(scenes)["scene_to_split"]
        all_cases = [
            (scene, variant) for scene in scenes
            for variant in ("clean", "wrong_entity", "computation")
        ]
        for scene in scenes:
            self.assertEqual(len({
                assignments[s] for s, variant in all_cases if s == scene
            }), 1)


class SingleTurnCaseSemanticsTests(unittest.TestCase):
    def test_published_single_turn_is_not_claimed_to_be_globally_recovered(self):
        from environment.data_export.export import _case_for_branch
        scene = {"task_id": "scene-1", "question": "Check the evidence"}
        clean = {"clean_reference_id": "clean-1"}
        sample_id = "scene-1-fork-round-02-candidate-001"
        entry = {"round_sample_id": sample_id, "main_decision_round": 2}
        journal = {
            "task_id": "scene-1",
            "round_sample_id": sample_id,
            "status": "single_turn_accepted",
            "episode": {
                "task_id": "scene-1", "branch_round": 2,
                "clean_reference_id": "clean-1",
                "format": "maestro_untrusted_single_turn.v1",
            },
            "private_audit": {"parallel_batches_verified": True},
        }
        case = _case_for_branch(scene, clean, entry, journal, published=True)
        self.assertTrue(case["recovery"]["decision_accepted"])
        self.assertIsNone(case["recovery"]["globally_recovered"])
        self.assertEqual(case["untrusted"]["publication_status"],
                         "single_turn_accepted")
        # Publication eligibility is independent of global recovery.
        self.assertTrue(case["recovery"]["decision_accepted"])


class PairedDatasetEndToEndTests(unittest.TestCase):
    def test_export_paired_case_and_views_from_original_journals(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_rows, clean_rows, clean_turns = [], [], []
            reports_by_scene = {}
            for i in range(3):
                scene, reports, base = sample(i)
                actual, turns, _, status = build_episode(scene, reports)
                self.assertEqual(status["status"], "accepted")
                scene_rows.append(scene)
                clean_rows.append(actual)
                clean_turns.extend(turns)
                reports_by_scene[scene["task_id"]] = reports
            paths = {
                "scenes": root / "stage1/scenes.jsonl",
                "clean": root / "stage3/episodes.jsonl",
                "turns": root / "stage3/main_turns.jsonl",
            }
            write(paths["scenes"], scene_rows)
            write(paths["clean"], clean_rows)
            write(paths["turns"], clean_turns)
            (root / "stage3/manifest.json").write_text(json.dumps({
                "scenes_sha256": digest(paths["scenes"]),
                "max_attempts": 16, "sub_model": "qwen3-8b",
                "model_alias": "model_1",
            }), encoding="utf-8")
            scene = scene_rows[0]
            scene_id = scene["task_id"]
            tid = scene["macro_graph"]["reference_waves"][0][0]
            ep, turns, audit = build_branch(
                scene, reports_by_scene[scene_id], clean_rows[0],
                branch_round=2, faults=[Fault(tid, "missing")],
            )
            self.assertEqual(audit["outcome"], "recovered")
            cid = scene_id + "-branch-02-fault-001"
            cid2 = scene_id + "-branch-02-fault-002"
            stage4 = root / "stage4"
            write(stage4 / "round_index.jsonl", [{
                "round_sample_id": item,
                "question_id": scene_id,
                "main_decision_round": 2,
                "scenario_blueprint": "latest_fault",
            } for item in (cid, cid2)])
            write(stage4 / "episodes.jsonl", [
                {**ep, "round_sample_id": item}
                for item in (cid, cid2)
            ])
            # Both candidates deliberately have identical display sample_id
            # and visible prompts, but MUST have distinct candidate identity.
            write(stage4 / "main_turns.jsonl", [
                {**turn, "round_sample_id": item}
                for item in (cid, cid2) for turn in turns
            ])
            journal = {
                "task_id": scene_id,
                "round_sample_id": cid,
                "status": "recovered", "episode": ep,
                "private_audit": audit,
                "turns": turns,
            }
            (stage4 / "journal").mkdir(parents=True, exist_ok=True)
            for index, item in enumerate((cid, cid2)):
                payload = {**journal, "round_sample_id": item}
                (stage4 / "journal" / f"{index:06d}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            (stage4 / "manifest.json").write_text(json.dumps({
                "input_sha256": [digest(paths["scenes"]), "report-hash"],
                "settings": {
                    "max_rounds": 16, "sub_model": "qwen3-8b",
                    "model_alias": "model_1",
                },
            }), encoding="utf-8")
            out = root / "unified"
            progress_stream = io.StringIO()
            with redirect_stderr(progress_stream):
                manifest = export_dataset(
                    scenes_path=paths["scenes"],
                    clean_episodes_path=paths["clean"],
                    clean_turns_path=paths["turns"],
                    stage4_dir=stage4, output_dir=out, progress=True,
                )
            self.assertIn("Stage4 Cases", progress_stream.getvalue())
            self.assertIn("stage4 Main Decisions", progress_stream.getvalue())
            self.assertIn("Write decisions.jsonl", progress_stream.getvalue())
            self.assertEqual(manifest["accepted_fault_cases"], 2)
            quality = json.loads(
                (out / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(quality["decision_count"],
                             manifest["decision_count"])
            self.assertEqual(quality["source_counts"]["untrusted"],
                             manifest["decision_sources"]["untrusted"])
            cases = read_jsonl(out / "cases.jsonl")
            paired = next(row for row in cases if row["case_id"] == cid)
            self.assertEqual(paired["clean"], clean_rows[0])
            self.assertEqual(paired["untrusted"]["episode"], ep)
            self.assertTrue(paired["fault"]["applied_events"])
            self.assertTrue(paired["recovery"]["globally_recovered"])
            self.assertEqual(paired["clean_reference_id"],
                             ep["clean_reference_id"])
            self.assertTrue(all("split" not in x for x in cases))
            decisions = read_jsonl(out / "decisions.jsonl")
            labels = read_jsonl(out / "private/decision_labels.jsonl")
            target = [row for row in decisions if row["case_id"] == cid]
            self.assertEqual(len(target), len(turns))
            sibling = [row for row in decisions if row["case_id"] == cid2]
            self.assertEqual(len(sibling), len(turns))
            self.assertTrue(set(row["decision_id"] for row in target).isdisjoint(
                {row["decision_id"] for row in sibling}
            ))
            self.assertTrue(all(
                row["prompt"][0]["role"] == "user" and
                "private_audit" not in row["prompt"][0]["content"] and
                "split" not in row
                for row in decisions
            ))
            self.assertEqual(len(labels), len(decisions))
            self.assertTrue(all(
                x["ground_truth"] and "split" not in x for x in labels
            ))
            self.assertFalse((out / "splits.json").exists())
            self.assertFalse((out / "adapters").exists())
            self.assertEqual(set(manifest["base_file_sha256"]), {
                "cases.jsonl", "decisions.jsonl",
                "private/decision_labels.jsonl",
            })

            with self.assertRaises(FileExistsError):
                export_dataset(
                    scenes_path=paths["scenes"],
                    clean_episodes_path=paths["clean"],
                    clean_turns_path=paths["turns"],
                    stage4_dir=stage4, output_dir=out,
                )

    def test_invalid_case_clean_fingerprint_is_rejected(self):
        from environment.data_export.export import _case_for_branch
        scene = {"task_id": "q1", "question": "Q"}
        clean = {
            "task_id": "q1", "clean_reference_id": "clean-ref",
        }
        journal = {
            "task_id": "q1", "round_sample_id": "q1-branch-02-fault-01",
            "status": "recovered",
            "episode": {
                "task_id": "q1", "branch_round": 2,
                "clean_reference_id": "wrong-ref",
            },
        }
        index = {
            "round_sample_id": "q1-branch-02-fault-01",
            "main_decision_round": 2,
        }
        with self.assertRaisesRegex(ValueError, "clean_reference"):
            _case_for_branch(
                scene, clean, index, journal,
                published=True,
            )


if __name__ == "__main__":
    unittest.main()
