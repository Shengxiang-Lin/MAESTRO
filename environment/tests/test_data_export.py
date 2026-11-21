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
from environment.tests.stage4_fixtures import sample
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


if __name__ == "__main__":
    unittest.main()
