"""Canonical paired data + immutable Scene split regression tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dataset.prepare_da_grpo import digest, read_jsonl
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
            manifest = export_dataset(
                scenes_path=paths["scenes"],
                clean_episodes_path=paths["clean"],
                clean_turns_path=paths["turns"],
                stage4_dir=stage4, output_dir=out,
                require_llm=False,
            )
            self.assertEqual(manifest["accepted_fault_cases"], 2)
            splits = json.loads(
                (out / "splits.json").read_text(encoding="utf-8")
            )["scene_to_split"]
            case_split = splits[scene_id]
            cases = read_jsonl(out / "cases" / (case_split + ".jsonl"))
            paired = next(row for row in cases if row["case_id"] == cid)
            self.assertEqual(paired["clean"], clean_rows[0])
            self.assertEqual(paired["untrusted"]["episode"], ep)
            self.assertTrue(paired["fault"]["applied_events"])
            self.assertTrue(paired["recovery"]["globally_recovered"])
            self.assertEqual(paired["clean_reference_id"],
                             ep["clean_reference_id"])
            decisions = read_jsonl(
                out / "decisions" / (case_split + ".jsonl")
            )
            target = [row for row in decisions if row["case_id"] == cid]
            self.assertEqual(len(target), len(turns))
            sibling = [row for row in decisions if row["case_id"] == cid2]
            self.assertEqual(len(sibling), len(turns))
            self.assertTrue(set(row["decision_id"] for row in target).isdisjoint(
                {row["decision_id"] for row in sibling}
            ))
            self.assertEqual({
                row["source"] for row in target
            }, {"untrusted"})
            self.assertTrue(all(
                row["prompt"][0]["role"] == "user" and
                "subtask_history" not in row["prompt"][0] and
                "private_audit" not in row["prompt"][0]["content"]
                for row in target
            ))
            # All Scene variants inherit the one frozen allocation.
            for split in ("train", "val", "test"):
                for row in read_jsonl(
                    out / "decisions" / (split + ".jsonl")
                ):
                    self.assertEqual(row["split"], splits[row["scene_id"]])
                sft = read_jsonl(
                    out / "adapters/da-sft" / (split + ".jsonl")
                )
                self.assertTrue(all(
                    set(row) == {"prompt", "completion"} for row in sft
                ))
                for row in read_jsonl(
                    out / "adapters/da-grpo" / (split + ".jsonl")
                ):
                    self.assertEqual(row["extra_info"]["split"], split)
                    self.assertEqual(
                        row["prompt"][0]["role"], "user",
                    )
            with self.assertRaises(FileExistsError):
                export_dataset(
                    scenes_path=paths["scenes"],
                    clean_episodes_path=paths["clean"],
                    clean_turns_path=paths["turns"],
                    stage4_dir=stage4, output_dir=out,
                    require_llm=False,
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
                split="train", published=True,
            )


if __name__ == "__main__":
    unittest.main()
