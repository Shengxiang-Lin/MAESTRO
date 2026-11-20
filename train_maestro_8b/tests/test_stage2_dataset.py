"""CPU contracts for canonical Environment -> MAESTRO GRPO conversion."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from environment.data_export.splits import make_splits
from train_maestro_8b.data.prepare_grpo import convert, prepare, sample_id


def decision(scene: str, rnd: int, *, source: str = "untrusted",
             mode: str = "llm_reviewed") -> dict:
    case = f"{scene}-fork-round-02" if source == "untrusted" else f"clean:{scene}"
    did = (f"stage4:{case}:round:{rnd:02d}" if source == "untrusted"
           else f"stage3:{scene}-round-{rnd:02d}")
    return {
        "decision_id": did, "case_id": case, "scene_id": scene,
        "source": source, "decision_round": rnd, "generation_mode": mode,
        "question": f"Question about {scene}",
        "prompt": [{"role": "user", "content": f"Visible {source} {scene} round {rnd}"}],
        "visible_state": {"subtask_history": "Visible Sub-Agent report"},
        "reference_action": {
            "action": "delegate_task", "reasoning": "Find evidence",
            "params": {"tasks": [{
                "task_instruction": "Verify the source",
                "model": "model_1", "tools": ["GoogleSearchAction"],
            }]},
        },
    }


def teacher(row: dict, *, scorable: bool = True) -> dict:
    state = None
    if row["source"] == "untrusted":
        state = {
            "schema": "maestro.decision_reward_state.v1",
            "scorable": scorable, "decision_round": row["decision_round"],
        }
    return {
        "decision_id": row["decision_id"],
        "scene_id": row["scene_id"],
        "case_id": row["case_id"],
        "ground_truth": "SECRET_GOLD",
        "reference_action": row["reference_action"],
        "reward_state": state,
    }


class Stage2CanonicalConversionTests(unittest.TestCase):
    def setUp(self):
        self.clean = decision("scene_A", 1, source="clean")
        self.fault = decision("scene_A", 2)
        self.fallback = decision("scene_B", 3, mode="contract_frozen")
        self.other = decision("scene_C", 2)
        self.decisions = [self.clean, self.fault, self.fallback, self.other]
        self.teachers = {row["decision_id"]: teacher(row) for row in self.decisions}
        self.splits = {"scene_A": "train", "scene_B": "val", "scene_C": "test"}

    def test_global_preserves_prompt_and_reward_state_join(self):
        rows, private, quality = convert(
            self.decisions, self.teachers, self.splits, reward_mode="global",
        )
        self.assertEqual(quality["selected_counts"], {
            "train": 1, "val": 0, "test": 1,
        })
        self.assertEqual(quality["excluded"]["clean_without_stage4_reward_state"], 1)
        self.assertEqual(quality["excluded"]["non_reviewed:contract_frozen"], 1)
        row = rows["train"][0]
        self.assertEqual(row["prompt"], self.fault["prompt"])
        self.assertEqual(row["extra_info"]["sample_id"],
                         "scene_A-fork-round-02-decision-round-02")
        self.assertEqual(row["extra_info"]["decision_id"], self.fault["decision_id"])
        self.assertEqual(private[0]["round_sample_id"], self.fault["case_id"])
        self.assertEqual(private[0]["state"],
                         self.teachers[self.fault["decision_id"]]["reward_state"])
        self.assertNotIn("SECRET_GOLD", str(row["prompt"]))
        self.assertNotIn("reward_state", str(row["prompt"]))

    def test_da_four_dimensional_mode_accepts_stage3_and_stage4(self):
        rows, private, quality = convert(
            self.decisions, self.teachers, self.splits, reward_mode="da",
        )
        self.assertEqual(quality["selected_by_source"], {
            "clean": 1, "untrusted": 2,
        })
        self.assertEqual(len(rows["train"]), 2)
        self.assertEqual(sample_id(self.clean), self.clean["decision_id"])
        self.assertEqual(len(private), 2)

    def test_include_fallback_is_explicit(self):
        rows, _, _ = convert(
            self.decisions, self.teachers, self.splits,
            reward_mode="global", include_fallback=True,
        )
        self.assertEqual(len(rows["val"]), 1)

    def test_missing_and_unscorable_state_not_silently_trained(self):
        teachers = {key: dict(value) for key, value in self.teachers.items()}
        teachers[self.fault["decision_id"]]["reward_state"] = None
        teachers[self.other["decision_id"]]["reward_state"] = {
            **teachers[self.other["decision_id"]]["reward_state"],
            "scorable": False,
        }
        with self.assertRaisesRegex(ValueError, "No reward-compatible"):
            convert(self.decisions, teachers, self.splits, reward_mode="global")

    def test_frozen_scene_splits_are_stable(self):
        one = make_splits(["scene_A", "scene_B", "scene_C", "scene_D"], seed=17)
        two = make_splits(["scene_D", "scene_C", "scene_B", "scene_A"], seed=17)
        self.assertEqual(one, two)
        self.assertEqual(set(one["scene_to_split"].values()), {"train", "val", "test"})

    def test_cross_split_identical_prompt_fails(self):
        dup = decision("scene_C", 3)
        dup["prompt"] = self.fault["prompt"]
        teachers = {**self.teachers, dup["decision_id"]: teacher(dup)}
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            convert([self.fault, dup], teachers, self.splits)

    def test_invalid_canonical_id_does_not_join_private_reward(self):
        broken = dict(self.fault, decision_id="stage4:wrong-case:round:02")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            sample_id(broken)


class Stage2CanonicalPreparationTests(unittest.TestCase):
    def test_dry_run_verifies_canonical_conversion_without_parquet(self):
        scenes = ["scene_A", "scene_B", "scene_C", "scene_D"]
        rows = [decision(scene, 2) for scene in scenes]
        labels = {row["decision_id"]: teacher(row) for row in rows}
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "unified"
            source.mkdir()
            (source / "manifest.json").write_text(
                '{"schema":"maestro.dataset.v1"}', encoding="utf-8",
            )
            output = Path(tmp) / "prepared"
            with patch(
                "train_maestro_8b.data.prepare_grpo.load_unified",
                return_value=(
                    rows, labels, scenes,
                    {"schema": "maestro.dataset.v1",
                     "base_file_sha256": {"decisions.jsonl": "fake"}},
                ),
            ), patch("train_maestro_8b.data.prepare_grpo.verify_private_context",
                     return_value={"file_sha256": {}}), patch(
                "train_maestro_8b.data.prepare_grpo.validate_selected_claims"
            ):
                summary = prepare(
                    source_dir=source, output_dir=output,
                    reward_mode="global", dry_run=True,
                )
            self.assertEqual(summary["schema"], "maestro.stage2_grpo_prepared.v2")
            self.assertEqual(sum(summary["counts"].values()), 4)
            self.assertFalse(output.exists())


    def test_parquet_round_trip_from_audited_unified_files(self):
        """Real canonical JSONL -> GRPO Parquet -> private state join."""
        import json
        import pyarrow.parquet as pq
        from environment.data_export.source_io import digest
        from train_maestro_8b.policy.grpo_runtime import load_dataset_rows
        from train_maestro_8b.policy.trl_reward_adapter import (
            load_private_states, select_scorable_rows,
        )

        scenes = ["scene_A", "scene_B", "scene_C", "scene_D"]
        decisions = [decision(scene, 2) for scene in scenes]
        labels = [teacher(row) for row in decisions]
        cases = []
        for row in decisions:
            scene = row["scene_id"]
            cases.append({
                "case_id": "clean:" + scene, "scene_id": scene, "kind": "clean",
            })
            cases.append({
                "case_id": row["case_id"], "scene_id": scene,
                "kind": "fault_injected",
                "recovery": {"globally_recovered": True},
                "fault": {"atoms": [{"kind": "fabricated_claim"}]},
            })
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "unified"
            (base / "private").mkdir(parents=True)
            for name, items in (
                ("cases.jsonl", cases),
                ("decisions.jsonl", decisions),
                ("private/decision_labels.jsonl", labels),
            ):
                (base / name).write_text(
                    "".join(json.dumps(item) + "\n" for item in items),
                    encoding="utf-8",
                )
            from environment.data_export.private_context import publish_private_context
            claim_path = root / "stage4_claims.jsonl"
            claim_records = [{
                "schema": "claim_source_dependency",
                "private_only": True,
                "task_id": row["scene_id"],
                "round_sample_id": row["case_id"],
                "main_can_act_from_round": row["decision_round"],
                "claim_nodes": [],
                "claim_states": {},
            } for row in decisions]
            claim_path.write_text(
                "".join(json.dumps(row) + "\n" for row in claim_records),
                encoding="utf-8",
            )
            source_manifest = {
                "schema": "maestro.dataset.v1",
                "base_file_sha256": {
                    name: digest(base / name)
                    for name in ("cases.jsonl", "decisions.jsonl",
                                 "private/decision_labels.jsonl")
                },
            }
            world = {
                name: {
                    "task_id": name, "question": "Question",
                    "macro_graph": {"tasks": [{"task_id": "A", "depends_on": []}]},
                    "world": {"sources": {}},
                }
                for name in scenes
            }
            publish_private_context(
                base, scenes=world, decisions=decisions,
                claim_path=claim_path, manifest=source_manifest,
            )
            (base / "manifest.json").write_text(
                json.dumps(source_manifest), encoding="utf-8",
            )
            # Neither source Stage1 nor Stage4 files are needed after export.
            claim_path.unlink()
            output = root / "prepared"
            arguments = {"source_dir": base, "output_dir": output,
                         "reward_mode": "global"}
            summary = prepare(**arguments)
            self.assertEqual(sum(summary["counts"].values()), 4)
            self.assertEqual(summary["private_state_rows"], 4)
            self.assertTrue((output / "private/scenes.jsonl").is_file())
            self.assertTrue((output / "private/claim_state_supervision.jsonl").is_file())
            self.assertEqual(digest(output / "private/scenes.jsonl"),
                             source_manifest["private_context"]["file_sha256"]["private/scenes.jsonl"])
            self.assertTrue(all(
                pq.read_metadata(output / (part + ".parquet")).num_rows ==
                summary["counts"][part]
                for part in ("train", "val", "test")
            ))
            private = load_private_states(
                output / "private/decision_reward_states.jsonl"
            )
            converted = load_dataset_rows(output / "train.parquet")
            compatible, report = select_scorable_rows(converted, private)
            self.assertEqual(len(compatible), summary["counts"]["train"])
            self.assertEqual(report["excluded"], {})
            self.assertTrue(prepare(**arguments, reuse_if_unchanged=True)["reused"])
            with self.assertRaisesRegex(FileExistsError, "different recipe"):
                prepare(**arguments, include_fallback=True)
            # Changing a frozen private World must fail, even though the
            # model-visible decision prompts are untouched.
            with (base / "private/scenes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "checksum_changed"):
                prepare(**arguments, dry_run=True)
            (base / "private/scenes.jsonl").write_text(
                (output / "private/scenes.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with (base / "decisions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"corrupted":true}\n')
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare(**arguments, dry_run=True)


if __name__ == "__main__":
    unittest.main()
