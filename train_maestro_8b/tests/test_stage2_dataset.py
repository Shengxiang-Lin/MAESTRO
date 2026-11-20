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
            ), patch("train_maestro_8b.data.prepare_grpo.assert_global_sources"):
                summary = prepare(
                    source_dir=source, output_dir=output,
                    reward_mode="global", dry_run=True,
                )
            self.assertEqual(summary["schema"], "maestro.stage2_grpo_prepared.v2")
            self.assertEqual(sum(summary["counts"].values()), 4)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
