"""DA-SFT data join and split contracts; no GPU/model files required."""
import json
import unittest

from dataset.prepare_da_sft import build_sft_rows, comparable, group_split


class TestDAExport(unittest.TestCase):
    def setUp(self):
        self.scene = {"task_id": "scene-1", "question": "Find the matching item"}
        self.turn = {
            "sample_id": "scene-1-round-01", "task_id": "scene-1", "round": 1,
            "input": {"subtask_history": "No subtasks completed yet."},
            "output": {
                "action": "complete",
                "reasoning": "Evidence already identifies the single answer.",
                "params": {"answer": "42"},
            },
            "generation_mode": "llm_reviewed",
        }
        self.row = {
            "data_source": "maestro_stage3",
            "prompt": [{"role": "user", "content": "The exact O1 prompt"}],
            "reward_model": {"style": "rule", "ground_truth": "42"},
            "extra_info": {
                "sample_id": "scene-1-round-01",
                "question": "Find the matching item",
                "answer": "42",
                "subtask_history": "No subtasks completed yet.",
                "has_expert": True,
                "step_index": 1,
                "expert_action": "complete",
                # Arrow null-fills absent keys from heterogeneous structs.
                "expert_params": {"answer": "42", "tasks": None},
            },
        }
        self.turns = {("maestro_stage3", "scene-1-round-01"): self.turn}
        self.settings = {"stage3": {
            "max_attempts": 8, "sub_model": "qwen3-8b", "model_alias": "model_1",
        }}

    def convert(self, *, rows=None, turns=None, renderer=None, include_fallback=False):
        return build_sft_rows(
            rows if rows is not None else [self.row],
            turns if turns is not None else self.turns,
            {"scene-1": self.scene},
            self.settings,
            renderer=renderer or (lambda *args, **kwargs: "The exact O1 prompt"),
            include_fallback=include_fallback,
        )

    def test_preserves_full_reasoning_not_only_expert_params(self):
        examples, metadata, skipped = self.convert()
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["prompt"], self.row["prompt"])
        self.assertEqual(
            json.loads(examples[0]["completion"][0]["content"]),
            self.turn["output"],
        )
        self.assertEqual(metadata[0]["task_id"], "scene-1")
        self.assertFalse(skipped)

    def test_parquet_union_nulls_are_normalized(self):
        self.assertEqual(
            comparable({"tasks": None, "answer": "42"}),
            {"answer": "42"},
        )

    def test_exact_prompt_is_required(self):
        with self.assertRaisesRegex(ValueError, "Prompt drift"):
            self.convert(renderer=lambda *args, **kwargs: "Different O1 prompt")

    def test_missing_stage4_turn_fails_not_silently_skipped(self):
        stage4 = dict(self.row, data_source="maestro_stage4")
        with self.assertRaisesRegex(ValueError, "untrusted-turns"):
            self.convert(rows=[stage4])

    def test_original_expert_mismatch_fails(self):
        row = dict(self.row, extra_info={**self.row["extra_info"], "answer": "wrong"})
        with self.assertRaisesRegex(ValueError, "metadata drift"):
            self.convert(rows=[row])

    def test_mode_filter(self):
        self.turn["generation_mode"] = "teacher_grounded_completion_fallback"
        with self.assertRaisesRegex(ValueError, "All samples filtered"):
            self.convert()
        rows, _, _ = self.convert(include_fallback=True)
        self.assertEqual(len(rows), 1)

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate GRPO sample"):
            self.convert(rows=[self.row, self.row])

    def test_scene_group_split_includes_all_rounds_and_branches(self):
        rows = [{}, {}, {}, {}, {}]
        metadata = [
            {"task_id": "a"}, {"task_id": "a"},  # 2 Stage 3 rounds
            {"task_id": "a"},  # Stage 4 recovery branch
            {"task_id": "b"}, {"task_id": "c"},
        ]
        s1 = group_split(rows, metadata, 0.34, 42)
        s2 = group_split(rows, metadata, 0.34, 42)
        self.assertEqual(s1, s2)
        train_ids = {metadata[i]["task_id"] for i in s1["train"]}
        eval_ids = {metadata[i]["task_id"] for i in s1["eval"]}
        self.assertFalse(train_ids & eval_ids)
        self.assertEqual(set(s1["train"] + s1["eval"]), set(range(5)))
        with self.assertRaisesRegex(ValueError, "two different Scenes"):
            group_split([{}], [{"task_id": "a"}], 0.1, 42)


if __name__ == "__main__":
    unittest.main()
