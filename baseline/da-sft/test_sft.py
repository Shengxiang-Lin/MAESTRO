"""DA-SFT trainer config, token budget and loss mask preflight (CPU only)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("da_sft_training", HERE / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1:
            return [10, 11, 12]
        return [10, 11, 12, 13, 14, 15]


class TestConfig(unittest.TestCase):
    def test_defaults_no_vllm_and_completion_only_loss(self):
        cfg = train.load_config()
        args = train.sft_kwargs(cfg)
        self.assertTrue(args["completion_only_loss"])
        self.assertFalse(args["assistant_only_loss"])
        self.assertFalse(args["packing"])
        self.assertNotIn("use_vllm", args)
        self.assertNotIn("beta", args)
        self.assertEqual(cfg["model"]["max_seq_length"], 16384)
        self.assertEqual(cfg["model"]["eos_token"], "<|im_end|>")

    def test_token_budget_checks_whole_dialogue(self):
        row = {
            "prompt": [{"role": "user", "content": "Original state"}],
            "completion": [{"role": "assistant", "content": json.dumps({
                "action": "complete", "reasoning": "Done", "params": {"answer": "42"},
            })}],
        }
        summary = train.check_lengths([row], FakeTokenizer(), 3, 6, "train")
        self.assertEqual(summary["max_prompt_tokens"], 3)
        self.assertEqual(summary["max_total_tokens"], 6)
        with self.assertRaisesRegex(ValueError, "overlong SFT"):
            train.check_lengths([row], FakeTokenizer(), 3, 5, "train")
        with self.assertRaisesRegex(ValueError, "overlong SFT"):
            train.check_lengths([row], FakeTokenizer(), 2, 6, "train")

    def test_mask_inspection_uses_real_trainer_dataset(self):
        t = SimpleNamespace(train_dataset=[{
            "completion_mask": [0, 0, 0, 1, 1],
        }])
        result = train.check_mask(t)
        self.assertEqual(result["samples"][0], {
            "prompt_masked": 3, "supervised": 2,
        })
        with self.assertRaisesRegex(RuntimeError, "mask"):
            train.check_mask(SimpleNamespace(train_dataset=[{
                "completion_mask": [1, 1, 1],
            }]))
        with self.assertRaisesRegex(RuntimeError, "did not expose"):
            train.check_mask(SimpleNamespace(train_dataset=[{
                "input_ids": [10, 11, 12],
            }]))

    def test_row_shape_and_json_validation(self):
        import importlib.util
        if importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pyarrow not installed in unit test environment")
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.parquet"
            example = {
                "prompt": [{"role": "user", "content": "Query"}],
                "completion": [{"role": "assistant", "content": json.dumps({
                    "action": "complete", "reasoning": "Enough information",
                    "params": {"answer": "42"},
                })}],
            }
            pq.write_table(pa.Table.from_pylist([example]), path)
            self.assertEqual(train.load_rows(path), [example])


if __name__ == "__main__":
    unittest.main()
