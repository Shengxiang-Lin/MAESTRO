"""Inspect that DA-aligned prompt/output audits are exact and privacy-safe."""
import unittest
from collections import UserDict

from train_maestro_8b.data.inspect_io import inspect, _ids


class TinyTokenizer:
    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        return {"input_ids": list(range(len(text)))}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return list(range(len(messages[0]["content"]) + 3))



class TensorLike:
    """Torch/NumPy-like token container without a CUDA dependency."""
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class BatchEncodingTokenizer(TinyTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = super().apply_chat_template(
            messages, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return UserDict({
            "input_ids": TensorLike([ids]),
            "attention_mask": TensorLike([[1] * len(ids)]),
        })


class IOAuditTests(unittest.TestCase):
    def test_chat_template_batchencoding_and_tensor_like_are_supported(self):
        row = {
            "sample_id": "batchencoding",
            "prompt": "Main question with Sub-Agent reports",
            "reports": [{"start": 5, "end": 22}],
            "reference_action": {"action": "complete", "params": {"answer": "ok"}},
            "decision_state": {"scorable": True},
        }
        result = inspect([row], BatchEncodingTokenizer())
        self.assertEqual(
            result["prompt_da_chat_template_tokens"]["max"],
            len(row["prompt"]) + 3,
        )
        self.assertEqual(result["prompt_raw_tokens"]["max"], len(row["prompt"]))

    def test_token_ids_normalizer_accepts_multiple_single_example_shapes(self):
        ids = [2, 3, 5, 7]
        self.assertEqual(_ids(ids), ids)
        self.assertEqual(_ids(tuple(ids)), ids)
        self.assertEqual(_ids(TensorLike(ids)), ids)
        self.assertEqual(_ids(TensorLike([ids])), ids)
        self.assertEqual(_ids(UserDict({"input_ids": TensorLike([ids])})), ids)

    def test_token_ids_normalizer_rejects_accidental_multirow_batches(self):
        with self.assertRaisesRegex(ValueError, "batch of 2"):
            _ids(TensorLike([[1, 2], [3, 4]]))
        with self.assertRaisesRegex(TypeError, "without input_ids"):
            _ids(UserDict({"attention_mask": [1, 1]}))
        with self.assertRaisesRegex(TypeError, "unsupported"):
            _ids("not token IDs")
        with self.assertRaisesRegex(TypeError, "non-integer"):
            _ids([1, True, 3])

    def test_raw_input_reference_and_private_reward_separation(self):
        row = {
            "sample_id": "demo-01",
            "prompt": "Question plus cumulative Sub-Agent feedback.",
            "reports": [{"start": 14, "end": 40, "report_id": "1:0:A"}],
            "reference_action": {"action": "delegate_task", "params": {"tasks": []}},
            "decision_state": {"PRIVATE_SHOULD_NOT_PRINT": "SECRET-GOLD-LABEL"},
        }
        result = inspect(
            [row], TinyTokenizer(), max_prompt_tokens=12288,
            max_completion_tokens=4096, num_generations=8,
        )
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["candidate_group_size"], 8)
        self.assertEqual(result["completion_token_cap"], 4096)
        self.assertEqual(result["prompt_raw_tokens"]["max"], len(row["prompt"]))
        self.assertEqual(
            result["prompt_da_chat_template_tokens"]["max"],
            len(row["prompt"]) + 3,
        )
        self.assertNotIn("SECRET-GOLD-LABEL", str(result))
        self.assertEqual(
            result["reference_action_types"]["delegate_task"], 1
        )

    def test_overlong_and_reference_completion_are_reported_not_truncated(self):
        row = {
            "sample_id": "long",
            "prompt": "X" * 20,
            "reports": [],
            "reference_action": {"action": "complete", "params": {"answer":"a"*40}},
            "decision_state": {"scorable": True},
        }
        result = inspect(
            [row], TinyTokenizer(), max_prompt_tokens=12,
            max_completion_tokens=10,
        )
        self.assertEqual(result["overlong_prompt_count"], 1)
        self.assertEqual(result["overlong_reference_count"], 1)
        self.assertEqual(result["examples"][0]["raw_prompt_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
