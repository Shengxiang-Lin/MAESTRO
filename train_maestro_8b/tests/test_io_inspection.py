"""Inspect that DA-aligned prompt/output audits are exact and privacy-safe."""
import unittest

from train_maestro_8b.data.inspect_io import inspect


class TinyTokenizer:
    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        return {"input_ids": list(range(len(text)))}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return list(range(len(messages[0]["content"]) + 3))


class IOAuditTests(unittest.TestCase):
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
