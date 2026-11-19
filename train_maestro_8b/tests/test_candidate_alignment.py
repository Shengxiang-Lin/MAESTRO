"""Regression test: fault candidates sharing the same round stay distinct."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from train_maestro_8b.data.export_risk import index_states
from train_maestro_8b.data.export_policy import export, reference_key
from train_maestro_8b.data.contracts import VERSION, iter_jsonl
from environment.untrusted_orchestration.rewards.decision_reward_state import SCHEMA


class CandidateAlignmentTests(unittest.TestCase):
    def test_main_turn_id_is_not_a_provenance_contract(self):
        candidate = "task-[7].1-fork-round-02-candidate-004"
        structured = {
            "task_id": "task-[7].1",
            "round": 5,
            "round_sample_id": candidate,
            # Historic and new Stage4 turn ids may have different formats.
            "sample_id": "reviewed-main-turn-abc-no-round-suffix",
        }
        self.assertEqual(
            reference_key(structured),
            ("task-[7].1", candidate, 5),
        )
        self.assertEqual(
            reference_key({**structured, "sample_id": None}),
            ("task-[7].1", candidate, 5),
        )

    def test_missing_candidate_identity_does_not_guess(self):
        valid = {
            "task_id": "task-1",
            "round": 2,
            "round_sample_id": "task-1-fork-round-02-candidate-001",
        }
        with self.assertRaisesRegex(ValueError, "round_sample_id"):
            reference_key({**valid, "round_sample_id": ""})
        with self.assertRaisesRegex(ValueError, "valid integer round"):
            reference_key({**valid, "round": "2"})
        with self.assertRaisesRegex(ValueError, "valid integer round"):
            reference_key({**valid, "round": True})
        with self.assertRaisesRegex(ValueError, "valid task_id"):
            reference_key({**valid, "task_id": ""})

    def test_same_scene_fork_multiple_candidates_are_not_mixed(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            a = "task-1-fork-round-02-candidate-001"
            b = "task-1-fork-round-02-candidate-002"
            states = folder / "claim_state_supervision.jsonl"
            states.write_text(
                "".join(json.dumps({
                    "task_id": "task-1",
                    "round_sample_id": cid,
                    "main_can_act_from_round": 2,
                }) + "\n" for cid in [a, b]), encoding="utf-8",
            )
            joined = index_states(states)
            self.assertEqual(len(joined), 2)
            self.assertIn(("task-1", a), joined)
            self.assertIn(("task-1", b), joined)

            risk_dir = folder / "risk"
            risk_dir.mkdir()
            for cid in (a, b):
                row = {
                    "schema": VERSION,
                    "sample_id": cid + "-decision-02",
                    "scene_id": "task-1", "branch_round": 2,
                    "round_sample_id": cid, "decision_round": 2,
                    "prompt": "history a" if cid == a else "history b",
                    "reports": [{"report_id":"x", "start":0, "end":7,
                                 "risk":1, "mask":1}],
                }
                with (risk_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            turns = folder / "main_turns.jsonl"
            with turns.open("w", encoding="utf-8") as handle:
                for cid, answer in [(a,"first"),(b,"second")]:
                    handle.write(json.dumps({
                        "sample_id": f"reviewed-main-turn-{answer}",
                        "round_sample_id": cid,
                        "task_id": "task-1", "round": 2,
                        "output": {"action": "complete",
                                   "params": {"answer": answer}},
                    }) + "\n")
            private_states = folder / "decision_reward_states.jsonl"
            with private_states.open("w", encoding="utf-8") as handle:
                for cid in [a, b]:
                    handle.write(json.dumps({
                        "task_id": "task-1", "round_sample_id": cid,
                        "decision_round": 2,
                        "state": {
                            "schema": SCHEMA, "decision_round":2,
                            "max_parallel_tasks":4, "scorable":True,
                            "can_complete":True, "actionable":[],
                            "event_pending":False,
                        },
                    }) + "\n")
            output_dir = folder / "grpo"
            summary = export(risk_dir, turns, output_dir, private_states)
            self.assertEqual(summary["train"], 2)
            self.assertEqual(summary["missing_decision_state"], 0)
            rows = list(iter_jsonl(output_dir / "train.jsonl"))
            mapping = {
                row["sample_id"].split("-decision-")[0]:
                row["reference_action"]["params"]["answer"]
                for row in rows
            }
            self.assertEqual(mapping, {a:"first", b:"second"})
            self.assertTrue(all(row["decision_state"]["can_complete"]
                                for row in rows))


if __name__ == "__main__":
    unittest.main()
