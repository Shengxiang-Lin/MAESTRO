"""Regression test: fault candidates sharing the same round stay distinct."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from train_maestro_8b.export_reliability import index_states
from train_maestro_8b.export_grpo import export
from train_maestro_8b.risk_schema import VERSION, iter_jsonl


class CandidateAlignmentTests(unittest.TestCase):
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
                        "sample_id": "task-1-branch-02-round-02",
                        "round_sample_id": cid,
                        "task_id": "task-1", "round": 2,
                        "output": {"action": "complete",
                                   "params": {"answer": answer}},
                    }) + "\n")
            output_dir = folder / "grpo"
            summary = export(risk_dir, turns, output_dir)
            self.assertEqual(summary["train"], 2)
            rows = list(iter_jsonl(output_dir / "train.jsonl"))
            mapping = {
                row["sample_id"].split("-decision-")[0]:
                row["reference_action"]["params"]["answer"]
                for row in rows
            }
            self.assertEqual(mapping, {a:"first", b:"second"})


if __name__ == "__main__":
    unittest.main()
