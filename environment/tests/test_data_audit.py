"""Unsplit, trainer-agnostic base data validation contracts."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from environment.data_export.audit import audit_dataset, audit_output
from environment.data_export.source_io import digest


def example():
    scenes = ("a", "b", "c")
    cases = [{
        "case_id": "clean:" + scene,
        "scene_id": scene,
        "kind": "clean",
    } for scene in scenes]
    cases.append({
        "case_id": "fault:a:1",
        "scene_id": "a",
        "kind": "fault_injected",
        "fault": {"atoms": [{"kind": "wrong_retrieval"}]},
        "recovery": {"decision_accepted": True, "globally_recovered": None},
    })
    decisions = []
    for scene in scenes:
        decisions.append({
            "decision_id": "stage3:" + scene,
            "case_id": "clean:" + scene,
            "scene_id": scene,
            "source": "clean",
            "decision_round": 1,
            "generation_mode": "llm_reviewed",
            "question": "Question " + scene,
            "visible_state": {"subtask_history": "History " + scene},
            "prompt": [{"role": "user", "content": "Prompt " + scene}],
            "reference_action": {
                "action": "complete",
                "reasoning": "Verified",
                "params": {"answer": scene},
            },
        })
    decisions.append({
        **deepcopy(decisions[0]),
        "decision_id": "stage4:a:1",
        "case_id": "fault:a:1",
        "source": "untrusted",
        "decision_round": 2,
        "prompt": [{"role": "user", "content": "Next-decision prompt a"}],
    })
    labels = [{
        "schema": "maestro.decision_teacher.v1",
        "decision_id": row["decision_id"],
        "case_id": row["case_id"],
        "scene_id": row["scene_id"],
        "ground_truth": row["scene_id"],
        "reference_action": deepcopy(row["reference_action"]),
        "reward_state": None,
    } for row in decisions]
    return cases, decisions, labels


class BaseAuditTests(unittest.TestCase):
    def test_unsplit_valid_case_and_teacher(self):
        report = audit_dataset(*example())
        self.assertEqual(report["case_count"], 4)
        self.assertEqual(report["decision_count"], 4)
        self.assertEqual(report["source_counts"]["untrusted"], 1)
        self.assertEqual(report["fault_decisions_by_kind"]["wrong_retrieval"], 1)

    def test_duplicate_same_prompt_same_label_reports_only(self):
        args = example()
        args[1][-1]["prompt"] = deepcopy(args[1][0]["prompt"])
        report = audit_dataset(*args)
        self.assertEqual(report["exact_duplicate_decision_rows"], 1)
        self.assertEqual(report["conflicting_prompt_groups"], 0)

    def test_conflicting_label_reports_in_base(self):
        args = example()
        args[1][-1]["prompt"] = deepcopy(args[1][0]["prompt"])
        args[1][-1]["reference_action"]["params"]["answer"] = "different"
        args[2][-1]["reference_action"] = deepcopy(
            args[1][-1]["reference_action"]
        )
        report = audit_dataset(*args)
        self.assertEqual(report["conflicting_prompt_groups"], 1)

    def test_split_field_is_forbidden_in_base(self):
        args = example()
        args[1][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "method_specific_split"):
            audit_dataset(*args)

    def test_teacher_action_mismatch_rejected(self):
        args = example()
        args[2][-1]["reference_action"]["params"]["answer"] = "wrong"
        with self.assertRaisesRegex(ValueError, "teacher_decision_alignment"):
            audit_dataset(*args)

    def test_unaccepted_branch_cannot_provide_positive_decision(self):
        args = example()
        args[0][-1]["recovery"]["decision_accepted"] = False
        with self.assertRaisesRegex(ValueError, "not_published_positive"):
            audit_dataset(*args)

    def test_persisted_source_manifest_hash(self):
        cases, decisions, labels = example()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "private").mkdir()
            rows = {
                "cases.jsonl": cases,
                "decisions.jsonl": decisions,
                "private/decision_labels.jsonl": labels,
            }
            for name, content in rows.items():
                (root / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in content),
                    encoding="utf-8",
                )
            hashes = {name: digest(root / name) for name in rows}
            (root / "manifest.json").write_text(json.dumps({
                "base_file_sha256": hashes,
            }), encoding="utf-8")
            self.assertEqual(
                audit_output(root, verify_manifest=True)["decision_count"], 4,
            )
            (root / "decisions.jsonl").write_text(
                (root / "decisions.jsonl").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checksum_changed"):
                audit_output(root, verify_manifest=True)


if __name__ == "__main__":
    unittest.main()
