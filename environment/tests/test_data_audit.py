"""Deterministic audit for Scene splits, duplicate prompts and SFT adapters."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.data_export.audit import audit_dataset


def data():
    mapping = {"a": "train", "b": "val", "c": "test"}
    cases = [{
        "case_id": "clean:" + scene, "scene_id": scene,
        "split": split, "kind": "clean",
    } for scene, split in mapping.items()]
    cases.append({
        "case_id": "fault:a:1", "scene_id": "a", "split": "train",
        "kind": "fault_injected",
        "fault": {"atoms": [{"kind": "wrong_retrieval"}]},
        "recovery": {"globally_recovered": True},
    })
    decisions = []
    for scene, split in mapping.items():
        decisions.append({
            "decision_id": "stage3:" + scene,
            "case_id": "clean:" + scene, "scene_id": scene,
            "split": split, "source": "clean", "decision_round": 1,
            "generation_mode": "llm_reviewed",
            "prompt": [{"role": "user", "content": "Question " + scene}],
            "reference_action": {
                "action": "complete", "reasoning": "Evidence checked",
                "params": {"answer": scene},
            },
        })
    decisions.append({
        **deepcopy(decisions[0]),
        "decision_id": "stage4:a:1", "case_id": "fault:a:1",
        "source": "untrusted", "decision_round": 2,
        "prompt": [{"role": "user", "content": "Recovery context a"}],
    })
    adapters = [(row["split"], {
        "prompt": deepcopy(row["prompt"]),
        "completion": [{"role": "assistant", "content": json.dumps(
            row["reference_action"], ensure_ascii=False
        )}],
    }) for row in decisions]
    return cases, decisions, adapters, mapping


class DataAuditTests(unittest.TestCase):
    def test_valid_data_and_fault_distribution(self):
        report = audit_dataset(*data())
        self.assertEqual(report["decision_split_counts"], {
            "train": 2, "val": 1, "test": 1,
        })
        self.assertEqual(report["source_counts"]["untrusted"], 1)
        self.assertEqual(report["fault_decisions_by_kind"]["wrong_retrieval"], 1)
        self.assertEqual(report["cross_split_prompt_groups"], 0)

    def test_duplicated_prompt_with_same_label_only_warns(self):
        args = data()
        args[1][3]["prompt"] = deepcopy(args[1][0]["prompt"])
        args[2][3][1]["prompt"] = deepcopy(args[1][0]["prompt"])
        report = audit_dataset(*args)
        self.assertEqual(report["exact_duplicate_decision_rows"], 1)
        self.assertEqual(report["conflicting_prompt_groups"], 0)

    def test_conflicting_label_optional_strict_mode(self):
        args = data()
        args[1][3]["prompt"] = deepcopy(args[1][0]["prompt"])
        args[2][3][1]["prompt"] = deepcopy(args[1][0]["prompt"])
        args[1][3]["reference_action"]["params"]["answer"] = "different"
        args[2][3][1]["completion"][0]["content"] = json.dumps(
            args[1][3]["reference_action"]
        )
        report = audit_dataset(*args)
        self.assertEqual(report["conflicting_prompt_groups"], 1)
        with self.assertRaisesRegex(ValueError, "conflicting_labels"):
            audit_dataset(*args, strict_conflicts=True)

    def test_cross_split_prompt_collision_fails(self):
        args = data()
        args[1][1]["prompt"] = deepcopy(args[1][0]["prompt"])
        args[2][1][1]["prompt"] = deepcopy(args[1][0]["prompt"])
        with self.assertRaisesRegex(ValueError, "cross_split_identical_prompts"):
            audit_dataset(*args)

    def test_scene_split_and_adapter_drift_fail(self):
        args = data()
        args[1][3]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "decision_scene_split_mismatch"):
            audit_dataset(*args)
        args = data()
        args[2][3][1]["completion"][0]["content"] = "{}"
        with self.assertRaisesRegex(ValueError, "adapter_completion_drift"):
            audit_dataset(*args)

    def test_unpublished_stage4_cannot_be_positive(self):
        args = data()
        args[0][3]["recovery"]["globally_recovered"] = False
        with self.assertRaisesRegex(ValueError, "unpublished_or_mismatched_case"):
            audit_dataset(*args)


if __name__ == "__main__":
    unittest.main()
