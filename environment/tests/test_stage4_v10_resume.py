"""Interrupted v10 audit-fix migration and CPU worker regressions."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from environment.untrusted_orchestration.generate_turns import (
    _resume_v10_audit_fix, _atomic_json, _atomic_jsonl,
)


def plan(index: int, *, delayed: bool = False) -> dict:
    return {
        "index": index,
        "task_id": "q",
        "branch_round": index + 2,
        "sample_id": f"q-R{index + 2}-candidate-001",
        "scenario_plan": {
            "scenario_id": f"q-R{index + 2}-candidate-001",
            "blueprint": ("delayed_consumed_error_recovery"
                          if delayed else "immediate_feedback_fault"),
            "fault_atoms": [{
                "kind": "wrong_retrieval",
                "task_id": "parent",
                "origin_round": index + 1,
            }],
        },
    }


def signed(version: str) -> dict:
    return {
        "version": "maestro.untrusted_orchestration.event_engine_v1",
        "input_sha256": ["s1", "s2"],
        "settings": {"max_scenes": 10, "max_rounds": 16},
        "source_code_sha256": {
            "trajectory_builder.py": version,
            "generate_turns.py": version,
            "config.py": version,
            "runtime/faults.py": "unchanged",
        },
    }


def make_files(root: Path, plans: list[dict]) -> None:
    _atomic_jsonl(root / "round_index.jsonl", [{
        "round_sample_id": p["sample_id"],
        "question_id": p["task_id"],
        "main_decision_round": p["branch_round"],
        "scenario_blueprint": p["scenario_plan"]["blueprint"],
        "atomic_fault_kinds": [
            a["kind"] for a in p["scenario_plan"]["fault_atoms"]
        ],
    } for p in plans])
    for p in plans:
        _atomic_json(root / "stage4a" / "journal" /
                     f"{p['index']:06d}.json", {
            "task_id": p["task_id"],
            "round_sample_id": p["sample_id"],
            "fork_round": p["branch_round"],
            "status": "recovered",
            "private_audit": {
                "event_scenario": deepcopy(p["scenario_plan"]),
            },
        })


class V10ExplicitResumeTests(TestCase):
    def test_reuses_signed_nondelayed_and_replays_delayed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plans = [plan(0), plan(1, delayed=True)]
            make_files(root, plans)
            earlier, current = signed("old"), signed("new")
            report = _resume_v10_audit_fix(root, earlier, current, plans)
            self.assertEqual(report["reused_nondelayed_journals"], 1)
            self.assertEqual(report["discarded_delayed_journals"], 1)
            self.assertTrue((root / "stage4a/journal/000000.json").is_file())
            self.assertFalse((root / "stage4a/journal/000001.json").exists())
            self.assertEqual(json.loads(
                (root / "manifest.json").read_text()), current)
            self.assertEqual(json.loads(
                (root / "manifest.before_v10_audit_fix.json").read_text()
            ), earlier)
            self.assertTrue(
                (root / "private/v10_resume_audit.json").is_file()
            )

    def test_never_migrates_changed_inputs_or_fault_executor(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plans = [plan(0)]
            make_files(root, plans)
            old, current = signed("old"), signed("new")
            changed = deepcopy(current)
            changed["settings"]["max_rounds"] = 10
            with self.assertRaisesRegex(
                ValueError, "input_or_generation_settings_changed"
            ):
                _resume_v10_audit_fix(root, old, changed, plans)
            changed = deepcopy(current)
            changed["source_code_sha256"]["runtime/faults.py"] = "changed"
            with self.assertRaisesRegex(
                ValueError, "unsupported_code_changes"
            ):
                _resume_v10_audit_fix(root, old, changed, plans)
            self.assertFalse((root / "manifest.json").exists())

    def test_refuses_candidate_drift_or_existing_language_journal(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plans = [plan(0)]
            make_files(root, plans)
            old, new = signed("old"), signed("new")
            drift = deepcopy(plans)
            drift[0]["sample_id"] = "new-unmatched-candidate"
            with self.assertRaisesRegex(
                ValueError, "candidate_identity_changed"
            ):
                _resume_v10_audit_fix(root, old, new, drift)
            _atomic_json(root / "journal/000000.json", {"status": "recovered"})
            with self.assertRaisesRegex(
                ValueError, "language_journals_already_present"
            ):
                _resume_v10_audit_fix(root, old, new, plans)
            self.assertFalse((root / "manifest.json").exists())
