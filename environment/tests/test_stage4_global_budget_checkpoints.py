"""Global Stage4 budget and per-item 4B/4C crash-safe replay."""
from __future__ import annotations

from copy import deepcopy
import unittest

from environment.untrusted_orchestration.scenario.candidate_selection import (
    select_recovered_candidates,
)
from environment.untrusted_orchestration.offline_pipeline import (
    author_branch_reports, author_branch_main,
)
from environment.untrusted_orchestration.runtime.faults import Fault
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.tests.test_stage4_short_branch import _trial, _journal
from environment.tests.test_untrusted_orchestration import sample


class GlobalTurnBudgetTests(unittest.TestCase):
    def test_budget_is_main_turns_not_episodes_and_balances_real_kinds(self):
        plans = [
            _trial(0, 2, "no_result"),
            _trial(1, 3, "wrong_retrieval"),
            _trial(2, 4, "incomplete_retrieval"),
        ]
        journals = [_journal("recovered") for _ in plans]
        for item in journals:
            item["episode"]["trajectory"] = [
                {"attempt": 5, "action": "delegate_task"},
                {"attempt": 6, "action": "complete"},
            ]
        picked, rows, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=None,
            target_main_turns=4, strategy="global_balanced", seed=17,
        )
        self.assertEqual(len(picked), 2)
        self.assertEqual(stats["estimated_selected_main_turns"], 4)
        self.assertEqual(len(stats["selected_recovered_by_fault_kind"]), 2)
        self.assertEqual(
            {row["selection_purpose"] for row in rows if row["candidate_index"]
             not in picked}, {"budget_reserve"},
        )
        # No valid Episode should be silently truncated to fill a quota.
        undersized, _, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=None,
            target_main_turns=1, strategy="global_balanced",
        )
        self.assertFalse(undersized)
        self.assertEqual(stats["estimated_selected_main_turns"], 0)

    def test_target_budget_rejects_invalid_sizes(self):
        with self.assertRaisesRegex(ValueError, "global_main_turn_budget"):
            select_recovered_candidates(
                [], [], max_selected_per_round=None,
                target_main_turns=0,
            )


class Stage4CheckpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.base = sample(0)
        cls.target = cls.scene["macro_graph"]["reference_waves"][0][0]
        cls.episode, _, cls.audit = build_branch(
            cls.scene, cls.reports, cls.base,
            branch_round=2, faults=[Fault(cls.target, "missing")],
        )

    def test_report_checkpoint_replay_matches_uncached(self):
        saved, ticks = [], []
        episode, audit = author_branch_reports(
            self.scene, self.reports, self.episode, self.audit,
            on_report_checkpoint=lambda row: saved.append(deepcopy(row)),
            on_report=lambda status: ticks.append(deepcopy(status)),
        )
        self.assertTrue(saved)
        self.assertEqual(len(saved), len(ticks))
        replay_ep, replay_audit = author_branch_reports(
            self.scene, self.reports, self.episode, self.audit,
            resume_reports=saved,
            on_report_checkpoint=lambda row: self.fail(
                "Checkpoint replay generated a new report"
            ),
        )
        self.assertEqual(episode, replay_ep)
        self.assertEqual(audit, replay_audit)
        bad = deepcopy(saved)
        bad[0]["task_id"] = "different-task"
        with self.assertRaisesRegex(
            ValueError, "stage4b_checkpoint_report_alignment_changed"
        ):
            author_branch_reports(
                self.scene, self.reports, self.episode, self.audit,
                resume_reports=bad,
            )

    def test_main_checkpoint_replay_matches_uncached(self):
        reviewed_ep, reviewed_audit = author_branch_reports(
            self.scene, self.reports, self.episode, self.audit,
        )
        saved = []
        output = author_branch_main(
            self.scene, reviewed_ep, reviewed_audit,
            on_main_checkpoint=lambda row: saved.append(deepcopy(row)),
        )
        self.assertTrue(saved)
        replay = author_branch_main(
            self.scene, reviewed_ep, reviewed_audit,
            resume_actions=saved,
            on_main_checkpoint=lambda row: self.fail(
                "Checkpoint replay generated a new Main action"
            ),
        )
        self.assertEqual(output, replay)
        bad = deepcopy(saved)
        bad[0]["prompt_sha256"] = "wrong"
        with self.assertRaisesRegex(
            ValueError, "stage4c_checkpoint_.*_drift"
        ):
            author_branch_main(
                self.scene, reviewed_ep, reviewed_audit,
                resume_actions=bad,
            )


if __name__ == "__main__":
    unittest.main()
