"""Round-local Main decision publication and anti-oracle regressions."""
from __future__ import annotations

from copy import deepcopy
import unittest

from environment.untrusted_orchestration.runtime.faults import Fault
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.offline_pipeline import (
    author_branch_reports, author_branch_main,
)
from environment.untrusted_orchestration.scenario.single_turn_selection import (
    select_single_turn_candidates,
)
from environment.tests.test_untrusted_orchestration import sample


class SingleTurnBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.clean = sample(0)

    def test_fork_emits_one_main_decision_without_executing_future_reports(self):
        task = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, turns, audit = build_branch(
            self.scene, self.reports, self.clean, branch_round=2,
            faults=[Fault(task, "missing")], single_turn=True,
        )
        self.assertEqual(audit["outcome"], "single_turn_planned")
        self.assertEqual(audit["decision_unit"], "single_turn")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["round"], 2)
        self.assertEqual(len(audit["evidence_plan"]), 1)
        self.assertTrue(audit["parallel_batches_verified"])
        self.assertIn("no result", turns[0]["input"]["subtask_history"])
        self.assertEqual(len(audit["runtime"]["attempt_versions"][task]), 1)
        self.assertEqual(len(episode["trajectory"]), 2)
        current = episode["trajectory"][-1]
        self.assertEqual(current["action"], "delegate_task")
        self.assertTrue(current["result"]["pending"])
        self.assertEqual(current["result"]["subtask_results"], [])
        self.assertFalse(episode["recovered"])

    def test_stage4b_c_publish_one_turn_without_terminal_gold_recovery(self):
        task = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, _, audit = build_branch(
            self.scene, self.reports, self.clean, branch_round=2,
            faults=[Fault(task, "missing")], single_turn=True,
        )
        authored_episode, authored_audit = author_branch_reports(
            self.scene, self.reports, episode, audit,
        )
        final_episode, turns, final_audit = author_branch_main(
            self.scene, authored_episode, authored_audit,
        )
        self.assertEqual(final_audit["outcome"], "single_turn_accepted")
        self.assertEqual(len(turns), 1)
        self.assertEqual(final_episode["trajectory"][-1]["result"]["subtask_results"], [])
        self.assertEqual(turns[0]["round"], 2)


def trial(i, kind="wrong_retrieval"):
    return {"index": i, "task_id": "Q", "branch_round": 2,
            "scenario_plan": {"fault_atoms": [{"kind": kind}]}}


def journal(issue="Different returned keys", *, status="single_turn_planned"):
    return {"status": status,
            "episode": {"trajectory": [{
                "attempt": 2, "action": "delegate_task",
                "result": {"pending": True, "subtask_results": []},
            }]},
            "private_audit": {
                "parallel_batches_verified": True,
                "observable_issue_snapshots": [{
                    "decision_round": 2,
                    "issues": ([] if issue is None else [{
                        "reported_round": 1,
                        "task_id": "A",
                        "kind": "lookup_key_disagreement",
                        "evidence": issue,
                    }]),
                }],
                "runtime": {"detected_conflicts": []},
                "evidence_plan": [{
                    "evidence_requests": [{
                        "operation": "verify_conflicting_evidence",
                        "target_fields": ["entity_id"],
                    }]
                }],
            }}


class SingleTurnSelectionTests(unittest.TestCase):
    def test_selects_distinct_publicly_observable_cases_not_future_recovery(self):
        planned = [trial(i) for i in range(4)]
        results = [
            journal("Different returned keys"),
            journal("Different returned keys"),
            journal("Different returned years"),
            journal(None),
        ]
        selected, rows, stats = select_single_turn_candidates(
            planned, results, target_main_turns=3,
            max_selected_per_round=None, seed=17,
        )
        self.assertEqual(len(selected), 2)
        self.assertNotIn(3, selected)
        self.assertEqual(rows[1]["selection_reason"], "duplicate_public_issue_and_action")
        self.assertEqual(stats["single_turn_actionable"], 2)

    def test_budget_one_and_missing_reports_are_separate_faults(self):
        planned = [trial(0, "no_result"), trial(1, "result_summary_conflict")]
        results = [
            journal("No result returned"),
            journal("Result and Summary disagree"),
        ]
        selected, rows, stats = select_single_turn_candidates(
            planned, results, target_main_turns=1,
            max_selected_per_round=None, seed=17,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(stats["estimated_selected_main_turns"], 1)
        self.assertEqual(
            [r["selection_purpose"] for r in rows].count("budget_reserve"), 1
        )


if __name__ == "__main__":
    unittest.main()
