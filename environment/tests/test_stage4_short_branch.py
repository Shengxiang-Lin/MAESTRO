"""Short-branch Stage-4 sampling and evidence-only diagnostic regressions."""
from __future__ import annotations

from copy import deepcopy
import unittest

from environment.untrusted_orchestration.round_sampling import (
    select_feedback_rounds,
)
from environment.untrusted_orchestration.candidate_selection import (
    select_recovered_candidates, classify_trial,
)
from environment.untrusted_orchestration.config import load_config
from environment.untrusted_orchestration.faults import Fault
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.tests.test_untrusted_orchestration import sample


def _trial(index, round_id, kind):
    return {
        "index": index,
        "sample_id": f"Q-fork-{round_id}-candidate-{index}",
        "task_id": "Q",
        "branch_round": round_id,
        "incoming_task_ids": [],
        "faults": [],
        "scenario_plan": {"fault_atoms": [{"kind": kind}]},
    }


def _journal(outcome, *, invalidation=False, verify=False):
    return {
        "status": outcome,
        "episode": {
            "trajectory": [{"attempt": 1, "action": "delegate_task"}],
        },
        "private_audit": {
            "execution_outcome": outcome,
            "runtime": {
                "invalidations": [{"task": "C"}] if invalidation else [],
                "detected_conflicts": [{"field": "id"}] if verify else [],
                "revalidations": [],
            },
            "fault_events_applied": [{"kind": "one"}],
            "evidence_plan": [{
                "evidence_requests": [{
                    "operation": (
                        "verify_conflicting_evidence" if verify
                        else "retry_missing_evidence"
                    ),
                    "target_fields": ["identifier"],
                    "authorized_source_ids": ["source_A"],
                }]
            }],
        },
    }


class StratifiedForkTests(unittest.TestCase):
    def test_round_sampling_is_seeded_and_covers_position_buckets(self):
        rounds = [(idx + 2, {"round": idx + 2}) for idx in range(9)]
        selected = select_feedback_rounds(
            rounds, limit=3, strategy="stratified", seed=17
        )
        self.assertEqual(selected, select_feedback_rounds(
            rounds, limit=3, strategy="stratified", seed=17
        ))
        self.assertEqual(len(selected), 3)
        self.assertIn(selected[0], rounds[:3])
        self.assertIn(selected[1], rounds[3:6])
        self.assertIn(selected[2], rounds[6:])
        self.assertEqual(
            select_feedback_rounds(
                rounds, limit=3, strategy="chronological", seed=17
            ),
            rounds[:3],
        )
        self.assertEqual(
            select_feedback_rounds(
                rounds, limit=None, strategy="stratified", seed=17
            ),
            rounds,
        )

    def test_evidence_aware_sampling_prefers_frozen_verifier_positions(self):
        rounds = [(i + 2, {"attempt": i + 1}) for i in range(6)]
        graph = {"reference_waves": [
            ["A"], ["B"], ["C"], ["D"], ["E"], ["F"],
        ], "tasks": [
            {"task_id": tid, "depends_on": [],
             "source_ids": (["alpha", "beta"] if tid == "E" else ["alpha"]),
             "macro_role": ("verification" if tid == "B" else "investigation"),
             "operator_chain": (
                 [{"op": "join"}] if tid == "E" else []
             )}
            for tid in "ABCDEF"
        ]}
        scene = {"macro_graph": graph}
        result = select_feedback_rounds(
            rounds, limit=2, strategy="evidence_aware", seed=17,
            scene=scene,
        )
        self.assertEqual([row[0] for row in result], [3, 6])
        self.assertEqual(result, select_feedback_rounds(
            rounds, limit=2, strategy="evidence_aware", seed=17,
            scene=scene,
        ))
        with self.assertRaisesRegex(
            ValueError, "evidence_aware_sampling_requires_scene"
        ):
            select_feedback_rounds(
                rounds, limit=2, strategy="evidence_aware", seed=17,
            )

    def test_observability_comes_from_report_snapshots_not_fault_kind(self):
        plan = _trial(0, 2, "wrong_retrieval")
        plan["faults"] = [Fault("A", "wrong")]
        journal = _journal("incorrect_result_after_actual_propagation")
        journal["private_audit"]["observable_issue_snapshots"] = [
            {"decision_round": 2, "issues": [{
                "task_id": "A", "kind": "lookup_key_disagreement",
                "evidence": "Explicitly returned keys differ.",
            }]},
        ]
        row = classify_trial(plan, journal)
        self.assertEqual(row["observability"], "immediately_observable")
        self.assertEqual(row["first_visible_issue_round"], 2)
        journal["private_audit"]["observable_issue_snapshots"][0][
            "issues"] = []
        journal["private_audit"]["observable_issue_snapshots"].append({
            "decision_round": 4, "issues": [{
                "task_id": "A", "kind": "cross_source_conflict",
                "evidence": "A later cited report disagrees.",
            }],
        })
        row = classify_trial(plan, journal)
        self.assertEqual(row["observability"], "eventually_observable")
        self.assertEqual(row["first_visible_issue_round"], 4)
        journal["private_audit"]["observable_issue_snapshots"][1][
            "issues"] = []
        row = classify_trial(plan, journal)
        self.assertEqual(row["observability"], "not_observed_in_replay")
        self.assertIsNone(row["first_visible_issue_round"])

    def test_balances_recovered_behaviors_across_rounds(self):
        plans = [
            _trial(0, 2, "wrong_retrieval"),
            _trial(1, 2, "incomplete_retrieval"),
            _trial(2, 3, "wrong_retrieval"),
            _trial(3, 3, "incomplete_retrieval"),
        ]
        journals = [
            _journal("recovered", invalidation=True),
            _journal("recovered"),
            _journal("recovered", invalidation=True),
            _journal("recovered"),
        ]
        picked, rows, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=1,
            strategy="stratified", seed=17,
        )
        self.assertEqual(picked, {0, 3})
        self.assertEqual(
            {rows[idx]["selection_purpose"] for idx in picked},
            {"positive"},
        )
        self.assertEqual(
            stats["candidates_selected_for_language_review"], 2,
        )

    def test_failed_execution_can_be_reviewed_without_positive_sft(self):
        plans = [
            _trial(0, 2, "no_result"),
            _trial(1, 2, "wrong_retrieval"),
        ]
        journals = [
            _journal("recovered"),
            _journal("incorrect_result_after_actual_propagation"),
        ]
        picked, rows, stats = select_recovered_candidates(
            plans, journals, max_selected_per_round=1,
            max_diagnostics_per_scene=1,
        )
        self.assertEqual(picked, {0, 1})
        self.assertEqual(rows[0]["selection_purpose"], "positive")
        self.assertEqual(rows[1]["selection_purpose"], "diagnostic")
        self.assertIsNone(rows[1]["recovery_rank"])
        self.assertEqual(stats["diagnostics_selected_for_language_review"], 1)

    def test_canonical_pilot_enables_short_stratified_replay(self):
        cfg = load_config()
        self.assertEqual(cfg["round_sampling_strategy"], "evidence_aware")
        self.assertEqual(cfg["branches_per_scene"], 3)
        self.assertEqual(cfg["max_post_fork_rounds"], 8)
        self.assertEqual(cfg["max_diagnostics_per_scene"], 1)


class ShortReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, cls.reports, cls.clean = sample(0)

    def test_horizon_expiry_is_never_successful_recovery(self):
        target = self.scene["macro_graph"]["reference_waves"][0][0]
        episode, turns, audit = build_branch(
            self.scene, deepcopy(self.reports), self.clean,
            branch_round=2,
            faults=[Fault(target, "missing")],
            max_post_fork_rounds=1,
        )
        self.assertEqual(audit["execution_outcome"], "short_replay_horizon")
        self.assertEqual(audit["outcome"], "short_replay_horizon")
        self.assertEqual(turns, [])
        self.assertFalse(episode["recovered"])

    def test_horizon_must_be_positive(self):
        target = self.scene["macro_graph"]["reference_waves"][0][0]
        with self.assertRaisesRegex(ValueError, "max_post_fork_rounds"):
            build_branch(
                self.scene, self.reports, self.clean,
                branch_round=2, faults=[Fault(target, "missing")],
                max_post_fork_rounds=0,
            )


if __name__ == "__main__":
    unittest.main()
