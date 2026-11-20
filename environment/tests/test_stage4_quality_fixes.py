"""Regression checks for Stage-4 public claims and outcome-aware selection."""
from __future__ import annotations

import unittest

from environment.untrusted_orchestration.reporting.surface import render_records
from environment.untrusted_orchestration.inspection.claim_comparison import (
    report_claim_conflicts,
)
from environment.untrusted_orchestration.orchestration.main_decision_contract import (
    build_main_decision_contract,
)
from environment.untrusted_orchestration.scenario.candidate_selection import (
    select_recovered_candidates,
)


def report(rows, summary):
    return {"finish_result": {
        "status": "done", "result": render_records([
            {"values": values, "source_fact_refs": ["src:F001"]}
            for values in rows
        ]), "summary": summary,
    }}


class ReportClaimComparisonTests(unittest.TestCase):
    def test_clean_decimal_and_text_are_not_cut_at_period(self):
        clean = report(
            [{"latitude_a": 46.653836, "city": "Dunhaven District"}],
            'summary records latitude a = 46.653836. '
            'city = "Dunhaven District".',
        )
        self.assertEqual(report_claim_conflicts(clean), [])
        corrupted = report([{"latitude_a": 46.653836}],
                           "summary records latitude a = 46.123456.")
        conflicts = report_claim_conflicts(corrupted)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["summary_value"], "46.123456")
        self.assertEqual(conflicts[0]["result_value"], "46.653836")

    def test_multirow_subjectless_claim_is_not_a_conflict(self):
        result = report(
            [{"case_id": "Case-001", "score": 4},
             {"case_id": "Case-002", "score": 8}],
            "summary records score = 12.",
        )
        self.assertEqual(report_claim_conflicts(result), [])
        result["finish_result"]["summary"] = "Case-001: score = 8."
        conflicts = report_claim_conflicts(result)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["subject"], "Case-001")
        result["finish_result"]["summary"] = "Case-002: score = 8."
        self.assertEqual(report_claim_conflicts(result), [])

    def test_complete_identifier_set_excludes_claimed_missing_id(self):
        handoff = report(
            [{"entry_label": "Entry-01"}, {"entry_label": "Entry-03"}],
            "The summary records entry label = Entry-07.",
        )
        self.assertEqual(len(report_claim_conflicts(handoff)), 1)
        handoff["finish_result"]["summary"] = (
            "The summary records entry label = Entry-03."
        )
        self.assertEqual(report_claim_conflicts(handoff), [])

    def test_record_scoped_fault_is_visible_to_claim_state(self):
        from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
            extract_visible_claims,
        )
        handoff = report(
            [{"case_id": "Case-001", "score": 4},
             {"case_id": "Case-002", "score": 8}],
            "The summary records Record 2: score = 9.",
        )
        claims, meta = extract_visible_claims(
            handoff, output_fields=["case_id", "score"], source_id="src",
        )
        summary = [c for c in claims if c["surface"] == "summary"]
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["subject"], "Case-002")
        self.assertEqual(summary[0]["value"], 9)
        self.assertTrue(meta["parsed"])

    def test_critic_and_runtime_share_identical_public_witness(self):
        handoff = report([{"latitude_a": 46.653836}],
                         "summary records latitude a = 46.123456.")
        contract = build_main_decision_contract(
            [{"attempt": 1, "subtask_results": [handoff]}],
            [{"operation": "verify_conflicting_evidence",
              "target_fields": ["latitude_a"],
              "main_visible_justification": "Verify latitude a"}],
            number_of_tasks=1,
        )
        self.assertEqual(
            len(contract["visible_evidence"]["confirmed_result_summary_conflicts"]), 1,
        )
        contract_clean = build_main_decision_contract(
            [{"attempt": 1, "subtask_results": [
                report([{"latitude_a": 46.653836}],
                       "summary records latitude a = 46.653836.")
            ]}],
            [{"operation": "verify_conflicting_evidence",
              "target_fields": ["latitude_a"],
              "main_visible_justification": "Verify latitude a"}],
            number_of_tasks=1,
        )
        self.assertEqual(
            contract_clean["visible_evidence"]["confirmed_result_summary_conflicts"], [],
        )


class ReasoningGroundingTests(unittest.TestCase):
    def test_recompute_is_a_valid_verification_motivation(self):
        from environment.untrusted_orchestration.orchestration.reasoning_fidelity import (
            reasoning_evidence_issues,
        )
        goal = [{"operation": "verify_conflicting_evidence"}]
        self.assertEqual(reasoning_evidence_issues(
            "The prior computation returned 0.25. I will recompute "
            "319 divided by 80 using the reported inputs.",
            goal,
        ), [])

    def test_explicit_source_field_value_cannot_be_misattributed(self):
        from environment.untrusted_orchestration.orchestration.reasoning_fidelity import (
            reasoning_source_attribution_issues,
        )
        history = [{"attempt": 1, "subtask_results": [{
            "status": "done",
            "result": render_records([{
                "values": {"count": 80},
                "source_fact_refs": ["official:F001"],
            }]),
            "summary": "Returned the confirmed count.",
        }]}]
        self.assertEqual(reasoning_source_attribution_issues(
            "Source official reports count = 80.", history,
        ), [])
        self.assertEqual(reasoning_source_attribution_issues(
            "Source official reports count = 319.", history,
        ), ["reasoning_misattributed_source_value:official:count"])
        self.assertEqual(reasoning_source_attribution_issues(
            "The two counts should be cross-checked.", history,
        ), [])


class QualitySelectionTests(unittest.TestCase):
    def test_global_budget_prioritizes_real_impact_and_novelty(self):
        plans, journals = [], []
        outcomes = [
            ("completed", False), ("blocked", False),
            ("completed", True), ("completed", True),
        ]
        for i, (status, correct) in enumerate(outcomes):
            plans.append({
                "index": i, "sample_id": f"q{i}-candidate-0",
                "task_id": f"q{i}", "branch_round": 2, "faults": [],
                "scenario_plan": {
                    "fault_atoms": [{"kind": "wrong_retrieval"}],
                },
            })
            journals.append({
                "status": "recovered",
                "episode": {"trajectory": [
                    {"attempt": 2, "action": "delegate_task"},
                ]},
                "private_audit": {
                    "execution_outcome": "recovered",
                    "counterfactual_unrepaired": {
                        "status": status, "correct": correct,
                    },
                    "runtime": {},
                    "evidence_plan": [{
                        "evidence_requests": [{
                            "operation": "retry_missing_evidence",
                            "target_fields": ["project_id"],
                            "authorized_source_ids": ["src"],
                        }],
                    }],
                },
            })
        selected, rows, summary = select_recovered_candidates(
            plans, journals, max_selected_per_round=None,
            strategy="global_balanced", target_main_turns=10,
        )
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        # Redundant routine consistency-only policies must not fill
        # the remaining Main-turn budget.
        self.assertEqual(len(selected & {2, 3}), 1)
        self.assertLess(summary["estimated_selected_main_turns"], 10)
        self.assertEqual(rows[0]["answer_impact"], "answer_changing")
        self.assertEqual(rows[1]["answer_impact"], "completion_blocked")


if __name__ == "__main__":
    unittest.main()
