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


class PublicVerificationFocusTests(unittest.TestCase):
    @staticmethod
    def state(*, disputed=True):
        from types import SimpleNamespace
        from environment.untrusted_orchestration.state_machine import ExecutionState
        state = ExecutionState.__new__(ExecutionState)
        state.scene = {"macro_graph": {"terminal_task_id": "M1"}}
        state.tasks = {"M1": {
            "task_id": "M1", "depends_on": [], "operator_chain": [],
            "source_ids": ["src"],
            "outputs": {"typed_relation": ["entity_final", "metric_count"]},
        }}
        state.current = {"M1": SimpleNamespace(version=1, status="done", runtime={})}
        state.public_reports = {"M1": report(
            [{"entity_final": "PRJ-A", "metric_count": 20}],
            "The summary records metric count = " + ("21" if disputed else "20") + ".",
        )}
        state.cross_source_checks_enabled = False
        state._public_conflict_cache = None
        state.detected_conflicts = []
        return state

    def test_multioutput_macro_uses_claim_focus_not_reduced_schema(self):
        from unittest.mock import patch
        from environment.untrusted_orchestration.runtime.evidence_planning import (
            evidence_request, evidence_assignment,
        )
        state = self.state()
        request = evidence_request(state, "M1", round_number=2, retry=True)
        self.assertEqual(request["operation"], "verify_conflicting_evidence")
        self.assertEqual(request["target_fields"], ["entity_final", "metric_count"])
        self.assertEqual(request["verification_focus"], {
            "basis": "result_summary_disagreement",
            "subject": "PRJ-A",
            "attribute": "metric_count",
        })
        with patch(
            "environment.untrusted_orchestration.runtime.evidence_planning._assignment",
            return_value={
                "target_instruction": "Retrieve entity_final and metric_count from src.",
                "context": "Earlier src report has already arrived.",
                "source_ids": ["src"], "output_fields": ["entity_final", "metric_count"],
            },
        ):
            assignment = evidence_assignment(
                state, "M1", catalog={}, model_alias="model_1",
                request=request,
            )
        self.assertIn("metric_count claim for PRJ-A", assignment["target_instruction"])
        self.assertIn("Result and Summary", assignment["target_instruction"])
        self.assertEqual(assignment["focus_fields"], ["entity_final", "metric_count"])
        self.assertIn("Specific public claim to recheck", assignment["context"])
        self.assertNotIn("verified metric_count = 20", assignment["target_instruction"])

    def test_clean_report_does_not_invent_focus(self):
        from environment.untrusted_orchestration.runtime.evidence_planning import (
            evidence_request,
        )
        state = self.state(disputed=False)
        request = evidence_request(state, "M1", round_number=2, retry=False)
        self.assertIsNone(request["verification_focus"])
        self.assertEqual(request["operation"], "retrieve_required_evidence")

    def test_current_cross_report_event_can_focus_but_stale_one_cannot(self):
        from environment.untrusted_orchestration.runtime.evidence_planning import (
            evidence_request,
        )
        state = self.state(disputed=False)
        event = {
            "recheck_task_id": "M1", "recheck_task_version": 1,
            "claim": {"subject": "PRJ-A", "attribute": "metric_count"},
            "observed_values": [20, 21],
        }
        state.active_public_conflicts = lambda: [event]
        request = evidence_request(state, "M1", round_number=3, retry=True)
        self.assertEqual(request["verification_focus"], {
            "basis": "cross_report_disagreement",
            "subject": "PRJ-A", "attribute": "metric_count",
        })
        event["recheck_task_version"] = 0
        request = evidence_request(state, "M1", round_number=3, retry=True)
        self.assertIsNone(request["verification_focus"])

    def test_stage4c_requires_the_same_public_result_summary_witness(self):
        from environment.untrusted_orchestration.offline_pipeline import (
            _visibility_issues,
        )
        request = {
            "request_id": "R02-EM1-F01",
            "operation": "verify_conflicting_evidence",
            "target_fields": ["entity_final", "metric_count"],
            "known_identifiers": [],
            "verification_focus": {
                "basis": "result_summary_disagreement",
                "subject": "PRJ-A", "attribute": "metric_count",
            },
            "main_visible_justification": "Result and Summary differ on metric count.",
        }
        bad_report = self.state(disputed=False).public_reports["M1"]["finish_result"]
        good_report = self.state(disputed=True).public_reports["M1"]["finish_result"]
        good_history = [{"attempt": 1, "subtask_results": [good_report]}]
        bad_history = [{"attempt": 1, "subtask_results": [bad_report]}]
        self.assertEqual(_visibility_issues([request], good_history, str(good_history)), [])
        self.assertIn(
            "R02-EM1-F01:unseen_verification_focus",
            _visibility_issues([request], bad_history, str(bad_history)),
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


    def test_direct_public_focus_must_be_named_in_main_reasoning(self):
        from environment.untrusted_orchestration.orchestration.reasoning_fidelity import (
            reasoning_evidence_issues,
        )
        goal = [{
            "operation": "verify_conflicting_evidence",
            "target_fields": ["entity_final", "metric_count"],
            "verification_focus": {
                "basis": "result_summary_disagreement",
                "subject": "PRJ-A", "attribute": "metric_count",
            },
        }]
        self.assertEqual(reasoning_evidence_issues(
            "The public metric count statements disagree; I should "
            "recheck metric_count before using this report.", goal,
        ), [])
        self.assertEqual(reasoning_evidence_issues(
            "The reports conflict, so I will verify the evidence.", goal,
        ), ["missing_specific_verification_focus:metric_count"])
        broader = [{**goal[0], "verification_focus": {
            "basis": "cross_report_disagreement",
            "subject": "PRJ-A", "attribute": "returned_records",
        }}]
        self.assertEqual(reasoning_evidence_issues(
            "The earlier and later results disagree; I should recheck.", broader,
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
