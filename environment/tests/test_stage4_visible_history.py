"""V10.4.3: only Main-visible history can justify prior-attempt claims."""
from __future__ import annotations

import unittest

from environment.untrusted_orchestration.main_decision_contract import (
    build_main_decision_contract, critic_denies_confirmed_conflict,
)
from environment.untrusted_orchestration.visible_history import (
    public_field_activity, visible_history_reasoning_issues,
)
from environment.untrusted_orchestration.publication_guard import (
    review_turn_surface,
)
from environment.untrusted_orchestration.reasoning_fidelity import (
    reasoning_evidence_issues,
)


def history(*reports):
    return [{"attempt": 5, "subtask_results": list(reports)}]


def report(instruction, result, summary="", status="done"):
    return {
        "instruction": instruction, "result": result,
        "summary": summary, "status": status,
        "trace_summary": "A source lookup was performed.",
    }


def goal(field, *, operation="retrieve_required_evidence"):
    return {
        "operation": operation, "target_fields": [field],
        "main_visible_justification": "The question still requires " + field,
    }


class PriorRetrievalGroundingTests(unittest.TestCase):
    def test_actual_v10_4_2_counts_not_yet_queried(self):
        prior = history(
            report(
                "Identify the candidates from the visual source. "
                "Return candidate_id and marker_code.",
                "candidate id: Case-016, marker code: MK-0531",
            ),
            report(
                "Join candidate_id from the earlier Sub-Agent results and "
                "return marker_code.",
                "marker code: MK-0531",
            ),
        )
        fields = [goal("completed_count"), goal("total_count")]
        activity = public_field_activity(prior, fields)
        self.assertEqual(
            [r["explicit_prior_request_count"] for r in activity],
            [0, 0],
        )
        wrong = (
            "The previous attempts failed to retrieve these counts "
            "from stat_a and stat_b, likely due to bad key matching. "
            "I will fetch completed_count and total_count now."
        )
        self.assertEqual(
            visible_history_reasoning_issues(wrong, activity),
            ["unfounded_prior_failed_retrieval_claim"],
        )
        correct = (
            "The candidate identifier is available, but completed_count "
            "and total_count have not yet been requested. I will obtain "
            "both counts from their authorized sources."
        )
        self.assertEqual(visible_history_reasoning_issues(correct, activity), [])

    def test_real_previous_failed_lookup_is_not_rejected(self):
        prior = history(report(
            "Look up completed_count for the candidate key.",
            "", "No usable final result was delivered.", status="missing",
        ))
        activity = public_field_activity(prior, [goal("completed_count")])
        self.assertEqual(activity[0]["explicit_prior_request_count"], 1)
        self.assertEqual(
            activity[0]["missing_or_partial_prior_request_count"], 1,
        )
        self.assertEqual(visible_history_reasoning_issues(
            "The previous attempt failed to retrieve completed_count; "
            "I should retry the lookup.", activity,
        ), [])

    def test_unrelated_failed_macro_does_not_authorize_count_failure(self):
        prior = history(report(
            "Retrieve the marker_code for Case-016.",
            "", "No usable final result.", "missing",
        ))
        activity = public_field_activity(prior, [
            goal("completed_count"), goal("total_count")
        ])
        self.assertEqual(visible_history_reasoning_issues(
            "The previous attempts failed to retrieve these counts, so "
            "completed_count and total_count must be obtained again.",
            activity,
        ), ["unfounded_prior_failed_retrieval_claim"])

    def test_field_status_uses_public_instruction_not_gold_or_planner(self):
        prior = history(report(
            "Return marker_code.", "The record contains completed_count: 36.",
            "No prior count lookup was delegated."
        ))
        activity = public_field_activity(prior, [goal("completed_count")])
        self.assertEqual(activity[0]["explicit_prior_request_count"], 0)
        self.assertEqual(activity[0]["explicit_completed_delivery_count"], 0)


class VisibleBooleanContradictionTests(unittest.TestCase):
    def test_boolean_true_false_is_confirmed_in_actual_flat_history(self):
        prior = history(report(
            "Return active.",
            '[{"active": true}]',
            "The summary records active = False.",
        ))
        plan = [{
            "operation": "verify_conflicting_evidence",
            "target_fields": ["active"],
            "main_visible_justification": (
                "The returned result and summary disagree about active: "
                "result true, summary false."
            ),
        }]
        contract = build_main_decision_contract(prior, plan, number_of_tasks=1)
        conflicts = contract["visible_evidence"][
            "confirmed_result_summary_conflicts"
        ]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["field"], "active")
        self.assertEqual(conflicts[0]["result_value"], "true")
        self.assertEqual(conflicts[0]["summary_value"], "false")
        self.assertEqual(contract["contract_version"], "v10_4_3")

    def test_boolean_identical_report_is_not_flagged(self):
        plan = [goal("active", operation="verify_conflicting_evidence")]
        contract = build_main_decision_contract(
            history(report("Return active.", '[{"active": true}]',
                           "The summary records active = True.")),
            plan, number_of_tasks=1,
        )
        self.assertEqual(
            contract["visible_evidence"]["confirmed_result_summary_conflicts"],
            [],
        )

    def test_bool_conflict_denial_can_be_re_reviewed_not_auto_passed(self):
        contract = build_main_decision_contract(
            history(report(
                "Return active.", '[{"active": true}]',
                "The summary records active = False.",
            )),
            [goal("active", operation="verify_conflicting_evidence")],
            number_of_tasks=1,
        )
        self.assertTrue(critic_denies_confirmed_conflict(
            {"verdict": "fail",
             "reason": "There is no actual contradiction between the "
                       "result and summary."}, contract,
        ))
        self.assertFalse(critic_denies_confirmed_conflict(
            {"verdict": "fail",
             "reason": "The Author falsely claims it already fetched "
                       "a different independent source."}, contract,
        ))


class ProseHandleIsolationTests(unittest.TestCase):
    def test_workspace_source_handle_not_allowed_in_main_reasoning(self):
        self.assertIn(
            "synthetic_source_handle_in_main_reasoning",
            visible_history_reasoning_issues(
                "I will inspect workspace://visual again.", [],
            )
        )
        self.assertEqual(
            visible_history_reasoning_issues(
                "I will inspect the visual source again.", []
            ), []
        )

    def test_reserved_fixture_url_banned_only_from_main_prose(self):
        url = "https://evidence-32a6327c9bd944f3.example.invalid/archive/record"
        base = {
            "input": {
                "subtask_history": "The source link is " + url,
            },
            "output": {
                "action": "delegate_task",
                "reasoning": "I will recheck the archive record.",
                "params": {
                    "tasks": [{
                        "task_instruction": "Use the known archive URL: " + url,
                        "context": "Retrieve active.",
                        "model": "model_1", "tools": [],
                    }],
                },
            },
        }
        self.assertEqual(review_turn_surface(base), [])
        base["output"]["reasoning"] = (
            "I will inspect " + url + " before answering."
        )
        self.assertIn(
            "synthetic_source_handle_in_main_reasoning:reasoning",
            review_turn_surface(base),
        )

    def test_unfinished_verification_cannot_be_claimed_as_done(self):
        self.assertIn(
            "premature_completion_without_required_verification",
            reasoning_evidence_issues(
                "We can now complete the task without rechecking.",
                [goal("active", operation="verify_conflicting_evidence")],
            ),
        )


if __name__ == "__main__":
    unittest.main()
