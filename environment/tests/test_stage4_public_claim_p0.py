"""Stage 4 P0: public-only claim parsing and observable Result contradictions."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from environment.untrusted_orchestration.inspection.claim_comparison import (
    public_report_conflicts, report_claim_conflicts,
    result_internal_claim_conflicts,
)
from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
    extract_visible_claims,
)
from environment.untrusted_orchestration.orchestration.main_decision_contract import (
    build_main_decision_contract, critic_denies_confirmed_conflict,
)
from environment.untrusted_orchestration.reporting.surface import render_records
from environment.untrusted_orchestration.runtime.evidence_planning import (
    evidence_assignment, evidence_request,
)
from environment.untrusted_orchestration.offline_pipeline import _visibility_issues
from environment.tests.test_stage4_quality_fixes import PublicVerificationFocusTests


def handoff(rows, *, summary="Completed the evidence retrieval.", refs=True):
    return {"finish_result": {
        "status": "done",
        "result": render_records([{
            "values": row,
            "source_fact_refs": ["metric_a:F004"] if refs else [],
        } for row in rows]),
        "summary": summary,
    }}


def disagreement():
    return handoff([
        {"entity_final": "Alderwick Valley", "date_a": "2012-02-18"},
        {"entity_final": "Alderwick Valley", "date_a": "2015-07-22"},
    ])


class ResultInternalConflictTests(unittest.TestCase):
    def test_shared_explicit_record_source_detects_conflicting_values(self):
        report = disagreement()
        found = result_internal_claim_conflicts(report)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["kind"], "intra_result_conflict")
        self.assertEqual(found[0]["field"], "date_a")
        self.assertEqual(found[0]["subject"], "Alderwick Valley")
        self.assertEqual(found[0]["observed_values"], [
            "2012-02-18", "2015-07-22",
        ])
        self.assertEqual(found[0]["source_fact_refs"], ["metric_a:F004"])
        # Old Result/Summary evidence remains separate and backward compatible.
        self.assertEqual(report_claim_conflicts(report), [])
        self.assertEqual(len(public_report_conflicts(report)), 1)

    def test_no_shared_fact_ref_or_strong_record_id_abstains(self):
        self.assertEqual(result_internal_claim_conflicts(handoff([
            {"entity_final": "Alderwick Valley", "metric_a": 7},
            {"entity_final": "Alderwick Valley", "metric_a": 8},
        ], refs=False)), [])

    def test_different_subjects_do_not_conflict(self):
        self.assertEqual(result_internal_claim_conflicts(handoff([
            {"entity_final": "Alderwick Valley", "metric_a": 7},
            {"entity_final": "Bellwater Region", "metric_a": 8},
        ])), [])

    def test_different_explicit_time_scopes_are_not_same_observation(self):
        self.assertEqual(result_internal_claim_conflicts(handoff([
            {"entity_final": "Alderwick Valley", "observation_year": 2020,
             "metric_a": 7},
            {"entity_final": "Alderwick Valley", "observation_year": 2021,
             "metric_a": 8},
        ])), [])

    def test_numeric_equivalence_and_repeated_identical_values_not_fault(self):
        self.assertEqual(result_internal_claim_conflicts(handoff([
            {"entity_final": "Alderwick Valley", "metric_a": 1},
            {"entity_final": "Alderwick Valley", "metric_a": 1.0},
        ])), [])

    def test_explicit_shared_record_key_can_bind_without_factrefs(self):
        found = result_internal_claim_conflicts(handoff([
            {"entity_final": "PRJ-A", "record_key_1": "RK-01", "metric_a": 7},
            {"entity_final": "PRJ-A", "record_key_1": "RK-01", "metric_a": 8},
        ], refs=False))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["field"], "metric_a")

    def test_state_planner_and_critic_share_the_same_witness(self):
        state = PublicVerificationFocusTests.state(disputed=False)
        state.public_reports["M1"] = handoff([
            {"entity_final": "PRJ-A", "metric_count": 20},
            {"entity_final": "PRJ-A", "metric_count": 21},
        ])
        report = state.public_reports["M1"]
        reason = state._visible_report_internal_conflict("M1")
        self.assertIn("returned Result contains conflicting values", reason)
        self.assertEqual(state.public_verification_focus("M1", issue=reason), {
            "basis": "intra_result_conflict",
            "subject": "PRJ-A",
            "attribute": "metric_count",
            "observed_round": 1,
        })
        request = evidence_request(state, "M1", round_number=2, retry=True)
        self.assertEqual(request["operation"], "verify_conflicting_evidence")
        self.assertEqual(request["target_fields"], ["entity_final", "metric_count"])
        self.assertEqual(request["verification_focus"]["basis"],
                         "intra_result_conflict")
        with patch(
            "environment.untrusted_orchestration.runtime.evidence_planning._assignment",
            return_value={
                "target_instruction": "Read entity_final and metric_count from src.",
                "context": "The report is returned.",
                "source_ids": ["src"], "output_fields": ["entity_final", "metric_count"],
            },
        ):
            assignment = evidence_assignment(
                state, "M1", catalog={}, model_alias="model_1",
                request=request,
            )
        self.assertIn("two explicit rows in the received Result",
                      assignment["target_instruction"])
        self.assertEqual(_visibility_issues(
            [request],
            [{"attempt": 1, "subtask_results": [report]}],
            str(report) + str(request),
        ), [])
        contract = build_main_decision_contract(
            [{"attempt": 1, "subtask_results": [report]}],
            [request], number_of_tasks=1,
        )
        visible = contract["visible_evidence"]
        self.assertEqual(visible["confirmed_result_summary_conflicts"], [])
        self.assertEqual(len(visible["confirmed_public_conflicts"]), 1)
        self.assertEqual(visible["confirmed_public_conflicts"][0]["kind"],
                         "intra_result_conflict")
        self.assertTrue(critic_denies_confirmed_conflict({
            "verdict": "fail", "reason": "The conflict is not present."
        }, contract))

        clean = deepcopy(report)
        clean["finish_result"]["result"] = handoff([
            {"entity_final": "PRJ-A", "metric_count": 20},
        ])["finish_result"]["result"]
        self.assertIsNone(state._visible_report_internal_conflict("unknown"))
        state.public_reports["M1"] = clean
        self.assertIsNone(state._visible_report_internal_conflict("M1"))
        self.assertIn("unseen_verification_focus", " ".join(_visibility_issues(
            [request],
            [{"attempt": 1, "subtask_results": [clean]}],
            str(clean) + str(request),
        )))


class ConservativeProseExtractionTests(unittest.TestCase):
    def extract(self, result, fields=("case_id",)):
        return extract_visible_claims({
            "finish_result": {"status": "done", "result": result, "summary": ""}
        }, output_fields=list(fields), source_id=None)

    def test_real_clean_history_oxford_list(self):
        claims, meta = self.extract(
            "Three dossiers match both clues: "
            "Dossier-a, Dossier-b, and Dossier-c."
        )
        self.assertTrue(meta["parsed"])
        self.assertIn("natural_answer_identifier_list", meta["parse_mode"])
        self.assertEqual({item["subject"] for item in claims},
                         {"Dossier-a", "Dossier-b", "Dossier-c"})
        self.assertTrue(all(item["attribute"] == "identifier_present"
                            and item["value"] is True for item in claims))

    def test_existing_numeric_counted_list_accepts_oxford(self):
        claims, meta = self.extract(
            "The current task returned 3 case identifiers: "
            "Case-011, Case-012, and Case-013."
        )
        self.assertEqual(len(claims), 3)
        self.assertTrue(meta["parsed"])

    def test_mismatched_count_rejected(self):
        claims, _ = self.extract(
            "Three dossiers match both clues: Dossier-a, and Dossier-b."
        )
        self.assertEqual(claims, [])

    def test_negated_or_qualified_list_rejected(self):
        for text in (
            "No dossiers match both clues: Dossier-a, Dossier-b.",
            "Two dossiers excluded: Dossier-a and Dossier-b.",
            "Two dossiers match both clues: Dossier-a, Dossier-b, but not Dossier-c.",
            "Dossier-a and Dossier-b were examined but not returned.",
        ):
            with self.subTest(text=text):
                claims, _ = self.extract(text)
                self.assertEqual(claims, [])

    def test_quoted_string_and_date_are_explicit_claims(self):
        claims, meta = self.extract(
            'Dossier-abc: date_a = 2012-02-18. '
            'Dossier-abc: owner = "Alice".',
            fields=("date_a", "owner"),
        )
        self.assertTrue(meta["parsed"])
        self.assertEqual(
            {(x["attribute"], x["value"]) for x in claims},
            {("date_a", "2012-02-18"), ("owner", "Alice")},
        )
        self.assertTrue(all(x["fact_refs"] == [] for x in claims))

    def test_no_undeclared_or_unscoped_claim(self):
        claims, _ = self.extract(
            'Dossier-abc: secret = "hidden"; owner = "Alice".',
            fields=("owner",),
        )
        self.assertEqual(claims, [])
        claims, _ = self.extract("Owner is Alice.", fields=("owner",))
        self.assertEqual(claims, [])


if __name__ == "__main__":
    unittest.main()
