"""Executed-evidence report contracts and publication fidelity tests."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from environment.generate_question.program_ir import Relation
from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.public_language import (
    sanitize_public_prose, sanitize_public_report,
)
from environment.untrusted_orchestration.publication_guard import (
    review_turn_surface,
)
from environment.untrusted_orchestration.report_evidence import (
    observed_handoff_discrepancies,
)
from environment.untrusted_orchestration.report_author import (
    _observed_contract, _review_denies_witnessed_handoff,
    naturalize_report,
)
from environment.untrusted_orchestration.report_contract import (
    build_report_contract, render_delivered_result,
)
from environment.untrusted_orchestration.state_machine import Attempt

runner = importlib.import_module(
    "environment.untrusted_orchestration.generate_turns"
)


def step(operation, identities, *, output_count=None):
    return {
        "operation": operation,
        "execution_status": "done",
        "output_row_count": (
            len(identities) if output_count is None else output_count
        ),
        "output_schema": ["candidate_id", "marker_code"],
        "sampled_result_rows": [
            {"values": {"candidate_id": key, "marker_code": marker},
             "source_fact_refs": ["visual:F001"]}
            for key, marker in identities
        ],
    }


class ReportObservationTests(unittest.TestCase):
    def setUp(self):
        # From V10.1 candidate 000002: filter returned Case-002/Case-016;
        # identity-preserving project then returned Case-004.
        self.filter = step("filter_eq", [
            ("Case-002", "MK-6243"),
            ("Case-016", "MK-0531"),
        ])
        self.project = step("project", [
            ("Case-004", "MK-4525"),
        ])

    def test_real_visible_handoff_mismatch_is_supported(self):
        observations = observed_handoff_discrepancies(
            [self.filter, self.project]
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(
            observations[0]["later_unmatched_identities"], ["Case-004"],
        )
        self.assertEqual(
            observations[0]["earlier_observed_identities"],
            ["Case-002", "Case-016"],
        )
        self.assertEqual(observations[0]["identity_field"], "candidate_id")

    def test_preview_truncation_does_not_prove_missing_record(self):
        truncated = step("filter_eq", [
            ("Case-002", "MK-6243"),
        ], output_count=5)
        self.assertEqual(
            observed_handoff_discrepancies([truncated, self.project]), []
        )
        transformed = step("lookup_by_key", [
            ("Case-004", "MK-4525"),
        ])
        self.assertEqual(
            observed_handoff_discrepancies([self.filter, transformed]), []
        )
        self.assertEqual(
            observed_handoff_discrepancies([
                self.filter, step("project", [
                    ("Case-002", "MK-6243"),
                ]),
            ]), [],
        )

    def test_contract_uses_actual_observed_steps_not_gold(self):
        task = {
            "task_id": "read-visual", "source_ids": ["visual"],
            "operator_chain": [{
                "op": "filter_eq", "source_id": "visual",
            }, {"op": "project", "source_id": "visual"}],
        }
        actual = Attempt(
            task_id="read-visual", version=1, status="done",
            product=Relation(("candidate_id", "marker_code"), ({
                "values": {"candidate_id": "Case-004",
                           "marker_code": "MK-4525"},
                "support": ("visual:F001",),
            },)),
            available_fields=("candidate_id", "marker_code"),
            used_parent_versions={},
            runtime={"observed_operator_steps": [
                self.filter, self.project,
            ]},
        )
        contract = _observed_contract(
            task, actual, "Read visual candidates."
        )
        self.assertEqual(
            contract["observed_handoff_discrepancies"][0]
            ["later_unmatched_identities"], ["Case-004"],
        )
        self.assertNotIn("gold", json.dumps(contract).lower())
        self.assertTrue(_review_denies_witnessed_handoff({
            "verdict": "fail",
            "reason": "The report falsely claims a logical inconsistency "
                      "between filter and project.",
        }))

    def test_witness_review_remains_failable(self):
        task = {
            "task_id": "read-visual", "source_ids": ["visual"],
            "operator_chain": [{
                "op": "filter_eq", "source_id": "visual",
            }, {"op": "project", "source_id": "visual"}],
        }
        actual = Attempt(
            task_id="read-visual", version=1, status="done",
            product=Relation(("candidate_id", "marker_code"), ({
                "values": {"candidate_id": "Case-004",
                           "marker_code": "MK-4525"},
                "support": ("visual:F001",),
            },)),
            available_fields=("candidate_id", "marker_code"),
            used_parent_versions={},
            runtime={"observed_operator_steps": [
                self.filter, self.project,
            ]},
        )

        class Author:
            def complete_json(self, system, request):
                return {
                    "result": (
                        "The delivered record contains candidate id "
                        "Case-004 and marker code MK-4525."
                    ),
                    "summary": (
                        "The preceding filter returned Case-002 and "
                        "Case-016, but the later projection returned "
                        "Case-004. These two executed observations disagree."
                    ),
                    "trace_summary": (
                        "The recorded filter and subsequent projection "
                        "both completed, but their candidate IDs differ."
                    ),
                }

        class Critic:
            def __init__(self, second_pass):
                self.calls = 0
                self.second_pass = second_pass
            def complete_json(self, system, request):
                self.calls += 1
                if self.calls == 2 and self.second_pass:
                    return {
                        "verdict": "pass", "grounded": True,
                        "scope_preserved": True,
                        "natural": True, "useful": True,
                    }
                return {
                    "verdict": "fail", "grounded": False,
                    "scope_preserved": False, "natural": False,
                    "useful": False,
                    "reason": "The report falsely claims a logical inconsistency "
                              "between filter and project output.",
                    "repair_hint": "Remove the claim of logical inconsistency.",
                }

        accepted_critic = Critic(second_pass=True)
        report, status = naturalize_report(
            task, actual, instruction="Read visual candidate records.",
            model_alias="model_1", reviewed=None, author=Author(),
            critic=accepted_critic, max_revisions=0,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(accepted_critic.calls, 2)
        self.assertTrue(status["critic_witness_rereviewed"])
        self.assertIn("Case-004", report["finish_result"]["result"])

        reject_critic = Critic(second_pass=False)
        _, rejected = naturalize_report(
            task, actual, instruction="Read visual candidate records.",
            model_alias="model_1", reviewed=None, author=Author(),
            critic=reject_critic, max_revisions=0,
        )
        self.assertEqual(rejected["status"], "fallback")
        self.assertEqual(reject_critic.calls, 2)


class PublicSurfaceTests(unittest.TestCase):
    def test_internal_ids_removed_only_from_prose(self):
        report = {
            "task_instruction": "Combine work_M1 with work_M2.",
            "trace_summary": "Step m3_op01 ended without a result.",
            "finish_result": {
                "status": "done",
                "result": "Derived from work_M4.",
                "summary": "Operation m3_op01 executed.",
            },
            "rows": [{
                "values": {"candidate_id": "Case-004"},
                "source_fact_refs": ["visual:F001"],
            }],
        }
        clean = sanitize_public_report(report)
        self.assertNotIn("work_M", json.dumps(clean))
        self.assertNotIn("m3_op01", json.dumps(clean))
        self.assertEqual(clean["rows"], report["rows"])
        self.assertEqual(report["task_instruction"],
                         "Combine work_M1 with work_M2.")
        original = {"input": {"subtask_history": "work_M1 used m3_op01"},
                    "output": {"action": "complete",
                               "reasoning": "Finish the task.",
                               "params": {}}}
        self.assertTrue(any(
            "synthetic_execution_identifier_leak" in issue
            for issue in review_turn_surface(original)
        ))
        original["input"]["subtask_history"] = sanitize_public_prose(
            original["input"]["subtask_history"]
        )
        self.assertEqual(review_turn_surface(original), [])


class LanguageExceptionTests(unittest.TestCase):
    def test_main_failure_keeps_true_4a_4b_evidence(self):
        scene, reports, base = sample(0)
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes, report_path, base_path = [
                root / name for name in
                ("scenes.jsonl", "reports.jsonl", "base.jsonl")
            ]
            scenes.write_text(json.dumps(scene) + "\n")
            report_path.write_text("".join(
                json.dumps(r) + "\n" for r in reports
            ))
            base_path.write_text(json.dumps(base) + "\n")
            with patch.object(
                runner, "author_branch_main",
                side_effect=ValueError(
                    "stage4c_dispatch_o1_contract_invalid:internal_teacher_language"
                ),
            ):
                summary = runner.generate_turns(
                    scenes_path=scenes, reports_path=report_path,
                    base_episodes_path=base_path, output_dir=root / "out",
                    max_scenes=1, branches_per_scene=1, workers=2,
                    no_progress=True, scenario_mode="event_driven",
                    candidate_strategy="enumerate",
                    max_candidates_per_round=8,
                    max_distractors_per_fault=2,
                    max_selected_per_round=1,
                    max_language_attempts_per_round=1,
                    delayed_evidence_probability=0.0,
                )
            choices = [
                json.loads(row) for row in
                (root / "out/private/candidate_selection.jsonl")
                .read_text().splitlines()
                if json.loads(row)["selected_for_language_review"]
            ]
            self.assertEqual(len(choices), 1)
            chosen = choices[0]["candidate_index"]
            stage4a = json.loads(
                (root / "out/stage4a/journal" /
                 f"{chosen:06d}.json").read_text()
            )
            stage4c = json.loads(
                (root / "out/journal" /
                 f"{chosen:06d}.json").read_text()
            )
            self.assertEqual(stage4a["status"], "recovered")
            self.assertEqual(stage4c["status"], "rejected_validation")
            self.assertEqual(
                stage4c["private_audit"]["execution_outcome"], "recovered"
            )
            self.assertEqual(
                stage4c["private_audit"]["language_exception"]["phase"],
                "stage4c",
            )
            self.assertGreater(
                len(stage4c["private_audit"]["subagent_report_reviews"]), 0
            )
            self.assertEqual(stage4c["private_audit"]["main_action_reviews"], [])
            self.assertEqual(
                summary["candidates_execution_recovered"],
                summary["execution_recovered_branches"],
            )
            self.assertEqual(summary["reviewed_turns"], 0)


class StructuredFactSpineTests(unittest.TestCase):
    """V10.2's partial-year failure: observed is not delivered."""

    @staticmethod
    def partial_attempt():
        task = {
            "task_id": "read-year", "source_ids": ["register"],
            "depends_on": [],
            "operator_chain": [{"op": "scan", "source_id": "register"}],
        }
        attempt = Attempt(
            task_id="read-year", version=1, status="partial",
            product=Relation(("case_id", "reported_year_b"), ({
                "values": {
                    "case_id": "Case-004", "reported_year_b": None,
                }, "support": ("register:F001",),
            },)),
            available_fields=("case_id",),
            used_parent_versions={}, runtime={
                "observed_operator_steps": [{
                    "operation": "scan", "execution_status": "done",
                    "output_row_count": 1,
                    "output_schema": ["case_id", "reported_year_b"],
                    "sampled_result_rows": [{
                        "values": {
                            "case_id": "Case-004",
                            "reported_year_b": 2003,
                        },
                        "source_fact_refs": ["register:F001"],
                    }],
                    "supporting_fact_refs": ["register:F001"],
                }],
                "missing_fields": ["reported_year_b"],
                "rows_complete": True,
                "delivery_fault_kind": "field_not_delivered",
            },
        )
        return task, attempt

    def test_positive_intermediate_does_not_fill_null_delivery(self):
        task, attempt = self.partial_attempt()
        contract = build_report_contract(task, attempt, "Read year.")
        spine = contract["fact_spine"]
        self.assertEqual(spine["delivery"]["null_fields"],
                         ["reported_year_b"])
        self.assertIsNone(
            spine["delivery"]["result"][0]["values"]["reported_year_b"]
        )
        self.assertEqual(
            spine["partial_intermediate_observations"][0]
            ["observed_values_sample"], [2003],
        )
        self.assertIn(
            "not delivered", spine["delivery"]["result_text"].lower()
        )
        self.assertIn(
            "unavailable", spine["mandatory_summary_clauses"][0].lower()
        )
        self.assertNotIn("gold", json.dumps(contract).lower())

    def test_wrong_claim_cannot_be_auto_corrected_into_accepted_report(self):
        task, attempt = self.partial_attempt()

        class WrongAuthor:
            def complete_json(self, system, payload):
                return {
                    "result": (
                        "The final reported_year_b: 2003 was delivered "
                        "for Case-004."
                    ),
                    "summary": "The final delivered year is 2003.",
                    "trace_summary": (
                        "The scan returned a completed year reading."
                    ),
                }

        class NoCritic:
            def complete_json(self, system, payload):
                raise AssertionError("Guard must fail before critic")

        _, meta = naturalize_report(
            task, attempt, instruction="Read register records.",
            model_alias="model_1", reviewed=None,
            author=WrongAuthor(), critic=NoCritic(), max_revisions=0,
        )
        self.assertEqual(meta["status"], "fallback")
        self.assertEqual(meta["failures"][0]["stage"], "guard")

    def test_critic_gets_locked_delivered_values_and_mandatory_gap(self):
        task, attempt = self.partial_attempt()

        class Author:
            def complete_json(self, system, payload):
                spine = json.loads(payload)[
                    "observed_current_attempt"
                ]["fact_spine"]
                return {
                    "result": spine["delivery"]["result_text"],
                    "summary": (
                        "The scan observed a year internally, but the "
                        "final handoff does not deliver that field."
                    ),
                    "trace_summary": (
                        "The recorded scan completed and the attempt "
                        "returned a partial result."
                    ),
                }

        class Critic:
            def __init__(self):
                self.input = None
            def complete_json(self, system, payload):
                self.input = json.loads(payload)
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True,
                    "natural": True, "useful": True,
                }

        critic = Critic()
        report, meta = naturalize_report(
            task, attempt, instruction="Read register records.",
            model_alias="model_1", reviewed=None, author=Author(),
            critic=critic, max_revisions=0,
        )
        self.assertEqual(meta["status"], "accepted", meta)
        self.assertTrue(meta["delivered_result_locked"])
        self.assertEqual(meta["report_contract_version"], "v10_3")
        self.assertEqual(
            report["finish_result"]["result"],
            critic.input["locked_fact_spine"]["delivery"][
                "result_text"
            ],
        )
        self.assertIn(
            "unavailable", report["finish_result"]["summary"].lower()
        )
        self.assertNotIn(
            "reported year b: 2003",
            report["finish_result"]["result"].lower(),
        )

    def test_two_field_author_compiles_partial_result_without_rewriting(self):
        task, attempt = self.partial_attempt()

        class Author:
            def complete_json(self, system, payload):
                self.assertion_payload = json.loads(payload)
                return {
                    "summary": (
                        "The scan observed a year in an intermediate step, "
                        "but the final reported_year_b field is unavailable."
                    ),
                    "trace_summary": (
                        "The assigned scan completed, and its final "
                        "handoff contains a partial record."
                    ),
                }

        class Critic:
            def complete_json(self, system, payload):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True, "useful": True,
                }

        author = Author()
        report, meta = naturalize_report(
            task, attempt, instruction="Read register records.",
            model_alias="model_1", reviewed=None,
            author=author, critic=Critic(), max_revisions=0,
        )
        self.assertEqual(meta["status"], "accepted", meta)
        self.assertTrue(meta["result_compiled_after_author_omission"])
        self.assertTrue(meta["delivered_result_locked"])
        self.assertIn(
            "not delivered", report["finish_result"]["result"].lower()
        )
        self.assertNotIn(
            "reported year b: 2003",
            report["finish_result"]["result"].lower(),
        )
        self.assertEqual(
            "result" in author.assertion_payload[
                "observed_current_attempt"]["fact_spine"]["delivery"],
            True,
        )

    def test_multi_row_compiler_preserves_returned_record_identifiers(self):
        result = render_delivered_result({
            "actual_delivered_rows": [
                {"values": {"case_id": "Case-010"}},
                {"values": {"case_id": "Case-012"}},
            ],
            "delivery_status": "partial",
            "unresolved_fields": ["remaining_rows"],
        })
        self.assertIn("Case-010", result)
        self.assertIn("Case-012", result)
        self.assertIn("incomplete delivered row set", result)

    def test_missing_author_can_omit_result_without_creating_finish_result(self):
        task = {
            "task_id": "missing", "source_ids": ["table"],
            "depends_on": [],
            "operator_chain": [{"op": "scan", "source_id": "table"}],
        }
        attempt = Attempt(
            task_id="missing", version=1, status="missing",
            product=None, available_fields=(),
            used_parent_versions={}, runtime={},
        )

        class Author:
            def complete_json(self, system, payload):
                return {
                    "summary": (
                        "The assigned attempt returned no usable evidence "
                        "for the requested field."
                    ),
                    "trace_summary": (
                        "The scan did not deliver an inspectable final "
                        "result for this handoff."
                    ),
                }

        class Critic:
            def complete_json(self, system, payload):
                return {
                    "verdict": "pass", "grounded": True,
                    "scope_preserved": True, "natural": True, "useful": True,
                }

        report, meta = naturalize_report(
            task, attempt, instruction="Read table.",
            model_alias="model_1", reviewed=None,
            author=Author(), critic=Critic(), max_revisions=0,
        )
        self.assertEqual(meta["status"], "accepted", meta)
        self.assertIsNone(report["finish_result"])
        self.assertTrue(meta["result_compiled_after_author_omission"])

    def test_witnessed_mismatch_is_not_assigned_attack_cause(self):
        task = {
            "task_id": "read-visual", "source_ids": ["visual"],
            "depends_on": [],
            "operator_chain": [
                {"op": "filter_eq", "source_id": "visual"},
                {"op": "project", "source_id": "visual"},
            ],
        }
        attempt = Attempt(
            task_id="read-visual", version=1, status="done",
            product=Relation(("candidate_id", "marker_code"), ({
                "values": {
                    "candidate_id": "Case-004", "marker_code": "MK-4525",
                }, "support": ("visual:F001",),
            },)),
            available_fields=("candidate_id", "marker_code"),
            used_parent_versions={},
            runtime={"observed_operator_steps": [
                step("filter_eq", [
                    ("Case-002", "MK-6243"),
                    ("Case-016", "MK-0531"),
                ]),
                step("project", [("Case-004", "MK-4525")]),
            ]},
        )
        contract = build_report_contract(
            task, attempt, "Read visual candidates.",
        )
        summary = " ".join(contract["fact_spine"][
            "mandatory_summary_clauses"
        ])
        self.assertIn("Case-004", summary)
        self.assertIn("Case-002", summary)
        self.assertNotIn("attack", summary.lower())


if __name__ == "__main__":
    unittest.main()
