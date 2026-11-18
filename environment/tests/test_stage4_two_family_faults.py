"""Stage-4 two-family fault contracts; no backend or private Oracle required."""
from __future__ import annotations

import json
import unittest

from environment.generate_question.execution.program_ir import Relation
from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.atomic_faults import AtomicFault
from environment.untrusted_orchestration.faults import run_attempt, relation_rows
from environment.untrusted_orchestration.report_author import naturalize_report
from environment.untrusted_orchestration.state_machine import (
    Attempt, ExecutionState,
)


class IntraReportFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, _, _ = sample(0)

    def test_computation_corrupts_one_operator_but_not_input_or_provenance(self):
        task = {
            "task_id": "calc", "depends_on": ["prior"],
            "source_ids": [], "input_schema": {"input_0": ["case_id"]},
            "external_inputs": ["prior_output"],
            "operator_chain": [{
                "id": "counted", "op": "count", "inputs": ["prior_output"],
            }],
            "program_node_id": "counted",
            "outputs": {"typed_relation": ["count"]},
        }
        original = Relation(("case_id",), ({
            "values": {"case_id": "Case-001"}, "support": ("src:F001",),
        }, {
            "values": {"case_id": "Case-002"}, "support": ("src:F002",),
        }))
        before = relation_rows(original)
        clean, _ = run_attempt(task, parents=[original], sources={})
        faulted, audit = run_attempt(
            task, parents=[original], sources={},
            fault=AtomicFault("computation_error", "calc", 2).as_legacy_fault(),
        )
        self.assertEqual(relation_rows(original), before)
        self.assertEqual(clean.rows[0]["values"]["count"], 2)
        self.assertEqual(faulted.rows[0]["values"]["count"], 3)
        self.assertEqual(faulted.rows[0]["support"], clean.rows[0]["support"])
        self.assertEqual(audit["kind"], "miscomputed_numeric_output")
        self.assertEqual(audit["calculation_operation"], "count")
        self.assertEqual(audit["observed_operator_steps"][-1][
            "sampled_result_rows"][0]["values"]["count"], 3)

    def test_done_marked_partial_field_is_visible_without_hidden_gold(self):
        task = {"task_id": "T", "source_ids": [], "operator_chain": [],
                "depends_on": [],
                "outputs": {"typed_relation": ["case_id", "metric"]}}
        attempt = Attempt(
            task_id="T", version=1, status="partial",
            product=Relation(("case_id", "metric"), ({
                "values": {"case_id": "Case-001", "metric": None},
                "support": ("real:F001",),
            },)),
            available_fields=("case_id",),
            used_parent_versions={}, runtime={
                "missing_fields": ["metric"],
                "rows_complete": True,
                "delivery_fault_kind": "field_not_delivered",
                "report_fault_kind": "done_but_incomplete",
                "reported_status": "done",
            },
        )
        report, meta = naturalize_report(
            task, attempt, instruction="Return the case metric.",
            model_alias="model_1", reviewed=None,
        )
        self.assertEqual(meta["status"], "accepted")
        self.assertEqual(report["finish_result"]["status"], "done")
        returned = json.loads(report["finish_result"]["result"])
        self.assertIsNone(returned[0]["values"]["metric"])
        self.assertEqual(returned[0]["source_fact_refs"], ["real:F001"])

        state = ExecutionState(self.scene, {})
        state.tasks = {"T": task}
        state.scene = {
            **self.scene,
            "macro_graph": {
                **self.scene["macro_graph"],
                "terminal_task_id": "T",
            },
        }
        state.current["T"] = attempt
        state.set_public_report("T", report, round_number=2)
        self.assertIn("marked done", state.visible_discrepancy("T"))
        self.assertEqual(state.observable_issues()[0]["kind"],
                         "done_but_incomplete")
        self.assertIn("T", state.scheduling_state()["repair"])

        # The private missing field must NOT create a Main-visible
        # discrepancy when the delivered report has the full value.
        safe = json.loads(report["finish_result"]["result"])
        safe[0]["values"]["metric"] = 12
        report["finish_result"]["result"] = json.dumps(safe)
        state.set_public_report("T", report, round_number=2)
        self.assertIsNone(state.visible_discrepancy("T"))
        self.assertEqual(state.observable_issues(), [])

    def test_arithmetic_conflict_needs_visible_input_values(self):
        state = ExecutionState(self.scene, {})
        parent = {"task_id": "prior", "source_ids": ["stat"],
                  "depends_on": [], "operator_chain": [],
                  "outputs": {"typed_relation": ["a", "b"]}}
        calc = {"task_id": "calc", "depends_on": ["prior"],
                "source_ids": [],
                "input_schema": {"input_0": ["a", "b"]},
                "external_inputs": ["prior_node"],
                "operator_chain": [{
                    "id": "derive", "op": "derive_arithmetic",
                    "inputs": ["prior_node"],
                    "left_field": "a", "right_field": "b",
                    "output_field": "difference",
                    "operation": "absolute_difference", "round_digits": 0,
                }],
                "outputs": {"typed_relation": ["difference"]}}
        state.tasks = {"prior": parent, "calc": calc}
        state.current = {
            "prior": Attempt(
                "prior", 1, "done",
                Relation(("a", "b"), ({
                    "values": {"a": 10, "b": 6},
                    "support": ("stat:F001",),
                },)), ("a", "b"), {}, {},
            ),
            "calc": Attempt(
                "calc", 1, "done",
                Relation(("difference",), ({
                    "values": {"difference": 5},
                    "support": ("stat:F001",),
                },)), ("difference",), {"prior": 1}, {},
            ),
        }
        state.set_public_report("prior", {"finish_result": {
            "status": "done", "result": json.dumps([{
                "values": {"a": 10, "b": 6},
                "source_fact_refs": ["stat:F001"],
            }]), "summary": "The two original values were returned.",
        }}, round_number=1)
        state.set_public_report("calc", {"finish_result": {
            "status": "done", "result": json.dumps([{
                "values": {"difference": 5},
                "source_fact_refs": ["stat:F001"],
            }]), "summary": "The requested difference was calculated.",
        }}, round_number=2)
        self.assertIn("imply 4", state.visible_discrepancy("calc"))
        self.assertEqual(
            next(row for row in state.observable_issues()
                 if row["task_id"] == "calc")["kind"],
            "arithmetic_inconsistency",
        )
        # Private values do not make a Main-visible mismatch.
        state.public_reports["prior"]["finish_result"]["result"] = (
            "The original observations were reviewed."
        )
        self.assertIsNone(state.visible_discrepancy("calc"))

    def test_wrong_scope_only_if_filter_is_in_public_instruction(self):
        state = ExecutionState(self.scene, {})
        task = {
            "task_id": "scope", "depends_on": [],
            "source_ids": ["ledger"],
            "operator_chain": [{
                "op": "filter_eq", "field": "year", "value": 2024,
            }], "outputs": {"typed_relation": ["case_id", "year"]},
        }
        state.tasks = {"scope": task}
        state.current["scope"] = Attempt(
            "scope", 1, "done",
            Relation(("case_id", "year"), ({
                "values": {"case_id": "Case-002", "year": 2024},
                "support": ("ledger:F002",),
            },)), ("case_id", "year"), {}, {},
        )
        report = {"task_instruction": "Find the 2024 year record.",
                  "finish_result": {
                      "status": "done",
                      "result": json.dumps([{
                          "values": {"case_id": "Case-002", "year": 2023},
                          "source_fact_refs": ["ledger:F001"],
                      }]), "summary": "Found a matching record.",
                  }}
        state.set_public_report("scope", report, round_number=2)
        self.assertIn("requested scope", state.visible_discrepancy("scope"))
        self.assertEqual(state.observable_issues()[0]["kind"],
                         "scope_constraint_mismatch")
        report["task_instruction"] = "Find a related record."
        state.set_public_report("scope", report, round_number=2)
        self.assertIsNone(state.visible_discrepancy("scope"))

    def test_fabricated_report_retains_cited_distractor_origin(self):
        task = {"task_id": "T", "source_ids": ["ledger"],
                "operator_chain": [], "depends_on": [],
                "outputs": {"typed_relation": ["case_id", "metric"]}}
        attempt = Attempt(
            "T", 1, "done",
            Relation(("case_id", "metric"), ({
                "values": {"case_id": "Case-001", "metric": 12},
                "support": ("ledger:F001",),
            },)),
            ("case_id", "metric"), {}, {
                "report_fault_kind": "fabricated_claim",
                "report_fault_payload": {
                    "field": "metric", "replacement": 15,
                    "source_fact_refs": ["ledger:F002"],
                },
            },
        )
        report, _ = naturalize_report(
            task, attempt, instruction="Return the case metric.",
            model_alias="model_1", reviewed=None,
        )
        delivered = json.loads(report["finish_result"]["result"])
        self.assertEqual(delivered[0]["values"]["metric"], 15)
        self.assertEqual(delivered[0]["source_fact_refs"],
                         ["ledger:F002"])
        self.assertEqual(attempt.product.rows[0]["values"]["metric"], 12)


if __name__ == "__main__":
    unittest.main()
