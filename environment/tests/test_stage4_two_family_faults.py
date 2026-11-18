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
            model_alias="model_1",
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
            model_alias="model_1",
        )
        delivered = json.loads(report["finish_result"]["result"])
        self.assertEqual(delivered[0]["values"]["metric"], 15)
        self.assertEqual(delivered[0]["source_fact_refs"],
                         ["ledger:F002"])
        self.assertEqual(attempt.product.rows[0]["values"]["metric"], 12)


if __name__ == "__main__":
    unittest.main()
