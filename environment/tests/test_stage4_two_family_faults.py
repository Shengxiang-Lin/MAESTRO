"""Stage-4 two-family fault contracts; no backend or private Oracle required."""
from __future__ import annotations

import json
import unittest

from environment.generate_question.execution.program_ir import Relation
from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.atomic_faults import AtomicFault
from environment.untrusted_orchestration.faults import run_attempt, relation_rows
from environment.untrusted_orchestration.report_author import naturalize_report
from environment.untrusted_orchestration.visible_claim_extraction import (
    extract_visible_claims,
)
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
                "program_node_id": "derive",
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
        state.set_public_report("calc", {
            "task_instruction": (
                "Compute the absolute difference between a and b."
            ),
            "finish_result": {
                "status": "done", "result": json.dumps([{
                    "values": {"difference": 5},
                    "source_fact_refs": ["stat:F001"],
                }]), "summary": "The requested difference was calculated.",
            },
        }, round_number=2)
        self.assertIn("imply 4", state.visible_discrepancy("calc"))
        self.assertEqual(
            next(row for row in state.observable_issues()
                 if row["task_id"] == "calc")["kind"],
            "arithmetic_inconsistency",
        )
        # A hidden arithmetic operator is not an instruction Main
        # received: without a visible formula, do not teach it to
        # identify an inconsistency from the private program node.
        state.public_reports["calc"]["task_instruction"] = (
            "Summarize the preceding record."
        )
        self.assertIsNone(state.visible_discrepancy("calc"))
        state.public_reports["calc"]["task_instruction"] = (
            "Compute the absolute difference between a and b."
        )
        # A visible formula without actually delivered operands is also
        # insufficient for an arithmetic contradiction.
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

    def test_different_source_year_fields_are_not_automatically_same_fact(self):
        def claims(field):
            report = {"finish_result": {
                "status": "done",
                "result": json.dumps([{
                    "values": {"case_id": "Case-001", field: 2024},
                    "source_fact_refs": ["report:F001"],
                }]), "summary": "A year was observed.",
            }}
            found, _ = extract_visible_claims(
                report, output_fields=["case_id", field],
                source_id="report",
            )
            return [x for x in found if x["original_field"] == field][0]
        self.assertNotEqual(
            claims("reported_year_a")["attribute"],
            claims("reported_year_b")["attribute"],
        )

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

    def test_4c_done_null_uses_flattened_o1_history(self):
        from environment.untrusted_orchestration.offline_pipeline import (
            _public_done_null_fields, _visibility_issues,
        )
        entries = [{
            "attempt": 1, "subtask_results": [{
                "status": "done", "result": json.dumps([{
                    "values": {"case_id": "Dossier-01", "metric_b": None},
                    "source_fact_refs": ["metric_b:F001"],
                }]),
                "summary": "A finding was returned.",
                "instruction": "Return metric_b for the identified case.",
            }],
        }]
        self.assertEqual(_public_done_null_fields(entries), {"metric_b"})
        requests = [{
            "request_id": "R02-Ework_M8-F01",
            "operation": "complete_partial_evidence",
            "target_fields": ["metric_b"],
            "known_identifiers": [],
        }]
        # O1 formats the history as text, NOT a JSON array. The evidence
        # checker must read the structured delivered history separately.
        self.assertEqual(_visibility_issues(
            requests, entries, "Round 1: Sub-Agent finished (done)."
        ), [])
        entries[0]["subtask_results"][0]["result"] = json.dumps([{
            "values": {"case_id": "Dossier-01", "metric_b": 87},
            "source_fact_refs": ["metric_b:F001"],
        }])
        self.assertEqual(_visibility_issues(
            requests, entries, "Round 1: Sub-Agent finished (done)."
        ), ["R02-Ework_M8-F01:missing_status"])

    def test_dossier_identifier_visible_result_summary_conflict(self):
        from environment.untrusted_orchestration.main_decision_contract import (
            build_main_decision_contract,
        )
        entries = [{"attempt": 1, "subtask_results": [{
            "status": "done",
            "result": json.dumps([{
                "values": {"case_id": "Dossier-001"},
                "source_fact_refs": ["M1:F001"],
            }]),
            "summary": "The summary records case id = Dossier-002.",
        }]}]
        contract = build_main_decision_contract(
            entries, [{
                "operation": "verify_conflicting_evidence",
                "target_fields": ["case_id"],
                "main_visible_justification": "The result and summary disagree.",
            }], number_of_tasks=1,
        )
        conflicts = contract["visible_evidence"][
            "confirmed_result_summary_conflicts"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["summary_value"], "dossier-002")
        self.assertEqual(conflicts[0]["result_value"], "dossier-001")

    def test_latest_batch_synthetic_witness_is_real_and_prior_scene_unchanged(self):
        from copy import deepcopy
        from environment.untrusted_orchestration.latest_batch_witness import (
            propose_witness_patch, patch_scene,
        )
        from environment.untrusted_orchestration.state_machine import (
            deterministic_report,
        )
        scene = deepcopy(self.scene)
        scene["world"]["sources"]["local_primary"] = {
            "schema": ["case_id", "metric"],
            "access_mode": "browse",
            "access_policy": {"mode": "browse"},
            "modality": "text",
            "title": "Primary register",
            "observations": [{
                "observation_id": "local_primary.OBS001",
                "values": {"case_id": "Case-011", "metric": 87},
                "fact_refs": ["local_primary:F001"],
                "evidence_origin_id": "primary_root",
                "observed_content": "Case-011 has metric 87.",
            }, {
                "observation_id": "local_primary.OBS002",
                "values": {"case_id": "Case-012", "metric": 55},
                "fact_refs": ["local_primary:F002"],
                "evidence_origin_id": "distractor_root",
                "observed_content": "Case-012 has metric 55.",
            }],
        }
        scene["world"]["sources"]["local_secondary"] = {
            "schema": ["case_id", "metric"],
            "access_mode": "browse", "access_policy": {"mode": "browse"},
            "modality": "text", "title": "Secondary register",
            "observations": [{
                "observation_id": "local_secondary.OBS001",
                "values": {"case_id": "Case-011", "metric": 22},
                "fact_refs": ["local_secondary:F001"],
                "evidence_origin_id": "secondary_root",
                "observed_content": "A secondary observation.",
            }],
        }
        task = {
            "task_id": "target", "depends_on": [],
            "source_ids": ["local_primary", "local_secondary"],
            "external_inputs": [], "input_schema": {},
            "operator_chain": [{
                "id": "load", "op": "scan", "inputs": [],
                "source_id": "local_primary",
            }, {
                "id": "secondary", "op": "scan", "inputs": [],
                "source_id": "local_secondary",
            }, {
                "id": "filter", "op": "filter_eq", "inputs": ["load"],
                "field": "case_id", "value": "Case-011",
            }],
            "program_node_id": "filter",
            "outputs": {"typed_relation": ["case_id", "metric"]},
        }
        state = ExecutionState(scene, {})
        state.tasks = {"target": task}
        state.attempts = {"target": []}
        attempt = state.register("target")
        public = deterministic_report(
            task, attempt,
            instruction="Look up Case-011 metric from local_primary.",
            model_alias="model_1",
        )
        public["finish_result"]["result"] = json.dumps([{
            "values": {"case_id": "Case-011", "metric": 55},
            "source_fact_refs": ["local_primary:F002"],
        }])
        state.set_public_report("target", public, round_number=1)
        diagnostic = {}
        patch = propose_witness_patch(
            state, "target", branch_round=2, diagnostic=diagnostic,
        )
        self.assertIsNotNone(patch)
        self.assertEqual(diagnostic["status"], "generated")
        self.assertEqual(diagnostic["macro_source_count"], 2)
        # Composite FactRefs are NOT independent single-origin claims.
        mixed = deepcopy(public)
        mixed["finish_result"]["result"] = json.dumps([{
            "values": {"case_id": "Case-011", "metric": 55},
            "source_fact_refs": [
                "local_primary:F002", "local_secondary:F001",
            ],
        }])
        state.set_public_report("target", mixed, round_number=1)
        skipped = {}
        self.assertIsNone(propose_witness_patch(
            state, "target", branch_round=2, diagnostic=skipped,
        ))
        self.assertEqual(
            skipped["reason"], "no_single_origin_direct_source_claim",
        )
        state.set_public_report("target", public, round_number=1)
        self.assertEqual(patch["claim"]["attribute"], "metric")
        self.assertEqual(patch["claim"]["faulty_delivered_value"], 55)
        self.assertEqual(patch["claim"]["clean_observation_value"], 87)
        self.assertFalse(patch["empirically_independent"])
        patched = patch_scene(scene, patch)
        self.assertNotIn(patch["source_id"], scene["world"]["sources"])
        self.assertNotIn(patch["task_id"], {
            x["task_id"] for x in scene["macro_graph"]["tasks"]
        })
        self.assertIn(patch["source_id"], patched["world"]["sources"])
        witness, _ = run_attempt(
            patch["task"], parents=[],
            sources={patch["source_id"]: patch["source"]},
        )
        self.assertEqual(relation_rows(witness)[0]["values"], {
            "case_id": "Case-011", "metric": 87,
        })
        self.assertEqual(
            relation_rows(witness)[0]["source_fact_refs"],
            [patch["source_id"] + ":F001"],
        )

        # Once both reports have been delivered in the latest batch,
        # only their PUBLIC cited claims can justify a cross-source check.
        state.scene = patched
        from environment.implement_subtask.realize import _fact_index
        state.fact_index = _fact_index(patched["world"])
        state.tasks[patch["task_id"]] = patch["task"]
        state.attempts[patch["task_id"]] = []
        state.origin_by_ref[patch["source_id"] + ":F001"] = (
            patch["source"]["observations"][0]["evidence_origin_id"]
        )
        new_attempt = state.register(patch["task_id"])
        new_report = deterministic_report(
            patch["task"], new_attempt,
            instruction="Inspect the comparison register.",
            model_alias="model_1",
        )
        state.set_public_report(patch["task_id"], new_report, round_number=1)
        state.cross_source_checks_enabled = True
        state._detect_explicit_cross_report_disagreements()
        self.assertEqual(len(state.detected_conflicts), 1)
        self.assertEqual(
            state.detected_conflicts[0]["conflict_timing"], "simultaneous"
        )
        self.assertEqual(
            state.detected_conflicts[0]["recheck_task_id"], "target"
        )

    def test_cross_field_values_do_not_prove_same_claim_disagreement(self):
        from environment.untrusted_orchestration.main_decision_contract import (
            build_main_decision_contract,
        )
        from environment.untrusted_orchestration.trajectory_builder import (
            _MAIN_REASONING_CRITIC_SYSTEM,
        )
        entries = [{"attempt": 1, "subtask_results": [{
            "status": "done",
            "result": json.dumps([{
                "values": {
                    "case_id": "Dossier-01",
                    "metric_a": 87, "metric_b": 55,
                },
                "source_fact_refs": ["primary:F001"],
            }]),
            "summary": (
                "Dossier-01 metric_a = 87, metric_b = 55. "
                "Comparison still pending."
            ),
        }]}]
        goals = [{
            "operation": "verify_conflicting_evidence",
            "target_fields": ["metric_a", "metric_b"],
            "main_visible_justification": (
                "A later comparison of metric_a and metric_b is pending."
            ),
        }]
        contract = build_main_decision_contract(
            entries, goals, number_of_tasks=1,
        )
        self.assertEqual(
            contract["visible_evidence"]["confirmed_result_summary_conflicts"],
            [],
        )
        self.assertIn("DIFFERENT attributes", _MAIN_REASONING_CRITIC_SYSTEM)
        self.assertIn("pending computation", _MAIN_REASONING_CRITIC_SYSTEM)

    def test_latest_batch_report_has_matching_dispatch_in_full_branch(self):
        from unittest.mock import patch as mocked
        from environment.untrusted_orchestration.atomic_faults import probe_atoms
        from environment.untrusted_orchestration.trajectory_builder import (
            build_branch,
        )
        from environment.tests.test_untrusted_orchestration import sample
        scene, reports, trusted = sample(0)
        eligible = probe_atoms(scene, [
            "fabricated_claim", "wrong_retrieval",
        ])
        first = scene["macro_graph"]["reference_waves"][0]
        choice = next((
            (tid, kind) for tid in first
            for kind in ("fabricated_claim", "wrong_retrieval")
            if kind in eligible.get(tid, [])
        ), None)
        if choice is None:
            self.skipTest("fixture has no first-round Wrong/Report candidate")
        tid, kind = choice
        atom = AtomicFault(kind, tid, origin_round=1)
        ident = "crosscheck_unit_test"
        witness_tid = "zz_aux_unit_test"
        patch = {
            "version": 1,
            "source_id": ident,
            "task_id": witness_tid,
            "feedback_round": 1,
            "compare_with": tid,
            "claim": {
                "subject": "Case-011", "subject_field": "case_id",
                "attribute": "metric", "faulty_delivered_value": 55,
                "clean_observation_value": 87,
            },
            "construction": "stage4_synthetic_comparison_source",
            "empirically_independent": False,
            "source": {
                "schema": ["case_id", "metric"],
                "access_mode": "browse",
                "access_policy": {"mode": "browse"},
                "modality": "text", "title": "Comparison register",
                "observations": [{
                    "observation_id": ident + ".OBS001",
                    "values": {"case_id": "Case-011", "metric": 87},
                    "fact_refs": [ident + ":F001"],
                    "evidence_origin_id": ident + ".root",
                    "observed_content": "Case-011 metric is 87.",
                }],
            },
            "task": {
                "task_id": witness_tid, "depends_on": [],
                "source_ids": [ident], "external_inputs": [],
                "input_schema": {},
                "operator_chain": [{
                    "id": "read_comparison", "op": "scan",
                    "inputs": [], "source_id": ident,
                }],
                "program_node_id": "read_comparison",
                "outputs": {"typed_relation": ["case_id", "metric"]},
            },
        }
        scenario = {
            "mode": "event_driven", "blueprint": "immediate_feedback_fault",
            "scenario_id": "latest-batch-fixture",
            "fault_atoms": [atom.to_private_dict()],
            "evidence_events": [],
        }
        with mocked(
            "environment.untrusted_orchestration.trajectory_builder."
            "propose_witness_patch", return_value=patch,
        ):
            episode, _, audit = build_branch(
                scene, reports, trusted, branch_round=2,
                faults=[atom.as_legacy_fault()],
                scenario_plan=scenario, max_rounds=16, max_retries=1,
                max_post_fork_rounds=4,
                max_parallel_tasks=len(first) + 1,
            )
        latest = episode["trajectory"][0]
        self.assertEqual(latest["attempt"], 1)
        self.assertEqual(latest["action"], "delegate_task")
        self.assertEqual(
            len(latest["params"]["tasks"]),
            len(latest["result"]["subtask_results"]),
        )
        self.assertIn(ident, latest["params"]["tasks"][-1][
            "task_instruction"])
        self.assertIn(ident + ":F001", latest["result"][
            "subtask_results"][-1]["finish_result"]["result"])
        self.assertIn(witness_tid, audit["runtime"]["attempt_versions"])
        self.assertEqual(audit["latest_batch_witness_patch"], patch)
        self.assertEqual(
            audit["latest_batch_witness_probe"]["status"], "generated",
        )
        self.assertNotIn(ident, scene["world"]["sources"])


if __name__ == "__main__":
    unittest.main()
