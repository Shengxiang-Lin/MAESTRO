"""Stage-4 two-family fault contracts; no backend or private Oracle required."""
from __future__ import annotations

import json
import unittest

from environment.untrusted_orchestration.reporting.result_text import delivered_records

from environment.generate_question.execution.program_ir import Relation
from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.runtime.atomic_faults import AtomicFault
from environment.untrusted_orchestration.runtime.faults import run_attempt, relation_rows
from environment.untrusted_orchestration.reporting.report_author import naturalize_report
from environment.untrusted_orchestration.inspection.visible_claim_extraction import (
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
        returned = delivered_records(report["finish_result"]["result"])
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

        # When Result lists a value but the SAME public report still
        # admits that extraction failed, this is a visible inconsistency,
        # NOT a leak from the hidden execution ledger. It merits checking.
        safe = delivered_records(report["finish_result"]["result"])
        safe[0]["values"]["metric"] = 12
        report["finish_result"]["result"] = json.dumps(safe)
        state.set_public_report("T", report, round_number=2)
        self.assertIn("unfinished evidence work",
                      state.visible_discrepancy("T"))
        # Removing the public admission leaves no basis for inferring the
        # private missing field from the hidden Attempt alone.
        report["finish_result"]["summary"] = (
            "The completed report lists the extracted metric of 12."
        )
        report["trace_summary"] = (
            "The available source record was inspected."
        )
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
        delivered = delivered_records(report["finish_result"]["result"])
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
        from environment.untrusted_orchestration.orchestration.main_decision_contract import (
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

    def test_latest_report_conflicts_with_prior_report_without_extra_task(self):
        from copy import deepcopy
        from environment.untrusted_orchestration.state_machine import (
            deterministic_report,
        )
        from environment.untrusted_orchestration.scenario.history_conflicts import (
            latest_history_disagreements,
        )
        scene = deepcopy(self.scene)
        scene["world"]["sources"]["local_primary"] = {
            "schema": ["case_id", "metric"],
            "access_mode": "browse", "access_policy": {"mode": "browse"},
            "modality": "text", "title": "Primary register",
            "observations": [{
                "observation_id": "local_primary.OBS001",
                "values": {"case_id": "Case-011", "metric": 87},
                "fact_refs": ["local_primary:F001"],
                "evidence_origin_id": "primary_root",
                "observed_content": "Case-011 metric = 87.",
            }],
        }
        task = {
            "depends_on": [], "source_ids": ["local_primary"],
            "external_inputs": [], "input_schema": {},
            "operator_chain": [{
                "id": "load", "op": "scan", "inputs": [],
                "source_id": "local_primary",
            }],
            "program_node_id": "load",
            "outputs": {"typed_relation": ["case_id", "metric"]},
        }
        state = ExecutionState(scene, {})
        state.tasks = {
            "prior": {**task, "task_id": "prior"},
            "latest": {**task, "task_id": "latest"},
        }
        state.attempts = {"prior": [], "latest": []}
        prior = state.register("prior")
        old_report = deterministic_report(
            state.tasks["prior"], prior,
            instruction="Read the primary register.", model_alias="model_1",
        )
        state.set_public_report("prior", old_report, round_number=1)

        new = state.register("latest")
        corrupted = deterministic_report(
            state.tasks["latest"], new,
            instruction="Read the register.", model_alias="model_1",
        )
        corrupted["finish_result"]["result"] = json.dumps([{
            "values": {"case_id": "Case-011", "metric": 55},
            "source_fact_refs": ["local_primary:F001"],
        }])
        state.set_public_report("latest", corrupted, round_number=2)
        issues = latest_history_disagreements(state)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["attribute"], "metric")
        self.assertEqual(issues[0]["earlier_value"], 87)
        self.assertEqual(issues[0]["latest_value"], 55)
        state.cross_source_checks_enabled = True
        state.scheduling_state()
        self.assertEqual(state.detected_conflicts[0]["kind"],
                         "cross_round_report_disagreement")
        self.assertEqual(state.detected_conflicts[0]["recheck_task_id"],
                         "latest")
        self.assertEqual(state.observable_issues()[0]["kind"],
                         "historical_report_conflict")

        # A real retry can resolve the discrepancy using the same source.
        retried = state.register("latest")
        corrected = deterministic_report(
            state.tasks["latest"], retried,
            instruction="Recheck cited records.", model_alias="model_1",
        )
        state.set_public_report("latest", corrected, round_number=3)
        state.scheduling_state()
        self.assertFalse(state.unresolved_cited_conflicts())
        self.assertEqual(len(state.tasks), 2)

    def test_history_anchor_uses_prior_executed_product_not_gold(self):
        from copy import deepcopy
        from environment.untrusted_orchestration.state_machine import (
            deterministic_report, public_history_item,
        )
        from environment.untrusted_orchestration.scenario.history_conflicts import (
            expose_executed_prior_record, latest_history_disagreements,
        )
        scene = deepcopy(self.scene)
        scene["world"]["sources"]["local_primary"] = {
            "schema": ["case_id", "metric"],
            "access_mode": "browse", "access_policy": {"mode": "browse"},
            "modality": "text", "title": "Primary register",
            "observations": [{
                "observation_id": "local_primary.OBS001",
                "values": {"case_id": "Case-011", "metric": 87},
                "fact_refs": ["local_primary:F001"],
                "evidence_origin_id": "primary_root",
                "observed_content": "Case-011 metric = 87.",
            }],
        }
        task = {
            "depends_on": [], "source_ids": ["local_primary"],
            "external_inputs": [], "input_schema": {},
            "operator_chain": [{
                "id": "load", "op": "scan", "inputs": [],
                "source_id": "local_primary",
            }],
            "program_node_id": "load",
            "outputs": {"typed_relation": ["case_id", "metric"]},
        }
        state = ExecutionState(scene, {})
        state.tasks = {
            "prior": {**task, "task_id": "prior"},
            "latest": {**task, "task_id": "latest"},
        }
        state.attempts = {"prior": [], "latest": []}
        a = state.register("prior")
        old = deterministic_report(
            state.tasks["prior"], a, instruction="Check source.",
            model_alias="model_1",
        )
        old["finish_result"]["result"] = "A record was reported."
        state.set_public_report("prior", old, round_number=1)
        b = state.register("latest")
        latest = deterministic_report(
            state.tasks["latest"], b, instruction="Read source.",
            model_alias="model_1",
        )
        latest["finish_result"]["result"] = json.dumps([{
            "values": {"case_id": "Case-011", "metric": 55},
            "source_fact_refs": ["local_primary:F001"],
        }])
        state.set_public_report("latest", latest, round_number=2)
        batches = [
            {"attempt": 1, "subtask_results": [public_history_item(old, 0)]},
            {"attempt": 2, "subtask_results": [public_history_item(latest, 0)]},
        ]
        traj = [
            {"attempt": 1, "result": {"subtask_results": [old]},
             "params": {"tasks": [{"task_instruction": "Check source."}]}},
            {"attempt": 2, "result": {"subtask_results": [latest]},
             "params": {"tasks": [{"task_instruction": "Read source."}]}},
        ]
        diagnostic = expose_executed_prior_record(
            state, batches, traj, latest_task_ids={"latest"},
        )
        self.assertEqual(diagnostic["status"], "exposed")
        self.assertIn("local_primary:F001",
                      traj[0]["result"]["subtask_results"][0][
                          "finish_result"]["result"])
        self.assertEqual(len(latest_history_disagreements(state)), 1)
        self.assertEqual(len(traj[0]["params"]["tasks"]), 1)

    def test_date_wording_is_guarded_against_exact_observed_iso_dates(self):
        from environment.untrusted_orchestration.reporting.report_validation import (
            claim_tokens, grounded_calendar_text,
        )
        actual = json.dumps({
            "answer": {
                "duration_days": 39,
                "midpoint_date": "2011-07-02",
                "observation_count": 12,
            },
            "date_a": "2011-06-13", "date_b": "2011-07-22",
        })
        summary = (
            "A 39-day gap from June 13 to July 22, 2011 produced "
            "the midpoint July 2, 2011, across 12 observations."
        )
        grounded = grounded_calendar_text(summary, actual)
        self.assertEqual(
            claim_tokens(grounded) - claim_tokens(actual), set()
        )
        self.assertEqual(
            grounded_calendar_text("The midpoint was July 3, 2011.", actual),
            "The midpoint was July 3, 2011.",
        )
        self.assertIn("3", claim_tokens(
            grounded_calendar_text("July 3, 2011", actual)
        ) - claim_tokens(actual))
        # A bare numeric claim of 2 is never admitted solely because a
        # supported date contains 02.
        self.assertIn("2", claim_tokens("There are 2 observations.")
                      - claim_tokens(actual))

    def test_report_guard_accepts_only_exact_delivered_factref_count(self):
        from environment.untrusted_orchestration.reporting.report_validation import (
            claim_tokens, grounded_reference_count_text,
        )
        rows = [{
            "values": {"answer": {
                "conclusion": "first",
                "first_count": 87,
                "second_count": 55,
            }},
            "source_fact_refs": [f"real:F{i:03d}" for i in range(1, 15)],
        }]
        available = json.dumps(rows, ensure_ascii=False)
        summary = (
            "The returned conclusion was first, with counts 87 and 55, "
            "supported by 14 fact references."
        )
        grounded = grounded_reference_count_text(summary, rows)
        self.assertNotIn("14", claim_tokens(grounded) - claim_tokens(available))
        self.assertIn(
            "15", claim_tokens(
                grounded_reference_count_text(
                    "The result cited 15 fact references.", rows,
                )
            ) - claim_tokens(available),
        )
        self.assertIn(
            "14", claim_tokens(
                grounded_reference_count_text(
                    "The result contained 14 observations.", rows,
                )
            ) - claim_tokens(available),
        )
        # Duplicated citations are not counted twice.
        rows[0]["source_fact_refs"].append("real:F014")
        self.assertNotIn("14", claim_tokens(
            grounded_reference_count_text(summary, rows)
        ) - claim_tokens(available))

    def test_prior_observation_is_executed_in_an_earlier_wave(self):
        from copy import deepcopy
        from environment.untrusted_orchestration.state_machine import (
            deterministic_report,
        )
        from environment.untrusted_orchestration.scenario.prior_observation import (
            propose_prior_observation, patch_scene, comparison_rows,
        )
        scene = deepcopy(self.scene)
        scene["world"]["sources"]["hist_source"] = {
            "schema": ["case_id", "metric"],
            "access_mode": "browse", "access_policy": {"mode": "browse"},
            "modality": "text", "title": "Case register",
            "observations": [{
                "observation_id": "hist_source.OBS001",
                "values": {"case_id": "Case-011", "metric": 87},
                "fact_refs": ["hist_source:F001"],
                "evidence_origin_id": "hist_root",
                "observed_content": "Case-011 has metric 87.",
            }, {
                "observation_id": "hist_source.OBS002",
                "values": {"case_id": "Case-012", "metric": 55},
                "fact_refs": ["hist_source:F002"],
                "evidence_origin_id": "hist_other",
                "observed_content": "Case-012 has metric 55.",
            }],
        }
        task = {
            "task_id": "target", "depends_on": [],
            "source_ids": ["hist_source"], "external_inputs": [],
            "input_schema": {},
            "operator_chain": [{
                "id": "scan", "op": "scan", "inputs": [],
                "source_id": "hist_source",
            }, {
                "id": "filter", "op": "filter_eq", "inputs": ["scan"],
                "field": "case_id", "value": "Case-011",
            }, {
                "id": "project", "op": "project", "inputs": ["filter"],
                "fields": ["metric"],
            }],
            "program_node_id": "project",
            "outputs": {"typed_relation": ["metric"]},
        }
        state = ExecutionState(scene, {})
        state.tasks = {"target": task}
        state.attempts = {"target": []}
        attempt = state.register("target")
        faulty = deterministic_report(task, attempt,
                                     instruction="Find the metric.",
                                     model_alias="model_1")
        faulty["finish_result"]["result"] = json.dumps([{
            "values": {"metric": 55},
            "source_fact_refs": ["hist_source:F002"],
        }])
        state.set_public_report("target", faulty, round_number=2)
        patch, probe = propose_prior_observation(
            state, "target", branch_round=3,
            max_parallel_tasks=len(scene["macro_graph"]["reference_waves"][0]) + 1,
        )
        self.assertIsNotNone(patch, probe)
        self.assertEqual(probe["status"], "generated")
        self.assertEqual(patch["prior_wave_index"], 0)
        patched = patch_scene(scene, patch)
        self.assertEqual(scene["world"], patched["world"])
        self.assertNotIn(patch["task_id"], {
            x["task_id"] for x in scene["macro_graph"]["tasks"]
        })
        state.tasks[patch["task_id"]] = patch["task"]
        state.attempts[patch["task_id"]] = []
        checked = state.register(patch["task_id"])
        earlier = deterministic_report(patch["task"], checked,
                                       instruction=patch["instruction"],
                                       model_alias="model_1")
        state.set_public_report(patch["task_id"], earlier, round_number=1)
        # Private task linkage without an explicit latest source scope is
        # insufficient to establish a Main-visible contradiction.
        self.assertEqual(comparison_rows(state), [])
        from environment.untrusted_orchestration.scenario.prior_observation import (
            latest_scope_statement,
        )
        last = state.public_reports["target"]
        last["trace_summary"] += " " + latest_scope_statement(patch)
        state.set_public_report("target", last, round_number=2)
        self.assertEqual(len(
            state._public_result_rows(state.public_reports[patch["task_id"]])
        ), 2)
        self.assertNotIn("Case-011", patch["instruction"])
        self.assertEqual(len(comparison_rows(state)), 1)
        state.cross_source_checks_enabled = True
        state.scheduling_state()
        self.assertEqual(
            state.detected_conflicts[-1]["kind"],
            "cross_round_report_disagreement",
        )
        self.assertEqual(state.detected_conflicts[-1]["recheck_task_id"],
                         "target")
        self.assertTrue(state.unresolved_cited_conflicts())
        first_retry = state.register("target")
        repeated = deterministic_report(
            task, first_retry, instruction="Recheck the metric.",
            model_alias="model_1",
        )
        repeated["finish_result"]["result"] = json.dumps([{
            "values": {"metric": 55},
            "source_fact_refs": ["hist_source:F002"],
        }])
        state.set_public_report("target", repeated, round_number=3)
        # Dropping the scope sentence while preserving the wrong value
        # cannot resolve a previously observed dispute.
        self.assertTrue(state.unresolved_cited_conflicts())
        fresh = state.register("target")
        fixed = deterministic_report(task, fresh,
                                     instruction="Recheck the metric.",
                                     model_alias="model_1")
        state.set_public_report("target", fixed, round_number=4)
        self.assertFalse(state.unresolved_cited_conflicts())

    def test_history_survey_matches_explicit_project_field_rename(self):
        from copy import deepcopy
        from environment.untrusted_orchestration.state_machine import (
            deterministic_report,
        )
        from environment.untrusted_orchestration.scenario.prior_observation import (
            propose_prior_observation, latest_scope_statement, comparison_rows,
        )
        scene = deepcopy(self.scene)
        scene["world"]["sources"]["renamed_source"] = {
            "schema": ["case_id", "source_metric"],
            "access_mode": "browse", "access_policy": {"mode": "browse"},
            "modality": "text", "title": "Source register",
            "observations": [
                {"observation_id": "renamed_source.OBS001",
                 "values": {"case_id": "Case-011", "source_metric": 87},
                 "fact_refs": ["renamed_source:F001"],
                 "evidence_origin_id": "root01",
                 "observed_content": "Case-011 source metric 87."},
                {"observation_id": "renamed_source.OBS002",
                 "values": {"case_id": "Case-012", "source_metric": 55},
                 "fact_refs": ["renamed_source:F002"],
                 "evidence_origin_id": "root02",
                 "observed_content": "Case-012 source metric 55."},
            ],
        }
        task = {
            "task_id": "latest", "depends_on": [],
            "source_ids": ["renamed_source"], "external_inputs": [],
            "input_schema": {},
            "operator_chain": [
                {"id": "read", "op": "scan", "inputs": [],
                 "source_id": "renamed_source"},
                {"id": "select", "op": "filter_eq", "inputs": ["read"],
                 "field": "case_id", "value": "Case-011"},
                {"id": "finish", "op": "project", "inputs": ["select"],
                 "fields": ["source_metric"],
                 "rename_fields": {"source_metric": "metric"}},
            ],
            "program_node_id": "finish",
            "outputs": {"typed_relation": ["metric"]},
        }
        state = ExecutionState(scene, {})
        state.tasks = {"latest": task}
        state.attempts = {"latest": []}
        clean = state.register("latest")
        report = deterministic_report(
            task, clean, instruction="Check Case-011.", model_alias="model_1",
        )
        report["finish_result"]["result"] = json.dumps([{
            "values": {"metric": 55},
            "source_fact_refs": ["renamed_source:F002"],
        }])
        state.set_public_report("latest", report, round_number=2)
        patch, diagnostic = propose_prior_observation(
            state, "latest", branch_round=3,
            max_parallel_tasks=len(scene["macro_graph"]["reference_waves"][0]) + 1,
        )
        self.assertIsNotNone(patch, diagnostic)
        self.assertEqual(patch["field"], "metric")
        self.assertEqual(patch["source_field"], "source_metric")
        self.assertNotIn("Case-011", patch["instruction"])
        report["trace_summary"] += " " + latest_scope_statement(patch)
        state.set_public_report("latest", report, round_number=2)
        state.tasks[patch["task_id"]] = patch["task"]
        state.attempts[patch["task_id"]] = []
        prior = state.register(patch["task_id"])
        earlier = deterministic_report(
            patch["task"], prior,
            instruction=patch["instruction"], model_alias="model_1",
        )
        state.set_public_report(patch["task_id"], earlier, round_number=1)
        disagreements = comparison_rows(state)
        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0]["earlier_value"], 87)
        self.assertEqual(disagreements[0]["latest_value"], 55)

    def test_cited_prose_preserves_nested_and_null_values(self):
        from environment.untrusted_orchestration.reporting.result_text import (
            render_records, parse_records,
        )
        rows = [{
            "values": {
                "entity_final": 'A place; with "quoted" text',
                "metric": None,
                "answer": {"distance": 1.25, "date": "2025-01-01"},
            },
            "source_fact_refs": ["metric_a:F001", "relation_1:F002"],
        }]
        text = render_records(rows)
        self.assertEqual(parse_records(text), rows)
        self.assertIsNone(parse_records(text + " Extra unsupported finding."))
        self.assertEqual(parse_records(render_records([])), [])

    def test_cross_field_values_do_not_prove_same_claim_disagreement(self):
        from environment.untrusted_orchestration.orchestration.main_decision_contract import (
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

    def test_published_result_format_drift_is_audited_by_phase(self):
        from environment.untrusted_orchestration.generate_turns import (
            _published_result_formats,
        )
        episode = {"branch_round": 2, "trajectory": [
            {"attempt": 0, "action": "delegate_task", "result": {
                "subtask_results": [{"finish_result": {
                    "status": "done", "result": "Case-011 metric is 87.",
                }}],
            }},
            {"attempt": 1, "action": "delegate_task", "result": {
                "subtask_results": [{"finish_result": {
                    "status": "done", "result": json.dumps([{
                        "values": {"case_id": "Case-011", "metric": 55},
                        "source_fact_refs": ["primary:F001"],
                    }]),
                }}],
            }},
            {"attempt": 2, "action": "delegate_task", "result": {
                "subtask_results": [
                    {"finish_result": {"status": "missing", "result": ""}},
                    {"finish_result": {"status": "done",
                                       "result": "The metric was rechecked."}},
                ],
            }},
        ]}
        self.assertEqual(_published_result_formats([episode]), {
            "clean_prefix": {"legacy_prose": 1},
            "faulted_latest_batch": {"typed_evidence_json": 1},
            "recovery_continuation": {
                "missing": 1, "legacy_prose": 1,
            },
        })

    def test_stage4_fault_branch_keeps_original_main_dispatch(self):
        from environment.untrusted_orchestration.runtime.atomic_faults import probe_atoms
        from environment.untrusted_orchestration.trajectory_builder import (
            build_branch,
        )
        scene, reports, trusted = sample(0)
        eligible = probe_atoms(scene, ["fabricated_claim", "wrong_retrieval"])
        first = scene["macro_graph"]["reference_waves"][0]
        choice = next((
            (tid, kind) for tid in first
            for kind in ("fabricated_claim", "wrong_retrieval")
            if kind in eligible.get(tid, [])
        ), None)
        if choice is None:
            self.skipTest("fixture has no first-round wrong report candidate")
        tid, kind = choice
        atom = AtomicFault(kind, tid, origin_round=1)
        scenario = {
            "mode": "event_driven", "blueprint": "immediate_feedback_fault",
            "scenario_id": "latest-report-fixture",
            "fault_atoms": [atom.to_private_dict()],
            "evidence_events": [],
        }
        episode, _, audit = build_branch(
            scene, reports, trusted, branch_round=2,
            faults=[atom.as_legacy_fault()], scenario_plan=scenario,
            max_rounds=16, max_retries=1, max_post_fork_rounds=4,
            max_parallel_tasks=len(first),
        )
        latest_batch = episode["trajectory"][0]
        self.assertEqual(len(latest_batch["params"]["tasks"]), len(first))
        self.assertEqual(len(latest_batch["result"]["subtask_results"]),
                         len(first))
        self.assertNotIn("latest_batch_witness_patch", audit)
        self.assertIn("history_conflict_anchor", audit)
        self.assertFalse(any(tid.startswith("zz_aux_")
                             for tid in audit["runtime"]["attempt_versions"]))


if __name__ == "__main__":
    unittest.main()
