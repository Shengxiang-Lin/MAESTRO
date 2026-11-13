"""Realistic Sub-Agent finish reporting over frozen Oracle Macro evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.streaming import StreamingError

from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate
from environment.implement_subtask.config import (
    DEFAULT_CONFIG, load_reporting_config,
)
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.naturalize import naturalize_subtasks
from environment.implement_subtask.reporting import author_critic_report, trace_narrative


def scene(i, difficulty="medium"):
    spec = make_spec(
        seed_record={
            "record_id": f"reporting-{i}",
            "topic": f"Fictional Study {i}",
            "domain_hint": "science",
            "source_modalities": ["table", "image", "audio", "text"],
        }, ordinal=i, run_seed=17, difficulty=difficulty,
    )
    result = compile_and_validate(spec, reference_question(spec))
    result["question_authoring"] = {"status": "accepted"}
    result["question_origin"] = "llm_accepted_frozen_spec"
    return result


class ReportAuthor:
    def __init__(self, *, fail_topic=None, forged_tool=False):
        self.calls = []
        self.fail_topic = fail_topic
        self.forged_tool = forged_tool

    def complete_json(self, system_prompt, user_prompt):
        contract = json.loads(user_prompt)["frozen_task_and_evidence"]
        self.calls.append(contract)
        answer_count = len(contract["exact_oracle_answer"])
        if (self.fail_topic is not None
                and self.fail_topic in contract["overall_question_for_context_only"]):
            return {
                "result": "I could not verify the requested records.",
                "summary": "The required source details could not be located in the assigned evidence.",
                "status": "partial",
            }
        return {
            "result": (
                "Checked the assigned source records and identified "
                f"{answer_count} matching row(s); "
                + ("no qualifying records were found." if answer_count == 0
                   else "the exact items are in the handoff.")
                + (" Used GoogleSearchAction." if self.forged_tool else "")
            ),
            "summary": (
                "Compared the relevant source observations with upstream "
                "workproducts and applied the recorded criteria. "
                f"The resulting handoff contains {answer_count} row(s); "
                "no additional unsupported eligibility condition was assumed."
            ),
            "status": "done",
        }


class ReportCritic:
    def __init__(self, *, reject_first=False):
        self.calls = []
        self.reject_first = reject_first

    def complete_json(self, system_prompt, user_prompt):
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        passed = not (self.reject_first and len(self.calls) == 1)
        return {
            "verdict": "pass" if passed else "fail",
            "grounded": passed,
            "answer_scope_preserved": True,
            "evidence_scope_respected": True,
            "natural": True,
            "reason": "Supported by frozen evidence." if passed
                      else "Explain source comparison more precisely.",
            "repair_hint": "Mention the actual matching workproduct.",
        }


class StructuredResultAuthor:
    """Mock the exact JSON-array pattern observed in the user's Qwen logs."""
    def __init__(self, *, wrong_subset=False, collapse_labels=False):
        self.calls = []
        self.wrong_subset = wrong_subset
        self.collapse_labels = collapse_labels

    def complete_json(self, system_prompt, user_prompt):
        request = json.loads(user_prompt)
        contract = request["frozen_task_and_evidence"]
        self.calls.append(contract)
        rows = deepcopy(contract["exact_oracle_answer"])
        if self.wrong_subset:
            rows = rows[:-1] if rows else [{"entry_label": "Invented"}]
        if self.collapse_labels and rows and all(
            set(item) == {"entry_label"} for item in rows
        ):
            rows = [item["entry_label"] for item in rows]
        return {
            "result": rows,
            "summary": (
                "Checked the assigned source observations against the "
                "provided upstream workproducts and returned the exact "
                "matching records without adding unrelated filters."
            ),
            "status": "done",
        }


class StageLocalScopeRegressions(unittest.TestCase):
    """Examples from the user's 30/31 Qwen trial: do not claim later work."""

    def _contract(self, rows, operations, fields, *, parents=None,
                  terminal=False):
        return {
            "exact_oracle_answer": rows,
            "terminal_output_schema_of_this_macro": fields,
            "executed_operations_in_this_macro": operations,
            "upstream_handoffs": parents or [],
            "macro_is_terminal": terminal,
        }

    def test_join_must_not_pretend_inactive_rows_are_filtered(self):
        from environment.implement_subtask.reporting import _stage_scope_issues
        contract = self._contract(
            [{"entry_label": "Entry-03", "active": False,
              "measured_value": 58, "group_id": "Collection group D"},
             {"entry_label": "Entry-04", "active": True,
              "measured_value": 0, "group_id": "Collection group C"}],
            ["join"],
            ["entry_label", "active", "measured_value", "group_id"],
            parents=["work_register_rows", "work_codebook_rows"],
        )
        self.assertEqual(_stage_scope_issues(
            "Joined register and codebook candidates, preserving both "
            "active and inactive entries for subsequent checks.", contract), [])
        self.assertIn(
            "claims_all_active_with_inactive_rows",
            _stage_scope_issues(
                "The join confirmed all retrieved rows are active.", contract,
            ),
        )

    def test_annotation_join_cannot_claim_register_disagreement(self):
        from environment.implement_subtask.reporting import _stage_scope_issues
        contract = self._contract(
            [{"route_code": "R-01", "observed_status": True}],
            ["join"], ["route_code", "observed_status"],
        )
        contract["available_provenance_sources"] = ["annotations", "glossary"]
        self.assertIn(
            "claims_comparison_without_register_evidence",
            _stage_scope_issues(
                "The result contains four rows where the observed status differs "
                "from the master register.", contract,
            ),
        )
        self.assertEqual(
            _stage_scope_issues(
                "I aligned the codes with the observed group statuses for the "
                "next comparison.", contract,
            ), [],
        )

    def test_glossary_scan_does_not_approve_or_filter(self):
        from environment.implement_subtask.reporting import _stage_scope_issues
        contract = self._contract(
            [{"route_code": "R-01", "group_id": "Collection group A"}],
            ["scan"], ["route_code", "group_id"],
        )
        for summary, name in (
            ("Scanned the glossary for active entries with matching codes.",
             "scan_claims_unobserved_active_status"),
            ("Scanned the glossary source for approved translated category labels.",
             "scan_claims_unobserved_approval"),
            ("Scanned the glossary and filtered rows to retain only approvals.",
             "scan_claims_unexecuted_filter"),
            ("The retrieved rows satisfied clearance conditions.",
             "scan_claims_unchecked_clearance"),
        ):
            with self.subTest(summary=summary):
                self.assertIn(name, _stage_scope_issues(summary, contract))
        self.assertEqual(
            _stage_scope_issues(
                "Read the glossary and returned its route-to-group mappings. "
                "Approval and clearance still need separate checks.", contract
            ),
            [],
        )

    def test_scan_can_truthfully_say_filtering_was_not_performed(self):
        from environment.implement_subtask.reporting import (
            _scan_claims_completed_filter, _stage_scope_issues,
        )
        contract = self._contract(
            [{"group_id": "Collection group C", "minimum_required": 68},
             {"group_id": "Collection group B", "minimum_required": 71}],
            ["scan"], ["group_id", "minimum_required"],
        )
        valid_summaries = [
            "Scanned the policy source and returned all thresholds without "
            "filtering any records; subsequent checks belong to downstream Macros.",
            "Read both policy rows without excluding any categories.",
            "No rows were filtered or removed during the source scan.",
            "The returned policy thresholds are for downstream filtering.",
            "Rows to be filtered in the next stage were preserved here.",
            "The source was scanned before any filtering took place.",
            "Retrieved the policy relation with all records retained; "
            "no eligibility decisions were made.",
        ]
        for summary in valid_summaries:
            with self.subTest(summary=summary):
                self.assertFalse(_scan_claims_completed_filter(summary))
                self.assertNotIn(
                    "scan_claims_unexecuted_filter",
                    _stage_scope_issues(summary, contract),
                )

    def test_scan_still_rejects_positive_unexecuted_filter_claim(self):
        from environment.implement_subtask.reporting import (
            _scan_claims_completed_filter, _stage_scope_issues,
        )
        contract = self._contract(
            [{"group_id": "Collection group C", "minimum_required": 68}],
            ["scan"], ["group_id", "minimum_required"],
        )
        impossible = [
            "Scanned the policy table and filtered rows by approval status.",
            "Excluded rows without authorization after reading the policy table.",
            "The policy scan retained only eligible records.",
            "The source scan removed all rows below the required threshold.",
            "Filtered the policy table and did not change any other fields.",
        ]
        for summary in impossible:
            with self.subTest(summary=summary):
                self.assertTrue(_scan_claims_completed_filter(summary))
                self.assertIn(
                    "scan_claims_unexecuted_filter",
                    _stage_scope_issues(summary, contract),
                )

    def test_private_guard_failure_keeps_full_summary_for_debugging(self):
        class InvalidSummaryAuthor:
            def complete_json(self, system_prompt, user_prompt):
                teacher = json.loads(user_prompt)["frozen_task_and_evidence"]
                return {
                    "result": teacher["exact_oracle_answer"],
                    "status": "done",
                    "summary": (
                        "Scanned the source and filtered records to the "
                        "eligible subset, even though the current operation "
                        "is only a read-only source scan."
                    ),
                }
        frozen = scene(31, "easy")
        record = next(
            row for row in realize_scene(frozen)["macro_tasks"]
            if [step["operation"] for step in row["trace_summary"]["steps"]]
            == ["scan"]
        )
        reviewed, result = author_critic_report(
            record, question=frozen["question"], author=InvalidSummaryAuthor(),
            critic=ReportCritic(), max_revisions=0,
        )
        self.assertIsNone(reviewed)
        self.assertEqual(result["status"], "failed")
        self.assertIn("candidate_summary", result["failures"][0])
        self.assertIn("filtered records",
                      result["failures"][0]["candidate_summary"])
        self.assertEqual(result["failures"][0]["candidate_result_type"],
                         "list")

    def test_intermediate_join_does_not_claim_unexecuted_time_check(self):
        from environment.implement_subtask.reporting import _stage_scope_issues
        contract = self._contract(
            [{"entry_label": "Entry-02", "event_step": 68,
              "group_id": "Collection group B"}],
            ["join"], ["entry_label", "event_step", "group_id"],
            parents=["work_active_events", "work_codebook_rows"],
        )
        self.assertIn(
            "claims_unexecuted_temporal_verification",
            _stage_scope_issues(
                "The mapped records were verified against event step "
                "constraints and satisfy the session matching rules.",
                contract,
            ),
        )
        self.assertEqual(
            _stage_scope_issues(
                "Joined the codebook and register by route code, "
                "producing one mapped candidate for the next time-window check.",
                contract,
            ), [],
        )

    def test_critic_repair_must_not_change_oracle_result_rows(self):
        record = realize_scene(scene(27, "hard"))["macro_tasks"][0]
        author = ReportAuthor()
        critic = ReportCritic(reject_first=True)
        revised, review = author_critic_report(
            record, question="Check Fictional Study 27 records",
            author=author, critic=critic, max_revisions=1,
        )
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(revised["answer"], record["answer"])
        self.assertEqual(
            revised["answer_with_provenance"], record["answer_with_provenance"]
        )
        self.assertEqual(len(critic.calls), 2)

class VerifiedListCriticRegressions(unittest.TestCase):
    def test_second_critic_can_correct_oracle_membership_hallucination(self):
        from environment.implement_subtask.oracle_review import (
            critic_oracle_contradiction,
        )
        frozen = scene(25, "easy")
        realized = realize_scene(frozen)
        final = next(
            row for row in realized["macro_tasks"]
            if row["macro_task_id"] == realized["terminal_macro_task_id"]
        )
        self.assertTrue(final["answer"])
        self.assertTrue(all(set(x) == {"entry_label"} for x in final["answer"]))

        class OracleMistakenCritic:
            def __init__(self):
                self.calls = []

            def complete_json(self, system_prompt, user_prompt):
                payload = json.loads(user_prompt)
                self.calls.append((system_prompt, payload))
                if len(self.calls) == 1:
                    label = final["answer"][0]["entry_label"]
                    return {
                        "verdict": "fail", "grounded": False,
                        "answer_scope_preserved": False,
                        "evidence_scope_respected": True,
                        "natural": True,
                        "reason": f"{label} is absent from exact_oracle_answer",
                        "repair_hint": "remove that entry",
                    }
                return {
                    "verdict": "pass", "grounded": True,
                    "answer_scope_preserved": True,
                    "evidence_scope_respected": True,
                    "natural": True,
                    "reason": "The exact enumerated list is verified; summary is grounded.",
                    "repair_hint": "",
                }

        author = StructuredResultAuthor()
        critic = OracleMistakenCritic()
        reviewed, result = author_critic_report(
            final, question=frozen["question"], author=author,
            critic=critic, is_terminal=True, max_revisions=0,
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(critic.calls), 2)
        self.assertTrue(
            result["critic_oracle_contradiction_rechecked"]
        )
        self.assertEqual(reviewed["answer"], final["answer"])
        self.assertEqual(reviewed["answer_with_provenance"],
                         final["answer_with_provenance"])
        self.assertIsNotNone(critic_oracle_contradiction(
            {"reason": "Entry-01 is absent from exact_oracle_answer"},
            "Entries: Entry-01, Entry-02.",
            [{"entry_label": "Entry-01"}, {"entry_label": "Entry-02"}], {},
        ))
        self.assertIsNone(critic_oracle_contradiction(
            {"reason": "Entry-99 is absent from exact_oracle_answer"},
            "Entries: Entry-01, Entry-02.",
            [{"entry_label": "Entry-01"}, {"entry_label": "Entry-02"}], {},
        ))


class EntryEvidenceMappingRegressions(unittest.TestCase):
    def test_wrong_fact_ref_is_not_accepted_as_an_entry_status(self):
        from environment.implement_subtask.oracle_review import (
            critic_oracle_contradiction, mismatched_entry_refs,
        )
        facts = {
            "register:F005": {"entry_label": "Entry-06", "active": False},
            "register:F007": {"entry_label": "Entry-01", "active": True},
        }
        error = critic_oracle_contradiction(
            {"verdict": "fail",
             "reason": "Entry-01 (register:F005) has active=false according to the evidence"},
            "Entries: Entry-01, Entry-02.",
            [{"entry_label": "Entry-01"}, {"entry_label": "Entry-02"}],
            facts,
        )
        self.assertEqual(error["type"], "critic_entry_fact_ref_mismatch")
        self.assertEqual(error["mismatches"][0]["actual_entry"], "Entry-06")
        self.assertTrue(mismatched_entry_refs(
            "Entry-01 (register:F005)", facts
        ))
        self.assertFalse(mismatched_entry_refs(
            "Entry-01 (register:F007)", facts
        ))

    def test_wrong_author_entry_fact_ref_is_rejected_by_guard(self):
        from environment.implement_subtask.reporting import (
            _guard, _teacher_contract,
        )
        from environment.implement_subtask.oracle_review import (
            positive_entry_provenance,
        )
        frozen = scene(25, "easy")
        realized = realize_scene(frozen)
        teacher = next(
            r for r in realized["macro_tasks"]
            if r["macro_task_id"] == realized["terminal_macro_task_id"]
        )
        positive = positive_entry_provenance(teacher)
        if not positive:
            self.skipTest("no register provenance in sampled teacher")
        label, refs = next((k, v) for k, v in positive.items() if v)
        wrong = "register:F999"
        text = (
            "Checked the register entries and validated the output "
            f"with the incorrectly attributed {label} ({wrong})."
        )
        from environment.implement_subtask.reporting import _normalize_author_result
        rows = teacher["answer"]
        contract = _teacher_contract(teacher, question=frozen["question"],
                                     is_terminal=True)
        contract["all_evidence_refs"] = {
            wrong, *(
                ref for step in teacher["trace_summary"]["steps"]
                for ref in step["supporting_fact_refs"]
            ),
        }
        candidate = {
            "status": "done", "result": rows, "summary": text,
        }
        self.assertIn(
            "summary_entry_fact_ref_mismatch",
            _guard(candidate, contract),
        )

    def test_real_oracle_facts_map_to_correct_entries(self):
        from environment.implement_subtask.oracle_review import entry_fact_index
        frozen = scene(25, "easy")
        realized = realize_scene(frozen)
        teacher = next(
            record for record in realized["macro_tasks"]
            if record["macro_task_id"] == realized["terminal_macro_task_id"]
        )
        fact_index = entry_fact_index(teacher)
        for row in teacher["answer_with_provenance"]:
            label = row["values"].get("entry_label")
            if label:
                self.assertTrue(any(
                    fact_index.get(ref, {}).get("entry_label") == label
                    for ref in row["source_fact_refs"]
                ))


class ActiveStatusPrecisionRegressions(unittest.TestCase):
    """Examples sampled from 42 false-positive 'all...active' rejections."""

    def test_scan_field_descriptions_are_not_all_active_assertions(self):
        from environment.implement_subtask.reporting import _stage_scope_issues
        contract = {
            "exact_oracle_answer": [
                {"entry_label": "Entry-03", "active": False},
                {"entry_label": "Entry-04", "active": True},
            ],
            "terminal_output_schema_of_this_macro": [
                "entry_label", "active",
            ],
            "executed_operations_in_this_macro": ["scan"],
            "upstream_handoffs": [],
            "macro_is_terminal": False,
        }
        true_descriptions = [
            "Scanned all 11 rows containing entry_label, route_code, active and measured_value.",
            "Returned all candidates regardless of active status.",
            "The register scan retrieved all 9 records containing entry_label, "
            "route_code, active status, and event_step fields.",
            "Returned all rows including those with varying active states.",
            "Preserved all 10 rows, including those with active=false.",
            "Returned all records preserving both active and inactive entries.",
            "Retrieved all rows without applying eligibility filters such as active.",
            "The output contains both active and inactive entries.",
        ]
        for description in true_descriptions:
            with self.subTest(description=description):
                self.assertNotIn(
                    "claims_all_active_with_inactive_rows",
                    _stage_scope_issues(description, contract),
                )

    def test_explicit_claim_all_records_active_is_still_rejected(self):
        from environment.implement_subtask.reporting import _stage_scope_issues
        contract = {
            "exact_oracle_answer": [
                {"entry_label": "Entry-03", "active": False},
                {"entry_label": "Entry-04", "active": True},
            ],
            "terminal_output_schema_of_this_macro": [
                "entry_label", "active",
            ],
            "executed_operations_in_this_macro": ["scan"],
            "upstream_handoffs": [],
            "macro_is_terminal": False,
        }
        false_assertions = [
            "All ten rows are active.",
            "Every record is active.",
            "Only active entries were returned.",
            "The scan returned exclusively active rows.",
            "All returned entries were marked as active.",
            "The records were all active.",
        ]
        for description in false_assertions:
            with self.subTest(description=description):
                self.assertIn(
                    "claims_all_active_with_inactive_rows",
                    _stage_scope_issues(description, contract),
                )


class ReportingUnitTests(unittest.TestCase):
    def test_real_finish_shape_and_frozen_gold(self):
        frozen = scene(23, "hard")
        realized = realize_scene(frozen)
        record = realized["macro_tasks"][-1]
        original = deepcopy(record)
        author, critic = ReportAuthor(), ReportCritic()
        revised, info = author_critic_report(
            record, question=realized["question"],
            author=author, critic=critic,
            is_terminal=True,
        )
        self.assertEqual(info["status"], "accepted")
        self.assertEqual(info["attempts"], 1)
        self.assertEqual(record, original)
        self.assertEqual(revised["answer"], original["answer"])
        self.assertEqual(revised["answer_with_provenance"],
                         original["answer_with_provenance"])
        self.assertEqual(revised["trace_summary"]["steps"],
                         original["trace_summary"]["steps"])
        self.assertEqual(revised["finish_result"]["status"], "done")
        self.assertEqual(
            set(revised["finish_result"]), {"result", "status", "summary"}
        )
        self.assertNotIn("Verified result:", revised["finish_result"]["result"])
        self.assertNotIn(
            json.dumps(record["answer"], ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")),
            revised["finish_result"]["result"],
        )
        self.assertEqual(revised["answer_text"], revised["finish_result"]["result"])
        self.assertEqual(revised["trace_summary"]["narrative"],
                         trace_narrative(original["trace_summary"]))
        self.assertNotEqual(revised["trace_summary"]["narrative"],
                            revised["finish_result"]["summary"])
        self.assertTrue(author.calls[0]["macro_is_terminal"])
        self.assertNotIn("world", author.calls[0])
        self.assertNotIn("gold_program", author.calls[0])
        self.assertEqual(len(critic.calls), 1)

    def test_structured_result_exact_match_accepted_and_made_readable(self):
        frozen = scene(24, "medium")
        candidate = realize_scene(frozen)["macro_tasks"][0]
        author = StructuredResultAuthor()
        critic = ReportCritic()
        revised, review = author_critic_report(
            candidate, question=frozen["question"],
            author=author, critic=critic,
            max_revisions=0,
        )
        self.assertIsNotNone(revised)
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(revised["answer"], candidate["answer"])
        self.assertIsInstance(revised["finish_result"]["result"], str)
        self.assertNotIn("row(s) returned", revised["finish_result"]["result"])
        self.assertNotIn("entry_label", revised["finish_result"]["result"])
        self.assertIn(str(len(candidate["answer"])), revised["finish_result"]["result"])
        self.assertEqual(
            critic.calls[0]["candidate_finish"]["result"],
            revised["finish_result"]["result"],
        )
        self.assertTrue(critic.calls[0]["author_result_was_structured"])

    def test_structured_entry_label_list_is_accepted_only_when_exact(self):
        frozen = scene(25, "easy")
        realized = realize_scene(frozen)
        final = next(
            row for row in realized["macro_tasks"]
            if row["macro_task_id"] == realized["terminal_macro_task_id"]
        )
        self.assertTrue(all(
            set(item) == {"entry_label"} for item in final["answer"]
        ))
        revised, status = author_critic_report(
            final, question=frozen["question"],
            author=StructuredResultAuthor(collapse_labels=True),
            critic=ReportCritic(), max_revisions=0, is_terminal=True,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(revised["answer"], final["answer"])

    def test_wrong_subset_or_invented_extra_row_still_fails(self):
        frozen = scene(26, "hard")
        realized = realize_scene(frozen)
        record = next(
            row for row in realized["macro_tasks"]
            if row["answer"] and len(row["answer"]) > 1
        )
        reviewed, report = author_critic_report(
            record, question=frozen["question"],
            author=StructuredResultAuthor(wrong_subset=True),
            critic=ReportCritic(), max_revisions=1,
        )
        self.assertIsNone(reviewed)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["failures"]), 2)
        self.assertTrue(all(
            "structured_result_not_equal_to_oracle" in attempt["errors"]
            for attempt in report["failures"]
        ))

    def test_empty_structured_answer_still_done(self):
        from environment.implement_subtask.reporting import _normalize_author_result
        text, errors = _normalize_author_result([], [], ["entry_label"])
        self.assertEqual(errors, [])
        self.assertEqual(text, "No records matched the assigned task.")

    def test_partial_from_author_never_counts_as_complete(self):
        record = realize_scene(scene(10, "medium"))["macro_tasks"][0]
        author = ReportAuthor(fail_topic="Fictional Study 10")
        reviewed, info = author_critic_report(
            record, question="Fictional Study 10 tasks",
            author=author, critic=ReportCritic(),
            max_revisions=1,
        )
        self.assertIsNone(reviewed)
        self.assertEqual(info["status"], "failed")
        self.assertEqual(info["attempts"], 2)
        self.assertTrue(all(x["stage"] == "guard" for x in info["failures"]))

    def test_unperformed_tools_cannot_be_invented(self):
        record = realize_scene(scene(12, "medium"))["macro_tasks"][0]
        reviewed, info = author_critic_report(
            record, question="Check Fictional Study 12 records",
            author=ReportAuthor(forged_tool=True),
            critic=ReportCritic(), max_revisions=0,
        )
        self.assertIsNone(reviewed)
        self.assertIn("claims_unperformed_tool_or_media_action",
                      " ".join(info["failures"][0]["errors"]))

    def test_independent_critic_uses_targeted_revision(self):
        record = realize_scene(scene(11, "medium"))["macro_tasks"][0]
        author, critic = ReportAuthor(), ReportCritic(reject_first=True)
        reviewed, info = author_critic_report(
            record, question="Check Fictional Study 11 records",
            author=author, critic=critic, max_revisions=2,
        )
        self.assertIsNotNone(reviewed)
        self.assertEqual(info["attempts"], 2)
        self.assertEqual(len(critic.calls), 2)

    def test_config_has_model_roles_and_does_not_break_oracle_mode(self):
        cfg = load_reporting_config(DEFAULT_CONFIG)
        self.assertEqual(cfg["model"], "local-qwen3.5-9b")
        self.assertEqual(cfg["max_revisions"], 2)
        self.assertTrue(str(cfg["source_subtasks"]).endswith("subtasks.jsonl"))


class ReportingBatchTests(unittest.TestCase):
    def _teacher_inputs(self, root, ids):
        teacher_path = root / "oracle.jsonl"
        index_path = root / "index.jsonl"
        with (teacher_path.open("w", encoding="utf-8") as out,
              index_path.open("w", encoding="utf-8") as index):
            for i in ids:
                result = realize_scene(scene(i, "medium"))
                index.write(json.dumps({
                    k: v for k, v in result.items() if k != "macro_tasks"
                }, ensure_ascii=False) + "\n")
                for task in result["macro_tasks"]:
                    out.write(json.dumps(task, ensure_ascii=False) + "\n")
        return teacher_path, index_path

    def test_complete_scene_atomic_acceptance_and_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subtasks, index = self._teacher_inputs(root, [27, 28])
            author = ReportAuthor(fail_topic="Fictional Study 28")
            summary = naturalize_subtasks(
                source_subtasks=subtasks, source_scene_index=index,
                output_dir=root / "reviewed", api_base="http://unused/v1",
                model="mock", workers=3, max_revisions=0,
                no_progress=True,
                clients=(author, ReportCritic()),
            )
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["attempted_scenes"], 2)
            self.assertEqual(summary["accepted_scenes"], 1)
            self.assertEqual(summary["failed_scenes"], 1)
            self.assertGreater(summary["author_critic_failed_macros"], 0)
            rows = [
                json.loads(line) for line in
                (root / "reviewed/subtasks.jsonl").read_text().splitlines()
            ]
            self.assertTrue(rows)
            self.assertEqual({x["scene_task_id"] for x in rows},
                             {scene(27, "medium")["task_id"]})
            self.assertTrue(all("finish_result" in x for x in rows))
            self.assertEqual(len((root / "reviewed/scene_index.jsonl")
                                 .read_text().splitlines()), 1)
            self.assertEqual(len((root / "reviewed/failed.jsonl")
                                 .read_text().splitlines()), 1)
            all_reviews = [
                json.loads(line) for line in
                (root / "reviewed/review_audit.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(all_reviews), summary["attempted_macro_tasks"])
            self.assertTrue(any(
                row["status"] == "accepted" for row in all_reviews
            ))
            self.assertTrue(any(
                row["status"] == "failed" for row in all_reviews
            ))
            with self.assertRaises(FileExistsError):
                naturalize_subtasks(
                    source_subtasks=subtasks, source_scene_index=index,
                    output_dir=root / "reviewed",
                    api_base="http://unused/v1", model="mock",
                    clients=(ReportAuthor(), ReportCritic()),
                    no_progress=True,
                )

    def test_streaming_macro_journal_and_completed_scene_publication(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oracle, index = self._teacher_inputs(root, [36, 37])
            output = root / "reports"
            class ObservingAuthor(ReportAuthor):
                def complete_json(self, system_prompt, user_prompt):
                    result = super().complete_json(system_prompt, user_prompt)
                    if len(self.calls) > 1:
                        audit = output / "review_audit.jsonl"
                        if audit.exists():
                            self.saw_prior_audit = bool(audit.read_text().strip())
                    return result
            author = ObservingAuthor()
            completed = naturalize_subtasks(
                source_subtasks=oracle, source_scene_index=index,
                output_dir=output, api_base="http://mock/v1",
                model="mock", clients=(author, ReportCritic()),
                workers=1, no_progress=True, flush_every=1,
            )
            self.assertEqual(completed["status"], "done")
            self.assertTrue(getattr(author, "saw_prior_audit", False))
            self.assertEqual(
                len((output / "subagent_reports_accepted.jsonl").read_text().splitlines()),
                completed["author_critic_passed_macros"],
            )
            self.assertEqual(
                len((output / "review_audit.jsonl").read_text().splitlines()),
                completed["attempted_macro_tasks"],
            )
            self.assertEqual(
                len((output / "scene_index.jsonl").read_text().splitlines()), 2,
            )
            self.assertEqual(completed["completed_macro_tasks"],
                             completed["attempted_macro_tasks"])

    def test_interrupted_reports_resume_without_recalling_persisted_macros(self):
        from environment.implement_subtask.naturalize import (
            run_ordered as real_run_ordered,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oracle, index = self._teacher_inputs(root, [38, 39])
            output = root / "reports"
            scenes = [
                json.loads(line) for line in index.read_text().splitlines()
            ]
            stopping_at = scenes[0]["macro_task_count"] - 1
            class Interrupted(RuntimeError):
                pass
            def truncate_after_scene(*args, **kwargs):
                prior_on_commit = kwargs["on_commit"]
                def after_commit(index, ok, row, meta):
                    prior_on_commit(index, ok, row, meta)
                    if index == stopping_at:
                        raise Interrupted("simulated process stop")
                kwargs["on_commit"] = after_commit
                return real_run_ordered(*args, **kwargs)

            with patch(
                "environment.implement_subtask.naturalize.run_ordered",
                side_effect=truncate_after_scene,
            ):
                with self.assertRaises(Interrupted):
                    naturalize_subtasks(
                        source_subtasks=oracle, source_scene_index=index,
                        output_dir=output, api_base="http://mock/v1",
                        model="mock", clients=(ReportAuthor(), ReportCritic()),
                        workers=1, no_progress=True,
                    )
            # The first complete DAG must already be visible, even when
            # no additional scene has completed.
            self.assertEqual(
                len((output / "scene_index.jsonl").read_text().splitlines()), 1
            )
            self.assertEqual(
                len((output / "subtasks.jsonl").read_text().splitlines()),
                scenes[0]["macro_task_count"],
            )
            checkpoint = json.loads(
                (output / "subagent_reports_progress.json").read_text()
            )
            self.assertEqual(checkpoint["completed"], stopping_at + 1)
            self.assertEqual(
                json.loads((output / "summary.json").read_text())["status"],
                "running",
            )
            with self.assertRaises(StreamingError):
                naturalize_subtasks(
                    source_subtasks=oracle, source_scene_index=index,
                    output_dir=output, api_base="http://mock/v1",
                    model="other", clients=(ReportAuthor(), ReportCritic()),
                    workers=1, no_progress=True,
                    overwrite=False,
                )
            resumed_author = ReportAuthor()
            final = naturalize_subtasks(
                source_subtasks=oracle, source_scene_index=index,
                output_dir=output, api_base="http://mock/v1",
                model="mock", clients=(resumed_author, ReportCritic()),
                workers=1, no_progress=True,
            )
            self.assertEqual(final["status"], "done")
            self.assertEqual(final["accepted_scenes"], 2)
            self.assertEqual(len(resumed_author.calls),
                             final["attempted_macro_tasks"] - stopping_at - 1)
            self.assertEqual(
                len((output / "subtasks.jsonl").read_text().splitlines()),
                final["published_macro_tasks"],
            )
            with self.assertRaises(FileExistsError):
                naturalize_subtasks(
                    source_subtasks=oracle, source_scene_index=index,
                    output_dir=output, api_base="http://mock/v1",
                    model="mock", clients=(ReportAuthor(), ReportCritic()),
                    workers=1, no_progress=True,
                )

    def test_reuse_completed_prior_accepts_and_retry_only_failures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oracle, index = self._teacher_inputs(root, [46, 47])
            prior_dir, repaired_dir = root / "prior", root / "repaired"
            prior = naturalize_subtasks(
                source_subtasks=oracle, source_scene_index=index,
                output_dir=prior_dir, api_base="http://mock/v1",
                model="mock", workers=2, no_progress=True, max_revisions=0,
                clients=(ReportAuthor(fail_topic="Fictional Study 47"),
                         ReportCritic()),
            )
            self.assertEqual(prior["status"], "partial")
            self.assertGreater(prior["author_critic_passed_macros"], 0)
            self.assertGreater(prior["author_critic_failed_macros"], 0)
            author = ReportAuthor()
            recovered = naturalize_subtasks(
                source_subtasks=oracle, source_scene_index=index,
                output_dir=repaired_dir, api_base="http://mock/v1",
                model="mock", workers=3, no_progress=True, max_revisions=0,
                reuse_from=prior_dir, clients=(author, ReportCritic()),
            )
            self.assertEqual(recovered["status"], "done")
            self.assertEqual(recovered["accepted_scenes"], 2)
            self.assertEqual(recovered["author_critic_failed_macros"], 0)
            self.assertEqual(
                recovered["reused_accepted_macros"],
                prior["author_critic_passed_macros"],
            )
            self.assertEqual(
                len(author.calls), prior["author_critic_failed_macros"],
            )
            self.assertEqual(
                len((repaired_dir / "review_audit.jsonl").read_text().splitlines()),
                recovered["attempted_macro_tasks"],
            )
            self.assertEqual(
                len((repaired_dir / "subtasks.jsonl").read_text().splitlines()),
                recovered["attempted_macro_tasks"],
            )

    def test_reuse_rejects_tampered_previous_accepted_journal(self):
        from environment.streaming import StreamingError
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            oracle, index = self._teacher_inputs(root, [48])
            old = root / "prior"
            naturalize_subtasks(
                source_subtasks=oracle, source_scene_index=index,
                output_dir=old, api_base="http://mock/v1",
                model="mock", workers=1, no_progress=True,
                clients=(ReportAuthor(), ReportCritic()),
            )
            journal = old / "subagent_reports_accepted.jsonl"
            journal.write_bytes(journal.read_bytes() + b'{"extra":"tamper"}\n')
            # Uncommitted trailing data are permitted. Tampering with the
            # committed prefix must NOT be allowed.
            orig = journal.read_bytes()
            journal.write_bytes(orig.replace(b'"accepted"', b'"tampered"', 1))
            with self.assertRaises(StreamingError):
                naturalize_subtasks(
                    source_subtasks=oracle, source_scene_index=index,
                    output_dir=root / "repaired", api_base="http://mock/v1",
                    model="mock", reuse_from=old, workers=1, no_progress=True,
                    clients=(ReportAuthor(), ReportCritic()),
                )

    def test_successful_batch_retains_one_response_per_oracle_macro(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, idx = self._teacher_inputs(root, [30])
            summary = naturalize_subtasks(
                source_subtasks=source, source_scene_index=idx,
                output_dir=root / "natural", api_base="http://unused/v1",
                model="mock", clients=(ReportAuthor(), ReportCritic()),
                no_progress=True,
            )
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["failed_scenes"], 0)
            self.assertEqual(summary["attempted_macro_tasks"],
                             summary["published_macro_tasks"])
            self.assertEqual(summary["author_critic_failed_macros"], 0)


if __name__ == "__main__":
    unittest.main()
