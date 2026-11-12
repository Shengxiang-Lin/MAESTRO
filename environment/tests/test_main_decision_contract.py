"""Stage-4C Main-visible decision contract and guarded reasoning tests."""
from __future__ import annotations

import json
import unittest

from environment.base_orchestration.action_author import (
    author_delegate_action, proposed_template_action,
)
from environment.untrusted_orchestration.main_decision_contract import (
    build_main_decision_contract, critic_denies_confirmed_conflict,
)
from environment.untrusted_orchestration.reasoning_fidelity import (
    reasoning_evidence_issues,
)
from environment.untrusted_orchestration.publication_guard import (
    review_turn_surface,
)
from environment.untrusted_orchestration.trajectory_builder import _history_batch
from environment.untrusted_orchestration.state_machine import public_history_item



def goal(field: str, result: str, summary: str) -> dict:
    return {
        "operation": "verify_conflicting_evidence",
        "target_fields": [field],
        "main_visible_justification": (
            f"The delivered result and summary disagree about "
            f"{field.replace('_', ' ')}: result lists {result}, "
            f"but the summary says {summary} for the returned record. "
            "Check the field against its source."
        ),
    }


def report(result: str, summary: str) -> dict:
    return {
        "finish_result": {
            "status": "done", "result": result, "summary": summary,
        },
        "trace_summary": "A keyed lookup and projection were completed.",
    }


def history(results: list[dict], round_no: int = 5) -> list[dict]:
    return [{
        "attempt": round_no,
        "is_parallel_batch": True,
        "subtask_results": results,
    }]


def assignment(field: str) -> dict:
    return {
        "target_instruction": (
            f"Check the discrepancy in {field} using the authorized source. "
            "Return only the requested field and its provenance."
        ),
        "context": (
            "Previous Sub-Agent returns disagree about the field. "
            "The inconsistency remains unresolved, so obtain verification."
        ),
        "source_ids": [],
        "output_fields": [field],
        "focus_fields": [field],
        "model": "model_1", "tools": [],
        "needs_upstream": False,
    }


def review_pass() -> dict:
    return {
        "verdict": "pass", "grounded": True,
        "assignment_fidelity": True, "o1_style": True,
        "reason": "The additional rationale is grounded.",
        "repair_hint": "",
    }


class MainVisibleFactTests(unittest.TestCase):
    def test_actual_000081_year_result_summary_difference(self):
        rows = history([
            report('[{"exhibition_year": 1999}]',
                   "The summary records exhibition year = 2017 for the returned record."),
        ])
        contract = build_main_decision_contract(
            rows, [goal("exhibition_year", "1999", "2017")],
            number_of_tasks=1,
        )
        conflicts = contract["visible_evidence"][
            "confirmed_result_summary_conflicts"
        ]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["field"], "exhibition_year")
        self.assertEqual(conflicts[0]["result_value"], "1999")
        self.assertEqual(conflicts[0]["summary_value"], "2017")
        self.assertEqual(contract["next_decision"]["action"], "delegate_task")
        self.assertIn("1999", contract["locked_visible_fact_core"])
        self.assertIn("2017", contract["locked_visible_fact_core"])
        self.assertNotIn("gold", json.dumps(contract).lower())
        self.assertNotIn("teacher", json.dumps(contract).lower())

    def test_actual_000280_metric_a_result_summary_difference(self):
        rows = history([
            report('[{"case_id": "Case-006", "metric_a": 90}]',
                   "The summary records metric a = 21 for the returned record."),
            report("The lookup returned metric b = 25.",
                   "The metric b measurement is 25."),
        ], round_no=4)
        contract = build_main_decision_contract(
            rows, [goal("case_id", "Case-006", "Case-006"),
                   goal("metric_a", "90", "21")], number_of_tasks=1,
        )
        found = contract["visible_evidence"]["confirmed_result_summary_conflicts"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["field"], "metric_a")
        self.assertEqual((found[0]["result_value"], found[0]["summary_value"]),
                         ("90", "21"))

    def test_planner_claim_is_not_evidence_and_same_report_is_required(self):
        contracts = [
            build_main_decision_contract(
                history([report('[{"exhibition_year": 1999}]',
                                "The summary records exhibition year = 1999.")]),
                [goal("exhibition_year", "1999", "2017")], number_of_tasks=1,
            ),
            build_main_decision_contract(
                history([
                    report('[{"exhibition_year": 1999}]',
                           "The summary records exhibition year = 1999."),
                    report('[{"exhibition_year": 2017}]',
                           "The summary records exhibition year = 2017."),
                ]),
                [goal("exhibition_year", "1999", "2017")], number_of_tasks=1,
            ),
        ]
        for contract in contracts:
            self.assertEqual(
                contract["visible_evidence"]["confirmed_result_summary_conflicts"], []
            )
            self.assertEqual(contract["locked_visible_fact_core"], "")

    def test_old_conflict_not_promoted_after_new_clean_batch(self):
        h = history([
            report('[{"exhibition_year": 1999}]',
                   "exhibition year = 2017."),
        ], round_no=3) + history([
            report('[{"exhibition_year": 1999}]',
                   "exhibition year = 1999."),
        ], round_no=4)
        contract = build_main_decision_contract(
            h, [goal("exhibition_year", "1999", "2017")],
            number_of_tasks=1,
        )
        self.assertEqual(contract["visible_evidence"]["latest_round"], 4)
        self.assertFalse(contract["locked_visible_fact_core"])

    def test_partial_missing_status_is_visible_not_a_verified_value(self):
        h = history([
            {"finish_result": {"status": "partial",
                               "result": "The count field was not delivered.",
                               "summary": "The handoff is incomplete."}},
            {"finish_result": None, "trace_summary": "No result returned."},
        ])
        contract = build_main_decision_contract(
            h, [{"operation": "complete_partial_evidence",
                 "target_fields": ["count"],
                 "main_visible_justification": "The count is missing."}],
            number_of_tasks=1,
        )
        gaps = contract["visible_evidence"]["delivery_gaps"]
        self.assertEqual([g["status"] for g in gaps],
                         ["partial", "missing_finish_result"])
        self.assertEqual(contract["locked_visible_fact_core"], "")


class GuardedMainAuthorTests(unittest.TestCase):
    def setUp(self):
        self.goal = goal("exhibition_year", "1999", "2017")
        self.contract = build_main_decision_contract(
            history([report('[{"exhibition_year": 1999}]',
                            "The summary records exhibition year = 2017.")]),
            [self.goal], number_of_tasks=1,
        )
        self.assignments = [assignment("exhibition_year")]

    def call(self, responses, reviews, *, max_revisions=0):
        class Author:
            def __init__(self, entries):
                self.entries, self.payloads = iter(entries), []
            def complete_json(self, system, request):
                self.payloads.append(json.loads(request))
                return next(self.entries)

        class Critic:
            def __init__(self, entries):
                self.entries, self.payloads = iter(entries), []
                self.systems = []
            def complete_json(self, system, request):
                self.systems.append(system)
                self.payloads.append(json.loads(request))
                return next(self.entries)

        author, critic = Author(responses), Critic(reviews)
        action, status = author_delegate_action(
            assignments=self.assignments,
            prompt="QUESTION: Identify the exhibition year.",
            author=author, critic=critic,
            freeze_assignment_prose=True,
            reasoning_goals=[self.goal],
            decision_contract=self.contract,
            critic_conflict_objection=lambda review: (
                critic_denies_confirmed_conflict(review, self.contract)
            ),
            reasoning_guard=lambda text: reasoning_evidence_issues(
                text, [self.goal],
            ),
            max_revisions=max_revisions,
        )
        return action, status, author, critic

    def test_reasoning_only_author_uses_fixed_action_and_visible_fact_core(self):
        action, status, author, critic = self.call(
            [{"reasoning": (
                "I should recheck the conflicting exhibition-year values "
                "against the authorized archive before drawing a conclusion."
            )}],
            [review_pass()],
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertEqual(status["decision_contract_version"], "v10_4")
        self.assertEqual(status["confirmed_visible_conflicts"], 1)
        self.assertTrue(status["locked_visible_fact_core_applied"])
        self.assertEqual(action["action"], "delegate_task")
        self.assertEqual(action["params"],
                         proposed_template_action(self.assignments)["params"])
        self.assertIn("1999", action["reasoning"])
        self.assertIn("2017", action["reasoning"])
        self.assertEqual(
            critic.payloads[0]["locked_main_visible_decision_facts"],
            self.contract,
        )
        self.assertIn(
            "locked_main_visible_decision_facts",
            author.payloads[0],
        )
        self.assertEqual(
            critic.payloads[0]["candidate_reasoning_only"],
            "I should recheck the conflicting exhibition-year values "
            "against the authorized archive before drawing a conclusion.",
        )
        self.assertEqual(review_turn_surface({
            "input": {"subtask_history": "The result lists 1999; summary says 2017."},
            "output": action,
        }), [])

    def test_false_denial_gets_independent_review_without_auto_pass(self):
        false_denial = {
            "verdict": "fail", "grounded": False,
            "assignment_fidelity": False, "o1_style": False,
            "reason": "The reasoning invents a factual contradiction.",
            "repair_hint": "Use the result 1999 and complete instead.",
        }
        self.assertTrue(critic_denies_confirmed_conflict(
            false_denial, self.contract
        ))
        actions = [{"reasoning": (
            "The returned year is disputed; independently verify the "
            "field against the archive before proceeding."
        )}]
        action, status, _, critic = self.call(
            actions, [false_denial, review_pass()]
        )
        self.assertEqual(status["status"], "accepted")
        self.assertTrue(status["critic_conflict_rereviewed"])
        self.assertEqual(len(critic.payloads), 2)
        self.assertIn("SECOND INDEPENDENT REVIEW", critic.systems[1])
        self.assertEqual(action["action"], "delegate_task")

        action, status, _, critic = self.call(
            actions, [false_denial, false_denial]
        )
        self.assertIsNone(action)
        self.assertEqual(status["status"], "failed")
        self.assertTrue(status["critic_conflict_rereviewed"])
        self.assertEqual(len(critic.payloads), 2)

    def test_explicit_complete_is_rejected_not_silently_recast(self):
        class ShouldNotReview:
            def complete_json(self, system, request):
                raise AssertionError("Invalid complete cannot reach Critic")
        class Author:
            def complete_json(self, system, request):
                return {"action": "complete", "reasoning":
                        "The result is sufficient so I can now complete the task.",
                        "params": {"answer": "1999"}}
        action, status = author_delegate_action(
            assignments=self.assignments, prompt="QUESTION",
            author=Author(), critic=ShouldNotReview(),
            max_revisions=0, freeze_assignment_prose=True,
            reasoning_goals=[self.goal],
            decision_contract=self.contract,
        )
        self.assertIsNone(action)
        self.assertEqual(
            status["failures"][0]["errors"],
            ["expected_delegate_task_action"],
        )

    def test_unsupported_reasoning_or_generation_leak_still_rejected(self):
        action, status, _, critic = self.call(
            [{"reasoning": (
                "The critic correctly identified the field mismatch and "
                "I will recheck the source before continuing."
            )}], [], max_revisions=0,
        )
        self.assertIsNone(action)
        self.assertEqual(len(critic.payloads), 0)
        self.assertIn("generation_process_leak",
                      status["failures"][0]["errors"])
        action, status, _, critic = self.call(
            [{"reasoning": (
                "The conflict is already verified; we can now complete "
                "the task without a further check."
            )}], [], max_revisions=0,
        )
        self.assertIsNone(action)
        self.assertEqual(len(critic.payloads), 0)
        self.assertTrue(any(
            failure["stage"] == "evidence_reasoning_guard"
            for failure in status["failures"]
        ))

    def test_unconfirmed_planner_claim_does_not_gain_fact_core(self):
        unproven = build_main_decision_contract(
            history([report('[{"exhibition_year": 1999}]',
                            "The summary records exhibition year = 1999.")]),
            [self.goal], number_of_tasks=1,
        )
        self.assertFalse(critic_denies_confirmed_conflict(
            {"verdict": "fail", "reason": "No conflict exists."}, unproven,
        ))


def executed_report(*, result: str | None, summary: str | None,
                    status: str = "done") -> dict:
    """Representative post-4B report, before the real O1 history adapter."""
    return {
        "task_instruction": "Look up the requested field in the source.",
        "model": "model_1", "steps_taken": 2,
        "statistics": {"max_steps": 30},
        "finish_result": ({
            "status": status, "result": result, "summary": summary,
        } if result is not None else None),
        "trace_summary": "The assigned key lookup and projection completed.",
    }


class MainVisibleO1HistoryIntegrationTests(unittest.TestCase):
    """Real _history_batch -> public_history_item -> contract, not fake nesting."""

    def test_actual_000081_flattened_history_confirms_year_conflict(self):
        raw = executed_report(
            result='[{"exhibition_year": 1999}]',
            summary=(
                "The summary records exhibition year = 2017 "
                "for the returned record."
            ),
        )
        entry = _history_batch(5, [raw])
        flat = entry["subtask_results"][0]
        self.assertNotIn("finish_result", flat)
        self.assertEqual(flat["result"], raw["finish_result"]["result"])
        self.assertEqual(flat["summary"], raw["finish_result"]["summary"])
        contract = build_main_decision_contract(
            [entry], [goal("exhibition_year", "1999", "2017")],
            number_of_tasks=1,
        )
        self.assertEqual(contract["visible_evidence"]["delivery_gaps"], [])
        conflicts = contract["visible_evidence"][
            "confirmed_result_summary_conflicts"
        ]
        self.assertEqual(
            [(v["field"], v["result_value"], v["summary_value"])
             for v in conflicts],
            [("exhibition_year", "1999", "2017")],
        )
        self.assertIn("1999", contract["locked_visible_fact_core"])
        self.assertIn("2017", contract["locked_visible_fact_core"])

    def test_actual_000280_parallel_batch_identifies_only_metric_conflict(self):
        raw = [
            executed_report(
                result='[{"case_id": "Case-006", "metric_a": 90}]',
                summary=(
                    "The summary records metric a = 21 "
                    "for the returned record."
                ),
            ),
            executed_report(
                result="The lookup for Case-006 returned metric b = 25.",
                summary="The metric b measurement is 25.",
            ),
        ]
        entry = _history_batch(4, raw)
        flat = entry["subtask_results"]
        self.assertEqual(len(flat), 2)
        self.assertFalse(any("finish_result" in item for item in flat))
        contract = build_main_decision_contract(
            [entry],
            [
                goal("case_id", "Case-006", "Case-006"),
                goal("metric_a", "90", "21"),
            ],
            number_of_tasks=1,
        )
        conflicts = contract["visible_evidence"][
            "confirmed_result_summary_conflicts"
        ]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["field"], "metric_a")
        self.assertEqual(conflicts[0]["result_value"], "90")
        self.assertEqual(conflicts[0]["summary_value"], "21")
        self.assertEqual(contract["visible_evidence"]["delivery_gaps"], [])

    def test_flat_missing_and_partial_not_spuriously_marked_missing_wrapper(self):
        missing = executed_report(result=None, summary=None)
        partial = executed_report(
            result="The delivered record contains case id: Case-006, metric a: none.",
            summary="The returned evidence is incomplete.",
            status="partial",
        )
        entry = _history_batch(5, [missing, partial])
        self.assertEqual([x["status"] for x in entry["subtask_results"]],
                         ["missing", "partial"])
        contract = build_main_decision_contract(
            [entry],
            [{
                "operation": "complete_partial_evidence",
                "target_fields": ["metric_a"],
                "main_visible_justification": "The metric is not delivered.",
            }],
            number_of_tasks=1,
        )
        gaps = contract["visible_evidence"]["delivery_gaps"]
        self.assertEqual(
            [x["status"] for x in gaps],
            ["missing", "partial"],
        )
        self.assertNotIn(
            "missing_finish_result", [x["status"] for x in gaps],
        )
        self.assertEqual(contract["locked_visible_fact_core"], "")

    def test_clean_latest_batch_does_not_inherit_old_conflict(self):
        conflicted = _history_batch(3, [
            executed_report(
                result='[{"exhibition_year": 1999}]',
                summary="The summary records exhibition year = 2017.",
            ),
        ])
        repaired = _history_batch(4, [
            executed_report(
                result='[{"exhibition_year": 1999}]',
                summary="The summary records exhibition year = 1999.",
            ),
        ])
        contract = build_main_decision_contract(
            [conflicted, repaired],
            [goal("exhibition_year", "1999", "2017")],
            number_of_tasks=1,
        )
        self.assertFalse(contract["visible_evidence"][
            "confirmed_result_summary_conflicts"
        ])
        self.assertEqual(contract["visible_evidence"]["latest_round"], 4)

    def test_false_critic_denial_from_real_o1_history_gets_rereview(self):
        case = GuardedMainAuthorTests()
        case.setUp()
        case.contract = build_main_decision_contract(
            [_history_batch(4, [
                executed_report(
                    result='[{"case_id": "Case-006", "metric_a": 90}]',
                    summary=(
                        "The summary records metric a = 21 "
                        "for the returned record."
                    ),
                ),
            ])],
            [goal("metric_a", "90", "21")],
            number_of_tasks=1,
        )
        case.goal = goal("metric_a", "90", "21")
        case.assignments = [assignment("metric_a")]
        denial = {
            "verdict": "fail", "grounded": False,
            "assignment_fidelity": False, "o1_style": False,
            "reason": (
                "The Author's reasoning invents a factual contradiction "
                "between Result=90 and Summary=21."
            ),
            "repair_hint": "Use the Result directly.",
        }
        self.assertTrue(critic_denies_confirmed_conflict(denial, case.contract))
        accepted, meta, _, critic = case.call(
            [{"reasoning": (
                "The reported metric is inconsistent, so the evidence "
                "must be rechecked against the permitted source."
            )}],
            [denial, review_pass()],
            max_revisions=0,
        )
        self.assertEqual(meta["status"], "accepted", meta)
        self.assertTrue(meta["critic_conflict_rereviewed"])
        self.assertEqual(len(critic.payloads), 2)
        self.assertEqual(meta["confirmed_visible_conflicts"], 1)
        self.assertIn("90", accepted["reasoning"])
        self.assertIn("21", accepted["reasoning"])

    def test_unproven_flat_history_does_not_trigger_special_rereview(self):
        clean = _history_batch(2, [
            executed_report(
                result='[{"metric_a": 90}]',
                summary="The metric a = 90 for the returned record.",
            ),
        ])
        contract = build_main_decision_contract(
            [clean], [goal("metric_a", "90", "21")],
            number_of_tasks=1,
        )
        denial = {
            "verdict": "fail",
            "reason": "No conflict exists in the visible report.",
            "repair_hint": "Do not invent a discrepancy.",
        }
        self.assertFalse(critic_denies_confirmed_conflict(denial, contract))


if __name__ == "__main__":
    unittest.main()
