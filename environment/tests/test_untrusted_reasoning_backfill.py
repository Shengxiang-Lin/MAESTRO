"""Main reasoning fidelity and causal candidate-backfill regression tests."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from environment.base_orchestration.action_author import author_delegate_action
from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.orchestration.reasoning_fidelity import (
    reasoning_evidence_issues,
)
from environment.untrusted_orchestration.scenario.candidate_selection import (
    select_recovered_candidates,
)

runner = importlib.import_module(
    "environment.untrusted_orchestration.generate_turns"
)


VERIFY = [{
    "operation": "verify_conflicting_evidence",
    "target_fields": ["marker_code"],
    "main_visible_justification": (
        "The returned joined result reports MK-6243 but earlier candidate "
        "sets disagree. Reconcile them before keyed lookup."
    ),
}]


class FakeAuthor:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete_json(self, system, payload):
        self.requests.append(json.loads(payload))
        return next(self.responses)


class FakeCritic:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.systems = []

    def complete_json(self, system, payload):
        self.systems.append(system)
        self.requests.append(json.loads(payload))
        return next(self.responses)


def passed():
    return {
        "verdict": "pass", "grounded": True,
        "assignment_fidelity": True, "o1_style": True,
        "reason": "The requested discrepancy is supported.",
        "repair_hint": "",
    }


def assignment():
    return {
        "target_instruction": (
            "Check the returned marker code against the previously "
            "reported visual and caption candidate lists."
        ),
        "context": "The prior reports disagree about the marker code.",
        "source_ids": [],
        "output_fields": ["marker_code"],
        "focus_fields": ["marker_code"],
        "model": "model_1",
        "tools": [],
        "needs_upstream": False,
    }


class Stage4ReasoningFidelityTests(unittest.TestCase):
    def test_reject_overconfident_join_while_verify_is_pending(self):
        incorrect = (
            "The previous round successfully resolved the conflicting "
            "candidate lists, confirming Case-002 with marker code MK-6243. "
            "This result is complete and verified. Continue with the table."
        )
        self.assertIn(
            "prematurely_claims_conflict_verified_or_resolved",
            reasoning_evidence_issues(incorrect, VERIFY),
        )
        correct = (
            "The new join returned MK-6243, but the two earlier reports "
            "disagree on that candidate. The binding remains unresolved; "
            "I will recheck the marker code against authorized evidence."
        )
        self.assertEqual(reasoning_evidence_issues(correct, VERIFY), [])
        self.assertEqual(reasoning_evidence_issues(incorrect, [{
            "operation": "retrieve_required_evidence"
        }]), [])

    def test_future_conditional_and_recheck_would_not_be_blocked(self):
        pending = (
            "The lookup reported KEY2-15385 while its upstream key was "
            "KEY2-17716. This mismatch needs another source check. Once "
            "this conflict is resolved, the completion rate can be calculated."
        )
        self.assertEqual(reasoning_evidence_issues(pending, VERIFY), [])
        separate = (
            "The two candidate sets were disjoint. To resolve this, I must "
            "re-examine the video evidence before the next lookup."
        )
        self.assertEqual(reasoning_evidence_issues(separate, VERIFY), [])

    def test_reasoning_guard_requests_revision_before_critic(self):
        wrong = (
            "The previous round successfully resolved the conflicting "
            "candidate lists. The result is fully verified."
        )
        correct = (
            "The result and summary disagree on the returned identifier. "
            "I will verify the marker code before using this value."
        )
        author = FakeAuthor([
            {"action": "delegate_task", "reasoning": wrong},
            {"action": "delegate_task", "reasoning": correct},
        ])
        critic = FakeCritic([passed()])
        action, status = author_delegate_action(
            assignments=[assignment()], prompt="History includes a conflict.",
            author=author, critic=critic, max_revisions=1,
            freeze_assignment_prose=True, reasoning_goals=VERIFY,
            reasoning_guard=lambda text: reasoning_evidence_issues(text, VERIFY),
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(status["attempts"], 2)
        self.assertEqual(len(critic.requests), 1)
        self.assertEqual(len(author.requests), 2)
        self.assertEqual(
            author.requests[1]["previous_feedback"][0]["stage"],
            "evidence_reasoning_guard",
        )
        self.assertEqual(action["reasoning"], correct)

    def test_critic_false_conflict_denial_gets_independent_rereview(self):
        goals = [{
            "operation": "verify_conflicting_evidence",
            "target_fields": ["match_year"],
            "main_visible_justification": (
                "The result lists 2008, but the summary says 2002."
            ),
        }]
        author = FakeAuthor([{
            "action": "delegate_task",
            "reasoning": (
                "The returned result lists match year 2008 but its "
                "summary says 2002. This disagreement requires a "
                "source check before the answer can be trusted."
            ),
        }])
        critic = FakeCritic([
            {"verdict": "fail", "grounded": False,
             "assignment_fidelity": False, "o1_style": True,
             "reason": "Invented discrepancy: both the result and summary say 2008",
             "repair_hint": "Do not invent a mismatch."},
            passed(),
        ])
        action, status = author_delegate_action(
            assignments=[assignment()],
            prompt="Result: match_year=2008; Summary: match year = 2002",
            author=author, critic=critic, max_revisions=0,
            freeze_assignment_prose=True, reasoning_goals=goals,
            reasoning_guard=lambda text: reasoning_evidence_issues(text, goals),
        )
        self.assertIsNotNone(action)
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(len(critic.requests), 2)
        self.assertIn("SECOND INDEPENDENT REVIEW", critic.systems[1])
        self.assertEqual(
            critic.requests[1]["main_visible_evidence_requests"], goals,
        )

    def test_conflict_rereview_does_not_override_repeated_rejection(self):
        author = FakeAuthor([{
            "action": "delegate_task",
            "reasoning": (
                "The joined result and earlier candidate lists disagree. "
                "I will verify the marker code before the next lookup."
            ),
        }])
        rejection = {
            "verdict": "fail", "grounded": False,
            "assignment_fidelity": False, "o1_style": True,
            "reason": "No discrepancy exists in the actual visible report",
            "repair_hint": "Insufficient evidence",
        }
        critic = FakeCritic([rejection, rejection])
        action, status = author_delegate_action(
            assignments=[assignment()], prompt="history",
            author=author, critic=critic, max_revisions=0,
            freeze_assignment_prose=True, reasoning_goals=VERIFY,
        )
        self.assertIsNone(action)
        self.assertEqual(status["status"], "failed")


class Stage4BackfillTests(unittest.TestCase):
    def test_failed_primary_uses_existing_second_4a_candidate(self):
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
            original = runner.author_branch_main
            calls = []

            def reject_first(*args, **kwargs):
                episode, turns, audit = original(*args, **kwargs)
                calls.append(audit["event_scenario"]["scenario_id"])
                if len(calls) == 1:
                    audit["outcome"] = "main_action_not_accepted"
                    audit["language_review_accepted"] = False
                    episode["outcome"] = "main_action_not_accepted"
                    episode["recovered"] = False
                    return episode, [], audit
                return episode, turns, audit

            with patch.object(runner, "author_branch_main",
                              side_effect=reject_first):
                summary = runner.generate_turns(
                    scenes_path=scenes, reports_path=report_path,
 output_dir=root / "out",
                    max_scenes=1, branches_per_scene=1, workers=2,
                    no_progress=True, scenario_mode="event_driven",
                    candidate_strategy="enumerate",
                    atomic_fault_kinds=[
                        "no_result", "incomplete_retrieval",
                        "wrong_retrieval", "wrong_entity_binding",
                        "report_missing", "result_summary_conflict",
                        "fabricated_claim",
                    ],
                    max_candidates_per_round=16,
                    max_distractors_per_fault=2,
                    max_selected_per_round=1,
                    max_language_attempts_per_round=2,
                    delayed_evidence_probability=0.0,
                )
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["candidate_language_backfill_attempts"], 1)
            self.assertEqual(summary["feedback_rounds_backfilled"], 1)
            self.assertEqual(summary["candidates_selected_for_language_review"], 2)
            self.assertEqual(summary["recovered"], 1)
            self.assertLessEqual(
                summary["main_actions_processed"],
                summary["main_actions_planned"],
            )
            self.assertGreaterEqual(summary["main_actions_skipped_downstream"], 0)
            selection = [
                json.loads(line) for line in
                (root / "out/private/candidate_selection.jsonl")
                .read_text().splitlines()
            ]
            self.assertEqual(sum(x["selected_for_language_review"]
                                 for x in selection), 2)
            self.assertEqual(
                sum(x.get("language_review_selection_reason") ==
                    "language_review_backfill_after_rejection"
                    for x in selection), 1,
            )


class CausalCandidateRankingTests(unittest.TestCase):
    def test_backfill_preserves_causal_rank_without_fault_name_reordering(self):
        kinds = [
            "result_summary_conflict",
            "result_summary_conflict",
            "wrong_retrieval",
        ]
        plans = [{
            "index": i, "sample_id": f"q-candidate-{i}",
            "task_id": "q", "branch_round": 2,
            "faults": [],
            "scenario_plan": {"fault_atoms": [{"kind": kind}]},
        } for i, kind in enumerate(kinds)]
        journals = [{
            "status": "recovered",
            "episode": {"trajectory": [{
                "attempt": 2, "action": "delegate_task"
            }]},
            "private_audit": {
                "execution_outcome": "recovered",
                "runtime": {},
                "fault_events_applied": [{}],
                "evidence_plan": [{
                    "evidence_requests": [{
                        "operation": "verify_conflicting_evidence",
                        "target_fields": [f"field_{i}"],
                        "authorized_source_ids": ["visual"],
                    }]
                }],
            },
        } for i in range(3)]
        selected, rows, _ = select_recovered_candidates(
            plans, journals, max_selected_per_round=1,
        )
        self.assertEqual(selected, {0})
        self.assertEqual(
            [r["candidate_index"] for r in
             sorted(rows, key=lambda r: r["recovery_rank"])],
            [0, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
