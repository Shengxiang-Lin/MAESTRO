"""V10.3.1 regression: no private reviewer feedback in public Main SFT.

Uses actual leak phrasing from V10.3's published Main reasoning.
The shared pattern must prevent acceptance at the Author stage AND
independently reject an otherwise-clean branch at publication time.
"""
from __future__ import annotations

import unittest

from environment.base_orchestration.action_author import (
    action_issues, author_delegate_action, completion_issues,
    proposed_template_action,
)
from environment.base_orchestration.generation_meta import (
    generation_process_leak,
)
from environment.untrusted_orchestration.publication_guard import (
    review_turn_surface,
)


LEAKS = [
    "The critic correctly identified that the key is not verified.",
    "The critic feedback correctly noted that no search was performed.",
    "The critic correctly notes the missing review status.",
    "The critic feedback confirms that the field is missing.",
    "The critic feedback indicates the current result is incomplete.",
    "As the reviewer identified, the draft was incorrect.",
    "The critic's assessment said that the evidence was incomplete.",
    "According to the critic feedback, we must retry.",
    "The evidence planner has selected the next investigation.",
    "The evidence planner already chose a delegate_task action.",
    "The pending request is requested_not_executed.",
    "The locked decision requires a lookup.",
    "The plan has been selected by the compiler.",
    "This is a code-owned decision contract.",
]


def assignment() -> dict:
    return {
        "target_instruction": (
            "Read the assigned source records and return the requested "
            "marker code without asserting a previously unverified match."
        ),
        "context": (
            "Current evidence: the marker code is not yet established "
            "by the previously delivered report."
        ),
        "source_ids": [], "output_fields": ["marker_code"],
        "focus_fields": ["marker_code"], "model": "model_1",
        "tools": [], "needs_upstream": False,
    }


def good_review() -> dict:
    return {
        "verdict": "pass", "grounded": True,
        "assignment_fidelity": True, "o1_style": True,
        "reason": "The current visible gap is described correctly.",
        "repair_hint": "",
    }


class PublicationLeakRegressionTests(unittest.TestCase):
    def test_actual_v10_3_leak_variants_are_all_detected(self):
        for leak in LEAKS:
            with self.subTest(leak=leak):
                self.assertTrue(generation_process_leak(leak))
                self.assertIn(
                    "generation_process_leak:reasoning",
                    review_turn_surface({
                        "input": {"subtask_history": "Sub-Agent returned a gap."},
                        "output": {
                            "action": "delegate_task",
                            "reasoning": leak,
                            "params": {"tasks": []},
                        },
                    }),
                )

    def test_normal_main_planning_language_is_not_internal_machinery(self):
        for prose in (
            "The previous reports disagree, so I should request a source check.",
            "The remaining task depends on the returned key.",
            "The report's plan lists a scheduled exhibition in 2017.",
            "The review of the public archive reports an incomplete record.",
        ):
            with self.subTest(prose=prose):
                self.assertFalse(generation_process_leak(prose))

    def test_real_world_film_criticism_is_not_meta_feedback(self):
        prose = (
            "A film critic examined the poster and reported that it "
            "includes a red background; the agent should read the caption."
        )
        self.assertFalse(generation_process_leak(prose))
        self.assertEqual(
            review_turn_surface({
                "input": {"subtask_history": prose},
                "output": {
                    "action": "complete",
                    "reasoning": "The poster description is sufficient.",
                    "params": {"answer": "red"},
                },
            }), [],
        )

    def test_input_history_and_assignment_are_also_scanned(self):
        self.assertTrue(review_turn_surface({
            "input": {"subtask_history": "The critic feedback indicates a gap."},
            "output": {
                "action": "delegate_task",
                "reasoning": "The missing code should be retrieved.",
                "params": {"tasks": []},
            },
        }))
        self.assertIn(
            "generation_process_leak:task_0_context",
            review_turn_surface({
                "input": {"subtask_history": "No result was returned."},
                "output": {
                    "action": "delegate_task",
                    "reasoning": "The missing code should be retrieved.",
                    "params": {"tasks": [{
                        "task_instruction": "Read marker code.",
                        "context": "The critic correctly noted the gap.",
                    }]},
                },
            }),
        )

    def test_main_action_guard_rejects_then_retries_clean_reasoning(self):
        tasks = [assignment()]

        class Author:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, payload):
                self.calls += 1
                return {
                    "action": "delegate_task",
                    "reasoning": (
                        LEAKS[0] if self.calls == 1 else
                        "The previously returned evidence does not establish "
                        "the marker code. Delegate a targeted lookup to "
                        "retrieve the field before continuing."
                    ),
                    "params": {"tasks": []},
                }

        class Critic:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, payload):
                self.calls += 1
                return good_review()

        author, critic = Author(), Critic()
        action, status = author_delegate_action(
            assignments=tasks, prompt="Find the required marker code.",
            author=author, critic=critic, max_revisions=2,
            freeze_assignment_prose=True,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(author.calls, 2)
        self.assertEqual(critic.calls, 1)
        self.assertEqual(status["attempts"], 2)
        self.assertFalse(generation_process_leak(action["reasoning"]))
        self.assertEqual(action["params"], proposed_template_action(tasks)["params"])
        self.assertEqual(action_issues(action, tasks, prompt="Question"), [])

    def test_completion_guard_rejects_private_reviewer_language(self):
        delivered = {"work": {
            "finish_result": {
                "result": "The delivered record is Case-002.",
                "summary": "The case was returned by the index.",
            }
        }}
        good = (
            "The previously delivered result explicitly states "
            "'The delivered record is Case-002.' This is sufficient "
            "to complete the current question without another lookup."
        )
        bad = good + " The critic correctly identified the final answer."
        template = {
            "action": "complete", "params": {"answer": "Case-002"},
            "reasoning": good,
        }
        self.assertEqual(
            completion_issues(template, expected_answer="Case-002",
                              delivered=delivered), [],
        )
        self.assertIn(
            "completion_generation_or_private_leak",
            completion_issues({
                **template, "reasoning": bad,
            }, expected_answer="Case-002", delivered=delivered),
        )


if __name__ == "__main__":
    unittest.main()
