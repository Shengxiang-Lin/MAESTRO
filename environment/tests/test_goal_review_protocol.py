"""Regressions for the Stage 2A local semantic reviewer protocol.

The 50-task v1 pilot failed primarily because a 9B model cannot reliably
produce verbatim Goal quotes and step IDs for every compacted operation.
"""
import json
import unittest

from environment.question_generation.goal_review import (
    GOAL_CERT_SCHEMA, GOAL_REVIEW_METHOD, validate_review, review_fragment_goal,
    deterministic_goal_issues,
)
from environment.tests.test_fragment_goal_isolation import _outline


class SmallCritic:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        self.calls.append(json.loads(user))
        return self.reply


class GoalReviewProtocolTests(unittest.TestCase):
    def setUp(self):
        self.outline = _outline()
        self.fragment = self.outline["tasks"][0]
        self.goal = (
            "Find the public entity set satisfying the population number "
            "of exactly 10 and return the matching entities."
        )

    def test_concise_equivalence_review_passes_without_step_quotes_or_ids(self):
        reply = {
            "verdict": "pass",
            "issues": [],
            "summary": "The answer-defining numeric condition and output are intact.",
        }
        critic = SmallCritic(reply)
        reviewed = review_fragment_goal(
            critic, self.outline, self.fragment, "", self.goal,
        )
        self.assertEqual(reviewed, reply)
        self.assertEqual(len(critic.calls), 1)
        self.assertIn("candidate_goal", critic.calls[0])
        self.assertNotIn("reference_for_public_literal_spelling_only", critic.calls[0])
        self.assertIn("v2", GOAL_CERT_SCHEMA)
        self.assertIn("v2", GOAL_REVIEW_METHOD)

    def test_valid_deterministic_goal_error_skips_critic(self):
        critic = SmallCritic({
            "verdict": "pass", "issues": [], "summary": "No error.",
        })
        wrong = "Find entities with exactly 20 residents and return their names."
        review = review_fragment_goal(
            critic, self.outline, self.fragment, "", wrong,
        )
        self.assertEqual(review["verdict"], "revise")
        self.assertTrue(any("missing_local_numeric:10" in i
                            for i in review["issues"]))
        self.assertEqual(critic.calls, [])

    def test_invalid_critic_json_retries_same_goal_not_new_goal(self):
        good = {
            "verdict": "pass", "issues": [],
            "summary": "The local output, number and comparison are preserved.",
        }

        class OnceMalformed(SmallCritic):
            def complete_json(self, system, user, **kwargs):
                packet = json.loads(user)
                self.calls.append(packet)
                if len(self.calls) == 1:
                    return {**good, "coverage": []}
                return good

        critic = OnceMalformed(good)
        audit = {}
        result = review_fragment_goal(
            critic, self.outline, self.fragment, "", self.goal, audit=audit,
        )
        self.assertEqual(result, good)
        self.assertEqual(audit["critic_calls"], 2)
        self.assertEqual(len(audit["protocol_failures"]), 1)
        self.assertEqual(
            critic.calls[0]["candidate_goal"],
            critic.calls[1]["candidate_goal"],
        )

    def test_revisions_still_need_specific_issues(self):
        with self.assertRaisesRegex(ValueError, "revision_without_issue"):
            validate_review({
                "verdict": "revise", "issues": [],
                "summary": "This goal is semantically wrong.",
            }, self.fragment, self.goal)

    def test_critic_cannot_pass_with_explicit_issues(self):
        with self.assertRaisesRegex(ValueError, "pass_with_issues"):
            validate_review({
                "verdict": "pass", "issues": ["AND incorrectly replaced by OR"],
                "summary": "Contains an answer-changing error.",
            }, self.fragment, self.goal)

    def test_protocol_failures_remain_uncertain_not_certified(self):
        bad = {
            "verdict": "pass", "issues": [], "summary": "Good.",
            "coverage": [{"step_id": "S123",
                          "goal_quote": "hallucinated quotation"}],
        }
        critic = SmallCritic(bad)
        reviewed = review_fragment_goal(
            critic, self.outline, self.fragment, "", self.goal,
        )
        self.assertEqual(reviewed["verdict"], "uncertain")
        self.assertIn("protocol_error", reviewed["summary"])

    def test_null_public_entity_type_does_not_cancel_filter(self):
        from environment.question_generation.goal_review import _REVIEW_SYSTEM
        self.assertIn("WITHHELD", _REVIEW_SYSTEM)
        self.assertIn("FILTER_TYPE", _REVIEW_SYSTEM)
        self.assertIn("satisfying A and B", _REVIEW_SYSTEM)


if __name__ == "__main__":
    unittest.main()
