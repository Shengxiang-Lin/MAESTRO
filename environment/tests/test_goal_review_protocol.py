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
        self.assertIn("v5", GOAL_CERT_SCHEMA)
        self.assertIn("v5", GOAL_REVIEW_METHOD)

    def test_valid_deterministic_goal_error_skips_critic(self):
        critic = SmallCritic({
            "verdict": "pass", "issues": [], "summary": "No error.",
        })
        wrong = "Find entities with exactly 20 residents and return their names."
        review = review_fragment_goal(
            critic, self.outline, self.fragment, "", wrong,
        )
        self.assertEqual(review["verdict"], "revise")
        self.assertTrue(any("unrelated_branch_numeric:20" in i
                            for i in review["issues"]))
        self.assertEqual(critic.calls, [])

    def test_summarized_intent_can_omit_all_numeric_thresholds(self):
        concise = (
            "Identify the relevant public entities that satisfy the "
            "specified population criteria and return those entities."
        )
        critic = SmallCritic({
            "verdict": "pass", "issues": [],
            "summary": "The population-oriented entity search and output are aligned.",
        })
        self.assertEqual(
            deterministic_goal_issues(self.outline, self.fragment, concise), [],
        )
        review = review_fragment_goal(
            critic, self.outline, self.fragment, "", concise,
        )
        self.assertEqual(review["verdict"], "pass")
        self.assertEqual(len(critic.calls), 1)

    def test_explicit_not_equal_misread_is_still_rejected(self):
        fragment = {
            "fragment_id": "fragment:0", "depends_on": [],
            "public_steps": [{
                "id": "S1", "operation": "FILTER_YEAR",
                "public_value": "1160", "comparison": "not equal to",
            }],
        }
        outline = {"tasks": [fragment]}
        self.assertEqual(
            deterministic_goal_issues(
                outline, fragment,
                "Find relevant people meeting the specified date constraints.",
            ), [],
        )
        self.assertIn(
            "not_equal_rewritten_as_ordered_comparison:1160",
            deterministic_goal_issues(
                outline, fragment,
                "Find relevant people who died after 1160.",
            ),
        )

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
        self.assertIn("Public null labels", _REVIEW_SYSTEM)
        self.assertIn("ALLOWED abstraction", _REVIEW_SYSTEM)
        self.assertIn("OUTPUT", _REVIEW_SYSTEM)
        self.assertIn("NO CONTRADICTION", _REVIEW_SYSTEM)


if __name__ == "__main__":
    unittest.main()
