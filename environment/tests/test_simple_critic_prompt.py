"""No-network regression for the *prompt contract* of simple Stage 2.

These are deliberately NOT claims that a live Qwen reviewer would classify
the examples correctly. Model-level semantic precision is measured with
separate audited experiments on frozen candidate/reference pairs.
"""
from __future__ import annotations

import json
import unittest

from environment.question_generation.question_review import (
    _AUTHOR_SYSTEM, _CRITIC_SYSTEM, author_question, critique_question,
)


# Authentic minimal excerpts from clear mistakes in the first-50 diagnostic
# set. Keep these in the prompt regression suite so future editorial changes
# cannot silently lose all guidance for scope, owner or relation type.
CASES = {
    "and_to_or": {
        "reference": "Let the second group consist of entities in both entities with a start date of 1915 drawn from the first group and entities with an address of 898 Pine Avenue drawn from the first group.",
        "candidate": "The second group includes entities that either have a start date of 1915 or have an address of 898 Pine Avenue.",
        "problem": "One entity must satisfy both filters; candidate permits either.",
    },
    "owner_shift": {
        "reference": "organizations listed as targets of relation R from events with a logo of crimson mountain emblem",
        "candidate": "organizations listed as targets of relation R from events, with a logo of crimson mountain emblem",
        "problem": "Candidate can move the event's logo onto the organization.",
    },
    "branch_narrowing": {
        "reference": "entities with population P drawn from the second group or administrative areas with demonym D in the relevant records",
        "candidate": "entities from the second group with population P or demonym D",
        "problem": "Independent demonym branch is incorrectly restricted.",
    },
    "missing_upstream_condition": {
        "reference": "places recorded as locations of organizations with floruit 958 and end date 1945",
        "candidate": "places recorded as locations of organizations with floruit 958",
        "problem": "The organization end date constraint is omitted.",
    },
}


class PromptRegressionTests(unittest.TestCase):
    def test_author_prompt_is_task_first_and_source_safe(self):
        self.assertIn("benchmark question", _AUTHOR_SYSTEM)
        self.assertIn("delegation blueprint", _AUTHOR_SYSTEM)
        self.assertIn("mechanical reference is ONLY a fallback", _AUTHOR_SYSTEM)
        self.assertIn("AND vs branch-wise OR", _AUTHOR_SYSTEM)
        self.assertIn("discoverable", _AUTHOR_SYSTEM)
        self.assertIn("intermediate results", _AUTHOR_SYSTEM)
        self.assertNotIn("A modest improvement is sufficient", _AUTHOR_SYSTEM)

    def test_critic_prompt_checks_contract_and_independent_pools(self):
        self.assertIn("semantic contract is authoritative", _CRITIC_SYSTEM)
        self.assertIn("candidate pools", " ".join(_CRITIC_SYSTEM.split()))
        self.assertIn("relation hop", _CRITIC_SYSTEM)
        self.assertIn("group containing X", _CRITIC_SYSTEM)
        self.assertIn("subtask", _CRITIC_SYSTEM)
        self.assertNotIn("risk_groups", _CRITIC_SYSTEM)

    def test_original_and_candidate_text_reach_one_simple_critic(self):
        class CapturingBackend:
            def __init__(self):
                self.calls = []
            def complete_json(self, system, user, **kwargs):
                self.calls.append((system, json.loads(user), kwargs))
                return {
                    "verdict": "revise", "naturalness": "acceptable",
                    "differences": [{
                        "reference_excerpt": "in both",
                        "candidate_excerpt": "either",
                        "problem": "AND changed to OR.",
                        "minimal_fix": "Change either to both and retain both filters.",
                    }],
                    "summary": "One group is now a union instead of an intersection.",
                }
        backend = CapturingBackend()
        case = CASES["and_to_or"]
        feedback = critique_question(
            backend, case["reference"], case["candidate"],
        )
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(
            set(backend.calls[0][1]),
            {"reference_question", "candidate_question"},
        )
        self.assertEqual(backend.calls[0][1]["reference_question"], case["reference"])
        self.assertEqual(backend.calls[0][1]["candidate_question"], case["candidate"])
        self.assertEqual(feedback["differences"][0]["minimal_fix"],
                         "Change either to both and retain both filters.")

    def test_author_revision_receives_exact_local_difference(self):
        class Writer:
            def __init__(self):
                self.inputs = []
            def complete_json(self, system, user, **kwargs):
                self.inputs.append(json.loads(user))
                return {"question": self.inputs[-1]["previous_draft"]}
        writer = Writer()
        previous = CASES["owner_shift"]["candidate"]
        feedback = {
            "verdict": "revise", "naturalness": "acceptable",
            "differences": [{
                "reference_excerpt": "events with a logo",
                "candidate_excerpt": "events, with a logo",
                "problem": CASES["owner_shift"]["problem"],
                "minimal_fix": "Attach the logo to the events, not organizations.",
            }],
            "summary": "Logo filter owner changed.",
        }
        author_question(
            writer, CASES["owner_shift"]["reference"],
            previous=previous, feedback=feedback,
        )
        self.assertEqual(writer.inputs[0]["previous_draft"], previous)
        self.assertEqual(writer.inputs[0]["reviewer_feedback"], feedback)
        self.assertNotIn("candidate_probe_specs", writer.inputs[0])


if __name__ == "__main__":
    unittest.main()
