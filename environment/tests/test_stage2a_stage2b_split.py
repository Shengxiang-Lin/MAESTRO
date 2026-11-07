"""Stage 2A/2B no-network gate, tamper replay, local feedback and state isolation."""
from __future__ import annotations

import copy
import json
import unittest

from environment.question_generation.goals_pipeline import generate_goal_record
from environment.question_generation.goal_review import (
    GOAL_CERT_SCHEMA, deterministic_goal_issues, validate_goal_record,
)
from environment.question_generation.realization import _realize_one
from environment.question_generation.verification import (
    CERTIFIED_SCHEMA, validate_surface_certificate,
)
from environment.tests.test_stage2_surface_certification import (
    Stage2SurfaceCertificationTests, natural_rephrase, PASS_CRITIQUE,
)
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.eligibility import (
    controlled_question_reference,
)
from environment.question_generation.scoped_surface import build_scoped_plan
from environment.delegation_blueprint import compile_delegation_outline


class GoalWriter:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append(data)
        assert "current_fragment" in data, "GoalWriter must never author a question"
        fragment = data["current_fragment"]
        numeric = [
            str(s["public_value"]) for s in fragment["public_steps"]
            if s.get("public_value") is not None
        ]
        text = ("Investigate the relevant recorded entities and preserve "
                "the specified relation, filter, and return requirements.")
        if numeric:
            text += " Required literal values: " + ", ".join(numeric) + "."
        return {"goal": text}


class GoalCritic:
    def __init__(self, verdict="pass"):
        self.verdict = verdict
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append(data)
        assert "candidate_goal" in data
        return {
            "verdict": self.verdict,
            "issues": [] if self.verdict == "pass" else ["Missing local relation details"],
            "summary": "Local answer-defining conditions are preserved." if
                       self.verdict == "pass" else "Need a more specific local goal.",
        }


class QuestionOnlyWriter:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append(data)
        assert "reference_question" in data
        return {"question": natural_rephrase(data["reference_question"])}


class QuestionOnlyCritic:
    def complete_json(self, system, user, **kwargs):
        assert "candidate_question" in json.loads(user)
        return copy.deepcopy(PASS_CRITIQUE)


class SplitPipelineTests(unittest.TestCase):
    def context(self):
        payload, world, capability, rendered = (
            Stage2SurfaceCertificationTests().data()
        )
        gold = compile_gold_solution({
            "task_id": payload["task_id"],
            "question": rendered.question,
            "capability_world": capability,
        }, world, rendered, schedule="compact")
        controlled = controlled_question_reference(gold, world, rendered)
        plan = build_scoped_plan(world, rendered.question, gold, controlled)
        outline = compile_delegation_outline(
            gold, world, rendered.question, plan, payload["resources"],
        )
        return payload, world, capability, rendered, outline

    def test_2a_certifies_and_2b_reuses_with_zero_goal_calls(self):
        payload, world, capability, rendered, outline = self.context()
        goal_author, goal_critic = GoalWriter(), GoalCritic()
        ok, certified, audit = generate_goal_record(
            payload, world, capability, goal_author, goal_critic,
            max_goal_revisions=0, blueprint_workers=2,
        )
        self.assertTrue(ok, (certified, audit))
        self.assertEqual(
            certified["goal_certificate"]["schema"], GOAL_CERT_SCHEMA,
        )
        self.assertEqual(
            len(goal_author.calls), len(outline["tasks"]),
        )
        self.assertEqual(
            len(goal_critic.calls), len(outline["tasks"]),
        )
        validate_goal_record(
            certified, payload, world, rendered.question, outline,
            oracle_answer=rendered.answer,
        )
        self.assertTrue(all(
            "reference_for_public_literal_spelling_only" not in c
            and "task_goal" not in c["global_output"]
            for c in goal_author.calls
        ))

        writer = QuestionOnlyWriter()
        ok, question, audit = _realize_one(
            payload, world, capability, writer, QuestionOnlyCritic(),
            goal_record=certified, max_revisions=0,
        )
        self.assertTrue(ok, (question, audit))
        self.assertEqual(len(writer.calls), 1)
        self.assertEqual(
            question["surface_certificate"]["schema"], CERTIFIED_SCHEMA,
        )
        self.assertEqual(
            question["goal_certificate"], certified["goal_certificate"],
        )
        self.assertEqual(
            question["goal_reviews"], certified["goal_reviews"],
        )
        self.assertEqual(
            question["delegation_blueprint"], certified["delegation_blueprint"],
        )
        eligibility = validate_surface_certificate(
            question, world, rendered.question,
        )
        self.assertTrue(eligibility["eligible"])
        self.assertTrue(any(
            event.get("stage") == "certified_blueprint_reused" and
            event.get("author_calls") == 0 for event in audit
        ))

    def test_2a_rejects_ambiguous_local_semantics_without_question(self):
        payload, world, capability, rendered, outline = self.context()
        ok, rejected, audit = generate_goal_record(
            payload, world, capability, GoalWriter(),
            GoalCritic(verdict="revise"),
            max_goal_revisions=1, blueprint_workers=2,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_goal_fidelity")
        self.assertNotIn("delegation_blueprint", rejected)
        self.assertTrue(any(
            event.get("stage") == "fragment_goal"
            and event.get("author_calls") == 2 for event in audit
        ))

    def test_persistent_goal_critic_protocol_error_does_not_rewrite_goal(self):
        payload, world, capability, rendered, outline = self.context()

        class BadJsonCritic:
            def __init__(self):
                self.calls = []

            def complete_json(self, system, user, **kwargs):
                self.calls.append(json.loads(user))
                return {
                    "verdict": "pass", "issues": [],
                    "summary": "Valid semantics",
                    "coverage": [{"step_id": "made-up", "goal_quote": "fake"}],
                }

        author, critic = GoalWriter(), BadJsonCritic()
        ok, rejected, audit = generate_goal_record(
            payload, world, capability, author, critic,
            max_goal_revisions=2, blueprint_workers=2,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_goal_fidelity")
        self.assertEqual(
            len(author.calls), len(outline["tasks"]),
            "A malformed Critic must never cause the Author to regenerate a Goal",
        )
        self.assertEqual(len(critic.calls), 2 * len(outline["tasks"]))
        self.assertTrue(all(
            entry.get("failure") == "rejected_goal_protocol"
            for entry in audit if entry.get("stage") == "fragment_goal"
        ))

    def test_2b_rejects_tampering_before_question_author(self):
        payload, world, capability, rendered, outline = self.context()
        ok, certified, audit = generate_goal_record(
            payload, world, capability, GoalWriter(), GoalCritic(),
            max_goal_revisions=0,
        )
        self.assertTrue(ok, (certified, audit))
        edits = [
            lambda r: r["delegation_blueprint"]["goals"][0].update(
                goal="Another unrelated Goal that changes the target entity."
            ),
            lambda r: r["goal_certificate"].update(
                program_sha256="0" * 64
            ),
            lambda r: r["goal_reviews"][0]["review"].update(
                verdict="uncertain"
            ),
        ]
        for change in edits:
            with self.subTest(edit=repr(change)):
                forged = copy.deepcopy(certified)
                change(forged)
                writer = QuestionOnlyWriter()
                ok, rejected, audit = _realize_one(
                    payload, world, capability, writer, QuestionOnlyCritic(),
                    goal_record=forged, max_revisions=0,
                )
                self.assertFalse(ok)
                self.assertEqual(rejected["reason"], "rejected_goal_certificate")
                self.assertEqual(len(writer.calls), 0)

    def test_detect_foreign_branch_numeric_without_llm(self):
        fragment0 = {
            "fragment_id": "fragment:0", "depends_on": [],
            "public_steps": [{
                "id": "S1", "operation": "FILTER_NUMBER",
                "public_value": "1160", "comparison": "not equal to",
            }],
        }
        fragment1 = {
            "fragment_id": "fragment:1", "depends_on": [],
            "public_steps": [{
                "id": "S2", "operation": "FILTER_NUMBER",
                "public_value": "1945", "comparison": "equal to",
            }],
        }
        outline = {"tasks": [fragment0, fragment1]}
        self.assertIn(
            "not_equal_rewritten_as_ordered_comparison:1160",
            deterministic_goal_issues(
                outline, fragment0,
                "Find entities that have dates after 1160 and return 1945.",
            ),
        )
        self.assertIn(
            "unrelated_branch_numeric:1945",
            deterministic_goal_issues(
                outline, fragment0,
                "Find entities with dates different from 1160 and 1945.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
