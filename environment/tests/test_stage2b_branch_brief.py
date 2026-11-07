"""No-network Stage2B Branch Brief compiler and independent full-contract Critic."""
from __future__ import annotations

import copy
import json
import unittest

from environment.delegation_blueprint import _structured_intent
from environment.question_generation.question_brief import compile_question_brief
from environment.question_generation.realization import _realize_one
from environment.question_generation.verification import validate_surface_certificate
from environment.tests.test_stage2_surface_certification import (
    Stage2SurfaceCertificationTests, PASS_CRITIQUE, natural_rephrase,
)
from environment.trajectory_simulation.compiler import compile_gold_solution


class CaptureBriefAuthor:
    def __init__(self, question):
        self.question = question
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        payload = json.loads(user)
        self.calls.append((system, payload))
        self_type = payload["question_brief"]
        assert self_type["workstreams"]
        return {"question": self.question}


class CaptureFullCritic:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        packet = json.loads(user)
        self.calls.append(packet)
        return copy.deepcopy(PASS_CRITIQUE)


class BranchQuestionBriefTests(unittest.TestCase):
    def fixture(self):
        source = Stage2SurfaceCertificationTests()
        payload, world, capability, rendered = source.data()
        record = source.certified(payload, world, capability)
        gold = compile_gold_solution({
            "task_id": payload["task_id"],
            "question": rendered.question,
            "capability_world": capability,
        }, world, rendered, schedule="compact")
        return payload, world, capability, rendered, gold, record

    def test_branch_brief_has_all_public_details_no_full_duplicate_inputs(self):
        payload, world, capability, rendered, gold, record = self.fixture()
        blueprint = record["delegation_blueprint"]
        brief = compile_question_brief(blueprint, world, rendered.question, gold)
        self.assertEqual(brief["schema"], "maestro.question_brief.branch_v1")
        self.assertTrue(brief["workstreams"])
        self.assertNotIn("task_semantics", brief)
        self.assertNotIn("reference_question", brief)
        self.assertNotIn("benchmark_task_brief", brief)
        all_steps = [
            step for stream in brief["workstreams"]
            for step in stream["precise_public_steps"]
        ]
        original = [
            step for task in blueprint["outline"]["tasks"]
            for step in task["public_steps"]
        ]
        self.assertEqual(len(all_steps), len(original))
        for step in all_steps:
            self.assertNotIn("goal", step)
            self.assertNotIn("capabilities", step)
        self.assertEqual(
            {stream["id"] for stream in brief["workstreams"]},
            set("B" + str(i + 1) for i in range(len(brief["workstreams"]))),
        )
        known = {stream["id"] for stream in brief["workstreams"]}
        for i, stream in enumerate(brief["workstreams"]):
            self.assertTrue(set(stream["depends_on"]) <=
                            {x["id"] for x in brief["workstreams"][:i]})
        self.assertIn("INTERSECT", brief["authoring_rules"])
        self.assertIn("UNION", brief["authoring_rules"])

    def test_stage2b_branch_author_gets_brief_but_critic_gets_reference(self):
        payload, world, capability, rendered, gold, record = self.fixture()
        author = CaptureBriefAuthor(natural_rephrase(rendered.question))
        critic = CaptureFullCritic()
        ok, row, audit = _realize_one(
            payload, world, capability, author, critic,
            goal_record=record, question_brief_mode="branch", max_revisions=0,
        )
        self.assertTrue(ok, (row, audit))
        self.assertEqual(len(author.calls), 1)
        system, packet = author.calls[0]
        self.assertEqual(set(packet), {"question_brief"})
        self.assertIn("PUBLIC branch-level Question Brief", system)
        self.assertEqual(len(critic.calls), 1)
        self.assertIn("reference_question", critic.calls[0])
        self.assertEqual(
            critic.calls[0]["reference_question"],
            audit[0]["scoped_question"],
        )
        self.assertEqual(
            row["authoring"]["question_brief_mode"], "branch",
        )
        self.assertIn(
            "question_brief_sha256", row["authoring"]["surface_review"],
        )
        replay = validate_surface_certificate(
            row, world, rendered.question,
        )
        self.assertTrue(replay["eligible"])

    def test_structured_intent_targets_are_local_not_global(self):
        blueprint = self.fixture()[-1]["delegation_blueprint"]
        outline = blueprint["outline"]
        for task in outline["tasks"]:
            spec = _structured_intent(outline, task)
            self.assertIn("intent_action", spec)
            self.assertIn("target_kind", spec)
            self.assertIn("expected_output", spec)
            self.assertEqual(
                spec["direct_upstream_inputs"], len(task["depends_on"]),
            )
            self.assertNotIn("task_semantics", spec)
            self.assertNotIn("expected_answer", spec)


if __name__ == "__main__":
    unittest.main()
