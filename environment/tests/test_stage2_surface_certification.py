"""End-to-end Stage-2 Author/Critic/Final-Judge behavior; no network."""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import Mock

from environment.tests.test_review_reliability import ReviewReliabilityTests
from environment.question_generation.realization import _realize_one
from environment.question_generation.verification import (
    METHOD, SCHEMA, REVIEW_CONTRACT, validate_surface_certificate,
)
from environment.trajectory_simulation.eligibility import digest


def natural_rephrase(reference):
    # Synthetic fixture for pipeline/JSON-certificate mechanics, not a model
    # naturalness benchmark. It must satisfy the independent surface-change
    # gate so these tests reach the Critic/Final-Judge stages.
    return (
        "Using the public facts and relationships described below, determine "
        "which result the question seeks, without assuming any source was "
        "provided in advance. " +
        reference.replace("Let the ", "Consider the ", 1)
    )


PASS_CRITIQUE = {
    "verdict": "pass", "issues": [], "summary": "Equivalent and clear.",
}
PASS_FINAL = {
    "verdict": "pass", "semantic_equivalence": "equivalent",
    "naturalness": "improved", "issues": [],
    "explanation": "The conditions have been retained.",
}
MISSING_CRITIQUE = {
    "verdict": "revise",
    "issues": [{
        "kind": "missing_constraint", "severity": "major",
        "explanation": "The date 1914 was omitted from the location clause.",
        "suggestion": "Restore the location's start date of 1914.",
    }],
    "summary": "Restore the missing location condition.",
}


class FakeAuthor:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append(data)
        return {"question": natural_rephrase(data["reference_question"])}


def supported_checks(data):
    candidate = data["candidate_question"]
    # The scripted judge uses real substrings; this tests the publisher's
    # witness and checklist protocol, NOT the model's semantic accuracy.
    witness = candidate[:min(60, len(candidate))]
    return [{
        "id": check["id"], "status": "preserved",
        "candidate_quote": witness,
        "explanation": "Scripted: condition retained.",
    } for check in data["semantic_checks"]]


class FakeCritic:
    def __init__(self, critiques=None, final=None):
        self.critiques = list(critiques or [PASS_CRITIQUE])
        self.final = final or PASS_FINAL
        self.calls = []
        self.final_inputs = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append((system, data))
        if "final independent language and semantic judge" in system:
            self.final_inputs.append(data)
            response = copy.deepcopy(self.final)
        else:
            response = copy.deepcopy(self.critiques.pop(0))
        response["checks"] = supported_checks(data)
        return response


class Stage2SurfaceCertificationTests(unittest.TestCase):
    def data(self):
        gold, world, rendered = ReviewReliabilityTests().rendered_fixture()
        world = {**world, "_rendered": rendered,
                 "_display_answer": rendered.answer}
        from environment.tests.test_gold_trajectory import fixture
        q, _, _ = fixture()
        capability = q["capability_world"]
        payload = {
            "task_id": rendered.task_id,
            "draft_question": rendered.question,
            "task_design": {"final_operation": "QUERY_NAME"},
            "resources": capability["resources"],
            "terminal_contract": {},
        }
        return payload, world, capability, rendered

    def test_first_draft_passes_critic_and_final_judge(self):
        payload, world, capability, rendered = self.data()
        author, critic = FakeAuthor(), FakeCritic()
        ok, row, audit = _realize_one(
            payload, world, capability, author, critic,
            max_revisions=2,
        )
        self.assertTrue(ok, str(row))
        self.assertEqual(row["publication_status"], "naturalized_certified")
        self.assertEqual(row["surface_certificate"]["method"], METHOD)
        self.assertEqual(row["surface_certificate"]["schema"], SCHEMA)
        self.assertEqual(row["surface_certificate"]["review_contract"],
                         REVIEW_CONTRACT)
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(len(critic.calls), 2)
        self.assertEqual(critic.final_inputs[0].keys(),
                         {"reference_question", "candidate_question",
                          "potential_literal_changes_to_check", "semantic_checks"})
        self.assertEqual(row["authoring"]["revision_rounds"], 0)
        self.assertNotEqual(row["question"], rendered.question)
        verified = validate_surface_certificate(
            row, world, rendered.question,
        )
        self.assertTrue(verified["eligible"])
        self.assertEqual(verified["stage2_method"], METHOD)

    def test_revisions_receive_previous_draft_and_specific_critic_feedback(self):
        payload, world, capability, rendered = self.data()
        author = FakeAuthor()
        critic = FakeCritic(critiques=[MISSING_CRITIQUE, PASS_CRITIQUE])
        ok, row, audit = _realize_one(
            payload, world, capability, author, critic,
            max_revisions=1,
        )
        self.assertTrue(ok, row)
        self.assertEqual(len(author.calls), 2)
        self.assertIn("previous_draft", author.calls[1])
        self.assertEqual(
            author.calls[1]["reviewer_feedback"]["issues"][0]["kind"],
            "missing_constraint",
        )
        self.assertEqual(row["authoring"]["revision_rounds"], 1)
        self.assertEqual(
            [r["critic_feedback"]["verdict"] for r in audit
             if "critic_feedback" in r], ["revise", "pass"],
        )

    def test_critic_failure_never_publishes_scoped_fallback(self):
        payload, world, capability, rendered = self.data()
        ok, rejected, audit = _realize_one(
            payload, world, capability,
            FakeAuthor(), FakeCritic(critiques=[MISSING_CRITIQUE]),
            max_revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["publication_status"], "rejected")
        self.assertEqual(rejected["reason"], "rejected_semantic")
        self.assertNotIn("question", rejected)
        self.assertEqual(len([r for r in audit if r["stage"] == "candidate"]), 1)

    def test_final_judge_can_veto_critic_pass(self):
        payload, world, capability, rendered = self.data()
        judge = {
            "verdict": "reject", "semantic_equivalence": "different",
            "naturalness": "improved",
            "issues": ["The death date is attached to the wrong entity."],
            "explanation": "Changed date ownership.",
        }
        critic = FakeCritic(final=judge)
        ok, rejected, audit = _realize_one(
            payload, world, capability, FakeAuthor(), critic,
            max_revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_semantic")
        self.assertEqual(len(critic.final_inputs), 1)
        self.assertIn("final_judgment", audit[-1])

    def test_uncertainty_fails_closed_not_published(self):
        payload, world, capability, rendered = self.data()
        judge = {
            "verdict": "uncertain", "semantic_equivalence": "uncertain",
            "naturalness": "improved", "issues": [],
            "explanation": "Ambiguous referent.",
        }
        ok, rejected, _ = _realize_one(
            payload, world, capability, FakeAuthor(),
            FakeCritic(final=judge), max_revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_uncertain")

    def test_tampering_with_question_program_or_judgment_invalidates_certificate(self):
        payload, world, capability, rendered = self.data()
        ok, row, _ = _realize_one(
            payload, world, capability, FakeAuthor(),
            FakeCritic(), max_revisions=0,
        )
        self.assertTrue(ok)
        cases = [
            ("question", lambda r, w: r.update(question="changed question?")),
            ("program", lambda r, w: w["program"].update(nodes=[])),
            ("review", lambda r, w: r["authoring"]["surface_review"]
             ["final_judgment"].update(verdict="reject")),
            ("certificate", lambda r, w: r["surface_certificate"]
             .update(method="verified_controlled_renderer")),
            ("status", lambda r, w: r.update(publication_status="canonical_only")),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                edited, copy_world = copy.deepcopy(row), copy.deepcopy(world)
                mutate(edited, copy_world)
                with self.assertRaises(ValueError):
                    validate_surface_certificate(
                        edited, copy_world, rendered.question,
                    )

    def test_no_llm_rejects_instead_of_publishing_a_reference(self):
        payload, world, capability, rendered = self.data()
        ok, rejected, _ = _realize_one(
            payload, world, capability, Mock(), Mock(), max_revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_validation_error")



    def test_v13_known_false_positive_near_copy_is_rejected(self):
        from environment.question_generation.semantic_checks import naturalness_gate
        ref = (
            "Let the first group consist of places with a start date of 1914 "
            "recorded as birthplaces of the person born on March 8, 2000. "
            "Let the second group consist of events that point to the first group."
        )
        draft = ref.replace("consist of", "be", 1)
        self.assertFalse(naturalness_gate(ref, draft)["improved_enough_to_review"])

    def test_explicit_intersection_to_union_is_not_approved(self):
        from environment.question_generation.semantic_checks import explicit_join_contradictions
        ref = (
            "Let the first group consist of administrative areas in Central Valley. "
            "Let the second group consist of entities in both entities with "
            "population 1203 from the first group and administrative areas "
            "with a neighbor of an entity with male population 488."
        )
        bad = (
            "First, identify areas in Central Valley to form Group 1. "
            "Next, define Group 2 as the union of: (1) entities with "
            "population 1203, and (2) areas with a neighbor relation."
        )
        self.assertEqual(
            explicit_join_contradictions(ref, bad),
            ["group_2_changed_intersection_to_union"],
        )
        self.assertFalse(explicit_join_contradictions(ref, ref))

    def test_missing_or_forged_grounding_blocks_final_certificate(self):
        from environment.question_generation.semantic_checks import (
            check_evidence, CheckFormatError,
        )
        sample = "Which people have a recorded spouse who died in this place?"
        checks = [{"id": "C1", "kind": "relation_scope",
                   "expected": "Preserve spouse direction",
                   "reference_excerpt": "whose spouse died in this place"}]
        for bad in (
            [], [{"id": "wrong", "status": "preserved",
                  "candidate_quote": "spouse", "explanation": "yes"}],
            [{"id": "C1", "status": "preserved",
              "candidate_quote": "invented relationship", "explanation": "yes"}],
            [{"id": "C1", "status": "preserved",
              "candidate_quote": "spouse", "explanation": ""}],
        ):
            with self.subTest(bad=bad), self.assertRaises(CheckFormatError):
                check_evidence(checks, bad, sample)

    def test_model_cannot_pass_a_known_changed_semantic_check(self):
        from environment.question_generation.question_review import (
            judge_question, critique_question,
        )
        class ContradictoryReviewer:
            def complete_json(self, system, user, **kwargs):
                data = json.loads(user)
                checks = supported_checks(data)
                checks[0]["status"] = "changed"
                if "final independent" in system:
                    return {**PASS_FINAL, "checks": checks}
                return {**PASS_CRITIQUE, "checks": checks}
        c = [{"id": "C1", "kind": "set_logic",
              "expected": "all branches", "reference_excerpt": "both groups"}]
        draft = "Which entities belong to either group?"
        feedback = critique_question(ContradictoryReviewer(), "both groups", draft, c)
        self.assertEqual(feedback["verdict"], "revise")
        self.assertTrue(feedback["issues"])
        judged = judge_question(ContradictoryReviewer(), "both groups", draft, c)
        self.assertNotEqual(judged["verdict"], "pass")

if __name__ == "__main__":
    unittest.main()
