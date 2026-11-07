"""Small no-network tests for the current Author → Critic → local repair loop."""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import Mock

from environment.tests.test_review_reliability import ReviewReliabilityTests
from environment.question_generation.realization import _realize_one, _meaningful_edit
from environment.question_generation.question_review import (
    ReviewFormatError, author_question, critique_question, passing_judgment,
)
from environment.question_generation.verification import (
    CERTIFIED_METHOD, CERTIFIED_SCHEMA, CERTIFIED_REVIEW_CONTRACT,
    _certificate, validate_surface_certificate,
)


def natural_rephrase(reference):
    if "Noah Yarrow" in reference and "'first'" in reference and "'second'" in reference:
        return (
            "Which entities are recorded as targets of both the 'first' and "
            "'second' relations from Noah Yarrow?"
        )
    return "For the records under discussion, which entities meet these conditions? " + reference


PASS_CRITIQUE = {
    "verdict": "pass", "naturalness": "acceptable",
    "differences": [], "summary": "The same conditions are retained.",
}
MISSING_CRITIQUE = {
    "verdict": "revise", "naturalness": "acceptable",
    "differences": [{
        "reference_excerpt": "the 'first' relation",
        "candidate_excerpt": "'first'",
        "problem": "The direction of the first relation is unclear.",
        "minimal_fix": "Keep the direction explicit; leave the second relation unchanged.",
    }],
    "summary": "Repair only the first relation direction.",
}


class FakeAuthor:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append(data)
        if "current_fragment" in data:
            return {"goal":
                "Investigate the specified public evidence to identify "
                "the requested entities and return the relevant result."
            }
        return {"question": natural_rephrase(data["reference_question"])}


class FakeGoalAuthor:
    def complete_json(self, system, user, **kwargs):
        packet = json.loads(user)
        fragment = packet["current_fragment"]
        values = [
            str(s["public_value"]) for s in fragment["public_steps"]
            if s.get("public_value") is not None
        ]
        text = (
            "Investigate the relevant recorded entities and preserve all "
            "specified public relation and filter requirements."
        )
        if values:
            text += " Public values: " + ", ".join(values) + "."
        return {"goal": text}


class FakeGoalCritic:
    def complete_json(self, system, user, **kwargs):
        packet = json.loads(user)
        assert "candidate_goal" in packet
        return {
            "verdict": "pass", "issues": [],
            "summary": "All answer-defining local semantics were preserved.",
        }


class FakeCritic:
    def __init__(self, reviews=None):
        self.reviews = list(reviews if reviews is not None else [PASS_CRITIQUE])
        self.calls = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append((system, data))
        return copy.deepcopy(self.reviews.pop(0))


class Stage2SurfaceCertificationTests(unittest.TestCase):
    def data(self):
        gold, world, rendered = ReviewReliabilityTests().rendered_fixture()
        world = {**world, "_rendered": rendered, "_display_answer": rendered.answer}
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

    def certified(self, payload, world, capability):
        from environment.question_generation.goals_pipeline import generate_goal_record
        ok, record, audit = generate_goal_record(
            payload, world, capability,
            FakeGoalAuthor(), FakeGoalCritic(), max_goal_revisions=0,
        )
        assert ok, str((record, audit))
        return record

    def run_one(self, *, reviews=None, revisions=0, author=None):
        payload, world, capability, rendered = self.data()
        author = author or FakeAuthor()
        critic = FakeCritic(reviews)
        goal_record = self.certified(payload, world, capability)
        result = _realize_one(
            payload, world, capability, author, critic,
            max_revisions=revisions, goal_record=goal_record,
        )
        return result, payload, world, rendered, author, critic

    def test_certified_blueprint_reused_with_one_question_author_one_critic(self):
        (ok, row, audit), payload, world, rendered, author, critic = self.run_one()
        self.assertTrue(ok, str((row, audit)))
        self.assertEqual(row["publication_status"], "naturalized_certified")
        self.assertEqual(row["surface_certificate"]["method"], CERTIFIED_METHOD)
        self.assertEqual(row["surface_certificate"]["schema"], CERTIFIED_SCHEMA)
        self.assertEqual(row["surface_certificate"]["review_contract"], CERTIFIED_REVIEW_CONTRACT)
        self.assertEqual(row["authoring"]["strategy"], "certified_goal_blueprint_question_author_critic")
        self.assertEqual(len(row["delegation_blueprint"]["goals"]),
                         len(row["delegation_blueprint"]["outline"]["tasks"]))
        self.assertTrue(any(a["stage"] == "certified_blueprint_reused" for a in audit))
        self.assertEqual(len(author.calls), 1)
        self.assertTrue(all("current_fragment" not in x for x in author.calls))
        self.assertEqual(len(critic.calls), 1)
        self.assertNotIn("risk_groups", critic.calls[0][1])
        self.assertNotIn("candidate_probes", row["authoring"]["surface_review"])
        self.assertNotIn("final_judgment", row["authoring"]["surface_review"])
        validated = validate_surface_certificate(row, world, rendered.question)
        self.assertTrue(validated["eligible"])
        self.assertEqual(validated["stage2_method"], CERTIFIED_METHOD)

    def test_specific_local_difference_returns_to_previous_draft(self):
        (ok, row, audit), _, _, _, author, critic = self.run_one(
            reviews=[MISSING_CRITIQUE, PASS_CRITIQUE], revisions=1,
        )
        self.assertTrue(ok, str((row, audit)))
        self.assertEqual(len(author.calls), 2)
        second = next(call for call in reversed(author.calls)
                      if "previous_draft" in call)
        self.assertEqual(second["previous_draft"], natural_rephrase(second["reference_question"]))
        diff = second["reviewer_feedback"]["differences"][0]
        self.assertIn("relation", diff["problem"])
        self.assertIn("second relation unchanged", diff["minimal_fix"])
        self.assertEqual(row["authoring"]["revision_rounds"], 1)
        self.assertEqual(sum(a.get("critic_calls", 0) for a in audit), 2)

    def test_critic_uncertain_quarantines_without_author_hallucinated_repair(self):
        uncertain = {
            "verdict": "uncertain", "naturalness": "acceptable",
            "differences": [], "summary": "Cannot resolve the relation's subject.",
        }
        (ok, row, audit), _, _, _, author, critic = self.run_one(
            reviews=[uncertain], revisions=2,
        )
        self.assertFalse(ok)
        self.assertEqual(row["reason"], "rejected_uncertain")
        self.assertEqual(len(author.calls), 1)  # Stage 2B never writes Goals
        self.assertEqual(len(critic.calls), 1)

    def test_nonsemantic_punctuation_changes_are_not_naturalization(self):
        self.assertFalse(_meaningful_edit(
            "Which places are in Paris?", "  WHICH places are in Paris!!!"
        ))
        self.assertTrue(_meaningful_edit(
            "Let the first group consist of places in Paris.",
            "The first group contains places in Paris.",
        ))

    def test_local_guard_retains_explicit_and_or_and_named_after_checks(self):
        from environment.question_generation.semantic_checks import (
            explicit_join_contradictions, relation_semantic_contradictions,
        )
        reference = "Let the second group consist of entities in both X and Y."
        candidate = "The second group consists of entities in either X or Y."
        self.assertTrue(explicit_join_contradictions(reference, candidate))
        self.assertEqual(relation_semantic_contradictions(
            "Which cities are named after the person?",
            "Which cities have matching names with the person?",
        ), ["named_after_replaced_by_name_equality"])

    def test_reviewer_cannot_pass_and_report_differences(self):
        invalid = {
            **PASS_CRITIQUE, "differences": MISSING_CRITIQUE["differences"],
        }
        (ok, row, audit), _, _, _, _, critic = self.run_one(
            reviews=[invalid, invalid], revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(row["reason"], "rejected_validation_error")
        self.assertEqual(audit[-1]["failure"], "critic_protocol_error")
        self.assertEqual(len(critic.calls), 2)

    def test_critic_revise_without_actionable_difference_is_invalid(self):
        invalid = {
            "verdict": "revise", "naturalness": "acceptable",
            "differences": [], "summary": "Unspecified semantic error.",
        }
        class Bad:
            def complete_json(self, system, user, **kwargs):
                return invalid
        with self.assertRaises(ReviewFormatError):
            critique_question(Bad(), "reference", "candidate")

    def test_critic_reviews_only_reference_and_candidate(self):
        (ok, _, _), _, _, _, _, critic = self.run_one()
        self.assertTrue(ok)
        sent = critic.calls[0][1]
        self.assertEqual(set(sent), {
            "reference_question", "candidate_question", "benchmark_task_brief",
        })
        self.assertIn("subtask_goals", sent["benchmark_task_brief"])

    def test_rehashed_certificate_cannot_authorize_changed_review(self):
        (ok, row, _), payload, world, rendered, _, _ = self.run_one()
        self.assertTrue(ok)
        mutations = (
            lambda r: r["authoring"]["surface_review"]["critic_review"].update(
                verdict="revise"),
            lambda r: r.update(question="Which unrelated people exist?"),
            lambda r: r["authoring"]["surface_review"].update(
                scoped_reference_sha256="0" * 64),
            lambda r: r["authoring"]["surface_review"].update(
                public_guards_sha256="0" * 64),
            lambda r: r["delegation_blueprint"]["goals"][0].update(
                goal="Investigate a completely different unrelated topic.",
            ),
            lambda r: r["delegation_blueprint"]["outline"]["waves"].reverse(),
        )
        for edit in mutations:
            with self.subTest(edit=str(edit)):
                forged = copy.deepcopy(row)
                edit(forged)
                forged["surface_certificate"] = _certificate(
                    forged["task_id"], world["program"], rendered.question,
                    forged["question"], forged["authoring"]["surface_review"],
                )
                with self.assertRaises(ValueError):
                    validate_surface_certificate(forged, world, rendered.question)

    def test_old_certificate_version_is_not_replayed(self):
        (ok, row, _), payload, world, rendered, _, _ = self.run_one()
        self.assertTrue(ok)
        row["surface_certificate"]["schema"] = "maestro.stage2_surface_cert.v11"
        with self.assertRaises(ValueError):
            validate_surface_certificate(row, world, rendered.question)

    def test_missing_model_cannot_publish_reference_fallback(self):
        payload, world, capability, rendered = self.data()
        goal_record = self.certified(payload, world, capability)
        ok, row, audit = _realize_one(
            payload, world, capability, Mock(), Mock(), max_revisions=1,
            goal_record=goal_record,
        )
        self.assertFalse(ok)
        self.assertEqual(row["reason"], "rejected_validation_error")
        self.assertNotIn("question", row)


if __name__ == "__main__":
    unittest.main()
