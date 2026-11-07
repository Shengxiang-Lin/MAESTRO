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
    # This fixture uses a genuinely reordered statement of the two relations
    # so style checks do not mistake a boilerplate prefix for naturalization.
    if "Noah Yarrow" in reference and "'first'" in reference and "'second'" in reference:
        return (
            "Which entities are recorded as targets of both the 'first' and "
            "'second' relations from Noah Yarrow?"
        )
    return "Which candidates satisfy all the requested public conditions? " + reference


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
    # Simulates review verdict formatting; not a proof of semantic accuracy.
    return [{
        "id": group["id"], "status": "preserved", "problem_ids": [],
        "explanation": "Scripted: all requirements in this category are preserved.",
    } for group in data["risk_groups"]]


class FakeCritic:
    def __init__(self, critiques=None, final=None):
        self.critiques = list(critiques or [PASS_CRITIQUE])
        self.final = final or PASS_FINAL
        self.calls = []
        self.final_inputs = []

    def complete_json(self, system, user, **kwargs):
        data = json.loads(user)
        self.calls.append((system, data))
        if "independent reader of candidate text only" in system:
            # This is a scripted output contract test, NOT semantic evidence.
            # The renderer fixture contains two FORWARD relations, an AND
            # result selection and a QUERY_NAME terminal.
            answers = []
            for spec in data["probes"]:
                question = spec["question"]
                if "actual subject" in question:
                    answer = "A_TO_B"
                elif "set-building clause" in question:
                    answer = "AND"
                elif "final candidate question" in question:
                    answer = "ENTITIES"
                elif "FIRST group" in question:
                    answer = "NOT_ALL"
                elif "must satisfy" in question:
                    answer = "SELECTED"
                elif "EITHER branch" in question:
                    answer = "AND"
                else:
                    answer = "UNCLEAR"
                answers.append({
                    "id": spec["id"], "answer": answer,
                    "evidence": data["candidate_question"],
                    "subject": "Noah Yarrow", "object": "entities",
                    "subject_role": "A", "object_role": "B", "owner": "",
                })
            return {"answers": answers}
        if "independent final semantic and language judge" in system:
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
        self.assertGreaterEqual(len(critic.calls), 3)
        self.assertEqual(
            row["authoring"]["surface_review"]["semantic_coverage"]["status"],
            "complete",
        )
        self.assertGreater(
            row["authoring"]["surface_review"]["semantic_coverage"]["required"], 0,
        )
        self.assertEqual(critic.final_inputs[0].keys(),
                         {"reference_question", "candidate_question",
                          "potential_literal_changes_to_check", "risk_groups"})
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
        self.assertTrue(ok, repr((row, audit)))
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
        self.assertEqual(rejected["reason"], "rejected_uncertain", str(rejected))

    def test_tampering_with_question_program_or_judgment_invalidates_certificate(self):
        payload, world, capability, rendered = self.data()
        ok, row, _ = _realize_one(
            payload, world, capability, FakeAuthor(),
            FakeCritic(), max_revisions=0,
        )
        self.assertTrue(ok, repr((row, _)))
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

    def test_stage3_rejects_missing_coverage_even_if_attacker_rehashes_certificate(self):
        from environment.question_generation.verification import _certificate
        payload, world, capability, rendered = self.data()
        ok, row, audit = _realize_one(
            payload, world, capability, FakeAuthor(),
            FakeCritic(), max_revisions=0,
        )
        self.assertTrue(ok, repr((row, audit)))
        self.assertTrue(
            row["authoring"]["surface_review"]["candidate_probes"],
        )
        cases = [
            ("missing_model_records", lambda review: review.update(
                candidate_probes=[])),
            ("false_complete_manifest", lambda review: review[
                "semantic_coverage"].update(verified=0)),
            ("fake_obligations", lambda review: review.update(
                coverage_obligation_sha256="0" * 64)),
            ("remove_evidence_status", lambda review: review[
                "candidate_probes"][0].pop("evidence_status")),
            ("forge_raw_model_answer", lambda review: review[
                "candidate_probes"][0].update(raw_answer="UNCLEAR")),
            ("forge_grounding_reason", lambda review: review[
                "candidate_probes"][0].update(
                    evidence_reason="model_returned_unclear")),
        ]
        for label, mutate in cases:
            with self.subTest(case=label):
                forged = copy.deepcopy(row)
                review = forged["authoring"]["surface_review"]
                mutate(review)
                forged["surface_certificate"] = _certificate(
                    forged["task_id"], world["program"], rendered.question,
                    forged["question"], review,
                )
                with self.assertRaises(ValueError):
                    validate_surface_certificate(
                        forged, world, rendered.question,
                    )

    def test_json_schema_does_not_require_recursive_checks_in_child(self):
        from environment.question_generation.question_review import (
            _CHECK_SCHEMA, _FINAL_SCHEMA, _CRITIC_SCHEMA,
        )
        self.assertEqual(
            set(_CHECK_SCHEMA["required"]),
            {"id", "status", "explanation"},
        )
        self.assertIn("checks", _CRITIC_SCHEMA["required"])
        self.assertIn("checks", _FINAL_SCHEMA["required"])
        self.assertNotIn("checks", _CHECK_SCHEMA["properties"])

    def test_protocol_retry_reviews_same_candidate_without_reauthoring(self):
        from environment.question_generation.question_review import ReviewFormatError
        class OneProtocolFailure(FakeCritic):
            failed_once = False
            def complete_json(self, system, user, **kwargs):
                if ("constructive question editor" in system
                        and not self.failed_once):
                    self.failed_once = True
                    raise ReviewFormatError("test_protocol_failure")
                return super().complete_json(system, user, **kwargs)
        payload, world, capability, _ = self.data()
        author, critic = FakeAuthor(), OneProtocolFailure()
        ok, row, audit = _realize_one(
            payload, world, capability, author, critic, max_revisions=0,
        )
        self.assertTrue(ok, repr((row, audit)))
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(audit[-1]["critic_calls"], 2)
        self.assertEqual(len(audit[-1]["critic_protocol_failures"]), 1)
        self.assertEqual(
            audit[-1]["critic_protocol_failures"][0]["schema_mode"], True,
        )

    def test_failed_final_protocol_retries_do_not_rewrite_good_candidate(self):
        from environment.question_generation.question_review import ReviewFormatError
        class BrokenFinal(FakeCritic):
            def complete_json(self, system, user, **kwargs):
                if "independent final" in system:
                    raise ReviewFormatError("final_model_json_invalid")
                return super().complete_json(system, user, **kwargs)
        payload, world, capability, _ = self.data()
        author = FakeAuthor()
        ok, rejected, audit = _realize_one(
            payload, world, capability, author, BrokenFinal(),
            max_revisions=2,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_validation_error")
        self.assertEqual(len(author.calls), 1)
        self.assertEqual(audit[-1]["critic_calls"], 1)
        self.assertEqual(audit[-1]["final_judge_calls"], 2)
        self.assertEqual(len(audit[-1]["final_judge_protocol_failures"]), 2)

    def test_no_llm_rejects_instead_of_publishing_a_reference(self):
        payload, world, capability, rendered = self.data()
        ok, rejected, _ = _realize_one(
            payload, world, capability, Mock(), Mock(), max_revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_validation_error")



    def test_same_filter_on_different_relation_owners_stays_distinct(self):
        from environment.question_generation.scoped_surface import ScopedPlan
        from environment.question_generation.semantic_checks import semantic_checks
        restriction = {
            "attribute": "population number", "comparison": "less than",
            "value": {"type": "number", "value": "1765"},
        }
        def relation(anchor):
            return {
                "definition": "RELATE", "relation": "location",
                "direction": "forward",
                "conditions": [copy.deepcopy(restriction)],
                "inputs": [{
                    "definition": "FIND", "anchor": anchor,
                    "conditions": [], "inputs": [],
                }],
            }
        combined = {
            "definition": "INTERSECT", "conditions": [],
            "inputs": [relation("Alpha"), relation("Beta")],
        }
        plan = ScopedPlan(
            [("Which matching entities satisfy both location branches?",)],
            [combined], combined,
        )
        checks = semantic_checks(plan)
        ownership = [item for item in checks
                     if item["kind"] == "condition_owner"
                     and "population number" in item["expected"]]
        self.assertEqual(len(ownership), 2)
        self.assertEqual(len({item["id"] for item in ownership}), 2)

    def test_scoped_contract_typed_literals_allow_date_paraphrases_not_code_loss(self):
        from environment.question_generation.scoped_surface import ScopedPlan
        from environment.question_generation.semantic_checks import missing_typed_literals
        tree = {
            "definition": "COUNT", "inputs": [{
                "definition": "FIND_ALL", "inputs": [],
                "conditions": [
                    {"attribute": "death date", "comparison": "equal to",
                     "value": {"type": "date", "value": "1997-08-01"}},
                    {"attribute": "telephone country code", "comparison": "equal to",
                     "value": {"type": "string", "value": "+650"}},
                    {"attribute": "population", "comparison": "equal to",
                     "value": {"type": "number", "value": "1203"}},
                ],
            }], "conditions": [],
        }
        plan = ScopedPlan([], [], tree)
        for date in ("1997-08-01", "August 1, 1997", "1 August 1997"):
            text = f"How many people died on {date}, with code +650 and population 1,203?"
            self.assertEqual(missing_typed_literals(plan, text), [])
            self.assertIn(
                "string:+650",
                missing_typed_literals(plan, text.replace("+650", "650")),
            )
            self.assertIn(
                "number:1203",
                missing_typed_literals(plan, text.replace("1,203", "1,204")),
            )

    def test_boilerplate_prefix_does_not_game_naturalization_gate(self):
        from environment.question_generation.semantic_checks import naturalness_gate
        ref = (
            "Which entities listed as targets of the first relation from Noah "
            "Yarrow are also targets of the second relation from Noah Yarrow?"
        )
        bad = (
            "Using the public records, please determine the requested "
            "result according to the following details. " + ref
        )
        self.assertFalse(naturalness_gate(ref, bad)["improved_enough_to_review"])

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

    def test_missing_or_invalid_group_review_cannot_certify(self):
        from environment.question_generation.semantic_checks import (
            validate_group_decisions, CheckFormatError,
        )
        groups = [{"id": "R1", "kind": "relation_scope",
                   "requirements": [{"expected": "Preserve spouse direction"}]}]
        for invalid in (
            [], [{"id": "wrong", "status": "preserved",
                  "explanation": "Confirmed relation ownership."}],
            [{"id": "R1", "status": "invalid",
              "explanation": "Confirmed relation ownership."}],
            [{"id": "R1", "status": "preserved",
              "explanation": "yes"}],
            [{"id": "R1", "status": "preserved",
              "explanation": "Confirmed relation ownership."},
             {"id": "R1", "status": "preserved",
              "explanation": "Confirmed relation ownership."}],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(CheckFormatError):
                validate_group_decisions(groups, invalid)

    def test_grouped_output_is_bounded_but_keeps_detailed_risks_as_input(self):
        from environment.question_generation.semantic_checks import (
            semantic_checks, grouped_semantic_checks,
        )
        from environment.question_generation.scoped_surface import ScopedPlan
        tree = {
            "definition": "COUNT", "conditions": [], "inputs": [{
                "definition": "INTERSECT", "conditions": [], "inputs": [
                    {"definition": "FIND_ALL", "conditions": [
                        {"attribute": "population", "comparison": "less than",
                         "value": {"type": "number", "value": str(1000 + i)}}
                        for i in range(12)
                    ], "inputs": []},
                    {"definition": "RELATE", "relation": "location",
                     "direction": "forward", "conditions": [], "inputs": [
                         {"definition": "FIND_ALL",
                          "conditions": [], "inputs": []}]},
                ]
            }],
        }
        plan = ScopedPlan([("How many match?",)], [tree], tree)
        atomic = semantic_checks(plan)
        grouped = grouped_semantic_checks(plan)
        self.assertGreater(len(atomic), 12)
        self.assertLessEqual(len(grouped), 4)
        self.assertEqual(
            len([r for group in grouped if group["kind"] == "condition_owner"
                 for r in group["requirements"]]), 12,
        )

    def test_v16_relation_meaning_drift_is_flagged_without_private_gold(self):
        from environment.question_generation.semantic_checks import (
            relation_semantic_contradictions,
        )
        reference = (
            "Which places are named after members of the target group, "
            "who are named after an earlier person?"
        )
        candidate = (
            "Which places have names that match members of the target group, "
            "who are named after an earlier person?"
        )
        self.assertEqual(
            relation_semantic_contradictions(reference, candidate),
            ["named_after_replaced_by_name_equality"],
        )
        self.assertFalse(relation_semantic_contradictions(
            reference,
            "Which places were named in honor of members of the target group?",
        ))
        self.assertFalse(relation_semantic_contradictions(
            "Which cities have matching names?",
            "What cities share the same name?",
        ))

    def test_v19_negative_review_without_atomic_ids_still_vetoes(self):
        from environment.question_generation.semantic_checks import (
            validate_group_decisions, CheckFormatError,
        )
        groups = [{"id": "R1", "kind": "relation_scope",
                   "requirements": [
                       {"id": "C4", "expected": "Preserve directed naming"},
                       {"id": "C5", "expected": "Preserve spouse"},
                   ]}]
        good = [{"id": "R1", "status": "changed",
                 "problem_ids": ["C4"],
                 "explanation": "Name equality replaces a naming relationship."}]
        validate_group_decisions(groups, good)
        # Negative groups remain vetoes even when the small model cannot
        # ground specific C-IDs. This removes avoidable v18 protocol errors.
        validate_group_decisions(groups, [{
            "id": "R1", "status": "changed",
            "explanation": "Relation was replaced with name equality.",
        }])
        for bad in (
            [{**good[0], "problem_ids": ["C7"]}],
            [{**good[0], "problem_ids": ["C4", "C4"]}],
            [{**good[0], "status": "preserved"}],
            [{**good[0], "explanation":
              "The directed naming relation is preserved correctly."}],
        ):
            with self.subTest(bad=bad), self.assertRaises(CheckFormatError):
                validate_group_decisions(groups, bad)

    def test_v16_relation_owner_route_is_explicit_in_atomic_requirements(self):
        from environment.question_generation.scoped_surface import ScopedPlan
        from environment.question_generation.semantic_checks import semantic_checks
        source = {"definition": "FIND", "anchor": "Alpha",
                  "conditions": [], "inputs": []}
        node = {
            "definition": "RELATE", "direction": "forward",
            "relation": "location", "inputs": [source],
            "conditions": [{"attribute": "population",
                            "comparison": "greater than",
                            "value": {"type": "number", "value": "1190"}}],
        }
        plan = ScopedPlan([("Which location entities have population >1190?",)],
                          [node], node)
        owners = [c for c in semantic_checks(plan)
                  if c["kind"] == "condition_owner"]
        self.assertEqual(len(owners), 1)
        self.assertIn("forward traversal of 'location'", owners[0]["expected"])

    def test_model_cannot_pass_a_known_changed_semantic_check(self):
        from environment.question_generation.question_review import (
            judge_question, critique_question,
        )
        class ContradictoryReviewer:
            def complete_json(self, system, user, **kwargs):
                data = json.loads(user)
                checks = supported_checks(data)
                checks[0]["status"] = "changed"
                checks[0]["problem_ids"] = ["C1"]
                checks[0]["explanation"] = (
                    "The candidate uses either group where both are required."
                )
                if "independent final" in system:
                    return {**PASS_FINAL, "checks": checks}
                return {**PASS_CRITIQUE, "checks": checks}
        c = [{"id": "R1", "kind": "set_logic",
              "requirements": [{"id": "C1", "expected": "all branches",
                                "reference_excerpt": "both groups"}]}]
        draft = "Which entities belong to either group?"
        feedback = critique_question(ContradictoryReviewer(), "both groups", draft, c)
        self.assertEqual(feedback["verdict"], "revise")
        self.assertTrue(feedback["issues"])
        judged = judge_question(ContradictoryReviewer(), "both groups", draft, c)
        self.assertNotEqual(judged["verdict"], "pass")

if __name__ == "__main__":
    unittest.main()
