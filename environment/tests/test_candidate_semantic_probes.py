"""Public candidate-only probes: v16 false-positive regression examples.

No private Gold, real local LLM or network call is needed.
"""
from __future__ import annotations

import json
import unittest

from environment.question_generation.scoped_surface import ScopedPlan
from environment.question_generation.semantic_probes import (
    CandidateProbeFormatError, compile_candidate_probes,
    probe_candidate_semantics, validate_candidate_probe_records,
    explicit_probe_contradictions,
)
from environment.question_generation.question_review import (
    critique_question, ReviewFormatError,
)


def join(op, *inputs):
    return {"definition": op, "conditions": [],
            "inputs": list(inputs)}


def leaf():
    return join("FIND_ALL")


class ProbeTests(unittest.TestCase):
    def test_v16_case3_union_substituted_for_intersection(self):
        first = "Let the first group consist of administrative areas in Foxford."
        second = (
            "Let the second group consist of entities in both entities "
            "whose location points to the first group and administrative "
            "areas with population under 1765 drawn from the first group."
        )
        third = (
            "Let the third group consist of entities in at least one "
            "of the second group or the first group."
        )
        plan = ScopedPlan([(first,), (second,), (third,)],
                          [leaf(), join("INTERSECT", leaf(), leaf()),
                           join("UNION", leaf(), leaf())], leaf())
        candidate = (
            "The first group consists of Foxford areas; the second group "
            "includes areas that are either near Foxford or under population "
            "1765; the third group includes areas in either earlier set."
        )
        probes = compile_candidate_probes(plan, " ".join((first, second, third)),
                                         candidate)
        self.assertEqual(
            [(p["group"], p["expected"]) for p in probes],
            [("second", "AND"), ("third", "OR")],
        )
        answers = [{
            "id": p["id"], "kind": p["kind"],
            "answer": "OR", "expected": p["expected"],
        } for p in probes]
        self.assertNotEqual(answers[0]["answer"], answers[0]["expected"])
        self.assertTrue(
            any("explicit_intersection_to_either_or" in e for e in
                explicit_probe_contradictions(probes, candidate)),
        )
        with self.assertRaises(ValueError):
            validate_candidate_probe_records(probes, answers)

    def test_v16_case5_final_union_does_not_inherit_group_scope(self):
        first = "Let the first group consist of administrative areas in Central Valley."
        second = (
            "Let the second group consist of entities in both entities "
            "with population 1203 drawn from the first group and "
            "administrative areas whose neighbor points to a member of "
            "the first group."
        )
        final = (
            "Among entities with a postal code of 91683 drawn from the "
            "second group or entities in both entities listed as targets "
            "of the neighbor relation from Clearford and administrative "
            "areas in the relevant records, which have minimum population?"
        )
        plan = ScopedPlan([(first,), (second,), (final,)],
                          [leaf(), join("INTERSECT", leaf(), leaf()),
                           join("SELECT_BETWEEN", leaf(), leaf())], leaf())
        candidate = (
            "The first group consists of areas in Central Valley. Define "
            "the second group to include areas with population 1203 and "
            "neighbors. Among entities in the second group that have postal "
            "code 91683 or are neighbors of Clearford, which has minimum "
            "population?"
        )
        probes = compile_candidate_probes(plan, " ".join((first, second, final)),
                                         candidate)
        self.assertIn(("final_union_scope", "NOT_ALL"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertIn(("group_join", "AND"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertTrue(
            any("union_branch_restricted_to_group" in e for e in
                explicit_probe_contradictions(probes, candidate)),
        )

    def test_v16_case8_member_scope_does_not_move_inside_both(self):
        fifth = (
            "Let the fifth group consist of organizations with a duns of "
            "492868983 drawn from entities in both entities with at least "
            "one member in the third group and the fourth group."
        )
        meaning = join(
            "FILTER_NUMBER",
            join("INTERSECT", leaf(), leaf()),
        )
        candidate = (
            "The fifth group consists of organizations with DUNS 492868983 "
            "drawn from entities that have at least one member in both "
            "the third group and the fourth group."
        )
        plan = ScopedPlan([(fifth,)], [meaning], meaning)
        probes = compile_candidate_probes(plan, fifth, candidate)
        self.assertIn(("member_owner", "SELF"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertIn(("group_join", "AND"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertTrue(
            any("member_scope_moved_into_intersection" in e for e in
                explicit_probe_contradictions(probes, candidate)),
        )
        faithful = (
            "The fifth group consists of organizations drawn from entities "
            "that have at least one member in the third group and that "
            "themselves belong to the fourth group."
        )
        self.assertEqual(
            explicit_probe_contradictions(
                compile_candidate_probes(plan, fifth, faithful), faithful,
            ), [],
        )

    def test_naturally_unlabelled_candidate_is_not_assigned_undefined_probe(self):
        second = "Let the second group consist of entities in both X and Y."
        meaning = join("INTERSECT", leaf(), leaf())
        plan = ScopedPlan([(second,)], [meaning], meaning)
        candidate = "Which entities meet both conditions X and Y?"
        self.assertEqual(
            compile_candidate_probes(plan, second, candidate), [],
        )

    def test_model_sees_only_candidate_and_probes_not_reference_or_expected(self):
        reference = "Let the second group consist of entities in both X and Y."
        candidate = "The second group includes entities in either X or Y."
        meaning = join("INTERSECT", leaf(), leaf())
        plan = ScopedPlan([(reference,)], [meaning], meaning)
        probes = compile_candidate_probes(plan, reference, candidate)
        class Reader:
            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                assert "reference_question" not in payload
                assert "expected" not in user
                assert "in both X and Y" not in user
                assert payload["candidate_question"] == candidate
                assert all(set(p) == {"id", "question"} for p in payload["probes"])
                return {"answers": [{
                    "id": p["id"], "answer": "OR",
                } for p in payload["probes"]]}
        observed = probe_candidate_semantics(
            Reader(), reference, candidate, probes,
        )
        self.assertEqual(observed[0]["answer"], "OR")
        self.assertEqual(observed[0]["expected"], "AND")
        with self.assertRaises(ValueError):
            validate_candidate_probe_records(probes, observed)

    def test_complete_probe_certificate_and_tampering(self):
        reference = "Let the second group consist of entities in both X and Y."
        meaning = join("INTERSECT", leaf(), leaf())
        plan = ScopedPlan([(reference,)], [meaning], meaning)
        candidate = "The second group consists of the intersection of X and Y."
        probes = compile_candidate_probes(plan, reference, candidate)
        valid = [
            {"id": p["id"], "kind": p["kind"], "expected": p["expected"],
             "answer": p["expected"]}
            for p in probes
        ]
        validate_candidate_probe_records(probes, valid)
        for bad in ([], [dict(valid[0], expected="OR")],
                    [dict(valid[0], answer="UNCLEAR")],
                    [dict(valid[0], kind="some_other_check")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_candidate_probe_records(probes, bad)

    def test_incomplete_probe_response_reports_expected_and_received_ids(self):
        class Reader:
            def complete_json(self, system, user, **kwargs):
                return {"answers": []}
        probes = [{"id": "P1", "kind": "group_join", "expected": "AND",
                   "question": "AND or OR?"}]
        with self.assertRaises(CandidateProbeFormatError) as ctx:
            probe_candidate_semantics(Reader(), "", "Some candidate", probes)
        self.assertEqual(ctx.exception.diagnostic["expected_ids"], ["P1"])
        self.assertEqual(ctx.exception.diagnostic["received_count"], 0)

    def test_incomplete_critic_output_has_bounded_diagnostic_shape(self):
        checks = [{"id": "R1", "kind": "set_logic",
                   "requirements": [{"id": "C1", "expected": "both"}]}]
        class BadCritic:
            def complete_json(self, system, user, **kwargs):
                return {"verdict": "pass", "issues": [],
                        "summary": "Looks fine", "checks": []}
        with self.assertRaises(ReviewFormatError) as ctx:
            critique_question(BadCritic(), "both", "either", checks)
        self.assertEqual(
            ctx.exception.diagnostic["expected_group_ids"], ["R1"],
        )
        self.assertEqual(
            ctx.exception.diagnostic["received_group_ids"], [],
        )
        self.assertEqual(ctx.exception.diagnostic["received_count"], 0)


if __name__ == "__main__":
    unittest.main()
