"""Stage-2 v18 candidate condition-owner and reviewer feedback regressions.

Synthetic PUBLIC ScopedPlan + scripted readers only: no real model, Gold, or
private world is passed to an LLM.
"""
from __future__ import annotations

import json
import unittest

from environment.question_generation.scoped_surface import ScopedPlan
from environment.question_generation.semantic_probes import (
    compile_candidate_probes, explicit_probe_contradictions,
    probe_candidate_semantics, validate_candidate_probe_records,
)
from environment.question_generation.semantic_checks import semantic_checks
from environment.question_generation.question_review import (
    ReviewFormatError, critique_question, _guard_unverified_omission_feedback,
)


def _scope(op, *, conditions=(), inputs=(), relation=None):
    node = {"definition": op, "conditions": list(conditions),
            "inputs": list(inputs)}
    if relation is not None:
        node.update(relation=relation, direction="backward")
    return node


def _plan_case3():
    first_ref = ("Let the first group consist of administrative areas "
                 "whose recorded 'location' points to Foxford.")
    second_ref = (
        "Let the second group consist of entities in both entities whose "
        "recorded 'location' points to a member of administrative areas "
        "with a geo of North Cedar region drawn from the first group and "
        "administrative areas with a population number less than 1,765 "
        "drawn from the first group."
    )
    first = _scope("FIND_ALL")
    related = _scope("RELATE", relation="location", inputs=[first])
    filtered_selected = _scope(
        "FILTER_NUMBER",
        conditions=[{
            "attribute": "population number", "comparison": "less than",
            "value": {"type": "number", "value": "1765"},
        }], inputs=[first],
    )
    second = _scope("INTERSECT", inputs=[related, filtered_selected])
    plan = ScopedPlan([(first_ref,), (second_ref,)],
                      [first, second], second)
    return plan, first_ref + " " + second_ref


class ConditionOwnerProbeTests(unittest.TestCase):
    def test_case3_requires_owner_probe_even_if_and_is_preserved(self):
        plan, reference = _plan_case3()
        candidate = (
            "The first group consists of administrative areas whose location "
            "points to Foxford; the second group consists of entities located "
            "in both (1) administrative areas with a geo of North Cedar "
            "region and (2) administrative areas with a population number "
            "less than 1,765."
        )
        probes = compile_candidate_probes(plan, reference, candidate)
        kinds = [(p["kind"], p["expected"]) for p in probes]
        self.assertIn(("group_join", "AND"), kinds)
        self.assertIn(("condition_owner", "SELECTED"), kinds)
        self.assertTrue(any(
            "selected_filter_moved_to_related_location" in reason
            for reason in explicit_probe_contradictions(probes, candidate)
        ))

    def test_faithful_candidate_probes_have_no_explicit_false_positive(self):
        plan, reference = _plan_case3()
        candidate = (
            "The second group consists of entities that both (1) have a "
            "recorded location pointing to qualifying places and (2) "
            "are themselves administrative areas with a population number "
            "less than 1,765, while the first group consists of places "
            "whose location points to Foxford."
        )
        probes = compile_candidate_probes(plan, reference, candidate)
        self.assertEqual(explicit_probe_contradictions(probes, candidate), [])
        self.assertIn(("condition_owner", "SELECTED"),
                      [(p["kind"], p["expected"]) for p in probes])

        class Reader:
            def complete_json(self, system, user, **kwargs):
                payload = json.loads(user)
                assert "reference_question" not in payload
                assert "expected" not in user
                assert "North Cedar" not in user
                return {"answers": [
                    {"id": p["id"], "answer": "SELECTED" if
                     "which entity must satisfy" in p["question"] else "AND"}
                    for p in payload["probes"]
                ]}
        records = probe_candidate_semantics(
            Reader(), reference, candidate, probes,
        )
        validate_candidate_probe_records(probes, records)
        altered = [dict(record) for record in records]
        altered[0]["answer"] = "RELATED"
        with self.assertRaises(ValueError):
            validate_candidate_probe_records(probes, altered)

    def test_no_owner_guess_when_no_joined_location_branch(self):
        ref = "Let the second group consist of entities in both A and B."
        plan = ScopedPlan(
            [(ref,)],
            [_scope("INTERSECT", inputs=[
                _scope("FIND_ALL"),
                _scope("FILTER_NUMBER", conditions=[
                    {"attribute": "population number", "comparison": "less than",
                     "value": {"type": "number", "value": "1765"}}]),
            ])],
            _scope("COUNT"),
        )
        probes = compile_candidate_probes(
            plan, ref, "The second group consists of entities in both A and B."
        )
        self.assertFalse(any(p["kind"] == "condition_owner" for p in probes))

    def test_select_between_is_a_union_even_if_inner_branch_intersects(self):
        tree = _scope("SELECT_BETWEEN", inputs=[
            _scope("INTERSECT", inputs=[_scope("FIND_ALL"),
                                        _scope("FIND_ALL")]),
            _scope("FIND_ALL"),
        ])
        plan = ScopedPlan([("Among branch A or branch B, which are smallest? "
                            "Include ties.",)], [tree], tree)
        checks = semantic_checks(plan)
        self.assertTrue(any(
            c["kind"] == "set_logic"
            and "FINAL RANKING POOL UNION (OR)" in c["expected"]
            for c in checks
        ))
        self.assertTrue(any(
            c["kind"] == "final_answer"
            and "UNION of the two independent candidate pools" in c["expected"]
            for c in checks
        ))

    def test_omission_claim_does_not_instruct_author_to_reinsert_present_value(self):
        issues = [{
            "kind": "condition_owner", "severity": "major",
            "explanation": "The candidate omits male population 488.",
            "suggestion": "Restore missing male population 488.",
        }]
        candidate = "Areas with a male population of 488 are adjacent."
        _guard_unverified_omission_feedback(candidate, issues)
        self.assertIn("unverified claim", issues[0]["explanation"])
        self.assertIn("owning entity", issues[0]["suggestion"])
        self.assertEqual(issues[0]["severity"], "major")

    def test_invalid_problem_id_diagnostics_include_offending_ids(self):
        checks = [{
            "id": "R1", "kind": "condition_owner",
            "requirements": [{"id": "C4", "expected": "Owner fixed."}],
        }]
        class WrongId:
            def complete_json(self, system, user, **kwargs):
                return {
                    "verdict": "revise", "summary": "Changed",
                    "issues": [],
                    "checks": [{
                        "id": "R1", "status": "changed",
                        "problem_ids": ["C99"],
                        "explanation": "Condition moved to a related entity.",
                    }],
                }
        with self.assertRaises(ReviewFormatError) as ctx:
            critique_question(WrongId(), "Reference", "Candidate", checks)
        detail = ctx.exception.diagnostic["group_details"][0]
        self.assertEqual(detail["problem_ids"], ["C99"])
        self.assertEqual(detail["unknown_problem_ids"], ["C99"])
        self.assertEqual(detail["allowed_problem_ids"], ["C4"])


if __name__ == "__main__":
    unittest.main()
