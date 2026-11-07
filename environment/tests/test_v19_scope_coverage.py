"""Stage 2 v19 regression on previously failing REAL v18 candidate texts.

All references are PUBLIC scoped questions. ScopedPlan meanings are narrow
synthetic projections of the public contract, not private World/Gold state.
No network or LLM inference is used.
"""
from __future__ import annotations

import json
import unittest

from environment.tests.test_condition_owner_probes import _plan_case3, _scope
from environment.question_generation.scoped_surface import ScopedPlan
from environment.question_generation.semantic_probes import (
    compile_candidate_probes, explicit_probe_contradictions,
    probe_candidate_semantics, validate_candidate_probe_records,
)
from environment.question_generation.question_review import critique_question


REFERENCE_5 = (
    "Let the first group consist of administrative areas with a geo of "
    "Central Valley region in the relevant records. "
    "Let the second group consist of entities in both entities with a "
    "population number of 1,203 drawn from the first group and "
    "administrative areas whose recorded 'neighbor' points to a member "
    "of entities with a male population of 488 drawn from the first group. "
    "Among entities with a postal code of 91683 drawn from the second "
    "group or entities in both entities listed as targets of the "
    "'neighbor' relation from Clearford and administrative areas in "
    "the relevant records, which entities have the smallest recorded "
    "female population? Include ties."
)
CANDIDATE_5_BAD = (
    "First, identify all administrative areas in the relevant records "
    "located in the Central Valley region. From this set, select those "
    "with a population of 1,203 to form the first group, and separately "
    "select administrative areas whose 'neighbor' points to an entity "
    "with a male population of 488 drawn from the first group; these "
    "form the second group. Next, consider entities drawn from the "
    "second group that have a postal code of 91683, along with entities "
    "listed as targets of the 'neighbor' relation from Clearford and "
    "any administrative areas in the relevant records. Among this "
    "combined set, which entities have the smallest recorded female "
    "population? Include ties."
)
REFERENCE_9 = (
    "Let the first group consist of Kingswell Airport if it has an iata "
    "code of BQG. Otherwise, this group is empty. Let the second group "
    "consist of cities in the United States with a start date of 1914 "
    "listed as targets of the 'named after' relation from members of "
    "the first group. Which ways with a logo of blue star emblem in "
    "the relevant records also have a start date of 1912 with a "
    "member of the second group as a recorded terminus?"
)
CANDIDATE_9_BAD = (
    "Assuming Kingswell Airport has the IATA code BQG, identify the "
    "cities in the United States that were targets of a 'named after' "
    "relation from Kingswell Airport and have a start date of 1914. "
    "Then, find the ways that have a blue star emblem logo, a start "
    "date of 1912, and a recorded terminus in one of those cities."
)


class V19ScopeCoverageTests(unittest.TestCase):
    def test_case3_label_free_real_candidate_triggers_owner_probe_and_guard(self):
        plan, reference = _plan_case3()
        candidate = (
            "First, identify all administrative areas where the recorded "
            "'location' points to Foxford. Second, define a set of entities "
            "that exist in administrative areas meeting both of the following "
            "criteria: their 'location' points to a member of the first group "
            "(where the area's geo is North Cedar region), and their "
            "'location' points to a member of the first group (where the "
            "area's population number is less than 1,765). Third, identify "
            "a set of entities that exist in administrative areas meeting "
            "at least one of the following criteria: the area's demonym is "
            "Atlas Demonym (and the area is in the second group), or the "
            "area has a female population of 8,007 and a male population "
            "of 490 (and the area is in the first group). How many distinct "
            "administrative areas are included in this third group?"
        )
        probes = compile_candidate_probes(plan, reference, candidate)
        self.assertTrue(any(
            p["kind"] == "condition_owner" and p["expected"] == "SELECTED"
            for p in probes
        ))
        self.assertNotIn("the second group consists", candidate.lower())
        self.assertIn(
            "selected_filter_moved_to_related_location",
            " ".join(explicit_probe_contradictions(probes, candidate)),
        )

    def test_case5_moved_first_group_boundary_is_caught(self):
        plan = ScopedPlan(
            [
                ("Let the first group consist of administrative areas with "
                 "a geo of Central Valley region in the relevant records.",),
                ("Let the second group consist of entities in both entities "
                 "with a population number of 1,203 drawn from the first "
                 "group and administrative areas with neighbor X.",),
            ],
            [
                _scope("FIND_ALL"),
                _scope("INTERSECT", inputs=[_scope("FIND_ALL"),
                                             _scope("FIND_ALL")]),
            ],
            _scope("SELECT_BETWEEN"),
        )
        probes = compile_candidate_probes(plan, REFERENCE_5, CANDIDATE_5_BAD)
        self.assertTrue(any(
            p["kind"] == "group_identity" and p["expected"] == "NOT_ALL"
            for p in probes
        ), probes)
        self.assertIn(
            "derived_filter_moved_to_first_group",
            " ".join(explicit_probe_contradictions(probes, CANDIDATE_5_BAD)),
        )
        faithful = (
            "The first group consists of all areas in Central Valley. "
            "The second group consists of entities both drawn from the "
            "first group with population 1,203 and satisfying a neighbor "
            "relation. Choose from the two independent candidate pools."
        )
        faithful_probes = compile_candidate_probes(plan, REFERENCE_5, faithful)
        self.assertEqual(explicit_probe_contradictions(faithful_probes, faithful), [])

    def test_case9_assuming_does_not_preserve_empty_branch(self):
        plan = ScopedPlan([], [], _scope("FIND_ALL"))
        probes = compile_candidate_probes(plan, REFERENCE_9, CANDIDATE_9_BAD)
        self.assertEqual(
            [(p["kind"], p["expected"]) for p in probes],
            [("conditional_scope", "CONDITIONAL")],
        )
        self.assertIn(
            "conditional_replaced_by_assumption",
            " ".join(explicit_probe_contradictions(probes, CANDIDATE_9_BAD)),
        )
        faithful = (
            "If Kingswell Airport has IATA code BQG, find the named-after "
            "cities and ways ending there; otherwise there are no matches."
        )
        safe_probes = compile_candidate_probes(plan, REFERENCE_9, faithful)
        self.assertEqual(explicit_probe_contradictions(safe_probes, faithful), [])

    def test_v19_model_receives_only_candidate_and_questions(self):
        plan = ScopedPlan([], [], _scope("FIND_ALL"))
        probes = compile_candidate_probes(plan, REFERENCE_9, CANDIDATE_9_BAD)
        class Reader:
            def complete_json(self, system, user, **kwargs):
                data = json.loads(user)
                assert data["candidate_question"] == CANDIDATE_9_BAD
                assert "reference_question" not in data
                assert "expected" not in user
                return {"answers": [
                    {"id": q["id"], "answer": "ASSUMED"}
                    for q in data["probes"]
                ]}
        records = probe_candidate_semantics(Reader(), REFERENCE_9,
                                            CANDIDATE_9_BAD, probes)
        self.assertEqual(records[0]["answer"], "ASSUMED")
        with self.assertRaises(ValueError):
            validate_candidate_probe_records(probes, records)

    def test_wrong_atomic_ids_in_negative_review_veto_without_protocol_error(self):
        checks = [
            {"id": "R1", "kind": "set_logic",
             "requirements": [{"id": "C3", "expected": "AND"}]},
            {"id": "R2", "kind": "relation_scope",
             "requirements": [{"id": "C4", "expected": "location"}]},
        ]
        class WrongIdReviewer:
            def complete_json(self, system, user, **kwargs):
                return {
                    "verdict": "pass", "issues": [],
                    "summary": "The location scope is changed.",
                    "checks": [
                        {"id": "R1", "status": "changed",
                         "explanation": "The candidate turns AND into OR.",
                         "problem_ids": ["C4"]},
                        {"id": "R2", "status": "preserved",
                         "explanation": "The location relation is retained."},
                    ],
                }
        result = critique_question(
            WrongIdReviewer(), "Both clauses", "Either clause", checks,
        )
        self.assertEqual(result["verdict"], "revise")
        self.assertEqual(result["checks"][0]["problem_ids"], [])
        self.assertEqual(result["checks"][0]["status"], "changed")
        self.assertTrue(result["protocol_notes"])

    def test_dict_group_map_normalizes_without_inventing_missing_groups(self):
        checks = [{"id": "R1", "kind": "set_logic",
                   "requirements": [{"id": "C1", "expected": "OR"}]}]
        class MapReviewer:
            def complete_json(self, system, user, **kwargs):
                return {
                    "verdict": "revise", "issues": [],
                    "summary": "Changed group",
                    "checks": {"R1": {
                        "status": "changed",
                        "explanation": "Changed AND to OR.",
                    }},
                }
        result = critique_question(MapReviewer(), "both", "either", checks)
        self.assertEqual(result["checks"][0]["id"], "R1")
        self.assertEqual(result["checks"][0]["status"], "changed")
        self.assertTrue(result["protocol_notes"])


if __name__ == "__main__":
    unittest.main()
