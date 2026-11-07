"""Stage-2 v20 public semantic coverage and tamper/false-positive regressions.

These tests use PUBLIC synthetic ScopedPlan trees and scripted readers.
They do NOT claim that a real language model verifies semantics reliably.
"""
from __future__ import annotations

import copy
import json
import unittest

from environment.question_generation.scoped_surface import ScopedPlan
from environment.question_generation.coverage_review import (
    MAX_COVERAGE_PROBES, PROBE_BATCH_SIZE, UnverifiableCoverage,
    compile_coverage_probes, coverage_manifest, validate_coverage_manifest,
)
from environment.question_generation.semantic_probes import probe_candidate_semantics


def node(op, *children, relation=None, direction=None, conditions=()):
    result = {"definition": op, "inputs": list(children),
              "conditions": list(conditions)}
    if relation is not None:
        result.update(relation=relation, direction=direction)
    return result


class CoverageReviewTests(unittest.TestCase):
    def plan(self):
        # A public result set requiring both a relation hop and an intersection.
        base = node("FIND_ALL")
        related = node("RELATE", base, relation="location",
                       direction="backward")
        other = node("FIND_ALL", conditions=[{
            "attribute": "population number", "comparison": "less than",
            "value": {"type": "number", "value": "1765"},
        }])
        intersect = node("INTERSECT", related, other)
        query = node("COUNT", intersect)
        reference = (
            "Let the first group consist of entities in both entities whose "
            "recorded 'location' points to a member of North Cedar areas "
            "and administrative areas with a population less than 1,765. "
            "How many distinct administrative areas are in the first group?"
        )
        return ScopedPlan([(reference.split(". ")[0] + ".",)],
                          [intersect], query), reference

    def test_relation_path_and_final_answer_are_compulsory_without_group_labels(self):
        plan, ref = self.plan()
        candidate = (
            "Find areas in North Cedar that have population below 1765. "
            "How many such areas are there?"
        )
        probes = compile_coverage_probes(plan, ref, candidate)
        self.assertIn(("relation_path", "BACKWARD"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertIn(("set_scope", "AND"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertIn(("answer_type", "COUNT"),
                      [(p["kind"], p["expected"]) for p in probes])
        self.assertEqual(len(probes), len(set(p["id"] for p in probes)))
        # Even though candidate stripped "the first group", no required
        # public relation hop is dropped.
        self.assertNotIn("the first group", candidate.lower())

    def test_candidate_only_model_sees_no_reference_or_gold(self):
        plan, ref = self.plan()
        candidate = "Count areas under 1765 that have a recorded location pointing to North Cedar."
        probes = compile_coverage_probes(plan, ref, candidate)
        class Reader:
            def complete_json(self, system, user, **kwargs):
                data = json.loads(user)
                self_keys = set(data)
                assert self_keys == {"candidate_question", "probes"}
                assert data["candidate_question"] == candidate
                assert "reference_question" not in user
                assert '"expected"' not in user
                return {"answers": [
                    {"id": p["id"], "answer": (
                        "BACKWARD" if "Which structure" in p["question"] else
                        "AND" if "set-building clause" in p["question"] else
                        "SELECTED" if "which entity must satisfy" in p["question"] else
                        "COUNT"
                    )} for p in data["probes"]
                ]}
        records = []
        for start in range(0, len(probes), PROBE_BATCH_SIZE):
            batch = probes[start:start + PROBE_BATCH_SIZE]
            records.extend(probe_candidate_semantics(
                Reader(), ref, candidate, batch,
            ))
        manifest = coverage_manifest(probes, records)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["required"], manifest["verified"])
        validate_coverage_manifest(probes, records, manifest)

    def test_wrong_hop_cannot_certify_even_when_count_and_and_are_correct(self):
        plan, ref = self.plan()
        probes = compile_coverage_probes(plan, ref, "Count direct North Cedar areas under 1765.")
        records = [{"id": p["id"], "kind": p["kind"],
                    "expected": p["expected"], "answer": (
                        "DIRECT" if p["kind"] == "relation_path" else p["expected"]
                    )} for p in probes]
        status = coverage_manifest(probes, records)
        self.assertEqual(status["status"], "incomplete")
        self.assertLess(status["verified"], status["required"])
        with self.assertRaises(ValueError):
            validate_coverage_manifest(probes, records, status)

    def test_tampering_and_missing_relation_records_fail_closed(self):
        plan, ref = self.plan()
        probes = compile_coverage_probes(plan, ref, "Count qualifying areas.")
        records = [{"id": p["id"], "kind": p["kind"],
                    "expected": p["expected"], "answer": p["expected"]}
                   for p in probes]
        manifest = coverage_manifest(probes, records)
        validate_coverage_manifest(probes, records, manifest)
        for label, altered_records, altered_manifest in (
            ("missing", records[:-1], manifest),
            ("relation", [
                dict(x, answer="FORWARD") if x["kind"] == "relation_path" else x
                for x in records
            ], manifest),
            ("hash", records, dict(manifest, required=0)),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_coverage_manifest(probes, altered_records, altered_manifest)

    def test_output_only_tasks_are_checked_not_zero_probe(self):
        plan = ScopedPlan([], [], node("QUERY_NAME", node("FIND_ALL")))
        probes = compile_coverage_probes(plan, "Which entities are listed?", "What entities are listed?")
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["kind"], "answer_type")
        self.assertEqual(probes[0]["expected"], "ENTITIES")
        with self.assertRaises(ValueError):
            validate_coverage_manifest(probes, [], coverage_manifest(probes, []))

    def test_all_public_relations_are_required_even_if_candidate_mentions_none(self):
        base = node("FIND_ALL")
        one = node("RELATE", base, relation="named after", direction="forward")
        two = node("RELATE", one, relation="location", direction="backward")
        three = node("RELATE", two, relation="participant", direction="backward")
        plan = ScopedPlan([("Locate entities along the public three-hop chain.",)],
                          [three], node("QUERY_NAME", three))
        probes = compile_coverage_probes(plan, "Reference describes three hops.", "Which entities?")
        paths = [(p["relation"], p["expected"]) for p in probes
                 if p["kind"] == "relation_path"]
        self.assertEqual(set(paths), {
            ("named after", "FORWARD"), ("location", "BACKWARD"),
            ("participant", "BACKWARD"),
        })
        self.assertEqual(len(paths), 3)

    def test_budget_fail_closed_not_silent_truncation(self):
        base = node("FIND_ALL")
        for i in range(MAX_COVERAGE_PROBES + 2):
            base = node("RELATE", base, relation=f"link_{i}", direction="backward")
        plan = ScopedPlan([("Many chained relations.",)], [base],
                          node("QUERY_NAME", base))
        with self.assertRaises(UnverifiableCoverage):
            compile_coverage_probes(plan, "Many chained relations.", "Which entities?")


if __name__ == "__main__":
    unittest.main()
