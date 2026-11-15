"""Regression checks for reusing canonical Stage-1 quality with typed Macros."""
from __future__ import annotations

from copy import deepcopy
import unittest

from environment.generate_question.typed_macro_quality import (
    audit_terminal_goal_coverage,
    audit_typed_handoffs,
    audit_typed_scene,
)
from environment.macro_v3.typed_scene import compose_scene, execute, validate_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph


def fixture():
    namespace = "https://yago-knowledge.org/resource/"
    p = "https://schema.org/"
    edges = []
    for i in range(14):
        edges.extend([
            Edge(namespace+f"Person{i}", p+"birthPlace", namespace+f"Town{i}"),
            Edge(namespace+f"Town{i}", p+"containedInPlace", namespace+f"State{i}"),
            Edge(namespace+f"State{i}", p+"containedInPlace", namespace+f"Country{i}"),
            Edge(namespace+f"Person{i}", p+"memberOf", namespace+f"Institute{i}"),
            Edge(namespace+f"Person{i}", p+"knows", namespace+f"Colleague{i}"),
        ])
    return KnowledgeGraph(edges=edges)


class CanonicalTypedQualityTests(unittest.TestCase):
    def test_all_requested_roles_reach_final_workproduct(self):
        kg = fixture()
        encountered = set()
        for ordinal in range(24):
            scene = compose_scene(kg, seed=5, ordinal=ordinal)
            audit = audit_typed_scene(scene, execute)
            self.assertTrue(audit["goal_coverage"]["passed"])
            self.assertTrue(audit["handoffs"]["passed"])
            self.assertEqual(
                audit["goal_coverage"]["expected_results"],
                audit["goal_coverage"]["delivered_results"],
            )
            encountered.update(n["role"] for n in scene["macro_plan"]["nodes"])
            self.assertTrue(validate_scene(scene, kg)["passed"])
        self.assertTrue({"comparison", "verification"} <= encountered)

    def test_final_dropped_verification_is_rejected(self):
        kg = fixture()
        scene = next(
            compose_scene(kg, seed=13, ordinal=i)
            for i in range(30)
            if any(n["role"] == "verification"
                   for n in compose_scene(kg, seed=13, ordinal=i)["macro_plan"]["nodes"])
        )
        broken = deepcopy(scene)
        terminal = broken["macro_plan"]["nodes"][-1]
        key = next(k for k, link in terminal["input_ports"].items()
                   if link["type"] == "VerifiedClaim")
        del terminal["input_ports"][key]
        with self.assertRaisesRegex(ValueError, "typed_goal_missing_result_handoff"):
            audit_terminal_goal_coverage(broken)

    def test_ineffective_edge_is_not_silently_accepted(self):
        kg = fixture()
        scene = compose_scene(kg, seed=21, ordinal=2)
        def unchanged(_scene, *, input_overrides=None):
            return [], _scene["gold_answer"][0]
        audit = audit_typed_handoffs(scene, unchanged)
        self.assertFalse(audit["passed"])
        self.assertEqual(len(audit["ineffective"]), audit["total_edges"])

    def test_original_evidence_carries_actual_witness(self):
        kg = fixture()
        scene = next(
            compose_scene(kg, seed=11, ordinal=i)
            for i in range(40)
            if "original_evidence" in
            compose_scene(kg, seed=11, ordinal=i)["macro_plan"]["nodes"][-1]["input_ports"]
        )
        self.assertTrue(scene["gold_answer"][0]["original_witness_refs"])
        audit = audit_typed_handoffs(scene, execute)
        original = next(row for row in audit["edges"]
                        if row["to_port"] == "original_evidence")
        self.assertTrue(original["answer_sensitive"] or original["execution_blocked"])


if __name__ == "__main__":
    unittest.main()
