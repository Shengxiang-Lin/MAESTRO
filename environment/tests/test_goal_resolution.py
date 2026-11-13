"""Goal Resolution v1: numeric derivation, source provenance and old replay."""
from __future__ import annotations
import copy
import unittest
from collections import Counter

from environment.generate_question.spec import make_spec, spec_issues
from environment.generate_question.question import reference_question, guard_question
from environment.generate_question.goal_resolution import (
    GOAL_RESOLUTION_VERSION, goal_values,
)
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.program_ir import (
    evaluate_program, answer_values, MissingSourceError,
)
from environment.implement_subtask.realize import realize_scene


def case(i):
    return make_spec(
        seed_record={"record_id": "goal-v1-"+str(i), "topic": "Premier League",
                     "domain_hint": "sports"},
        ordinal=i, run_seed=97, difficulty="hard" if i%2 else "medium",
        macro_first_v2_share=100,
    )


class GoalResolutionTests(unittest.TestCase):
    def test_compiler_and_world_use_derived_goal_values(self):
        counts = Counter()
        for i in range(100):
            spec = case(i)
            kind = spec["macro_plan"]["goal_resolution"]["kind"]
            counts[kind] += 1
            self.assertEqual(spec["goal_resolution_version"],
                             GOAL_RESOLUTION_VERSION)
            self.assertEqual(spec_issues(spec), [])
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [], (i, question))
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(len(scene["gold_answer"]), 1)
            if kind == "direct_lookup":
                self.assertIn("archive", scene["world"]["sources"])
            else:
                self.assertNotIn("archive", scene["world"]["sources"])
                self.assertIn("stat_a", scene["world"]["sources"])
                self.assertIn("stat_b", scene["world"]["sources"])
                program_ops = [
                    node["op"] for node in scene["gold_program"]["nodes"]
                ]
                expected_op = ("derive_comparison" if kind == "comparison"
                               else "derive_percentage")
                self.assertIn(expected_op, program_ops)
                self.assertEqual(program_ops.count("lookup_by_key") >= 2, True)
                oracle = realize_scene(scene)
                self.assertEqual(oracle["macro_tasks"][-1]["answer"], scene["gold_answer"])
                self.assertTrue(all(
                    entry["source_fact_refs"]
                    for entry in oracle["macro_tasks"][-1][
                        "answer_with_provenance"
                    ]
                ))
                for removed in ("stat_a", "stat_b"):
                    world = dict(scene["world"]["sources"])
                    world.pop(removed)
                    with self.assertRaises(MissingSourceError):
                        evaluate_program(scene["gold_program"], sources=world)
        for kind in ("direct_lookup", "comparison", "calculation"):
            self.assertGreater(counts[kind], 15)

    def test_typed_operations_and_provenance(self):
        self.assertEqual(goal_values("comparison", 8, 7), "first")
        self.assertEqual(goal_values("comparison", 3, 7), "second")
        self.assertEqual(goal_values("comparison", 4, 4), "equal")
        self.assertEqual(goal_values("calculation", 36, 48), 75)
        self.assertEqual(goal_values("calculation", 2, 3), 66)
        with self.assertRaises(ValueError):
            goal_values("calculation", 3, 0)
        with self.assertRaises(ValueError):
            goal_values("calculation", 9, 3)
        with self.assertRaises(ValueError):
            goal_values("comparison", True, 3)

if __name__ == "__main__":
    unittest.main()
