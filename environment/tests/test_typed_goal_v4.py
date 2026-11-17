"""Executable Stage 1 typed Goal v4 regression and migration tests."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import unittest

from environment.generate_question.typed_goal_v4 import VERSION, KINDS


def fixture():
    from environment.macro_v3.yago import Edge, KnowledgeGraph
    p = "http://schema.org/"
    return KnowledgeGraph(edges=[
        Edge("https://example.org/person", p+"birthPlace",
             "https://example.org/town"),
        Edge("https://example.org/town", p+"containedInPlace",
             "https://example.org/region"),
        Edge("https://example.org/region", p+"containedInPlace",
             "https://example.org/country"),
    ], input_file="typed-goal-v4-test.ttl", scanned=3)


class TypedGoalV4Tests(unittest.TestCase):
    def test_deterministic_three_goal_kinds_and_macro_budget(self):
        from environment.generate_question.task_plan import make_spec, validate_spec
        from environment.generate_question.question import (
            reference_question, guard_question,
        )
        from environment.generate_question.quality import (
            compile_and_validate, validate_replay,
        )
        from environment.implement_subtask.realize import realize_scene
        from environment.generate_question.pipeline import quality_report

        kg = fixture()
        kinds, examples, signatures = Counter(), {}, set()
        for ordinal in range(26):
            diff = "medium" if ordinal % 2 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=diff,
                world_mode="compose", composer=VERSION,
            )
            self.assertEqual(validate_spec(spec), [])
            plan = spec["macro_plan"]
            kind = plan["goal_spec"]["kind"]
            kinds[kind] += 1
            self.assertEqual(plan["composer_policy"], VERSION)
            self.assertLessEqual(len(plan["macro_tasks"]), 9)
            self.assertEqual(
                plan["macro_tasks"][-1]["output_schema"],
                {"answer": plan["goal"]["answer_type"]},
            )
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [])
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"]["passed"])
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            self.assertEqual(len(scene["gold_answer"]), 1)
            actual = scene["gold_answer"][0]["answer"]
            self.assertIsInstance(
                actual, int if kind == "aggregation" else str
            )
            signatures.add((kind, tuple(
                (t["operation"], tuple(t["depends_on"]))
                for t in plan["macro_tasks"]
            )))
            examples.setdefault(kind, scene)
        self.assertEqual(set(kinds), set(KINDS), kinds)
        self.assertGreater(len(signatures), 8)

        comparison = examples["comparison"]
        comp_plan = comparison["problem_spec"]["macro_plan"]
        tasks = comp_plan["macro_tasks"]
        left = next(t for t in tasks if t["operation"] == "lookup_metric_a")
        right = next(t for t in tasks if t["operation"] == "lookup_metric_b")
        self.assertEqual(left["depends_on"], right["depends_on"])
        self.assertEqual(
            tasks[-1]["depends_on"], [left["macro_id"], right["macro_id"]]
        )
        self.assertTrue(any(
            n["op"] == "derive_comparison"
            for n in comparison["gold_program"]["nodes"]
        ))
        self.assertNotEqual(comparison["gold_answer"][0]["answer"], "equal")
        self.assertEqual(
            {t["source_scope"][0] for t in (left, right)},
            {"metric_a", "metric_b"},
        )

        aggregation = examples["aggregation"]
        self.assertTrue(any(
            n["op"] == "group_count"
            for n in aggregation["gold_program"]["nodes"]
        ))
        self.assertTrue(any(
            n["op"] == "project"
            and n.get("rename_fields") == {"count": "answer"}
            for n in aggregation["gold_program"]["nodes"]
        ))
        self.assertEqual(
            aggregation["world"]["sources"]["observation_log"][
                "access_policy"]["key_cardinality"], "many"
        )
        qa = quality_report([comparison, aggregation], no_progress=True)
        self.assertEqual(qa["replay_pass"], 2)
        self.assertEqual(
            qa["evidence_acquisition"][
                "task_plan_keyed_source_contract_mismatches"], 0,
        )

    def test_numeric_goal_world_tampering_fails_closed(self):
        from environment.generate_question.task_plan import (
            make_spec, validate_spec, SALT,
        )
        from environment.schema import digest
        kg = fixture()
        example = None
        for ordinal in range(30):
            candidate = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty="medium",
                world_mode="compose", composer=VERSION,
            )
            if candidate["macro_plan"]["goal_spec"]["kind"] == "comparison":
                example = candidate
                break
        self.assertIsNotNone(example)
        broken = deepcopy(example)
        plan = broken["macro_plan"]
        world = plan["world_binding"]["world_model"]["goal_measurements"]
        target = plan["world_binding"]["target_case"]
        world["by_case"][target]["metric_a"] += 7
        # Even an adversary who recomputes the outer frozen hash cannot
        # change canonical evidence measures silently.
        broken["knowledge_contract_hash"] = digest(
            plan, broken["knowledge_witness"],
            broken["seed"], broken["ordinal"], SALT,
        )
        self.assertIn(
            "v4_metric_world_not_canonical", validate_spec(broken)
        )

    def test_v4_program_projection_rename_is_type_checked(self):
        from environment.generate_question.program_ir import validate_program
        valid = {
            "nodes": [
                {"id": "a", "op": "scan", "inputs": [],
                 "source_id": "observation_log"},
                {"id": "b", "op": "group_count", "inputs": ["a"],
                 "field": "entity_final"},
                {"id": "c", "op": "project", "inputs": ["b"],
                 "fields": ["count"], "rename_fields": {"count": "answer"}},
            ], "output": "c",
        }
        self.assertEqual(validate_program(valid), [])
        bad = deepcopy(valid)
        bad["nodes"][-1]["rename_fields"] = {"wrong": "answer"}
        self.assertIn("project_invalid_renames", validate_program(bad))


if __name__ == "__main__":
    unittest.main()
