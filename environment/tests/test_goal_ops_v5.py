"""Orthogonal operation-DAG × output-contract Stage 1 regression suite."""
from __future__ import annotations

import copy
import json
import unittest

from environment.generate_question.goal_ops_v5 import (
    VERSION, PROFILES, OUTPUTS, selected_profile_output, evaluate_goal_value,
    validate_operation_graph,
)


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
    ], input_file="operation-dag-v5-test.ttl", scanned=3)


class GoalOperationDagTests(unittest.TestCase):
    def test_type_checked_orthogonal_plans_and_full_replay(self):
        from environment.generate_question.task_plan import (
            make_spec, validate_spec,
        )
        from environment.generate_question.question import (
            reference_question, guard_question,
        )
        from environment.generate_question.quality import (
            compile_and_validate, validate_replay,
        )
        from environment.implement_subtask.realize import realize_scene
        from environment.generate_question.pipeline import quality_report

        chosen = {}
        for ordinal in range(120):
            profile, output = selected_profile_output(17, ordinal)
            chosen.setdefault((profile, output), ordinal)
        self.assertEqual(set(p for p, _ in chosen), set(PROFILES))
        self.assertEqual(set(o for _, o in chosen), set(OUTPUTS))

        scenes = []
        for (profile, shape), ordinal in chosen.items():
            spec = make_spec(
                fixture(), seed=17, ordinal=ordinal,
                difficulty="medium" if ordinal % 2 else "hard",
                world_mode="compose", composer=VERSION,
            )
            self.assertEqual(validate_spec(spec), [], (profile, shape))
            plan = spec["macro_plan"]
            goal = plan["goal_spec"]
            self.assertNotIn("kind", goal)
            self.assertEqual(goal["source_profile"], profile)
            self.assertEqual(goal["output_spec"]["kind"], shape)
            self.assertEqual(validate_operation_graph(goal), [])
            self.assertLessEqual(len(plan["macro_tasks"]), 9)
            self.assertGreaterEqual(len(goal["operations"]), 1)
            self.assertLessEqual(len(goal["operations"]), 8)
            self.assertEqual(
                plan["macro_tasks"][-1]["output_schema"],
                {"answer": plan["goal"]["answer_type"]},
            )
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [], (profile, shape))
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(
                scene["difficulty_audit"]["macro_first_audit"][
                    "candidate_information_necessity"]["passed"], True,
            )
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "source_counterfactuals_pass"])
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            answer = scene["gold_answer"][0]["answer"]
            source_world = plan["world_binding"]["world_model"].get(
                "goal_measurements"
            )
            if source_world:
                target = plan["world_binding"]["target_case"]
                expected = evaluate_goal_value(
                    source_world["by_case"][target], goal,
                )
                self.assertEqual(answer, expected, (profile,shape))
            if shape in {"structured_multi_field",
                         "evidence_based_explanation"}:
                self.assertEqual(
                    set(answer),
                    set(goal["output_spec"]["fields"]),
                )
            if profile != "entity_resolution":
                self.assertTrue(any(
                    node["op"] == "goal_format"
                    for node in scene["gold_program"]["nodes"]
                ))
            scenes.append(scene)

        report = quality_report(scenes, no_progress=True)
        self.assertEqual(report["evidence_acquisition"][
            "task_plan_keyed_source_contract_mismatches"], 0)
        self.assertEqual(
            set(report["knowledge_world_grounding"]["single_task_plan"][
                "goal_output_distribution"]), set(OUTPUTS),
        )

    def test_freeze_operations_and_output_fields(self):
        from environment.generate_question.task_plan import (
            make_spec, validate_spec, SALT,
        )
        from environment.schema import digest
        ordinal = next(
            i for i in range(160)
            if selected_profile_output(17, i)[0] == "comparison"
            and selected_profile_output(17, i)[1] ==
                "structured_multi_field"
        )
        spec = make_spec(
            fixture(), seed=17, ordinal=ordinal, difficulty="medium",
            world_mode="compose", composer=VERSION,
        )
        changed = copy.deepcopy(spec)
        goal = changed["macro_plan"]["goal_spec"]
        goal["output_spec"]["bindings"]["count_difference"] = "winner"
        changed["knowledge_contract_hash"] = digest(
            changed["macro_plan"], changed["knowledge_witness"],
            changed["seed"], changed["ordinal"], SALT,
        )
        self.assertIn("task_plan_typed_goal_not_frozen",
                      validate_spec(changed))
        bad = copy.deepcopy(spec)
        bgoal = bad["macro_plan"]["goal_spec"]
        bgoal["operations"][-1]["inputs"] = ["unknown"]
        self.assertTrue(validate_operation_graph(bgoal))
        self.assertNotEqual(
            json.dumps(goal, sort_keys=True),
            json.dumps(spec["macro_plan"]["goal_spec"], sort_keys=True),
        )

    def test_program_rejects_wrong_output_field_type(self):
        from environment.generate_question.program_ir import validate_program
        valid = {"nodes": [
            {"id": "a", "op": "scan", "inputs": [],
             "source_id": "any"},
            {"id": "b", "op": "goal_format", "inputs": ["a"],
             "output_kind": "single_number",
             "field_types": {"answer": "int"},
             "bindings": {"answer": "metric_a"},
             "output_field": "answer"},
        ], "output": "b"}
        self.assertEqual(validate_program(valid), [])
        invalid = copy.deepcopy(valid)
        invalid["nodes"][-1]["output_kind"] = "unsupported"
        self.assertIn("v5_invalid_goal_format",
                      validate_program(invalid))


if __name__ == "__main__":
    unittest.main()
