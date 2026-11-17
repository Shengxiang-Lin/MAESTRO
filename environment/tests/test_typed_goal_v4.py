"""End-to-end and tamper-evidence tests for 60-question-inspired v4 Goals."""
from __future__ import annotations

import copy
import unittest

from environment.generate_question.typed_goal_v4 import (
    VERSION, KINDS, selected_kind, _final_answer_for,
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
    ], input_file="typed-goal-v4-test.ttl", scanned=3)


class TypedGoalV4Tests(unittest.TestCase):
    def test_distinct_goal_type_sampling_and_executable_replay(self):
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

        kg = fixture()
        selected = {}
        for ordinal in range(240):
            kind = selected_kind(17, ordinal)
            selected.setdefault(kind, ordinal)
            if len(selected) == len(KINDS):
                break
        self.assertEqual(set(selected), set(KINDS))
        scenes = []
        for kind, ordinal in selected.items():
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=(
                    "hard" if ordinal % 2 else "medium"
                ), world_mode="compose", composer=VERSION,
            )
            self.assertEqual(validate_spec(spec), [], kind)
            plan = spec["macro_plan"]
            self.assertEqual(plan["goal_spec"]["kind"], kind)
            self.assertLessEqual(len(plan["macro_tasks"]), 9)
            self.assertEqual(
                plan["macro_tasks"][-1]["output_schema"],
                {"answer": plan["goal"]["answer_type"]},
            )
            draft = reference_question(spec)
            self.assertEqual(guard_question(draft, spec), [], kind)
            scene = compile_and_validate(spec, draft)
            self.assertEqual(validate_replay(scene)["status"], "valid", kind)
            self.assertTrue(scene["difficulty_audit"][
                "macro_first_audit"]["candidate_information_necessity"][
                    "passed"], kind)
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "source_counterfactuals_pass"], kind)
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            self.assertEqual(len(scene["gold_answer"]), 1)
            answer = scene["gold_answer"][0]["answer"]
            expected_type = {
                "numeric_ratio": float,
                "numeric_difference": int,
                "temporal_gap": int,
                "geographic_distance": int,
                "aggregation": int,
                "structured_answer": dict,
            }.get(kind, str)
            self.assertIsInstance(answer, expected_type, kind)
            if kind == "structured_answer":
                self.assertEqual(
                    set(answer),
                    {"entity", "higher_source", "count_difference"},
                )
            if kind != "entity_resolution":
                measurements = plan["world_binding"]["world_model"][
                    "goal_measurements"
                ]
                self.assertEqual(
                    answer,
                    _final_answer_for(
                        measurements["by_case"][
                            plan["world_binding"]["target_case"]
                        ], kind,
                    ),
                )
            scenes.append(scene)

        one = next(s for s in scenes if s["problem_spec"]["macro_plan"][
            "goal_spec"]["kind"] == "comparison")
        tasks = one["problem_spec"]["macro_plan"]["macro_tasks"]
        left = next(t for t in tasks if t["operation"] == "lookup_metric_a")
        right = next(t for t in tasks if t["operation"] == "lookup_metric_b")
        self.assertEqual(left["depends_on"], right["depends_on"])
        self.assertEqual(
            tasks[-1]["depends_on"], [left["macro_id"], right["macro_id"]]
        )

        ops_by_kind = {
            "numeric_difference": "derive_arithmetic",
            "numeric_ratio": "derive_arithmetic",
            "temporal_gap": "derive_date_gap",
            "geographic_distance": "derive_great_circle",
            "aggregation": "group_count",
            "comparison": "derive_comparison",
            "structured_answer": "compose_typed_result",
        }
        for kind, operation in ops_by_kind.items():
            scene = next(s for s in scenes if s["problem_spec"][
                "macro_plan"]["goal_spec"]["kind"] == kind)
            self.assertIn(
                operation, [node["op"] for node in scene[
                    "gold_program"]["nodes"]], kind,
            )
        report = quality_report(scenes, no_progress=True)
        self.assertEqual(report["evidence_acquisition"][
            "task_plan_keyed_source_contract_mismatches"], 0)

    def test_metric_world_and_goal_plan_are_hash_bound(self):
        from environment.generate_question.task_plan import (
            make_spec, validate_spec, SALT,
        )
        from environment.schema import digest
        kg = fixture()
        ordinal = next(i for i in range(90)
                       if selected_kind(17, i) == "numeric_difference")
        spec = make_spec(
            kg, seed=17, ordinal=ordinal, difficulty="medium",
            world_mode="compose", composer=VERSION,
        )
        bad = copy.deepcopy(spec)
        plan = bad["macro_plan"]
        world = plan["world_binding"]["world_model"]["goal_measurements"]
        target = plan["world_binding"]["target_case"]
        world["by_case"][target]["metric_a"] += 5
        bad["knowledge_contract_hash"] = digest(
            plan, bad["knowledge_witness"], bad["seed"], bad["ordinal"], SALT,
        )
        self.assertIn(
            "v4_metric_world_not_canonical", validate_spec(bad),
        )
        schema_changed = copy.deepcopy(spec)
        schema_changed["macro_plan"]["goal_spec"]["kind"] = "comparison"
        schema_changed["knowledge_contract_hash"] = digest(
            schema_changed["macro_plan"], schema_changed["knowledge_witness"],
            schema_changed["seed"], schema_changed["ordinal"], SALT,
        )
        self.assertIn(
            "task_plan_typed_goal_not_frozen",
            validate_spec(schema_changed),
        )

    def test_project_rename_and_numeric_contracts(self):
        from environment.generate_question.program_ir import (
            validate_program, ProgramError, great_circle_km,
        )
        program = {
            "nodes": [
                {"id": "a", "op": "scan", "inputs": [],
                 "source_id": "observation_log"},
                {"id": "b", "op": "group_count", "inputs": ["a"],
                 "field": "entity_final"},
                {"id": "c", "op": "project", "inputs": ["b"],
                 "fields": ["count"],
                 "rename_fields": {"count": "answer"}},
            ], "output": "c",
        }
        self.assertEqual(validate_program(program), [])
        bad = copy.deepcopy(program)
        bad["nodes"][-1]["rename_fields"] = {"wrong": "answer"}
        self.assertIn("project_invalid_renames", validate_program(bad))
        self.assertGreater(great_circle_km(30, -90, 31, -89), 0)
        with self.assertRaises(ProgramError):
            great_circle_km(999, -90, 31, -89)


if __name__ == "__main__":
    unittest.main()
