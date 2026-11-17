"""Output-first Goal construction and Question Critic contract regressions."""
from __future__ import annotations

import copy
import json
import unittest

from environment.generate_question.goal_ops_v6 import (
    VERSION, OUTPUT_KINDS, _backward_goal,
    evaluate_goal_value, selected_output, validate_operation_graph,
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
    ], input_file="output-first-v6-test.ttl", scanned=3)


class OutputFirstGoalTests(unittest.TestCase):
    def test_output_first_reuse_and_partial_order(self):
        # Output is sampled BEFORE the Goal graph. Mixed demands share
        # identical upstream Lookup nodes (common subexpression reuse).
        selected = [
            selected_output(17, ordinal) for ordinal in range(160)
        ]
        self.assertEqual({s["kind"] for s in selected}, set(OUTPUT_KINDS))
        self.assertGreater(len({
            tuple(s["bindings"].values()) for s in selected
        }), 15)
        combinations = set()
        for output in selected:
            goal = _backward_goal(output)
            self.assertEqual(validate_operation_graph(goal), [])
            ops = goal["operations"]
            self.assertEqual(len({n["id"] for n in ops}), len(ops))
            seen = set()
            for node in ops:
                self.assertTrue(set(node["inputs"]) <= seen)
                seen.add(node["id"])
            combinations.add(tuple(goal["capabilities"]))
        self.assertTrue(any(len(c) >= 2 for c in combinations))
        mixed = _backward_goal({
            "kind": "structured_multi_field",
            "fields": {"duration_days": "int", "distance_km": "int",
                       "entity": "str"},
            "bindings": {
                "duration_days": "duration_days",
                "distance_km": "distance_km", "entity": "entity",
            },
        })
        self.assertEqual(
            mixed["capabilities"], ["geographic", "temporal"]
        )
        self.assertEqual(
            sum(node["id"] == "a" for node in mixed["operations"]), 1
        )
        self.assertEqual(
            sum(node["id"] == "b" for node in mixed["operations"]), 1
        )
        self.assertEqual(
            set(mixed["source_fields"]["metric_a"]),
            {"entity_final", "date_a", "latitude_a", "longitude_a"},
        )
        tampered = copy.deepcopy(mixed)
        tampered["operations"].reverse()
        self.assertTrue(validate_operation_graph(tampered))

    def test_full_world_gold_replay_and_stage2(self):
        from environment.generate_question.task_plan import make_spec, validate_spec
        from environment.generate_question.question import (
            reference_question, guard_question,
        )
        from environment.generate_question.quality import (
            compile_and_validate, validate_replay,
        )
        from environment.implement_subtask.realize import realize_scene
        from environment.generate_question.pipeline import quality_report

        selected = {}
        for ordinal in range(180):
            output = selected_output(17, ordinal)
            # Cover every shape plus representative heterogeneous sources.
            signature = (
                output["kind"], tuple(sorted(_backward_goal(output)["capabilities"]))
            )
            selected.setdefault(signature, ordinal)
        self.assertEqual(
            {shape for shape, _ in selected}, set(OUTPUT_KINDS)
        )
        self.assertTrue(any(
            len(k[1]) >= 2 for k in selected
        ))
        scenes = []
        for (shape, capabilities), ordinal in selected.items():
            spec = make_spec(
                fixture(), seed=17, ordinal=ordinal,
                difficulty="medium" if ordinal % 2 else "hard",
                world_mode="compose", composer=VERSION,
            )
            self.assertEqual(validate_spec(spec), [], (shape,capabilities))
            plan = spec["macro_plan"]
            goal = plan["goal_spec"]
            self.assertNotIn("kind", goal)
            self.assertEqual(validate_operation_graph(goal), [])
            self.assertLessEqual(len(plan["macro_tasks"]), 11)
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [],
                             (shape,capabilities))
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            audit = scene["difficulty_audit"]["macro_first_audit"]
            self.assertTrue(audit["candidate_information_necessity"]["passed"])
            self.assertTrue(audit["source_counterfactuals_pass"])
            actual = scene["gold_answer"][0]["answer"]
            if goal["capabilities"]:
                world = plan["world_binding"]["world_model"][
                    "goal_measurements"
                ]
                target = plan["world_binding"]["target_case"]
                self.assertEqual(
                    actual, evaluate_goal_value(world["by_case"][target], goal)
                )
            if shape in ("structured_multi_field",
                         "evidence_based_explanation"):
                self.assertEqual(set(actual), set(goal["output_spec"]["fields"]))
            realized = realize_scene(scene, strict=True)
            self.assertEqual(
                len(realized["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            scenes.append(scene)

        report = quality_report(scenes, no_progress=True)
        self.assertEqual(
            set(report["knowledge_world_grounding"][
                "single_task_plan"]["goal_output_distribution"]),
            set(OUTPUT_KINDS),
        )
        self.assertEqual(report["evidence_acquisition"][
            "task_plan_keyed_source_contract_mismatches"], 0)

    def test_question_critic_uses_final_output_not_intermediate_entity(self):
        from environment.generate_question.task_plan import make_spec
        from environment.generate_question.question import naturalize_one
        from environment.generate_question.goal_ops_v5 import (
            selected_profile_output, goal_question as v5_goal_question,
        )

        class Author:
            def __init__(self, text):
                self.text = text
            def complete_json(self, system, request):
                return {"question": self.text}

        class Critic:
            def __init__(self):
                self.seen = None
            def complete_json(self, system, request):
                goal = json.loads(request)["frozen_contract"]
                self.seen = goal
                return {
                    "verdict": "pass", "all_constraints_preserved": True,
                    "no_unlicensed_facts": True,
                    "same_answer_scope": True, "natural": True,
                }

        kg = fixture()
        selections = [
            ("goal_conditioned_operation_dag_v5",
             next(i for i in range(100)
                  if selected_profile_output(17, i)[0] == "temporal_gap"),
             v5_goal_question),
            (VERSION,
             next(i for i in range(100)
                  if "temporal" in _backward_goal(
                      selected_output(17, i))["capabilities"]
                  and selected_output(17, i)["kind"] != "date_time"),
             None),
        ]
        for composer, ordinal, question_function in selections:
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty="medium",
                world_mode="compose", composer=composer,
            )
            from environment.generate_question.goal_ops_v6 import goal_question
            wanted = (question_function or goal_question)(spec["macro_plan"])
            candidate = (
                "Please use the supplied reference card concerning a record "
                "in the " + spec["scenario"]["topic"] + ". " + wanted
            )
            critic = Critic()
            _, info = naturalize_one(
                spec, author=Author(candidate), critic=critic,
                max_revisions=0,
            )
            self.assertEqual(info["status"], "accepted", info)
            self.assertEqual(critic.seen["semantic_goal_question"], wanted)
            self.assertEqual(
                critic.seen["final_output_spec"],
                spec["macro_plan"]["goal_spec"]["output_spec"],
            )
            self.assertTrue(critic.seen["goal_operations"])


if __name__ == "__main__":
    unittest.main()
