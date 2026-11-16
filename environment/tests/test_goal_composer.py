"""Small, YAGO-independent regression tests for the bounded Goal composer."""
from __future__ import annotations

import unittest

from environment.generate_question.goal_composer import compose_plan, VERSION


class GoalComposerTests(unittest.TestCase):
    def test_many_goals_are_replayably_composed(self):
        signatures = set()
        keyed = parallel = 0
        for ordinal in range(120):
            args = dict(
                seed=17, ordinal=ordinal,
                difficulty="medium" if ordinal % 2 else "hard",
                hops=2 if ordinal % 2 else 3,
                goal_keys=["doctoraladvisor", "fieldofwork"]
                if ordinal % 2 else
                ["birthplace", "containedinplace", "containedinplace"],
            )
            plan = compose_plan(**args)
            self.assertEqual(plan, compose_plan(**args))
            self.assertEqual(plan["composer_policy"], VERSION)
            tasks = plan["macro_tasks"]
            self.assertLessEqual(len(tasks), 9)
            self.assertEqual(tasks[-1]["output_schema"], {"answer": "str"})
            self.assertEqual(tasks[-1]["macro_id"], plan["terminal_macro_id"])
            previous = {}
            for task in tasks:
                self.assertEqual(task["input_contract"], {
                    mid: previous[mid]["output_schema"]
                    for mid in task["depends_on"]
                })
                previous[task["macro_id"]] = task
            discoveries = [
                t for t in tasks if t["operation"] == "match_clues"
            ]
            keyed += bool(any(t["depends_on"] for t in discoveries))
            parallel += len([t for t in discoveries if not t["depends_on"]]) > 1
            signatures.add(tuple(
                (t["operation"], tuple(t["depends_on"]),
                 tuple(t["source_scope"]))
                for t in tasks
            ))
        self.assertGreater(len(signatures), 8)
        self.assertGreater(keyed, 0)
        self.assertGreater(parallel, 0)


    def test_composed_world_executes_with_keyed_candidate_handoffs(self):
        from environment.generate_question.task_plan import make_spec, validate_spec
        from environment.generate_question.question import reference_question
        from environment.generate_question.quality import compile_and_validate, validate_replay
        from environment.implement_subtask.realize import realize_scene
        from environment.macro_v3.yago import Edge, KnowledgeGraph

        kg = KnowledgeGraph(edges=[
            Edge("https://test.org/p", "http://schema.org/birthPlace",
                 "https://test.org/t"),
            Edge("https://test.org/t", "http://schema.org/containedInPlace",
                 "https://test.org/r"),
            Edge("https://test.org/r", "http://schema.org/containedInPlace",
                 "https://test.org/c"),
        ], input_file="goal-composer-fixture.ttl", scanned=3)
        seen_keyed, seen_parallel = False, False
        for ordinal in range(18):
            spec = make_spec(
                kg, seed=17, ordinal=ordinal,
                difficulty="medium" if ordinal % 2 else "hard",
                world_mode="compose", composer=VERSION,
            )
            self.assertEqual(validate_spec(spec), [])
            q = reference_question(spec)
            scene = compile_and_validate(spec, q)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"]["passed"])
            realized = realize_scene(scene, strict=True)
            self.assertEqual(len(realized["macro_tasks"]),
                             scene["macro_graph"]["macro_task_count"])
            tasks = spec["macro_plan"]["macro_tasks"]
            dependent = [
                source_id for task in tasks
                if task["operation"] == "match_clues" and task["depends_on"]
                for source_id in task["source_scope"]
            ]
            for sid in dependent:
                self.assertEqual(scene["world"]["sources"][sid][
                    "access_policy"]["mode"], "keyed_lookup")
                self.assertTrue(any(
                    node["op"] == "lookup_by_key" and node["source_id"] == sid
                    for node in scene["gold_program"]["nodes"]
                ))
            seen_keyed |= bool(dependent)
            seen_parallel |= sum(
                t["operation"] == "match_clues" and not t["depends_on"]
                for t in tasks
            ) > 1
        self.assertTrue(seen_keyed)
        self.assertTrue(seen_parallel)

    def test_invalid_goal_and_difficulty_fail_closed(self):
        with self.assertRaises(ValueError):
            compose_plan(seed=1, ordinal=0, difficulty="easy", hops=2,
                         goal_keys=["birthplace", "location"])
        with self.assertRaises(ValueError):
            compose_plan(seed=1, ordinal=0, difficulty="hard", hops=3,
                         goal_keys=["birthplace", "location"])


if __name__ == "__main__":
    unittest.main()
