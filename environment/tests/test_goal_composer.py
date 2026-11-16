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

    def test_invalid_goal_and_difficulty_fail_closed(self):
        with self.assertRaises(ValueError):
            compose_plan(seed=1, ordinal=0, difficulty="easy", hops=2,
                         goal_keys=["birthplace", "location"])
        with self.assertRaises(ValueError):
            compose_plan(seed=1, ordinal=0, difficulty="hard", hops=3,
                         goal_keys=["birthplace", "location"])


if __name__ == "__main__":
    unittest.main()
