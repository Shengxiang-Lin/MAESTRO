"""The deterministic Scoped text is a reference, not an accepted fallback."""
import unittest

from environment.tests.test_stage2_surface_certification import (
    Stage2SurfaceCertificationTests,
)
from environment.question_generation.realization import _realize_one
from environment.question_generation.scoped_surface import (
    build_scoped_plan, read_scoped, read_scoped_semantics, render_scoped,
)
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.eligibility import (
    controlled_question_reference,
)


class ScopedSurfaceTests(unittest.TestCase):
    def fixture(self):
        payload, world, capability, rendered = Stage2SurfaceCertificationTests().data()
        realized = {**payload, "question": rendered.question,
                    "capability_world": capability}
        gold = compile_gold_solution(
            realized, world, rendered, schedule="compact",
        )
        plan = build_scoped_plan(
            world, rendered.question, gold,
            controlled_question_reference(gold, world, rendered),
        )
        return payload, world, capability, rendered, gold, plan

    def test_reference_is_deterministic_and_every_sentence_option_has_same_semantics(self):
        _, _, _, _, _, plan = self.fixture()
        self.assertTrue(plan)
        base = render_scoped(plan, [0] * len(plan))
        self.assertEqual(read_scoped(base, plan), [0] * len(plan))
        for i, options in enumerate(plan):
            for j in range(len(options)):
                choices = [0] * len(plan)
                choices[i] = j
                text = render_scoped(plan, choices)
                parsed = read_scoped_semantics(text, plan)
                self.assertEqual(parsed["query"], plan.query)
        self.assertIsNone(read_scoped(base + " Exclude Alice.", plan))

    def test_scoped_reference_is_never_automatically_published(self):
        payload, world, capability, rendered, _, plan = self.fixture()
        scoped = render_scoped(plan, [0] * len(plan))

        class CopyingAuthor:
            def complete_json(self, system, user, **kwargs):
                import json
                data = json.loads(user)
                if "current_fragment" in data:
                    return {"goal":
                        "Investigate the available public evidence and return "
                        "the result of this assigned information request."
                    }
                return {"question": scoped}

        class NoCritic:
            def complete_json(self, *args, **kwargs):
                raise AssertionError("unchanged reference must not be reviewed")

        ok, rejected, audit = _realize_one(
            payload, world, capability, CopyingAuthor(), NoCritic(),
            max_revisions=0,
        )
        self.assertFalse(ok)
        self.assertEqual(rejected["reason"], "rejected_quality")
        self.assertNotIn("question", rejected)
        self.assertEqual(audit[-1]["failure"], "not_naturalized")

    def test_relation_owner_dates_and_output_are_distinct_text_slots(self):
        def node(i, op, deps=(), args=()):
            return {
                "node_id": i, "op": op, "dependencies": list(deps),
                "inputs": list(args),
            }
        world = {"program": {"nodes": [
            node(0, "FIND_ALL"),
            node(1, "FILTER_DATE", (0,),
                 ("schema:deathDate", "=", "1997-08-01")),
            node(2, "RELATE", (1,), ("schema:deathPlace", "forward")),
            node(3, "FILTER_DATE", (2,),
                 ("schema:endDate", "=", "2004-05-03")),
            node(4, "RELATE", (3,), ("schema:deathPlace", "backward")),
            node(5, "FILTER_TYPE", (4,), ("schema:Person",)),
            node(6, "FIND_ALL"),
            node(7, "FILTER_STRING", (6,),
                 ("schema:telephoneCountryCode", "=", "+650")),
            node(8, "FILTER_NUMBER", (7,),
                 ("schema:malePopulation", "=", "488")),
            node(9, "INTERSECT", (5, 8)),
            node(10, "QUERY_NAME", (9,)),
        ]}}
        canonical = (
            "Person death date 1997-08-01 death place end date 2004-05-03 "
            "telephone country code +650 male population 488"
        )
        plan = build_scoped_plan(
            world, canonical, {"fragments": []},
            {"verified_identity_clues": []},
        )
        question = render_scoped(plan, [0] * len(plan))
        self.assertIn("August 1, 1997", question)
        self.assertIn("May 3, 2004", question)
        self.assertEqual(read_scoped(question, plan), [0] * len(plan))
        for changed in (
            question.replace("August 1, 1997", "August 2, 1997"),
            question.replace("+650", "650"),
            question.replace("belong to both", "belong to either"),
            question.replace("Which people", "How many people"),
        ):
            if changed != question:
                self.assertIsNone(read_scoped(changed, plan))


if __name__ == "__main__":
    unittest.main()
