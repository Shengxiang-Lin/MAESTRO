"""Numeric Goal witness diversity + versioned exact Question-scope regression.

Seeds are public topic-only metadata from a prior 100-Scene pilot; no
original MMQA answers, user data or media are used.
"""
from __future__ import annotations

import unittest

from environment.generate_question.spec import make_spec
from environment.generate_question.question import (
    reference_question, guard_question, naturalize_one,
)
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.macro_evidence_world import generate_world
from environment.generate_question.numeric_goal_world import (
    NUMERIC_GOAL_WORLD_VERSION,
)


CASES = (
    # ordinal, topic-only MMQA record_id, topic, domain_hint, difficulty, kind
    (2, "0fffaf96fddf92be904cbf61e927dfc3", "Julianne Moore TV", "general", "hard", "comparison"),
    (7, "c618f8bf57c59873def946899c5270a3", "The Office", "general", "hard", "comparison"),
    (9, "92057299c80994757cb90145f1e74c08", "Presidents Cup", "sports", "hard", "comparison"),
    (10, "d5cf090af397faec8d3e5c11ec8f292b", "PNC Park", "culture", "medium", "comparison"),
    (18, "c907e2f5fbffa6707512c00025c9b6b3", "Manuel Fernandes", "sports", "hard", "comparison"),
    (20, "6d5ccd7336ca7ecddb8db7af4da2b81e", "Vampire", "general", "medium", "comparison"),
    (23, "724c3ece94a544bbb23eb1540fa8776c", "Parma", "sports", "hard", "comparison"),
    (27, "2ddf958d3960e71cd1fbf67fcae2d50f", "Warrington", "sports", "hard", "comparison"),
    (0, "529d4345a1c5519e0aed79c5de6809e7", "Pomona, California", "civic", "medium", "calculation"),
    (5, "d746a8409fff8205ebfafa1c1af115fe", "Church of Denmark", "general", "hard", "calculation"),
    (8, "cbdbf87704640711cdf6baec3615ed28", "Brent McGrath's", "general", "hard", "calculation"),
    (12, "5616068b4c47f3bc50ba9acc7d66c6d5", "bass", "general", "hard", "calculation"),
    (13, "a25f8acdef7be75f57c13e3b8b72d7ad", "Paul Hurst", "general", "medium", "calculation"),
    (17, "f881d9fac67ec8cdc8d6c03771aa3d6", "Angel Valdivielso", "general", "medium", "calculation"),
    (22, "c3fd1b0738563468096f0409ff6bf685", "Sean Chiplock", "education", "hard", "calculation"),
    (24, "447ff16b6c76ced6e1081fe0a4a02788", "Essex", "logistics", "hard", "calculation"),
)


def spec_from(row, *, enabled):
    ordinal, record_id, topic, domain, difficulty, _kind = row
    return make_spec(
        seed_record={
            "record_id": record_id,
            "topic": topic,
            "domain_hint": domain,
        },
        ordinal=ordinal, run_seed=17, difficulty=difficulty,
        macro_first_v2_share=100, operator_grammar_share=100,
        numeric_goal_diversity_share=100 if enabled else 0,
        question_semantic_lock_share=100 if enabled else 0,
    )


class NumericGoalWorldTests(unittest.TestCase):
    def test_old_scene_world_remains_exact_and_new_world_replays(self):
        answers = {"comparison": set(), "calculation": set()}
        for row in CASES:
            with self.subTest(ordinal=row[0], kind=row[-1]):
                old = spec_from(row, enabled=False)
                new = spec_from(row, enabled=True)
                self.assertEqual(
                    new["numeric_goal_world_version"],
                    NUMERIC_GOAL_WORLD_VERSION,
                )
                self.assertNotIn("numeric_goal_world_version", old)
                self.assertNotIn("question_semantic_lock_version", old)
                old_world = generate_world(old)
                newer_world = generate_world(new)
                # Only the statistic payload may change; no source schema,
                # candidate identity, source count or observation id may.
                self.assertEqual(
                    set(old_world["sources"]), set(newer_world["sources"])
                )
                for sid in old_world["sources"]:
                    if sid not in {"stat_a", "stat_b"}:
                        self.assertEqual(
                            old_world["sources"][sid],
                            newer_world["sources"][sid],
                        )
                    else:
                        x = old_world["sources"][sid]["observations"]
                        y = newer_world["sources"][sid]["observations"]
                        self.assertEqual(
                            [z["observation_id"] for z in x],
                            [z["observation_id"] for z in y],
                        )
                scene = compile_and_validate(new, reference_question(new))
                self.assertEqual(
                    validate_replay(scene)["status"], "valid"
                )
                value = scene["gold_answer"][0][new["macro_plan"]["goal"]["answer_field"]]
                answers[row[-1]].add(value)
                if row[-1] == "calculation":
                    self.assertTrue(10 < value < 100)
                else:
                    self.assertIn(value, ("first", "second"))
                audit = scene["difficulty_audit"]["macro_first_audit"]
                self.assertTrue(audit["operator_grammar_equivalence"][
                    "all_macro_outputs_equivalent"
                ])
                self.assertTrue(audit["source_counterfactuals_pass"])
        self.assertEqual(answers["comparison"], {"first", "second"})
        self.assertGreaterEqual(len(answers["calculation"]), 5)
        self.assertNotEqual(answers["calculation"], {75})

    def test_old_statistic_target_values_still_frozen(self):
        for row in (CASES[0], CASES[-1]):
            spec = spec_from(row, enabled=False)
            sources = generate_world(spec)["sources"]
            a = sources["stat_a"]["observations"]
            b = sources["stat_b"]["observations"]
            if row[-1] == "comparison":
                self.assertTrue(any(
                    x["values"].get("metric_a") == 90 for x in a
                ))
                self.assertTrue(any(
                    x["values"].get("metric_b") == 25 for x in b
                ))
            else:
                self.assertTrue(any(
                    x["values"].get("completed_count") == 36 for x in a
                ))
                self.assertTrue(any(
                    x["values"].get("total_count") == 48 for x in b
                ))


class QuestionSemanticLockTests(unittest.TestCase):
    def test_calculation_requires_floor_and_whole_percent(self):
        row = next(x for x in CASES if x[-1] == "calculation")
        spec = spec_from(row, enabled=True)
        good = reference_question(spec)
        self.assertFalse(guard_question(good, spec))
        removed = good.replace("(rounded down)", "")
        self.assertIn(
            "question_calculation_round_down_missing",
            guard_question(removed, spec)
        )
        stripped = good.replace("whole percent", "rate")
        self.assertIn(
            "question_calculation_whole_percent_missing",
            guard_question(stripped, spec)
        )
        old = spec_from(row, enabled=False)
        self.assertNotIn(
            "question_calculation_round_down_missing",
            guard_question(removed, old),
        )

    def test_repair_prompt_receives_explicit_required_semantics(self):
        row = next(x for x in CASES if x[-1] == "calculation")
        spec = spec_from(row, enabled=True)
        base = reference_question(spec)
        bad = base.replace("(rounded down)", "")
        # Distinct enough to avoid the reference similarity gate.
        topic = spec["scenario"]["topic"]
        clues = spec["macro_plan"]["clues"]
        good = (
            f"For the item in {topic} identified by a "
            f"{clues['emblem_color']} emblem and {clues['emblem_shape']} "
            "motif, what completion rate in whole percent, rounded "
            "down, follows from the two independently recorded counts?"
        )
        self.assertEqual(guard_question(good, spec), [])

        class Author:
            payloads = []
            def complete_json(self, system, request):
                import json
                p = json.loads(request)
                self.payloads.append(p)
                return {"question": bad if len(self.payloads) == 1 else good}

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass",
                    "all_constraints_preserved": True,
                    "no_unlicensed_facts": True,
                    "same_answer_scope": True,
                    "natural": True,
                }

        author = Author()
        result, audit = naturalize_one(
            spec, author=author, critic=Critic(), max_revisions=1,
        )
        self.assertEqual(audit["status"], "accepted", audit)
        self.assertEqual(result, good)
        self.assertEqual(len(author.payloads), 2)
        requirement = author.payloads[0]["contract"][
            "mandatory_question_semantics"
        ]
        self.assertIn("ROUND DOWN", requirement["mandatory_computation"])
        self.assertTrue(requirement["no_additional_output_fields"])
        self.assertTrue(author.payloads[1]["contract"]["repair_priority"])

    def test_original_rejected_question_errors_are_explicit(self):
        # Real failed V1.1 cases: omitted floor rule, coordinated output,
        # then 'flagged' was replaced with an unlicensed paraphrase.
        numeric = spec_from(
            next(x for x in CASES if x[-1] == "calculation"), enabled=True
        )
        misleading = reference_question(numeric).replace("(rounded down)", "")
        self.assertIn(
            "question_calculation_round_down_missing",
            guard_question(misleading, numeric),
        )


if __name__ == "__main__":
    unittest.main()
