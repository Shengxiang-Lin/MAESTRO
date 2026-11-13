"""Counterfactual statistic joins use explicit shared source key contracts.

The two regression records with index_hops=1/2 come from the accepted
MMQA topic-only Stage-1 seed bank; neither depends on MMQA media files.
Historically both failed during counterfactual mutation with KeyError:
'metric_b' when the first key in metric_b_lookup.output_schema was
mistaken for a join key.
"""
from __future__ import annotations

import unittest

from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.macro_evidence_world import mutate_witness


CASE_SEEDS = (
    # hop, ordinal, MMQA record id, topic, domain hint, difficulty
    (0, 9, "92057299c80994757cb90145f1e74c08",
     "Presidents Cup", "sports", "hard"),
    (1, 10, "d5cf090af397faec8d3e5c11ec8f292b",
     "PNC Park", "culture", "medium"),
    (2, 2, "0fffaf96fddf92be904cbf61e927dfc3",
     "Julianne Moore TV", "general", "hard"),
)


class ComparisonCounterfactualKeyTests(unittest.TestCase):
    def test_comparison_replay_with_zero_one_two_index_hops(self):
        for hops, ordinal, record_id, topic, domain, difficulty in CASE_SEEDS:
            for share in (0, 100):
                with self.subTest(hops=hops, operator_grammar_share=share):
                    spec = make_spec(
                        seed_record={
                            "record_id": record_id, "topic": topic,
                            "domain_hint": domain,
                            "source_modalities": ["image", "text", "table"],
                        },
                        ordinal=ordinal, run_seed=17,
                        difficulty=difficulty,
                        macro_first_v2_share=100,
                        operator_grammar_share=share,
                    )
                    self.assertEqual(
                        spec["macro_plan"]["goal_resolution"]["kind"],
                        "comparison",
                    )
                    self.assertEqual(spec["macro_plan"]["index_hops"], hops)
                    scene = compile_and_validate(spec, reference_question(spec))
                    self.assertEqual(validate_replay(scene)["status"], "valid")
                    self.assertEqual(
                        set(scene["gold_answer"][0]), {"higher_count_source"},
                    )
                    if share:
                        audit = scene["difficulty_audit"]["macro_first_audit"][
                            "operator_grammar_equivalence"
                        ]
                        self.assertTrue(audit["all_macro_outputs_equivalent"])
                        self.assertTrue(
                            audit["final_answer_and_provenance_equivalent"]
                        )

    def test_statistic_mutation_changes_answer_without_metric_key_lookup(self):
        _hops, ordinal, record_id, topic, domain, difficulty = CASE_SEEDS[-1]
        spec = make_spec(
            seed_record={
                "record_id": record_id, "topic": topic,
                "domain_hint": domain,
            },
            ordinal=ordinal, run_seed=17,
            difficulty=difficulty, macro_first_v2_share=100,
            operator_grammar_share=100,
        )
        scene = compile_and_validate(spec, reference_question(spec))
        plan = spec["macro_plan"]
        self.assertEqual(
            set(plan["source_schemas"]["stat_a"]) &
            set(plan["source_schemas"]["stat_b"]),
            {"record_key_2"},
        )
        # The real output_schema can list metric_b before the actual join key.
        output = plan["macro_tasks"][-2]["output_schema"]
        self.assertEqual(next(iter(sorted(output))), "metric_b")
        for source in ("stat_a", "stat_b"):
            mutated = mutate_witness(
                scene["world"]["sources"], source,
                scene["difficulty_audit"]["answer_provenance"], spec,
            )
            self.assertNotEqual(
                mutated[source], scene["world"]["sources"][source],
            )


if __name__ == "__main__":
    unittest.main()
