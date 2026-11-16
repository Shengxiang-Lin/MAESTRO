"""Typed relational operator grammar and legacy exact-replay regressions."""
from __future__ import annotations

from collections import Counter
import copy
import unittest

from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.macro_planning import (
    compile_macro_solvers, validate_spec,
)
from environment.generate_question.operator_grammar import (
    VERSION, allowed_solver_variants, canonical_spec,
)
from environment.generate_question.program_ir import evaluate_program


def sample(index: int, share: int):
    topics = (
        ("Sports festival", "sports"),
        ("Local cinema program", "film"),
        ("Community event recordings", "general"),
        ("Museum catalog", "culture"),
        ("Experiment register", "science"),
        ("University archives", "education"),
        ("Transport incident records", "logistics"),
    )
    topic, domain = topics[index % len(topics)]
    return make_spec(
        seed_record={
            "record_id": "typed-grammar-" + str(index),
            "topic": topic, "domain_hint": domain,
            "source_modalities": ["image", "audio", "table"],
        },
        ordinal=index, run_seed=31,
        difficulty="hard" if index % 2 else "medium",
        macro_first_v2_share=100,
        operator_grammar_share=share,
    )


class TypedOperatorGrammarTests(unittest.TestCase):
    def test_opt_in_changes_no_legacy_scene_and_preserves_schema(self):
        for index in range(8):
            legacy = sample(index, share=0)
            explicit_disabled = sample(index, share=0)
            typed = sample(index, share=100)
            self.assertEqual(legacy, explicit_disabled)
            self.assertEqual(typed["operator_grammar_version"], VERSION)
            self.assertNotIn("operator_grammar_version", legacy)
            self.assertEqual(canonical_spec(typed), legacy)
            self.assertEqual(validate_spec(typed), [])
            self.assertEqual(validate_spec(legacy), [])

    def test_local_variants_have_typed_bounded_choices(self):
        self.assertEqual(
            len(allowed_solver_variants("video_audio_alignment", {})), 4
        )
        self.assertEqual(
            allowed_solver_variants(
                "visual_candidates", {"visual_modality": "image"}
            ), ("canonical",),
        )
        self.assertEqual(
            len(allowed_solver_variants(
                "visual_candidates", {"visual_modality": "video"}
            )), 2
        )
        self.assertEqual(
            allowed_solver_variants("key_resolution_1", {}),
            ("canonical",),
        )

    def test_variant_assignment_is_stable_and_not_world_rng_dependent(self):
        signature = Counter()
        patterns = Counter()
        noncanonical = 0
        for index in range(40):
            typed = sample(index, share=100)
            old = canonical_spec(typed)
            self.assertEqual(validate_spec(old), [])
            self.assertEqual(typed["macro_plan"]["clues"],
                             old["macro_plan"]["clues"])
            self.assertEqual(typed["macro_plan"]["index_hops"],
                             old["macro_plan"]["index_hops"])
            self.assertEqual(
                [task["evidence_plan"] for task in
                 typed["macro_plan"]["macro_tasks"]],
                [task["evidence_plan"] for task in
                 old["macro_plan"]["macro_tasks"]],
            )
            self.assertEqual(
                typed["macro_plan"],
                sample(index, share=100)["macro_plan"],
            )
            pattern = typed["macro_plan"]["pattern"]
            patterns[pattern] += 1
            for task in typed["macro_plan"]["macro_tasks"]:
                signature[(task["role"], task["solver_variant"])] += 1
                noncanonical += task["solver_variant"] != "canonical"
        self.assertGreaterEqual(len(patterns), 4)
        self.assertGreater(noncanonical, 0)
        self.assertGreaterEqual(len(signature), 20)

    def test_compilation_preserves_exact_macro_results_and_factrefs(self):
        patterns, noncanonical, late_video = set(), 0, False
        for index in range(34):
            typed = sample(index, share=100)
            patterns.add(typed["macro_plan"]["pattern"])
            noncanonical += sum(
                row["solver_variant"] != "canonical"
                for row in typed["macro_plan"]["macro_tasks"]
            )
            late_video |= any(
                row["role"] == "video_audio_alignment"
                and row["solver_variant"] != "canonical"
                for row in typed["macro_plan"]["macro_tasks"]
            )
            scene = compile_and_validate(typed, reference_question(typed))
            self.assertEqual(validate_replay(scene)["status"], "valid")
            audit = scene["difficulty_audit"]["macro_first_audit"][
                "operator_grammar_equivalence"
            ]
            self.assertTrue(audit["all_macro_outputs_equivalent"])
            self.assertTrue(audit["final_answer_and_provenance_equivalent"])
            self.assertEqual(audit["verified_macro_outputs"],
                             scene["macro_graph"]["macro_task_count"])
        self.assertGreaterEqual(len(patterns), 4)
        self.assertGreater(noncanonical, 0)
        self.assertTrue(late_video, "must test alternate filter-join placement")

    def test_corrupt_variant_and_unknown_version_fail_closed(self):
        spec = sample(8, share=100)
        altered = copy.deepcopy(spec)
        altered["macro_plan"]["macro_tasks"][0]["solver_variant"] = "unknown"
        self.assertTrue(validate_spec(altered))
        with self.assertRaises(ValueError):
            compile_macro_solvers(altered)
        altered = copy.deepcopy(spec)
        altered["operator_grammar_version"] = "maestro.unknown.v999"
        self.assertIn("invalid_operator_grammar_version",
                      validate_spec(altered))

    def test_stage2_oracle_and_stage4_trusted_prefix_accept_new_program(self):
        import json
        from environment.implement_subtask.realize import realize_scene
        from environment.untrusted_orchestration.trusted_prefix import (
            compile_trusted_prefixes,
        )
        from environment.untrusted_orchestration.trajectory_builder import (
            build_branch,
        )
        from environment.untrusted_orchestration.faults import Fault

        typed = sample(3, share=100)
        scene = compile_and_validate(typed, reference_question(typed))
        oracle = realize_scene(scene, strict=True)
        self.assertEqual(oracle["macro_task_count"],
                         scene["macro_graph"]["macro_task_count"])
        reports = [{
            **row,
            "finish_result": {
                "status": "done",
                "result": "Observed: " + json.dumps(row["answer"]),
                "summary": "Grounded observations " + json.dumps(row["answer"]),
            },
            "report_authoring": {"status": "accepted"},
        } for row in oracle["macro_tasks"]]
        episodes, reviews = compile_trusted_prefixes(
            [scene], reports, max_rounds=16,
            sub_model="qwen3-8b", model_alias="model_1",
        )
        self.assertEqual(reviews[0]["status"], "accepted", reviews)
        clean = episodes[scene["task_id"]]
        self.assertEqual(clean["trajectory"][-1]["action"], "complete")
        first = scene["macro_graph"]["reference_waves"][0][0]
        _episode, turns, audit = build_branch(
            scene, reports, clean, branch_round=2,
            faults=[Fault(first, "missing")],
        )
        self.assertEqual(audit["outcome"], "recovered")
        self.assertTrue(turns)

    def test_canonical_1_2_4_configs_share_latest_stage1_bank(self):
        from pathlib import Path
        from environment.config import load_config as load_questions_config
        import yaml

        cfg = Path(__file__).resolve().parents[1] / "configs"
        questions = load_questions_config(cfg / "questions_task_plan.yaml")
        reports = yaml.safe_load((cfg / "subagent_reports.yaml").read_text())
        untrusted = yaml.safe_load(
            (cfg / "untrusted_orchestration.yaml").read_text()
        )
        generation = questions["generation"]
        self.assertEqual(generation["knowledge_variant"], "task_plan")
        self.assertEqual(generation["knowledge_world_mode"], "compose")
        self.assertTrue(generation["knowledge_yago"])
        scene_path = generation["output_dir"] + "/constructed/scenes.jsonl"
        self.assertEqual(reports["implementation"]["scenes"], scene_path)
        self.assertEqual(untrusted["input"]["scenes"], scene_path)
        self.assertEqual(
            untrusted["input"]["reports"],
            reports["reporting"]["output_dir"] + "/subtasks.jsonl",
        )

    def test_old_scene_exact_gold_remains_valid(self):
        legacy = sample(2, share=0)
        scene = compile_and_validate(legacy, reference_question(legacy))
        replay = validate_replay(scene)
        self.assertEqual(replay["status"], "valid")
        self.assertNotIn(
            "operator_grammar_equivalence",
            scene["difficulty_audit"]["macro_first_audit"],
        )
        self.assertNotIn("operator_grammar_version", legacy)


if __name__ == "__main__":
    unittest.main()
