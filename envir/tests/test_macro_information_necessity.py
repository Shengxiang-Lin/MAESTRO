"""Macro information necessity, Goal output fidelity and 1:2:3 mix."""
from __future__ import annotations

import copy
import json
import unittest
from collections import Counter

from envir.generate_question.spec import make_spec, difficulty_sequence
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.information_necessity import (
    POLICY, VARIABLE_POLICY, audit_candidate_information,
)
from envir.generate_question.pipeline import quality_report


def spec_for(i, *, difficulty="hard", topic="Premier League",
             hint="sports"):
    return make_spec(
        seed_record={
            "record_id": f"info-needed-{i}", "topic": topic,
            "domain_hint": hint, "source_modalities": ["image", "table"],
        }, ordinal=i, run_seed=61, difficulty=difficulty,
        macro_first_v2_share=100,
    )


def sample_of(pattern):
    for i in range(300):
        spec = spec_for(i)
        if spec["macro_plan"]["pattern"] == pattern:
            return spec
    raise AssertionError("missing pattern")


class InformationNecessityTests(unittest.TestCase):
    def test_default_difficulty_ratio_and_small_targets(self):
        from pathlib import Path
        import yaml
        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
        )
        mix = tuple(config["generation"]["difficulty_mix"])
        self.assertEqual(mix, (17, 33, 50))
        seq = difficulty_sequence(target=100, mix=mix, run_seed=17)
        self.assertEqual(Counter(seq), {
            "easy": 17, "medium": 33, "hard": 50,
        })
        self.assertEqual(seq, difficulty_sequence(
            target=100, mix=mix, run_seed=17
        ))
        small = difficulty_sequence(target=6, mix=mix, run_seed=17)
        self.assertEqual(Counter(small), {
            "easy": 1, "medium": 2, "hard": 3,
        })
        self.assertEqual(len(difficulty_sequence(
            target=101, mix=mix, run_seed=17
        )), 101)

    def test_joint_multimodal_branches_are_both_informative(self):
        checked = []
        scenes = []
        for i in range(100):
            spec = spec_for(i, difficulty="medium" if i % 2 else "hard")
            scene = compile_and_validate(spec, reference_question(spec))
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertTrue(spec["macro_information_necessity_version"])
            if spec["macro_plan"]["pattern"] != "multimodal_event":
                continue
            proof = audit_candidate_information(scene)
            self.assertEqual(spec["macro_plan"]["information_necessity_policy"],
                             VARIABLE_POLICY)
            profile = spec["macro_plan"]["candidate_profile"]
            self.assertEqual(proof["left_candidate_count"],
                             profile["left_candidate_count"])
            self.assertEqual(proof["right_candidate_count"],
                             profile["right_candidate_count"])
            self.assertTrue(proof["candidate_count_matches_profile"])
            self.assertGreaterEqual(proof["left_candidate_count"], 2)
            self.assertGreaterEqual(proof["right_candidate_count"], 2)
            self.assertEqual(proof["intersection_count"], 1)
            self.assertTrue(proof["passed"])
            self.assertEqual(scene["difficulty_audit"][
                "macro_first_audit"]["candidate_information_necessity"], proof)
            checked.append(proof)
            scenes.append(scene)
        self.assertGreaterEqual(len(checked), 10)
        report = quality_report(scenes, no_progress=True)
        self.assertEqual(report["macro_first_v2"]["information_necessity"][
            "candidate_discrimination_passed"], len(scenes))

    def test_old_semantic_v2_scene_replays_without_changed_world(self):
        spec = sample_of("multimodal_event")
        old = copy.deepcopy(spec)
        from envir.generate_question.macro_first_v2 import sample_macro_plan
        old.pop("scene_realism_version")
        old.pop("goal_resolution_version", None)
        plan, roles = sample_macro_plan(
            seed=old["seed"], ordinal=old["ordinal"],
            record_id=old["source_anchor"]["source_record_id"],
            topic=old["scenario"]["topic"], difficulty=old["difficulty_target"],
            domain="sports", information_need=True, realism_v3=False,
        )
        old["macro_plan"] = plan
        old["sources"] = roles
        # Reconstruct a PR106-style frozen V2 scene without the new marker.
        scene = compile_and_validate(old, reference_question(old))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        old_audit = scene["difficulty_audit"]["macro_first_audit"][
            "candidate_information_necessity"
        ]
        self.assertEqual(old_audit["policy"], POLICY)
        self.assertEqual(old_audit["left_candidate_count"], 2)
        self.assertEqual(old_audit["right_candidate_count"], 2)

    def test_context_cannot_be_redundant_in_new_multimodal(self):
        spec = sample_of("multimodal_event")
        scene = compile_and_validate(spec, reference_question(spec))
        new = copy.deepcopy(scene)
        # Recode the actual context source to include both visual candidates
        # in the matching tag set. The world will fail strict regeneration,
        # and the local information-necessity audit must flag the shortcut.
        values = new["world"]["sources"]["context"]["observations"]
        first = {row["values"]["candidate_id"]
                 for row in next(x for x in scene["task_feedback"]
                                 if x["task_id"] == "work_M1")["workproduct"]}
        for row in values:
            if row["values"]["candidate_id"] in first:
                row["values"]["case_tag"] = spec["macro_plan"]["clues"]["case_tag"]
        # Reject either by witness constraint or by exact provenance replay;
        # there must be no accepted redundant branch.
        result = validate_replay(new)
        self.assertEqual(result["status"], "invalid")

    def test_single_goal_output_and_extra_output_detection(self):
        target = None
        for i in range(250):
            candidate = spec_for(i, topic="The Office", hint="culture")
            if candidate["macro_plan"]["goal"]["answer_field"] == "production_year":
                target = candidate
                break
        self.assertIsNotNone(target)
        q = reference_question(target)
        self.assertEqual(guard_question(q, target), [])
        bad = q.replace(
            "What is", "Which episode is involved, and what is"
        ).replace(
            "which confirmed production year", "what confirmed production year"
        )
        # Explicit second output should be rejected even if the correct
        # requested year is still present in the question.
        bad = "Which episode appears in The Office investigation, and what is " + (
            "the confirmed production year for its identified case?"
        )
        self.assertTrue(
            {"semantic_additional_output_field_requested",
             "semantic_additional_output_requested"}
            & set(guard_question(bad, target))
        )
        contract = compile_and_validate(target, q)["semantic_evidence_contract"]
        self.assertEqual(contract["required_output_fields"],
                         ["production_year"])
        self.assertEqual(contract["semantic_goal_type"], "int")
        self.assertEqual(contract["requested_entities"], [])

    def test_domain_metadata_only_changes_new_versioned_scenes(self):
        from envir.generate_question.macro_first_v2 import (
            domain_hint_v2, domain_hint_v3,
        )
        self.assertEqual(domain_hint_v2("Swansea City", "sports"), "sports")
        self.assertEqual(domain_hint_v2("Unidentified subject", "culture"),
                         "general")
        self.assertEqual(domain_hint_v3("Unidentified subject", "culture"),
                         "culture")
        self.assertEqual(domain_hint_v3("Unidentified subject", "environment"),
                         "environment")
        spec = spec_for(13, topic="Unknown civic event", hint="civic")
        self.assertEqual(spec["macro_plan"]["domain"], "civic")
        self.assertIn(spec["macro_plan"]["goal"]["answer_field"],
                      {"civic_site", "permit_status", "inspection_year"})


if __name__ == "__main__":
    unittest.main()
