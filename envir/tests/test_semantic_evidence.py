"""Regression tests for V2 Semantic Evidence Contracts.

The cases reflect previously accepted but unsupported question requirements:
clearance filters, phantom audio, fabricated visual emblems, exact elapsed
time, reversed video/audio order and substituted output goals.
"""
from __future__ import annotations

import copy
import unittest

from envir.generate_question.spec import make_spec, spec_issues
from envir.generate_question.question import guard_question, reference_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.semantic_evidence import (
    CONTRACT_VERSION, compile_semantic_contract, semantic_text_issues,
    semantic_source_title_issues,
)
from envir.generate_question.intent import intent_contract
from envir.generate_question.scenario import _narrative_issues


def make(i: int, *, difficulty="hard"):
    return make_spec(
        seed_record={
            "record_id": f"semantic-{i}",
            "topic": "Premier League" if i % 2 else "The Office",
            "domain_hint": "sports" if i % 2 else "film",
            "source_modalities": ["image", "video", "audio", "text"],
        },
        ordinal=i, run_seed=37, difficulty=difficulty,
        macro_first_v2_share=100,
    )


def specimen(pattern: str, *, medium=False, image=None, audio=None):
    for i in range(400):
        spec = make(i, difficulty="medium" if medium else "hard")
        plan = spec["macro_plan"]
        if plan["pattern"] != pattern:
            continue
        if image is not None and (
            plan["clues"]["visual_modality"] == "image"
        ) is not image:
            continue
        if audio is not None and (
            plan["clues"]["caption_modality"] == "audio"
        ) is not audio:
            continue
        return spec
    raise AssertionError(f"No matching seeded case for {pattern}")


class SemanticEvidenceTests(unittest.TestCase):
    def test_new_specs_have_frozen_program_grounded_contract(self):
        for i in range(100):
            spec = make(i, difficulty="hard" if i % 2 else "medium")
            self.assertEqual(spec_issues(spec), [])
            self.assertEqual(spec["semantic_evidence_contract_version"],
                             CONTRACT_VERSION)
            contract = compile_semantic_contract(spec)
            self.assertTrue(contract["applied_predicates"])
            self.assertEqual(
                sorted(contract["sources"]),
                sorted(source["source_id"] for source in spec["sources"])
            )
            reference = reference_question(spec)
            self.assertEqual(guard_question(reference, spec), [])
            scene = compile_and_validate(spec, reference)
            self.assertEqual(scene["semantic_evidence_contract"], contract)
            self.assertEqual(validate_replay(scene)["status"], "valid")

    def test_hard_v2_intent_does_not_add_clearance(self):
        spec = specimen("disputed_reports")
        intent = intent_contract(spec)
        self.assertNotIn("clearance", str(intent).lower())
        self.assertNotIn("cleared", str(intent).lower())
        contract = compile_semantic_contract(spec)
        self.assertIn("clearance", contract["prohibited_new_filters"])

    def test_clearance_added_to_question_and_source_title_is_rejected(self):
        spec = specimen("disputed_reports")
        q = reference_question(spec)
        q = q.replace("?", ", provided the entry-specific clearance is marked cleared?")
        self.assertIn("semantic_unlicensed_clearance", guard_question(q, spec))
        contract = compile_semantic_contract(spec)
        terminal_source = spec["macro_plan"].get(
            "goal_resolution", {}
        ).get("input_source_ids", ["archive"])[0]
        self.assertIn("semantic_unlicensed_clearance",
                      semantic_source_title_issues(
                          terminal_source, "Final Clearance and Review Register",
                          spec, contract))

        original = spec["scenario"]
        proposal = {
            "scenario": {k: original[k] for k in (
                "setting", "user_role", "purpose", "motivation", "entity_kind"
            )},
            "source_titles": [
                {"source_id": source["source_id"],
                 "title": ("Final Clearance and Review Register"
                           if source["source_id"] == terminal_source
                           else source["title"])}
                for source in spec["sources"]
            ],
        }
        self.assertIn("semantic_unlicensed_clearance",
                      _narrative_issues(spec, proposal))

    def test_nonexistent_audio_and_time_in_image_only_scene(self):
        spec = specimen("serial_identity", image=True)
        q = reference_question(spec)
        q = q.replace("?", " with an audio cue occurring within one minute?")
        failures = guard_question(q, spec)
        self.assertIn("semantic_unsupported_audio", failures)
        self.assertIn("semantic_unsupported_numeric_time", failures)

    def test_video_emblem_and_exact_time_interval_cannot_be_added(self):
        spec = specimen("multimodal_event")
        q = reference_question(spec)
        bad_visual = q.replace("?", " with a red star emblem in the footage?")
        problems = guard_question(bad_visual, spec)
        self.assertIn("semantic_unapplied_color_clue", problems)
        self.assertIn("semantic_unobserved_visual_emblem", problems)
        fabricated = q.replace(
            "?", " with the spoken cue exactly 45 seconds after the action?"
        )
        problems = guard_question(fabricated, spec)
        self.assertIn("semantic_unsupported_numeric_time", problems)
        self.assertIn("semantic_unsupported_time_interval", problems)
        reversed_order = q.replace(
            "?", " with the audio cue occurring before the video action?"
        )
        self.assertIn("semantic_reversed_multimodal_order",
                      guard_question(reversed_order, spec))

    def test_parallel_video_and_audio_cannot_invent_time_alignment(self):
        spec = specimen("parallel_cues", image=False, audio=True)
        q = reference_question(spec)
        wrong = q.replace(
            "?", " where the video action happens after the spoken audio cue?"
        )
        self.assertIn("semantic_unlicensed_crossmodal_order",
                      guard_question(wrong, spec))

    def test_unlicensed_goal_substitution_in_scenario_and_question(self):
        spec = specimen("parallel_cues", image=True)
        q = reference_question(spec).replace(
            "?", " and also return the founding year?"
        )
        self.assertIn("semantic_goal_event_substitution", guard_question(q, spec))
        proposal = {
            "scenario": {
                "setting": f"An investigation of {spec['scenario']['topic']} records",
                "user_role": "researcher",
                "purpose": "Confirm the founding year from working evidence",
                "motivation": "Resolve conflicting reference signals",
                "entity_kind": "records",
            },
            "source_titles": [
                {"source_id": source["source_id"], "title": source["title"]}
                for source in spec["sources"]
            ],
        }
        self.assertIn("semantic_goal_event_substitution",
                      _narrative_issues(spec, proposal))

    def test_llm_questions_are_checked_in_strict_replay(self):
        spec = specimen("disputed_reports")
        scene = compile_and_validate(spec, reference_question(spec))
        fake = copy.deepcopy(scene)
        fake["question"] = scene["question"].replace(
            "?", ", only if the separate clearance table marks it cleared?"
        )
        fake["public"]["question"] = fake["question"]
        fake["question_origin"] = "llm_accepted"
        result = validate_replay(fake)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("semantic_unlicensed_clearance", result["issues"])

        fake = copy.deepcopy(scene)
        before = fake["semantic_evidence_contract"]["goal"]["answer_type"]
        fake["semantic_evidence_contract"]["goal"]["answer_type"] = (
            "int" if before == "str" else "str"
        )
        self.assertIn("semantic_evidence_contract_not_frozen",
                      validate_replay(fake)["issues"])

    def test_domain_hint_metadata_changes_only_versioned_goals(self):
        from envir.generate_question.macro_planning import (
            domain_hint, domain_hint_v2,
        )
        self.assertEqual(domain_hint("Sports reference collection", "sports"), "general")
        self.assertEqual(domain_hint_v2("Sports reference collection", "sports"), "sports")
        self.assertEqual(domain_hint_v2("Generic company contract", "commerce"), "business")
        self.assertEqual(domain_hint_v2("Unidentified cultural event", "culture"), "general")

    def test_retrospective_quality_diagnostics_do_not_mutate_old_scene(self):
        from envir.generate_question.pipeline import quality_report
        spec = specimen("disputed_reports")
        spec.pop("semantic_evidence_contract_version")
        spec.pop("macro_information_necessity_version", None)
        spec.pop("scene_realism_version", None)
        spec.pop("goal_resolution_version", None)
        spec.pop("evidence_tool_contract_version", None)
        from envir.generate_question.macro_planning import (
            sample_macro_plan, domain_hint,
        )
        old_plan, old_roles = sample_macro_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            record_id=spec["source_anchor"]["source_record_id"],
            topic=spec["scenario"]["topic"],
            difficulty=spec["difficulty_target"],
            domain=domain_hint(
                spec["scenario"]["topic"],
                spec["source_anchor"]["domain_hint"],
            ), information_need=False, realism_v3=False,
        )
        spec["macro_plan"] = old_plan
        spec["sources"] = old_roles
        scene = compile_and_validate(spec, reference_question(spec))
        scene = copy.deepcopy(scene)
        scene["question"] = scene["question"].replace(
            "?", ", provided a separate clearance table marks the case cleared?"
        )
        scene["public"]["question"] = scene["question"]
        # Older V2 scenes were previously accepted through the model Critic;
        # the new retrospective report must flag them without retroactively
        # invalidating their frozen Gold World or replay path.
        scene["question_origin"] = "llm_accepted"
        metrics = quality_report([scene], no_progress=True)
        self.assertEqual(metrics["replay_pass"], 1)
        audit = metrics["semantic_evidence_diagnostics"]
        self.assertEqual(audit["v2_scenes_checked"], 1)
        self.assertEqual(audit["versioned_contract_enforced_scenes"], 0)
        self.assertEqual(audit["scenes_with_flags"], 1)
        self.assertIn("semantic_unlicensed_clearance", audit["flag_counts"])

    def test_older_v2_scenes_remain_replay_compatible(self):
        spec = specimen("disputed_reports")
        spec.pop("semantic_evidence_contract_version")
        spec.pop("macro_information_necessity_version", None)
        spec.pop("scene_realism_version", None)
        spec.pop("goal_resolution_version", None)
        spec.pop("evidence_tool_contract_version", None)
        from envir.generate_question.macro_planning import (
            sample_macro_plan, domain_hint,
        )
        old_plan, old_roles = sample_macro_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            record_id=spec["source_anchor"]["source_record_id"],
            topic=spec["scenario"]["topic"],
            difficulty=spec["difficulty_target"],
            domain=domain_hint(
                spec["scenario"]["topic"],
                spec["source_anchor"]["domain_hint"],
            ), information_need=False, realism_v3=False,
        )
        spec["macro_plan"] = old_plan
        spec["sources"] = old_roles
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertNotIn("semantic_evidence_contract", scene)
        self.assertEqual(validate_replay(scene)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
