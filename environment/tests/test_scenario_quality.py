"""Scenario field repair and topic-quality regression tests."""
from __future__ import annotations

import unittest

from environment.generate_question.authoring.scenario_contracts import repair_scenario_fields
from environment.generate_question.scenario import _narrative_issues


class ScenarioFieldRepairTests(unittest.TestCase):
    def test_long_motivation_repairs_without_changing_frozen_semantics(self):
        spec = {
            "scenario": {"topic": "Premier League"},
            "macro_plan": {"goal": {"public_label": "match result"}},
        }
        original = {
            "scenario": {
                "setting": "Investigation of a match event",
                "user_role": "research analyst",
                "purpose": "Confirm the requested match result",
                "motivation": "Compare records for the final match result. " * 18,
                "entity_kind": "match events",
            },
            "source_titles": [{"source_id": "visual", "title": "Visual annotations"}],
        }
        modified, repairs = repair_scenario_fields(spec, original)
        self.assertGreater(len(original["scenario"]["motivation"]), 250)
        self.assertLessEqual(len(modified["scenario"]["motivation"]), 250)
        self.assertIn("Premier League", modified["scenario"]["setting"])
        self.assertEqual(original["source_titles"], modified["source_titles"])
        self.assertEqual(original["scenario"]["purpose"],
                         modified["scenario"]["purpose"])
        self.assertIn("too_long", {r["reason"] for r in repairs})
        self.assertIn("topic_prefix", {r["reason"] for r in repairs})

    def test_short_and_non_string_fields_have_explicit_reasons(self):
        spec = {"scenario": {"topic": "Some League"},
                "macro_plan": {"goal": {"public_label": "match result"}}}
        draft = {"scenario": {"setting": "ok", "user_role": None,
                              "purpose": "Find the match result",
                              "motivation": "", "entity_kind": 99}}
        repaired, flags = repair_scenario_fields(spec, draft)
        self.assertTrue(all(isinstance(repaired["scenario"][k], str)
                            for k in repaired["scenario"]))
        self.assertTrue(any(x["reason"] == "too_short" for x in flags))
        self.assertTrue(any(x["reason"] == "invalid_type_or_missing"
                            for x in flags))


class TopicQualityTests(unittest.TestCase):
    def test_reject_placeholder_not_real_topic(self):
        from environment.generate_question.authoring.scenario_contracts import topic_seed_issues
        self.assertIn(
            "topic_generated_placeholder",
            topic_seed_issues({
                "topic": "sports reference collection 7b56ebb",
                "scene_topic_policy": "original_mmqa_source_bundle_fallback",
            }),
        )
        self.assertIn(
            "topic_ambiguous_subject",
            topic_seed_issues({"topic": "the man"}),
        )
        self.assertEqual(topic_seed_issues({
            "topic": "Premier League",
            "scene_topic_policy": "original_mmqa_entity_metadata",
        }), [])



if __name__ == '__main__':
    unittest.main()
