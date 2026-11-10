"""Regression tests for Scenario repair and accepted difficulty quotas."""
from __future__ import annotations

import unittest
from envir.generate_question.difficulty_quotas import (
    candidate_budget, select_accepted_scenes,
)
from envir.generate_question.scenario_repair import (
    diagnose_fields, repair_draft,
)
from envir.generate_question.spec import make_spec
from envir.generate_question.scenario import _narrative_issues
from envir.generate_question.source_title_contract import (
    canonicalize_scenario_proposal,
)


def sample():
    return make_spec(
        seed_record={"record_id": "repair-001", "topic": "Premier League",
                     "domain_hint": "sports"},
        ordinal=0, run_seed=17, difficulty="hard",
        macro_first_v2_share=100,
    )


class QuotaAndRepairTests(unittest.TestCase):
    def test_bounded_candidate_pool_and_deterministic_accepted_selection(self):
        self.assertEqual(candidate_budget(
            100, 1000, enabled=True, multiplier=2), 200)
        self.assertEqual(candidate_budget(
            100, 120, enabled=True, multiplier=2), 120)
        self.assertEqual(candidate_budget(
            100, 1000, enabled=False, multiplier=2), 100)
        with self.assertRaises(ValueError):
            candidate_budget(100, 200, enabled=True, multiplier=5)
        reviewed = [
            {"difficulty_target": level, "id": i}
            for i, level in enumerate(
                ["hard"] * 6 + ["easy"] * 3 + ["medium"] * 4
            )
        ]
        chosen, deficits = select_accepted_scenes(
            reviewed, {"easy": 2, "medium": 3, "hard": 5}
        )
        self.assertEqual(
            [sum(x["difficulty_target"] == level for x in chosen)
             for level in ("easy", "medium", "hard")],
            [2, 3, 5],
        )
        self.assertEqual(deficits, {"easy": 0, "medium": 0, "hard": 0})
        partial, missing = select_accepted_scenes(
            reviewed[:8], {"easy": 2, "medium": 3, "hard": 5}
        )
        self.assertEqual(missing, {"easy": 0, "medium": 3, "hard": 0})
        self.assertEqual(len(partial), 7)

    def test_motivation_length_and_missing_topic_are_repaired_before_critic(self):
        spec = sample()
        draft = {
            "scenario": {
                "setting": "A review of working competition records",
                "user_role": "analyst",
                "purpose": "resolve the requested output",
                "motivation": ("Review records carefully. " * 18).strip(),
                "entity_kind": "matches",
            },
        }
        draft = canonicalize_scenario_proposal(spec, draft)
        problems = diagnose_fields(spec, draft)
        self.assertIn("too_long", problems["motivation"])
        self.assertIn("topic_missing", problems["setting"])
        repaired, audit = repair_draft(spec, draft)
        self.assertEqual(
            set(audit["changes"]), {"setting", "motivation"},
        )
        self.assertIn("Premier League", repaired["scenario"]["setting"])
        self.assertLessEqual(len(repaired["scenario"]["motivation"]), 250)
        self.assertEqual(_narrative_issues(spec, repaired), [])
        self.assertEqual(draft["scenario"]["setting"],
                         "A review of working competition records")

    def test_nonstrings_are_repaired_without_changing_goal_or_sources(self):
        spec = sample()
        draft = canonicalize_scenario_proposal(spec, {
            "scenario": {
                "setting": "Premier League: review working evidence",
                "user_role": None,
                "purpose": "resolve the requested output",
                "motivation": "Use WorkProduct to verify the findings.",
                "entity_kind": "matches",
            },
        })
        corrected, audit = repair_draft(spec, draft)
        self.assertEqual(
            corrected["scenario"]["user_role"],
            spec["scenario"]["user_role"],
        )
        self.assertNotIn("WorkProduct", corrected["scenario"]["motivation"])
        self.assertEqual(_narrative_issues(spec, corrected), [])
        self.assertIn("implementation_term", audit["changes"]["motivation"])
        self.assertEqual(
            corrected["source_titles"], draft["source_titles"]
        )


if __name__ == "__main__":
    unittest.main()
