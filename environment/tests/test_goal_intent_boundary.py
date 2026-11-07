"""Intent summary cannot replace precise Stage2B/Stage3 semantic contracts."""
from __future__ import annotations

import json
import unittest

from environment.delegation_blueprint import (
    _GOAL_SYSTEM, naturalize_fragment_goal,
)
from environment.trajectory_simulation.natural_language import (
    main_single_task_prompt,
)
from environment.tests.test_fragment_goal_isolation import _outline


class IntentAuthor:
    def __init__(self):
        self.packets = []

    def complete_json(self, system, user, **kwargs):
        self.packets.append(json.loads(user))
        return {
            "goal": (
                "Identify the relevant public entities meeting the specified "
                "population criteria and return the matching entity set."
            )
        }


class IntentBoundaryTests(unittest.TestCase):
    def test_short_goal_is_allowed_to_omit_local_thresholds(self):
        outline = _outline()
        author = IntentAuthor()
        fragment = outline["tasks"][0]
        goal = naturalize_fragment_goal(
            author, outline, fragment,
            "Which entities meet both conditions?",
        )["goal"]
        self.assertNotIn("10", goal)
        self.assertIn("population criteria", goal)
        self.assertIn("omit numeric thresholds", _GOAL_SYSTEM)
        self.assertEqual(
            author.packets[0]["current_fragment"]["public_steps"],
            fragment["public_steps"],
        )

    def test_stage3_still_supplies_full_checks_separately_from_intent(self):
        short_goal = (
            "Find events linked to individuals meeting the specified "
            "biographical and physical criteria."
        )
        exact_instruction = (
            "Find persons born before 1840, with weight below 94.9 kg, "
            "and return the events they won."
        )
        payload = {
            "public_state": {"question": "Which events meet all criteria?"},
            "delegation_blueprint": {
                "goals": [{
                    "fragment_id": "fragment:0", "goal": short_goal,
                }],
            },
            "gold_guidance": {"ready": [{"fragment_id": "fragment:0"}]},
            "completed_subtask_contracts": [],
            "visible_resources": [],
        }
        canonical = {
            "action": "delegate_task",
            "params": {"tasks": [{
                "task_instruction": exact_instruction,
                "tools": ["ExtractUrlContentAction"],
                "context": "Public scenario context",
            }]},
        }
        prompt = main_single_task_prompt(payload, canonical, 0)
        self.assertIn(short_goal, prompt)
        self.assertIn(exact_instruction, prompt)
        self.assertIn("preplanned_goal", prompt)
        self.assertIn("required_public_checks", prompt)
        self.assertIn("intent hint", prompt)
        self.assertIn("Preserve EVERY necessary hop", prompt)
        self.assertIn("94.9 kg", prompt)


if __name__ == "__main__":
    unittest.main()
