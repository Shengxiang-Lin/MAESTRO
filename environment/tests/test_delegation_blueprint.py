"""No-network shared intent: Stage 2 → Subtask Bank → Main wording tests."""
from __future__ import annotations

import copy
import json
import unittest

from environment.delegation_blueprint import (
    VERSION, compile_delegation_outline, goal_for_fragment,
    validate_delegation_blueprint,
)
from environment.tests.test_stage2_surface_certification import (
    Stage2SurfaceCertificationTests,
)
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.backend import PassthroughExecutor
from environment.trajectory_simulation.subtask_bank import build_subtask_bank
from environment.trajectory_simulation.natural_language import main_single_task_prompt


class DelegationBlueprintTests(unittest.TestCase):
    def prepared(self):
        (ok, row, audit), payload, world, rendered, author, critic = (
            Stage2SurfaceCertificationTests().run_one()
        )
        self.assertTrue(ok, str((row, audit)))
        return row, payload, world, rendered, author

    def test_compact_fragment_identity_waves_and_answer_boundary(self):
        row, payload, world, rendered, author = self.prepared()
        blueprint = row["delegation_blueprint"]
        outline = blueprint["outline"]
        gold = compile_gold_solution(
            row, world, rendered, schedule="compact",
        )
        self.assertEqual(blueprint["version"], VERSION)
        self.assertEqual(
            outline["waves"], gold["waves"],
        )
        self.assertEqual(
            {f["fragment_id"] for f in outline["tasks"]},
            {f["fragment_id"] for f in gold["fragments"]},
        )
        self.assertEqual(
            {f["fragment_id"]: f["depends_on"] for f in outline["tasks"]},
            {f["fragment_id"]: f["depends_on"] for f in gold["fragments"]},
        )
        serialized = json.dumps(blueprint).lower()
        for hidden in (
            "expected_answer", "operation_results", "support_resource_ids",
            "symbolic_trace", "source_claims", "fact_ids", "raw_answer",
        ):
            self.assertNotIn(hidden, serialized)
        self.assertTrue(
            validate_delegation_blueprint(
                blueprint, outline, payload=row, oracle_answer=rendered.answer,
            )
        )
        # Stage 2A generated and reviewed the Goals independently.
        # Stage 2B only requests the final user Question.
        self.assertEqual(len(author.calls), 1)
        self.assertIn("reference_question", author.calls[0])
        self.assertNotIn("current_fragment", author.calls[0])
        self.assertIn("goal_certificate", row)
        self.assertEqual(len(row["goal_reviews"]), len(outline["tasks"]))

    def test_fragment_tampering_and_gold_answer_leak_are_rejected(self):
        row, payload, world, rendered, author = self.prepared()
        blueprint = row["delegation_blueprint"]
        tampered = copy.deepcopy(blueprint)
        tampered["outline"]["tasks"][0]["depends_on"].append("future")
        with self.assertRaisesRegex(ValueError, "outline_mismatch"):
            validate_delegation_blueprint(tampered, blueprint["outline"])
        leaked = copy.deepcopy(blueprint)
        leaked["goals"][0]["goal"] = (
            "Return the final answer " + str(rendered.answer[0]) + "."
        )
        with self.assertRaisesRegex(ValueError, "blueprint_goal_boundary"):
            validate_delegation_blueprint(
                leaked, blueprint["outline"], payload=row,
                oracle_answer=rendered.answer,
            )

    def test_bank_carries_same_blueprint_without_adding_to_public_state(self):
        row, payload, world, rendered, author = self.prepared()
        bank = build_subtask_bank(
            row, world, rendered,
            executor=PassthroughExecutor(), schedule="compact",
        )
        self.assertEqual(
            bank["delegation_blueprint"], row["delegation_blueprint"],
        )
        self.assertEqual(
            bank["ready_order"], row["delegation_blueprint"]["outline"]["waves"],
        )
        for wave in bank["waves"]:
            self.assertNotIn("delegation_blueprint", wave["state"])
            self.assertNotIn("final_answer", wave["state"])

    def test_main_prompt_receives_only_current_fragment_goal(self):
        row, payload, world, rendered, author = self.prepared()
        goals = row["delegation_blueprint"]["goals"]
        first = goals[0]
        other = goals[1] if len(goals) > 1 else None
        canonical = {
            "action": "delegate_task", "params": {
                "tasks": [{
                    "task_instruction": "Identify the relevant entities.",
                    "tools": ["GoogleSearchAction"],
                    "context": "Only public context.",
                }],
            },
        }
        guidance = {"ready": [{"fragment_id": first["fragment_id"]}]}
        prompt = main_single_task_prompt({
            "public_state": {"question": row["question"]},
            "gold_guidance": guidance,
            "delegation_blueprint": row["delegation_blueprint"],
        }, canonical, 0)
        self.assertIn(first["goal"], prompt)
        if other is not None and other["goal"] != first["goal"]:
            self.assertNotIn(other["goal"], prompt)
        self.assertEqual(
            goal_for_fragment(row["delegation_blueprint"], first["fragment_id"]),
            first["goal"],
        )


if __name__ == "__main__":
    unittest.main()
