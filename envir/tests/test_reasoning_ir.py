"""Regression tests for the isolated, opt-in V2 Reasoning Program pilot."""
from __future__ import annotations

import copy
import unittest

from envir.macro.program_ir import (
    MissingSourceError, ProgramError, answer_values, evaluate_program,
    program_complexity, validate_program,
)
from envir.macro.source_entity_world import build_linked_case, public_case
from envir.macro.difficulty_validation import audit_linked_case


class ReasoningIRTests(unittest.TestCase):
    def test_medium_requires_actual_join_outputs(self):
        case = build_linked_case(seed=17, difficulty="medium")
        self.assertEqual(case["gold_answer"], [
            {"player_label": "Player-01"}, {"player_label": "Player-03"}
        ])
        stats = program_complexity(case["gold_program"])
        self.assertEqual(stats["join_count"], 2)
        self.assertEqual(stats["source_count"], 3)
        self.assertGreaterEqual(stats["reasoning_depth"], 4)

    def test_hard_adds_independent_filter_and_join(self):
        case = build_linked_case(seed=17, difficulty="hard")
        self.assertEqual(case["gold_answer"], [{"player_label": "Player-01"}])
        stats = program_complexity(case["gold_program"])
        self.assertEqual(stats["join_count"], 3)
        self.assertEqual(stats["source_count"], 4)
        self.assertGreater(stats["reasoning_depth"], 4)

    def test_witnesses_propagate_from_all_required_sources(self):
        for difficulty in ("medium", "hard"):
            case = build_linked_case(difficulty=difficulty)
            result = evaluate_program(
                case["gold_program"], sources=case["world"]["sources"]
            )
            self.assertEqual(answer_values(result), case["gold_answer"])
            expected = set(case["world"]["sources"])
            for item in result["answer"]:
                actual = {ref.split(":")[0] for ref in item["source_fact_refs"]}
                self.assertEqual(actual, expected)

    def test_provenance_and_source_schema_integrity(self):
        case = build_linked_case()
        result = evaluate_program(case["gold_program"],
                                  sources=case["world"]["sources"])
        self.assertEqual(result["trace"][-1]["node_id"], "unique_answer")
        self.assertIn("board:F001", result["trace"][-1]["source_fact_refs"])
        broken = copy.deepcopy(case["world"]["sources"])
        broken["ledger"]["observations"][0]["values"].pop("active")
        with self.assertRaises(ProgramError):
            evaluate_program(case["gold_program"], sources=broken)

    def test_missing_source_does_not_fall_back_to_teacher_world(self):
        case = build_linked_case()
        sources = dict(case["world"]["sources"])
        del sources["codebook"]
        with self.assertRaises(MissingSourceError):
            evaluate_program(case["gold_program"], sources=sources)

    def test_counterfactual_sources_really_change_answer(self):
        for difficulty in ("medium", "hard"):
            report = audit_linked_case(
                build_linked_case(difficulty=difficulty)
            )
            self.assertTrue(report["all_sources_counterfactually_relevant"])
            self.assertFalse(report["active_only_baseline_exact"])
            self.assertTrue(all(
                item["missing_source_blocks_this_program"]
                for item in report["source_tests"].values()
            ))

    def test_seed_changes_codes_without_changing_answer(self):
        cases = [build_linked_case(seed=i) for i in range(20)]
        code_orders = {
            tuple(row["values"]["position_code"]
                  for row in case["world"]["sources"]["codebook"]["observations"])
            for case in cases
        }
        self.assertGreater(len(code_orders), 1)
        self.assertTrue(all(case["gold_answer"] == cases[0]["gold_answer"]
                            for case in cases))

    def test_public_view_does_not_leak_program_entities_or_gold(self):
        case = build_linked_case(difficulty="hard")
        public = public_case(case)
        for key in ("gold_program", "world", "gold_answer", "entities"):
            self.assertNotIn(key, public)
        self.assertEqual(
            {x["modality_hint"] for x in public["available_sources"]},
            {"table", "text", "image"},
        )
        self.assertTrue(case["synthetic_only"])

    def test_no_disconnected_dummy_work(self):
        program = copy.deepcopy(build_linked_case()["gold_program"])
        program["nodes"].insert(1, {
            "id": "unused", "op": "filter_eq", "inputs": ["ledger_rows"],
            "field": "active", "value": True,
        })
        self.assertIn("unused_node_not_required_for_output",
                      validate_program(program))

    def test_no_forward_reference_or_bypass(self):
        case = build_linked_case()
        program = copy.deepcopy(case["gold_program"])
        program["nodes"][1]["inputs"] = ["code_rows"]
        self.assertIn("invalid_arity_or_non_topological_dependency",
                      validate_program(program))
        with self.assertRaises(ProgramError):
            evaluate_program(program, sources=case["world"]["sources"])

    def test_join_does_not_overwrite_conflicting_shared_field(self):
        program = {
            "nodes": [
                {"id": "a", "op": "scan", "inputs": [], "source_id": "a"},
                {"id": "b", "op": "scan", "inputs": [], "source_id": "b"},
                {"id": "c", "op": "join", "inputs": ["a", "b"],
                 "left_on": "id", "right_on": "id"},
            ],
            "output": "c",
        }
        sources = {
            "a": {"schema": ["id", "shared"], "observations": [{
                "values": {"id": "x", "shared": 1}, "fact_refs": ["a:F1"]}]},
            "b": {"schema": ["id", "shared"], "observations": [{
                "values": {"id": "x", "shared": 2}, "fact_refs": ["b:F1"]}]},
        }
        self.assertEqual(
            answer_values(evaluate_program(program, sources=sources)), []
        )


if __name__ == "__main__":
    unittest.main()
