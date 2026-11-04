"""Regression: raw KQA operator casing must not yield empty compositions."""
from __future__ import annotations

import unittest

from environment.task_construction.deep_dependency.candidate_selection import _program_from_kqa
from environment.task_construction.deep_dependency.composer import _fragment
from environment.task_construction.data_sources.kqa import record_to_ir
from environment.task_construction.quality.hardness import complexity_profile
import random


class CandidateKQANormalizationTests(unittest.TestCase):
    def test_raw_kqa_record_is_normalized_and_abstracted(self):
        raw = {
            "question_type": "single",
            "program": [
                {"function": "Find", "inputs": ["real-world-entity"], "dependencies": []},
                {"function": "Relate", "inputs": ["real relation", "forward"], "dependencies": [0]},
                {"function": "FilterNum", "inputs": ["real attribute", 42, ">"], "dependencies": [1]},
                {"function": "Count", "inputs": [], "dependencies": [2]},
            ],
        }
        ir = _program_from_kqa(raw)
        self.assertIsNotNone(ir)
        self.assertEqual([node.op for node in ir.nodes],
                         ["FIND", "RELATE", "FILTER_NUMBER", "COUNT"])
        self.assertEqual(ir.nodes[0].inputs, ("<ENTITY_OR_TYPE>",))
        self.assertEqual(ir.nodes[1].inputs, ("<REL>", "forward"))
        self.assertEqual(ir.nodes[2].inputs[0], "<ATTR>")
        self.assertNotIn("real-world-entity", ir.canonical_signature())
        self.assertEqual(ir.canonical_signature(),
                         record_to_ir(raw).canonical_signature())

    def test_composer_recognizes_entity_fragment(self):
        record = {"program": [
            {"function": "Find", "inputs": ["X"], "dependencies": []},
            {"function": "Relate", "inputs": ["P", "forward"], "dependencies": [0]},
            {"function": "Relate", "inputs": ["Q", "forward"], "dependencies": [1]},
            {"function": "What", "inputs": [], "dependencies": [2]},
        ]}
        ir = _program_from_kqa(record)
        fragment = _fragment(ir, min_depth=2, max_depth=4, rng=random.Random(17))
        self.assertGreaterEqual(len(fragment), 2)
        self.assertIn("RELATE", [node.op for node in fragment])

    def test_empty_or_invalid_record_is_skipped(self):
        self.assertIsNone(_program_from_kqa({"program": []}))
        self.assertIsNone(_program_from_kqa({"not_a_program": 1}))


if __name__ == "__main__":
    unittest.main()
