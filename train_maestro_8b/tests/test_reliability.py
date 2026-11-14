"""Tests for window boundaries and conservative Stage-4 risk-label export."""
import unittest

from train_maestro_8b.data.contracts import VERSION, validate
from train_maestro_8b.data.contracts import windows, span_to_tokens
from train_maestro_8b.data.export_risk import (
    branch_from_sample_id, history_spans, label_reports,
)


class WindowTests(unittest.TestCase):
    def test_short_and_long_cover_all_tokens(self):
        self.assertEqual(windows(10, 13, 128, 96), [(10, 13)])
        parts = windows(0, 350, 128, 96)
        self.assertEqual(parts[0], (0, 128))
        self.assertEqual(parts[-1], (222, 350))
        self.assertEqual(set(i for left, right in parts for i in range(left, right)),
                         set(range(350)))
        self.assertTrue(all(right - left <= 128 for left, right in parts))

    def test_offsets_overlap_multibyte_text(self):
        offsets = [(0, 2), (2, 6), (6, 10), (10, 12)]
        self.assertEqual(span_to_tokens(offsets, 3, 10), (1, 3))

    def test_sample_rejects_overlap_or_missing_labels(self):
        sample = {"schema": VERSION, "prompt": "abcdefghijklmnop",
                  "reports": [{"report_id": "a", "start": 0, "end": 4,
                               "risk": 0, "mask": 1},
                              {"report_id": "b", "start": 4, "end": 8,
                               "risk": 1, "mask": 0}]}
        validate(sample)
        sample["reports"][1]["start"] = 3
        with self.assertRaises(ValueError):
            validate(sample)


class ExportTests(unittest.TestCase):
    def test_branch_id_and_native_history_report_blocks(self):
        self.assertEqual(branch_from_sample_id("task-A", 
                         "task-A-fork-round-03-candidate-005"), 3)
        text = ("[Attempt 1] 🔄 Parallel Batch (2 subtasks)\n"
                "  [1] ✅ done | Model: model_1 | Steps: 2/30\n"
                "  ├─ Task: First\n  ├─ Result: value\n  └─ Summary: one\n"
                "  [2] ✅ done | Model: model_1 | Steps: 2/30\n"
                "  ├─ Task: Second\n  ├─ Result: value\n  └─ Summary: two\n\n"
                "[Attempt 2] 🔄 Parallel Batch (1 subtasks)\n"
                "  [1] ⚠️ partial | Model: model_1 | Steps: 2/30\n"
                "  ├─ Task: Retry\n  └─ Result: (no result)\n\n"
                "---\nSummary: 2/3 subtasks done")
        spans = history_spans(text, [(1, 2), (2, 1)])
        self.assertEqual([(r, s) for r, s, _, _ in spans],
                         [(1, 0), (1, 1), (2, 0)])
        self.assertIn("Summary: one", text[spans[0][2]:spans[0][3]])
        self.assertNotIn("Summary: two", text[spans[0][2]:spans[0][3]])

    def test_visible_dispute_masks_unparseable_and_superseded(self):
        snapshots = [
            {"main_can_act_from_round": 2, "report_events": [
                {"macro_task_id": "a", "status": "done", "visible_claim_count": 1},
                {"macro_task_id": "b", "status": "done", "visible_claim_count": 0},
            ]},
            {"main_can_act_from_round": 3, "report_events": [
                {"macro_task_id": "c", "status": "done", "visible_claim_count": 1},
                {"macro_task_id": "a", "status": "partial", "visible_claim_count": 0},
            ]},
        ]
        state = {"claim_nodes": [
            {"claim_id": "b1", "report_task_id": "c"},
        ], "claim_states": {
            "k": {"status": "disputed", "supporting_claim_ids": ["b1"]}
        }, "claim_dependency_edges": []}
        slots = [(1, 0, 0, 4), (1, 1, 4, 8),
                 (2, 0, 8, 12), (2, 1, 12, 16)]
        labels = label_reports(state, snapshots, slots)
        self.assertEqual([(x["risk"], x["mask"]) for x in labels],
                         [(0, 0), (0, 0), (1, 1), (1, 1)])
        self.assertEqual(labels[0]["label_reason"], "superseded")
        self.assertEqual(labels[1]["label_reason"], "unparsed")

    def test_visible_dependency_ambiguity_not_negative(self):
        snapshots = [{"main_can_act_from_round": 2, "report_events": [
            {"macro_task_id": "parent", "status": "done", "visible_claim_count": 1},
            {"macro_task_id": "child", "status": "done", "visible_claim_count": 1},
        ]}]
        state = {"claim_nodes": [
            {"claim_id": "p", "report_task_id": "parent"},
            {"claim_id": "c", "report_task_id": "child"},
        ], "claim_states": {
            "k": {"status": "disputed", "supporting_claim_ids": ["p"]},
            "x": {"status": "unverified", "supporting_claim_ids": ["c"]},
        }, "claim_dependency_edges": [
            {"parent_claim_id": "p", "child_claim_id": "c"},
        ]}
        rows = label_reports(state, snapshots, [(1, 0, 0, 4), (1, 1, 4, 8)])
        self.assertEqual((rows[0]["risk"], rows[0]["mask"]), (1, 1))
        self.assertEqual((rows[1]["risk"], rows[1]["mask"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
