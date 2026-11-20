"""Frozen evaluator-only Scene World/claim source regression tests."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from environment.data_export.audit import audit_output
from environment.data_export.private_context import (
    attach_to_unified, collect_private_context, verify_private_context,
)
from environment.data_export.source_io import digest
from environment.tests.test_data_audit import example


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FrozenGlobalContextTests(unittest.TestCase):
    def sample_sources(self, root: Path) -> tuple[Path, Path, Path]:
        base, stage4 = root / "unified", root / "stage4"
        cases, decisions, labels = example()
        source_files = {
            "cases.jsonl": cases,
            "decisions.jsonl": decisions,
            "private/decision_labels.jsonl": labels,
        }
        for name, rows in source_files.items():
            write(base / name, rows)
        scenes = root / "stage1/scenes.jsonl"
        write(scenes, [{
            "task_id": key,
            "question": "Question " + key,
            "gold_answer": [{"answer": key}],
            "world": {"sources": {"secret_source": {
                "observations": [{"values": {"secret": "PRIVATE_WORLD"}}],
            }}},
            "macro_graph": {"tasks": [{"task_id": "t1", "depends_on": []}]},
        } for key in ("a", "b", "c")])
        (stage4 / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (stage4 / "manifest.json").write_text('{"settings":1}\n', encoding="utf-8")
        write(stage4 / "main_turns.jsonl", [{"round_sample_id": "fault:a:1"}])
        write(stage4 / "private/claim_state_supervision.jsonl", [{
            "schema": "claim_source_dependency",
            "task_id": "a",
            "private_only": True,
            "round_sample_id": "fault:a:1",
            "main_can_act_from_round": 2,
            "claim_nodes": [{
                "observed_in_round": 1,
                "subject": "project",
                "attribute": "status",
                "value": "Pending",
            }],
            "claim_states": {},
        }])
        manifest = {
            "schema": "maestro.dataset.v1",
            "base_file_sha256": {
                name: digest(base / name) for name in source_files
            },
            "source_sha256": {
                "scenes": digest(scenes),
                "stage4_manifest": digest(stage4 / "manifest.json"),
                "stage4_main_turns": digest(stage4 / "main_turns.jsonl"),
            },
        }
        (base / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        return base, scenes, stage4

    def test_backfill_preserves_frozen_base_and_can_delete_original_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, scenes, stage4 = self.sample_sources(root)
            before = {
                name: digest(base / name)
                for name in ("decisions.jsonl", "cases.jsonl",
                             "private/decision_labels.jsonl")
            }
            result = attach_to_unified(
                unified_dir=base, scenes_path=scenes, stage4_dir=stage4,
            )
            self.assertEqual(result["required_claim_snapshots"], 1)
            self.assertEqual(result["present_claim_snapshots"], 1)
            self.assertEqual(result["missing_claim_snapshots"], 0)
            self.assertEqual(verify_private_context(base)["scene_count"], 3)
            self.assertEqual(audit_output(base, verify_manifest=True)["decision_count"], 4)
            self.assertEqual(before, {
                name: digest(base / name) for name in before
            })
            self.assertNotIn("PRIVATE_WORLD", (base / "decisions.jsonl").read_text())
            self.assertIn("PRIVATE_WORLD", (base / "private/scenes.jsonl").read_text())
            scenes.unlink()
            (stage4 / "private/claim_state_supervision.jsonl").unlink()
            (stage4 / "main_turns.jsonl").unlink()
            (stage4 / "manifest.json").unlink()
            self.assertEqual(verify_private_context(base)["present_claim_snapshots"], 1)
            with (base / "private/scenes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "checksum_changed"):
                audit_output(base, verify_manifest=True)

    def test_mismatched_source_is_rejected_without_modifying_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, scenes, stage4 = self.sample_sources(root)
            original = (base / "manifest.json").read_bytes()
            with scenes.open("a", encoding="utf-8") as stream:
                stream.write('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "source_mismatch:scenes"):
                attach_to_unified(
                    unified_dir=base, scenes_path=scenes, stage4_dir=stage4,
                )
            self.assertEqual((base / "manifest.json").read_bytes(), original)

    def test_future_or_duplicate_claims_fail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, scenes_path, stage4 = self.sample_sources(root)
            scenes = {
                item["task_id"]: item for item in
                [json.loads(line) for line in scenes_path.read_text().splitlines()]
            }
            decisions = example()[1]
            claims = stage4 / "private/claim_state_supervision.jsonl"
            record = json.loads(claims.read_text())
            record["claim_nodes"][0]["observed_in_round"] = 2
            write(claims, [record])
            with self.assertRaisesRegex(ValueError, "future_claim"):
                collect_private_context(scenes, decisions, claims)
            record["claim_nodes"][0]["observed_in_round"] = 1
            write(claims, [record, record])
            with self.assertRaisesRegex(ValueError, "duplicate_claim"):
                collect_private_context(scenes, decisions, claims)


if __name__ == "__main__":
    unittest.main()
