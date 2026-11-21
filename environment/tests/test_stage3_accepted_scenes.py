"""Stage-3 accepted Scene publication is stable and safe for Stage 4."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from environment.base_orchestration.generate_turns import _rebuild_publication


class AcceptedScenePublicationTests(TestCase):
    def test_rebuild_outputs_only_complete_accepted_scenes(self):
        scenes = [{"task_id": "one", "world": {"name": "original"}},
                  {"task_id": "two", "world": {"name": "other"}}]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            journal = output / "journal"
            journal.mkdir()
            (journal / "000000.json").write_text(json.dumps({
                "task_id": "one",
                "status": {"status": "accepted"},
                "episode": {"task_id": "one"},
                "turns": [{"task_id": "one", "round": 1}],
                "private_audit": [{"task_id": "one", "round": 1}],
            }))
            (journal / "000001.json").write_text(json.dumps({
                "task_id": "two",
                "status": {"status": "rejected", "cause": "review_failed"},
                "episode": None, "turns": [], "private_audit": [],
            }))
            episodes, turns, audits, failed = _rebuild_publication(
                scenes, journal, output,
            )
            selected = [
                json.loads(line) for line in
                (output / "accepted_scenes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(selected, scenes[:1])
            self.assertEqual(len(episodes), len(turns))
            self.assertEqual(len(audits), len(failed))
            self.assertEqual([r["task_id"] for r in episodes], ["one"])
            self.assertEqual([r["task_id"] for r in failed], ["two"])

            # Rebuilding after a restart must not duplicate accepted Scenes.
            _rebuild_publication(scenes, journal, output)
            self.assertEqual(
                (output / "accepted_scenes.jsonl").read_text().count("\n"), 1,
            )
