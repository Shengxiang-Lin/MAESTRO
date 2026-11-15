"""Fast unit tests for frozen split manifests and dataset ID selection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.experiment_splits import (
    MANIFESTS, OMNI_FINAL_100, OMNI_PRETEST_60, load_task_ids,
)
from benchmark.omnigaia.orchestra_benchmark import (
    OmniGAIAConfig, OmniGAIAOrchestraBenchmark,
)


class ExperimentSplitTests(unittest.TestCase):
    def test_frozen_sizes_and_non_overlap(self):
        self.assertEqual([len(OMNI_PRETEST_60), len(OMNI_FINAL_100)], [60, 100])
        self.assertFalse(set(OMNI_PRETEST_60).intersection(OMNI_FINAL_100))
        self.assertEqual(sorted(len(v) for v in MANIFESTS.values()), [60, 100, 100])

    def test_id_manifest_validation(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "ids.txt"
            path.write_text("# allowed comment\n9\n 7 # note\n", encoding="utf-8")
            self.assertEqual(load_task_ids(path), ["9", "7"])
            path.write_text("9\n9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_task_ids(path)
            path.write_text("9,7\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one task ID"):
                load_task_ids(path)

    def _fixture(self, root: Path, manifest: str, max_tasks=None):
        dataset = root / "test.jsonl"
        dataset.write_text(
            "".join(json.dumps(row) + "\n" for row in [
                {"id": 7, "question": "Question 7", "Level": "Easy"},
                {"id": 9, "question": "Question 9", "Level": "Hard"},
                {"id": 11, "question": "Question 11", "Level": "Medium"},
            ]), encoding="utf-8",
        )
        selected = root / "ids.txt"
        selected.write_text(manifest, encoding="utf-8")
        return OmniGAIAConfig(
            dataset_path=dataset, attachments_dir=root,
            task_ids_file=selected, max_tasks=max_tasks,
        )

    def test_selected_ids_preserve_manifest_order(self):
        with TemporaryDirectory() as directory:
            config = self._fixture(Path(directory), "9\n7\n")
            benchmark = OmniGAIAOrchestraBenchmark(config)
            self.assertEqual([level["id"] for level in benchmark.list_levels()], [9, 7])

    def test_missing_id_and_limit_raise(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing from"):
                OmniGAIAOrchestraBenchmark(self._fixture(root, "100\n"))
            with self.assertRaisesRegex(ValueError, "would truncate"):
                OmniGAIAOrchestraBenchmark(self._fixture(root, "9\n7\n", max_tasks=1))


if __name__ == "__main__":
    unittest.main()
