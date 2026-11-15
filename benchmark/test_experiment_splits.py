"""Unit tests for metadata-derived fixed-seed benchmark sampling."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmark.experiment_splits import (
    MM_FINAL, OMNI_FINAL, PRETEST, generate_splits, load_task_ids,
)
from benchmark.omnigaia.orchestra_benchmark import (
    OmniGAIAConfig, OmniGAIAOrchestraBenchmark,
)


def synthetic_metadata():
    """Small but diverse 9-domain / 22-subtask benchmark fixtures."""
    omni = []
    counter = 1
    groups = ("Geography & Travel", "History & Society", "Technology",
              "Sports", "Arts & Culture", "Movies", "Science & Nature",
              "Finance & Commerce", "Food & Nutrition")
    for domain in groups:
        for number in range(26):
            omni.append({
                "id": counter,
                "category": domain,
                "Level": ("Easy", "Medium", "Hard")[number % 3],
                "task_type": ("image + audio", "video w/ audio")[number % 2],
                "total_steps": 4 + number % 7,
                "omni_modal_input": [{"id": f"media_{counter}"}],
                "required_external_tools": ["code_executor"] if number % 3 else [],
            })
            counter += 1

    mm = []
    mid = 1
    types = {
        "Geography": 4,
        "Media": 6,
        "Technology": 6,
        "Society": 3,
        "Academics": 3,
    }
    for category, count in types.items():
        for subtask in range(count):
            for number in range(20):
                mm.append({
                    "id": mid,
                    "category": category,
                    "subtask": f"{category}-{subtask}",
                    "level": 2 if number % 7 == 0 else 1,
                    "images": [f"image_{mid}"] if number % 2 else [],
                    "source": ["source1", "source2"],
                })
                mid += 1
    return omni, mm


class ExperimentSplitTests(unittest.TestCase):
    def test_seed_reproducibility_and_strata(self):
        omni, mm = synthetic_metadata()
        samples, report = generate_splits(omni, mm, seed=17, trials=60)
        same, _ = generate_splits(list(reversed(omni)), list(reversed(mm)),
                                  seed=17, trials=60)
        changed, _ = generate_splits(omni, mm, seed=18, trials=60)
        self.assertEqual(samples, same)
        self.assertNotEqual(samples, changed)
        self.assertEqual([len(samples[x]) for x in (PRETEST, OMNI_FINAL, MM_FINAL)],
                         [60, 100, 100])
        self.assertFalse(set(samples[PRETEST]) & set(samples[OMNI_FINAL]))
        self.assertEqual(report["selected"]["omni_pretest"]["levels"],
                         {"Easy": 20, "Medium": 20, "Hard": 20})
        self.assertEqual(len(report["selected"]["mm_final"]["categories"]), 5)
        self.assertEqual(report["selected"]["mm_final"]["subtask_coverage"], 22)

    def test_missing_and_duplicate_manifest_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ids.txt"
            path.write_text("# Comment\n9\n 7 # note\n", encoding="utf-8")
            self.assertEqual(load_task_ids(path), ["9", "7"])
            path.write_text("9\n9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_task_ids(path)
            path.write_text("9,7\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one task ID"):
                load_task_ids(path)

    def _fixture(self, root: Path, ids: str, max_tasks=None):
        dataset = root / "test.jsonl"
        dataset.write_text(
            "".join(json.dumps(row) + "\n" for row in [
                {"id": 7, "question": "Question 7", "Level": "Easy"},
                {"id": 9, "question": "Question 9", "Level": "Hard"},
                {"id": 11, "question": "Question 11", "Level": "Medium"},
            ]), encoding="utf-8",
        )
        manifest = root / "ids.txt"
        manifest.write_text(ids, encoding="utf-8")
        return OmniGAIAConfig(dataset_path=dataset, attachments_dir=root,
                              task_ids_file=manifest, max_tasks=max_tasks)

    def test_runner_loads_in_manifest_order(self):
        with TemporaryDirectory() as directory:
            benchmark = OmniGAIAOrchestraBenchmark(self._fixture(Path(directory), "9\n7\n"))
            self.assertEqual([x["id"] for x in benchmark.list_levels()], [9, 7])

    def test_missing_ids_or_truncation_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing from"):
                OmniGAIAOrchestraBenchmark(self._fixture(root, "100\n"))
            with self.assertRaisesRegex(ValueError, "would truncate"):
                OmniGAIAOrchestraBenchmark(self._fixture(root, "9\n7\n", max_tasks=1))


if __name__ == "__main__":
    unittest.main()
