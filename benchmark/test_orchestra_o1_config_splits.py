"""Regression checks for Orchestra-o1's dataset/stage config organization."""
from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "baseline" / "orchestra-o1" / "config"

STAGES = {
    "omnigaia/pretest": (
        12, "OmniGAIA", "data/OmniGAIA/splits/pretest_60_ids.txt",
        "logs/omnigaia_pretest/"
    ),
    "omnigaia/final": (
        12, "OmniGAIA", "data/OmniGAIA/splits/final_100_ids.txt",
        "logs/omnigaia_final/"
    ),
    "mm_browsecomp/final": (
        7, "MM-BrowseComp", "data/MM-BrowseComp/splits/final_100_ids.txt",
        "logs/mm_browsecomp_final/"
    ),
}


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return result


class OrchestraO1ConfigSplitTests(unittest.TestCase):
    def test_stage_directory_counts_and_no_legacy_files(self):
        for subdir, (expected, *_rest) in STAGES.items():
            with self.subTest(stage=subdir):
                self.assertEqual(
                    len(list((CONFIG_ROOT / subdir).glob("*.yaml"))), expected
                )
        legacy = CONFIG_ROOT / "benchmarks"
        self.assertFalse(legacy.exists() and any(legacy.glob("*.yaml")))

    def test_all_configs_pin_expected_ids_and_isolate_logs(self):
        output_folders: set[str] = set()
        for stage, (count, dataset, task_ids, prefix) in STAGES.items():
            configs = sorted((CONFIG_ROOT / stage).glob("*.yaml"))
            self.assertEqual(len(configs), count)
            for path in configs:
                with self.subTest(stage=stage, name=path.name):
                    cfg = read_config(path)
                    self.assertEqual(cfg.get("dataset_type"), dataset)
                    self.assertEqual(cfg.get("task_ids_file"), task_ids)
                    self.assertIn("max_tasks", cfg)
                    self.assertIsNone(cfg["max_tasks"])
                    self.assertIsNone(cfg.get("level_filter"))
                    self.assertIsInstance(cfg.get("main_model"), str)
                    self.assertTrue(cfg.get("sub_models"))
                    result = cfg["result_folder"]
                    self.assertTrue(result.startswith(prefix))
                    self.assertEqual(
                        cfg["trajectory_folder"], result + "/trajectories"
                    )
                    self.assertNotIn(result, output_folders)
                    output_folders.add(result)

    def test_omnigaia_model_roles_and_settings_match(self):
        pretest = CONFIG_ROOT / "omnigaia/pretest"
        final = CONFIG_ROOT / "omnigaia/final"
        self.assertEqual(
            {p.name for p in pretest.glob("*.yaml")},
            {p.name for p in final.glob("*.yaml")},
        )
        stage_specific = {"task_ids_file", "result_folder", "trajectory_folder"}
        for pretest_file in sorted(pretest.glob("*.yaml")):
            with self.subTest(config=pretest_file.name):
                cfg_a = read_config(pretest_file)
                cfg_b = read_config(final / pretest_file.name)
                for key in stage_specific:
                    cfg_a.pop(key, None)
                    cfg_b.pop(key, None)
                self.assertEqual(cfg_a, cfg_b)

    def test_fixed_id_manifests_have_expected_lengths(self):
        paths = [
            ("OmniGAIA/splits/pretest_60_ids.txt", 60),
            ("OmniGAIA/splits/final_100_ids.txt", 100),
            ("MM-BrowseComp/splits/final_100_ids.txt", 100),
        ]
        for relative, size in paths:
            with self.subTest(manifest=relative):
                manifest = ROOT / "data" / relative
                ids = [line.strip() for line in manifest.read_text().splitlines()
                       if line.strip()]
                self.assertEqual(len(ids), size)
                self.assertEqual(len(set(ids)), size)


if __name__ == "__main__":
    unittest.main()
