"""Cross-baseline contract: one unified source, two method-owned data outputs."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from environment.data_export.source_io import digest


ROOT = Path(__file__).resolve().parents[1]


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sft = load_script("baseline/da-sft/prepare_data.py", "sft_prepare")
grpo = load_script("baseline/da-grpo/prepare_data.py", "grpo_prepare")


def fixture(root: Path):
    source = root / "environment/data/unified"
    (source / "private").mkdir(parents=True)
    cases, decisions, teachers = [], [], []
    for i in range(10):
        scene = f"scene-{i:02d}"
        cases.append({
            "case_id": "clean:" + scene,
            "scene_id": scene,
            "kind": "clean",
        })
        action = {
            "action": "complete",
            "reasoning": "Evidence verified " + scene,
            "params": {"answer": scene},
        }
        decisions.append({
            "schema": "maestro.main_decision.v1",
            "decision_id": "stage3:" + scene,
            "case_id": "clean:" + scene,
            "scene_id": scene,
            "source": "clean",
            "decision_round": 1,
            "generation_mode": "llm_reviewed",
            "question": "Find source " + scene,
            "visible_state": {"subtask_history": "No subtasks completed yet."},
            "prompt": [{"role": "user", "content": "O1 visible prompt " + scene}],
            "reference_action": action,
        })
    cases.append({
        "case_id": "fault:scene-00:1",
        "scene_id": "scene-00",
        "kind": "fault_injected",
        "fault": {"atoms": [{"kind": "wrong_retrieval"}]},
        "recovery": {"globally_recovered": True},
    })
    decisions.append({
        **deepcopy(decisions[0]),
        "decision_id": "stage4:scene-00:1",
        "case_id": "fault:scene-00:1",
        "source": "untrusted",
        "decision_round": 2,
        "generation_mode": "llm_reviewed",
        "prompt": [{
            "role": "user",
            "content": "O1 recovery prompt with visible reports",
        }],
    })
    decisions.append({
        **deepcopy(decisions[0]),
        "decision_id": "stage4:scene-00:2",
        "case_id": "fault:scene-00:1",
        "source": "untrusted",
        "decision_round": 3,
        "generation_mode": "template",
        "prompt": [{
            "role": "user", "content": "O1 fallback example",
        }],
    })
    for row in decisions:
        teachers.append({
            "schema": "maestro.decision_teacher.v1",
            "decision_id": row["decision_id"],
            "scene_id": row["scene_id"],
            "case_id": row["case_id"],
            "ground_truth": row["scene_id"],
            "reference_action": deepcopy(row["reference_action"]),
            "reward_state": None,
        })
    files = {
        "cases.jsonl": cases,
        "decisions.jsonl": decisions,
        "private/decision_labels.jsonl": teachers,
    }
    for name, rows in files.items():
        (source / name).write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ), encoding="utf-8")
    manifest = {
        "schema": "maestro.dataset.v1",
        "base_file_sha256": {
            name: digest(source / name) for name in files
        },
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return source


@unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None,
                     "pyarrow required by method-local Parquet preparation")
class BaselineDataPreparationTests(unittest.TestCase):
    def test_sft_and_grpo_own_outputs_share_scene_split(self):
        import pyarrow.parquet as pq

        with TemporaryDirectory() as temp:
            root = Path(temp)
            src = fixture(root)
            sft_out = root / "baseline/da-sft/data"
            grpo_out = root / "baseline/da-grpo/data"
            a = sft.prepare(src, sft_out)
            b = grpo.prepare(src, grpo_out)
            self.assertEqual(a["counts"], b["counts"])
            self.assertEqual(a["scene_split"], b["scene_split"])
            self.assertEqual(a["counts"]["train"] + a["counts"]["val"] +
                             a["counts"]["test"], 11)
            self.assertEqual(sum(a["generation_mode_skipped"].values()), 1)
            self.assertTrue((sft_out / "train.parquet").exists())
            self.assertTrue((sft_out / "eval.parquet").exists())
            self.assertTrue((sft_out / "test.parquet").exists())
            self.assertTrue((grpo_out / "train.parquet").exists())
            self.assertTrue((grpo_out / "val.parquet").exists())
            self.assertTrue((grpo_out / "test.parquet").exists())
            self.assertFalse((src / "adapters").exists())
            for split in ("train", "val", "test"):
                sft_path = "eval" if split == "val" else split
                sft_rows = pq.read_table(
                    sft_out / (sft_path + ".parquet"),
                ).to_pylist()
                grpo_rows = pq.read_table(
                    grpo_out / (split + ".parquet"),
                ).to_pylist()
                self.assertEqual(len(sft_rows), len(grpo_rows))
                self.assertTrue(all(
                    set(row) == {"prompt", "completion"}
                    for row in sft_rows
                ))
                self.assertEqual(
                    [row["prompt"] for row in sft_rows],
                    [row["prompt"] for row in grpo_rows],
                )
                for row in grpo_rows:
                    self.assertEqual(set(row), {
                        "data_source", "prompt", "reward_model", "extra_info",
                    })
                    self.assertTrue(row["extra_info"]["has_expert"])
                    self.assertEqual(
                        row["reward_model"]["ground_truth"],
                        row["extra_info"]["answer"],
                    )
                    self.assertNotIn("extra_info", row["prompt"][0])
                    self.assertNotIn("reward_model", row["prompt"][0])
            with self.assertRaises(FileExistsError):
                sft.prepare(src, sft_out)

    def test_include_fallback_and_reuse_external_split_map(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            src = fixture(root)
            output = root / "sft"
            main = sft.prepare(src, output)
            extra = grpo.prepare(
                src, root / "grpo", include_fallback=True,
                split_map=output / "splits.json",
            )
            self.assertEqual(extra["counts"]["train"] +
                             extra["counts"]["val"] + extra["counts"]["test"], 12)
            self.assertEqual(main["scene_split"], extra["scene_split"])

    def test_dry_run_does_not_write_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            src = fixture(root)
            out = root / "only_dry"
            self.assertEqual(sft.prepare(src, out, dry_run=True)["counts"]["test"], 1)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
