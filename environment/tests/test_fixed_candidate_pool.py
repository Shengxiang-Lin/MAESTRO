"""Fixed candidate-pool mode does not impose an accepted-world quota."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from environment.config import parse_with_config
from environment.task_construction.deep_dependency import candidate_selection as cs
from environment.task_construction.deep_dependency.world_generation import _stop_after_target


class FixedCandidatePoolTests(unittest.TestCase):
    def test_path_expansion_uses_pool_size_not_legacy_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.yaml"
            p.write_text(yaml.safe_dump({
                "version": 1,
                "task_construction": {
                    "target": 100,
                    "all_valid_candidates": True,
                    "candidate_pool_size": 2000,
                },
                "question_generation": {
                    "manifest": "environment/data/frozen_manifest_{target}.json",
                },
            }), encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--manifest")
            args = parse_with_config(parser, "question_generation", ["--config", str(p)])
            self.assertTrue(args.manifest.endswith("frozen_manifest_2000.json"))

    def test_no_world_quota_stop_when_pool_mode_is_active(self):
        self.assertTrue(_stop_after_target(100, 100, False))
        self.assertFalse(_stop_after_target(100, 100, True))
        self.assertFalse(_stop_after_target(800, 100, True))

    def test_exact_validated_pool_with_no_second_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            catalog = directory / "catalog.json"
            taxonomy = directory / "taxonomy.json"
            catalog.write_text("{}", encoding="utf-8")
            taxonomy.write_text("{}", encoding="utf-8")
            output = directory / "semantic_candidates_pool_5.jsonl"
            row = {
                "program": {"nodes": [], "metadata": {"deep_dependency": {}}},
                "binding": {"relation_bindings": {}, "attribute_bindings": {}},
                "source_program_signature": "test-signature",
                "terminal_family": "entity_lookup",
                "topology_signature": "test-topology",
            }
            reservoir = [dict(row, candidate_id=f"candidate-{i:06d}") for i in range(5)]
            stats = {
                "attempts": 15, "rejections": {}, "examples": [],
                "workers": 1, "proposal_batch_size": 2,
            }
            class DummyProgram:
                def canonical_signature(self):
                    return "source-signature"

            with (
                patch.object(cs, "_iter_kqa_programs", return_value=[DummyProgram()]),
                patch.object(cs, "prepare_composition_pools",
                             return_value=([object()], [object()])),
                patch.object(cs.TaxonomyIndex, "from_json", return_value=object()),
                patch("environment.task_construction.deep_dependency.parallel_candidates.input_signature",
                      return_value={"test": True}),
                patch("environment.task_construction.deep_dependency.parallel_candidates.prepare_worker_context"),
                patch("environment.task_construction.deep_dependency.parallel_candidates.run_batches",
                      return_value=(reservoir, stats)) as run_batches,
            ):
                result = cs._generate_world_candidate_shortlist(
                    ["ignored.json"], catalog, taxonomy, output,
                    shortlist_target=5, world_target=5,
                    candidate_multiplier=8.0, all_valid_candidates=True,
                    candidate_workers=1,
                )
            self.assertEqual(run_batches.call_args.kwargs["target"], 5)
            self.assertEqual(result["reservoir_target"], 5)
            self.assertEqual(result["accepted"], 5)
            self.assertEqual(result["diversity"]["policy"], "deferred_to_world_generation")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["candidate_id"] for r in rows],
                             [r["candidate_id"] for r in reservoir])


if __name__ == "__main__":
    unittest.main()
