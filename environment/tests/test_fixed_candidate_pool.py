"""Fixed-size validated candidate pool: one production flow, no world quota."""
from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from environment.config import parse_with_config
from environment.task_construction.deep_dependency import candidate_selection as cs
from environment.task_construction.deep_dependency.world_generation import generate_world_batch
from environment.task_construction.pipeline import build_parser


class FixedCandidatePoolTests(unittest.TestCase):
    def test_path_expansion_uses_only_candidate_pool_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"test.yaml"
            p.write_text(yaml.safe_dump({
                "version":1,
                "task_construction":{"candidate_pool_size":2000},
                "question_generation":{
                    "manifest":"environment/data/frozen_manifest_{target}.json",
                },
            }),encoding="utf-8")
            parser=argparse.ArgumentParser()
            parser.add_argument("--manifest")
            args=parse_with_config(parser,"question_generation",
                                   ["--config",str(p)])
            self.assertTrue(args.manifest.endswith("frozen_manifest_2000.json"))

    def test_legacy_quota_flags_are_removed(self):
        parser=build_parser()
        args=parser.parse_args(["--candidate-pool-size","20"])
        self.assertEqual(args.candidate_pool_size,20)
        for flag in ("--target","--shortlist-target","--candidate-multiplier",
                     "--all-valid-candidates","--allow-partial-batch"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag,"100"] if flag!="--all-valid-candidates"
                                      and flag!="--allow-partial-batch" else [flag])
        self.assertNotIn("target",inspect.signature(generate_world_batch).parameters)
        self.assertNotIn("reserve_candidates_path",
                         inspect.signature(generate_world_batch).parameters)

    def test_exact_validated_pool_without_second_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            catalog=directory/"catalog.json"
            taxonomy=directory/"taxonomy.json"
            catalog.write_text("{}",encoding="utf-8")
            taxonomy.write_text("{}",encoding="utf-8")
            output=directory/"semantic_candidates_pool_5.jsonl"
            row={
                "program":{"nodes":[],"metadata":{"deep_dependency":{}}},
                "binding":{"relation_bindings":{},"attribute_bindings":{}},
                "source_program_signature":"test-signature",
                "terminal_family":"entity_lookup",
                "topology_signature":"test-topology",
            }
            candidates=[
                dict(row,candidate_id=f"candidate-{i:06d}") for i in range(5)
            ]
            stats={"attempts":15,"rejections":{},"examples":[],
                   "workers":1,"proposal_batch_size":2}

            class DummyProgram:
                def canonical_signature(self):
                    return "source-signature"

            with (
                patch.object(cs,"_iter_kqa_programs",
                             return_value=[DummyProgram()]),
                patch.object(cs,"prepare_composition_pools",
                             return_value=([object()],[object()])),
                patch.object(cs.TaxonomyIndex,"from_json",return_value=object()),
                patch("environment.task_construction.deep_dependency.parallel_candidates.input_signature",
                      return_value={"test":True}),
                patch("environment.task_construction.deep_dependency.parallel_candidates.prepare_worker_context"),
                patch("environment.task_construction.deep_dependency.parallel_candidates.run_batches",
                      return_value=(candidates,stats)) as run_batches,
            ):
                result=cs._generate_candidate_pool(
                    ["ignored.json"],catalog,taxonomy,output,
                    pool_size=5,candidate_workers=1,
                )
            self.assertEqual(run_batches.call_args.kwargs["target"],5)
            self.assertEqual(result["candidate_pool_size"],5)
            self.assertEqual(result["validated_candidates"],5)
            self.assertTrue(result["pool_filled"])
            rows=[json.loads(line) for line in output.read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual([r["candidate_id"] for r in rows],
                             [r["candidate_id"] for r in candidates])
            self.assertFalse(output.with_name(
                "semantic_candidates_pool_5.reservoir.jsonl").exists())


if __name__=="__main__":
    unittest.main()
