"""Focused tests for the dataset-driven Task-First question generator."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from envir.macro.scene_catalog import load_scenario_seeds, parse_composition
from envir.macro.data_taskfirst import UnsuitableSeed, compose_seed_taskfirst
from envir.macro.v4_builder import build_one
from envir.macro.v4_validation import validate_clean_scene


KQA = [{
    "signature": "fixture-kopl",
    "program": {"nodes": []},
    "logical_depth": 4,
    "branch_nodes": 1,
    "merge_nodes": 1,
    "operator_count": 4,
}]


def seed(tree="Compare(Compose(TableQ,ImageQ),TextQ)"):
    return {
        "record_id": "mmqa-fixture-1",
        "question": "Which artist has the earlier exhibition date?",
        "topic": "two artists",
        "entities": ["Artist A", "Artist B"],
        "domain_hint": "culture",
        "operation": "comparison",
        "question_type": tree,
        "logical_pattern": "comparison",
        "composition": parse_composition(tree),
        "source_modalities": ["table", "image", "text"],
        "supporting_sources": [
            {"modality": "table", "doc_id": "t1",
             "provenance": "mmqa_supporting_context"},
            {"modality": "image", "doc_id": "i1",
             "provenance": "mmqa_supporting_context"},
            {"modality": "text", "doc_id": "x1",
             "provenance": "mmqa_supporting_context"},
        ],
        "yago_matches": [],
    }


class MMQATaskFirstTests(unittest.TestCase):
    def test_nested_composition_parser_preserves_dependencies(self):
        tree = parse_composition("Compare(Compose(TableQ,ImageQ),TextQ)")
        self.assertEqual(tree["op"], "Compare")
        self.assertEqual(tree["children"][0]["op"], "Compose")
        self.assertEqual(tree["children"][0]["children"][1]["op"], "ImageQ")
        self.assertEqual(tree["children"][1]["op"], "TextQ")

    def test_catalog_preserves_source_ids_and_does_not_copy_answers(self):
        record = {
            "qid": "mmqa-001",
            "question": "Which artist has the earlier exhibition date?",
            "answers": [{"answer": "SECRET_ANSWER"}],
            "wiki_entities_in_question": ["Artist A", "Artist B"],
            "metadata": {
                "type": "Compare(TableQ,ImageQ)",
                "supporting_context": [
                    {"doc_id": "table-101", "doc_part": "table"},
                    {"doc_id": "image-202", "doc_part": "image"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "MMQA_train.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            seeds, stats = load_scenario_seeds(path)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(stats["unique_scenario_seeds"], 1)
        self.assertEqual([s["doc_id"] for s in seeds[0]["supporting_sources"]],
                         ["table-101", "image-202"])
        self.assertNotIn("SECRET_ANSWER", json.dumps(seeds))

    def test_macro_dag_compiled_before_question(self):
        record = seed()
        plan = compose_seed_taskfirst(
            seed_record=record, ordinal=0, run_seed=17, kqa=KQA)
        tasks = plan["macro_graph"]["tasks"]
        self.assertGreaterEqual(len(tasks), 7)
        self.assertGreaterEqual(len(plan["macro_graph"]["edges"]), 7)
        self.assertNotEqual(plan["question"], record["question"])
        from envir.macro.goal_grammar import GOALS
        self.assertIn(plan["goal"]["intent"], GOALS)
        self.assertEqual(
            plan["macro_graph"]["terminal_handoffs"][0]["type"],
            GOALS[plan["goal"]["intent"]]["artifact"],
        )
        self.assertEqual(plan["source_priors"]["MultiModalQA"]["record_id"],
                         record["record_id"])
        self.assertEqual(plan["source_priors"]["KQA"]["signature"], "fixture-kopl")
        self.assertTrue(all(edge["type"] for edge in plan["macro_graph"]["edges"]))

    def test_complete_clean_replay_and_question_provenance(self):
        scene = build_one(
            ordinal=0, seed=17, kqa=KQA, seed_record=seed())
        self.assertEqual([], validate_clean_scene(scene))
        self.assertEqual(scene["public"]["question"], scene["question"])
        self.assertNotEqual(scene["question"], seed()["question"])
        self.assertEqual(scene["goal_evidence_coverage"]["can_finish_cleanly"], True)
        self.assertGreaterEqual(len(scene["macro_graph"]["tasks"]), 7)
        self.assertEqual(scene["world"]["real_multimodal_files"], False)

    def test_two_independent_sources_yield_real_work_products(self):
        record = seed("TextQ")
        record["supporting_sources"] = record["supporting_sources"][:2]
        record["source_modalities"] = ["table", "image"]
        plan = compose_seed_taskfirst(
            seed_record=record, ordinal=1, run_seed=17, kqa=KQA)
        self.assertEqual(len(plan["macro_graph"]["tasks"]), 5)
        self.assertEqual(len(plan["macro_graph"]["edges"]), 6)
        self.assertIn("entity_reference_alignment",
                      {task["work_type"] for task in plan["macro_graph"]["tasks"]})

    def test_every_mmqa_supporting_source_is_assigned_to_reachable_work(self):
        import copy
        record = seed("Compose(TableQ,ImageListQ)")
        record["supporting_sources"].extend([
            {"modality": "image", "doc_id": "i2",
             "provenance": "mmqa_supporting_context"},
            {"modality": "image", "doc_id": "i3",
             "provenance": "mmqa_supporting_context"},
        ])
        scene = build_one(ordinal=8, seed=17, kqa=KQA, seed_record=record)
        expected = {(s["modality"], s["doc_id"])
                    for s in record["supporting_sources"]}
        actual = {
            (src["modality_hint"], src["original_doc_id"])
            for src in scene["world"]["sources"].values()
        }
        self.assertEqual(actual, expected)
        self.assertEqual(scene["macro_graph"]["source_coverage"]["assigned"], 5)
        self.assertEqual([], validate_clean_scene(scene))
        changed = copy.deepcopy(scene)
        changed["macro_graph"]["tasks"][1]["source_requirements"].pop()
        self.assertIn("mmqa_supporting_source_coverage_incomplete",
                      validate_clean_scene(changed))

    def test_source_observations_and_typed_traces_are_computational(self):
        scene = build_one(ordinal=5, seed=17, kqa=KQA, seed_record=seed())
        sources = scene["world"]["sources"]
        rows = [{
            fact["field"]: fact["value"]
            for fact in source["facts"]
        } for source in sources.values()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({item["candidate_key"] for item in rows}), 3)
        self.assertEqual(len({item["original_source_handle"] for item in rows}), 3)
        self.assertEqual(len({item["synthetic_attribute_score"] for item in rows}) >= 2, True)
        self.assertTrue(all(item["observation_provenance"] ==
                            "simulated_not_raw_perception" for item in rows))
        for feedback in scene["feedback_bank"].values():
            operations = [
                item for item in feedback["summary_trace"]
                if item["action"] == "ExecuteTypedEvidenceOperation"
            ]
            self.assertEqual(len(operations), 1)
            self.assertTrue(operations[0]["result"]["simulation_only"])
            work = next(iter(feedback["answer"].values()))
            self.assertEqual(work["operation_result"], operations[0]["result"])
        self.assertEqual([], validate_clean_scene(scene))

    def test_original_question_does_not_determine_goal_or_question(self):
        import copy
        base = seed()
        variant = copy.deepcopy(base)
        variant["question"] = "What is the unrelated result of a different comparison?"
        variant["operation"] = "temporal"
        first = compose_seed_taskfirst(
            seed_record=base, ordinal=12, run_seed=17, kqa=KQA)
        second = compose_seed_taskfirst(
            seed_record=variant, ordinal=12, run_seed=17, kqa=KQA)
        self.assertEqual(first["question"], second["question"])
        self.assertEqual(first["goal"], second["goal"])
        self.assertEqual(first["macro_graph"]["tasks"],
                         second["macro_graph"]["tasks"])
        self.assertEqual(first["macro_graph"]["edges"],
                         second["macro_graph"]["edges"])

    def test_independent_scene_goal_random_pairs_yield_different_goals(self):
        from envir.macro.goal_grammar import GOALS
        patterns = set()
        for ordinal in range(64):
            plan = compose_seed_taskfirst(
                seed_record=seed(), ordinal=ordinal, run_seed=17, kqa=KQA)
            patterns.add(plan["goal"]["intent"])
            self.assertNotEqual(plan["question"], seed()["question"])
            self.assertIn(plan["goal"]["intent"], GOALS)
            self.assertEqual(
                plan["macro_graph"]["tasks"][-1]["question_blueprint"]["pattern_id"],
                plan["goal"]["intent"],
            )
        self.assertGreaterEqual(len(patterns), 7)

    def test_nested_macro_graph_has_stages_and_typed_handoffs(self):
        plan = compose_seed_taskfirst(
            seed_record=seed(), ordinal=0, run_seed=17, kqa=KQA)
        tasks = plan["macro_graph"]["tasks"]
        work_types = [task["work_type"] for task in tasks]
        self.assertIn("reference_chain_resolution", work_types)
        from envir.macro.goal_grammar import GOALS
        self.assertIn(GOALS[plan["goal"]["intent"]]["assessment"], work_types)
        self.assertIn("evidence_consistency_review", work_types)
        self.assertIn("answer_evidence_synthesis", work_types)
        self.assertGreaterEqual(
            plan["macro_graph"]["complexity"]["macro_dependency_depth"], 4)
        self.assertGreaterEqual(len(plan["macro_graph"]["edges"]), 7)

    def test_single_source_is_not_inflated_into_fake_macro_dag(self):
        record = seed("TextQ")
        record["source_modalities"] = ["text"]
        record["supporting_sources"] = [record["supporting_sources"][-1]]
        with self.assertRaises(UnsuitableSeed):
            compose_seed_taskfirst(
                seed_record=record, ordinal=1, run_seed=17, kqa=KQA)


    def test_replay_detects_terminal_goal_and_source_provenance_tampering(self):
        import copy
        scene = build_one(ordinal=3, seed=17, kqa=KQA, seed_record=seed())
        modified = copy.deepcopy(scene)
        modified["macro_graph"]["tasks"][-1]["goal"] = "An unrelated goal"
        self.assertIn("macro_goal_terminal_blueprint_drift",
                      validate_clean_scene(modified))
        modified = copy.deepcopy(scene)
        modified["macro_graph"]["tasks"][0]["source_requirements"][0][
            "original_doc_id"] = "fabricated-source"
        self.assertIn("mmqa_original_source_binding_drift",
                      validate_clean_scene(modified))

    def test_exact_yago_context_is_optional_and_unambiguous(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from envir.macro.scene_catalog import enrich_with_yago
        item = seed()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "yago-facts.ttl"
            path.write_text("", encoding="utf-8")
            triple = SimpleNamespace(
                subject="https://yago-knowledge.org/resource/Artist_A",
                predicate="http://schema.org/occupation",
                object="https://yago-knowledge.org/resource/Painter",
                object_is_literal=False,
            )
            with patch("environment.task_construction.data_sources.yago.iter_turtle_triples",
                       return_value=iter([triple])):
                result = enrich_with_yago([item], path)
        self.assertEqual(result["matched_seeds"], 1)
        self.assertEqual(item["yago_matches"][0]["label"], "Artist A")
        self.assertEqual(item["yago_matches"][0]["facts"][0]["predicate"],
                         "occupation")

    def test_v45_author_critic_stays_graph_grounded(self):
        import json as json_module
        from envir.macro.v4_question_review import naturalize_one
        scene = build_one(ordinal=0, seed=17, kqa=KQA, seed_record=seed())
        seen = []

        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                payload = json_module.loads(user)
                seen.append(("author", payload))
                return {"question": scene["question"]}

        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                payload = json_module.loads(user)
                seen.append(("critic", payload))
                return {
                    "goal_alignment": "same",
                    "task_sufficiency": "sufficient",
                    "user_framing": "user_goal",
                    "unsupported_additions": [],
                    "verdict": "pass",
                    "naturalness": "acceptable",
                    "summary": "Same final information need, with graph support.",
                    "differences": [],
                }

        question, meta = naturalize_one(
            scene, author=Author(), critic=Critic(), max_revisions=1)
        self.assertEqual(question, scene["question"])
        self.assertEqual(meta["status"], "accepted")
        self.assertEqual(meta["review_policy_version"],
                         "v4.5_graph_primary_benchmark_style")
        self.assertIn("macro_task_graph", seen[0][1])
        self.assertNotIn("world", json_module.dumps(seen[0][1]))
        self.assertNotIn("optional_reference_hint", seen[1][1])

    def test_cli_dataset_build_manifest_and_replay(self):
        import contextlib
        import io
        from envir.macro.v4_cli import main

        kqa_record = {
            "question": "Which groups share a member?",
            "program": [
                {"function": "Find", "inputs": ["A"], "dependencies": []},
                {"function": "Relate", "inputs": ["member", "forward"],
                 "dependencies": [0]},
                {"function": "Find", "inputs": ["B"], "dependencies": []},
                {"function": "Relate", "inputs": ["member", "forward"],
                 "dependencies": [2]},
                {"function": "And", "inputs": [], "dependencies": [1, 3]},
                {"function": "Count", "inputs": [], "dependencies": [4]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kqa_path = root / "train.json"
            mmqa_path = root / "MMQA_train.jsonl"
            output = root / "output"
            kqa_path.write_text(json.dumps([kqa_record]), encoding="utf-8")
            records = []
            for i in range(4):
                records.append({
                    "qid": f"q{i}",
                    "question": f"Which person matches evidence in case {i}?",
                    "metadata": {
                        "type": "Compose(TableQ,ImageQ)",
                        "supporting_context": [
                            {"doc_id": f"t{i}", "doc_part": "table"},
                            {"doc_id": f"i{i}", "doc_part": "image"},
                        ],
                    },
                })
            mmqa_path.write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
                encoding="utf-8",
            )
            args = [
                "build", "--kqa", str(kqa_path), "--mmqa", str(mmqa_path),
                "--target", "3", "--max-mmqa-records", "4",
                "--output-dir", str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main([
                    "replay", "--scenes",
                    str(output / "private_clean_scenes_v4.jsonl"),
                ]), 0)
            summary = json.loads((output / "macro_v4_summary.json").read_text())
            self.assertEqual(summary["statistics"]["generated"], 3)
            self.assertEqual(summary["original_pattern_coverage"]
                             ["MultiModalQA"]["unique_scenario_seeds"], 4)
            manifest = [
                json.loads(line) for line in
                (output / "scenario_seed_manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(manifest), 3)
            self.assertEqual(len({row["mmqa_record_id"] for row in manifest}), 3)
            self.assertTrue(all(
                row["question"] != row["original_mmqa_question_audit_only"]
                for row in manifest
            ))
            self.assertTrue((output / "scene_bank_v4.jsonl").exists())
            bank = [
                json.loads(line) for line in
                (output / "scene_bank_v4.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(bank), 4)
            self.assertTrue(all("question" not in item for item in bank))
            patterns = json.loads((output / "goal_pattern_bank_v4.json").read_text())
            self.assertEqual(patterns["total_patterns"], 8)
            self.assertEqual(summary["statistics"]["original_mmqa_question_copies"], 0)
            with self.assertRaises(FileExistsError):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args[:args.index("--target")] + [
                        "--target", "5", "--max-mmqa-records", "4",
                        "--output-dir", str(root / "too_many")])


if __name__ == "__main__":
    unittest.main()
