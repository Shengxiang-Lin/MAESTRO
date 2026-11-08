"""Contract tests for the standalone TaskGraph-first MAESTRO environment.

Covers the entire pipeline without requiring external source downloads or LLM
API calls; in particular, MMQA original questions never drive final goals.
"""
from __future__ import annotations

import copy
import gzip
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from envir.macro.recipe_planner import UnsuitableSeed, compose_seed_taskfirst
from envir.macro.recipes import RECIPES_BY_OUTPUT
from envir.macro.builder import build_one, build_batch
from envir.macro.goal_grammar import GOALS
from envir.macro.question import QUESTION_VARIANTS
from envir.macro.scene_catalog import load_scenario_seeds, parse_composition
from envir.macro.source_contexts import load_supporting_contexts
from envir.macro.validation import validate_clean_scene


KQA = [{
    "signature": "fixture-kopl",
    "program": {"nodes": [{"function": "Find", "dependencies": []}]},
    "logical_depth": 4,
    "branch_nodes": 1,
    "merge_nodes": 1,
    "operator_count": 4,
}]


def scene(index: int = 0, n_sources: int = 3,
          composition: str = "Compare(Compose(TableQ,ImageQ),TextQ)") -> dict:
    modalities = ("table", "image", "text")
    return {
        "record_id": f"mmqa-fixture-{index}",
        "question": f"Which unrelated source answer was correct for record {index}?",
        "topic": f"Subject {index}",
        "entities": [f"Subject {index}"],
        "domain_hint": ("sports", "culture", "science", "civic")[index % 4],
        "original_question_operation": "comparison",
        "question_type": composition,
        "logical_pattern": "comparison",
        "composition": parse_composition(composition),
        "source_modalities": list(modalities[:min(n_sources, 3)]),
        "supporting_sources": [
            {"modality": modalities[i % 3], "doc_id": f"{index}-doc-{i}",
             "provenance": "mmqa_supporting_context"}
            for i in range(n_sources)
        ],
        "yago_matches": [],
    }


class BackwardRecipePlannerTests(unittest.TestCase):
    def test_multiple_recipes_produce_the_same_work_product(self):
        self.assertGreaterEqual(len(RECIPES_BY_OUTPUT["GoalReport"]), 3)
        self.assertGreaterEqual(len(RECIPES_BY_OUTPUT["GoalAssessment"]), 3)
        self.assertGreaterEqual(len(RECIPES_BY_OUTPUT["AlignedCandidates"]), 2)
        self.assertGreaterEqual(len(RECIPES_BY_OUTPUT["EvidenceReview"]), 2)

    def test_backwards_planning_does_not_depend_on_original_mmqa_qa(self):
        first = scene(8)
        altered = copy.deepcopy(first)
        altered["question"] = "Was the unrelated historical question about 1987?"
        altered["original_question_operation"] = "temporal"
        altered["logical_pattern"] = "intersection"
        altered["question_type"] = "TextQ"
        altered["composition"] = parse_composition("TextQ")
        left = compose_seed_taskfirst(
            seed_record=first, ordinal=8, run_seed=17, kqa=KQA)
        right = compose_seed_taskfirst(
            seed_record=altered, ordinal=8, run_seed=17, kqa=KQA)
        self.assertEqual(left["macro_graph"], right["macro_graph"])
        self.assertEqual(left["goal"], right["goal"])
        self.assertEqual(left["question"], right["question"])
        self.assertNotEqual(left["question"], first["question"])

    def test_goal_and_question_are_derived_from_terminal_workproduct(self):
        plan = compose_seed_taskfirst(
            seed_record=scene(1), ordinal=1, run_seed=17, kqa=KQA)
        terminal = plan["macro_graph"]["terminal_handoffs"][0]
        tasks = {t["task_id"]: t for t in plan["macro_graph"]["tasks"]}
        final = tasks[terminal["from_task"]]
        intent = plan["goal"]["intent"]
        self.assertIn(intent, GOALS)
        self.assertEqual(final["question_blueprint"]["pattern_id"], intent)
        self.assertEqual(terminal["type"], GOALS[intent]["artifact"])
        self.assertTrue(plan["question"].endswith(("?", ".")))
        self.assertNotIn("unrelated source answer", plan["question"])
        self.assertTrue(all(edge["type"] for edge in plan["macro_graph"]["edges"]))

    def test_minimal_sources_are_not_inflated(self):
        with self.assertRaises(UnsuitableSeed):
            compose_seed_taskfirst(
                seed_record=scene(4, 1), ordinal=4, run_seed=17, kqa=KQA)

    def test_replay_and_exact_original_source_coverage(self):
        payload = scene(11, 9)
        result = build_one(ordinal=11, seed=17, kqa=KQA,
                           seed_record=payload)
        self.assertEqual([], validate_clean_scene(result))
        expected = {
            (item["modality"], item["doc_id"])
            for item in payload["supporting_sources"]
        }
        observed = {
            (doc["modality_hint"], doc["original_doc_id"])
            for doc in result["world"]["sources"].values()
        }
        self.assertEqual(expected, observed)
        self.assertTrue(3 <= len(result["macro_graph"]["tasks"]) <= 9)
        for feedback in result["feedback_bank"].values():
            self.assertEqual(len([
                t for t in feedback["summary_trace"]
                if t["action"] == "ExecuteTypedEvidenceOperation"
            ]), 1)

    def test_100_scenes_have_recipe_and_topology_diversity(self):
        inputs = [
            scene(i, n_sources=(i % 8) + 2,
                  composition=(
                    "Compare(Compose(TableQ,ImageQ),TextQ)"
                    if i % 2 else "ImageListQ"
                  ))
            for i in range(150)
        ]
        results, summary = build_batch(
            kqa=KQA, scenario_seeds=inputs, target=100, seed=17)
        self.assertEqual(summary["generated"], 100)
        self.assertEqual(summary["missing_original_supporting_sources"], 0)
        self.assertEqual(summary["original_mmqa_question_copies"], 0)
        self.assertGreater(len(summary["macro_work_types"]), 8)
        self.assertGreaterEqual(summary["macro_graph_signatures"], 12)
        self.assertGreaterEqual(len(summary["topology_distribution"]), 2)
        self.assertGreaterEqual(summary["unlabeled_dag_topologies"], 7)
        self.assertGreaterEqual(summary["unique_recipe_sequences"], 12)
        self.assertGreaterEqual(summary["unique_recipe_ids"], 8)
        self.assertEqual(summary["recipe_selection"],
                         "backward_recursive_typed_workproduct_grammar")
        self.assertTrue(all(
            3 <= len(x["macro_graph"]["tasks"]) <= 9 for x in results
        ))
        self.assertTrue(all(
            x["macro_graph"]["complexity"]["macro_dependency_depth"] >= 2
            for x in results
        ))
        # Every task has a traceable recipe rather than a forced MMQA operator.
        self.assertTrue(all(
            len(x["macro_graph"]["provenance_recipe_routes"]) ==
            len(x["macro_graph"]["tasks"]) for x in results
        ))
        self.assertEqual([], validate_clean_scene(results[0]))

    def test_question_templates_have_multiple_phrasings_per_pattern(self):
        self.assertEqual(set(QUESTION_VARIANTS), set(GOALS))
        self.assertTrue(all(len(variants) >= 2
                            for variants in QUESTION_VARIANTS.values()))

    def test_scene_catalog_preserves_sources_but_never_copies_answers(self):
        record = {
            "qid": "mmqa-001",
            "question": "What did Example Artist produce?",
            "answers": [{"answer": "SECRET_GOLD_ANSWER"}],
            "wiki_entities_in_question": ["Example Artist"],
            "metadata": {
                "type": "Compose(TableQ,ImageQ)",
                "supporting_context": [
                    {"doc_id": "t-001", "doc_part": "table"},
                    {"doc_id": "i-002", "doc_part": "image"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "MMQA_train.jsonl.gz"
            with gzip.open(p, "wt", encoding="utf-8") as writer:
                writer.write(json.dumps(record) + "\n")
            records, stats = load_scenario_seeds(p)
        self.assertEqual(stats["unique_scenario_seeds"], 1)
        self.assertEqual(records[0]["topic"], "Example Artist")
        self.assertEqual(len(records[0]["supporting_sources"]), 2)
        self.assertNotIn("SECRET_GOLD_ANSWER", json.dumps(records))

    def test_original_contexts_are_provenance_labeled(self):
        record = scene(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "MMQA_tables.jsonl"
            p.write_text(json.dumps({
                "id": "2-doc-0",
                "table": {"table_name": "Source table",
                          "header": [{"column_name": "Score"}],
                          "table_rows": [[{"text": "62"}]]},
            }) + "\n", encoding="utf-8")
            contexts, stats = load_supporting_contexts([record], directory)
        self.assertEqual(stats["available_context_excerpts"], 1)
        self.assertEqual(contexts[("table", "2-doc-0")]["provenance"],
                         "original_mmqa_table_text")

    def test_author_critic_preserves_graph_and_uses_new_reference(self):
        from envir.macro.question_review import naturalize_one
        payload = scene(12)
        result = build_one(
            ordinal=12, seed=17, kqa=KQA, seed_record=payload)
        initial = result["question"]
        calls = []

        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                request = json.loads(user)
                calls.append(("author", request))
                return {"question": initial}

        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                request = json.loads(user)
                calls.append(("critic", request))
                return {
                    "objective_equivalent": "yes",
                    "solution_preserved": "yes",
                    "required_work_preserved": "yes",
                    "source_faithful": "yes",
                    "answer_equivalent": "yes",
                    "user_framing": "user_goal",
                    "unsupported_additions": [],
                    "verdict": "pass",
                    "naturalness": "acceptable",
                    "summary": "The question preserves the frozen contract.",
                    "differences": [],
                }

        natural, meta = naturalize_one(
            result, author=Author(), critic=Critic(), max_revisions=0)
        self.assertEqual(natural, initial)
        self.assertEqual(meta["status"], "accepted")
        self.assertNotEqual(initial, payload["question"])
        self.assertEqual(
            calls[0][1]["semantic_authority"],
            "frozen_semantic_contract_and_macro_task_graph",
        )
        self.assertIn("macro_task_graph", calls[0][1])
        self.assertIn("semantic_contract", calls[0][1])
        self.assertIn("semantic_contract_sha256", meta)
        self.assertEqual(meta["review_policy_version"],
                         "taskfirst_semantic_contract_v1")
        self.assertNotIn("world", calls[0][1])
        self.assertNotIn("source_priors", json.dumps(calls[0][1]))

    def test_semantic_contract_is_derived_without_original_mmqa_question(self):
        from envir.macro.semantic_contract import (
            semantic_contract, semantic_surface_issues,
        )
        original = build_one(
            ordinal=13, seed=17, kqa=KQA, seed_record=scene(13))
        changed = copy.deepcopy(original)
        changed["source_priors"]["MultiModalQA"]["original_question"] = (
            "Which entirely different historical answer is correct?"
        )
        self.assertEqual(semantic_contract(original), semantic_contract(changed))
        contract = semantic_contract(original)
        self.assertEqual(contract["contract_version"], "question_semantics_v1")
        self.assertNotIn("original_question", json.dumps(contract))
        self.assertNotIn("world", contract)
        self.assertEqual(
            len(contract["source_inventory"]),
            len(original["public"]["available_source_descriptions"]),
        )
        self.assertIn(
            "invented_personal_or_attached_source",
            semantic_surface_issues(
                "Which records from my account summary indicate the highest score?",
                contract,
            ),
        )

    def test_strict_critic_rejects_false_positive_and_uncertainty(self):
        from envir.macro.question_review import (
            _validate_review, CRITIC_SCHEMA_SEMANTIC, SEMANTIC_REVIEW_POLICY,
        )
        valid = {
            "objective_equivalent": "yes",
            "solution_preserved": "yes",
            "required_work_preserved": "yes",
            "source_faithful": "yes",
            "answer_equivalent": "yes",
            "user_framing": "user_goal",
            "unsupported_additions": [],
            "verdict": "pass",
            "naturalness": "acceptable",
            "summary": "Identical operation and answer under the same evidence.",
            "differences": [],
        }
        self.assertIn("source_faithful", CRITIC_SCHEMA_SEMANTIC["required"])
        self.assertEqual(
            _validate_review(valid, policy=SEMANTIC_REVIEW_POLICY)["verdict"],
            "pass",
        )
        rejected = dict(valid)
        rejected["source_faithful"] = "no"
        rejected["unsupported_additions"] = ["my account"]
        with self.assertRaisesRegex(ValueError, "semantic_pass_contradicts"):
            _validate_review(rejected, policy=SEMANTIC_REVIEW_POLICY)
        uncertain = dict(valid)
        uncertain["objective_equivalent"] = "uncertain"
        uncertain["verdict"] = "revise"
        uncertain["differences"] = [{
            "reference_excerpt": "",
            "candidate_excerpt": "perhaps",
            "problem": "Operation cannot be established",
            "minimal_fix": "Use the contract operation",
        }]
        with self.assertRaisesRegex(ValueError, "uncertainty_cannot"):
            _validate_review(uncertain, policy=SEMANTIC_REVIEW_POLICY)

    def test_surface_guard_rejects_invented_account_without_critic_call(self):
        from envir.macro.question_review import naturalize_one
        original = build_one(
            ordinal=19, seed=17, kqa=KQA, seed_record=scene(19))
        attempts = []

        class BadAuthor:
            def complete_json(self, system, user, *, response_schema=None):
                attempts.append("author")
                return {"question": (
                    "Using my account summary for Subject 19, which "
                    "candidate should be selected according to the records?"
                )}

        class CriticNever:
            def complete_json(self, *args, **kwargs):
                attempts.append("critic")
                raise AssertionError("the semantic guard should reject first")

        question, metadata = naturalize_one(
            original, author=BadAuthor(), critic=CriticNever(), max_revisions=0)
        self.assertEqual(question, original["question"])
        self.assertEqual(metadata["status"], "fallback")
        self.assertEqual(attempts, ["author"])
        self.assertIn(
            "invented_personal_or_attached_source",
            metadata["audit"][0]["deterministic_guards"],
        )

    def test_frozen_semantics_fingerprint_catches_contract_drift(self):
        from envir.macro.question_review import naturalize_one
        from envir.macro.validation import validate_clean_scene
        from envir.macro.question_review import _fingerprint, semantic_contract
        original = build_one(
            ordinal=17, seed=17, kqa=KQA, seed_record=scene(17))
        natural = copy.deepcopy(original)
        reference = natural["question"]
        semantic = semantic_contract(natural)
        natural["question_authoring"] = {
            "review_policy_version": "taskfirst_semantic_contract_v1",
            "pipeline": "taskfirst_question_semantic_author_critic_v1",
            "semantic_contract_sha256": "wrong-hash",
            "macro_task_view_sha256": "dummy",
            "macro_graph_sha256": _fingerprint(natural["macro_graph"]),
            "reference_sha256": _fingerprint(reference),
            "reference_question": reference,
            "final_question": reference,
            "status": "fallback", "reason": "author_error",
            "accepted_review": None, "audit": [],
            "semantic_equivalence_certified": False,
        }
        self.assertIn(
            "semantic_contract_fingerprint_mismatch",
            validate_clean_scene(natural),
        )
        self.assertEqual(semantic["authorized_topic"], "Subject 17")

    def test_cli_build_and_replay(self):
        from envir.macro.cli import main
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kqa = root / "train.json"
            mmqa = root / "MMQA_train.jsonl"
            output = root / "output"
            kqa.write_text(json.dumps([kqa_record]), encoding="utf-8")
            rows = []
            for i in range(10):
                rows.append({
                    "qid": f"q{i}",
                    "question": f"What happened to Some Entity in year {2000 + i}?",
                    "wiki_entities_in_question": [f"Subject {i}"],
                    "metadata": {
                        "type": "Compose(TableQ,ImageQ)",
                        "supporting_context": [
                            {"doc_id": f"t{i}", "doc_part": "table"},
                            {"doc_id": f"v{i}", "doc_part": "image"},
                        ],
                    },
                })
            mmqa.write_text(
                "\n".join(json.dumps(x) for x in rows) + "\n",
                encoding="utf-8",
            )
            command = [
                "build", "--kqa", str(kqa), "--mmqa", str(mmqa),
                "--target", "5", "--max-mmqa-records", "10",
                "--output-dir", str(output),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
                self.assertEqual(main([
                    "replay", "--scenes", str(output / "scenes.jsonl")
                ]), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["statistics"]["generated"], 5)
            self.assertEqual(summary["statistics"]["original_mmqa_question_copies"], 0)
            self.assertTrue((output / "scene_bank.jsonl").exists())
            self.assertTrue((output / "goal_bank.json").exists())
            self.assertTrue((output / "recipe_bank.json").exists())
            self.assertEqual(len((output / "source_manifest.jsonl")
                                 .read_text().splitlines()), 5)
            with self.assertRaises(FileExistsError):
                with redirect_stdout(io.StringIO()):
                    main(command)


if __name__ == "__main__":
    unittest.main()
