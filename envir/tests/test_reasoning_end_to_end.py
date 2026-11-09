"""End-to-end tests for V2: the real CLI, executable dependencies and audit."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from envir.macro.cli import parser
from envir.macro.program_ir import MissingSourceError, ProgramError
from envir.reasoning.spec import (
    DIFFICULTIES, FAMILIES, difficulty_sequence, make_spec, spec_issues,
)
from envir.reasoning.world import build_scene
from envir.reasoning.question import guard_question, reference_question
from envir.reasoning.compiler import (
    V2Session, compile_macro_tasks, oracle_schedule, run_macro_tasks,
)
from envir.reasoning.quality import (
    compile_and_validate, validate_replay, audit_scene,
    weak_single_source_baseline,
)
from envir.reasoning.pipeline import build_v2, replay_v2, audit_v2, naturalize_v2, read_jsonl


def seed(ordinal=0):
    return {
        "record_id": f"mmqa-{ordinal:05d}",
        "topic": f"Fictional Topic {ordinal}",
        "domain_hint": "sports",
        "source_modalities": ["table", "image", "text"],
        "supporting_sources": [],
    }


def scene(*, difficulty="medium", ordinal=0, family=None):
    spec = make_spec(
        seed_record=seed(ordinal), ordinal=ordinal, run_seed=17,
        difficulty=difficulty,
    )
    if family is not None:
        spec["family"] = family
        spec["requirements"]["answer_kind"] = (
            "count" if family == "coverage_count" else
            "category_counts" if family == "category_distribution" else
            "single_entry" if family in {"priority", "chronology"} else
            "entry_list"
        )
    return compile_and_validate(spec, reference_question(spec))


class V2ReasoningTests(unittest.TestCase):
    def test_real_program_compiles_into_macro_tasks(self):
        for difficulty in DIFFICULTIES:
            data = scene(difficulty=difficulty)
            self.assertEqual(validate_replay(data)["status"], "valid")
            self.assertEqual(data["gold_answer"], oracle_schedule(data)["answer"])
            self.assertEqual(data["gold_answer"], run_macro_tasks(
                data["macro_graph"], sources=data["world"]["sources"]
            )["answer"])
            self.assertEqual(
                data["macro_graph"]["macro_task_count"],
                len(data["gold_program"]["nodes"]),
            )

    def test_each_difficulty_meets_published_structure(self):
        expectations = {
            "easy": (1, 0),
            "medium": (3, 2),
            "hard": (4, 3),
        }
        for difficulty, (n_sources, n_joins) in expectations.items():
            data = scene(difficulty=difficulty)
            self.assertEqual(len(data["world"]["sources"]), n_sources)
            self.assertGreaterEqual(
                data["difficulty_audit"]["reasoning"]["join_count"], n_joins)
            self.assertTrue(data["difficulty_audit"]["replay_exact"])

    def test_six_families_across_medium_and_hard(self):
        for difficulty in ("medium", "hard"):
            for family in FAMILIES:
                with self.subTest(difficulty=difficulty, family=family):
                    data = scene(difficulty=difficulty, family=family)
                    self.assertEqual(validate_replay(data)["status"], "valid")
                    report = data["difficulty_audit"]
                    self.assertFalse(report["one_source_baseline"]["exact_match"])
                    self.assertTrue(all(
                        item["changing_source_changes_answer"]
                        for item in report["counterfactual_source_tests"].values()
                    ))

    def test_media_modalities_include_simulated_video_and_audio(self):
        modalities = set()
        for i in range(30):
            data = scene(difficulty="medium", ordinal=i)
            source = data["world"]["sources"]["annotations"]
            modalities.add(source["modality"])
            self.assertIn("synthetic_", source["representation"])
            self.assertFalse(source["original_mmqa_handle_used"])
        self.assertIn("video", modalities)
        self.assertIn("audio", modalities)
        self.assertIn("image", modalities)

    def test_session_requires_source_acquisition_before_scan(self):
        data = scene(difficulty="hard")
        session = V2Session(data)
        first = data["macro_graph"]["tasks"][0]
        with self.assertRaises(MissingSourceError):
            session.run_task(first["task_id"])
        result = session.read_source("register")
        self.assertIn("observations", result)
        result["observations"].clear()
        self.assertTrue(data["world"]["sources"]["register"]["observations"])
        session.run_task(first["task_id"])
        with self.assertRaises(ProgramError):
            session.run_task(first["task_id"])
        check = session.verify(first["task_id"])
        self.assertTrue(check["valid_evidence_closure"])
        self.assertFalse(check["independent_source_verification"])

    def test_direct_world_source_leak_is_not_public(self):
        data = scene(difficulty="hard")
        published = data["public"]
        for field in ("gold_answer", "world", "gold_program",
                      "entities", "task_feedback", "macro_graph"):
            self.assertNotIn(field, published)
        serialized = json.dumps(published)
        self.assertNotIn("F001", serialized)
        self.assertNotIn("Entry-01", serialized)
        self.assertNotIn("fictional-001", serialized)

    def test_ungranted_source_cannot_be_used_by_non_scan_task(self):
        data = scene(difficulty="medium")
        graph = copy.deepcopy(data["macro_graph"])
        node = next(x for x in graph["tasks"] if x["operation"] == "join")
        node["source_ids"] = ["glossary"]
        with self.assertRaises(ProgramError):
            run_macro_tasks(graph, sources=data["world"]["sources"])

    def test_change_source_facts_invalidates_replay(self):
        data = scene(difficulty="medium")
        bad = copy.deepcopy(data)
        record = bad["world"]["sources"]["register"]["observations"][0]["values"]
        record["active"] = not record["active"]
        result = validate_replay(bad)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("world_not_reproducible_from_frozen_spec", result["issues"])

    def test_bad_macro_task_dependency_invalidates_replay(self):
        data = scene(difficulty="hard")
        bad = copy.deepcopy(data)
        dependent = next(t for t in bad["macro_graph"]["tasks"]
                         if t["depends_on"])
        dependent["depends_on"] = []
        result = validate_replay(bad)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("macro_graph_not_program_compiled", result["issues"])

    def test_spec_guards_and_question_guards(self):
        data = scene(difficulty="hard", family="coverage_count")
        question = data["question"]
        self.assertEqual(guard_question(question, data["problem_spec"]), [])
        self.assertTrue(guard_question(question.replace("clearance", "extra").replace("cleared", "selected"),
                                       data["problem_spec"]))
        spec = copy.deepcopy(data["problem_spec"])
        spec["sources"] = []
        self.assertIn("wrong_source_count", spec_issues(spec))

    def test_difficulty_mix_exact_counts(self):
        values = difficulty_sequence(target=101, mix=(20, 50, 30), run_seed=17)
        self.assertEqual(len(values), 101)
        self.assertEqual(len([x for x in values if x == "medium"]), 51)
        with self.assertRaises(ValueError):
            difficulty_sequence(target=100, mix=(20, 30, 30), run_seed=17)

    def test_old_cli_remains_available_with_v2_commands(self):
        for command in ("build", "naturalize", "realize-world", "replay",
                        "build-v2", "naturalize-v2", "replay-v2", "audit-v2"):
            self.assertIn(command, parser()._subparsers._group_actions[0].choices)


class GenerationQualityRegressionTests(unittest.TestCase):
    def test_v3_varies_actual_answers_and_program_structures(self):
        variants = []
        for ordinal in range(60):
            spec = make_spec(
                seed_record=seed(ordinal), ordinal=ordinal,
                run_seed=17, difficulty=(
                    "easy" if ordinal < 15
                    else "medium" if ordinal < 40
                    else "hard"
                ),
            )
            self.assertEqual(spec["version"], "maestro.problem_spec.v3")
            sample = compile_and_validate(spec, reference_question(spec))
            self.assertEqual(validate_replay(sample)["status"], "valid")
            self.assertEqual(sample["difficulty_audit"]["status"], "valid")
            variants.append(sample)
        import collections
        structural = {
            tuple((node["op"], tuple(node["inputs"]))
                  for node in row["gold_program"]["nodes"])
            for row in variants
        }
        answers = {
            json.dumps(row["gold_answer"], ensure_ascii=False, sort_keys=True)
            for row in variants
        }
        self.assertGreater(len(structural), 14)
        self.assertGreater(len(answers), 15)
        self.assertTrue(all(
            all(test["changing_source_changes_answer"]
                for test in row["difficulty_audit"][
                    "counterfactual_source_tests"].values())
            for row in variants
        ))

    def test_hard_priority_answers_are_not_always_entry_one(self):
        selected = set()
        for ordinal in range(40):
            sample = scene(
                difficulty="hard", ordinal=ordinal, family="priority"
            )
            selected.add(sample["gold_answer"][0]["entry_label"])
        self.assertGreaterEqual(len(selected), 4)

    def test_private_entity_registry_matches_shuffled_source_rows(self):
        sample = scene(difficulty="hard", ordinal=33)
        register = {
            x["values"]["entry_label"]: x["values"]["label_code"]
            for x in sample["world"]["sources"]["register"]["observations"]
        }
        glossary = {
            x["values"]["label_code"]: x["values"]["category_name"]
            for x in sample["world"]["sources"]["glossary"]["observations"]
        }
        for entity in sample["world"]["entities"]:
            self.assertEqual(
                entity["category_name"],
                glossary[register[entity["entry_label"]]],
            )

    def test_legacy_frozen_specs_remain_replayable(self):
        spec = make_spec(
            seed_record=seed(9), ordinal=9, run_seed=17, difficulty="hard"
        )
        spec["version"] = "maestro.problem_spec.v2"
        frozen = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(
            frozen["world"]["version"], "maestro.source_observation_world.v2"
        )
        self.assertEqual(
            frozen["gold_program"]["version"], "maestro.reasoning_program.v2"
        )
        self.assertEqual(validate_replay(frozen)["status"], "valid")

    def test_frozen_reference_is_visible_to_question_author_and_critic(self):
        from envir.reasoning.question import naturalize_one
        spec = make_spec(
            seed_record=seed(12), ordinal=12,
            run_seed=17, difficulty="hard"
        )
        reference = reference_question(spec)
        seen = []
        class EchoAuthor:
            def complete_json(self, system_prompt, user_prompt):
                payload = json.loads(user_prompt)
                self_ref = payload["contract"]["frozen_reference_question"]
                seen.append(("author", self_ref))
                return {"question": self_ref}
        class AcceptingCritic:
            def complete_json(self, system_prompt, user_prompt):
                payload = json.loads(user_prompt)
                seen.append(("critic", payload[
                    "frozen_contract"]["frozen_reference_question"]))
                return {
                    "verdict": "pass",
                    "all_constraints_preserved": True,
                    "no_unlicensed_facts": True,
                    "same_answer_scope": True,
                    "natural": True,
                    "reason": "mock review of reference wording",
                }
        q, audit = naturalize_one(
            spec, author=EchoAuthor(), critic=AcceptingCritic(),
            max_revisions=0,
        )
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(q, reference)
        self.assertEqual(seen, [("author", reference), ("critic", reference)])

    def test_bad_question_reports_missing_mark_and_preview(self):
        from envir.reasoning.question import naturalize_one
        spec = make_spec(
            seed_record=seed(11), ordinal=11,
            run_seed=17, difficulty="medium"
        )
        class BadAuthor:
            def complete_json(self, *_args):
                return {"question": "Entries"}
        class NeverCritic:
            def complete_json(self, *_args):
                raise AssertionError("guard should reject before critic")
        q, audit = naturalize_one(
            spec, author=BadAuthor(), critic=NeverCritic(),
            max_revisions=0
        )
        self.assertEqual(audit["status"], "fallback")
        failure = audit["failures"][0]
        self.assertEqual(failure["candidate_preview"], "Entries")
        self.assertIn("question_missing_interrogative", failure["errors"])
        self.assertIn("question_length_out_of_bounds", failure["errors"])


class V2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.legacy = self.root / "legacy_scenes.jsonl"
        with self.legacy.open("w", encoding="utf-8") as handle:
            for i in range(24):
                handle.write(json.dumps({
                    "task_id": f"macro-{i}",
                    "scenario": {
                        "topic": f"Fictional Topic {i}",
                        "domain": "sports",
                    },
                    "source_priors": {"MultiModalQA": {
                        "record_id": f"mmqa-{i}",
                        "source_modalities": ["table", "image", "text"],
                        "supporting_sources": [],
                    }},
                }) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, target=12):
        output = self.root / "v2"
        return build_v2(Namespace(
            scenes=self.legacy, mmqa=None, output_dir=output,
            target=target, max_records=24, seed=17,
            difficulty_mix="25,50,25", overwrite=False,
        ))

    def test_missing_example_scene_path_gives_actionable_alternative(self):
        path = self.root / "taskfirst_recursive_1000" / "scenes.jsonl"
        with self.assertRaisesRegex(FileNotFoundError, "--mmqa") as caught:
            read_jsonl(path)
        self.assertIn(str(path), str(caught.exception))
        self.assertIn("find envir environment", str(caught.exception))

    def test_batch_outputs_and_strict_replay(self):
        summary = self.build(12)
        self.assertEqual(summary["status"], "done")
        self.assertEqual(summary["generated"], 12)
        self.assertEqual(sum(summary["difficulty_target_distribution"].values()), 12)
        output = self.root / "v2" / "scenes.jsonl"
        replay = replay_v2(Namespace(scenes=output, max_records=None))
        self.assertEqual(replay["status"], "valid")
        self.assertEqual(replay["checked"], 12)
        aggregate = audit_v2(Namespace(scenes=output, max_records=None))
        self.assertEqual(aggregate["replay_pass"], 12)
        self.assertEqual(aggregate["one_source_shortcut_successes"], 3)
        self.assertEqual(aggregate["counterfactual_full_pass"], 12)
        self.assertTrue((self.root / "v2" / "questions.jsonl").exists())
        self.assertTrue((self.root / "v2" / "difficulty_audit.jsonl").exists())

    def test_pre_world_scenario_author_critic_retains_frozen_contract(self):
        from envir.reasoning.scenario import enrich_one
        spec = make_spec(seed_record=seed(4), ordinal=4,
                         run_seed=17, difficulty="hard")
        frozen_program = copy.deepcopy(spec["requirements"])
        frozen_sources = [(x["source_id"], x["modality"]) for x in spec["sources"]]
        proposal = {
            "scenario": {
                "setting": f"a fictional staffing review involving {spec['scenario']['topic']}",
                "user_role": "audit coordinator",
                "purpose": "reconcile independent simulated reports",
                "motivation": "prepare a traceable staffing summary",
                "entity_kind": "recorded roster entries",
            },
            "source_titles": [
                {"source_id": x["source_id"], "title": x["title"] + " copy"}
                for x in spec["sources"]
            ],
        }
        class Author:
            def complete_json(self, *_args, **_kwargs):
                return proposal
        class Critic:
            def complete_json(self, *_args, **_kwargs):
                return {
                    "verdict": "pass", "topic_coherent": True,
                    "same_relation_semantics": True,
                    "source_faithful": True, "no_unlicensed_facts": True,
                    "reason": "fixture",
                }
        updated, review = enrich_one(spec, author=Author(), critic=Critic(),
                                     max_revisions=0)
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(updated["requirements"], frozen_program)
        self.assertEqual(
            [(x["source_id"], x["modality"]) for x in updated["sources"]],
            frozen_sources,
        )
        self.assertEqual(
            validate_replay(compile_and_validate(
                updated, reference_question(updated)))["status"], "valid",
        )

    def test_does_not_overwrite_existing_outputs(self):
        self.build(3)
        with self.assertRaises(FileExistsError):
            self.build(3)

    def test_llm_failure_falls_back_without_world_mutation(self):
        self.build(4)
        class BrokenClient:
            def complete_json(self, *_args, **_kwargs):
                raise RuntimeError("intentional artificial backend error")
        src = self.root / "v2" / "scenes.jsonl"
        out = self.root / "naturalized"
        with patch("envir.reasoning.pipeline._llm_clients",
                   return_value=(BrokenClient(), BrokenClient())):
            summary = naturalize_v2(Namespace(
                scenes=src, output_dir=out, max_records=4,
                question_workers=2, max_revisions=1,
                question_api_base="mock", question_model="mock",
                question_critic_api_base=None,
                question_critic_model=None,
                question_api_key_env="OPENAI_API_KEY",
                question_timeout=4, overwrite=False,
            ))
        self.assertEqual(summary["question_results"].get("fallback"), 4)
        self.assertEqual(replay_v2(Namespace(
            scenes=out / "scenes.jsonl", max_records=None
        ))["status"], "valid")


class CanonicalReasoningWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scenes = self.root / "source_bank.jsonl"
        with self.scenes.open("w", encoding="utf-8") as f:
            for ordinal in range(15):
                f.write(json.dumps({
                    "scenario": {
                        "topic": f"Fictional Collection {ordinal}",
                        "domain": "books",
                    },
                    "task_id": f"source-{ordinal}",
                    "source_priors": {
                        "MultiModalQA": {
                            "record_id": f"topic-{ordinal}",
                            "source_modalities": ["text", "image", "table"],
                            "supporting_sources": [],
                        },
                    },
                }) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_canonical_generate_quality_validate_and_legacy_aliases(self):
        from envir.reasoning.workflow import generate, quality
        args = parser().parse_args([
            "generate", "--scenes", str(self.scenes),
            "--output-dir", str(self.root / "new_run"),
            "--target", "12", "--max-records", "15",
            "--difficulty-mix", "25,50,25", "--no-progress",
        ])
        summary = generate(args)
        self.assertEqual(summary["status"], "done")
        self.assertEqual(summary["generated"], 12)
        self.assertEqual(summary["question_authoring"], {"off": 12})
        bank = self.root / "new_run" / "constructed" / "scenes.jsonl"
        public = self.root / "new_run" / "constructed" / "questions.jsonl"
        report_file = self.root / "new_run" / "quality_report.json"
        self.assertTrue(bank.is_file())
        self.assertTrue(public.is_file())
        self.assertTrue(report_file.is_file())
        stored = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(stored["checked"], 12)
        self.assertEqual(stored["replay_pass"], 12)
        self.assertEqual(stored["replay_fail"], 0)
        self.assertTrue(stored["reasoning"]["unique_structural_program_signatures"])
        self.assertTrue(stored["question_surface"]["unique_abstract_shapes"])
        self.assertEqual(
            replay_v2(Namespace(scenes=bank, max_records=None,
                                no_progress=True))["status"], "valid"
        )
        self.assertEqual(quality(parser().parse_args([
            "quality", "--scenes", str(bank), "--no-progress",
        ]))["status"], "valid")
        for alias in ("build-v2", "replay-v2", "audit-v2",
                      "naturalize-v2", "generate", "quality",
                      "validate", "refine-questions"):
            self.assertIn(
                alias, parser()._subparsers._group_actions[0].choices
            )
        with self.assertRaises(FileExistsError):
            generate(args)

    def test_partial_question_configuration_fails_before_generation(self):
        from envir.reasoning.workflow import generate
        args = parser().parse_args([
            "generate", "--scenes", str(self.scenes),
            "--output-dir", str(self.root / "partial"),
            "--target", "3", "--question-model", "local-test",
            "--no-progress",
        ])
        with self.assertRaisesRegex(ValueError, "both"):
            generate(args)
        self.assertFalse((self.root / "partial").exists())

    def test_parallel_progress_tracks_actual_completions_without_reordering(self):
        from envir.reasoning.pipeline import _parallel_ordered
        from time import sleep

        completed = []
        class FakeBar:
            def __init__(self, *args, **kwargs):
                self.total = kwargs["total"]
                self.updates = 0
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def update(self, n):
                self.updates += n

        bars = []
        def progress(*args, **kwargs):
            instance = FakeBar(*args, **kwargs)
            bars.append(instance)
            return instance

        def work(x):
            if x == 0:
                sleep(0.04)
            completed.append(x)
            return x * 10

        with patch("envir.reasoning.pipeline.tqdm", progress):
            results = _parallel_ordered(
                [0, 1, 2], work, workers=3,
                desc="Scenario Author/Critic", no_progress=False
            )
        self.assertEqual(results, [0, 10, 20])
        self.assertEqual(bars[0].total, 3)
        self.assertEqual(bars[0].updates, 3)
        self.assertNotEqual(completed, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
