"""Regression tests for the ONLY maintained Stage-1 Question generator."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.__main__ import parser
from environment.config import DEFAULT_CONFIG, args_from_config, load_config
from environment.generate_question.spec import make_spec
from environment.generate_question.world import compile_program, generate_world
from environment.generate_question.question import reference_question, guard_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.macro_contract import macro_question_contract
from environment.generate_question.surface_quality import (
    question_output_issues, unsupported_macro_predicates,
)
from environment.generate_question.stages import (
    construct_scenes, read_jsonl, review_questions, _jsonl,
)
from environment.generate_question.pipeline import quality_report, generate_questions


def spec(ordinal=0, difficulty="medium", family=None):
    data = make_spec(
        seed_record={
            "record_id": f"test-record-{ordinal}",
            "topic": f"Fictional Collection {ordinal}",
            "domain_hint": "science",
            "source_modalities": ["image", "video", "audio", "table", "text"],
        },
        ordinal=ordinal, run_seed=17, difficulty=difficulty,
    )
    if family is not None:
        data["family"] = family
        data["reasoning_pattern"] = {
            "window_alignment": "time_window",
            "status_disagreement": "status_disagreement",
            "threshold_eligibility": "dynamic_threshold",
        }.get(family, "approval")
        data["requirements"]["answer_kind"] = (
            "count" if family == "coverage_count" else
            "category_counts" if family == "category_distribution" else
            "single_entry" if family in {"priority", "chronology"} else
            "entry_evidence" if family == "evidence_reconcile" else
            "entry_list"
        )
    return data


def scene(ordinal=0, difficulty="medium", family=None):
    frozen = spec(ordinal, difficulty, family)
    return compile_and_validate(frozen, reference_question(frozen))


class ConfigAndCliTests(unittest.TestCase):
    def test_yaml_is_canonical_and_fully_resolved(self):
        values = load_config(DEFAULT_CONFIG)
        args = args_from_config(values)
        self.assertEqual(args.target, 100)
        self.assertEqual(args.difficulty_mix, "17,33,50")
        self.assertFalse(hasattr(args, "quota_refill"))
        self.assertFalse(hasattr(args, "max_candidate_multiplier"))
        self.assertTrue(args.mmqa.is_absolute())
        self.assertEqual(args.scenario_workers, 8)
        self.assertEqual(args.question_workers, 8)
        self.assertIsNone(args.question_api_key_env)

    def test_cli_exposes_one_generation_command(self):
        sub = parser()._subparsers._group_actions[0].choices
        self.assertEqual(
            set(sub), {"generate-questions", "quality", "implement-subtasks", "naturalize-subtasks", "generate-untrusted-turns", "run"}
        )
        self.assertNotIn("build-v2", sub)
        self.assertNotIn("refine-questions", sub)

    def test_config_cli_override_and_unknown_keys(self):
        cfg = load_config(DEFAULT_CONFIG)
        args = args_from_config(cfg, target=15, output_dir="environment/data/test_15")
        self.assertEqual(args.target, 15)
        self.assertTrue(str(args.output_dir).endswith("environment/data/test_15"))
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "config.yaml"
            data = DEFAULT_CONFIG.read_text(encoding="utf-8")
            file.write_text(data + "\nunknown_top_level: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_config(file)


class ContractTests(unittest.TestCase):
    def test_compiled_graph_executes_and_replays_across_semantics(self):
        patterns = set()
        for ordinal in range(40):
            frozen = spec(ordinal, "hard")
            result = compile_and_validate(frozen, reference_question(frozen))
            self.assertEqual(validate_replay(result)["status"], "valid")
            self.assertLessEqual(result["macro_graph"]["macro_task_count"], 9)
            self.assertEqual(
                macro_question_contract(frozen, result["macro_graph"])["answer_kind"],
                frozen["requirements"]["answer_kind"],
            )
            patterns.add(frozen["reasoning_pattern"])
        self.assertEqual(patterns, {"macro_first_v2"})

    def test_explicit_no_archived_is_not_allowed_without_a_graph_filter(self):
        entry = scene(4, "easy", "priority")
        question = reference_question(entry["problem_spec"])
        bad = question + " Ensure no archived records are included."
        self.assertIn(
            "unlicensed_archived_filter",
            guard_question(bad, entry["problem_spec"], entry["macro_graph"]),
        )
        self.assertNotIn(
            "unlicensed_archived_filter",
            guard_question(question, entry["problem_spec"], entry["macro_graph"]),
        )

    def test_other_unsupported_filter_fields_are_rejected(self):
        entry = scene(6, "easy", "reconcile")
        original = reference_question(entry["problem_spec"])
        for added in (
            "Exclude unpublished records.",
            "Keep only certified entries.",
            "No funded records are eligible.",
            "Remove deleted entries.",
        ):
            with self.subTest(added=added):
                issues = unsupported_macro_predicates(
                    original + " " + added,
                    entry["problem_spec"], entry["macro_graph"],
                )
                self.assertTrue(issues)

    def test_active_is_not_equivalent_to_archived(self):
        entry = scene(11, "easy", "reconcile")
        original = reference_question(entry["problem_spec"])
        self.assertEqual(
            unsupported_macro_predicates(
                original, entry["problem_spec"], entry["macro_graph"]),
            [],
        )
        self.assertIn(
            "unlicensed_archived_filter",
            guard_question(
                original + " Include no archived records.",
                entry["problem_spec"], entry["macro_graph"],
            ),
        )

    def test_output_entry_label_not_abbreviated_code(self):
        entry = scene(9, "easy", "reconcile")
        frozen = entry["problem_spec"]
        original = reference_question(frozen)
        self.assertEqual(
            question_output_issues(original, frozen, entry["macro_graph"]), [],
        )
        for changed in (
            "Please list the abbreviated labels of qualifying entries.",
            "Return the label codes of all qualifying entries.",
            "Identify the route codes of the qualifying entries.",
        ):
            with self.subTest(changed=changed):
                candidate = original + " " + changed
                self.assertIn(
                    "requested_code_instead_of_entry_label",
                    guard_question(candidate, frozen, entry["macro_graph"]),
                )

    def test_raw_output_field_can_be_rendered_as_public_label(self):
        from environment.generate_question.surface_quality import normalize_question_punctuation
        entry = scene(9, "easy", "reconcile")
        frozen, graph = entry["problem_spec"], entry["macro_graph"]
        raw = reference_question(frozen) + " Return the entry_label values."
        self.assertIn("internal_implementation_leak", guard_question(raw, frozen, graph))
        question = normalize_question_punctuation(raw)
        self.assertIn("entry label", question)
        self.assertNotIn("internal_implementation_leak", guard_question(question, frozen, graph))

    def test_count_cannot_ask_for_entry_labels(self):
        entry = scene(8, "easy", "coverage_count")
        frozen = entry["problem_spec"]
        changed = (
            reference_question(frozen)
            + " Please list individual entry labels as the final answer."
        )
        self.assertIn(
            "requested_list_instead_of_count",
            guard_question(changed, frozen, entry["macro_graph"]),
        )

    def test_no_hidden_gold_question_material(self):
        entry = scene(15, "hard")
        contract = macro_question_contract(entry["problem_spec"], entry["macro_graph"])
        text = json.dumps(contract)
        self.assertNotIn("gold_answer", text)
        self.assertNotIn('"world":', text)

    def test_question_changes_cannot_change_world_or_graph(self):
        entry = scene(18, "hard")
        before = copy.deepcopy(entry["macro_graph"])
        altered = copy.deepcopy(entry)
        altered["question"] += " Ensure no archived records are included."
        altered["public"]["question"] = altered["question"]
        self.assertEqual(altered["macro_graph"], before)
        self.assertEqual(validate_replay(altered)["status"], "invalid")


class StageAndQualityTests(unittest.TestCase):
    def test_strict_fallback_segregation(self):
        original = [scene(i, "medium") for i in range(6)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "constructed.jsonl"
            _jsonl(src, original)
            args = args_from_config(load_config(DEFAULT_CONFIG))
            args.scenes = src
            args.output_dir = root / "questions"
            args.max_records = None

            def fake_review(frozen, *, author, critic, max_revisions,
                            macro_graph):
                return reference_question(frozen), {
                    "status": "accepted" if frozen["ordinal"] % 2 else "fallback",
                    "attempts": 1, "semantic_equivalence_proven": False,
                }

            with patch("environment.generate_question.stages._llm_clients",
                       return_value=(object(), object())), patch(
                "environment.generate_question.stages.naturalize_one",
                side_effect=fake_review,
            ):
                summary = review_questions(args)
            self.assertEqual(summary["attempted"], 6)
            self.assertEqual(summary["generated"], 3)
            self.assertEqual(summary["failed"], 3)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(len(read_jsonl(args.output_dir / "scenes.jsonl")), 3)
            self.assertEqual(len(read_jsonl(args.output_dir / "questions.jsonl")), 3)
            self.assertEqual(len(read_jsonl(args.output_dir / "failed.jsonl")), 3)

    def test_question_reports_are_written_during_calls_and_can_resume(self):
        from environment.generate_question.stages import (
            run_ordered as real_run_ordered,
        )
        original = [scene(i, "medium") for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "constructed.jsonl"
            _jsonl(source, original)
            args = args_from_config(load_config(DEFAULT_CONFIG))
            args.scenes = source
            args.output_dir = root / "questions"
            args.max_records = None
            args.question_workers = 1
            called = []
            def author(specification, *, author, critic, max_revisions,
                       macro_graph):
                called.append(specification["task_id"])
                if len(called) > 2:
                    self.assertGreater(
                        len(read_jsonl(args.output_dir / "scenes.jsonl")), 0,
                        "The first Question should already be on disk before "
                        "the next Author returns"
                    )
                return reference_question(specification), {
                    "status": "accepted", "attempts": 1,
                }
            class Interrupted(RuntimeError):
                pass
            def stop_after_two(*items, **options):
                old = options["on_commit"]
                def callback(index, good, payload, meta):
                    old(index, good, payload, meta)
                    if index == 1:
                        raise Interrupted("question stopped")
                options["on_commit"] = callback
                return real_run_ordered(*items, **options)
            with patch(
                "environment.generate_question.stages._llm_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.stages.naturalize_one",
                side_effect=author,
            ), patch(
                "environment.generate_question.stages.run_ordered",
                side_effect=stop_after_two,
            ):
                with self.assertRaises(Interrupted):
                    review_questions(args)
            self.assertEqual(
                json.loads(
                    (args.output_dir / "question_authoring_progress.json").read_text()
                )["completed"], 2,
            )
            self.assertEqual(
                json.loads((args.output_dir / "summary.json").read_text())["status"],
                "running",
            )
            before = len(called)
            with patch(
                "environment.generate_question.stages._llm_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.stages.naturalize_one",
                side_effect=author,
            ):
                summary = review_questions(args)
            self.assertEqual(len(called) - before, 3)
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["generated"], 5)
            self.assertEqual(len(read_jsonl(args.output_dir / "scenes.jsonl")), 5)
            self.assertEqual(len(read_jsonl(args.output_dir / "questions.jsonl")), 5)
            self.assertEqual(len(read_jsonl(args.output_dir / "failed.jsonl")), 0)
            self.assertTrue(
                (args.output_dir / "question_authoring_accepted.jsonl").exists()
            )
            with self.assertRaises(FileExistsError):
                review_questions(args)

    def test_scenario_checkpoint_recovers_before_compilation(self):
        from environment.generate_question.stages import (
            run_ordered as real_run_ordered,
        )
        args = args_from_config(load_config(DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args.output_dir = root / "constructed"
            args.target = 4
            args.macro_first_v2_share = 100  # canonical V1.2
            args.scenario_workers = 1
            records = [{
                "record_id": f"item-{i}",
                "topic": f"Fictional Collection {i}",
                "domain_hint": "science",
                "source_modalities": ["image", "text", "table"],
            } for i in range(4)]
            called = []
            def enrich(specification, **kwargs):
                called.append(specification["task_id"])
                row = copy.deepcopy(specification)
                row["scenario_authoring"] = {
                    "status": "accepted", "attempts": 1
                }
                return row, {}
            class Interrupted(RuntimeError):
                pass
            def stop_after_two(*items, **options):
                callback = options.get("on_commit")
                def interruption(index, good, payload, meta):
                    if callback:
                        callback(index, good, payload, meta)
                    if index == 1:
                        raise Interrupted("scenario interrupted")
                options["on_commit"] = interruption
                return real_run_ordered(*items, **options)
            patches = (
                patch(
                    "environment.generate_question.sources.load_scenario_seeds",
                    return_value=(records, {}),
                ),
                patch("environment.generate_question.stages._scenario_clients",
                      return_value=(object(), object())),
                patch("environment.generate_question.scenario.enrich_one",
                      side_effect=enrich),
            )
            with patches[0], patches[1], patches[2], patch(
                "environment.generate_question.stages.run_ordered",
                side_effect=stop_after_two,
            ):
                with self.assertRaises(Interrupted):
                    construct_scenes(args)
            self.assertEqual(
                json.loads(
                    (args.output_dir / "scenario_authoring_progress.json").read_text()
                )["completed"], 2,
            )
            before = len(called)
            with patches[0], patches[1], patches[2]:
                result = construct_scenes(args)
            self.assertEqual(len(called) - before, 2)
            self.assertEqual(result["generated"], 4)
            self.assertEqual(result["status"], "done")
            self.assertEqual(len(read_jsonl(args.output_dir / "scenes.jsonl")), 4)

    def test_quality_marks_bad_predicate_non_publishable(self):
        entry = scene(3, "easy", "reconcile")
        bad = copy.deepcopy(entry)
        bad["question"] += " No archived records may remain."
        bad["public"]["question"] = bad["question"]
        report = quality_report([bad], no_progress=True)
        self.assertEqual(report["replay_pass"], 0)
        self.assertEqual(report["question_publishable"], 0)

    def test_generate_questions_reuses_completed_stages_after_interruption(self):
        args = args_from_config(load_config(DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as td:
            args.output_dir = Path(td) / "pipeline"
            args.target = 4
            args.macro_first_v2_share = 100  # canonical V1.2
            args.quota_refill = False
            args.no_progress = True
            records = [{
                "record_id": f"new-{i}",
                "topic": f"Fictional Collection {i}",
                "domain_hint": "science",
                "source_modalities": ["image", "text", "table"],
            } for i in range(4)]
            def scenario_author(specification, **_):
                modified = copy.deepcopy(specification)
                modified["scenario_authoring"] = {
                    "status": "accepted", "attempts": 1,
                }
                return modified, {}
            def question_author(specification, **_):
                return reference_question(specification), {
                    "status": "accepted", "attempts": 1,
                }
            with patch(
                "environment.generate_question.sources.load_scenario_seeds",
                return_value=(records, {}),
            ), patch(
                "environment.generate_question.stages._scenario_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.stages._llm_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.scenario.enrich_one",
                side_effect=scenario_author,
            ), patch(
                "environment.generate_question.stages.naturalize_one",
                side_effect=question_author,
            ):
                first = generate_questions(args)
            self.assertEqual(first["generated"], 4)
            self.assertEqual(first["status"], "done")
            self.assertEqual(
                len(read_jsonl(args.output_dir / "constructed/scenes.jsonl")), 4
            )
            self.assertEqual(
                len(read_jsonl(args.output_dir / "questions/scenes.jsonl")), 4
            )
            (args.output_dir / "summary.json").unlink()
            # Simulate a crash after all expensive LLM calls but BEFORE the
            # final unified summary. Re-run must reuse both completed stages.
            with patch(
                "environment.generate_question.stages._scenario_clients",
                side_effect=AssertionError("scenario must not rerun"),
            ), patch(
                "environment.generate_question.stages._llm_clients",
                side_effect=AssertionError("question must not rerun"),
            ):
                recovered = generate_questions(args)
            self.assertEqual(recovered["generated"], 4)
            self.assertEqual(recovered["status"], "done")
            self.assertEqual(first["question_authoring"],
                             recovered["question_authoring"])

    def test_stage1_progress_shows_real_work_by_default(self):
        """The default CLI must not freeze at a 0/3 Phase counter.

        Scenario and Question review get their per-candidate tqdm bars.
        Optional validation and quality-audit bars remain opt-in.
        """
        from environment.generate_question.pipeline import generate_questions
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            calls = []
            args = args_from_config(load_config(DEFAULT_CONFIG))
            args.target = 2
            args.no_progress = False

            def constructed(options):
                calls.append(("scenario_compile", options.no_progress))
                (options.output_dir / "scenes.jsonl").parent.mkdir(
                    parents=True, exist_ok=True
                )
                (options.output_dir / "scenes.jsonl").write_text(
                    '{"task_id":"fixture"}\n', encoding="utf-8"
                )
                return {
                    "generated": 1, "attempted": 2,
                    "seed_source": "fixture", "failed": 1,
                    "failed_file": str(options.output_dir / "failed.jsonl"),
                    "scenario_authoring": {"accepted": 1, "rejected": 1},
                }

            def reviewed(options):
                calls.append(("question_review", options.no_progress))
                self.assertEqual(options.verbose_progress, args.verbose_progress)
                return {"failed": 0, "question_results": {"accepted": 1}}

            def audited(options):
                calls.append(("quality_audit", options.no_progress))
                return {
                    "status": "valid", "question_publishable": 1,
                    "difficulty_distribution": {"medium": 1},
                    "checked": 1, "macro_graph_diversity": {},
                    "replay_pass": 1, "replay_fail": 0,
                    "reasoning": {"unique_structural_program_signatures": 1},
                    "question_surface": {"unique_abstract_shapes": 1},
                    "interpretation_limits": [],
                }

            with patch("environment.generate_question.pipeline.construct_scenes",
                       side_effect=constructed), patch(
                "environment.generate_question.pipeline.review_questions",
                side_effect=reviewed,
            ), patch(
                "environment.generate_question.pipeline.audit_questions",
                side_effect=audited,
            ):
                for i, (quiet, verbose) in enumerate((
                    (False, False), (False, True), (True, True),
                )):
                    args.output_dir = Path(td) / f"run_{i}"
                    args.no_progress = quiet
                    args.verbose_progress = verbose
                    calls.clear()
                    result = generate_questions(args)
                    self.assertEqual(result["status"], "done")
                    self.assertEqual(calls, [
                        ("scenario_compile", quiet),
                        ("question_review", quiet),
                        ("quality_audit", quiet or not verbose),
                    ])

    def test_single_candidate_batch_keeps_all_valid_without_refill(self):
        args = args_from_config(load_config(DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as td:
            args.output_dir = Path(td) / "fixed_candidates"
            args.target = 6
            args.macro_first_v2_share = 100  # canonical V1.2
            args.no_progress = True
            records = [{
                "record_id": f"fixed-pool-{i}",
                "topic": f"Fictional Collection {i}",
                "domain_hint": "science",
                "source_modalities": ["image", "text", "table"],
            } for i in range(6)]
            submitted_scenarios, submitted_questions = [], []

            def scenario_author(specification, **_):
                submitted_scenarios.append(specification["task_id"])
                row = copy.deepcopy(specification)
                row["scenario_authoring"] = {
                    "status": (
                        "fallback" if specification["ordinal"] == 1
                        else "accepted"
                    ),
                    "attempts": 1,
                }
                return row, {}

            def question_author(specification, **_):
                submitted_questions.append(specification["task_id"])
                return reference_question(specification), {
                    "status": (
                        "fallback" if specification["ordinal"] == 3
                        else "accepted"
                    ),
                    "attempts": 1,
                }

            with patch(
                "environment.generate_question.sources.load_scenario_seeds",
                return_value=(records, {}),
            ), patch(
                "environment.generate_question.stages._scenario_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.stages._llm_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.scenario.enrich_one",
                side_effect=scenario_author,
            ), patch(
                "environment.generate_question.stages.naturalize_one",
                side_effect=question_author,
            ):
                result = generate_questions(args)

            # Exactly six initial candidates were reviewed; no candidate
            # replacement after either rejection and no difficulty quota gate.
            self.assertEqual(len(submitted_scenarios), 6)
            self.assertEqual(len(submitted_questions), 5)
            self.assertEqual(result["attempted"], 6)
            self.assertEqual(result["generated"], 4)
            self.assertEqual(result["accepted"], 4)
            self.assertEqual(result["rejected"], 2)
            self.assertEqual(result["failed_scenario"], 1)
            self.assertEqual(result["failed_question"], 1)
            self.assertAlmostEqual(result["acceptance_rate"], 4 / 6, places=4)
            self.assertEqual(result["status"], "done")
            self.assertEqual(
                result["selection_policy"],
                "fixed_initial_candidates_no_refill_no_quota_filter",
            )
            self.assertFalse((args.output_dir / "rounds").exists())
            self.assertEqual(
                len(read_jsonl(args.output_dir / "questions" / "scenes.jsonl")),
                4,
            )
            self.assertEqual(
                len(read_jsonl(args.output_dir / "constructed" / "failed.jsonl")),
                1,
            )
            self.assertEqual(
                len(read_jsonl(args.output_dir / "questions" / "failed.jsonl")),
                1,
            )

    def test_scenario_fallback_not_a_success(self):
        args = args_from_config(load_config(DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as td:
            args.output_dir = Path(td) / "constructed"
            args.target = 4
            records = [{
                "record_id": f"item-{i}",
                "topic": f"Fictional Collection {i}",
                "domain_hint": "science",
                "source_modalities": ["image", "text", "table"],
            } for i in range(4)]

            def invalid(specification, **_):
                altered = copy.deepcopy(specification)
                altered["scenario_authoring"] = {
                    "status": "fallback", "attempts": 1,
                }
                return altered, {}

            with patch(
                "environment.generate_question.sources.load_scenario_seeds",
                return_value=(records, {}),
            ), patch(
                "environment.generate_question.stages._scenario_clients",
                return_value=(object(), object()),
            ), patch(
                "environment.generate_question.scenario.enrich_one",
                side_effect=invalid,
            ):
                result = construct_scenes(args)
            self.assertEqual(result["generated"], 0)
            self.assertEqual(result["failed"], 4)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(read_jsonl(args.output_dir / "failed.jsonl")), 4)
            self.assertEqual(read_jsonl(args.output_dir / "questions.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
