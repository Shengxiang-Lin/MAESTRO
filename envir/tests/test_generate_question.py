"""Regression tests for the ONLY maintained Stage-1 Question generator."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from envir.__main__ import parser
from envir.config import DEFAULT_CONFIG, args_from_config, load_config
from envir.generate_question.spec import make_spec
from envir.generate_question.world import compile_program, generate_world
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.macro_contract import macro_question_contract
from envir.generate_question.surface_quality import (
    question_output_issues, unsupported_macro_predicates,
)
from envir.generate_question.stages import (
    construct_scenes, read_jsonl, review_questions, _jsonl,
)
from envir.generate_question.pipeline import quality_report


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
        self.assertEqual(args.difficulty_mix, "20,50,30")
        self.assertTrue(args.mmqa.is_absolute())
        self.assertEqual(args.scenario_workers, 8)
        self.assertEqual(args.question_workers, 8)
        self.assertIsNone(args.question_api_key_env)

    def test_cli_exposes_one_generation_command(self):
        sub = parser()._subparsers._group_actions[0].choices
        self.assertEqual(set(sub), {"generate-questions", "quality"})
        self.assertNotIn("build-v2", sub)
        self.assertNotIn("refine-questions", sub)

    def test_config_cli_override_and_unknown_keys(self):
        cfg = load_config(DEFAULT_CONFIG)
        args = args_from_config(cfg, target=15, output_dir="envir/data/test_15")
        self.assertEqual(args.target, 15)
        self.assertTrue(str(args.output_dir).endswith("envir/data/test_15"))
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
        self.assertEqual(patterns, {
            "approval", "time_window", "status_disagreement", "dynamic_threshold",
        })

    def test_explicit_no_archived_is_not_allowed_without_a_graph_filter(self):
        entry = scene(4, "medium", "priority")
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
        entry = scene(6, "medium", "reconcile")
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

    def test_approval_filter_only_when_in_graph(self):
        entry = scene(12, "hard", "threshold_eligibility")
        question = reference_question(entry["problem_spec"])
        bad = question + " Include only independently approved policy groups."
        self.assertIn(
            "unlicensed_independent_approval_filter",
            guard_question(bad, entry["problem_spec"], entry["macro_graph"]),
        )

    def test_no_hidden_gold_question_material(self):
        entry = scene(15, "hard", "window_alignment")
        contract = macro_question_contract(entry["problem_spec"], entry["macro_graph"])
        text = json.dumps(contract)
        self.assertNotIn("gold_answer", text)
        self.assertNotIn('"world":', text)

    def test_question_changes_cannot_change_world_or_graph(self):
        entry = scene(18, "medium", "status_disagreement")
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

            with patch("envir.generate_question.stages._llm_clients",
                       return_value=(object(), object())), patch(
                "envir.generate_question.stages.naturalize_one",
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

    def test_quality_marks_bad_predicate_non_publishable(self):
        entry = scene(3, "medium", "reconcile")
        bad = copy.deepcopy(entry)
        bad["question"] += " No archived records may remain."
        bad["public"]["question"] = bad["question"]
        report = quality_report([bad], no_progress=True)
        self.assertEqual(report["replay_pass"], 0)
        self.assertEqual(report["question_publishable"], 0)

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
                "envir.generate_question.sources.load_scenario_seeds",
                return_value=(records, {}),
            ), patch(
                "envir.generate_question.stages._scenario_clients",
                return_value=(object(), object()),
            ), patch(
                "envir.generate_question.scenario.enrich_one",
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
