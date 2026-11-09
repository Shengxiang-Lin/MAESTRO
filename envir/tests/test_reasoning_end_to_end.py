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
        spec["reasoning_pattern"] = {
            "window_alignment": "time_window",
            "status_disagreement": "status_disagreement",
            "threshold_eligibility": "dynamic_threshold",
        }.get(family, "approval")
        spec["requirements"]["answer_kind"] = (
            "count" if family == "coverage_count" else
            "category_counts" if family == "category_distribution" else
            "single_entry" if family in {"priority", "chronology"} else
            "entry_evidence" if family == "evidence_reconcile" else
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
            self.assertLessEqual(
                data["macro_graph"]["macro_task_count"], 9)
            self.assertEqual(
                data["macro_graph"]["primitive_operator_count"],
                len(data["gold_program"]["nodes"]))
            self.assertLessEqual(
                data["macro_graph"]["macro_task_count"],
                data["macro_graph"]["primitive_operator_count"])

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
        sid = first["source_ids"][0]
        result = session.read_source(sid)
        self.assertIn("observations", result)
        result["observations"].clear()
        self.assertTrue(data["world"]["sources"][sid]["observations"])
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
        node = next(
            x for x in graph["tasks"]
            if any(step["op"] == "join" for step in x.get(
                "operator_chain", [{"op": x["operation"]}]
            ))
        )
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
            self.assertEqual(spec["version"], "maestro.problem_spec.v4")
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
        sample = scene(difficulty="hard", ordinal=33, family="priority")
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
        spec["reasoning_pattern"] = "approval"
        spec["family"] = "priority"
        spec["requirements"]["answer_kind"] = "single_entry"
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
        spec["version"] = "maestro.problem_spec.v3"
        spec["reasoning_pattern"] = "approval"
        spec["family"] = "priority"
        spec["requirements"]["answer_kind"] = "single_entry"
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
        self.assertIn("question_not_actionable_request", failure["errors"])
        self.assertIn("question_length_out_of_bounds", failure["errors"])


class SemanticPatternContractTests(unittest.TestCase):
    def test_three_new_semantic_families_with_true_media_predicates(self):
        from envir.reasoning.pattern_world import modality_fields
        patterns = set()
        media_used = set()
        for ordinal in range(44):
            sample = scene(difficulty="hard", ordinal=ordinal)
            spec = sample["problem_spec"]
            pattern = spec["reasoning_pattern"]
            patterns.add(pattern)
            self.assertLessEqual(sample["macro_graph"]["macro_task_count"], 9)
            self.assertEqual(validate_replay(sample)["status"], "valid")
            if pattern == "approval":
                continue
            source = sample["world"]["sources"]["annotations"]
            mod = source["modality"]
            fields = modality_fields(mod, pattern)
            expected = set(fields) if isinstance(fields, tuple) else {fields}
            predicates = {step["right_field"] for step in
                          sample["gold_program"]["nodes"]
                          if step["op"] == "compare_fields"}
            self.assertTrue(expected & predicates)
            if mod in {"image", "video", "audio"}:
                media_used.add(mod)
        self.assertEqual(patterns, {
            "approval", "time_window",
            "status_disagreement", "dynamic_threshold",
        })
        self.assertTrue(media_used)

    def test_evidence_output_contains_actual_cited_facts(self):
        sample = scene(difficulty="hard", ordinal=1, family="evidence_reconcile")
        self.assertEqual(sample["problem_spec"]["requirements"]["answer_kind"],
                         "entry_evidence")
        self.assertTrue(sample["gold_answer"])
        for row in sample["gold_answer"]:
            self.assertEqual(set(row), {"entry_label", "evidence_refs"})
            self.assertEqual(len(row["evidence_refs"]), 4)
            self.assertTrue(all(":" in ref for ref in row["evidence_refs"]))
        self.assertEqual(validate_replay(sample)["status"], "valid")

    def test_question_author_receives_structured_intent_not_template(self):
        from envir.reasoning.question import naturalize_one
        spec = make_spec(seed_record=seed(15), ordinal=15,
                         run_seed=17, difficulty="medium")
        seen = []
        class SyntheticAuthor:
            def complete_json(self, system_prompt, user_prompt):
                p = json.loads(user_prompt)
                seen.append(p["contract"])
                return {"question": "Not a full question"}
        class NeverCritic:
            def complete_json(self, *_args):
                raise AssertionError("malformed question should fail guard")
        question, result = naturalize_one(
            spec, author=SyntheticAuthor(), critic=NeverCritic(),
            max_revisions=0,
        )
        self.assertEqual(result["status"], "fallback")
        self.assertNotIn("frozen_reference_question", seen[0])
        self.assertIn("intent", seen[0])
        self.assertIn("style", seen[0])

    def test_compact_task_rejects_unauthorized_grants_and_missing_inputs(self):
        data = scene(difficulty="hard", ordinal=7)
        bad = copy.deepcopy(data["macro_graph"])
        composite = next(t for t in bad["tasks"] if t["operator_chain"]
                         and len(t["operator_chain"]) > 1)
        composite["source_ids"] = ["unexpected-source"]
        with self.assertRaises((ProgramError, MissingSourceError)):
            run_macro_tasks(bad, sources=data["world"]["sources"])

    def test_legacy_v3_source_data_is_still_replayable(self):
        spec = make_spec(seed_record=seed(10), ordinal=10,
                         run_seed=17, difficulty="medium")
        spec["version"] = "maestro.problem_spec.v3"
        spec["reasoning_pattern"] = "approval"
        spec["family"] = "priority"
        spec["requirements"]["answer_kind"] = "single_entry"
        result = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(result["gold_program"]["version"],
                         "maestro.reasoning_program.v3")
        self.assertEqual(validate_replay(result)["status"], "valid")


class SemanticLanguageQualityTests(unittest.TestCase):
    def test_closed_time_window_rejects_strictly_inside(self):
        from envir.reasoning.surface_quality import question_language_warnings
        data = scene(difficulty="medium", ordinal=0,
                     family="window_alignment")
        spec = data["problem_spec"]
        question = reference_question(spec)
        self.assertFalse(question_language_warnings(question, spec))
        contradiction = question.replace(
            "inclusive time window", "closed time window"
        ).replace(
            "Which entries fall inside", "Which entries fall strictly inside"
        )
        self.assertIn(
            "strict_vs_inclusive_time_window",
            question_language_warnings(contradiction, spec),
        )
        self.assertIn(
            "strict_vs_inclusive_time_window",
            guard_question(contradiction, spec),
        )

    def test_rejected_unlicensed_year_and_allowed_original_topic_year(self):
        from envir.reasoning.surface_quality import unlicensed_calendar_years
        spec = make_spec(seed_record=seed(7), ordinal=7,
                         run_seed=17, difficulty="medium")
        question = reference_question(spec)
        self.assertIn(
            "unlicensed_calendar_year_not_in_topic",
            guard_question("For the 2024 cycle, " + question, spec),
        )
        self.assertEqual(
            unlicensed_calendar_years("2024 membership audit", "2024 membership"),
            [],
        )

    def test_question_mark_normalization_preserves_meaning(self):
        from envir.reasoning.surface_quality import (
            normalize_question_punctuation, question_language_warnings
        )
        data = scene(difficulty="medium", ordinal=2,
                     family="threshold_eligibility")
        reference = reference_question(data["problem_spec"])
        self.assertEqual(normalize_question_punctuation(reference + "?"),
                         reference)
        self.assertIn("repeated_question_marks",
                      question_language_warnings(
                          reference + "?", data["problem_spec"]))

    def test_scenario_guard_rejects_new_calendar_year(self):
        from envir.reasoning.scenario import _narrative_issues
        spec = make_spec(seed_record=seed(5), ordinal=5,
                         run_seed=17, difficulty="hard")
        proposal = {
            "scenario": {
                "setting": f"The fictional 2024 {spec['scenario']['topic']} files",
                "user_role": "audit researcher",
                "purpose": "check multiple source annotations",
                "motivation": "avoid incomplete labels",
                "entity_kind": "register entries",
            },
            "source_titles": [
                {"source_id": x["source_id"], "title": x["title"] + " archive"}
                for x in spec["sources"]
            ],
        }
        self.assertIn(
            "unlicensed_calendar_year_not_in_topic",
            _narrative_issues(spec, proposal),
        )

    def test_quality_flags_question_wording_despite_exact_replay(self):
        from envir.reasoning.workflow import quality_report
        data = scene(difficulty="medium", ordinal=0,
                     family="window_alignment")
        faulty = copy.deepcopy(data)
        faulty["question"] = (
            "For a 2024 fictional audit of Fictional Topic 0, which active "
            "entries have event steps strictly inside the closed time windows "
            "after mapping their codes through the glossary??"
        )
        faulty["public"]["question"] = faulty["question"]
        faulty["question_origin"] = "llm_accepted_frozen_spec"
        report = quality_report([faulty], no_progress=True)
        self.assertEqual(report["replay_pass"], 1)
        counts = report["question_surface"]["semantic_language_warnings"]["counts"]
        self.assertEqual(counts["strict_vs_inclusive_time_window"], 1)
        self.assertEqual(counts["repeated_question_marks"], 1)
        self.assertEqual(counts["unlicensed_calendar_year_not_in_topic"], 1)


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
        self.assertLessEqual(aggregate["one_source_shortcut_successes"], 3)
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
        self.assertEqual(summary["generated"], 0)
        self.assertEqual(summary["failed"], 4)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(read_jsonl(out / "scenes.jsonl"), [])
        self.assertEqual(read_jsonl(out / "questions.jsonl"), [])
        self.assertEqual(len(read_jsonl(out / "failed.jsonl")), 4)
        self.assertEqual(replay_v2(Namespace(
            scenes=src, max_records=None
        ))["status"], "valid")


class MacroGraphAndStrictAcceptanceTests(unittest.TestCase):
    def test_distinct_macro_dependency_plans_preserve_exact_gold(self):
        from envir.reasoning.macro_contract import (
            macro_graph_profile, macro_question_contract,
        )
        for pattern, family in (
            ("time_window", "window_alignment"),
            ("status_disagreement", "status_disagreement"),
            ("dynamic_threshold", "threshold_eligibility"),
        ):
            with self.subTest(pattern=pattern):
                base = make_spec(
                    seed_record=seed(38), ordinal=38,
                    run_seed=17, difficulty="hard",
                )
                base["reasoning_pattern"] = pattern
                base["family"] = family
                base["requirements"]["answer_kind"] = "entry_list"
                base.pop("macro_plan_variant", None)
                legacy = compile_and_validate(base, reference_question(base))
                signatures = set()
                for variant in range(4):
                    plan = copy.deepcopy(base)
                    plan["macro_plan_variant"] = variant
                    scene_data = compile_and_validate(
                        plan, reference_question(plan))
                    self.assertEqual(scene_data["gold_answer"], legacy["gold_answer"])
                    self.assertLessEqual(scene_data["macro_graph"]["macro_task_count"], 9)
                    self.assertEqual(validate_replay(scene_data)["status"], "valid")
                    contract = macro_question_contract(
                        plan, scene_data["macro_graph"])
                    self.assertEqual(contract["answer_kind"], "entry_list")
                    signatures.add(macro_graph_profile(
                        scene_data["macro_graph"])["signature"])
                self.assertGreaterEqual(len(signatures), 3)

    def test_question_author_sees_compiled_macro_contract(self):
        from envir.reasoning.question import naturalize_one
        sample = scene(difficulty="hard", ordinal=24,
                       family="window_alignment")
        received = []
        class Author:
            def complete_json(self, system_prompt, user_prompt):
                received.append(json.loads(user_prompt)["contract"])
                return {"question": "Malformed"}
        class NeverCritic:
            def complete_json(self, *_args):
                raise AssertionError("guard rejects malformed draft")
        naturalize_one(
            sample["problem_spec"], author=Author(),
            critic=NeverCritic(), max_revisions=0,
            macro_graph=sample["macro_graph"],
        )
        contract = received[0]
        self.assertIn("compiled_macro_task_contract", contract)
        self.assertIn("macro_dependency_plan", contract)
        self.assertEqual(
            contract["compiled_macro_task_contract"]["answer_kind"],
            "entry_list")
        self.assertEqual(
            len(contract["macro_dependency_plan"]),
            sample["macro_graph"]["macro_task_count"])
        self.assertNotIn("gold_answer", json.dumps(contract))
        self.assertNotIn("world", json.dumps(contract))

    def test_fallback_questions_are_private_failures_not_public_successes(self):
        from envir.reasoning.pipeline import refine_bank
        from envir.reasoning.pipeline import build_bank
        from envir.reasoning.question import reference_question
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "seeds.jsonl"
            with src.open("w", encoding="utf-8") as fp:
                for ordinal in range(6):
                    fp.write(json.dumps({
                        "scenario": {
                            "topic": f"Fictional Topic {ordinal}",
                            "domain": "sports",
                        },
                        "task_id": f"legacy-{ordinal}",
                        "source_priors": {"MultiModalQA": {
                            "record_id": f"fixture-{ordinal}",
                            "source_modalities": ["table", "image", "text"],
                        }},
                    }) + "\n")
            built = root / "constructed"
            build_bank(parser().parse_args([
                "build-v2", "--scenes", str(src), "--target", "6",
                "--output-dir", str(built), "--no-progress",
            ]))
            def fake_author(spec, *, author, critic, max_revisions,
                            macro_graph=None):
                return reference_question(spec), {
                    "status": "accepted" if spec["ordinal"] % 2 else "fallback",
                    "attempts": 1, "semantic_equivalence_proven": False,
                }
            output = root / "reviewed"
            with patch("envir.reasoning.pipeline._llm_clients",
                       return_value=(object(), object())), patch(
                 "envir.reasoning.pipeline.naturalize_one",
                 side_effect=fake_author):
                result = refine_bank(parser().parse_args([
                    "refine-questions", "--scenes", str(built / "scenes.jsonl"),
                    "--output-dir", str(output),
                    "--question-api-base", "http://localhost:1000/v1",
                    "--question-model", "fake", "--no-progress",
                ]))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["attempted"], 6)
            self.assertEqual(result["generated"], 3)
            self.assertEqual(result["failed"], 3)
            self.assertEqual(len(read_jsonl(output / "scenes.jsonl")), 3)
            self.assertEqual(len(read_jsonl(output / "questions.jsonl")), 3)
            self.assertEqual(len(read_jsonl(output / "failed.jsonl")), 3)
            self.assertTrue(all(
                row["question_authoring"]["status"] == "accepted"
                for row in read_jsonl(output / "scenes.jsonl")))

    def test_scenario_review_fallback_is_not_compiled_or_published(self):
        from envir.reasoning.pipeline import build_bank
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "seeds.jsonl"
            with src.open("w", encoding="utf-8") as fp:
                for i in range(5):
                    fp.write(json.dumps({
                        "scenario": {"topic": f"Test Topic {i}",
                                     "domain": "sports"},
                        "task_id": f"legacy-{i}",
                    }) + "\n")
            class BrokenAuthor:
                def complete_json(self, *_args):
                    raise RuntimeError("deliberate review failure")
            out = root / "out"
            with patch("envir.reasoning.pipeline._scenario_clients",
                       return_value=(BrokenAuthor(), BrokenAuthor())):
                summary = build_bank(parser().parse_args([
                    "build-v2", "--scenes", str(src), "--target", "5",
                    "--output-dir", str(out),
                    "--scenario-api-base", "http://localhost:1000/v1",
                    "--scenario-model", "fake", "--no-progress",
                ]))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["attempted"], 5)
            self.assertEqual(summary["generated"], 0)
            self.assertEqual(summary["failed"], 5)
            self.assertEqual(read_jsonl(out / "questions.jsonl"), [])
            self.assertEqual(len(read_jsonl(out / "failed.jsonl")), 5)


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

    def test_full_hundred_topic_bank_quality_and_task_budget(self):
        from envir.reasoning.workflow import generate
        with self.scenes.open("a", encoding="utf-8") as writer:
            for ordinal in range(15, 120):
                writer.write(json.dumps({
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
        args = parser().parse_args([
            "generate", "--scenes", str(self.scenes),
            "--output-dir", str(self.root / "hundred_run"),
            "--target", "100", "--max-records", "120",
            "--difficulty-mix", "20,50,30", "--no-progress",
        ])
        summary = generate(args)
        self.assertEqual(summary["generated"], 100)
        self.assertEqual(summary["quality"]["replay_pass"], 100)
        report = json.loads((self.root / "hundred_run" /
                             "quality_report.json").read_text(encoding="utf-8"))
        patterns = report["semantic_reasoning_pattern_distribution"]
        for name in ("approval", "time_window", "status_disagreement",
                     "dynamic_threshold"):
            self.assertGreater(patterns.get(name, 0), 5)
        self.assertEqual(report["reasoning"]["macro_tasks_above_nine"], 0)
        self.assertEqual(report["replay_pass"], 100)
        self.assertGreaterEqual(
            report["reasoning"]["unique_structural_program_signatures"], 15
        )
        self.assertTrue(
            report["reasoning"]["media_specific_observation_predicates_used"]
        )
        graphs = report["macro_graph_diversity"]
        self.assertGreater(graphs["distinct_executable_graph_signatures"], 24)
        for pattern in ("time_window", "status_disagreement",
                        "dynamic_threshold"):
            self.assertGreaterEqual(
                graphs["distinct_by_semantic_pattern"][pattern], 3
            )
        self.assertEqual(graphs["graph_to_question_contract_failures"], {})

    def test_historical_question_fallback_is_not_publishable_on_recheck(self):
        from envir.reasoning.workflow import quality_report
        sample = scene(difficulty="medium", ordinal=2)
        sample["question_authoring"] = {
            "status": "fallback", "attempts": 3,
            "failures": [{"stage": "critic"}],
        }
        report = quality_report([sample], no_progress=True)
        self.assertEqual(report["replay_pass"], 1)
        self.assertEqual(report["question_publishable"], 0)
        self.assertEqual(report["question_failed_fallback"], 1)
        self.assertEqual(report["status"], "partial")

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


class ReviewPrecisionRegressionTests(unittest.TestCase):
    def test_imperative_question_with_correct_source_contract_is_valid(self):
        from envir.reasoning.question import guard_question, reference_question
        sample = scene(
            difficulty="medium", ordinal=15,
            family="status_disagreement"
        )
        spec = sample["problem_spec"]
        original = reference_question(spec)
        imperative = original.replace(
            "Which entries have different statuses across the two records?",
            "Please identify entries with status discrepancies between "
            "the two records."
        )
        self.assertNotIn("?", imperative)
        self.assertNotIn(
            "question_not_actionable_request",
            guard_question(imperative, spec),
        )
        self.assertNotIn(
            "status_disagreement_missing",
            guard_question(imperative, spec),
        )
        self.assertEqual(guard_question(imperative, spec), [])

    def test_real_discrepancy_and_differ_synonyms_are_recognized(self):
        original = scene(
            difficulty="medium", ordinal=16, family="status_disagreement"
        )
        spec = original["problem_spec"]
        reference = reference_question(spec)
        for description in (
            "Please list entries with discrepancies between the statuses.",
            "Please identify entries whose statuses differ across records.",
            "Please list entries with mismatches in reported status.",
        ):
            with self.subTest(description=description):
                candidate = reference.replace(
                    "Which entries have different statuses across the two records?",
                    description
                )
                self.assertEqual(guard_question(candidate, spec), [])

    def test_topic_matching_retains_identity_without_overrejecting_punctuation(self):
        from envir.reasoning.surface_quality import topic_mentioned
        self.assertTrue(
            topic_mentioned("The fictional Green Lantern register",
                            "Green Lantern (film)")
        )
        self.assertTrue(
            topic_mentioned("Review the 2012-13 season", "2012–13")
        )
        self.assertTrue(
            topic_mentioned("Brent McGrath\u2019s register",
                            "Brent McGrath's")
        )
        self.assertFalse(
            topic_mentioned("The Apex Logistics fleet",
                            "Brent McGrath's")
        )
        self.assertFalse(
            topic_mentioned("The Swansea register", "Swansea City")
        )

    def test_descriptive_sentence_without_request_stays_rejected(self):
        from envir.reasoning.surface_quality import is_explicit_user_request
        self.assertFalse(
            is_explicit_user_request(
                "The independent archive has a descriptive record with "
                "several fields and no explicit instruction to the reader."
            )
        )
        self.assertTrue(
            is_explicit_user_request(
                "For the fictional archival register, please identify "
                "which active entries have a valid independent confirmation."
            )
        )

    def test_macro_contract_rejects_reversed_earliest_selection(self):
        from envir.reasoning.macro_contract import macro_question_contract
        sample = scene(
            difficulty="medium", ordinal=21, family="chronology"
        )
        altered = copy.deepcopy(sample["macro_graph"])
        reversed_sort = False
        for task in altered["tasks"]:
            for part in task.get(
                "operator_chain", [{"op": task["operation"]}]
            ):
                if part["op"] == "sort":
                    part["descending"] = True
                    reversed_sort = True
        self.assertTrue(reversed_sort)
        with self.assertRaisesRegex(
            ValueError, "wrong_selection_direction"
        ):
            macro_question_contract(sample["problem_spec"], altered)

    def test_older_spec_retains_required_question_mark(self):
        data = scene(
            difficulty="medium", ordinal=18, family="reconcile"
        )
        spec = copy.deepcopy(data["problem_spec"])
        spec["version"] = "maestro.problem_spec.v3"
        text = reference_question(spec)
        without_mark = text.rstrip("?")
        self.assertIn(
            "question_missing_interrogative",
            guard_question(without_mark, spec),
        )


class MacroQuestionAdditionalPredicateTests(unittest.TestCase):
    def test_threshold_approval_is_not_in_macro_graph(self):
        from envir.reasoning.surface_quality import (
            unsupported_macro_predicates,
        )
        sample = scene(difficulty="hard", ordinal=23,
                       family="threshold_eligibility")
        spec = sample["problem_spec"]
        for added in (
            "Only entries mapped to an approved group qualify.",
            "Only entries mapped to independently approved policy groups qualify.",
        ):
            question = reference_question(spec) + " " + added
            self.assertIn(
                "unlicensed_independent_approval_filter",
                unsupported_macro_predicates(question, spec),
            )
            self.assertIn(
                "unlicensed_independent_approval_filter",
                guard_question(question, spec),
            )
        # The original high-precision reference must remain acceptable.
        self.assertEqual(guard_question(reference_question(spec), spec), [])

    def test_time_window_cannot_acquire_approval_predicate(self):
        sample = scene(difficulty="medium", ordinal=21,
                       family="window_alignment")
        spec = sample["problem_spec"]
        corrupted = (
            reference_question(spec)
            + " Choose only the approved simulation session window."
        )
        self.assertIn(
            "unlicensed_independent_approval_filter",
            guard_question(corrupted, spec),
        )
        self.assertEqual(guard_question(reference_question(spec), spec), [])

    def test_archived_state_is_not_implicit_in_active_flag(self):
        sample = scene(difficulty="easy", ordinal=8, family="eligibility")
        spec = sample["problem_spec"]
        for added in (
            "Please exclude any inactive or archived entries.",
            "Retain only valid non-archived entries.",
        ):
            with self.subTest(added=added):
                q = reference_question(spec) + " " + added
                self.assertIn(
                    "unlicensed_archive_status_filter",
                    guard_question(q, spec),
                )

    def test_easy_chronology_does_not_invent_independent_approval(self):
        sample = scene(difficulty="easy", ordinal=4, family="chronology")
        spec = sample["problem_spec"]
        original = reference_question(spec)
        q = (
            f"For {spec['scenario']['topic']}, identify the single active "
            "entry that qualifies for the official approved register, "
            "specifically the one with the earliest relative step."
        )
        self.assertIn(
            "unlicensed_easy_approval_filter",
            guard_question(q, spec),
        )
        self.assertEqual(guard_question(original, spec), [])

    def test_surface_normalization_repairs_format_only(self):
        from envir.reasoning.surface_quality import (
            normalize_question_punctuation,
        )
        raw = (
            "Please identify active records using diagram_step_start and "
            "diagram_step_end with measured_value meeting minimum_required.??"
        )
        cleaned = normalize_question_punctuation(raw)
        self.assertEqual(
            cleaned,
            "Please identify active records using diagram interval start step "
            "and diagram interval end step with measured value meeting "
            "minimum required threshold?",
        )
        self.assertEqual(
            normalize_question_punctuation(
                'Please identify all active entries."'
            ),
            "Please identify all active entries.",
        )
        self.assertEqual(
            normalize_question_punctuation(
                'For "The Office" records, identify active entries.'
            ),
            'For "The Office" records, identify active entries.',
        )

    def test_language_quality_report_surfaces_accepted_bad_predicate(self):
        from envir.reasoning.workflow import quality_report
        sample = scene(difficulty="medium", ordinal=5,
                       family="threshold_eligibility")
        corrupted = copy.deepcopy(sample)
        corrupted["question"] = (
            reference_question(sample["problem_spec"])
            + " Keep only entries assigned to an approved policy group."
        )
        corrupted["public"]["question"] = corrupted["question"]
        report = quality_report([corrupted], no_progress=True)
        # The stricter Question guard deliberately invalidates this
        # historically accepted but semantically unlicensed task.
        self.assertEqual(report["replay_pass"], 0)
        self.assertEqual(report["replay_fail"], 1)
        warnings = report["question_surface"][
            "semantic_language_warnings"]["counts"]
        self.assertEqual(
            warnings["unlicensed_independent_approval_filter"], 1
        )


if __name__ == "__main__":
    unittest.main()
