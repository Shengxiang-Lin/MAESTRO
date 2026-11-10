"""Clue-Chain: executable information acquisition, not static table joins."""
from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.generate_question.spec import make_spec, spec_issues
from environment.generate_question.question import reference_question, guard_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.compiler import QuestionSession, oracle_schedule
from environment.generate_question.program_ir import (
    MissingSourceError, ProgramError, Relation, _scan, _lookup_by_key,
    evaluate_program, answer_values,
)
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.naturalize import naturalize_subtasks


def clue_spec(i=2):
    return make_spec(
        seed_record={
            "record_id": f"clue-{i}",
            "topic": f"Fictional Aviation Collection {i}",
            "domain_hint": "history",
            "source_modalities": ["table", "image", "text"],
        },
        ordinal=i, run_seed=17, difficulty="medium",
        clue_chain_share=100,
    )


def clue_scene(i=2):
    spec = clue_spec(i)
    scene = compile_and_validate(spec, reference_question(spec))
    scene["question_authoring"] = {"status": "accepted"}
    return scene


class ClueChainContracts(unittest.TestCase):
    def test_stage1_config_cli_and_batch_construction(self):
        from environment.config import load_config, args_from_config, DEFAULT_CONFIG
        from environment.__main__ import parser
        from environment.generate_question.stages import construct_scenes, read_jsonl
        cfg = load_config(DEFAULT_CONFIG)
        self.assertFalse(hasattr(args_from_config(cfg), "clue_chain_share"))
        parsed = parser().parse_args([
            "generate-questions", "--config", str(DEFAULT_CONFIG),
            "--target", "10", "--macro-first-v2-share", "100",
            "--output-dir", "environment/data/clue-pilot-test",
        ])
        self.assertEqual(parsed.macro_first_v2_share, 100)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = args_from_config(cfg, output_dir=root / "constructed",
                                    target=10, no_progress=True)
            args.difficulty_mix = "0,100,0"
            args.clue_chain_share = 100
            args.macro_first_v2_share = 0  # explicit archived pilot mode
            args.scenario_model = None
            args.scenario_api_base = None
            args.mmqa = root / "source-not-required-by-mock.jsonl"
            records = [{
                "record_id": f"fresh-clue-{i}",
                "topic": f"Fictional Museum {i}",
                "domain_hint": "history",
                "source_modalities": ["image", "table", "text"],
            } for i in range(10)]
            with patch(
                "environment.generate_question.sources.load_scenario_seeds",
                return_value=(records, {}),
            ):
                summary = construct_scenes(args)
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["generated"], 10)
            scenes = read_jsonl(root / "constructed/scenes.jsonl")
            self.assertEqual(len(scenes), 10)
            self.assertTrue(all(
                s["problem_spec"]["reasoning_pattern"] == "clue_chain"
                for s in scenes
            ))
            self.assertTrue(all(
                s["difficulty_audit"]["evidence_acquisition"][
                    "clue_counterfactual_pass"
                ] for s in scenes
            ))

    def test_new_pattern_is_opt_in_and_existing_specs_unchanged(self):
        legacy = make_spec(
            seed_record={
                "record_id": "old", "topic": "Old Catalog",
                "domain_hint": "history",
            }, ordinal=2, run_seed=17, difficulty="medium",
        )
        self.assertNotEqual(legacy["reasoning_pattern"], "clue_chain")
        self.assertNotIn("investigation", legacy)
        self.assertEqual(spec_issues(legacy), [])
        self.assertEqual(clue_spec()["reasoning_pattern"], "clue_chain")
        self.assertEqual(clue_spec()["family"], "accession_year")
        self.assertEqual(spec_issues(clue_spec()), [])
        with self.assertRaisesRegex(ValueError, "clue_chain_share"):
            make_spec(
                seed_record={"topic": "Bad"},
                ordinal=1, run_seed=17, difficulty="medium",
                clue_chain_share=101,
            )

    def test_question_contains_both_clues_and_keyed_evidence_path(self):
        spec = clue_spec()
        question = reference_question(spec)
        self.assertEqual(guard_question(question, spec), [])
        features = spec["investigation"]["visual_features"]
        self.assertIn(features["emblem_color"], question)
        self.assertIn(features["emblem_shape"], question)
        self.assertIn("index", question.lower())
        self.assertIn("archive", question.lower())
        self.assertIn("year", question.lower())
        self.assertIn("Fictional Aviation Collection", question)
        self.assertIn("visual_clue_features_missing", guard_question(
            question.replace(features["emblem_color"], "pink"), spec,
        ))

    def test_full_scene_replay_and_separate_keyed_macro_boundaries(self):
        scene = clue_scene()
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertEqual(len(scene["gold_answer"]), 1)
        self.assertEqual(set(scene["gold_answer"][0]), {"accession_year"})
        nodes = scene["gold_program"]["nodes"]
        lookups = [row for row in nodes if row["op"] == "lookup_by_key"]
        self.assertEqual(
            [(x["source_id"], x["input_key_field"]) for x in lookups],
            [("index", "marker_code"), ("archive", "accession_id")],
        )
        graph = scene["macro_graph"]
        self.assertGreaterEqual(graph["critical_path_depth"], 3)
        self.assertEqual(
            [step["op"] for task in graph["tasks"]
             for step in task["operator_chain"]].count("lookup_by_key"), 2,
        )
        self.assertEqual(len([
            task for task in graph["tasks"]
            if any(part["op"] == "lookup_by_key"
                   for part in task["operator_chain"])
        ]), 2)
        for task in graph["tasks"]:
            if any(part["op"] == "lookup_by_key"
                   for part in task["operator_chain"]):
                self.assertEqual(len(task["operator_chain"]), 1)
                self.assertEqual(len(task["depends_on"]), 1)
                self.assertEqual(len(task["source_ids"]), 1)
        audit = scene["difficulty_audit"]
        self.assertEqual(audit["reasoning"]["keyed_lookup_count"], 2)
        self.assertEqual(
            audit["evidence_acquisition"]["full_read_blocked"],
            {"index": True, "archive": True},
        )
        self.assertEqual(
            audit["evidence_acquisition"]["unauthorized_full_read_success_rate"],
            0,
        )
        self.assertTrue(audit["evidence_acquisition"]["clue_counterfactual_pass"])
        self.assertTrue(audit["evidence_acquisition"]["nontrivial_decoys"])

    def test_zero_upstream_lookup_and_wrong_key_have_no_access(self):
        scene = clue_scene()
        source = scene["world"]["sources"]["index"]
        with self.assertRaisesRegex(MissingSourceError,
                                    "full_source_scan_forbidden"):
            _scan("index", source)
        node = next(
            x for x in scene["gold_program"]["nodes"]
            if x["op"] == "lookup_by_key" and x["source_id"] == "index"
        )
        ungrounded = Relation(("marker_code",), ({
            "values": {"marker_code": "M-DOES-NOT-EXIST"}, "support": (),
        },))
        with self.assertRaisesRegex(ProgramError, "missing_upstream_provenance"):
            _lookup_by_key(node, ungrounded, source)
        nonexistent = Relation(("marker_code",), ({
            "values": {"marker_code": "M-DOES-NOT-EXIST"},
            "support": ("illustration:F001",),
        },))
        result = _lookup_by_key(node, nonexistent, source)
        self.assertEqual(result.rows, ())
        # A syntactically valid key with no matching record does not reveal
        # source observations and cannot produce the true year.
        wrong_sources = deepcopy(scene["world"]["sources"])
        illustrated = wrong_sources["illustration"]["observations"]
        target = next(row for row in illustrated
                      if all(row["values"][k] == v for k, v in
                             scene["problem_spec"]["investigation"][
                                 "visual_features"].items()))
        target["values"]["marker_code"] = "M-DOES-NOT-EXIST"
        self.assertEqual(answer_values(evaluate_program(
            scene["gold_program"], sources=wrong_sources
        )), [])

    def test_runtime_cannot_browse_index_or_archive(self):
        scene = clue_scene()
        session = QuestionSession(scene)
        before = session.public_view
        self.assertNotIn("world", before)
        self.assertEqual({
            row["source_id"]: row["access_mode"]
            for row in before["available_source_descriptions"]
        }, {
            "illustration": "browse", "index": "keyed_lookup",
            "archive": "keyed_lookup",
        })
        for source in ("index", "archive"):
            with self.assertRaisesRegex(MissingSourceError,
                                        "full_source_read_forbidden"):
                session.read_source(source)
        tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
        first_lookup = next(t for t in tasks.values()
                            if t["source_ids"] == ["index"])
        with self.assertRaisesRegex(ProgramError, "missing_upstream_workproduct"):
            session.run_task(first_lookup["task_id"])
        # Reference schedule makes only one broad read (illustration);
        # both later retrievals are distinct costed lookup actions.
        reference = oracle_schedule(scene)
        self.assertEqual(reference["answer"], scene["gold_answer"])
        actions = reference["action_log"]
        self.assertEqual([
            x["source_id"] for x in actions if x["action"] == "read_source"
        ], ["illustration"])
        keyed = [x for x in actions if x["action"] == "lookup_by_key"]
        self.assertEqual([x["source_id"] for x in keyed],
                         ["index", "archive"])
        self.assertTrue(all(len(x["keys"]) == 1 for x in keyed))
        # Manually follow exact reference waves to verify final source
        # provenance WITHOUT granting a broad read of either keyed store.
        session.read_source("illustration")
        for wave in scene["macro_graph"]["reference_waves"]:
            for task_id in wave:
                session.run_task(task_id)
        terminal = scene["macro_graph"]["terminal_task_id"]
        self.assertTrue(session.verify(terminal)["valid_evidence_closure"])
        self.assertEqual(session.final_answer(), scene["gold_answer"])

    def test_mutating_visual_target_repoints_downstream_archive(self):
        scene = clue_scene(3)
        sources = deepcopy(scene["world"]["sources"])
        picture = sources["illustration"]["observations"]
        features = scene["problem_spec"]["investigation"]["visual_features"]
        match = next(x for x in picture if all(
            x["values"][field] == wanted for field, wanted in features.items()
        ))
        other_marker = next(x["values"]["marker_code"] for x in picture
                            if x is not match)
        match["values"]["marker_code"] = other_marker
        rerouted = answer_values(evaluate_program(scene["gold_program"],
                                                   sources=sources))
        self.assertNotEqual(rerouted, scene["gold_answer"])

    def test_quality_report_exposes_evidence_acquisition_metrics(self):
        from environment.generate_question.pipeline import quality_report
        scenes = [clue_scene(2), clue_scene(3)]
        report = quality_report(scenes, no_progress=True)
        self.assertEqual(report["status"], "valid")
        metrics = report["evidence_acquisition"]
        self.assertEqual(metrics["scenes_with_keyed_clue_chain"], 2)
        self.assertEqual(metrics["keyed_lookup_count"], 4)
        self.assertEqual(metrics["mean_evidence_acquisition_depth"], 2)
        self.assertEqual(metrics["clue_counterfactual_pass_rate"], 1.0)
        self.assertEqual(metrics["unauthorized_full_read_success_rate"], 0.0)

    def test_subtask_oracle_includes_keyed_evidence(self):
        scene = clue_scene()
        realized = realize_scene(scene)
        self.assertEqual(realized["macro_task_count"],
                         scene["macro_graph"]["macro_task_count"])
        lookups = [row for row in realized["macro_tasks"]
                   if any(step["operation"] == "lookup_by_key"
                          for step in row["trace_summary"]["steps"])]
        self.assertEqual(len(lookups), 2)
        self.assertEqual(
            [x["authorized_source_ids"] for x in lookups],
            [["index"], ["archive"]],
        )
        for rec in lookups:
            step = rec["trace_summary"]["steps"][0]
            self.assertIn("upstream", step["action"])
            self.assertGreaterEqual(len(step["supporting_fact_refs"]), 2)
        terminal = next(row for row in realized["macro_tasks"]
                        if row["macro_task_id"] ==
                        realized["terminal_macro_task_id"])
        self.assertEqual(terminal["answer"], scene["gold_answer"])
        self.assertTrue({
            "illustration", "index", "archive"
        } <= set(terminal["trace_summary"]["available_evidence_source_ids"]))


class SimpleAuthor:
    def complete_json(self, system_prompt, user_prompt):
        payload = json.loads(user_prompt)["frozen_task_and_evidence"]
        return {
            "result": payload["exact_oracle_answer"],
            "summary": (
                "I processed the supplied evidence for this stage and "
                "returned the matching observations without inventing "
                "unperformed source accesses or downstream filters."
            ),
            "status": "done",
        }


class SimpleCritic:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "verdict": "pass",
            "grounded": True,
            "answer_scope_preserved": True,
            "evidence_scope_respected": True,
            "natural": True,
            "reason": "Exact stage-local evidence and result.",
            "repair_hint": "",
        }


class ClueChainStage2Integration(unittest.TestCase):
    def test_oracle_and_naturalize_two_clue_scenes(self):
        from environment.implement_subtask.pipeline import materialize_subtasks
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scenes = root / "scenes.jsonl"
            with scenes.open("w", encoding="utf-8") as handle:
                for i in (2, 3):
                    handle.write(json.dumps(clue_scene(i)) + "\n")
            oracle = materialize_subtasks(
                scenes_path=scenes, output_dir=root / "oracle",
                strict_replay=True, no_progress=True,
            )
            self.assertEqual(oracle["accepted_scenes"], 2)
            self.assertEqual(oracle["failed_scenes"], 0)
            reviewed = naturalize_subtasks(
                source_subtasks=root / "oracle/subtasks.jsonl",
                source_scene_index=root / "oracle/scene_index.jsonl",
                output_dir=root / "reviewed", api_base="http://unused/v1",
                model="mock", clients=(SimpleAuthor(), SimpleCritic()),
                workers=2, no_progress=True, max_revisions=0,
            )
            self.assertEqual(reviewed["status"], "done")
            self.assertEqual(reviewed["accepted_scenes"], 2)
            self.assertEqual(reviewed["published_macro_tasks"],
                             oracle["materialized_macro_tasks"])


if __name__ == "__main__":
    unittest.main()
