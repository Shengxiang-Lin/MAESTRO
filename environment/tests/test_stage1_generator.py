"""Regression tests for the maintained Stage 1 Goal/World/Question pipeline."""
from __future__ import annotations

from collections import Counter
import copy
import json
import unittest

from environment.generate_question.goal_plan import (
    OUTPUT_KINDS, _build_operation_dag, build_measurements,
    evaluate_goal_value, goal_question, operation_subgraph_leq,
    sample_output, validate_operation_graph,
)


def fixture():
    from environment.macro_v3.yago import Edge, KnowledgeGraph
    prefix = "http://schema.org/"
    return KnowledgeGraph(edges=[
        Edge("https://example.org/person", prefix+"birthPlace",
             "https://example.org/town"),
        Edge("https://example.org/town", prefix+"containedInPlace",
             "https://example.org/region"),
        Edge("https://example.org/region", prefix+"containedInPlace",
             "https://example.org/country"),
    ], input_file="stage1-generator-test.ttl", scanned=3)


class Stage1GeneratorTests(unittest.TestCase):
    def test_single_supported_config_and_output_first_partial_order(self):
        from environment.config import DEFAULT_CONFIG, args_from_config, load_config
        self.assertEqual(DEFAULT_CONFIG.name, "questions.yaml")
        config = load_config(DEFAULT_CONFIG)
        args = args_from_config(config)
        self.assertEqual(args.difficulty_mix, "0,50,50")
        self.assertIsNotNone(args.knowledge_yago)
        modified = copy.deepcopy(config)
        modified["generation"]["task_plan_composer"] = "legacy"
        self.assertIn("task_plan_composer", modified["generation"])
        self.assertNotIn("task_plan_composer", config["generation"])
        selections = [sample_output(17, i) for i in range(240)]
        self.assertEqual({q["kind"] for q in selections}, set(OUTPUT_KINDS))
        self.assertGreater(len({
            tuple(q["bindings"].values()) for q in selections
        }), 15)
        for output in selections:
            goal = _build_operation_dag(output)
            self.assertEqual(validate_operation_graph(goal), [])
            seen = set()
            for node in goal["operations"]:
                self.assertTrue(set(node["inputs"]) <= seen)
                seen.add(node["id"])
        smaller = _build_operation_dag({
            "kind": "single_number", "fields": {"answer": "int"},
            "bindings": {"answer": "difference"},
        })
        larger = _build_operation_dag({
            "kind": "structured_multi_field",
            "fields": {"difference": "int", "ratio": "float"},
            "bindings": {"difference": "difference", "ratio": "ratio"},
        })
        self.assertTrue(operation_subgraph_leq(smaller, larger))
        self.assertFalse(operation_subgraph_leq(larger, smaller))

    def test_retired_config_keys_are_rejected(self):
        from environment.config import load_config, DEFAULT_CONFIG
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import yaml
        valid = yaml.safe_load(DEFAULT_CONFIG.read_text())
        for section, key, value in (
            ("generation", "task_plan_composer", "legacy"),
            ("generation", "knowledge_variant", "graph_first"),
            ("generation", "knowledge_world_mode", "witnessed"),
            ("input", "mmqa", "legacy.jsonl"),
        ):
            with self.subTest(section=section, key=key):
                bad = copy.deepcopy(valid)
                bad[section][key] = value
                with TemporaryDirectory() as temp:
                    path = Path(temp) / "invalid.yaml"
                    path.write_text(yaml.safe_dump(bad))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_canonical_cli_construction_with_real_rdf_fixture(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from environment.config import DEFAULT_CONFIG, load_config, args_from_config
        from environment.generate_question.stages import construct_scenes
        from environment.generate_question.question import reference_question
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rdf = root / "yago-4.6-facts.ttl"
            rdf.write_text(
                "@prefix p: <https://example.org/> .\n"
                "@prefix s: <http://schema.org/> .\n"
                "p:person s:birthPlace p:town .\n"
                "p:town s:containedInPlace p:region .\n"
                "p:region s:containedInPlace p:country .\n"
            )
            opts = args_from_config(
                load_config(DEFAULT_CONFIG),
                output_dir=root / "generated", target=3,
                no_progress=True,
            )
            opts.knowledge_yago = rdf
            opts.max_records = 20
            opts.question_model = None
            opts.question_api_base = None
            opts.scenario_model = None
            opts.scenario_api_base = None
            result = construct_scenes(opts)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 3)
            rows = [
                json.loads(line)
                for line in (root / "generated" / "scenes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["problem_spec"]["macro_plan"][
                "composer_policy"] == "output_first" for row in rows))
            self.assertTrue(all(
                row["gold_answer"] and row["question"] ==
                reference_question(row["problem_spec"]) for row in rows
            ))

    def test_seeded_measurements_have_no_constant_answer_shortcut(self):
        seen = {}
        binding = {
            "target_case": "target",
            "candidate_paths": {
                "target": [{"object_label": "Alpha"}],
                "shadow1": [{"object_label": "Beta"}],
                "shadow2": [{"object_label": "Gamma"}],
            },
        }
        for ordinal in range(600):
            goal = _build_operation_dag(sample_output(17, ordinal))
            if not goal["capabilities"]:
                continue
            measured = build_measurements(
                plan={"goal_spec": goal}, binding=binding,
                seed=17, ordinal=ordinal,
            )
            answer = evaluate_goal_value(measured["by_case"]["target"], goal)
            for name, semantic in goal["output_spec"]["bindings"].items():
                value = answer[name] if isinstance(answer, dict) else answer
                seen.setdefault(semantic, set()).add(
                    json.dumps(value, sort_keys=True)
                )
            other = [
                evaluate_goal_value(row, goal)
                for case, row in measured["by_case"].items()
                if case != "target"
            ]
            self.assertTrue(all(x != answer for x in other))
        for semantic in (
            "duration_days", "distance_km", "observation_count",
            "difference", "ratio", "first_count", "second_count",
        ):
            self.assertGreaterEqual(
                len(seen.get(semantic, [])), 8,
                (semantic, seen.get(semantic)),
            )
        self.assertEqual(len(seen["conclusion"]), 2)

    def test_world_gold_replay_necessity_and_stage2(self):
        from environment.generate_question.task_plan import make_spec, validate_spec
        from environment.generate_question.question import reference_question, guard_question
        from environment.generate_question.quality import (
            compile_and_validate, validate_replay,
        )
        from environment.implement_subtask.realize import realize_scene
        from environment.generate_question.pipeline import quality_report

        selected = {}
        for ordinal in range(180):
            output = sample_output(17, ordinal)
            profile = _build_operation_dag(output)
            key = (output["kind"], tuple(profile["capabilities"]))
            selected.setdefault(key, ordinal)
        self.assertEqual({k[0] for k in selected}, set(OUTPUT_KINDS))
        self.assertTrue(any(len(k[1]) >= 2 for k in selected))
        scenes = []
        for (kind, caps), ordinal in selected.items():
            spec = make_spec(
                fixture(), seed=17, ordinal=ordinal,
                difficulty="medium" if ordinal%2 else "hard",
            )
            self.assertEqual(validate_spec(spec), [], (kind, caps))
            plan = spec["macro_plan"]
            self.assertEqual(validate_operation_graph(plan["goal_spec"]), [])
            self.assertLessEqual(len(plan["macro_tasks"]), 11)
            question = reference_question(spec)
            self.assertEqual(guard_question(question, spec), [], (kind, caps))
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            audit = scene["difficulty_audit"]["macro_first_audit"]
            self.assertTrue(audit["candidate_information_necessity"]["passed"])
            self.assertTrue(audit["source_counterfactuals_pass"])
            answer = scene["gold_answer"][0]["answer"]
            if caps:
                world = plan["world_binding"]["world_model"]["goal_measurements"]
                expected = evaluate_goal_value(
                    world["by_case"][plan["world_binding"]["target_case"]],
                    plan["goal_spec"],
                )
                self.assertEqual(answer, expected)
            realized = realize_scene(scene, strict=True)
            self.assertEqual(len(realized["macro_tasks"]),
                             scene["macro_graph"]["macro_task_count"])
            scenes.append(scene)
        report = quality_report(scenes, no_progress=True)
        task_stats = report["knowledge_world_grounding"]["single_task_plan"]
        self.assertEqual(set(task_stats["goal_output_distribution"]), set(OUTPUT_KINDS))
        self.assertFalse(task_stats["answer_shortcut_warning_fields"])
        self.assertEqual(report["evidence_acquisition"][
            "task_plan_keyed_source_contract_mismatches"], 0)

    def test_media_evidence_has_executable_source_affordances(self):
        from environment.generate_question.task_plan import make_spec
        from environment.generate_question.task_plan_runtime import compile_macro_solvers
        from environment.generate_question.auditing.decision_opportunities import (
            audit_decision_opportunities,
        )
        seen = set()
        for ordinal in range(36):
            spec = make_spec(
                fixture(), seed=17, ordinal=ordinal,
                difficulty="medium",
            )
            plan = spec["macro_plan"]
            seen.add(plan["mode"])
            catalog = {s["source_id"]: s for s in spec["sources"]}
            for sid, modality in (("metric_a", "image"),
                                  ("metric_b", "audio")):
                if sid not in plan["goal_spec"]["source_fields"]:
                    continue
                self.assertEqual(catalog[sid]["modality"], modality)
                self.assertEqual(catalog[sid]["access_mode"], "browse")
            if plan["goal_spec"]["source_fields"]:
                compiled = compile_macro_solvers(spec)
                media_scan = {
                    n["source_id"] for n in compiled["nodes"]
                    if n["op"] == "scan" and
                    n.get("source_id") in {"metric_a", "metric_b"}
                }
                self.assertEqual(media_scan, {
                    sid for sid in ("metric_a", "metric_b")
                    if sid in plan["goal_spec"]["source_fields"]
                })
            report = audit_decision_opportunities(spec)
            self.assertGreaterEqual(report["max_parallel_width"], 1)
            self.assertGreaterEqual(report["keyed_handoffs"], 1)
        self.assertIn("trace", seen)
        self.assertIn("event", seen)

    def test_critic_receives_final_goal_and_exact_fields(self):
        from environment.generate_question.task_plan import make_spec
        from environment.generate_question.question import naturalize_one

        class Author:
            def __init__(self, question):
                self.question = question
            def complete_json(self, system, request):
                return {"question": self.question}

        class Critic:
            def __init__(self):
                self.contract = None
            def complete_json(self, system, request):
                self.contract = json.loads(request)["frozen_contract"]
                return {
                    "verdict": "pass",
                    "all_constraints_preserved": True,
                    "no_unlicensed_facts": True,
                    "same_answer_scope": True, "natural": True,
                }

        ordinal = next(i for i in range(100)
                       if sample_output(17, i)["kind"] == "structured_multi_field")
        spec = make_spec(fixture(), seed=17, ordinal=ordinal, difficulty="medium")
        request = (
            "Using the supplied reference card concerning a record in the "
            + spec["scenario"]["topic"] + ", "
            + goal_question(spec["macro_plan"])
        )
        critic = Critic()
        _, result = naturalize_one(
            spec, author=Author(request), critic=critic,
            max_revisions=0,
        )
        self.assertEqual(result["status"], "accepted", result)
        self.assertEqual(
            critic.contract["final_output_spec"],
            spec["macro_plan"]["goal_spec"]["output_spec"],
        )
        self.assertEqual(
            critic.contract["semantic_goal_question"],
            goal_question(spec["macro_plan"]),
        )
        self.assertTrue(critic.contract["goal_operations"])


if __name__ == "__main__":
    unittest.main()
