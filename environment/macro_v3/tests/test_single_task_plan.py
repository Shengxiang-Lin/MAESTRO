"""Single Task Plan: direct witnessed YAGO grounding, no old-spec nesting."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question.task_plan import (
    make_spec, validate_spec, CALIBRATED, BLIND_EVENTS, GOAL_QUESTIONS,
)
from environment.generate_question.task_plan_runtime import audit_information
from environment.generate_question.task_plan import (
    SEMANTIC_GOALS, _loaded_yago_unique, _semantic_recipe,
)
from environment.generate_question.question import guard_question
from environment.generate_question.knowledge_macro import YagoKnowledgeIndex
from environment.generate_question.world import compile_program, generate_world
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    RDF_READY = True
except ImportError:
    RDF_READY = False


def graph_fixture(n=54):
    d, p = "https://example.org/", "http://schema.org/"
    edges = []
    for i in range(n):
        person, town, region, country, institute = (
            d+f"Person{i}", d+f"Town{i}", d+f"Region{i}",
            d+f"Country{i}", d+f"Institution{i}",
        )
        edges += [
            Edge(person, p+"birthPlace", town),
            Edge(town, p+"containedInPlace", region),
            Edge(region, p+"containedInPlace", country),
            Edge(person, p+"memberOf", institute),
        ]
    return KnowledgeGraph(
        edges=edges, input_file="fixture-yago-facts.ttl",
        scanned=len(edges),
    )


class DirectTaskPlanTests(unittest.TestCase):
    def test_one_plan_without_any_legacy_backbone(self):
        graph = graph_fixture()
        indexed = set(graph.edges)
        seen_modes, sizes = set(), set()
        from environment.generate_question.knowledge_macro import YagoKnowledgeIndex
        index = YagoKnowledgeIndex(graph)
        for ordinal in range(9):
            spec = make_spec(graph, seed=17, ordinal=ordinal,
                             difficulty="medium" if ordinal % 2 == 0 else "hard",
                             index=index)
            self.assertFalse(validate_spec(spec))
            self.assertEqual(spec["reasoning_pattern"], "task_plan")
            self.assertEqual(spec["family"], "task_plan")
            self.assertEqual(spec["macro_plan"]["difficulty_sampling_policy"],
                             CALIBRATED)
            self.assertEqual(spec["macro_plan"]["evidence_sampling_policy"],
                             BLIND_EVENTS)
            self.assertEqual(spec["macro_plan"]["question_sampling_policy"],
                             GOAL_QUESTIONS)
            self.assertTrue(not any(
                "backbone" in field for field in spec
            ))
            self.assertFalse(any(k in spec for k in (
                "task_first_v1_backbone", "hybrid_backbone_v3",
                "task_first_relation_sketch", "task_first_grounding",
            )))
            plan = spec["macro_plan"]
            seen_modes.add(plan["mode"])
            sizes.add(len(plan["macro_tasks"]))
            self.assertLessEqual(len(plan["macro_tasks"]), 9)
            self.assertGreaterEqual(len(plan["macro_tasks"]), 4)
            for facts in plan["world_binding"]["candidate_paths"].values():
                for fact in facts:
                    self.assertIn(Edge(
                        fact["subject"], fact["predicate"], fact["object"]
                    ), indexed)
            if plan["mode"] == "verify":
                fork = plan["world_binding"]["real_fork"]
                for fact in fork["candidate_forks"].values():
                    self.assertIn(Edge(
                        fact["subject"], fact["predicate"], fact["object"]
                    ), indexed)
            question = reference_question(spec)
            scene = compile_and_validate(spec, question)
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(generate_world(spec), scene["world"])
            realized = realize_scene(scene, strict=True)
            self.assertEqual(len(realized["macro_tasks"]),
                             scene["macro_graph"]["macro_task_count"])
            self.assertEqual(
                scene["gold_answer"], [{
                    "answer": plan["world_binding"]["candidate_paths"][
                        plan["world_binding"]["target_case"]][-1][
                            "object_label"]
                }]
            )
            info = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(info["passed"], info)
            self.assertTrue(info["non_target_field_substitution"]["passed"])
            self.assertNotIn("followed by", question.casefold())
            self.assertTrue(all(info["clue_necessary"].values()))
            self.assertTrue(all(info["discovery_necessary"].values()))
            if plan["condition"]:
                self.assertTrue(info["event_field_necessary"])
            if plan["mode"] == "verify":
                self.assertTrue(info["real_fork_necessary"])
                self.assertTrue(all(info["simulated_sources_necessary"].values()))
            else:
                self.assertIsNone(info["real_fork_necessary"])
        self.assertEqual(seen_modes, {"trace", "event", "verify"})
        self.assertGreaterEqual(len(sizes), 3)

    def test_cross_field_leakage_negative_control(self):
        # Replay the old naming shortcut in the NEW materialized World.
        # The original condition-removal test would still pass, but
        # replacing the condition with event_name must now catch it.
        kg = graph_fixture()
        for ordinal, difficulty in (
            (0, "medium"), (0, "hard"),
            (1, "hard"), (3, "hard"),
        ):
            spec = make_spec(
                kg, seed=17, ordinal=ordinal, difficulty=difficulty,
            )
            self.assertIsNotNone(spec["macro_plan"]["condition"])
            scene = compile_and_validate(
                spec, reference_question(spec)
            )
            audit = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(audit["non_target_field_substitution"]["passed"])
            self.assertTrue(audit["event_field_necessary"])
            case = spec["macro_plan"]["world_binding"]["target_case"]
            from environment.generate_question.task_plan_runtime import _project
            pid = _project(spec["seed"], spec["ordinal"], case)
            target_records = [
                r["values"] for r in scene["world"]["sources"][
                    "event_ledger"]["observations"]
                if r["values"]["project_id"] == pid
            ]
            self.assertEqual(
                len({r["event_name"] for r in target_records}), 1
            )
            # No alternative descriptive field can replace the selected
            # event_status or event_year predicate for this project.
            attacked = deepcopy(scene)
            condition = spec["macro_plan"]["condition"]
            for obs in attacked["world"]["sources"][
                "event_ledger"]["observations"
            ]:
                r = obs["values"]
                eligible = (r["event_status"] == "completed"
                            if condition["field"] == "event_status"
                            else r["event_year"] >= 2025)
                r["event_name"] = (
                    "Qualified project session" if eligible
                    else "Unqualified project session"
                )
            bypass = audit_information(attacked)
            self.assertFalse(bypass["passed"])
            self.assertTrue(bypass["event_field_necessary"])
            self.assertFalse(bypass[
                "non_target_field_substitution"]["tested_fields"][
                    "event_name"]["passed"]
            )
            self.assertTrue(any(
                row["unique_shortcut"]
                for row in bypass["non_target_field_substitution"][
                    "tested_fields"]["event_name"]["attempts"].values()
            ))

    def test_structural_difficulty_stratification(self):
        kg = graph_fixture()
        plans = {"medium": [], "hard": []}
        modes = {"medium": set(), "hard": set()}
        for ordinal in range(12):
            for level in ("medium", "hard"):
                spec = make_spec(
                    kg, seed=17, ordinal=ordinal,
                    difficulty=level,
                )
                plan = spec["macro_plan"]
                plans[level].append(len(plan["macro_tasks"]))
                modes[level].add(plan["mode"])
        self.assertGreater(
            min(plans["hard"]), max(plans["medium"])
        )
        self.assertEqual(modes["medium"], {"trace", "event"})
        self.assertEqual(modes["hard"], {"event", "verify"})

    def test_semantic_goal_questions_are_witnessed_and_scoped(self):
        kg = graph_fixture()
        index = YagoKnowledgeIndex(kg)
        goals = set()
        for ordinal in range(8):
            level = "medium" if ordinal % 2 == 0 else "hard"
            spec = make_spec(
                kg, seed=17, ordinal=ordinal,
                difficulty=level, index=index,
            )
            plan = spec["macro_plan"]
            self.assertEqual(plan["semantic_grounding_policy"], SEMANTIC_GOALS)
            sem = plan["goal"]["semantic"]
            paths = plan["world_binding"]["candidate_paths"]
            target_path = paths[plan["world_binding"]["target_case"]]
            self.assertIsNotNone(_semantic_recipe(target_path))
            self.assertEqual(sem["relation_sequence"], [
                edge["predicate"] for edge in target_path
            ])
            self.assertEqual(sem["type_evidence"],
                             "predicate_signature_heuristic_not_rdf_type")
            self.assertTrue(plan["world_binding"]["semantic_audit"][
                "loaded_yago_singleton_witness"])
            self.assertFalse(plan["world_binding"]["semantic_audit"][
                "rdf_type_verification"])
            for candidate in paths.values():
                self.assertTrue(_loaded_yago_unique(
                    index, [Edge(
                        e["subject"], e["predicate"], e["object"]
                    ) for e in candidate]
                ))
            question = reference_question(spec)
            self.assertIn(sem["question"], question)
            self.assertNotIn("relation target", question)
            self.assertNotIn("followed by", question)
            self.assertFalse(guard_question(question, spec))
            changed = question.replace(
                sem["question"],
                "Which unrelated location is mentioned in these records?"
            )
            self.assertIn(
                "task_plan_semantic_goal_question_missing",
                guard_question(changed, spec),
            )
            self.assertFalse(validate_spec(spec))
            goals.add(sem["profile"])
        self.assertTrue(goals)

    def test_reject_uninterpretable_and_multivalued_yago(self):
        p = "http://schema.org/"
        graph = KnowledgeGraph(
            edges=[
                Edge("https://test.org/a", p+"location", "https://test.org/b"),
                Edge("https://test.org/b", p+"follows", "https://test.org/c"),
            ],
            input_file="bad-meaningful-route.ttl",
            scanned=2,
        )
        with self.assertRaisesRegex(
            ValueError, "no_goal_expressible_singleton_yago"
        ):
            make_spec(graph, seed=17, ordinal=0, difficulty="medium")
        kg = graph_fixture()
        extra = Edge(
            "https://example.org/Person0",
            p+"birthPlace", "https://example.org/AlternativeTown",
        )
        kg.edges.append(extra)
        index = YagoKnowledgeIndex(kg)
        original = next(x for x in kg.edges
                        if x.subject == "https://example.org/Person0"
                        and x.predicate == p+"birthPlace"
                        and x.object != extra.object)
        self.assertFalse(_loaded_yago_unique(index, [original]))

    def test_spec_tamper_and_replay_fail_closed(self):
        kg = graph_fixture()
        spec = make_spec(kg, seed=17, ordinal=1, difficulty="medium")
        broken = deepcopy(spec)
        broken["macro_plan"]["macro_tasks"][-1]["depends_on"] = []
        self.assertTrue(validate_spec(broken))
        broken = deepcopy(spec)
        case = broken["macro_plan"]["world_binding"]["target_case"]
        broken["macro_plan"]["world_binding"]["candidate_paths"][case][0][
            "object"] = "urn:fake"
        self.assertTrue(validate_spec(broken))
        scene = compile_and_validate(spec, reference_question(spec))
        altered = deepcopy(scene)
        one = next(iter(altered["world"]["sources"]))
        altered["world"]["sources"][one]["observations"][0][
            "values"]["cue"] = "unmatched"
        self.assertEqual(validate_replay(altered)["status"], "invalid")


@unittest.skipUnless(RDF_READY, "requires pyoxigraph")
class TaskPlanCanonicalCliTests(unittest.TestCase):
    def test_full_canonical_cli_and_quality_report(self):
        from environment.config import load_config, args_from_config
        from environment.generate_question.pipeline import generate_questions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ttl = root/"yago-fixture.ttl"
            lines = [
                "@prefix x: <https://example.org/> .",
                "@prefix s: <http://schema.org/> .",
            ]
            for i in range(54):
                lines.extend([
                    f"x:Person{i} s:birthPlace x:Town{i} .",
                    f"x:Town{i} s:containedInPlace x:Region{i} .",
                    f"x:Region{i} s:containedInPlace x:Country{i} .",
                    f"x:Person{i} s:memberOf x:Institution{i} .",
                ])
            ttl.write_text("\n".join(lines), encoding="utf-8")
            conf = load_config("environment/configs/questions_task_plan.yaml")
            conf["generation"]["knowledge_yago"] = str(ttl)
            conf["generation"]["output_dir"] = str(root/"output")
            conf["generation"]["target"] = 6
            args = args_from_config(conf)
            args.no_progress = True
            result = generate_questions(args)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 6)
            report = json.loads(
                (args.output_dir/"quality_report.json").read_text()
            )
            self.assertEqual(report["replay_pass"], 6)
            stats = report["knowledge_world_grounding"]["single_task_plan"]
            self.assertEqual(stats["scenes"], 6)
            self.assertEqual(stats["semantic_grounding"]["scenes"], 6)
            self.assertEqual(
                stats["semantic_grounding"]["loaded_graph_uniqueness_verified"], 6
            )
            self.assertGreaterEqual(
                stats["semantic_grounding"]["sampling_attempts"], 6
            )
            self.assertTrue(
                stats["semantic_grounding"]["goal_profile_distribution"]
            )
            self.assertEqual(stats["necessity_passed"], 6)
            self.assertEqual(len(stats["mode_distribution"]), 3)
            self.assertGreaterEqual(stats["macro_task_count_range"][1],
                                    stats["macro_task_count_range"][0])
            scenes = [
                json.loads(x) for x in (
                    args.output_dir/"constructed"/"scenes.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(len(scenes), 6)
            for scene in scenes:
                self.assertEqual(validate_replay(scene)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
