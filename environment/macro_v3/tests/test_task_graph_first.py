"""Macro-first => relation sketch => YAGO + simulated World => old Stage-1.

These tests prove the abstract DAG is sampled BEFORE facts, and that all
three distinct DAGs execute with the original Gold/FactRefs/Stage2 compiler.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question import task_graph_first as taskfirst
from environment.generate_question.macro_graph_sampler import (
    sample_graph, validate_graph, SHAPES,
)
from environment.generate_question.relational_task_planner import (
    plan_relations, validate_sketch,
)
from environment.generate_question.world_grounder import validate_grounding
from environment.generate_question.evidence_planner import validate_evidence
from environment.generate_question.question import reference_question
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.world import compile_program, generate_world
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    RDF_READY = True
except ImportError:
    RDF_READY = False


def fixture_kg(n=48):
    d, p = "https://example.org/", "http://schema.org/"
    edges = []
    for i in range(n):
        human, town, region, country, institution = (
            d+f"Researcher{i}", d+f"Town{i}",
            d+f"Region{i}", d+f"Country{i}", d+f"Institution{i}",
        )
        edges.extend([
            Edge(human, p+"birthPlace", town),
            Edge(town, p+"containedInPlace", region),
            Edge(region, p+"containedInPlace", country),
            Edge(human, p+"memberOf", institution),
        ])
    return KnowledgeGraph(
        edges=edges, input_file="synthetic-fixture-yago.ttl",
        scanned=len(edges),
    )


class MacroGraphFirstTests(unittest.TestCase):
    def test_samples_three_world_free_typed_partial_orders(self):
        seen = set()
        for shape in SHAPES:
            graph = sample_graph(
                seed=17, ordinal=10, difficulty="hard",
                forced_shape=shape,
            )
            self.assertFalse(validate_graph(graph))
            self.assertEqual(len(graph["nodes"]), 9)
            self.assertTrue(graph["source_free"] and graph["relation_free"])
            self.assertFalse(any(
                "https://" in str(node) or "YAGO-4" in str(node)
                for node in graph["nodes"]
            ))
            sketch = plan_relations(graph)
            self.assertFalse(validate_sketch(sketch, graph))
            self.assertEqual(sketch["knowledge"]["main_chain_hops"], [2, 3])
            self.assertEqual(sketch["synthetic"]["required_event_status"],
                             "completed")
            seen.add(tuple(
                (node["role"], tuple(node["depends_on"]))
                for node in graph["nodes"]
            ))
        self.assertEqual(len(seen), 3)

    def test_real_world_grounding_and_full_legacy_execution(self):
        kg = fixture_kg()
        from environment.generate_question.knowledge_macro import YagoKnowledgeIndex
        index = YagoKnowledgeIndex(kg)
        observed_shapes = set()
        true_edges = set(kg.edges)
        for ordinal in range(6):
            spec = taskfirst.make_spec(
                kg, seed=17, ordinal=ordinal,
                difficulty="hard" if ordinal % 2 else "medium",
                min_hops=2, max_hops=3, index=index,
            )
            self.assertFalse(taskfirst.validate_spec(spec))
            graph = spec["task_first_macro_graph"]
            sketch = spec["task_first_relation_sketch"]
            grounding = spec["task_first_grounding"]
            evidence = spec["task_first_evidence"]
            self.assertFalse(validate_graph(graph))
            self.assertFalse(validate_sketch(sketch, graph))
            self.assertFalse(validate_grounding(grounding, graph, sketch))
            self.assertFalse(validate_evidence(evidence, graph, sketch,
                                                grounding))
            observed_shapes.add(graph["shape"])
            # Every candidate is supported by a witnessed *real* RDF fork.
            for fact in spec["macro_plan"]["real_fork"]["candidate_forks"].values():
                self.assertIn(Edge(
                    fact["subject"], fact["predicate"], fact["object"]
                ), true_edges)
            scene = compile_and_validate(spec, reference_question(spec))
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(generate_world(spec), scene["world"])
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(
                len(realize_scene(scene, strict=True)["macro_tasks"]),
                scene["macro_graph"]["macro_task_count"],
            )
            info = scene["difficulty_audit"]["macro_first_audit"][
                "candidate_information_necessity"
            ]
            self.assertTrue(info["passed"], info)
            self.assertTrue(info["identification_unique"])
            self.assertTrue(all(info["each_clue_necessary"].values()))
            self.assertTrue(all(info["each_discovery_necessary"].values()))
            self.assertTrue(all(info["simulated_directory_necessity"].values()))
            self.assertTrue(info["required_field_necessity"]["event_status"])
            self.assertTrue(info["required_field_necessity"][
                "real_fork_motif_object"])
            self.assertTrue(info["all_discovery_bypass_ambiguous"])
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "handoff_counterfactuals"]["all_handoffs_necessary"])
            self.assertEqual(len(scene["gold_answer"]), 1)
            self.assertIn("association", {
                row["role"] for row in spec["macro_plan"]["macro_tasks"]
            })
            self.assertIn("synthesis", {
                row["role"] for row in spec["macro_plan"]["macro_tasks"]
            })
            self.assertGreater(
                sum(row["values"]["event_status"] == "scheduled"
                    for row in scene["world"]["sources"][
                        "event_ledger"]["observations"]), 0
            )
            self.assertEqual(
                info["context_fields_not_credited"]["event_records"][:2],
                ["organizer_id", "organizer_name"],
            )
        self.assertEqual(observed_shapes, set(SHAPES))

    def test_frozen_graph_and_evidence_tamper_protection(self):
        kg = fixture_kg()
        spec = taskfirst.make_spec(
            kg, seed=17, ordinal=2, difficulty="medium",
        )
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        broken = deepcopy(spec)
        broken["task_first_macro_graph"]["nodes"][-1][
            "depends_on"] = []
        self.assertTrue(taskfirst.validate_spec(broken))
        broken = deepcopy(spec)
        broken["task_first_evidence"]["scheduled_event_records"][
            spec["macro_plan"]["target_case"]]["event_status"] = "completed"
        self.assertTrue(taskfirst.validate_spec(broken))
        broken_scene = deepcopy(scene)
        broken_scene["world"]["sources"]["event_ledger"]["observations"][0][
            "values"]["event_status"] = "cancelled"
        self.assertEqual(validate_replay(broken_scene)["status"], "invalid")


@unittest.skipUnless(RDF_READY, "pyoxigraph required")
class TaskGraphFirstCanonicalCliTests(unittest.TestCase):
    def test_user_generate_questions_cli_with_real_rdf_file(self):
        from environment.config import load_config, args_from_config
        from environment.generate_question.pipeline import generate_questions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ttl = root/"yago-4.6-facts.ttl"
            records = [
                "@prefix x: <https://example.org/> .",
                "@prefix s: <http://schema.org/> .",
            ]
            for i in range(48):
                records.extend([
                    f"x:Researcher{i} s:birthPlace x:Town{i} .",
                    f"x:Town{i} s:containedInPlace x:Region{i} .",
                    f"x:Region{i} s:containedInPlace x:Country{i} .",
                    f"x:Researcher{i} s:memberOf x:Institution{i} .",
                ])
            ttl.write_text("\n".join(records), encoding="utf-8")
            conf = load_config(
                "environment/configs/questions_task_graph_first.yaml"
            )
            conf["generation"]["knowledge_yago"] = str(ttl)
            conf["generation"]["output_dir"] = str(root/"output")
            conf["generation"]["target"] = 3
            args = args_from_config(conf)
            args.no_progress = True
            result = generate_questions(args)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 3)
            qa = json.loads((args.output_dir/"quality_report.json").read_text())
            self.assertEqual(qa["replay_pass"], 3)
            stats = qa["knowledge_world_grounding"]["task_graph_first_v1"]
            self.assertEqual(stats["scenes"], 3)
            self.assertEqual(stats["information_necessity_passed"], 3)
            self.assertEqual(stats["completed_event_status_causal"], 3)
            self.assertEqual(stats["real_yago_fork_causal"], 3)
            self.assertEqual(stats["both_reference_sources_causal"], 3)
            self.assertEqual(stats["distinct_graph_role_topologies"], 3)
            rows = [
                json.loads(line)
                for line in (args.output_dir/"constructed"/"scenes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(
                validate_replay(row)["status"] == "valid"
                for row in rows
            ))


if __name__ == "__main__":
    unittest.main()
