"""Canonical YAGO knowledge bridge: actual legacy Program/Quality/Stage-2 replay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question.knowledge_macro import (
    KNOWLEDGE_VERSION, make_spec, validate_spec,
)
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.compiler import run_macro_tasks
from environment.generate_question.program_ir import evaluate_program, answer_values
from environment.generate_question.world import generate_world, compile_program
from environment.generate_question.macro_contract import validate_handoffs
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    HAS_RDF = True
except ImportError:
    HAS_RDF = False


def graph_fixture() -> KnowledgeGraph:
    e, p = "https://yago-knowledge.org/resource/", "http://schema.org/"
    edges = []
    for i in range(16):
        person = e + f"Researcher{i}"
        location = e + f"City{i}"
        region = e + f"Region{i}"
        country = e + f"Country{i}"
        edges.extend([
            Edge(person, p + "birthPlace", location),
            Edge(location, p + "containedInPlace", region),
            Edge(region, p + "containedInPlace", country),
            Edge(person, p + "memberOf", e + f"University{i}"),
        ])
    return KnowledgeGraph(edges=edges, input_file="fixture-yago4.6-facts.ttl",
                          scanned=len(edges))


class KnowledgeBridgeTests(unittest.TestCase):
    def test_compile_and_strict_replay(self):
        graph = graph_fixture()
        for ordinal in range(12):
            spec = make_spec(graph, seed=17, ordinal=ordinal,
                             difficulty="medium" if ordinal % 2 else "hard")
            self.assertFalse(validate_spec(spec))
            question = reference_question(spec)
            scene = compile_and_validate(spec, question)
            self.assertEqual(scene["difficulty_audit"]["status"], "valid")
            self.assertTrue(scene["difficulty_audit"]["replay_exact"])
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "handoff_counterfactuals"]["all_handoffs_necessary"])
            self.assertTrue(scene["difficulty_audit"]["macro_first_audit"][
                "source_counterfactuals_pass"])
            self.assertEqual(scene["gold_answer"], [{"answer": spec["macro_plan"][
                "real_facts"][-1]["object_label"]}])
            self.assertEqual(
                set(x["macro_role"] for x in scene["macro_graph"]["tasks"]),
                {"discovery", "identification", "investigation"})
            self.assertGreaterEqual(scene["macro_graph"]["macro_task_count"], 4)
            self.assertEqual(
                set(spec["macro_plan"]["source_schemas"]),
                set(scene["world"]["sources"]))
            oracle = run_macro_tasks(
                scene["macro_graph"], sources=scene["world"]["sources"])
            self.assertEqual(oracle["answer"], scene["gold_answer"])
            self.assertEqual(len(realize_scene(scene, strict=True)["macro_tasks"]),
                             scene["macro_graph"]["macro_task_count"])

    def test_freeze_and_tamper_detection(self):
        graph = graph_fixture()
        spec = make_spec(graph, seed=8, ordinal=3, difficulty="hard")
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(generate_world(spec), scene["world"])
        self.assertEqual(compile_program(spec), scene["gold_program"])
        broken = deepcopy(scene)
        broken["world"]["sources"]["relation_1"]["observations"][0][
            "values"][list(broken["world"]["sources"]["relation_1"][
                "observations"][0]["values"])[-1]] = "urn:tampered"
        self.assertEqual(validate_replay(broken)["status"], "invalid")
        changed = deepcopy(spec)
        changed["macro_plan"]["real_facts"][0]["object"] = "urn:forged"
        self.assertTrue(validate_spec(changed))

    def test_different_macro_granularity(self):
        graph = graph_fixture()
        plans = [
            make_spec(graph, seed=21, ordinal=i, difficulty="hard")["macro_plan"]
            for i in range(25)
        ]
        self.assertGreater(len({len(p["macro_tasks"]) for p in plans}), 1)
        self.assertGreater(len({
            (len(p["macro_tasks"]), len(p["real_facts"]),
             tuple(map(tuple, p["relation_groups"])))
            for p in plans
        }), 2)


@unittest.skipUnless(HAS_RDF, "pyoxigraph required for YAGO RDF parsing")
class KnowledgeCanonicalCliTests(unittest.TestCase):


if __name__ == "__main__":
    unittest.main()
