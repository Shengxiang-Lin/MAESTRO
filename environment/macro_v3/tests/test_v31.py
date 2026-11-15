"""V3.1 regression: coarse-source tasks, genuine handoffs, YAGO and replay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.macro_v3.v31 import (
    compose_v31, choose_real_path, execute_macros, validate_v31,
)
from environment.macro_v3.yago import Edge, KnowledgeGraph
from environment.macro_v3.__main__ import main as cli_main

try:
    import pyoxigraph  # noqa: F401
    RDF = True
except ImportError:
    RDF = False


def graph_fixture() -> KnowledgeGraph:
    p = "http://schema.org/"
    e = "https://yago-knowledge.org/resource/"
    edges = []
    for n in range(12):
        person = e + f"Historian{n}"
        city = e + f"City{n}"
        region = e + f"Region{n}"
        country = e + f"Country{n}"
        edges.extend([
            Edge(person, p + "birthPlace", city),
            Edge(city, p + "containedInPlace", region),
            Edge(region, p + "containedInPlace", country),
            Edge(person, p + "memberOf", e + f"Organization{n}"),
        ])
    return KnowledgeGraph(edges=edges)


class MacroV31Tests(unittest.TestCase):
    def test_coarse_macro_and_fact_replay(self):
        graph = graph_fixture()
        a = compose_v31(graph, seed=17, ordinal=2, min_hops=2, max_hops=3)
        b = compose_v31(graph, seed=17, ordinal=2, min_hops=2, max_hops=3)
        self.assertEqual(a, b)
        self.assertTrue(validate_v31(a, graph)["passed"])
        self.assertGreaterEqual(len(a["world"]["real_fact_path"]), 2)
        self.assertGreaterEqual(len(a["quality_report"]["roles"]), 2)
        for task in a["macro_plan"]["macro_tasks"]:
            if task["solver_strategy"] == "investigate_bundle":
                self.assertEqual(len(task["source_scope"]), 2)
            if task["solver_strategy"] == "investigate_chain":
                self.assertGreaterEqual(len(task["source_scope"]), 3)
        self.assertTrue(all(len(row["internal_trace"]) >= 1
                            for row in a["task_feedback"]))
        self.assertEqual(a["question_authoring"]["status"], "draft")
        self.assertTrue(a["execution_assumptions"]["simulated_media"])

    def test_source_counterfactual_and_case_attributes(self):
        scene = compose_v31(graph_fixture(), seed=5, ordinal=4)
        self.assertTrue(scene["quality_report"]["all_clue_sources_necessary"])
        self.assertIn("archival", scene["question"].lower())
        self.assertNotIn("item meeting visual observation", scene["question"].lower())
        bad = deepcopy(scene)
        sid = bad["question_contract"]["clue_source_ids"][0]
        source = bad["world"]["sources"][sid]
        for obs in source["observations"]:
            obs["values"]["cue"] = bad["question_contract"]["clue_values"][sid]
        with self.assertRaises(ValueError):
            validate_v31(bad, graph_fixture())

    def test_conflict_resolution_and_parallel_fork(self):
        graph = graph_fixture()
        seen = set()
        for i in range(40):
            scene = compose_v31(graph, seed=29, ordinal=i)
            if scene["quality_report"]["uses_conflict_resolution"]:
                answer = scene["gold_answer"][0]
                self.assertEqual(len(answer), 3)
                self.assertIn(answer["evidence_agreement"],
                              {"corroborated", "discrepancy_resolved"})
                seen.add(answer["evidence_agreement"])
                audit = next(t for t in scene["macro_plan"]["macro_tasks"]
                             if t["solver_strategy"] == "review_case")
                self.assertEqual(len(audit["source_scope"]), 3)
                self.assertTrue(validate_v31(scene, graph)["passed"])
        self.assertEqual(seen, {"corroborated", "discrepancy_resolved"})

    def test_exact_real_path_minimum(self):
        graph = graph_fixture()
        import random
        for seed in range(5):
            path = choose_real_path(graph, random.Random(seed),
                                    min_hops=3, max_hops=3)
            self.assertEqual(len(path), 3)
            self.assertEqual(path[0].object, path[1].subject)
        with self.assertRaises(ValueError):
            choose_real_path(
                KnowledgeGraph(edges=[Edge("x", "p", "y")]),
                random.Random(1), min_hops=2, max_hops=3,
            )

    def test_tamper_detected(self):
        graph = graph_fixture()
        scene = compose_v31(graph, seed=21, ordinal=3)
        broken = deepcopy(scene)
        broken["world"]["sources"]["research_request"]["observations"][0][
            "values"]["route"] = "urn:unrelated"
        with self.assertRaises(ValueError):
            validate_v31(broken, graph)


@unittest.skipUnless(RDF, "requires pyoxigraph for RDF loader")
class MacroV31CliTests(unittest.TestCase):
    def test_isolated_cli_variant(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "yago-4.6-facts.ttl"
            path.write_text(
                """@prefix x: <https://example.org/> .
@prefix s: <http://schema.org/> .
x:A s:birthPlace x:B .
x:B s:containedInPlace x:C .
x:C s:containedInPlace x:D .
x:E s:memberOf x:F .
x:F s:location x:G .
""", encoding="utf-8",
            )
            out = root / "generated"
            self.assertEqual(
                cli_main(["--variant", "v3.1", "--yago", str(path),
                          "--output-dir", str(out), "--target", "3",
                          "--seed", "7", "--max-triples", "100"]),
                0,
            )
            result = json.loads((out / "summary.json").read_text())
            self.assertEqual(result["contract"], "maestro.graph_grounded_scene.v3.1")
            self.assertEqual(result["written"], 3)
            self.assertFalse(result["stage2_compatible"])
            data = [json.loads(line) for line in
                    (out / "drafts" / "scenes.jsonl").read_text().splitlines()]
            self.assertEqual(len(data), 3)
            self.assertTrue(all(x["quality_report"]["passed"] for x in data))


if __name__ == "__main__":
    unittest.main()
