from __future__ import annotations

from copy import deepcopy
import unittest

from environment.macro_v3.composer import compose_scene, replay_scene, validate_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph


def graph_fixture():
    root = "https://yago-knowledge.org/resource/"
    pred = "http://schema.org/"
    edges = []
    for n in range(9):
        a, b, c = (root + f"Person{n}", root + f"City{n}", root + f"Country{n}")
        edges.extend([
            Edge(a, pred + "birthPlace", b),
            Edge(b, pred + "containedInPlace", c),
            Edge(a, pred + "memberOf", root + f"Organization{n}"),
        ])
    return KnowledgeGraph(edges=edges)


class MacroV3Tests(unittest.TestCase):
    def test_deterministic_and_replay(self):
        graph = graph_fixture()
        a = compose_scene(graph, seed=17, ordinal=2, min_hops=2, max_hops=2)
        b = compose_scene(graph, seed=17, ordinal=2, min_hops=2, max_hops=2)
        self.assertEqual(a, b)
        self.assertEqual(replay_scene(a, graph)["passed"], True)
        self.assertTrue(a["macro_graph"]["reference_waves"])
        self.assertGreater(len(a["task_feedback"]), 3)
        self.assertTrue(all(item["fact_refs"] for item in a["task_feedback"]))
        self.assertEqual(a["question_authoring"]["status"], "draft")

    def test_branch_necessity_and_provenance(self):
        graph = graph_fixture()
        scene = compose_scene(graph, seed=5, ordinal=1)
        self.assertTrue(scene["quality_report"]["all_branch_sources_necessary"])
        self.assertTrue(scene["world"]["real_fact_path"])
        altered = deepcopy(scene)
        task = next(t for t in altered["macro_plan"]["macro_tasks"]
                    if t["solver_strategy"] == "filter_candidates")
        source = altered["world"]["sources"][task["source_scope"][0]]
        for obs in source["observations"]:
            obs["values"]["cue"] = task["solver_params"]["cue"]
        with self.assertRaises(ValueError):
            validate_scene(altered)

    def test_optional_branch_creates_real_parallel_dependency(self):
        graph = graph_fixture()
        checked = 0
        for ordinal in range(30):
            scene = compose_scene(graph, seed=4, ordinal=ordinal)
            tasks = scene["macro_plan"]["macro_tasks"]
            if any(t["solver_strategy"] == "check_status" for t in tasks):
                checked += 1
                final = tasks[-1]
                self.assertEqual(final["solver_strategy"], "synthesize")
                self.assertEqual(len(final["depends_on"]), 2)
                self.assertEqual(len(scene["gold_answer"][0]), 2)
                replay_scene(scene, graph)
        self.assertGreater(checked, 0)

    def test_progressive_clue_edges_are_executable(self):
        graph = graph_fixture()
        progressive = []
        for ordinal in range(35):
            scene = compose_scene(graph, seed=91, ordinal=ordinal)
            if scene["quality_report"]["progressive_clue_tasks"]:
                progressive.append(scene)
                replay_scene(scene, graph)
                branches = [t for t in scene["macro_plan"]["macro_tasks"]
                            if t["solver_strategy"] == "filter_candidates"]
                for task in branches:
                    self.assertEqual(set(task["input_contract"]), set(task["depends_on"]))
        self.assertTrue(progressive, "at least one graph must have progressive evidence")

    def test_replay_tamper_detection(self):
        graph = graph_fixture()
        scene = compose_scene(graph, seed=3, ordinal=1)
        scene["world"]["sources"]["entity_index"]["observations"][0]["values"]["entity_uri"] = "urn:tampered"
        with self.assertRaises(ValueError):
            replay_scene(scene, graph)


if __name__ == "__main__":
    unittest.main()
