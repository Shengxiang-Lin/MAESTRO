"""Typed Macro DAG v3.2: grammar, causal edges, YAGO and end-to-end replay."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.macro_v3.graph_generator import generate_graph
from environment.macro_v3.graph_validator import validate_graph
from environment.macro_v3.typed_scene import compose_scene, validate_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph
from environment.macro_v3.__main__ import main as cli_main

try:
    import pyoxigraph  # noqa: F401
    HAS_RDF = True
except ImportError:
    HAS_RDF = False


def kg_fixture():
    root = "https://yago-knowledge.org/resource/"
    ns = "http://schema.org/"
    facts = []
    for i in range(16):
        person = root + f"Subject{i}"
        place = root + f"Place{i}"
        district = root + f"District{i}"
        country = root + f"Country{i}"
        facts.extend([
            Edge(person, ns + "birthPlace", place),
            Edge(place, ns + "containedInPlace", district),
            Edge(district, ns + "containedInPlace", country),
            Edge(person, ns + "memberOf", root + f"Institution{i}"),
            Edge(person, ns + "knows", root + f"Colleague{i}"),
        ])
    return KnowledgeGraph(edges=facts)


class TypedGraphTests(unittest.TestCase):
    def test_many_graphs_vary_but_obey_role_and_partial_order(self):
        signatures = set()
        roles = Counter()
        with_cross_layer = 0
        for ordinal in range(100):
            graph = generate_graph(seed=17, ordinal=ordinal)
            result = validate_graph(graph)
            signatures.add(result["graph_signature"])
            roles.update(result["role_counts"])
            self.assertLessEqual(result["macro_count"], 12)
            self.assertLessEqual(result["max_parallel"], 3)
            self.assertLessEqual(result["wave_count"], 7)
            if result["cross_wave_edges"]:
                with_cross_layer += 1
        self.assertGreater(len(signatures), 12)
        self.assertGreater(with_cross_layer, 0)
        self.assertTrue({"discovery", "identification", "investigation",
                         "verification", "comparison", "synthesis"} <= set(roles))

    def test_type_and_fake_edge_rejection(self):
        a = generate_graph(seed=6, ordinal=2)
        b = deepcopy(a)
        node = next(n for n in b["nodes"] if n["role"] == "investigation")
        node["input_ports"]["subject"]["type"] = "CandidateSet"
        with self.assertRaisesRegex(ValueError, "illegal_macro_role_transition"):
            validate_graph(b)

        c = deepcopy(a)
        node = next(n for n in c["nodes"] if n["role"] == "synthesis")
        node["depends_on"].insert(0, "fake_upstream")
        with self.assertRaisesRegex(ValueError, "unconsumed_or_fabricated"):
            validate_graph(c)

    def test_specific_cross_layer_port_is_real_data(self):
        for ordinal in range(35):
            graph = generate_graph(seed=18, ordinal=ordinal)
            terminal = graph["nodes"][-1]
            if "original_evidence" not in terminal["input_ports"]:
                continue
            link = terminal["input_ports"]["original_evidence"]
            self.assertEqual(link["type"], "EvidenceRefs")
            self.assertEqual(
                next(n["role"] for n in graph["nodes"]
                     if n["macro_id"] == link["from_macro"]),
                "discovery",
            )
            self.assertIn(link["from_macro"], terminal["depends_on"])
            self.assertTrue(validate_graph(graph)["cross_wave_edges"])
            return
        self.fail("Expected a sampled source-to-final cross-layer handoff")


class TypedWorldTests(unittest.TestCase):
    def test_ground_and_replay_real_relation(self):
        kg = kg_fixture()
        sample = compose_scene(kg, seed=17, ordinal=3,
                               min_hops=2, max_hops=3)
        twin = compose_scene(kg, seed=17, ordinal=3,
                             min_hops=2, max_hops=3)
        self.assertEqual(sample, twin)
        audit = validate_scene(sample, kg)
        self.assertTrue(audit["source_necessary"])
        self.assertGreaterEqual(audit["yago_facts_used"], 2)
        self.assertEqual(sample["question_authoring"]["status"], "draft")
        self.assertTrue(sample["gold_answer"][0]["findings"])
        self.assertGreaterEqual(len(sample["macro_graph"]["reference_waves"]), 3)
        for report in sample["task_feedback"]:
            self.assertGreater(len(report["internal_trace"]), 0)
            self.assertEqual(len(report["output_ports"]), len(report["workproduct"]))

    def test_repeated_sampling_contains_comparisons_and_verifications(self):
        kg = kg_fixture()
        seen = Counter()
        for ordinal in range(35):
            scene = compose_scene(kg, seed=22, ordinal=ordinal)
            validate_scene(scene, kg)
            seen.update(scene["quality_report"]["role_counts"])
        self.assertGreater(seen["comparison"], 0)
        self.assertGreater(seen["verification"], 0)

    def test_tampering_detected(self):
        kg = kg_fixture()
        scene = compose_scene(kg, seed=6, ordinal=7)
        changed = deepcopy(scene)
        src = changed["world"]["real_fact_path"][0]["source_id"]
        actual = changed["world"]["real_fact_path"][0]
        target = next(r for r in changed["world"]["sources"][src]["observations"]
                      if r["values"]["subject_uri"] == actual["subject"]
                      and r["values"]["predicate"] == actual["predicate"])
        target["values"]["object_uri"] = "urn:fake:entity"
        with self.assertRaises(ValueError):
            validate_scene(changed, kg)
        wrong = deepcopy(scene)
        wrong["macro_plan"]["nodes"][-1]["input_ports"]["entity"]["type"] = "FactBundle"
        with self.assertRaises(ValueError):
            validate_scene(wrong, kg)


@unittest.skipUnless(HAS_RDF, "pyoxigraph required for RDF loader smoke test")
class TypedCliTests(unittest.TestCase):
    def test_v32_cli_is_opt_in_and_writes_draft_bank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rdf = root / "yago-4.6-facts.ttl"
            rdf.write_text(
                """@prefix x: <https://example.org/> .
@prefix s: <http://schema.org/> .
x:A s:birthPlace x:B .
x:A s:memberOf x:F .
x:B s:containedInPlace x:C .
x:C s:containedInPlace x:D .
x:E s:memberOf x:G .
x:E s:birthPlace x:H .
x:H s:containedInPlace x:J .
""", encoding="utf-8",
            )
            out = root / "v32"
            code = cli_main([
                "--variant", "v3.2", "--yago", str(rdf),
                "--output-dir", str(out), "--target", "3",
                "--min-hops", "1", "--max-hops", "3",
                "--max-triples", "100", "--max-edges", "100",
            ])
            self.assertEqual(code, 0)
            result = json.loads((out / "summary.json").read_text())
            self.assertEqual(result["contract"], "maestro.graph_grounded_scene.v3.2")
            self.assertFalse(result["stage2_compatible"])
            rows = [json.loads(line) for line in
                    (out / "drafts" / "scenes.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(x["quality_report"]["passed"] for x in rows))


if __name__ == "__main__":
    unittest.main()
