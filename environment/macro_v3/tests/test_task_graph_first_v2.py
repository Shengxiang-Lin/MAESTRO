"""Graph-first v2: no redundant event_id port; distinct local relation tasks.

Tests include the v1 shortcut as a NEGATIVE CONTROL: merely carrying an
event_id in the solver DAG is not enough if both events lead to the same
candidate references. v2 rejects that bypass with a re-planned execution.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from environment.generate_question import task_graph_first as v1
from environment.generate_question import task_graph_first_v2 as v2
from environment.generate_question.relational_task_planner import (
    plan_relations_v2, validate_relations_v2,
)
from environment.generate_question.evidence_planner import (
    validate_evidence_v2, audit_information_v2,
)
from environment.generate_question.question import reference_question
from environment.generate_question.quality import (
    compile_and_validate, validate_replay,
)
from environment.generate_question.program_ir import (
    evaluate_program, answer_values,
)
from environment.generate_question.world import compile_program, generate_world
from environment.implement_subtask.realize import realize_scene
from environment.macro_v3.yago import Edge, KnowledgeGraph

try:
    import pyoxigraph  # noqa: F401
    RDF_READY = True
except ImportError:
    RDF_READY = False


def graph_fixture(n=48):
    domain, p = "https://example.org/", "http://schema.org/"
    edges = []
    for i in range(n):
        a, b, c, d, e = (
            domain + f"Person{i}", domain + f"Town{i}",
            domain + f"Region{i}", domain + f"Country{i}",
            domain + f"Organization{i}",
        )
        edges.extend([
            Edge(a, p + "birthPlace", b),
            Edge(b, p + "containedInPlace", c),
            Edge(c, p + "containedInPlace", d),
            Edge(a, p + "memberOf", e),
        ])
    return KnowledgeGraph(
        edges=edges, input_file="synthetic_fixture_yago.ttl",
        scanned=len(edges),
    )


def project_only_bypass(scene: dict) -> list[dict]:
    """Replace event_id handoff with project_id (the old false dependency)."""
    program = deepcopy(scene["gold_program"])
    spec = scene["problem_spec"]
    directory_mids = {
        item["macro_id"] for item in spec["macro_plan"]["macro_tasks"]
        if item["information_operation"] in {"directory_a", "directory_b"}
    }
    ids = {
        node_id for row in program["macro_solver_groups"]
        if row["macro_id"] in directory_mids
        for node_id in row["node_ids"]
    }
    changed = 0
    for op in program["nodes"]:
        if (op["id"] in ids and op["op"] == "join"
                and op.get("left_on") == "event_id"):
            op["left_on"] = "project_id"
            op["right_on"] = "project_id"
            changed += 1
    if changed != 2:
        raise ValueError("test_can_only_bypass_event_keyed_DAG")
    by_project = deepcopy(scene["world"]["sources"])
    for sid in ("link_a", "link_b"):
        by_project[sid]["schema"] = [
            col for col in by_project[sid]["schema"]
            if col != "event_id"
        ]
        for observation in by_project[sid]["observations"]:
            observation["values"].pop("event_id")
    return answer_values(evaluate_program(
        program, sources=by_project,
    ))


class TaskGraphFirstPortTests(unittest.TestCase):
    def test_old_v1_shortcut_is_reproducible(self):
        # The v1 example must remain frozen-replayable, but should fail
        # the new, stronger dependency-level alternative-plan criterion.
        graph = graph_fixture()
        spec = next(
            v1.make_spec(graph, seed=17, ordinal=i, difficulty="medium")
            for i in range(4)
            if v1.sample_graph(
                seed=17, ordinal=i, difficulty="medium"
            )["shape"] == "event_gated_association"
        )
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertEqual(project_only_bypass(scene), scene["gold_answer"])

    def test_v2_all_six_graph_and_relation_combinations(self):
        graph = graph_fixture()
        from environment.generate_question.knowledge_macro import YagoKnowledgeIndex
        index = YagoKnowledgeIndex(graph)
        combinations = set()
        for ordinal in range(6):
            spec = v2.make_spec(
                graph, seed=17, ordinal=ordinal,
                difficulty="medium" if ordinal % 2 == 0 else "hard",
                index=index,
            )
            self.assertFalse(v2.validate_spec(spec))
            plan = spec["task_first_v2_relation_sketch"]
            self.assertFalse(validate_relations_v2(
                plan, spec["task_first_macro_graph"]
            ))
            self.assertFalse(validate_evidence_v2(
                spec["task_first_v2_evidence"], plan,
                spec["task_first_v1_backbone"],
            ))
            scene = compile_and_validate(spec, reference_question(spec))
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(generate_world(spec), scene["world"])
            self.assertEqual(compile_program(spec), scene["gold_program"])
            self.assertEqual(len(realize_scene(scene, strict=True)["macro_tasks"]),
                             len(spec["macro_plan"]["macro_tasks"]))
            self.assertEqual(len(scene["gold_answer"]), 1)
            audit = scene["difficulty_audit"]["macro_first_audit"]
            self.assertTrue(audit["handoff_counterfactuals"][
                "all_handoffs_necessary"])
            self.assertTrue(audit["source_counterfactuals_pass"])
            detail = audit["candidate_information_necessity"]
            self.assertTrue(detail["passed"], detail)
            self.assertTrue(detail["identification_unique"])
            self.assertTrue(all(detail["each_clue_necessary"].values()))
            self.assertTrue(all(detail["each_discovery_necessary"].values()))
            self.assertTrue(all(detail["simulated_directory_necessity"].values()))
            self.assertTrue(all(detail["local_relation_field_necessary"].values()))
            field = plan["event_relation"]["field"]
            predicate = next(
                node for node in scene["gold_program"]["nodes"]
                if node["op"] == plan["event_relation"]["operator"]
                and node.get("field") == field
                and node.get("value") == plan["event_relation"]["value"]
            )
            self.assertTrue(predicate["inputs"])
            shape = spec["task_first_macro_graph"]["shape"]
            combos_key = (shape, plan["event_goal"])
            combinations.add(combos_key)
            if shape == "event_gated_association":
                self.assertTrue(detail["event_handoff_alternative_plan"]["required"])
                self.assertTrue(detail["event_handoff_alternative_plan"][
                    "source_candidate_sets_differ_by_event"])
                self.assertTrue(detail["event_handoff_alternative_plan"][
                    "bypass_introduces_real_alternatives"])
                self.assertNotEqual(project_only_bypass(scene),
                                    scene["gold_answer"])
            else:
                self.assertFalse(detail["event_handoff_alternative_plan"]["required"])
        self.assertEqual(len(combinations), 6)

    def test_provenance_and_frozen_contract_detection(self):
        graph = graph_fixture()
        spec = v2.make_spec(graph, seed=17, ordinal=0,
                            difficulty="medium")
        self.assertTrue(spec["task_first_v2_evidence"]["event_facts_policy"])
        self.assertTrue(all(
            value["origin"] == "yago"
            for value in spec["macro_plan"]["real_fork"]["candidate_forks"].values()
        ))
        tampered = deepcopy(spec)
        tampered["task_first_v2_evidence"]["causal_port"]["field"] = "project_id"
        self.assertTrue(v2.validate_spec(tampered))
        tampered = deepcopy(spec)
        tampered["task_first_v2_relation_sketch"]["event_goal"] = "unrelated"
        self.assertTrue(v2.validate_spec(tampered))
        scene = compile_and_validate(spec, reference_question(spec))
        broken = deepcopy(scene)
        broken["world"]["sources"]["event_ledger"]["observations"][0][
            "values"]["event_year"] = 1901
        self.assertEqual(validate_replay(broken)["status"], "invalid")


@unittest.skipUnless(RDF_READY, "requires RDF loader")
class TaskGraphFirstV2CliTests(unittest.TestCase):
    def test_canonical_cli_with_six_relation_task_combinations(self):
        from environment.config import load_config, args_from_config
        from environment.generate_question.pipeline import generate_questions
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rdf = root / "yago-fixture.ttl"
            entries = [
                "@prefix x: <https://example.org/> .",
                "@prefix s: <http://schema.org/> .",
            ]
            for i in range(48):
                entries.extend([
                    f"x:Person{i} s:birthPlace x:Town{i} .",
                    f"x:Town{i} s:containedInPlace x:Region{i} .",
                    f"x:Region{i} s:containedInPlace x:Country{i} .",
                    f"x:Person{i} s:memberOf x:Organization{i} .",
                ])
            rdf.write_text("\n".join(entries), encoding="utf-8")
            cfg = load_config(
                "environment/macro_v3/tests/fixtures/questions_task_graph_first_v2.yaml"
            )
            cfg["generation"]["knowledge_yago"] = str(rdf)
            cfg["generation"]["output_dir"] = str(root / "generated")
            cfg["generation"]["target"] = 6
            args = args_from_config(cfg)
            args.no_progress = True
            result = generate_questions(args)
            self.assertEqual(result["status"], "done", result)
            self.assertEqual(result["generated"], 6)
            quality = json.loads(
                (args.output_dir / "quality_report.json").read_text()
            )
            self.assertEqual(quality["replay_pass"], 6)
            stats = quality["knowledge_world_grounding"][
                "task_graph_first_v2"
            ]
            self.assertEqual(stats["scenes_checked"], 6)
            self.assertEqual(stats["information_necessity_passed"], 6)
            self.assertEqual(stats["status_or_year_field_necessary"], 6)
            self.assertEqual(stats["event_keyed_cases"], 2)
            self.assertEqual(stats["event_keyed_bypass_blocked"], 2)
            self.assertEqual(stats["unique_local_relation_signatures"], 6)
            self.assertEqual(len(stats["local_event_relation_distribution"]), 2)


if __name__ == "__main__":
    unittest.main()
