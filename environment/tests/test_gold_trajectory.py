"""Offline regressions: first-stage fragments -> O1 Main/Sub-Agent gold episodes."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from environment.trajectory_simulation.backend import PassthroughPlanner, PassthroughExecutor
from environment.trajectory_simulation.compiler import (
    GoldCompilationError, compile_gold_solution, private_guidance, ready_fragments,
)
from environment.trajectory_simulation.contract import (
    feedback_schema, answers_equal, validate_feedback,
)
from environment.trajectory_simulation.pipeline import (
    GoldTrajectoryError, plan_trajectories_jsonl, synthesize_one,
    synthesize_trajectories_jsonl,
)


def fixture(*, dependent=False, terminal_evidence=False):
    """Composition links are the authoritative subtask boundaries."""
    nodes = [
        {"node_id": 0, "op": "FIND", "dependencies": [], "inputs": ["entity:A"]},
        {"node_id": 1, "op": "RELATE", "dependencies": [0],
         "inputs": ["r:first", "forward"]},
        {"node_id": 2, "op": "RELATE", "dependencies": [1] if dependent else [0],
         "inputs": ["r:second", "forward"]},
    ]
    if dependent:
        nodes.append({"node_id": 3, "op": "QUERY_NAME", "dependencies": [2], "inputs": []})
    else:
        nodes.extend([
            {"node_id": 3, "op": "INTERSECT", "dependencies": [1, 2], "inputs": []},
            {"node_id": 4, "op": "QUERY_NAME", "dependencies": [3], "inputs": []},
        ])
    if terminal_evidence:
        nodes.append({
            "node_id": 5, "op": "QUERY_ATTR", "dependencies": [4],
            "inputs": ["r:height"],
        })
    last = nodes[-1]["node_id"]
    record = {
        "program": {
            "nodes": nodes,
            "metadata": {"deep_dependency": {
                "composition_links": [
                    {"fragment": 0, "stage": 0, "producer_node_id": None,
                     "output_node_id": 1},
                    {"fragment": 1, "stage": 1 if dependent else 0,
                     "producer_node_id": 1 if dependent else None,
                     "output_node_id": 2},
                ],
                "orchestration_motif": "chain" if dependent else "fork_join",
            }},
        },
        "world": {
            "entities": {
                "A": "schema:Person", "B": "schema:Person",
                **({"C": "schema:Person"} if dependent else {}),
            },
            "facts": [
                {"fact_id": "f1", "subject": "A", "predicate": "r:first",
                 "object": "B", "object_kind": "entity", "source_id": "s1"},
                {"fact_id": "f2", "subject": "B" if dependent else "A",
                 "predicate": "r:second",
                 "object": "C" if dependent else "B",
                 "object_kind": "entity", "source_id": "s2"},
            ] + ([{
                "fact_id": "f3", "subject": "B", "predicate": "r:height",
                "object": 175, "object_kind": "number", "source_id": "s3",
            }] if terminal_evidence else []),
            "sources": {
                "s1": {"source_id": "s1", "source_type": "text",
                       "fact_ids": ["f1"], "metadata": {}},
                "s2": {"source_id": "s2", "source_type": "text",
                       "fact_ids": ["f2"], "metadata": {}},
                **({"s3": {"source_id": "s3", "source_type": "table",
                           "fact_ids": ["f3"], "metadata": {}}}
                   if terminal_evidence else {}),
            },
            "metadata": {},
        },
    }
    rendered_sources = [
        SimpleNamespace(source_id="s1", source_type="text",
                        title="First reference", content="Alice first Bob"),
        SimpleNamespace(source_id="s2", source_type="text",
                        title="Second reference",
                        content="Bob second Carol" if dependent else "Alice second Bob"),
    ]
    resources = [
        {"resource_id": "s1", "source_type": "text", "visibility": "provided",
         "anchor_dependency_resource_ids": []},
        {"resource_id": "s2", "source_type": "text",
         "visibility": "discoverable" if dependent else "provided",
         "anchor_dependency_resource_ids": ["s1"] if dependent else []},
    ]
    if terminal_evidence:
        resources.append({
            "resource_id": "s3", "source_type": "table", "visibility": "provided",
            "anchor_dependency_resource_ids": [],
        })
        rendered_sources.append(SimpleNamespace(
            source_id="s3", source_type="table",
            title="Height lookup", content="Bob has height 175",
        ))
    realized = {
        "task_id": "world-test",
        "question": "Find the result of the composed relations for Alice.",
        "capability_world": {"resources": resources, "binding_audit": {}},
    }
    rendered = SimpleNamespace(
        task_id="world-test",
        answer=[175] if terminal_evidence else ["Carol"] if dependent else ["Bob"],
        hidden_metadata={
            "entity_aliases": {"A": "Alice", "B": "Bob", "C": "Carol"},
            "literal_surfaces": {},
            "literal_surfaces_by_predicate": {},
        },
        sources=rendered_sources,
    )
    return realized, record, rendered


class FragmentCompilationTests(unittest.TestCase):
    def test_parallel_fragments_share_question_but_not_sources(self):
        question, world, rendered = fixture()
        gold = compile_gold_solution(question, world, rendered)
        self.assertEqual(len(gold["fragments"]), 2)
        self.assertEqual(gold["waves"], [["fragment:0", "fragment:1"]])
        self.assertEqual(gold["fragments"][0]["resource_ids"], ["s1"])
        self.assertEqual(gold["fragments"][1]["resource_ids"], ["s2"])
        self.assertEqual(gold["fragments"][0]["expected_answer"], ["Bob"])
        self.assertEqual(gold["fragments"][1]["expected_answer"], ["Bob"])
        self.assertEqual(gold["final_answer"], ["Bob"])
        self.assertEqual(gold["policy"], "first_stage_composition_links_fragment_gold")

    def test_dependency_and_resource_anchor_are_respected(self):
        gold = compile_gold_solution(*fixture(dependent=True))
        self.assertEqual(gold["waves"], [["fragment:0"], ["fragment:1"]])
        self.assertEqual(gold["fragments"][1]["depends_on"], ["fragment:0"])
        self.assertEqual(gold["fragments"][1]["resource_ids"], ["s2"])
        self.assertEqual(gold["fragments"][1]["support_resource_ids"], ["s1", "s2"])
        self.assertEqual(gold["fragments"][1]["expected_answer"], ["Carol"])

    def test_terminal_evidence_gets_its_own_fragment(self):
        gold = compile_gold_solution(*fixture(terminal_evidence=True))
        self.assertEqual(gold["fragments"][-1]["fragment_id"], "fragment:terminal")
        self.assertEqual(gold["fragments"][-1]["expected_answer"], [175])
        self.assertEqual(gold["fragments"][-1]["resource_ids"], ["s3"])
        self.assertEqual(gold["fragments"][-1]["support_resource_ids"], ["s1", "s2", "s3"])
        self.assertEqual(gold["waves"][-1], ["fragment:terminal"])

    def test_missing_composition_metadata_rejected_without_fallback(self):
        question, world, rendered = fixture()
        world["program"]["metadata"] = {}
        with self.assertRaisesRegex(GoldCompilationError, "composition_links"):
            compile_gold_solution(question, world, rendered)

    def test_private_guidance_has_operations_not_future_answers(self):
        gold = compile_gold_solution(*fixture(dependent=True))
        prompt_state = private_guidance(gold, set())
        self.assertEqual(len(prompt_state["ready"]), 1)
        self.assertNotIn("expected_answer", json.dumps(prompt_state))
        self.assertNotIn("Carol", json.dumps(prompt_state))
        self.assertNotIn("observations", json.dumps(prompt_state))
        self.assertFalse(prompt_state["may_complete"])


class FragmentTrajectoryTests(unittest.TestCase):
    def test_public_o1_state_and_private_supervision_are_separate(self):
        result = synthesize_one(*fixture(dependent=True),
                                PassthroughPlanner(), PassthroughExecutor())
        self.assertTrue(result["validation"]["accepted"])
        self.assertEqual([step["action"] for step in result["trajectory"]],
                         ["delegate_task", "delegate_task", "complete"])
        first = result["trajectory"][0]
        self.assertEqual(set(first["state_before"]),
                         {"question", "history", "available_tools"})
        self.assertEqual(first["state_before"]["history"], [])
        second = result["trajectory"][1]
        self.assertEqual(len(second["state_before"]["history"]), 1)
        public = json.dumps(result["trajectory"])
        self.assertNotIn("fragment:0", public)
        self.assertNotIn("symbolic_trace", public)
        self.assertNotIn("private_gold_guidance", public)
        self.assertEqual(result["gold_supervision"]["parallel_waves"],
                         [["fragment:0"], ["fragment:1"]])
        self.assertEqual(result["trajectory"][-1]["params"]["answer"], '["Carol"]')

    def test_subagent_result_is_symbolically_fixed_not_model_copied_facts(self):
        result = synthesize_one(*fixture(),
                                PassthroughPlanner(), PassthroughExecutor())
        replies = result["trajectory"][0]["result"]["subtask_results"]
        self.assertEqual([reply["finish_result"]["result"] for reply in replies],
                         ['["Bob"]', '["Bob"]'])
        self.assertNotIn("grounded_claims", json.dumps(replies))
        self.assertNotIn("fact_id", json.dumps(replies))
        self.assertIn("trace_summary", replies[0])
        schema = feedback_schema()
        self.assertEqual(set(schema["properties"]), {"summary", "trace_summary"})

    def test_parallel_subagents_are_independent_requests(self):
        class SlowExecutor(PassthroughExecutor):
            def __init__(self):
                self.lock = threading.Lock()
                self.active = self.peak = 0
            def respond(self, payload, prompt):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(.03)
                value = super().respond(payload, prompt)
                with self.lock:
                    self.active -= 1
                return value
        executor = SlowExecutor()
        result = synthesize_one(*fixture(), PassthroughPlanner(), executor,
                                subagent_workers=2)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(executor.peak, 2)

    def test_invalid_number_or_private_handle_rejected(self):
        class BadExecutor(PassthroughExecutor):
            def respond(self, payload, prompt):
                return {"summary": "Found 99999 unobserved records.",
                        "trace_summary": "Read fragment:0."}
        with self.assertRaisesRegex(GoldTrajectoryError, "unsupported_numeric_claim"):
            synthesize_one(*fixture(), PassthroughPlanner(), BadExecutor(),
                           max_attempts=1)

    def test_main_cannot_drop_gold_parallel_fragment(self):
        class IncompletePlanner(PassthroughPlanner):
            def decide(self, payload, prompt):
                value = super().decide(payload, prompt)
                if value["action"] == "delegate_task":
                    value["params"]["tasks"].pop()
                return value
        with self.assertRaisesRegex(GoldTrajectoryError,
                                    "parallel_gold_fragment_count_mismatch"):
            synthesize_one(*fixture(), IncompletePlanner(),
                           PassthroughExecutor(), max_attempts=1)

    def test_final_answer_checked_after_all_fragments(self):
        class WrongPlanner(PassthroughPlanner):
            def decide(self, payload, prompt):
                value = super().decide(payload, prompt)
                if value["action"] == "complete":
                    value["params"]["answer"] = '["Incorrect"]'
                return value
        with self.assertRaisesRegex(GoldTrajectoryError, "final_answer_mismatch"):
            synthesize_one(*fixture(), WrongPlanner(),
                           PassthroughExecutor(), max_attempts=1)

    def test_answer_contract_supports_set_list_numeric_and_boolean(self):
        self.assertTrue(answers_equal('["B","A"]', ["A", "B"], {"collection_mode": "set"}))
        self.assertFalse(answers_equal('["B","B"]', ["A", "B"], {"collection_mode": "set"}))
        self.assertTrue(answers_equal("1,000", 1000, {}))
        self.assertFalse(answers_equal("true", 1, {}))
        self.assertTrue(answers_equal("true", True, {}))

    def test_jsonl_plan_and_synthesis(self):
        question, world, rendered = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qs = root/"questions.jsonl"
            qs.write_text(json.dumps(question)+"\n", encoding="utf-8")
            mapping = {"world-test": (world, rendered)}
            with patch("environment.trajectory_simulation.pipeline._world_map",
                       return_value=mapping):
                inspection = plan_trajectories_jsonl(
                    root/"worlds.jsonl", qs, root/"plan.jsonl")
                self.assertEqual(inspection["ready"], 1)
                production = synthesize_trajectories_jsonl(
                    root/"worlds.jsonl", qs, root/"episodes.jsonl",
                    planner=PassthroughPlanner(), executor=PassthroughExecutor(),
                    progress=False)
                self.assertEqual(production["accepted"], 1)
                output = json.loads((root/"episodes.jsonl").read_text().strip())
                self.assertTrue(output["validation"]["final_answer_verified"])
                self.assertEqual(output["gold_supervision"]["policy"],
                                 "first_stage_composition_links_fragment_gold")


if __name__ == "__main__":
    unittest.main()
