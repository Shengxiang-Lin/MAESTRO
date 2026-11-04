"""Native O1 dispatch, per-task simulation, causality and answer tests."""
from __future__ import annotations

import json
import threading
import time
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from environment.tests.test_gold_trajectory import fixture
from environment.question_generation.backend import OpenAICompatibleBackend
from environment.trajectory_simulation.backend import (
    LLMPlanner, PassthroughExecutor, PassthroughPlanner,
)
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.grounding import text_issues
from environment.trajectory_simulation.o1_contract import (
    evidence_prefix, feedback_schema, validate_action,
)
from environment.trajectory_simulation.pipeline import GoldTrajectoryError, synthesize_one


class NativeO1Tests(unittest.TestCase):
    def test_feedback_schema_keeps_assigned_public_evidence_without_private_handles(self):
        import re
        gold = compile_gold_solution(*fixture(dependent=True), full_environment=True)
        discovery = next(task for task in gold["tasks"] if task["kind"] == "discover")
        scope = [discovery]
        prefix = evidence_prefix(scope)
        pattern = feedback_schema({"scopes": scope})["properties"]["finish_result"]["properties"]["result"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, prefix + "\nInspection is still needed."))
        self.assertIsNone(re.fullmatch(pattern, "No matches."))
        self.assertNotIn("discover:", prefix)
        self.assertNotIn("r:second", prefix)
        self.assertIn("contents not inspected", prefix)

    def test_native_llm_uses_action_schema_and_preserves_temperature(self):
        planner = LLMPlanner(OpenAICompatibleBackend("http://localhost/v1", "test", temperature=.2))
        payload = {"feasible_work": [{
            "simulation_subtask_id": "subtask:s1"}], "evidence_sufficient": False}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"action":"delegate_task"}'},
                         "finish_reason": "stop"}]}).encode()
        with patch("environment.trajectory_simulation.backend.urllib.request.urlopen",
                   return_value=response) as request:
            planner.decide(payload, "test")
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["temperature"], .2)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertIn("task_bindings", schema["properties"])
        self.assertNotIn("task_bindings", schema["properties"]["params"]["properties"])

    def test_public_reply_cannot_omit_facts_despite_correct_private_audit(self):
        class Simulator(PassthroughExecutor):
            def respond(self, payload, prompt):
                result = super().respond(payload, prompt)
                result["finish_result"]["result"] = (
                    payload["scopes"][0]["resource_id"] + ": no matching entries")
                return result
        with self.assertRaisesRegex(GoldTrajectoryError, "result_omits_assigned_fact"):
            synthesize_one(*fixture(), PassthroughPlanner(), Simulator(), max_attempts=1)

    def test_numeric_units_and_terminal_punctuation_do_not_split_values(self):
        self.assertEqual(text_issues("Height is 1.7609m.", [{"height": 1.7609}]), [])
        self.assertIn("unsupported_numeric_claim_in_summary",
                      text_issues("Height is 1.7610m.", [{"height": 1.7609}]))
        self.assertIn("unsupported_numeric_claim_in_summary",
                      text_issues("Population is 1191.", [{"population": 1190}]))

    def test_correct_answer_cannot_complete_before_any_observation(self):
        class Main(PassthroughPlanner):
            def decide(self, payload, prompt):
                return {"action": "complete", "reasoning": "",
                        "params": {"answer": '["B"]'}}
        self.assertEqual(compile_gold_solution(*fixture())["required_fact_ids"], [])
        with self.assertRaisesRegex(GoldTrajectoryError, "decision_evidence_incomplete"):
            synthesize_one(*fixture(), Main(), PassthroughExecutor(), max_attempts=1)

    def test_private_bindings_inside_params_get_actionable_rejection(self):
        action = {
            "action": "delegate_task", "reasoning": "",
            "params": {"tasks": [{}], "task_bindings": [
                {"task_index": 0, "subtask_ids": ["subtask:s1"]}]},
        }
        self.assertEqual(validate_action(action, [], False, ["B"], {})[1],
                         ["move_task_bindings_to_top_level_outside_params"])

    def test_wire_format_matches_o1_delegate_and_complete(self):
        output = synthesize_one(*fixture(), PassthroughPlanner(), PassthroughExecutor())
        self.assertTrue(output["success"])
        self.assertNotIn("steps", output)
        self.assertEqual([step["action"] for step in output["trajectory"]],
                         ["delegate_task", "complete"])
        step = output["trajectory"][0]
        self.assertEqual(len(step["params"]["tasks"]), 2)
        for task, reply in zip(step["params"]["tasks"], step["result"]["subtask_results"]):
            self.assertEqual(task["model"], "model_1")
            self.assertEqual(reply["task_instruction"], task["task_instruction"])
            self.assertTrue(reply["done"])
            self.assertEqual(reply["finish_result"]["status"], "done")
            self.assertIsInstance(reply["finish_result"]["result"], str)
            self.assertIsInstance(reply["trace_summary"], str)
            self.assertNotIn("grounded_claims", reply)
            self.assertNotIn("trace", reply)
        public = [{key: step[key] for key in ("action", "params", "result", "state_before")}
                  for step in output["trajectory"]]
        self.assertNotIn("task_bindings", json.dumps(public))
        self.assertNotIn("gold_answer_claims", json.dumps(public))
        self.assertEqual(json.loads(output["trajectory"][-1]["params"]["answer"]), ["B"])

    def test_main_can_dispatch_only_part_of_ready_work(self):
        class PartialMain(PassthroughPlanner):
            def decide(self, payload, prompt):
                result = super().decide(payload, prompt)
                if result["action"] == "delegate_task":
                    result["params"]["tasks"] = result["params"]["tasks"][:1]
                    result["task_bindings"] = result["task_bindings"][:1]
                return result
        output = synthesize_one(*fixture(), PartialMain(), PassthroughExecutor())
        delegates = [step for step in output["trajectory"] if step["action"] == "delegate_task"]
        self.assertEqual([len(step["params"]["tasks"]) for step in delegates], [1, 1])
        self.assertEqual(len(delegates[1]["state_before"]["history"]), 1)

    def test_one_simulator_request_per_task_not_per_source(self):
        class GroupedMain(PassthroughPlanner):
            def decide(self, payload, prompt):
                result = super().decide(payload, prompt)
                if result["action"] == "delegate_task":
                    result["params"]["tasks"] = result["params"]["tasks"][:1]
                    result["task_bindings"] = [{
                        "task_index": 0,
                        "subtask_ids": [item["simulation_subtask_id"] for item in payload["feasible_work"]],
                    }]
                return result
        class Simulator(PassthroughExecutor):
            def __init__(self):
                self.calls = []
            def respond(self, payload, prompt):
                self.calls.append(len(payload["scopes"]))
                return super().respond(payload, prompt)
        simulator = Simulator()
        output = synthesize_one(*fixture(), GroupedMain(), simulator)
        self.assertEqual(simulator.calls, [2])
        self.assertEqual(output["trajectory"][0]["result"]["total_subtasks"], 1)
        self.assertEqual(len(output["trajectory"][0]["result"]["subtask_results"]), 1)

    def test_discovery_reply_is_also_generated_by_simulator_without_contents(self):
        outer = self
        class Simulator(PassthroughExecutor):
            def __init__(self):
                self.kinds = []
            def respond(self, payload, prompt):
                self.kinds.extend(task["kind"] for task in payload["scopes"])
                for task in payload["scopes"]:
                    if task["kind"] == "discover":
                        outer.assertNotIn("content", task["observation"])
                        outer.assertNotIn("A has r:second B", prompt)
                return super().respond(payload, prompt)
        simulator = Simulator()
        output = synthesize_one(*fixture(dependent=True), PassthroughPlanner(), simulator)
        self.assertEqual(simulator.kinds, ["inspect", "discover", "inspect"])
        self.assertEqual(len(output["trajectory"]), 4)

    def test_main_prompt_has_history_but_no_oracle_or_future_observations(self):
        outer = self
        class Main(PassthroughPlanner):
            def decide(self, payload, prompt):
                outer.assertNotIn("offline_answer", prompt)
                outer.assertNotIn("gold_answer_claims", prompt)
                outer.assertNotIn("A has r:second B", prompt)
                outer.assertNotIn("symbolic_trace", prompt)
                if payload["public_state"]["history"]:
                    outer.assertIn("finish_result", prompt)
                    outer.assertIn("trace_summary", prompt)
                    outer.assertNotIn("_internal_metadata", json.dumps(payload["public_state"]["history"]))
                return super().decide(payload, prompt)
        synthesize_one(*fixture(dependent=True), Main(), PassthroughExecutor())

    def test_correct_final_answer_is_not_vetoed_by_a_summary_critic(self):
        class Main(PassthroughPlanner):
            def review(self, prompt):
                raise AssertionError("Native Main has no separate summary critic")
            def summarize(self, payload, prompt):
                raise AssertionError("Native Main has no separate summary stage")
        class Simulator(PassthroughExecutor):
            def review(self, prompt):
                raise AssertionError("Private facts have deterministic validation")
        output = synthesize_one(*fixture(), Main(), Simulator())
        self.assertTrue(output["validation"]["final_answer_verified"])

    def test_wrong_final_answer_still_fails_oracle_validation(self):
        class Main(PassthroughPlanner):
            def decide(self, payload, prompt):
                result = super().decide(payload, prompt)
                if result["action"] == "complete":
                    result["params"]["answer"] = '["C"]'
                return result
        with self.assertRaisesRegex(GoldTrajectoryError, "final_answer_mismatch"):
            synthesize_one(*fixture(), Main(), PassthroughExecutor(), max_attempts=1)

    def test_corrupted_simulator_claims_are_rejected(self):
        class Simulator(PassthroughExecutor):
            def respond(self, payload, prompt):
                result = super().respond(payload, prompt)
                result["grounded_claims"][payload["scopes"][0]["subtask_id"]] = []
                return result
        with self.assertRaisesRegex(GoldTrajectoryError, "scoped_claims_not_grounded"):
            synthesize_one(*fixture(), PassthroughPlanner(), Simulator(), max_attempts=1)

    def test_parallel_calls_follow_main_chosen_task_count(self):
        class Simulator(PassthroughExecutor):
            def __init__(self):
                self.lock = threading.Lock()
                self.active = self.peak = 0
            def respond(self, payload, prompt):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(.025)
                result = super().respond(payload, prompt)
                with self.lock:
                    self.active -= 1
                return result
        simulator = Simulator()
        synthesize_one(*fixture(), PassthroughPlanner(), simulator, subagent_workers=2)
        self.assertEqual(simulator.peak, 2)

    def test_alternative_provider_can_replace_minimum_proof_source(self):
        realized, world, rendered = fixture(dependent=True)
        resources = realized["capability_world"]["resources"]
        resources[0].update(requires_anchors=[], produces_anchors=["a"])
        resources[1].update(requires_anchors=["a"], produces_anchors=[])
        resources.append({**resources[0], "resource_id": "s3"})
        world["world"]["sources"]["s3"] = {
            "source_id": "s3", "source_type": "text", "fact_ids": ["f1"], "metadata": {}}
        rendered.sources.append(SimpleNamespace(
            source_id="s3", source_type="text", title="Source 3", content="A has r:first B"))
        class Main(PassthroughPlanner):
            def decide(self, payload, prompt):
                ready = deepcopy(payload["feasible_work"])
                if not payload["evidence_sufficient"]:
                    preferred = [item for item in ready if item["resource_id"] != "s1"]
                    payload = {**payload, "feasible_work": preferred[:1]}
                return super().decide(payload, prompt)
        output = synthesize_one(realized, world, rendered, Main(), PassthroughExecutor())
        self.assertEqual(output["validation"]["inspected_resource_ids"], ["s2", "s3"])
        self.assertEqual(output["gold_supervision"]["acquisition_alternatives"]["s2"],
                         [["s1", "s3"]])
        self.assertIn("s1", output["gold_supervision"]["available_resource_ids"])
        self.assertTrue(output["success"])

    def test_unknown_or_duplicate_scope_is_rejected(self):
        gold = compile_gold_solution(*fixture(), full_environment=True)
        ready = gold["tasks"]
        payload = {
            "public_state": {"question": gold["question"]},
            "feasible_work": [{
                "kind": task["kind"], "resource_id": task["resource_id"],
                "simulation_subtask_id": task["subtask_id"], "suggested_tools": ["ExtractUrlContentAction"],
            } for task in ready],
            "evidence_sufficient": False,
        }
        action = PassthroughPlanner().decide(payload, "")
        action["task_bindings"][1]["subtask_ids"] = ["subtask:s1"]
        self.assertIn("scope_assigned_more_than_once",
                      validate_action(action, ready, False, ["B"], gold["answer_contract"])[1])
        action["task_bindings"][1]["subtask_ids"] = ["subtask:unknown"]
        self.assertFalse(validate_action(action, ready, False, ["B"], gold["answer_contract"])[0])

    def test_negated_unavailable_read_and_unread_state_are_allowed(self):
        self.assertEqual(text_issues("Contents remain unread; future inspection is pending."), [])
        self.assertEqual(text_issues("Did not read unavailable future contents."), [])
        self.assertIn("claims_unavailable_observation",
                      text_issues("Read unavailable future contents."))


if __name__ == "__main__":
    unittest.main()
