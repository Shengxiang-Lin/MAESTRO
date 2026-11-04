"""No-network two-role gold trajectory simulation and grounding tests."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from environment.trajectory_simulation.backend import (
    PassthroughExecutor, PassthroughPlanner,
)
from environment.trajectory_simulation.compiler import (
    compile_gold_solution, public_plan, ready_subtasks,
)
from environment.trajectory_simulation.pipeline import (
    GoldTrajectoryError, plan_trajectories_jsonl,
    _synthesize_reference_one as synthesize_one, synthesize_trajectories_jsonl,
)
from environment.trajectory_simulation.validator import validate_subagent_result


def fixture(dependent=False):
    record = {
        "program": {
            "nodes": [
                {"node_id": 0, "op": "FIND", "dependencies": [], "inputs": ["entity:A"]},
                {"node_id": 1, "op": "RELATE", "dependencies": [0],
                 "inputs": ["r:first", "forward"]},
                {"node_id": 2, "op": "RELATE", "dependencies": [0],
                 "inputs": ["r:second", "forward"]},
                {"node_id": 3, "op": "INTERSECT", "dependencies": [1, 2], "inputs": []},
                {"node_id": 4, "op": "QUERY_NAME", "dependencies": [3], "inputs": []},
            ],
            "metadata": {},
        },
        "world": {
            "entities": {"A": "schema:Person", "B": "schema:Person"},
            "facts": [
                {"fact_id": "f1", "subject": "A", "predicate": "r:first",
                 "object": "B", "object_kind": "entity", "source_id": "s1"},
                {"fact_id": "f2", "subject": "A", "predicate": "r:second",
                 "object": "B", "object_kind": "entity", "source_id": "s2"},
            ],
            "sources": {
                "s1": {"source_id": "s1", "source_type": "text",
                       "fact_ids": ["f1"], "metadata": {}},
                "s2": {"source_id": "s2", "source_type": "text",
                       "fact_ids": ["f2"], "metadata": {}},
            },
            "metadata": {},
        },
    }
    realized = {
        "task_id": "world-test",
        "question": "Who is connected to A by both relations?",
        "capability_world": {
            "resources": [
                {"resource_id": "s1", "source_type": "text", "visibility": "provided",
                 "role": "entry", "required_capability_ids": ["text_cap"],
                 "anchor_dependency_resource_ids": []},
                {"resource_id": "s2", "source_type": "text",
                 "visibility": "discoverable" if dependent else "provided",
                 "role": "discovery", "required_capability_ids": ["text_cap"],
                 "anchor_dependency_resource_ids": ["s1"] if dependent else []},
            ],
            "capabilities": [
                {"capability_id": "text_cap", "specialist": "DocumentAgent"}
            ],
        },
    }
    rendered = SimpleNamespace(
        task_id="world-test", answer=["B"],
        sources=[
            SimpleNamespace(source_id="s1", source_type="text",
                            title="Source 1", content="A has r:first B"),
            SimpleNamespace(source_id="s2", source_type="text",
                            title="Source 2", content="A has r:second B"),
        ],
    )
    return realized, record, rendered


class GoldTrajectoryTests(unittest.TestCase):
    def test_compiler_reuses_symbolic_proof_and_program_dependency(self):
        realized, record, rendered = fixture()
        plan = compile_gold_solution(realized, record, rendered)
        self.assertEqual(plan["symbolic_answer"], ["B"])
        self.assertEqual(plan["final_answer"], ["B"])
        self.assertEqual(plan["proof_source_ids"], ["s1", "s2"])
        self.assertEqual(plan["gold_parallel_waves"],
                         [["subtask:s1", "subtask:s2"]])
        self.assertEqual(len(ready_subtasks(plan, set())), 2)
        public = public_plan(plan)
        self.assertNotIn("final_answer", public)
        self.assertNotIn("gold_answer_claims", json.dumps(public))
        self.assertEqual(plan["symbolic_trace"]["3"], ["B"])

    def test_gold_claims_use_public_rendered_aliases(self):
        realized, record, rendered = fixture()
        rendered.hidden_metadata = {
            "entity_aliases": {"A": "Alice", "B": "Bob"},
            "literal_surfaces_by_predicate": {},
            "literal_surfaces": {},
        }
        rendered.answer = ["Bob"]
        rendered.sources[0].content = "Alice — first: Bob."
        rendered.sources[1].content = "Alice — second: Bob."
        gold = compile_gold_solution(realized, record, rendered)
        claims = gold["tasks"][0]["gold_answer_claims"]
        self.assertEqual(claims, [{
            "subject": "Alice", "predicate": "first",
            "object": "Bob", "object_kind": "entity",
        }])
        self.assertNotIn("fact_id", json.dumps(claims))
        output = synthesize_one(
            realized, record, rendered,
            PassthroughPlanner(), PassthroughExecutor(),
        )
        self.assertTrue(output["validation"]["accepted"])
        self.assertEqual(output["final"]["answer"], ["Bob"])

    def test_dependency_separates_discovery_from_inspection(self):
        realized, record, rendered = fixture(dependent=True)
        plan = compile_gold_solution(realized, record, rendered)
        self.assertEqual(plan["gold_parallel_waves"],
                         [["subtask:s1"], ["discover:s2"], ["subtask:s2"]])
        self.assertEqual(
            [task["subtask_id"] for task in ready_subtasks(plan, {"subtask:s1"})],
            ["discover:s2"],
        )

    def test_two_role_offline_synthesis_preserves_all_gold_answers(self):
        realized, record, rendered = fixture(dependent=True)
        output = synthesize_one(
            realized, record, rendered,
            PassthroughPlanner(), PassthroughExecutor(),
            subagent_workers=2,
        )
        self.assertTrue(output["validation"]["accepted"])
        self.assertEqual(output["final"]["answer"], ["B"])
        self.assertEqual(len(output["steps"]), 3)
        self.assertEqual([len(r["subagent_results"]) for r in output["steps"]],
                         [1, 1, 1])
        self.assertEqual(output["steps"][1]["completed_before"],
                         ["subtask:s1"])
        self.assertEqual(output["metadata"]["clean_trajectory"], True)

    def test_parallel_subagent_calls_and_no_future_evidence_in_main(self):
        realized, record, rendered = fixture()
        class InspectorPlanner(PassthroughPlanner):
            def plan(self, payload, prompt):
                self_outer.assertNotIn("gold_answer_claims", json.dumps(payload["public_plan"]))
                self_outer.assertEqual(payload["completed_subtask_ids"], [])
                self_outer.assertNotIn("A has r:second B", prompt)
                self_outer.assertNotIn("expected_answer", prompt)
                self_outer.assertNotIn("gold_answer_claims", prompt)
                return super().plan(payload, prompt)
        self_outer = self
        class SlowExecutor(PassthroughExecutor):
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.peak = 0
            def execute(self, payload, prompt):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(.025)
                response = super().execute(payload, prompt)
                with self.lock:
                    self.active -= 1
                return response
        worker = SlowExecutor()
        trajectory = synthesize_one(
            realized, record, rendered, InspectorPlanner(), worker,
            subagent_workers=2,
        )
        self.assertTrue(trajectory["validation"]["accepted"])
        self.assertGreaterEqual(worker.peak, 2)

    def test_wrong_grounded_subtask_answer_is_rejected(self):
        realized, record, rendered = fixture()
        class WrongExecutor(PassthroughExecutor):
            def execute(self, payload, prompt):
                value = super().execute(payload, prompt)
                value["answer"] = []
                return value
        with self.assertRaisesRegex(GoldTrajectoryError,
                                    "answer_claims_not_grounded"):
            synthesize_one(
                realized, record, rendered,
                PassthroughPlanner(), WrongExecutor(), max_attempts=2,
            )

    def test_wrong_final_answer_is_rejected(self):
        realized, record, rendered = fixture()
        class WrongMain(PassthroughPlanner):
            def finalize(self, payload, prompt):
                value = super().finalize(payload, prompt)
                value["answer"] = ["C"]
                return value
        with self.assertRaisesRegex(GoldTrajectoryError, "final_answer_mismatch"):
            synthesize_one(
                realized, record, rendered,
                WrongMain(), PassthroughExecutor(), max_attempts=2,
            )

    def test_parallel_plan_cannot_drop_or_invent_tasks(self):
        realized, record, rendered = fixture()
        class WrongMain(PassthroughPlanner):
            def plan(self, payload, prompt):
                result = super().plan(payload, prompt)
                result["parallel_subtasks"].pop()
                return result
        with self.assertRaisesRegex(GoldTrajectoryError, "parallel_group_mismatch"):
            synthesize_one(
                realized, record, rendered,
                WrongMain(), PassthroughExecutor(), max_attempts=2,
            )

    def test_plan_only_and_synthesis_jsonl(self):
        realized, record, rendered = fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            question_file = folder / "questions.jsonl"
            question_file.write_text(json.dumps(realized) + "\n", encoding="utf-8")
            mapping = {"world-test": (record, rendered)}
            with patch("environment.trajectory_simulation.pipeline._world_map",
                       return_value=mapping):
                summary = plan_trajectories_jsonl(
                    folder/"worlds.jsonl", question_file, folder/"plan.jsonl"
                )
                self.assertEqual(summary["ready"], 1)
                output = synthesize_trajectories_jsonl(
                    folder/"worlds.jsonl", question_file, folder/"trajectories.jsonl",
                    planner=PassthroughPlanner(), executor=PassthroughExecutor(),
                    progress=False,
                )
                self.assertEqual(output["accepted"], 1)
                row = json.loads((folder/"trajectories.jsonl").read_text().strip())
                self.assertTrue(row["validation"]["final_answer_verified"])
                self.assertEqual(json.loads(row["trajectory"][-1]["params"]["answer"]), ["B"])


class GoldRegressionTests(unittest.TestCase):
    def test_review_backend_bounds_json_without_changing_generation_backend(self):
        from unittest.mock import patch
        from environment.question_generation.backend import OpenAICompatibleBackend,BackendError
        from environment.trajectory_simulation.backend import LLMPlanner
        backend=OpenAICompatibleBackend("http://localhost/v1","test",temperature=.2)
        role=LLMPlanner(backend)
        with patch("environment.trajectory_simulation.backend.urllib.request.urlopen") as request:
            response=request.return_value.__enter__.return_value
            response.read.return_value=json.dumps({
                "choices":[{"finish_reason":"stop","message":{"content":'{"accept":true,"issues":[]}'}}]
            }).encode()
            self.assertTrue(role.review("Check a summary.")["accept"])
            body=json.loads(request.call_args.args[0].data)
            self.assertEqual(body["response_format"]["type"],"json_schema")
            schema=body["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["properties"]["issues"]["maxItems"],3)
            self.assertEqual(body["temperature"],0.0)
            self.assertEqual(backend.temperature,.2)
            response.read.return_value=json.dumps({
                "choices":[{"finish_reason":"length","message":{"content":'{"accept":'}}]
            }).encode()
            with self.assertRaisesRegex(BackendError,"truncated"):
                role.review("Check a summary.")

    def test_all_selected_anchor_providers_remain_or_dependencies(self):
        r,w,rendered=fixture()
        w["world"]["entities"]["C"]="schema:Person"
        w["world"]["facts"].append({
            "fact_id":"f3","subject":"B","predicate":"r:third",
            "object":"C","object_kind":"entity","source_id":"s3"})
        w["world"]["sources"]["s3"]={
            "source_id":"s3","source_type":"text","fact_ids":["f3"],"metadata":{}}
        w["program"]["nodes"][-1].update(op="RELATE",inputs=["r:third","forward"])
        w["program"]["nodes"].append({
            "node_id":5,"op":"QUERY_NAME","dependencies":[4],"inputs":[]})
        rendered.answer=["C"]
        rendered.sources.append(SimpleNamespace(
            source_id="s3",source_type="text",title="Source 3",content="B third C"))
        resources=r["capability_world"]["resources"]
        for resource in resources:
            resource.update(requires_anchors=[],produces_anchors=["a"])
        resources.append({
            **resources[0],"resource_id":"s3","visibility":"discoverable",
            "requires_anchors":["a"],"produces_anchors":[]})
        plan=compile_gold_solution(r,w,rendered)
        task=next(t for t in plan["tasks"] if t["subtask_id"]=="discover:s3")
        self.assertEqual(task["depends_on"],[])
        self.assertEqual(task["depends_on_any"],[["subtask:s1","subtask:s2"]])
        for provider in ("s1","s2"):
            self.assertIn(task,ready_subtasks(plan,{"subtask:"+provider}))
        self.assertNotIn(task,ready_subtasks(plan,set()))
        output=synthesize_one(r,w,rendered,PassthroughPlanner(),PassthroughExecutor())
        self.assertTrue(output["validation"]["accepted"])

    def test_required_anchors_are_and_but_providers_are_or(self):
        from environment.trajectory_simulation.acquisition import resolve_dependencies
        resources=[
            {"resource_id":"a1","visibility":"provided","produces_anchors":["a"]},
            {"resource_id":"a2","visibility":"provided","produces_anchors":["a"]},
            {"resource_id":"b","visibility":"provided","produces_anchors":["b"]},
            {"resource_id":"target","visibility":"discoverable","requires_anchors":["a","b"]},
        ]
        deps,_=resolve_dependencies(resources,{})
        task={"subtask_id":"target","depends_on":[],
              "depends_on_any":[sorted(group) for group in deps["target"]]}
        self.assertEqual(ready_subtasks({"tasks":[task]},{"a1"}),[])
        self.assertEqual(ready_subtasks({"tasks":[task]},{"a2","b"}),[task])

    def test_evidence_selection_does_not_force_first_provider(self):
        r,w,rendered=fixture(dependent=True)
        resources=r["capability_world"]["resources"]
        resources[0].update(produces_anchors=["a"],requires_anchors=[])
        resources[1].update(produces_anchors=[],requires_anchors=["a"])
        resources.append({
            **resources[0],"resource_id":"s0","anchor_dependency_resource_ids":[]})
        w["world"]["sources"]["s0"]={
            "source_id":"s0","source_type":"text","fact_ids":[],"metadata":{}}
        rendered.sources.append(SimpleNamespace(
            source_id="s0",source_type="text",title="Extra",content=""))
        plan=compile_gold_solution(r,w,rendered)
        self.assertEqual(plan["required_resource_ids"],["s1","s2"])
        self.assertEqual(plan["acquisition_alternatives"]["s2"],[["s0","s1"]])

    def test_or_cycle_cannot_publish_same_wave_anchor(self):
        from environment.trajectory_simulation.acquisition import resolve_dependencies
        with self.assertRaisesRegex(ValueError,"unreachable"):
            resolve_dependencies([
                {"resource_id":"a","visibility":"discoverable",
                 "requires_anchors":["b"],"produces_anchors":["a"]},
                {"resource_id":"b","visibility":"discoverable",
                 "requires_anchors":["a"],"produces_anchors":["b"]},
            ],{})

    def test_grouped_review_keeps_titles_provenance_and_latest_read_state(self):
        from environment.trajectory_simulation.grounding import grouped_evidence,review_prompt
        tasks={
            "d":{"subtask_id":"d","kind":"discover","resource_id":"source_02",
                 "observation":{"title":"Reference 1"}},
            "i":{"subtask_id":"i","kind":"inspect","resource_id":"source_02",
                 "observation":{"title":"Reference 1"}},
        }
        claims,state=grouped_evidence(tasks,[
            {"subtask_id":"d","answer":[{"resource_id":"source_02","title":"Reference 1"}]},
            {"subtask_id":"i","answer":[{"subject":"Alice","predicate":"height","object":170}]},
        ])
        self.assertEqual(state["inspected_resource_ids"],["source_02"])
        self.assertEqual(state["discovered_unread_resource_ids"],[])
        self.assertEqual(claims[1]["resource_id"],"source_02")
        self.assertEqual(claims[1]["title"],"Reference 1")
        prompt=review_prompt("Read Source 02.",claims,context=state)
        self.assertIn("Never infer ownership",prompt)
        self.assertIn('"resource_title_aliases":{"Reference 1":"source_02"}',prompt)
        self.assertIn("AUTHORITATIVE OBSERVED FACT from INSPECTED source_02",prompt)

    def test_summary_numeric_format_is_semantically_normalized(self):
        from environment.trajectory_simulation.grounding import text_issues
        self.assertEqual(text_issues("Population is 1,190.0.",[{"population":1190}]),[])
        self.assertEqual(text_issues("Population is 1190.",[{"population":"1,190"}]),[])
        self.assertIn("unsupported_numeric_claim_in_summary",
                      text_issues("Population is 1191.",[{"population":1190}]))

    def test_final_reviewer_sees_verified_answer_but_main_does_not(self):
        r,w,rendered=fixture()
        w["program"]["nodes"][-1].update(op="COUNT")
        rendered.answer=1
        outer=self
        class Main(PassthroughPlanner):
            def finalize(self,payload,prompt):
                outer.assertNotIn("expected_answer",prompt)
                outer.assertNotIn("verified_final_answer",prompt)
                outer.assertIn("JOIN facts",prompt)
                return {"action":"complete","answer":1,"summary":"The final count is 1."}
            def review(self,prompt):
                if '"main_final_summary"' in prompt:
                    outer.assertIn('"verified_final_answer":1',prompt)
                    outer.assertIn('"inspected_resource_ids":["s1","s2"]',prompt)
                return {"accept":True,"issues":[]}
        output=synthesize_one(r,w,rendered,Main(),PassthroughExecutor())
        self.assertEqual(output["final"]["answer"],1)
        self.assertTrue(output["validation"]["accepted"])

    def test_shared_anchor_provider_is_an_alternative_not_an_and_parent(self):
        r,w,rendered=fixture(dependent=True)
        cap=r["capability_world"]
        cap["resources"][0].update(produces_anchors=["a"],requires_anchors=[])
        cap["resources"][1].update(produces_anchors=["b"],requires_anchors=["a"],anchor_dependency_resource_ids=["s1","s3"])
        cap["resources"].append({**cap["resources"][1],"resource_id":"s3","requires_anchors":["b"],"produces_anchors":["a"],"anchor_dependency_resource_ids":["s2"]})
        w["world"]["sources"]["s3"]={"source_id":"s3","source_type":"text","fact_ids":[],"metadata":{}}
        rendered.sources.append(SimpleNamespace(source_id="s3",source_type="text",title="Extra",content=""))
        cap["binding_audit"]={"progressive_discovery":{"anchor_registry":{"question_anchor_ids":[]}}}
        plan=compile_gold_solution(r,w,rendered)
        self.assertEqual(plan["required_resource_ids"],["s1","s2"])
        tasks={t["subtask_id"]:t for t in plan["tasks"]}
        self.assertEqual(tasks["discover:s2"]["depends_on"],["subtask:s1"])
        self.assertEqual(tasks["subtask:s2"]["depends_on"],["discover:s2"])
        self.assertNotIn("A has r:second B",json.dumps(tasks["discover:s2"]["observation"]))

    def test_unreachable_anchor_is_not_hidden_by_compiler(self):
        from environment.trajectory_simulation.compiler import GoldCompilationError
        r,w,rendered=fixture(dependent=True)
        r["capability_world"]["resources"][0].update(requires_anchors=[],produces_anchors=["a"])
        r["capability_world"]["resources"][1].update(requires_anchors=["missing"],produces_anchors=[])
        r["capability_world"]["binding_audit"]={"progressive_discovery":{"anchor_registry":{"question_anchor_ids":[]}}}
        with self.assertRaisesRegex(GoldCompilationError,"unreachable frozen"):
            compile_gold_solution(r,w,rendered)

    def test_comparison_keeps_losing_candidate_attribute(self):
        r,w,rendered=fixture()
        w["program"]["nodes"]=[
            {"node_id":0,"op":"FIND","dependencies":[],"inputs":["entity:A"]},
            {"node_id":1,"op":"FIND","dependencies":[],"inputs":["entity:B"]},
            {"node_id":2,"op":"SELECT_BETWEEN","dependencies":[0,1],"inputs":["height","greater"]}]
        for index,fact in enumerate(w["world"]["facts"]):
            fact.update(subject="A" if index==0 else "B",predicate="height",object=10+index*10,object_kind="number")
        plan=compile_gold_solution(r,w,rendered)
        self.assertEqual(plan["required_resource_ids"],["s1","s2"])
        self.assertEqual(plan["required_fact_ids"],["f1","f2"])

    def test_final_set_order_numeric_format_and_boolean_types(self):
        from environment.trajectory_simulation.validator import validate_final
        def check(a,b,contract=None):return validate_final({"action":"complete","answer":a,"summary":"Evidence supports this answer."},b,contract)[0]
        self.assertTrue(check(["Bob","Alice"],["Alice","Bob"],{"collection_mode":"set"}))
        self.assertFalse(check(["Bob","Alice"],["Alice","Bob"],{"collection_mode":"ordered"}))
        self.assertTrue(check("1,190.0",1190))
        self.assertFalse(check(True,1))
        self.assertFalse(check(1,True))
        self.assertFalse(check(["Alice"],["Alice","Alice"],{"collection_mode":"multiset"}))
        self.assertTrue(check('["Bob","Alice"]',["Alice","Bob"],{"collection_mode":"set"}))
        self.assertTrue(check("Alice",["Alice"],{"collection_mode":"set"}))
        self.assertFalse(check("Alice and Bob",["Alice","Bob"],{"collection_mode":"set"}))
        self.assertFalse(check('["Alice"]',["Alice","Alice"],{"collection_mode":"multiset"}))

    def test_semantic_review_requires_an_actual_quote_and_reason(self):
        from environment.trajectory_simulation.grounding import semantic_review
        class Reviewer:
            offline=False
            def review(self,prompt):
                return {"accept":False,"issues":[
                    {"quote":"Alice is 999 metres tall.","reason":"Not in evidence"}]}
        okay,issues=semantic_review(
            Reviewer(),"Alice is 170 metres tall.",[{"height":170}])
        self.assertFalse(okay)
        self.assertEqual(issues,["summary_review_invalid_rejection"])
        class ValidReviewer:
            offline=False
            def review(self,prompt):
                return {"accept":False,"issues":[
                    {"quote":"Alice is short.","reason":"Evidence says Alice is tall."}]}
        okay,issues=semantic_review(
            ValidReviewer(),"Alice is short.",[{"description":"Alice is tall."}])
        self.assertFalse(okay)
        self.assertIn("Alice is short.",issues[0])
        class ContradictoryReviewer(ValidReviewer):
            def review(self,prompt):
                response=super().review(prompt)
                response["accept"]=True
                return response
        self.assertFalse(semantic_review(
            ContradictoryReviewer(),"Alice is short.",
            [{"description":"Alice is tall."}])[0])

    def test_false_summary_is_rejected_even_when_structured_facts_match(self):
        task={"subtask_id":"s","resource_id":"r","gold_answer_claims":[{"subject":"Alice","predicate":"height","object":170}]}
        result={"subtask_id":"s","status":"success","answer":task["gold_answer_claims"],"evidence_source_ids":["r"],"answer_summary":"Alice is 999 metres tall.","trajectory_summary":"Read unavailable future image."}
        okay,issues=validate_subagent_result(result,task)
        self.assertFalse(okay)
        self.assertTrue(any("unsupported_numeric" in e for e in issues))
        self.assertTrue(any("unavailable_observation" in e for e in issues))

    def test_observed_resource_title_is_valid_summary_context(self):
        task={"subtask_id":"s","resource_id":"source_01","gold_answer_claims":[{"subject":"Alice","predicate":"height","object":170}],"observation":{"resource_id":"source_01","title":"Reference 4","content":"Alice height: 170."}}
        result={"subtask_id":"s","status":"success","answer":task["gold_answer_claims"],"evidence_source_ids":["source_01"],"answer_summary":"Reference 4 records Alice's height as 170.","trajectory_summary":"Read Reference 4."}
        self.assertTrue(validate_subagent_result(result,task)[0])

    def test_resource_shorthand_is_not_a_factual_quantity(self):
        from environment.trajectory_simulation.grounding import text_issues
        self.assertEqual(text_issues("Read sources 01, 03, and 05.",[],allowed_ids=["source_01","source_03","source_05"]),[])
        self.assertIn("unknown_resource_reference",text_issues("Read source 999.",[],allowed_ids=["source_01"]))
        self.assertEqual(text_issues(
            "Pending work must inspect sources 01, 03, and 05.",[],
            allowed_ids=["source_02"],
            pending_ids=["subtask:source_01","subtask:source_03","subtask:source_05"]),[])

    def test_final_summary_review_is_not_a_second_question_solver(self):
        from environment.trajectory_simulation.grounding import review_prompt
        for kind in ("main_final_summary","main_round_summary","inspect"):
            prompt=review_prompt("The answer is false.",[],
                                 question="PRIVATE QUESTION ONLY FOR MAIN",
                                 context={"kind":kind,"verified_final_answer":False})
            self.assertNotIn("PRIVATE QUESTION ONLY FOR MAIN",prompt)
            self.assertIn('"verified_final_answer":false',prompt)

    def test_final_summary_rejection_audit_separates_correct_answer(self):
        from environment.trajectory_simulation.pipeline import _stage
        with self.assertRaises(GoldTrajectoryError) as caught:
            _stage(
                lambda feedback:{"action":"complete","answer":False,"summary":"Bad attribution"},
                lambda value:(False,["summary_semantic_review:wrong source"]),
                stage="main_final_answer",max_attempts=1,
                diagnostics={"expected_answer":False,"answer_contract":{"collection_mode":"scalar"}})
        self.assertTrue(caught.exception.audit["attempts"][0]["answer_validation"]["accepted"])

    def test_false_single_source_requirement_is_rejected_before_llm_review(self):
        from environment.trajectory_simulation.grounding import semantic_review,review_prompt
        class Reviewer:
            def review(self,prompt):
                raise AssertionError("Invalid co-location reasoning should be rejected directly")
        okay,issues=semantic_review(
            Reviewer(),"No single source has both attributes, so the answer is false.",[],
            context={"kind":"main_final_summary","verified_final_answer":False})
        self.assertFalse(okay)
        self.assertIn("invalid_single_source_requirement_in_final_reasoning",issues)
        prompt=review_prompt(
            "The height does not equal 1.6224 m.",[{"height":1.8544}],
            question="Compare to 1.6224 m for people born in 1674.",
            context={"kind":"main_final_summary","verified_final_answer":False})
        self.assertIn('"public_constraint_values":["1.6224","1674"]',prompt)
        self.assertNotIn("Compare to",prompt)

    def test_rejected_response_is_preserved_for_diagnosis(self):
        from environment.trajectory_simulation.pipeline import _stage
        try:
            _stage(lambda feedback:{"answer":"wrong"},lambda value:(False,["final_answer_mismatch"]),stage="final",max_attempts=2,diagnostics={"expected_answer":"right"})
        except GoldTrajectoryError as exc:
            self.assertEqual(len(exc.audit["attempts"]),2)
            self.assertEqual(exc.audit["attempts"][0]["response"]["answer"],"wrong")
            self.assertEqual(exc.audit["expected_answer"],"right")
        else:self.fail("invalid response was accepted")

    def test_semantic_summary_rejection_blocks_trajectory(self):
        r,w,rendered=fixture()
        class RejectSummary(PassthroughExecutor):
            def review(self,prompt):return {"accept":False,"issues":["unsupported entity"]}
        with self.assertRaisesRegex(GoldTrajectoryError,"summary_semantic_review"):
            synthesize_one(r,w,rendered,PassthroughPlanner(),RejectSummary(),max_attempts=1)


if __name__ == "__main__":
    unittest.main()
