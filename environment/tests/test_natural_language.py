"""Naturalistic O1 prose must not break gold replay, source causality or SFT export."""
from __future__ import annotations

import json
import re
import unittest

from environment.tests.test_gold_trajectory import fixture
from environment.trajectory_simulation.backend import (
    LLMPlanner, LLMExecutor, PassthroughPlanner, PassthroughExecutor,
)
from environment.trajectory_simulation.contract import validate_feedback
from environment.trajectory_simulation.evidence import read_packet
from environment.trajectory_simulation.exporter import export_episode
from environment.trajectory_simulation.natural_language import (
    realize_main, realize_feedback_natural, natural_public_context,
)
from environment.trajectory_simulation.pipeline import synthesize_one


def main_payload():
    guidance = {
        "may_complete": False,
        "ready": [{
            "suggested_tools": ["GoogleSearchAction"],
            "operations": [
                {"op": "FIND", "find_kind": "entity", "inputs": ["Alice"]},
                {"op": "FILTER_YEAR", "inputs": ["r:birth", "=", "1912"]},
            ],
        }],
        "available_tools": ["GoogleSearchAction"],
    }
    task = {
        "task_instruction": (
            "Required checks:\n1. Identify Alice.\n"
            "2. From the result of instruction 1, keep only entities whose r:birth = 1912."
        ),
        "context": (
            "Execute the instructions in dependency order; each input refers only "
            "to the result explicitly named in that instruction."
            "\n[Public task packet]\n"
            '{"version":2,"scene":"omnigaia","completed_inputs":{},'
            '"known_resource_ids":[]}'
        ),
        "model": "model_1",
        "tools": ["GoogleSearchAction"],
    }
    canonical = {
        "action": "delegate_task", "reasoning": "Delegate the required checks.",
        "params": {"tasks": [task]},
    }
    payload = {
        "gold_guidance": guidance,
        "offline_oracle_answer": ["Alice"],
        "answer_contract": {"collection_mode": "set"},
        "public_state": {
            "question": "Find Alice whose birth year is 1912.",
            "available_tools": ["GoogleSearchAction"],
        },
        "visible_resources": [],
        "completed_subtask_contracts": [],
    }
    return payload, canonical


class NaturalLanguageTests(unittest.TestCase):
    def test_natural_task_has_public_v3_replay_contract_and_fixed_transport(self):
        payload, canonical = main_payload()
        natural = realize_main(payload, canonical, {
            "reasoning": "The requested person can be checked against the dated records.",
            "task_instructions": [
                "Check the supplied records for Alice and identify whether her recorded "
                "birth year is 1912. Return the matching person with supporting evidence.",
            ],
        })
        task = natural["params"]["tasks"][0]
        self.assertNotIn("Required checks", task["task_instruction"])
        self.assertNotIn("1.", task["task_instruction"])
        self.assertEqual(task["tools"], canonical["params"]["tasks"][0]["tools"])
        self.assertEqual(task["model"], "model_1")
        packet = read_packet(task)
        self.assertEqual(packet["version"], 3)
        self.assertIn("1. Identify Alice.", packet["execution_checks"])
        self.assertEqual(packet["known_resource_ids"], [])
        self.assertEqual(
            task["context"],
            natural_public_context(canonical["params"]["tasks"][0]["context"],
                                   canonical["params"]["tasks"][0]["task_instruction"]),
        )
        self.assertEqual(
            read_packet(canonical["params"]["tasks"][0])["version"], 2,
        )

    def test_main_rejects_missing_anchors_or_invented_numbers(self):
        payload, canonical = main_payload()
        for candidate in (
            "Check records for Alice and return the matching person.",
            "Check records for Alice born in 1912 and 1999.",
            "Find a person born in 1912 and return the match.",
        ):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    realize_main(payload, canonical, {
                        "reasoning": "Examine the records.",
                        "task_instructions": [candidate],
                    })

    def test_canonical_sibling_can_remain_unmodified_after_failed_rewrite(self):
        payload, canonical = main_payload()
        result = realize_main(payload, canonical, {
            "reasoning": "",
            "task_instructions": [canonical["params"]["tasks"][0]["task_instruction"]],
        })
        self.assertEqual(result["params"]["tasks"][0], canonical["params"]["tasks"][0])
        self.assertEqual(read_packet(result["params"]["tasks"][0])["version"], 2)

    def test_real_backend_writer_requires_semantic_critic(self):
        payload, canonical = main_payload()

        class FakeBackend:
            model = "local-test"
            base_url = "test://local"

            def __init__(self, verdict):
                self.calls = []
                self.verdict = verdict

            def complete_json(self, system_prompt, prompt):
                self.calls.append((system_prompt, prompt))
                if len(self.calls) == 1:
                    return {
                        "task_instruction": (
                            "Identify Alice in the records, check the recorded 1912 birth year, "
                            "and return the person with evidence."
                        ),
                    }
                return self.verdict

        bad = LLMPlanner(FakeBackend({"equivalent": False, "issues": ["missing relation"]}))
        rejected=bad.naturalize_action(payload, canonical)
        self.assertEqual(
            rejected["params"]["tasks"][0],
            canonical["params"]["tasks"][0],
        )
        self.assertEqual(bad.last_natural_audit["rewritten_tasks"],0)
        self.assertEqual(bad.request_counts["natural_main_review"], 1)

        good = LLMPlanner(FakeBackend({"equivalent": True, "issues": []}))
        output = good.naturalize_action(payload, canonical)
        self.assertIn("Alice", output["params"]["tasks"][0]["task_instruction"])
        self.assertEqual(good.request_counts["natural_main_write"], 1)
        self.assertEqual(good.request_counts["natural_main_review"], 1)

    def test_main_parallel_tasks_keep_approved_sibling_and_reject_bad_one(self):
        import copy
        payload,canonical=main_payload()
        second=copy.deepcopy(canonical["params"]["tasks"][0])
        second["task_instruction"]=second["task_instruction"].replace("Alice","Bob")
        canonical["params"]["tasks"].append(second)
        second_fragment=copy.deepcopy(payload["gold_guidance"]["ready"][0])
        second_fragment["operations"][0]["inputs"]=["Bob"]
        payload["gold_guidance"]["ready"].append(second_fragment)

        class Backend:
            model="test"
            base_url="test://local"
            calls=0
            def complete_json(self, system, prompt):
                self.calls+=1
                if self.calls==1:
                    return {"task_instruction":
                        "Identify Alice from the supplied records, verify her recorded birth year is 1912, "
                        "and return the matching name with source evidence."}
                if self.calls==2:
                    return {"equivalent":True,"issues":[]}
                return {"task_instructions":["invalid multi-task output"]}

        planner=LLMPlanner(Backend())
        result=planner.naturalize_action(payload,canonical)
        self.assertEqual(planner.last_natural_audit["rewritten_tasks"],1)
        self.assertEqual(planner.last_natural_audit["task_count"],2)
        self.assertEqual(len(planner.last_natural_audit["errors"]),1)
        self.assertEqual(planner.request_counts["natural_main_write"],2)
        self.assertEqual(planner.request_counts["natural_main_review"],1)
        self.assertEqual(read_packet(result["params"]["tasks"][0])["version"],3)
        self.assertEqual(read_packet(result["params"]["tasks"][1])["version"],2)
        self.assertEqual(result["params"]["tasks"][1],second)
        self.assertEqual(result["reasoning"],"")

    def test_realistic_main_complete_uses_fixed_answer_without_llm_request(self):
        payload,canonical=main_payload()
        payload["gold_guidance"]["may_complete"]=True
        canonical={"action":"complete","reasoning":"template",
                   "params":{"answer":"[\"Alice\"]"}}
        class NoCalls:
            model="test"
            base_url="test://local"
            def complete_json(self,*args):
                raise AssertionError("complete must not invoke the LLM")
        planner=LLMPlanner(NoCalls())
        result=planner.naturalize_action(payload,canonical)
        self.assertEqual(result["reasoning"],"")
        self.assertEqual(result["params"]["answer"],"[\"Alice\"]")
        self.assertEqual(planner.request_counts["natural_main_write"],0)

    def test_report_writer_rejects_fabricated_numbers_and_tool_calls(self):
        payload = {"fragment": {
            "operations": [{"op": "FIND", "inputs": ["Alice"]}],
            "operation_results": [{"answer": ["Alice"]}],
            "expected_answer": ["Alice"],
            "observations": [{
                "resource_id": "source_01", "title": "First reference",
                "claims": [{"subject": "Alice", "predicate": "r:birth",
                            "object": "1912", "object_kind": "year"}],
            }],
            "known_observations": [],
        }}
        valid = {
            "summary": "The supplied reference identifies Alice as the match.",
            "trace_summary": "First reference records Alice with birth year 1912, "
                             "supporting the identification.",
        }
        self.assertEqual(realize_feedback_natural(payload, valid), valid)
        with self.assertRaises(ValueError):
            realize_feedback_natural(payload, {
                **valid, "trace_summary": "A web search was executed and found Alice in 1999.",
            })
        with self.assertRaises(ValueError):
            realize_feedback_natural(payload, {
                **valid, "summary": "The record identifies Bob as the match.",
                "trace_summary": "No other result was found.",
            })

    def test_report_may_name_verified_simulated_tools_but_not_unsupported_ones(self):
        fragment = {
            "scene": "omnigaia",
            "expected_answer": ["Alice"],
            "operations": [{"op":"FIND", "inputs":["Alice"]}],
            "operation_results": [{"answer":["Alice"], "input_values":[]}],
            "observations": [{
                "resource_id":"source_01","source_type":"text","title":"First reference",
                "claims":[{"subject":"Alice","predicate":"r:birth",
                           "object":"1912","object_kind":"year"}],
                "visibility":"discoverable",
            }],
            "known_observations": [], "acquisition_waves":[["source_01"]],
            "suggested_tools":["GoogleSearchAction","ExtractUrlContentAction"],
        }
        from environment.trajectory_simulation.tool_provenance import (
            reconstruct_tool_observations,
        )
        plan = reconstruct_tool_observations(fragment)
        self.assertFalse(plan["live_tool_calls"])
        self.assertEqual(
            [step["tool"] for step in plan["steps"]],
            ["GoogleSearchAction","ExtractUrlContentAction"],
        )
        payload = {"fragment": fragment, "tool_provenance":plan}
        valid = {
            "summary":"Alice is identified in the public record.",
            "trace_summary":"- GoogleSearchAction located First reference (source_01).\n"
                            "- ExtractUrlContentAction inspected First reference and "
                            "found Alice's r:birth year as 1912.\n"
                            "- The recorded result identifies Alice.",
        }
        self.assertEqual(realize_feedback_natural(payload,valid),valid)
        with self.assertRaisesRegex(ValueError,"unverified_tool_action_claim"):
            realize_feedback_natural(payload,{
                **valid,"trace_summary":valid["trace_summary"]+
                "\n- VideoAnalysisAction read a video file.",
            })
        # Real O1 action summaries can use ordinary verbs, with no class
        # names. They may also concisely omit tool mechanics altogether.
        fluent = {
            **valid,
            "trace_summary": (
                "An initial search located First reference. "
                "I then checked the webpage and confirmed that it records "
                "Alice's birth year as 1912, supporting the result."
            ),
        }
        self.assertEqual(realize_feedback_natural(payload, fluent), fluent)
        concise = {
            **valid,
            "trace_summary": "First reference identifies Alice born in 1912.",
        }
        self.assertEqual(realize_feedback_natural(payload, concise), concise)
        with self.assertRaisesRegex(ValueError,"unverified_tool_action_claim"):
            realize_feedback_natural(payload,{
                **valid,"trace_summary":fluent["trace_summary"] +
                " I also analyzed the supplied video.",
            })
        with self.assertRaisesRegex(ValueError,"unverified_tool_action_claim"):
            realize_feedback_natural(payload,{
                **valid,"trace_summary":fluent["trace_summary"] +
                " The speech recording was transcribed.",
            })
        with self.assertRaisesRegex(ValueError,"unverified_tool_action_claim"):
            realize_feedback_natural(payload,{
                **valid,"trace_summary":valid["trace_summary"]+
                "\n- I clicked an unrelated webpage.",
            })

    def test_writer_source_guide_links_each_claim_to_correct_reference(self):
        from environment.trajectory_simulation.tool_provenance import (
            reconstruct_tool_observations, source_evidence_guide,
            unsupported_tool_actions,
        )
        from environment.trajectory_simulation.natural_language import (
            feedback_writer_prompt, feedback_reviewer_prompt,
        )
        fragment = {
            "scene":"omnigaia","expected_answer":["Alice"],
            "operations":[{"op":"FIND","inputs":["Alice"]}],
            "operation_results":[{"answer":["Alice"],"input_values":[]}],
            "suggested_tools":["ExtractUrlContentAction","ImageAnalysisAction"],
            "observations":[
                {"resource_id":"source_01","title":"Text page","source_type":"text",
                 "visibility":"provided",
                 "claims":[{"subject":"Alice","predicate":"r:birth",
                            "object":"1912","object_kind":"year"}]},
                {"resource_id":"source_02","title":"Photo","source_type":"image",
                 "visibility":"provided",
                 "claims":[{"subject":"Bob","predicate":"r:birth",
                            "object":"1920","object_kind":"year"}]},
            ],
            "known_observations":[],
        }
        plan = reconstruct_tool_observations(fragment)
        guide = source_evidence_guide(fragment,plan)
        self.assertEqual(len(guide),2)
        self.assertEqual(guide[0]["permitted_tools"],["ExtractUrlContentAction"])
        self.assertEqual(guide[1]["permitted_tools"],["ImageAnalysisAction"])
        self.assertEqual(guide[0]["claims"][0]["subject"],"Alice")
        self.assertEqual(guide[1]["claims"][0]["subject"],"Bob")
        self.assertEqual(
            unsupported_tool_actions("Examined the supplied image to check Bob.",plan),
            set(),
        )
        self.assertEqual(
            unsupported_tool_actions("Transcribed the audio recording.",plan),
            {"ParseAudioAction"},
        )
        payload={"fragment":fragment,"task_instruction":"Find Alice",
                 "tool_provenance":plan}
        prompt=feedback_writer_prompt(payload,{"summary":"Alice","trace_summary":"template"})
        self.assertIn('"source_evidence_guide"',prompt)
        self.assertIn('"source_id":"source_02"',prompt)
        self.assertIn("class names",prompt)
        review=feedback_reviewer_prompt(payload,{
            "summary":"Alice matches.","trace_summary":"The text page records Alice's birth year."
        })
        self.assertIn('"source_evidence_guide"',review)
        self.assertIn("grounded",review)

    def test_empty_subagent_json_gets_one_compact_retry_and_strict_review(self):
        from environment.trajectory_simulation.natural_language import (
            feedback_repair_prompt,
        )
        fragment={
            "scene":"omnigaia","expected_answer":["Alice"],
            "operations":[
                {"op":"FIND","inputs":["Alice"]},
                {"op":"FILTER_YEAR","inputs":["r:birth","=","1912"]},
            ],
            "operation_results":[
                {"answer":["Alice"],"input_values":[]},
                {"answer":["Alice"],"input_values":["Alice"]},
            ],
            "observations":[{
                "resource_id":"source_01","title":"Reference 1","source_type":"text",
                "claims":[
                    {"subject":"Alice","predicate":"r:birth","object":"1912"},
                    {"subject":"Unrelated","predicate":"postal code","object":"99999"},
                ],
            }],
            "known_observations":[],
        }
        payload={
            "fragment":fragment,
            "task_instruction":"Find Alice born in 1912.",
            "tool_provenance":{"steps":[]},
        }
        valid={
            "summary":"Alice is identified as the matching person.",
            "trace_summary":(
                "Checked Reference 1, which records Alice's birth year "
                "as 1912. The year matches the assigned condition."
            ),
        }
        shortened=feedback_repair_prompt(payload,valid)
        self.assertIn('"p":"r:birth"',shortened)
        self.assertNotIn('"p":"postal code"',shortened)
        self.assertIn("NONEMPTY",shortened)

        class Backend:
            model="test"
            base_url="test://local"
            def __init__(self):
                self.calls=[]
            def complete_json(self,system,prompt):
                self.calls.append((system,prompt))
                if len(self.calls)==1:
                    return {"summary":"","trace_summary":None}
                if len(self.calls)==2:
                    return valid
                return {"grounded":True,"issues":[]}

        backend=Backend()
        executor=LLMExecutor(backend)
        self.assertEqual(executor.naturalize_feedback(payload,valid),valid)
        self.assertEqual(executor.request_counts["natural_feedback_write"],2)
        self.assertEqual(executor.request_counts["natural_feedback_retry"],1)
        self.assertEqual(executor.request_counts["natural_feedback_review"],1)
        self.assertEqual(len(backend.calls),3)
        self.assertIn("relevant_public_source_facts",backend.calls[1][1])

    def test_repeated_empty_subagent_json_preserves_field_level_diagnostics(self):
        payload={"fragment":{
            "expected_answer":["Alice"],
            "operations":[{"op":"FIND","inputs":["Alice"]}],
            "operation_results":[{"answer":["Alice"],"input_values":[]}],
            "observations":[],"known_observations":[],
        },"task_instruction":"Find Alice.","tool_provenance":{"steps":[]}}
        class EmptyBackend:
            model="test"
            base_url="test://local"
            calls=0
            def complete_json(self,*args):
                self.calls+=1
                return {"summary":None,"trace_summary":" "}
        backend=EmptyBackend()
        executor=LLMExecutor(backend)
        with self.assertRaisesRegex(
            ValueError,r"natural_language_missing;writes=2;summary=null;trace_summary=blank"
        ):
            executor.naturalize_feedback(payload,{})
        self.assertEqual(backend.calls,2)
        self.assertEqual(executor.request_counts["natural_feedback_review"],0)
        self.assertEqual(executor.request_counts["natural_feedback_retry"],1)

    def test_unverified_action_does_not_get_format_retries(self):
        payload={"fragment":{
            "expected_answer":["Alice"],
            "operations":[{"op":"FIND","inputs":["Alice"]}],
            "operation_results":[{"answer":["Alice"],"input_values":[]}],
            "observations":[],"known_observations":[],
        },"task_instruction":"Find Alice.","tool_provenance":{"steps":[]}}
        class FakeBackend:
            model="test"
            base_url="test://local"
            calls=0
            def complete_json(self,*args):
                self.calls+=1
                return {"summary":"Alice is the result.",
                        "trace_summary":"I analyzed the supplied video and found Alice."}
        backend=FakeBackend()
        executor=LLMExecutor(backend)
        with self.assertRaisesRegex(ValueError,"unverified_tool_action_claim"):
            executor.naturalize_feedback(payload,{})
        self.assertEqual(backend.calls,1)
        self.assertEqual(executor.request_counts["natural_feedback_retry"],0)

    def test_public_tool_plan_preserves_acquisition_order_and_reused_sources(self):
        from environment.trajectory_simulation.tool_provenance import (
            reconstruct_tool_observations,
        )
        def source(sid, source_type, visibility="discoverable"):
            return {
                "resource_id":sid,"source_type":source_type,
                "visibility":visibility,"title":f"Reference {sid}",
                "claims":[{"observation_key":f"claim:{sid}",
                           "subject":"Alice","predicate":"r:record",
                           "object":"yes","object_kind":"string"}],
            }
        fragment = {
            "scene":"omnigaia","observations":[
                source("source_02","image_report"),
                source("source_01","text")
            ],
            "known_observations":[source("source_00","video_report","provided")],
            "acquisition_waves":[["source_01"],["source_02"]],
            "suggested_tools":["GoogleSearchAction","ExtractUrlContentAction",
                               "ImageAnalysisAction"],
        }
        plan=reconstruct_tool_observations(fragment)
        self.assertEqual(
            [(step["kind"],step["tool"],step["resource_id"]) for step in plan["steps"]],
            [
                ("source_discovery","GoogleSearchAction","source_01"),
                ("source_inspection","ExtractUrlContentAction","source_01"),
                ("source_discovery","GoogleSearchAction","source_02"),
                ("source_inspection","ImageAnalysisAction","source_02"),
                ("previously_available_evidence",None,"source_00"),
            ],
        )
        self.assertTrue(all("query" not in step for step in plan["steps"]))
        self.assertTrue(all("timestamp" not in step for step in plan["steps"]))
        self.assertTrue(all("claim_keys" in step for step in plan["steps"]))

    def test_public_replay_and_sft_export_work_with_natural_main_task_prose(self):
        class NaturalPlanner(PassthroughPlanner):
            style_provider = "natural"

            def naturalize_action(self, payload, canonical):
                if canonical["action"] == "complete":
                    return realize_main(payload, canonical, {
                        "reasoning": "The reported findings resolve the requested question.",
                    })
                natural = []
                for task in canonical["params"]["tasks"]:
                    # Deterministic test-only equivalent paraphrase: no numbered checklist.
                    lines = [
                        re.sub(r"^\d+\.\s+", "", line)
                        for line in task["task_instruction"].splitlines()
                        if re.match(r"^\d+\.\s+", line)
                    ]
                    natural.append(
                        "Review the relevant reference material. " + " ".join(lines)
                    )
                return realize_main(payload, canonical, {
                    "reasoning": "I will delegate the independent checks.",
                    "task_instructions": natural,
                })

        class NaturalExecutor(PassthroughExecutor):
            style_provider = "natural"

            def naturalize_feedback(self, payload, canonical):
                return canonical

        q, record, rendered = fixture(dependent=True)
        episode = synthesize_one(
            q, record, rendered, NaturalPlanner(), NaturalExecutor(),
            schedule="compact",
        )
        self.assertTrue(episode["validation"]["public_evidence_replay_verified"])
        self.assertEqual(episode["metadata"]["natural_language_fallbacks"], 0)
        self.assertTrue(episode["metadata"]["natural_language_attempted"])
        actions = [step for step in episode["trajectory"] if step["action"] == "delegate_task"]
        self.assertTrue(actions)
        self.assertTrue(all(
            read_packet(task)["version"] == 3
            for step in actions for task in step["params"]["tasks"]
        ))
        self.assertTrue(all(
            "Required checks:" not in task["task_instruction"]
            for step in actions for task in step["params"]["tasks"]
        ))
        records = export_episode(episode, require_training=False)
        self.assertEqual(len(records), len(episode["trajectory"]))
        self.assertTrue(all("target_json" in row for row in records))
        for row in records:
            target=json.loads(row["target_json"])
            if target["action"]=="delegate_task":
                for task in target["params"]["tasks"]:
                    # Teacher can replay the original public packet, while
                    # student learns only a natural O1 context.
                    self.assertNotIn("[Public task packet]",task["context"])
                    self.assertNotIn("execution_checks",task["context"])
                    self.assertTrue(task["task_instruction"].strip())
        self.assertTrue(any(
            "execution_checks" in task["context"]
            for step in actions for task in step["params"]["tasks"]
        ))

    def test_failing_naturalization_falls_back_to_verified_contract(self):
        class FailingPlanner(PassthroughPlanner):
            style_provider = "natural"

            def naturalize_action(self, payload, canonical):
                raise ValueError("semantic_reviewer_rejected")

        q, record, rendered = fixture(dependent=True)
        episode = synthesize_one(
            q, record, rendered, FailingPlanner(), PassthroughExecutor(),
            schedule="compact",
        )
        self.assertTrue(episode["validation"]["accepted"])
        self.assertGreater(episode["metadata"]["natural_language_fallbacks"], 0)
        self.assertTrue(any(a.get("fallback") for a in episode["generation_audit"]))
        delegated = next(step for step in episode["trajectory"]
                         if step["action"] == "delegate_task")
        self.assertEqual(
            read_packet(delegated["params"]["tasks"][0])["version"], 2,
        )
        export_episode(episode, require_training=False)


if __name__ == "__main__":
    unittest.main()
