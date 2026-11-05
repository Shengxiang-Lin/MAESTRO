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
