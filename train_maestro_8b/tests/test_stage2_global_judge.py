"""CPU-only regression tests: GLOBAL per-Action Judge, factual provenance and strict t-1."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train_maestro_8b.policy.global_judge import (
    DIMENSIONS, RUBRIC_VERSION, build_global_judge_prompt,
    global_user_payload, make_global_judge, parse_global_judge_response,
)
from train_maestro_8b.policy.global_judge_context import (
    SCHEMA, build_global_context, index_claim_snapshots, load_global_contexts,
)
from train_maestro_8b.policy.trl_reward_adapter import (
    make_stage2_reward_function, o1_contract_reason,
)
from train_maestro_8b.tests.test_stage2_trl_reward import SAMPLE, VISIBLE, row, state


def scene():
    return {
        "task_id": "scene_A",
        "gold_answer": [{"status": "Approved"}],
        "public": {"available_source_descriptions": [
            {"source_id": "source_A", "title": "public registry",
             "purpose": "status evidence", "modality_hint": "text"},
        ]},
        "world": {"sources": {
            "source_A": {
                "title": "Registry", "modality": "text",
                "observations": [{
                    "observation_id": "source_A.OBS001",
                    "fact_refs": ["source_A:F001"],
                    "values": {"status": "Approved"},
                }],
            },
        }},
        "macro_graph": {
            "terminal_task_id": "B",
            "tasks": [
                {"task_id": "A", "depends_on": [], "source_ids": ["source_A"],
                 "operator_chain": [{"op": "scan", "source_id": "source_A"}],
                 "outputs": {"typed_relation": ["status"]}},
                {"task_id": "B", "depends_on": ["A"], "source_ids": [],
                 "operator_chain": [{"op": "project"}],
                 "outputs": {"typed_relation": ["status"]}},
            ],
        },
    }


def claim_record():
    return {
        "task_id": "scene_A", "round_sample_id": "scene_A-fork-round-02",
        "main_can_act_from_round": 3,
        "schema": "claim_source_dependency", "schema_version": 1,
        "private_only": True,
        "claim_nodes": [
            {"claim_id": "c1", "report_task_id": "A", "observed_in_round": 2,
             "source_id": "source_A", "evidence_origin_id": "origin:A",
             "subject": "project", "attribute": "status", "value": "Pending",
             "delivery_status": "done", "private_fault_kind": "fabricated_claim"},
        ],
        "claim_states": {
            "project|status": {
                "subject": "project", "attribute": "status",
                "status": "disputed", "sufficient_to_accept": False,
                "missing_evidence": ["resolve_disagreement"],
                "answer_supporting_subject_private": True,
            },
        },
        "claim_relations": [], "claim_dependency_edges": [],
        "task_dependency_edges": [
            {"parent_task_id": "A", "child_task_id": "B",
             "relation": "executed_task_depends_on"},
        ],
        "report_events": [
            {"macro_task_id": "A", "source_id": "source_A",
             "status": "done", "visible_claim_count": 1},
        ],
        "transitions": [
            {"subject": "project", "before": "unverified", "after": "disputed",
             "next_main_decision_round": 3,
             "answer_supporting_subject_private": True},
        ],
    }


def response():
    return json.dumps({
        "task_tool_alignment": 8.0,
        "evidence_grounding": 7.5,
        "recovery_quality": 6.0,
        "global_orchestration_quality": 7.0,
        "reason": "Targets a relevant source but does not verify all conflicting reports.",
    })


class GlobalContextTests(unittest.TestCase):
    def test_evidence_facts_truth_and_task_graph_in_separate_lanes(self):
        ctx = build_global_context(scene(), state(), claim_record(), round_number=3)
        self.assertEqual(ctx["schema"], SCHEMA)
        self.assertEqual(ctx["task_graph"]["tasks"][1]["depends_on"], ["A"])
        self.assertEqual(ctx["observed_evidence"]["claim_nodes"][0]["value"], "Pending")
        self.assertEqual(
            ctx["evaluator_only_reference"]["reference_source_facts_for_reported_sources"][0]
            ["reference_observations"][0]["values"]["status"], "Approved",
        )
        self.assertEqual(ctx["recovery_reference"]["observable_issues"][0]["status"], "disputed")
        self.assertNotIn("fabricated_claim", str(ctx))
        self.assertNotIn("answer_supporting_subject_private", str(ctx))
        self.assertEqual(ctx["evaluator_only_reference"]["reference_final_answer"],
                         [{"status": "Approved"}])

    def test_future_claim_and_duplicate_or_missing_snapshot_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "claim.jsonl"
            record = claim_record()
            file.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(len(index_claim_snapshots(file)), 1)
            file.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                index_claim_snapshots(file)
            record["claim_nodes"][0]["observed_in_round"] = 3
            file.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "future"):
                index_claim_snapshots(file)

    def test_exact_sample_and_round_join(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "scenes.jsonl", Path(temp) / "claim.jsonl"
            a.write_text(json.dumps(scene()) + "\n")
            b.write_text(json.dumps(claim_record()) + "\n")
            selected = row()
            info = json.loads(selected["extra_info_json"])
            info["step_index"] = 3
            selected["extra_info_json"] = json.dumps(info)
            contexts, summary = load_global_contexts(
                scenes_path=a, claim_states_path=b,
                private_states={SAMPLE: state()}, rows=[selected],
            )
            self.assertEqual(summary["with_observed_evidence"], 1)
            self.assertEqual(contexts[SAMPLE]["decision_round"], 3)
            b.write_text("")
            with self.assertRaisesRegex(ValueError, "Missing pre-decision"):
                load_global_contexts(
                    scenes_path=a, claim_states_path=b,
                    private_states={SAMPLE: state()}, rows=[selected],
                )


class GlobalJudgeTests(unittest.TestCase):
    def test_parse_continuous_calibrated_scoring(self):
        x = parse_global_judge_response(response())
        expected = .10 + .15 * .8 + .15 * .75 + .25 * .6 + .35 * .7
        self.assertAlmostEqual(x["reward"], expected)
        self.assertEqual(set(x["dimensions"]), set(DIMENSIONS))
        with self.assertRaises(ValueError):
            parse_global_judge_response('{"task_tool_alignment": 5, "reason": "incomplete"}')
        with self.assertRaises(ValueError):
            parse_global_judge_response(response().replace('8.0', 'true'))

    def test_o1_optional_fields_not_hard_rejected(self):
        concise = {
            "action": "delegate_task", "params": {"tasks": [
                {"task_instruction": "Verify source status", "model": "model_1"},
            ]},
        }
        self.assertIsNone(o1_contract_reason(json.dumps(concise)))
        self.assertEqual(o1_contract_reason("bad"), "invalid_json")
        self.assertEqual(o1_contract_reason(
            '{"action":"complete","params":{"answer":""}}'), "missing_completion_answer")

    def test_global_mode_scores_semantic_mismatch_instead_of_hard_minus_one(self):
        class FakeGlobalJudge:
            def __init__(self):
                self.calls = []
            def score_many(self, prompt, candidates):
                self.calls.append((prompt, candidates))
                return [{
                    "reward": .62,
                    "dimensions": {k: 6.0 for k in DIMENSIONS}, "cached": False,
                } for _ in candidates]
        judge = FakeGlobalJudge()
        global_ctx = build_global_context(scene(), state(), claim_record(), round_number=3)
        fn = make_stage2_reward_function(
            private_states={SAMPLE: state()}, prompts={SAMPLE: VISIBLE},
            global_contexts={SAMPLE: global_ctx},
            mode="global", judge=judge, num_generations=2,
        )
        # Syntax valid, wrong evidence field: the old exact binder fails.
        candidate = json.dumps({"action": "delegate_task",
                                "params": {"tasks": [{
                                    "task_instruction": "Verify status",
                                    "model": "model_1",
                                }]}})
        outputs = fn(
            completions=[candidate, "not JSON"],
            data_source=["maestro_stage4"]*2, ground_truth=["PRIVATE GOLD"]*2,
            extra_info_json=[row()["extra_info_json"]]*2,
        )
        self.assertEqual(outputs, [.62, -1])
        self.assertEqual(len(judge.calls), 1)
        self.assertIn("evaluator_only_global_context", judge.calls[0][0])
        self.assertNotIn("PRIVATE GOLD", str(judge.calls))
        self.assertNotIn("PRIVATE_ORACLE_NEVER_SEND", str(judge.calls))
        self.assertEqual(judge.calls[0][1], [candidate])

    def test_api_request_contains_reference_not_teacher_or_hidden_fault_label(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"TEST_JUDGE_KEY": "token"},
        ):
            registry = Path(temp) / "models.yaml"
            registry.write_text(
                "models:\n  fake:\n    backend: api\n    api_type: openai\n"
                "    base_url: https://localhost/v1\n"
            )
            judge = make_global_judge(
                registry_path=str(registry), model="fake",
                api_key_env="TEST_JUDGE_KEY", workers=1, timeout=5,
                retries=0, cache_path=str(Path(temp) / "global.jsonl"),
            )
            ctx = build_global_context(scene(), state(), claim_record(), round_number=3)
            prompt = build_global_judge_prompt(VISIBLE, ctx)
            payload = global_user_payload(prompt, '{"action":"complete","params":{"answer":"X"}}')
            self.assertIn("observed_evidence", str(payload))
            self.assertNotIn("fabricated_claim", str(payload))
            sent = []
            class Response:
                def raise_for_status(self): pass
                def json(self):
                    return {"choices": [{"message": {"content": response()}}]}
            def fake_post(url, *, headers, json, timeout):
                sent.append(json)
                return Response()
            with patch("train_maestro_8b.policy.semantic_judge.requests.post", side_effect=fake_post):
                scores = judge.score_many(prompt, ['{"action":"complete","params":{"answer":"X"}}'])
            self.assertEqual(len(sent), 1)
            self.assertIn("GLOBAL", sent[0]["messages"][0]["content"])
            self.assertIn("global_orchestration_quality", sent[0]["messages"][0]["content"])
            self.assertEqual(scores[0]["reward"], parse_global_judge_response(response())["reward"])
            self.assertIn(RUBRIC_VERSION, (Path(temp) / "global.jsonl").read_text())
            self.assertNotIn("PRIVATE_ORACLE_NEVER_SEND", str(sent))


if __name__ == "__main__":
    unittest.main()
