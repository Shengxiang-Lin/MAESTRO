"""Semantic specification BEFORE numeric World/EvidenceGraph instantiation.

Uses compiled TaskGraph fixtures and mocked LLMs. No real media or GPU.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_taskfirst import KQA, scene
from envir.macro.builder import build_one
from envir.macro.semantic_world import (
    VERSION, plan_input, goal_contract_from_plan, source_manifest_from_plan,
    validate_proposal, materialize_spec, validate_spec,
    realize_spec_one, compile_semantic_world_scene,
    validate_semantic_world_scene,
)
from envir.macro.semantic_world_cli import realize_world
from envir.macro.question_review import macro_task_view
from envir.macro.realization_brief import question_realization_brief
from envir.macro.semantic_contract import semantic_contract
from envir.macro.validation import validate_clean_scene


def proposal_for(plan):
    topic = plan["scenario"]["topic"]
    manifest = source_manifest_from_plan(plan)
    need = {
        "lookup": "Identify the positively marked submissions",
        "comparison": "Compare and rank the fictional submissions by internal score",
        "aggregation": "Count the submissions with a positive recorded review mark",
        "intersection": "Find which archive groups recur across source records",
        "composition": "Link fictional archive entries by their shared reference groups",
        "temporal": "Arrange submissions by their relative order and find the earliest",
        "prioritization": "Prioritize submissions in descending order by review score",
        "feasibility": "Find eligible submissions marked positive with scores at least 50",
    }[plan["goal"]["intent"]]
    return {
        "status": "realizable", "reason": "none",
        "scenario": {
            "subject_type": "fictional sports award records",
            "setting": (
                f"A fictional sports archive for {topic} collects award "
                "submission records for an internal inventory."),
            "user_role": "sports-archive curator",
            "motivation": (
                "Prepare an internally consistent inventory review of "
                "the fictional award submissions."),
        },
        "semantic_bindings": {
            "candidate_kind": "fictional award submission",
            "score": "internal review points on a synthetic 0 to 100 scale",
            "positive_flag": "positive internal review mark",
            "relative_time_index": "relative order in an archival intake batch",
            "group_key": "archival submission cohort",
        },
        "source_bindings": [{
            "source_id": row["source_id"],
            "candidate_label": f"Submission {i+1}",
            "source_role": (
                "hypothetical " + row["modality"] +
                "-source metadata record for an award submission"),
        } for i, row in enumerate(manifest)],
        "user_need": need + f" for {topic}.",
    }


def pass_review():
    return {
        "topic_coherence": "yes",
        "field_coherence": "yes",
        "source_faithfulness": "yes",
        "goal_compatibility": "yes",
        "no_unsupported_media_or_history": "yes",
        "verdict": "pass",
        "issues": [],
        "summary": "Coherent fictional archive and Goal without new facts.",
    }


class Author:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0
        self.inputs = []

    def complete_json(self, system, user, *, response_schema=None):
        self.calls += 1
        self.inputs.append(json.loads(user))
        return copy.deepcopy(self.proposal)


class Critic:
    def __init__(self, review=None):
        self.calls = 0
        self.inputs = []
        self.review = review or pass_review()

    def complete_json(self, system, user, *, response_schema=None):
        self.calls += 1
        self.inputs.append(json.loads(user))
        return copy.deepcopy(self.review)


class SemanticWorldTests(unittest.TestCase):
    def setUp(self):
        self.original = build_one(
            ordinal=12, seed=17, kqa=KQA, seed_record=scene(12))
        self.proposal = proposal_for(self.original)

    def test_llm_receives_plan_only_not_old_world_or_original_mmqa(self):
        request = plan_input(self.original)
        raw = json.dumps(request, ensure_ascii=False)
        original_question = self.original["source_priors"][
            "MultiModalQA"]["original_question"]
        self.assertNotIn(original_question, raw)
        self.assertNotIn("original_mmqa_question", raw)
        self.assertNotIn("source_records", request)
        self.assertNotIn("feedback_bank", raw)
        self.assertNotIn("synthetic_attribute_score", raw.replace(
            json.dumps(request["goal_contract"]), ""))
        self.assertIn("goal_contract", request)
        self.assertEqual(
            request["goal_contract"], goal_contract_from_plan(self.original))
        self.assertEqual(
            {row["source_id"] for row in request["source_manifest"]},
            set(self.original["world"]["sources"]),
        )

    def test_semantic_spec_precedes_values_and_all_evidence_replays(self):
        author, critic = Author(self.proposal), Critic()
        spec, review = realize_spec_one(
            self.original, author=author, critic=critic)
        self.assertEqual(review["status"], "accepted")
        self.assertEqual((author.calls, critic.calls), (1, 1))
        self.assertEqual(spec["version"], VERSION)
        self.assertEqual([], validate_spec(self.original, spec))
        # No score, flag, candidate IDs or sample value is ever model-written.
        self.assertNotIn("fact_bindings", spec)
        self.assertNotIn("candidate_key", spec)
        self.assertNotIn("score_value", spec)
        realized = compile_semantic_world_scene(self.original, spec)
        self.assertEqual([], validate_clean_scene(realized))
        self.assertEqual([], validate_semantic_world_scene(realized))
        self.assertEqual(
            realized["macro_graph"], self.original["macro_graph"])
        self.assertEqual(realized["goal"], self.original["goal"])
        self.assertEqual(realized["question"], self.original["question"])
        self.assertNotEqual(realized["world"], self.original["world"])
        self.assertNotEqual(realized["feedback_bank"],
                            self.original["feedback_bank"])
        for source in realized["world"]["sources"].values():
            facts = {row["field"]: row for row in source["facts"]}
            score = facts["synthetic_attribute_score"]["value"]
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            self.assertIsInstance(
                facts["synthetic_attribute_flag"]["value"], bool)
            self.assertIn(
                "internal review points",
                facts["synthetic_attribute_score"]["statement"])
            self.assertNotIn("original_context_excerpt", facts)
            self.assertIn("semantic_candidate_label", source)
        self.assertIn(
            "synthetic_world_specification", semantic_contract(realized))
        self.assertEqual(
            macro_task_view(realized)["scenario"]["domain"],
            spec["scenario"]["subject_type"])
        self.assertEqual(
            macro_task_view(self.original)["scenario"]["domain"],
            self.original["scenario"]["domain"])
        brief = question_realization_brief(realized)
        self.assertIn("hypothetical_scene", brief)
        self.assertIn("internal review points",
                      str(brief["hypothetical_scene"]))
        # Rebuilding from the SAME spec and frozen Plan is reproducible.
        other = compile_semantic_world_scene(self.original, spec)
        self.assertEqual(realized["world"], other["world"])
        self.assertEqual(
            realized["task_evidence_graphs"], other["task_evidence_graphs"])

    def test_unsorted_source_bindings_keep_same_frozen_spec(self):
        swapped = copy.deepcopy(self.proposal)
        swapped["source_bindings"].reverse()
        self.assertEqual([], validate_proposal(self.original, swapped))
        first = materialize_spec(self.original, self.proposal, pass_review())
        second = materialize_spec(self.original, swapped, pass_review())
        self.assertEqual(first, second)

    def test_source_ids_and_world_numeric_values_cannot_be_tampered(self):
        broken = copy.deepcopy(self.proposal)
        broken["source_bindings"][0]["source_id"] = "invented"
        self.assertIn("source_inventory_changed",
                      validate_proposal(self.original, broken))
        spec = materialize_spec(
            self.original, self.proposal, pass_review())
        realized = compile_semantic_world_scene(self.original, spec)
        broken = copy.deepcopy(realized)
        sid = sorted(broken["world"]["sources"])[0]
        for fact in broken["world"]["sources"][sid]["facts"]:
            if fact["field"] == "synthetic_attribute_score":
                fact["value"] = -999
                break
        self.assertIn(
            "semantic_world_values_or_source_roles_drift",
            validate_clean_scene(broken))
        broken = copy.deepcopy(realized)
        broken["semantic_world_spec"]["semantic_bindings"]["score"] = (
            "actual historical goals")
        self.assertTrue(validate_clean_scene(broken))

    def test_empty_possible_answer_is_not_reason_to_reject_schema(self):
        # The schema is constructed BEFORE values are assigned. A Goal
        # may later yield zero eligible candidates or no shared group;
        # these are valid answers, not incompatible Scene/Goal pairs.
        proposal = copy.deepcopy(self.proposal)
        proposal["user_need"] = (
            "Count positively marked fictional submissions for Subject 12.")
        self.assertEqual([], validate_proposal(self.original, proposal))
        spec = materialize_spec(self.original, proposal, pass_review())
        self.assertEqual([], validate_spec(self.original, spec))

    def test_no_actual_source_media_claim(self):
        broken = copy.deepcopy(self.proposal)
        broken["source_bindings"][0]["source_role"] = (
            "A photo shows the verified real winner")
        self.assertIn(
            "unsupported_actual_media_perception",
            validate_proposal(self.original, broken))

    def test_explicit_not_realizable_does_not_force_rebuild(self):
        rejected = {
            "status": "not_realizable",
            "reason": "The topic is not interpretable with these sources.",
            "scenario": {k: "" for k in (
                "subject_type", "setting", "user_role", "motivation")},
            "semantic_bindings": {k: "" for k in (
                "candidate_kind", "score", "positive_flag",
                "relative_time_index", "group_key")},
            "source_bindings": [], "user_need": "",
        }
        spec, review = realize_spec_one(
            self.original, author=Author(rejected), critic=Critic())
        self.assertIsNone(spec)
        self.assertEqual("not_realizable", review["status"])
        self.assertEqual([], validate_clean_scene(self.original))

    def test_streaming_command_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "base.jsonl"
            input_path.write_text(
                json.dumps(self.original, ensure_ascii=False) + "\n",
                encoding="utf-8")
            folder = root / "result"
            args = argparse.Namespace(
                scenes=input_path, output_dir=folder, max_records=1,
                grounding_api_base="http://127.0.0.1:8807/v1",
                grounding_model="local-qwen3.5-9b",
                grounding_critic_api_base=None, grounding_critic_model=None,
                grounding_api_key_env="OPENAI_API_KEY",
                grounding_critic_api_key_env=None,
                grounding_workers=2, grounding_temperature=.6,
                grounding_critic_temperature=0.0,
                grounding_max_tokens=2000, grounding_critic_max_tokens=1400,
                grounding_timeout=180, grounding_no_json_mode=False,
                max_revisions=1, flush_every=1, no_progress=True,
                overwrite=False,
            )
            author, critic = Author(self.proposal), Critic()
            with patch("envir.macro.semantic_world_cli._clients",
                       return_value=(author, critic)):
                result = realize_world(args)
                again = realize_world(args)
            self.assertEqual(result["statistics"]["accepted"], 1)
            self.assertEqual(again["statistics"]["accepted"], 1)
            self.assertEqual((author.calls, critic.calls), (1, 1))
            actual = json.loads((folder / "scenes.jsonl").read_text().strip())
            self.assertEqual([], validate_clean_scene(actual))
            self.assertEqual(actual["semantic_world_spec"]["version"], VERSION)
            self.assertEqual(
                len((folder / "accepted_scenes.jsonl").read_text().splitlines()),
                1)
            self.assertNotIn("semantic_world_spec", self.original)


if __name__ == "__main__":
    unittest.main()
