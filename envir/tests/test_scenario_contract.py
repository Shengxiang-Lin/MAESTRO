"""v4 Semantic Scenario Contract: exact World binding and Goal-only realization.

No GPU, API, original MMQA files, or network required.
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
from envir.macro.scene_grounding_cli import ground_scenes
from envir.macro.scenario_contract import (
    VERSION, constructor_input, ground_one_contract,
    frozen_question_intent, validate_contract_proposal,
    validate_scenario_contract, materialize_contract,
    source_fact_view,
)
from envir.macro.realization_brief import question_realization_brief
from envir.macro.semantic_contract import semantic_contract
from envir.macro.validation import validate_clean_scene


def proposal_for(row):
    sources = source_fact_view(row)
    topic = row["scenario"]["topic"]
    intent = row["goal"]["intent"]
    needs = {
        "lookup": "Identify the entries with a positive archive mark",
        "composition": "Link the archive references by their recorded group",
        "comparison": "Rank the entries by their internal recorded scores",
        "intersection": "Find the groups shared across multiple archive records",
        "aggregation": "Count the entries with a positive archive mark",
        "temporal": "Find the relative sequence and earliest archive entry",
        "prioritization": "Prioritize the entries by their recorded scores",
        "feasibility": "Find eligible entries with positive marks and scores of at least 50",
    }
    return {
        "status": "realizable",
        "reason": "none",
        "scenario": {
            "setting": (
                "In a fictional sports archive associated with " + topic +
                ", a curator checks a set of invented award submissions."
            ),
            "user_role": "Sports-archive curator",
            "motivation": (
                "Prepare an internally consistent submission summary for " +
                topic + " using only the currently recorded facts."
            ),
        },
        "semantic_bindings": {
            "candidate_kind": "fictional sports award submissions",
            "score": "internal evaluation points, not real sporting results",
            "positive_flag": "positive internal submission mark",
            "relative_time_index": "relative record-entry order",
            "group_key": "internal evaluation cohort, not identity",
        },
        "source_bindings": [
            {
                "source_id": source["source_id"],
                "candidate_label": "Fictional submission " + chr(65 + index),
                "source_role": (
                    "Hypothetical " + source["modality"] +
                    "-source metadata entry for this submission"
                ),
            }
            for index, source in enumerate(sources)
        ],
        "user_need": needs[intent] + " for " + topic + ".",
    }


def pass_review():
    return {
        "domain_plausibility": "yes",
        "semantic_coherence": "yes",
        "source_faithfulness": "yes",
        "goal_compatibility": "yes",
        "evidence_boundary": "yes",
        "user_need_alignment": "yes",
        "verdict": "pass",
        "issues": [],
        "summary": "Hypothetical domain roles bind exactly to existing synthetic data.",
    }


class Author:
    def __init__(self, *, answer="realizable"):
        self.calls = 0
        self.inputs = []
        self.answer = answer

    def complete_json(self, system, text, *, response_schema=None):
        self.calls += 1
        user = json.loads(text)
        self.inputs.append(user)
        assert "original_mmqa_question" not in json.dumps(user)
        if self.answer == "not_realizable":
            return {
                "status": "not_realizable",
                "reason": "No coherent domain interpretation is possible.",
                "scenario": {k: "" for k in (
                    "setting", "user_role", "motivation")},
                "semantic_bindings": {k: "" for k in (
                    "candidate_kind", "score", "positive_flag",
                    "relative_time_index", "group_key")},
                "source_bindings": [],
                "user_need": "",
            }
        # Derive test fixture from actual scene only, not the model input.
        return copy.deepcopy(self.fixture)


class Critic:
    def __init__(self, answer=None):
        self.calls = 0
        self.inputs = []
        self.answer = answer if answer is not None else pass_review()

    def complete_json(self, system, text, *, response_schema=None):
        self.calls += 1
        self.inputs.append(json.loads(text))
        return copy.deepcopy(self.answer)


class ScenarioContractTests(unittest.TestCase):
    def setUp(self):
        self.original = build_one(
            ordinal=12, seed=17, kqa=KQA, seed_record=scene(12))

    def _models(self):
        author, critic = Author(), Critic()
        author.fixture = proposal_for(self.original)
        return author, critic

    def test_author_input_has_no_original_mmqa_question_or_excerpts(self):
        original_question = self.original["source_priors"][
            "MultiModalQA"]["original_question"]
        request = constructor_input(self.original)
        self.assertNotIn("original_mmqa_question", request)
        self.assertNotIn("original_mmqa_question_weak_style_cue", request)
        self.assertNotIn("original_excerpt_context_only", json.dumps(request))
        self.assertNotIn(original_question, json.dumps(request))
        self.assertEqual(
            request["goal_contract"], frozen_question_intent(self.original))
        self.assertEqual(len(request["task_operations"]),
                         len(self.original["macro_graph"]["tasks"]))
        self.assertEqual(len(request["source_records"]),
                         len(self.original["world"]["sources"]))
        self.assertIn("fact_ref", request["source_records"][0]["fields"]["score"])

    def test_full_contract_preserves_source_bindings_and_replay(self):
        author, critic = self._models()
        proposal = author.fixture
        # Deliberately reorder exact source roles; canonical binding must
        # be stable without requiring the LLM to return sorted source IDs.
        proposal["source_bindings"].reverse()
        overlay, review = ground_one_contract(
            self.original, constructor=author, critic=critic)
        self.assertEqual("accepted", review["status"])
        self.assertEqual((author.calls, critic.calls), (1, 1))
        self.assertEqual(VERSION, overlay["version"])
        self.assertEqual(overlay["question_intent"],
                         frozen_question_intent(self.original))
        enriched = copy.deepcopy(self.original)
        enriched["scene_grounding"] = overlay
        self.assertEqual([], validate_scenario_contract(enriched))
        self.assertEqual([], validate_clean_scene(enriched))
        self.assertEqual(
            enriched["world"], self.original["world"])
        self.assertEqual(
            enriched["macro_graph"], self.original["macro_graph"])
        self.assertEqual(
            enriched["task_evidence_graphs"],
            self.original["task_evidence_graphs"])
        self.assertEqual(
            enriched["feedback_bank"], self.original["feedback_bank"])
        for card in overlay["source_cards"]:
            self.assertIn(card["source_id"], enriched["world"]["sources"])
            for value in card["fact_bindings"].values():
                self.assertIn("fact_ref", value)
        self.assertIn(
            "synthetic_scenario_contract",
            semantic_contract(enriched))
        self.assertIn(
            "hypothetical_scene",
            question_realization_brief(enriched))
        self.assertNotIn(
            "synthetic_scene_interpretation",
            semantic_contract(enriched))

    def test_goal_operation_cannot_shift_even_if_prose_is_fluent(self):
        proposal = proposal_for(self.original)
        proposal["user_need"] = (
            "Investigate the historical founding year of Subject 12 teams.")
        self.assertIn("user_need_operation_missing",
                      validate_contract_proposal(self.original, proposal))

    def test_source_tamper_and_contract_change_are_detected(self):
        overlay = materialize_contract(
            self.original, proposal_for(self.original), pass_review())
        base = {**self.original, "scene_grounding": overlay}
        self.assertEqual([], validate_scenario_contract(base))
        wrong = copy.deepcopy(base)
        wrong["scene_grounding"]["source_cards"][0][
            "fact_bindings"]["score"]["value"] = -100
        self.assertIn("scenario_contract_materialization_drift",
                      validate_clean_scene(wrong))
        wrong = copy.deepcopy(base)
        wrong["scene_grounding"]["question_intent"]["operation"] = "lookup"
        self.assertIn("scenario_contract_goal_drift",
                      validate_clean_scene(wrong))
        wrong = copy.deepcopy(base)
        wrong["scene_grounding"]["scenario"]["motivation"] = (
            "Find an unrelated date in another source.")
        self.assertIn("scenario_contract_proposal_drift",
                      validate_clean_scene(wrong))
        wrong = copy.deepcopy(base)
        wrong["scene_grounding"]["source_cards"][0]["source_id"] = "invented"
        self.assertIn("source_inventory_changed",
                      validate_clean_scene(wrong))

    def test_no_real_image_perception_or_historical_dates(self):
        proposal = proposal_for(self.original)
        proposal["semantic_bindings"]["positive_flag"] = (
            "Visually confirmed actual logo in the original image.")
        self.assertIn("unsupported_media_perception",
                      validate_contract_proposal(self.original, proposal))
        proposal = proposal_for(self.original)
        proposal["scenario"]["setting"] += " In 2017 this team won."
        self.assertIn("unsupported_calendar_year",
                      validate_contract_proposal(self.original, proposal))

    def test_not_realizable_is_explicit_and_preserves_original(self):
        author, critic = Author(answer="not_realizable"), Critic()
        overlay, review = ground_one_contract(
            self.original, constructor=author, critic=critic)
        self.assertIsNone(overlay)
        self.assertEqual(review["status"], "not_realizable")
        self.assertEqual(author.calls, 1)
        self.assertEqual(critic.calls, 0)
        self.assertEqual([], validate_clean_scene(self.original))

    def test_critic_rejection_keeps_original(self):
        author, _ = self._models()
        critic = Critic(answer={
            **pass_review(), "semantic_coherence": "no",
            "verdict": "revise", "issues": [
                "A fictional review score cannot be treated as a birth date."
            ],
        })
        overlay, meta = ground_one_contract(
            self.original, constructor=author, critic=critic,
            max_revisions=0)
        self.assertIsNone(overlay)
        self.assertEqual(meta["status"], "rejected")
        self.assertEqual([], validate_clean_scene(self.original))

    def test_cli_v4_writes_and_resumes_and_tracks_realizability(self):
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "source.jsonl"
            src.write_text(
                json.dumps(self.original, ensure_ascii=False) + "\n",
                encoding="utf-8")
            out = Path(folder) / "v4"
            args = argparse.Namespace(
                scenes=src, output_dir=out, max_records=1,
                grounding_api_base="http://127.0.0.1:8807/v1",
                grounding_model="local-qwen3.5-9b",
                grounding_critic_api_base=None, grounding_critic_model=None,
                grounding_api_key_env="OPENAI_API_KEY",
                grounding_critic_api_key_env=None,
                grounding_workers=1, grounding_temperature=.7,
                grounding_critic_temperature=0.0,
                grounding_max_tokens=2200, grounding_critic_max_tokens=1400,
                grounding_timeout=180, grounding_no_json_mode=False,
                max_revisions=1, flush_every=1, no_progress=True,
                overwrite=False)
            author, critic = self._models()
            with patch("envir.macro.scene_grounding_cli._clients",
                       return_value=(author, critic)):
                stats = ground_scenes(args)
                self.assertEqual(stats["statistics"]["accepted"], 1)
                self.assertEqual(stats["statistics"]["not_realizable"], 0)
                self.assertEqual((author.calls, critic.calls), (1, 1))
                same = ground_scenes(args)
                self.assertEqual(same["statistics"]["accepted"], 1)
                self.assertEqual((author.calls, critic.calls), (1, 1))
            realized = json.loads((out / "scenes.jsonl").read_text().strip())
            self.assertEqual(realized["scene_grounding"]["version"], VERSION)
            self.assertEqual([], validate_clean_scene(realized))


if __name__ == "__main__":
    unittest.main()
