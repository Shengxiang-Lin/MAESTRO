"""Regression tests for post-TaskGraph synthetic World concretization.

All model calls are deterministic mocks. No internet, GPU or MMQA download.
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
from envir.macro.scene_grounding import (
    constructor_input, ground_one, materialize_grounding,
    source_records, validate_grounding, validate_proposal, validate_review,
    frozen_fingerprint, source_ablation_signal,
)
from envir.macro.scene_grounding_cli import ground_scenes
from envir.macro.semantic_contract import semantic_contract
from envir.macro.realization_brief import question_realization_brief
from envir.macro.validation import validate_clean_scene


def proposal_for(scene_value):
    records = source_records(scene_value)
    return {
        "scene_context": (
            "In a hypothetical editorial review of " +
            scene_value["scenario"]["topic"] +
            ", a curator cross-checks a set of fictional assessment entries."
        ),
        "candidate_kind": "fictional review submissions",
        "field_meanings": {
            "score": "evaluation points in the hypothetical review",
            "positive_flag": "a positive evaluation indicator",
            "relative_time_index": "the relative entry sequence",
            "group_key": "a hypothetical comparison cohort",
        },
        "source_roles": [{
            "source_id": row["source_id"],
            "candidate_label": "Submission " + chr(65 + i),
            "source_role": (
                "Hypothetical " + row["modality"] + "-reference "
                "assessment record"
            ),
        } for i, row in enumerate(records)],
        "user_need": (
            "Determine the requested source-supported outcome for "
            "the fictional review entries."
        ),
    }


def pass_review():
    return {
        "domain_plausibility": "yes",
        "frozen_facts": "yes",
        "source_roles": "yes",
        "solution_compatibility": "yes",
        "question_alignment": "yes",
        "verdict": "pass",
        "issues": [],
        "summary": "The fictional review labels match the frozen source facts.",
    }


class Author:
    def __init__(self):
        self.calls = 0
        self.payloads = []

    def complete_json(self, system, user, *, response_schema=None):
        self.calls += 1
        payload = json.loads(user)
        self.payloads.append(payload)
        rows = payload["source_records"]
        return {
            "scene_context": (
                "In a hypothetical editorial review of " + payload["topic"] +
                ", a curator cross-checks fictional assessment entries."
            ),
            "candidate_kind": "fictional review submissions",
            "field_meanings": {
                "score": "hypothetical evaluation points",
                "positive_flag": "the positive review indicator",
                "relative_time_index": "relative record sequence",
                "group_key": "grouped evaluation cohort",
            },
            "source_roles": [{
                "source_id": item["source_id"],
                "candidate_label": "Submission " + chr(65 + i),
                "source_role": "Simulated " + item["modality"] + " record",
            } for i, item in enumerate(rows)],
            "user_need": "Identify the result required by the existing source review.",
        }


class Critic:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user, *, response_schema=None):
        self.calls += 1
        return pass_review()


class WorldConcretizationTests(unittest.TestCase):
    def setUp(self):
        self.original = build_one(
            ordinal=12, seed=17, kqa=KQA, seed_record=scene(12))

    def test_constructor_sees_reference_question_world_and_evidencegraph(self):
        request = constructor_input(self.original)
        self.assertEqual(request["reference_question"], self.original["question"])
        self.assertEqual(request["original_mmqa_question_weak_style_cue"],
                         self.original["source_priors"]["MultiModalQA"]["original_question"])
        self.assertEqual(
            len(request["evidence_graph_operations"]),
            len(self.original["macro_graph"]["tasks"]),
        )
        self.assertEqual(
            {row["source_id"] for row in request["source_records"]},
            set(self.original["world"]["sources"]),
        )
        first = request["source_records"][0]
        for key in ("entry", "score", "positive_mark", "relative_order", "group"):
            self.assertIn("fact_ref", first["fields"][key])
            self.assertIn("value", first["fields"][key])
        # A simulation status may legitimately say "not_gold_answer";
        # it is the private Gold ANSWER KEY itself that must be excluded.
        self.assertNotIn("gold_answer", request)
        self.assertNotIn("gold_answer_key", request)

    def test_world_concretization_preserves_frozen_graph_evidence_and_values(self):
        author, critic = Author(), Critic()
        overlay, meta = ground_one(
            self.original, constructor=author, critic=critic)
        self.assertEqual(meta["status"], "accepted")
        self.assertEqual((author.calls, critic.calls), (1, 1))
        enriched = copy.deepcopy(self.original)
        original_hash = frozen_fingerprint(enriched)
        enriched["scene_grounding"] = overlay
        self.assertEqual(original_hash, frozen_fingerprint(enriched))
        self.assertEqual([], validate_grounding(enriched))
        self.assertEqual([], validate_clean_scene(enriched))
        for card in overlay["source_cards"]:
            source = enriched["world"]["sources"][card["source_id"]]
            facts = {item["field"]: item for item in source["facts"]}
            for field, output_name in [
                ("candidate_key", "entry"),
                ("synthetic_attribute_score", "score"),
                ("synthetic_attribute_flag", "positive_mark"),
                ("synthetic_time_index", "relative_order"),
                ("synthetic_group_key", "group"),
            ]:
                actual = card["fact_bindings"][output_name]
                self.assertEqual(actual["value"], facts[field]["value"])
                self.assertEqual(
                    actual["fact_ref"],
                    card["source_id"] + ":" + facts[field]["fact_id"],
                )
        enriched_contract = semantic_contract(enriched)
        enriched_brief = question_realization_brief(enriched)
        self.assertIn("synthetic_scene_interpretation", enriched_contract)
        self.assertIn("hypothetical_scene", enriched_brief)
        self.assertNotIn("macro_task_graph", enriched_brief)

    def test_source_ablation_is_diagnostic_not_rejection_rule(self):
        diagnostic = source_ablation_signal(self.original)
        self.assertEqual(diagnostic["goal_id"], "aggregation")
        self.assertEqual(diagnostic["source_count"], 3)
        # In this fixture one positive mark contributes to the aggregate;
        # two negative records do not change the count if omitted.
        self.assertEqual(len(diagnostic["influential_sources"]), 1)
        self.assertEqual(len(diagnostic["non_influential_sources"]), 2)
        proposal = proposal_for(self.original)
        overlay = materialize_grounding(
            self.original, proposal, pass_review())
        self.assertEqual(
            overlay["validation"]["goal_level_source_ablation"], diagnostic)
        self.assertEqual(
            [], validate_grounding({
                **self.original, "scene_grounding": overlay
            }))

    def test_natural_descriptions_do_not_fail_on_arbitrary_110_character_cap(self):
        proposal = proposal_for(self.original)
        proposal["field_meanings"]["score"] = (
            "A fictional internal confidence score between zero and one "
            "hundred, assigned by this hypothetical audit to compare "
            "the existing synthetic candidate records; it is not a "
            "real-world sport or financial measurement."
        )
        proposal["source_roles"][0]["source_role"] = (
            "Hypothetical source metadata record from a source of the "
            "existing modality, describing an invented review submission "
            "and its synthetic attributes, without claiming any real "
            "image was viewed or historical fact was verified."
        )
        proposal["user_need"] = (
            "For this hypothetical review case, identify the complete "
            "result required by the existing source evidence and preserve "
            "the established goal and candidate universe, without "
            "introducing new facts, sources, metrics or operations."
        )
        self.assertGreater(len(proposal["field_meanings"]["score"]), 110)
        self.assertGreater(len(proposal["source_roles"][0]["source_role"]), 110)
        self.assertEqual([], validate_proposal(self.original, proposal))
        reviewer = {**pass_review(), "summary": "Same semantics. " * 55}
        self.assertGreater(len(reviewer["summary"]), 600)
        output, issues = validate_review(reviewer)
        self.assertEqual(issues, [])
        self.assertEqual(output["verdict"], "pass")

    def test_v2_blocks_false_media_observation_v1_is_replayable(self):
        proposal = proposal_for(self.original)
        proposal["source_roles"][0]["source_role"] = (
            "Image-derived observation containing visual confirmation "
            "of the actual logo."
        )
        self.assertIn(
            "unsupported_media_perception_certainty",
            validate_proposal(self.original, proposal),
        )
        # Legacy accepted overlays retain their original validation
        # policy, instead of silently changing the replay contract.
        self.assertEqual(
            [], validate_proposal(self.original, proposal, legacy_v1=True)
        )
        previous = materialize_grounding(
            self.original, proposal, pass_review(),
            version="synthetic_world_concretization_v1",
        )
        self.assertEqual(
            [], validate_grounding({**self.original,
                                    "scene_grounding": previous})
        )

    def test_source_inventory_remains_strict_even_after_length_relaxation(self):
        proposal = proposal_for(self.original)
        proposal["source_roles"].append({
            "source_id": "new-unauthorized-source",
            "candidate_label": "Submission Z",
            "source_role": "Invented fourth source",
        })
        issues = validate_proposal(self.original, proposal)
        self.assertIn("source_role_count_mismatch", issues)
        self.assertIn("unknown_or_duplicate_source_id", issues)

    def test_v3_proposal_role_order_is_semantically_irrelevant(self):
        """Unsorted valid source list must not cause a replay failure."""
        proposal = proposal_for(self.original)
        proposal["source_roles"] = list(reversed(proposal["source_roles"]))
        self.assertEqual([], validate_proposal(self.original, proposal))
        overlay = materialize_grounding(
            self.original, proposal, pass_review())
        self.assertEqual(
            overlay["version"], "synthetic_world_concretization_v3")
        self.assertEqual(
            [], validate_grounding({**self.original, "scene_grounding": overlay}))
        normal = proposal_for(self.original)
        overlay_sorted = materialize_grounding(
            self.original, normal, pass_review())
        self.assertEqual(
            overlay["proposal_fingerprint"],
            overlay_sorted["proposal_fingerprint"])
        self.assertEqual(overlay["source_cards"], overlay_sorted["source_cards"])

        class UnsortedAuthor:
            def complete_json(self, system, user, *, response_schema=None):
                return proposal
        generated, review = ground_one(
            self.original, constructor=UnsortedAuthor(), critic=Critic(),
            max_revisions=0)
        self.assertEqual(review["status"], "accepted")
        self.assertEqual([], validate_grounding({
            **self.original, "scene_grounding": generated,
        }))

    def test_recoverable_nonjson_model_output_gets_a_bounded_retry(self):
        class FlakyAuthor:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, user, *, response_schema=None):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("model did not return a JSON object")
                return proposal_for(self_original)
        self_original = self.original
        author = FlakyAuthor()
        overlay, meta = ground_one(
            self.original, constructor=author, critic=Critic(),
            max_revisions=1)
        self.assertEqual(meta["status"], "accepted")
        self.assertEqual(meta["author_calls"], 2)
        self.assertEqual([], validate_grounding({
            **self.original, "scene_grounding": overlay,
        }))

    def test_tampered_value_or_source_id_is_rejected(self):
        overlay = materialize_grounding(
            self.original, proposal_for(self.original), pass_review())
        enriched = copy.deepcopy(self.original)
        enriched["scene_grounding"] = overlay
        self.assertEqual([], validate_grounding(enriched))
        broken = copy.deepcopy(enriched)
        broken["scene_grounding"]["source_cards"][0]["fact_bindings"]["score"]["value"] = -999
        self.assertIn("grounding_fact_bindings_changed",
                      validate_clean_scene(broken))
        broken = copy.deepcopy(enriched)
        broken["scene_grounding"]["source_cards"][0]["source_id"] = "new-source"
        self.assertTrue(validate_grounding(broken))
        broken = copy.deepcopy(enriched)
        broken["world"]["sources"][next(iter(broken["world"]["sources"]))]["facts"][0]["value"] = "altered"
        self.assertIn("grounding_world_fingerprint_changed",
                      validate_grounding(broken))

    def test_missing_source_or_invented_media_assertion_rejected_before_critic(self):
        proposal = proposal_for(self.original)
        proposal["source_roles"].pop()
        self.assertIn("source_role_count_mismatch",
                      validate_proposal(self.original, proposal))
        proposal = proposal_for(self.original)
        proposal["scene_context"] = "An image shows the winning player."
        self.assertIn("unobserved_media_content_claim",
                      validate_proposal(self.original, proposal))

    def test_critic_rejects_unsupported_scenario_without_rewriting_task(self):
        author = Author()
        class RejectCritic:
            def complete_json(self, *args, **kwargs):
                return {
                    **pass_review(),
                    "source_roles": "no",
                    "verdict": "revise",
                    "issues": ["The source label claims an extra document."],
                }
        overlay, review = ground_one(
            self.original, constructor=author, critic=RejectCritic(),
            max_revisions=0)
        self.assertIsNone(overlay)
        self.assertEqual(review["status"], "rejected")
        self.assertNotIn("scene_grounding", self.original)
        self.assertEqual([], validate_clean_scene(self.original))

    def test_critic_contradictory_pass_becomes_revise(self):
        verdict = {**pass_review(), "frozen_facts": "no",
                   "issues": ["Invented an original award winner"]}
        result, problems = validate_review(verdict)
        self.assertEqual(problems, [])
        self.assertEqual(result["verdict"], "revise")

    def test_cli_writes_and_resumes_without_model_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            source = home / "base.jsonl"
            source.write_text(
                json.dumps(self.original, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            out = home / "world50"
            args = argparse.Namespace(
                scenes=source, output_dir=out, max_records=1,
                grounding_api_base="http://127.0.0.1:8807/v1",
                grounding_model="local-qwen3.5-9b",
                grounding_critic_api_base=None, grounding_critic_model=None,
                grounding_api_key_env="OPENAI_API_KEY",
                grounding_critic_api_key_env=None,
                grounding_workers=2, grounding_temperature=.7,
                grounding_critic_temperature=0.0,
                grounding_max_tokens=2200, grounding_critic_max_tokens=1400,
                grounding_timeout=180, grounding_no_json_mode=False,
                max_revisions=1, flush_every=1, no_progress=True,
                overwrite=False,
            )
            author, critic = Author(), Critic()
            # Legacy v3 runner behavior remains supported by the
            # replay validator. Patch its legacy constructor explicitly,
            # while the new v4 pipeline is tested in test_scenario_contract.
            with patch("envir.macro.scene_grounding_cli._clients",
                       return_value=(author, critic)), patch(
                "envir.macro.scene_grounding_cli.ground_one_contract",
                side_effect=lambda sc, constructor, critic, max_revisions: (
                    ground_one(sc, constructor=constructor, critic=critic,
                               max_revisions=max_revisions)
                )
            ):
                result = ground_scenes(args)
                self.assertEqual(result["statistics"]["accepted"], 1)
                self.assertEqual((author.calls, critic.calls), (1, 1))
                resumed = ground_scenes(args)
                self.assertEqual(resumed["statistics"]["accepted"], 1)
                self.assertEqual((author.calls, critic.calls), (1, 1))
            rows = [
                json.loads(line) for line in
                (out / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual([], validate_clean_scene(rows[0]))
            self.assertEqual(rows[0]["public"], self.original["public"])


if __name__ == "__main__":
    unittest.main()
