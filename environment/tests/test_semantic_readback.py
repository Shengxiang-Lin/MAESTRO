"""Adversarial tests for blind free-text semantic readback.

Readback sees only a candidate's English text, never the expected query.
The deterministic verifier must reject hallucinated witnesses, omitted
filters, swapped RELATE direction, changed owner/join and string-code drift.
"""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import Mock, patch

from environment.question_generation.free_surface import (
    assess_free_style, free_rewrite_prompt, public_rewrite_guard,
)
from environment.question_generation.scoped_surface import ScopedPlan
from environment.question_generation.semantic_readback import (
    READBACK_VERSION, ReadbackResult, normalize_query,
    read_free_question, validate_readback_response,
)
from environment.question_generation.verification import (
    certify_free_surface, validate_surface_certificate,
)
from environment.question_generation.pipeline import _realize_one
from environment.question_generation.backend import (
    LLMQuestionAuthor, LLMSemanticCritic, OpenAICompatibleBackend,
)
from environment.tests.test_stage2_surface_certification import (
    Stage2SurfaceCertificationTests,
)
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.eligibility import digest


QUESTION = (
    "Which people are recorded as spouses of someone whose death date "
    "is August 1, 1997 in the relevant records?"
)


def example_parse():
    return {
        "uncertain": False, "reason": "",
        "query": {
            "definition": "QUERY_NAME",
            "conditions": [], "evidence": "Which people",
            "inputs": [{
                "definition": "RELATE", "relation": "spouse",
                "direction": "backward", "conditions": [],
                "evidence": "people are recorded as spouses of someone",
                "inputs": [{
                    "definition": "FIND_ALL",
                    "range": "bounded_public_task_evidence",
                    "conditions": [{
                        "attribute": "death date",
                        "comparison": "equal to",
                        "value": {"type": "date", "value": "1997-08-01"},
                        "evidence": "death date is August 1, 1997",
                    }],
                    "evidence": "someone whose death date is August 1, 1997",
                    "inputs": [],
                }],
            }],
        },
    }


class BlindReadbackTests(unittest.TestCase):
    def reference(self):
        return ScopedPlan([], [], copy.deepcopy(example_parse()["query"]))

    def test_two_question_only_parses_can_recover_typed_semantics(self):
        example = example_parse()
        backend = Mock()
        backend.complete_json.side_effect = [
            copy.deepcopy(example), copy.deepcopy(example),
        ]
        result = read_free_question(QUESTION, self.reference(), backend)
        self.assertTrue(result.eligible, result.reason)
        self.assertEqual(result.calls, 2)
        self.assertEqual(len(result.raw_responses), 2)
        for call in backend.complete_json.call_args_list:
            # The model does NOT receive the expected tree or any private
            # rendering context, even when the caller has access to it.
            user = json.loads(call.args[1])
            self.assertEqual(user, {"question": QUESTION})
            self.assertNotIn("death date", str(call.args[1]).split("question", 1)[0])
            self.assertNotIn("ProgramIR", str(call.args[1]))
        self.assertEqual(
            result.parsed_query,
            normalize_query(self.reference().query),
        )

    def test_readback_rejects_wrong_date_even_when_all_other_nodes_match(self):
        wrong = example_parse()
        wrong["query"]["inputs"][0]["inputs"][0]["conditions"][0]["value"]["value"] = "1997-08-02"
        with self.assertRaisesRegex(ValueError, "not_in_quote|mismatch"):
            validate_readback_response(wrong, QUESTION, self.reference().query)

    def test_readback_rejects_swapped_relation_direction(self):
        wrong = example_parse()
        wrong["query"]["inputs"][0]["direction"] = "forward"
        with self.assertRaisesRegex(ValueError, "typed_semantics_mismatch"):
            validate_readback_response(wrong, QUESTION, self.reference().query)

    def test_readback_rejects_condition_moved_to_other_owner(self):
        wrong = example_parse()
        filter_ = wrong["query"]["inputs"][0]["inputs"][0]["conditions"].pop()
        wrong["query"]["inputs"][0]["conditions"].append(filter_)
        with self.assertRaisesRegex(ValueError, "typed_semantics_mismatch"):
            validate_readback_response(wrong, QUESTION, self.reference().query)

    def test_readback_rejects_unsupported_claim_in_quote(self):
        wrong = example_parse()
        wrong["query"]["inputs"][0]["relation"] = "winner"
        wrong["query"]["inputs"][0]["evidence"] = "this relation is not in the text"
        with self.assertRaisesRegex(ValueError, "evidence_not_in_question"):
            validate_readback_response(wrong, QUESTION, self.reference().query)

    def test_readback_rejects_omitted_filter_or_added_branch(self):
        wrong = example_parse()
        wrong["query"]["inputs"][0]["inputs"][0]["conditions"] = []
        with self.assertRaisesRegex(ValueError, "typed_semantics_mismatch"):
            validate_readback_response(wrong, QUESTION, self.reference().query)
        for join in ("INTERSECT", "UNION"):
            merged = example_parse()
            parent = merged["query"]["inputs"][0]
            parent["definition"] = join
            parent["inputs"].append(copy.deepcopy(parent["inputs"][0]))
            with self.subTest(join=join), self.assertRaisesRegex(
                ValueError, "typed_semantics_mismatch"
            ):
                validate_readback_response(merged, QUESTION, self.reference().query)

    def test_string_code_and_numeric_quantity_are_distinct(self):
        from environment.question_generation.semantic_readback import normalize_query
        query = {"definition": "COUNT", "conditions": [], "inputs": [{
            "definition": "FIND_ALL",
            "range": "bounded_public_task_evidence",
            "conditions": [{
                "attribute": "telephone country code", "comparison": "equal to",
                "value": {"type": "string", "value": "+650"},
            }],
            "inputs": [],
        }]}
        modified = copy.deepcopy(query)
        modified["inputs"][0]["conditions"][0]["value"] = {
            "type": "number", "value": "650", "unit": "",
        }
        self.assertNotEqual(normalize_query(query), normalize_query(modified))
        modified["inputs"][0]["conditions"][0]["value"] = {
            "type": "string", "value": "650",
        }
        self.assertNotEqual(normalize_query(query), normalize_query(modified))

    def test_ambiguous_parse_is_not_eligible(self):
        backend = Mock()
        backend.complete_json.return_value = {
            "uncertain": True, "reason": "ambiguous pronoun",
            "query": example_parse()["query"],
        }
        result = read_free_question(QUESTION, self.reference(), backend)
        self.assertFalse(result.eligible)
        self.assertEqual(result.calls, 1)

    def test_typed_public_guard_allows_prose_date_but_not_modified_code(self):
        scoped = (
            "Which people died on 1997-08-01 and have telephone "
            "country code +650?"
        )
        same = (
            "Which people died on August 1, 1997 and have telephone "
            "country code +650?"
        )
        self.assertEqual(public_rewrite_guard(scoped, same), [])
        self.assertIn(
            "free_rewrite_invents_or_changes_public_literal",
            public_rewrite_guard(scoped, same.replace("+650", "650")),
        )
        self.assertIn(
            "free_rewrite_invents_or_changes_public_literal",
            public_rewrite_guard(scoped, same.replace("August 1, 1997",
                                                       "August 2, 1997")),
        )
        self.assertIn(
            "free_rewrite_adds_exclusion_or_uniqueness",
            public_rewrite_guard(scoped, same.replace("people", "other people")),
        )

    def test_readability_gate_does_not_reward_cosmetic_edit(self):
        scoped = (
            "Let the first group consist of people in the relevant records. "
            "Which people in the first group have a spouse?"
        )
        trivial = scoped.replace("people", "persons", 1)
        better = "Which people in the relevant records have a spouse?"
        self.assertFalse(assess_free_style(scoped, trivial).improved)
        self.assertTrue(assess_free_style(scoped, better).improved)
        self.assertEqual(
            json.loads(free_rewrite_prompt(scoped)),
            {"scoped_question": scoped},
        )


class CertificateRoundtripTests(unittest.TestCase):
    def fixture(self):
        payload, world, capability, rendered = Stage2SurfaceCertificationTests().data()
        result = ReadbackResult(
            eligible=True, reason="independent_typed_readback_matched",
            parsed_query=normalize_query(example_parse()["query"]),
            calls=2, audits=({"pass": 1, "status": "verified"},
                             {"pass": 2, "status": "verified"}),
            raw_responses=(copy.deepcopy(example_parse()),
                           copy.deepcopy(example_parse())),
        )
        plan = ScopedPlan([], [], copy.deepcopy(example_parse()["query"]))
        return payload, world, capability, rendered, plan, result

    def test_stage3_revalidates_stored_readback_evidence_without_llm(self):
        payload, world, capability, rendered, plan, result = self.fixture()
        cert, review = certify_free_surface(
            payload["task_id"], world, rendered.question,
            QUESTION, plan, result,
        )
        realized = {
            **payload, "question": QUESTION,
            "surface_certificate": cert,
            "authoring": {"surface_review": review},
            "capability_world": capability,
        }
        with patch(
            "environment.question_generation.verification.controlled_question_reference",
            return_value={"question": rendered.question},
        ), patch(
            "environment.question_generation.scoped_surface.build_scoped_plan",
            return_value=plan,
        ):
            verified = validate_surface_certificate(
                realized, world, rendered.question,
                gold={"task_id": payload["task_id"]}, rendered=rendered,
            )
            self.assertTrue(verified["eligible"])
            # Editing the sentence and recomputing all clear-text hashes is
            # still insufficient: the stored quotation witnesses no longer
            # appear in the actual user-facing question.
            changed = copy.deepcopy(realized)
            changed["question"] = QUESTION.replace("August 1", "August 2")
            review2 = changed["authoring"]["surface_review"]
            review2["question_sha256"] = digest(changed["question"])
            changed["surface_certificate"]["question_sha256"] = digest(changed["question"])
            changed["surface_certificate"]["review_sha256"] = digest(review2)
            with self.assertRaisesRegex(ValueError, "free_readback_text_witness_mismatch"):
                validate_surface_certificate(
                    changed, world, rendered.question,
                    gold={"task_id": payload["task_id"]}, rendered=rendered,
                )

    def test_free_mode_can_publish_only_after_positive_blind_verification(self):
        payload, world, capability, rendered, _, result = self.fixture()
        author = LLMQuestionAuthor(
            OpenAICompatibleBackend("http://test.local/v1", "stub"),
        )
        critic = LLMSemanticCritic(
            OpenAICompatibleBackend("http://test.local/v1", "stub"),
        )
        def good_rewrite(_, scoped_question, **kwargs):
            return scoped_question.replace(
                "Let the first group consist of", "The first group includes",
            )
        with patch(
            "environment.question_generation.scoped_surface.choose_scoped",
            side_effect=lambda _, groups: [0] * len(groups),
        ), patch(
            "environment.question_generation.free_surface.call_free_rewriter",
            side_effect=good_rewrite,
        ), patch(
            "environment.question_generation.free_surface.assess_free_style",
            return_value=Mock(improved=True, to_dict=lambda: {"improved": True}),
        ), patch(
            "environment.question_generation.semantic_readback.read_free_question",
            return_value=result,
        ), patch(
            "environment.question_generation.verification.certify_free_surface",
            return_value=(
                {"method": "verified_blind_semantic_readback_v1"},
                {"method": "verified_blind_semantic_readback_v1"},
            ),
        ) as certify:
            ok, row, audits = _realize_one(
                payload, world, capability, author, critic,
                question_max_attempts=1, use_critic=True,
                surface_mode="free",
            )
        self.assertTrue(ok)
        self.assertTrue(row["authoring"]["free_rewrite_verified"])
        self.assertEqual(row["authoring"]["strategy"],
                         "independent_free_semantic_readback")
        self.assertEqual(row["surface_certificate"]["method"],
                         "verified_blind_semantic_readback_v1")
        self.assertEqual(certify.call_count, 1)
        self.assertEqual(sum(a.get("blind_readback_calls", 0) for a in audits), 2)

    def test_explicit_free_mode_falls_back_to_scoped_when_readback_fails(self):
        from environment.question_generation.semantic_readback import ReadbackResult
        payload, world, capability, rendered, plan, _ = self.fixture()
        author = LLMQuestionAuthor(
            OpenAICompatibleBackend("http://test.local/v1", "stub"),
        )
        critic = LLMSemanticCritic(
            OpenAICompatibleBackend("http://test.local/v1", "stub"),
        )
        with patch(
            "environment.question_generation.scoped_surface.choose_scoped",
            side_effect=lambda _, groups: [0] * len(groups),
        ), patch(
            "environment.question_generation.free_surface.call_free_rewriter",
            side_effect=lambda _author, scoped_question, **kwargs: scoped_question.replace(
                "Let the first group consist of", "The first group includes",
            ),
        ), patch(
            "environment.question_generation.free_surface.assess_free_style",
            return_value=Mock(improved=True, to_dict=lambda: {"improved": True}),
        ), patch(
            "environment.question_generation.semantic_readback.read_free_question",
            return_value=ReadbackResult(False, "uncertain", None, 1, ()),
        ):
            ok, row, audits = _realize_one(
                payload, world, capability, author, critic,
                question_max_attempts=1, use_critic=True,
                surface_mode="free",
            )
        self.assertTrue(ok)
        self.assertFalse(row["authoring"]["free_rewrite_verified"])
        self.assertEqual(
            row["surface_certificate"]["method"], "verified_scoped_grammar_v2",
        )
        self.assertEqual(sum(a.get("blind_readback_calls", 0) for a in audits), 1)


if __name__ == "__main__":
    unittest.main()
