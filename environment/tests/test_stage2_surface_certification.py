"""Stage 2 publishes only ProgramIR-certified Surface questions."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import Mock, patch

from environment.tests.test_review_reliability import ReviewReliabilityTests
from environment.question_generation.backend import (
    LLMSemanticCritic, OpenAICompatibleBackend, PassthroughCriticBackend,
)
from environment.question_generation.pipeline import _realize_one
from environment.question_generation.verification import (
    SCHEMA, certify_surface, validate_surface_certificate,
)
from environment.trajectory_simulation.eligibility import digest


class Stage2SurfaceCertificationTests(unittest.TestCase):
    def data(self):
        gold,world,rendered=ReviewReliabilityTests().rendered_fixture()
        world={**world,"_rendered":rendered,"_display_answer":rendered.answer}
        # Use the frozen public capability from the same test fixture.
        from environment.tests.test_gold_trajectory import fixture
        q,_,_=fixture()
        capability=q["capability_world"]
        payload={
            "task_id":rendered.task_id,
            "draft_question":rendered.question,
            "task_design":{"final_operation":"QUERY_NAME"},
            "resources":capability["resources"],
            "terminal_contract":{},
        }
        return payload,world,capability,rendered

    def test_verified_renderer_has_certificate_without_llm_call(self):
        payload,world,capability,rendered=self.data()
        critic=PassthroughCriticBackend()
        cert,review=certify_surface(
            payload,world,capability,rendered.question,critic,
        )
        self.assertEqual(cert["schema"],SCHEMA)
        self.assertEqual(review["method"],"verified_controlled_renderer")
        realized={
            "task_id":payload["task_id"],
            "question":rendered.question,
            "surface_certificate":cert,
            "authoring":{"surface_review":review},
        }
        result=validate_surface_certificate(realized,world,rendered.question)
        self.assertTrue(result["eligible"])
        self.assertTrue(result["reviewed"])
        self.assertEqual(result["method"],"stage2_verified_programir_certificate")

    def test_rewrite_needs_full_graph_review_not_old_critic(self):
        payload,world,capability,rendered=self.data()
        critic=PassthroughCriticBackend()
        other=rendered.question+" In an unrelated galaxy."
        cert,review=certify_surface(payload,world,capability,other,critic)
        self.assertIsNone(cert)
        self.assertEqual(review["review_failure_kind"],"graph_reviewer_unavailable")

    def test_graph_rewrite_certificate_binds_all_inputs_and_review_audit(self):
        payload,world,capability,rendered=self.data()
        critic=LLMSemanticCritic(OpenAICompatibleBackend("test://local","test"))
        paraphrase=rendered.question+" Return the matching names."
        review={
            "eligible":True,"reviewed":True,"decision":"eligible",
            "method":"llm_contract_alignment","issues":[],
            "alignment_version":1,
            "question_sha256":digest(paraphrase),
            "program_sha256":digest(world["program"]),
        }
        with patch(
            "environment.question_generation.alignment.check_alignment",
            side_effect=[review, {**review, "method": "llm_scoped_programir_screen"}],
        ) as check, patch(
            "environment.question_generation.verification.check_question",
            return_value={**review, "method":"llm_semantic_screen"},
        ):
            cert,actual=certify_surface(
                payload,world,capability,paraphrase,critic,
            )
        self.assertEqual(actual["eligible"],review["eligible"])
        review=actual
        self.assertEqual(check.call_count,2)
        self.assertEqual(check.call_args.args[3],paraphrase)
        realized={
            "task_id":payload["task_id"],"question":paraphrase,
            "surface_certificate":cert,
            "authoring":{"surface_review":review},
        }
        self.assertTrue(validate_surface_certificate(realized,world,rendered.question)["eligible"])
        for name,mutate in (
            ("question",lambda r,w: r.update(question="tampered")),
            ("program",lambda r,w: w["program"].update(nodes=[])),
            ("canonical",lambda r,w: None),
            ("review",lambda r,w: r["authoring"]["surface_review"].update(eligible=False)),
            ("certificate",lambda r,w: r["surface_certificate"].update(method="verified_controlled_renderer")),
        ):
            with self.subTest(name=name):
                r=copy.deepcopy(realized);w=copy.deepcopy(world)
                mutate(r,w)
                reference="changed" if name=="canonical" else rendered.question
                with self.assertRaises(ValueError):
                    validate_surface_certificate(r,w,reference)

    def test_stage2_falls_back_to_verified_renderer_on_failed_author(self):
        payload,world,capability,rendered=self.data()
        author=Mock()
        author.author.return_value={
            "question":"Do irrelevant galaxies orbit one another?",
        }
        ok,row,audit=_realize_one(
            payload,world,capability,author,PassthroughCriticBackend(),
            question_max_attempts=2,use_critic=True,
        )
        self.assertTrue(ok,str(row))
        self.assertEqual(row["question"],rendered.question)
        self.assertTrue(row["authoring"]["surface_fallback"])
        self.assertEqual(row["surface_certificate"]["question_sha256"],digest(rendered.question))
        self.assertEqual(len(audit),3)

    def test_alignment_approval_cannot_override_full_graph_rejection(self):
        payload,world,capability,rendered=self.data()
        question=rendered.question+" Return the matching names."
        critic=LLMSemanticCritic(OpenAICompatibleBackend("test://local","test"))
        alignment={
            "eligible":True,"reviewed":True,"decision":"eligible",
            "method":"llm_contract_alignment","issues":[],
            "question_sha256":digest(question),
            "program_sha256":digest(world["program"]),
        }
        with patch("environment.question_generation.alignment.check_alignment",
            side_effect=[alignment, {"eligible":False,"reviewed":True,"decision":"mismatch",
                          "issues":["wrong owner"],"review_failure_kind":"semantic_mismatch"},
            ]):
            cert,review=certify_surface(payload,world,capability,question,critic)
        self.assertIsNone(cert)
        self.assertFalse(review["eligible"])
        self.assertEqual(review["issues"],["wrong owner"])

    def test_legacy_contract13_remains_consumable(self):
        payload,world,capability,rendered=self.data()
        cert,review=certify_surface(payload,world,capability,rendered.question,PassthroughCriticBackend())
        cert["review_contract"]=13
        realized={"task_id":payload["task_id"],"question":rendered.question,
                  "surface_certificate":cert,"authoring":{"surface_review":review}}
        self.assertEqual(validate_surface_certificate(realized,world,rendered.question)["stage2_review_contract"],13)

    def test_two_alignment_views_cannot_override_complete_graph_rejection(self):
        payload,world,capability,rendered=self.data()
        critic=LLMSemanticCritic(OpenAICompatibleBackend("test://local","test"))
        review={"eligible":True,"reviewed":True,"decision":"eligible",
                "method":"llm_contract_alignment","issues":[]}
        with patch("environment.question_generation.alignment.check_alignment",
                   return_value=review), patch(
            "environment.question_generation.verification.check_question",
            return_value={"eligible":False,"reviewed":True,"decision":"mismatch",
                          "method":"llm_semantic_screen","issues":["collapsed anchor scope"]},
        ) as final:
            cert,result=certify_surface(payload,world,capability,rendered.question+" Return names.",critic)
        final.assert_called_once()
        self.assertIsNone(cert)
        self.assertFalse(result["eligible"])


if __name__=="__main__":
    unittest.main()
