"""Regress executable question predicates and stage-local claim boundaries."""
from __future__ import annotations

import copy
import json
import unittest

from envir.generate_question.spec import make_spec, spec_issues
from envir.generate_question.question import reference_question, guard_question
from envir.generate_question.quality import compile_and_validate, validate_replay
from envir.generate_question.semantic_evidence import compile_semantic_contract
from envir.generate_question.topic_domain import infer_topic_domain
from envir.implement_subtask.realize import realize_scene
from envir.implement_subtask.reporting import _teacher_contract, _guard, author_critic_report
from envir.implement_subtask.stage_local_claims import (
    compile_claim_contract, stage_local_claim_issues, normalize_report_language,
)


def seeded(i, topic="Premier League", hint="sports", question=""):
    return make_spec(
        seed_record={
            "record_id": f"semantic-stage3-{i}",
            "topic": topic, "domain_hint": hint, "question": question,
        },
        ordinal=i, run_seed=31, difficulty="hard" if i % 2 else "medium",
        macro_first_v2_share=100,
    )


def pattern_spec(pattern, *, video=None):
    for i in range(300):
        spec = seeded(i)
        plan = spec["macro_plan"]
        if plan["pattern"] != pattern:
            continue
        if video is not None and (
            plan["clues"]["visual_modality"] == "video"
        ) != video:
            continue
        return spec
    raise AssertionError("missing sampled pattern")


class QuestionPredicateFidelityTests(unittest.TestCase):
    def test_new_specs_replay_and_reference_question_for_many_seeds(self):
        for i in range(90):
            spec = seeded(i)
            self.assertEqual(spec_issues(spec), [])
            q = reference_question(spec)
            self.assertEqual(guard_question(q, spec), [], (i, q))
            scene = compile_and_validate(spec, q)
            self.assertEqual(validate_replay(scene)["status"], "valid")

    def test_frame_lower_bound_is_not_action_duration_or_first_occurrence(self):
        spec = pattern_spec("serial_identity", video=True)
        reference = reference_question(spec)
        threshold = spec["macro_plan"]["clues"]["time_threshold_ms"] // 1000
        correct = f"The video shows the specified action at or after {threshold} seconds."
        contract = compile_semantic_contract(spec)
        from envir.generate_question.question_fidelity import question_fidelity_issues
        self.assertEqual(question_fidelity_issues(correct, contract), [])
        for claim, expected in (
            (f"The video action lasting at least {threshold} seconds.",
             "semantic_frame_timestamp_as_duration"),
            (f"The video action starting at {threshold} seconds.",
             "semantic_frame_lower_bound_as_start"),
            (f"The frame is visible at {threshold} seconds.",
             "semantic_frame_lower_bound_not_expressed"),
        ):
            with self.subTest(claim=claim):
                bad = reference + " " + claim
                self.assertIn(expected, guard_question(bad, spec))
        self.assertEqual(guard_question(reference, spec), [])

    def test_video_audio_comparison_is_not_strict(self):
        spec = pattern_spec("multimodal_event")
        original = reference_question(spec)
        self.assertIn("no later than", original)
        self.assertEqual(guard_question(original, spec), [])
        changed = original.replace(
            "no later than", "strictly before"
        )
        self.assertIn(
            "semantic_non_strict_order_claimed_strict",
            guard_question(changed, spec),
        )

    def test_two_requested_answers_with_video_identity_are_rejected(self):
        spec = pattern_spec("parallel_cues", video=True)
        good = reference_question(spec)
        bad = (
            f"Which video clip in {spec['scenario']['topic']} shows "
            f"the {spec['macro_plan']['clues']['scene_action']} action "
            f"at or after {spec['macro_plan']['clues']['time_threshold_ms']//1000} "
            f"seconds, and what is the confirmed "
            f"{spec['macro_plan']['goal']['public_label']}?"
        )
        self.assertIn("semantic_additional_output_requested",
                      guard_question(bad, spec))
        self.assertEqual(guard_question(good, spec), [])

    def test_previous_snapshot_without_marker_replays(self):
        spec = pattern_spec("serial_identity", video=True)
        spec.pop("question_fidelity_version")
        spec.pop("topic_domain_version")
        spec["source_anchor"].pop("topic_domain_cue", None)
        scene = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(scene)["status"], "valid")
        self.assertNotIn("question_fidelity_version",
                         scene["semantic_evidence_contract"])


class TopicDomainTests(unittest.TestCase):
    def test_source_question_only_supplies_entity_category_not_original_goal(self):
        examples = [
            ("Swansea City", "geography",
             "Which football club competed in the same league?", "sports"),
            ("Broad City", "geography",
             "Which episode of the television series featured the actor?", "film"),
            ("Sean Chiplock", "education",
             "Which voice actor performed the role?", "film"),
            ("PNC Park", "culture",
             "Which stadium hosted the football match?", "sports"),
        ]
        for i, (topic, hint, question, expected) in enumerate(examples):
            seed = {"topic": topic, "domain_hint": hint, "question": question}
            with self.subTest(topic=topic):
                self.assertEqual(infer_topic_domain(seed), expected)
                spec = seeded(i + 100, topic, hint, question)
                self.assertEqual(spec["macro_plan"]["domain"], expected)
                self.assertNotIn(question, json.dumps(spec["scenario"]))
                self.assertEqual(guard_question(reference_question(spec), spec), [])
        self.assertIsNone(infer_topic_domain({
            "question": "Which year was this recorded?"
        }))


class StageLocalClaimTests(unittest.TestCase):
    def test_scan_only_report_cannot_claim_independent_disagreement(self):
        spec = pattern_spec("disputed_reports")
        scene = compile_and_validate(spec, reference_question(spec))
        rows = realize_scene(scene)["macro_tasks"]
        a = next(r for r in rows if r["macro_task_id"] == "work_M1")
        dispute = next(r for r in rows if r["macro_task_id"] == "work_M3")
        contract = _teacher_contract(
            a, question=scene["question"], is_terminal=False
        )
        c = contract["stage_local_claim_contract"]
        self.assertFalse(c["reported_year_disagreement_established"])
        error = "This stage captured eight candidate cases where independent reports disagree on the year."
        self.assertIn("claims_unverified_independent_report_disagreement",
                      stage_local_claim_issues(error, c))
        candidate = {
            "result": a["answer"],
            "summary": error,
            "status": "done",
        }
        self.assertIn(
            "summary_claims_unverified_independent_report_disagreement",
            _guard(candidate, {**contract, "all_evidence_refs": [
                ref for step in a["trace_summary"]["steps"]
                for ref in step["supporting_fact_refs"]
            ]}),
        )
        self.assertEqual(stage_local_claim_issues(
            "Collected report A candidates for later comparison of reported years.", c
        ), [])
        verified = compile_claim_contract(dispute)
        self.assertTrue(verified["reported_year_disagreement_established"])
        self.assertEqual(stage_local_claim_issues(error, verified), [])

    def test_corroborated_dispute_requires_both_independent_branches(self):
        from envir.generate_question.information_necessity import (
            CORROBORATED_DISPUTE_POLICY, audit_candidate_information,
        )
        spec = pattern_spec("corroborated_dispute")
        scene = compile_and_validate(spec, reference_question(spec))
        report = audit_candidate_information(scene)
        self.assertEqual(report["policy"], CORROBORATED_DISPUTE_POLICY)
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["left_candidate_count"], 2)
        self.assertGreaterEqual(report["right_candidate_count"], 2)
        self.assertEqual(report["intersection_count"], 1)
        self.assertTrue(report["competing_goals_distinct"])
        self.assertEqual(scene["difficulty_audit"]["macro_first_audit"][
            "candidate_information_necessity"], report)
        q = scene["question"]
        self.assertIn(spec["macro_plan"]["clues"]["case_tag"], q)
        self.assertIn("disagree", q)
        self.assertEqual(validate_replay(scene)["status"], "valid")
        oracle = realize_scene(scene)
        compared = next(row for row in oracle["macro_tasks"]
                        if row["macro_task_id"] == "work_M3")
        context = next(row for row in oracle["macro_tasks"]
                       if row["macro_task_id"] == "work_M4")
        fused = next(row for row in oracle["macro_tasks"]
                     if row["macro_task_id"] == "work_M5")
        self.assertTrue(compile_claim_contract(compared)[
            "reported_year_disagreement_established"])
        self.assertFalse(compile_claim_contract(context)[
            "reported_year_disagreement_established"])
        inherited = compile_claim_contract(fused)
        self.assertTrue(inherited["reported_year_disagreement_established"])
        self.assertFalse(inherited["reported_year_disagreement_checked_locally"])
        self.assertIn("independent_report_year_disagreement",
                      inherited["verified_inherited_claims"])
        self.assertEqual(stage_local_claim_issues(
            "Combined records for which the independent report years disagree "
            "with the separately matching log entries.", inherited), [])

    def test_jargon_surface_rewrite_still_runs_critic(self):
        source = "The oracle output lists the cases supported by the evidence."
        transformed, edits = normalize_report_language(source)
        self.assertIn("verified source result", transformed)
        self.assertTrue(edits)
        self.assertNotIn("oracle", transformed.lower())

        spec = pattern_spec("serial_identity")
        scene = compile_and_validate(spec, reference_question(spec))
        record = realize_scene(scene)["macro_tasks"][0]

        class Author:
            def complete_json(self, system, request):
                expected = json.loads(request)["frozen_task_and_evidence"]["exact_oracle_answer"]
                return {
                    "result": expected, "status": "done",
                    "summary": (
                        "The oracle answer includes the requested candidate, "
                        "and the assigned source was checked against its recorded observations."
                    ),
                }

        class Critic:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, request):
                self.calls += 1
                return {
                    "verdict": "pass", "grounded": True,
                    "answer_scope_preserved": True,
                    "evidence_scope_respected": True, "natural": True,
                    "reason": "The source description is accurate.", "repair_hint": "",
                }

        critic = Critic()
        accepted, outcome = author_critic_report(
            record, question=scene["question"], author=Author(),
            critic=critic, max_revisions=0,
        )
        self.assertEqual(outcome["status"], "accepted")
        self.assertEqual(critic.calls, 1)
        self.assertTrue(outcome["surface_rewrites"])
        self.assertNotIn("oracle", accepted["finish_result"]["summary"].lower())
        self.assertEqual(accepted["answer"], record["answer"])


if __name__ == "__main__":
    unittest.main()
