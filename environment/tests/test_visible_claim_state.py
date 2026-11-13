"""Actual Main-visible (not World-synthetic) Claim-state transitions."""
from __future__ import annotations

import json
import unittest

from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.visible_evidence import (
    witnessed_source_summary, visible_year_claims,
)
from environment.untrusted_orchestration.visible_claim_state import (
    audit_visible_claim_states,
)


def year_state(round_snapshot: dict, subject: str) -> dict:
    """Find a specific reported-year Claim state in the unified state map."""
    return next(item for item in round_snapshot["claim_states"].values()
                if item["subject"] == subject
                and item["attribute"] == "reported_event_year")


def serial_dispute():
    for i in range(180):
        spec = make_spec(
            seed_record={
                "record_id": f"v14-public-state-{i:04d}",
                "topic": "Archive incident records", "domain_hint": "general",
            },
            ordinal=i, run_seed=17, difficulty="medium",
        )
        plan = spec["macro_plan"]
        if (plan["pattern"] == "disputed_reports"
                and plan["claim_verification_mode"] == "keyed_followup"):
            return compile_and_validate(spec, reference_question(spec))
    raise AssertionError("no_serial_report_pattern")


def public_result(record, *, status="done", summary=None, result=None):
    text = witnessed_source_summary(record)
    return {
        "task_instruction": "Review the assigned source",
        "finish_result": {
            "status": status,
            "result": text if result is None else result,
            "summary": text if summary is None else summary,
        } if status != "missing" else None,
        "trace_summary": "Read only the assigned source records.",
        "steps_taken": 1,
    }


def world_trajectory(scene):
    teacher = realize_scene(scene)
    by_id = {row["macro_task_id"]: row for row in teacher["macro_tasks"]}
    bindings, trajectory = {}, []
    for i, wave in enumerate(teacher["reference_waves"], 1):
        bindings[i] = list(wave)
        trajectory.append({
            "attempt": i, "action": "delegate_task",
            "result": {
                "subtask_results": [
                    public_result(by_id[tid]) for tid in wave
                ],
            },
        })
    return teacher, trajectory, bindings


class MainVisibleStateTests(unittest.TestCase):
    def test_actual_text_creates_genuine_cross_round_disagreement(self):
        scene = serial_dispute()
        teacher, trajectory, bindings = world_trajectory(scene)
        state = audit_visible_claim_states(
            scene, trajectory, task_ids_by_round=bindings,
        )
        self.assertTrue(state["applicable"])
        first, second = state["rounds"][:2]
        relevant = {
            t["subject"] for t in scene["evidence_ledger"][
                "reference_wave_trust"
            ]["transitions"] if t["answer_supporting_claim_pair"]
        }
        self.assertTrue(relevant)
        for subject in relevant:
            self.assertEqual(year_state(first, subject)["status"], "unverified")
            self.assertEqual(year_state(second, subject)["status"], "disputed")
        self.assertGreater(
            state["summary"]["late_visible_disputes"], 0,
        )
        self.assertEqual(first["main_can_act_from_round"], 2)
        self.assertEqual(second["main_can_act_from_round"], 3)

    def test_unreported_source_has_no_synthetic_world_claim(self):
        scene = serial_dispute()
        _, trajectory, bindings = world_trajectory(scene)
        # Second report exists in the World but is not returned to Main.
        second = trajectory[1]["result"]["subtask_results"][0]
        second["finish_result"] = None
        state = audit_visible_claim_states(
            scene, trajectory, task_ids_by_round=bindings,
        )
        self.assertEqual(state["summary"].get("late_visible_disputes", 0), 0)
        first = state["rounds"][0]["claim_states"]
        second_state = state["rounds"][1]["claim_states"]
        self.assertEqual(first, second_state)

    def test_wrong_delivered_value_is_not_silently_replaced_by_gold(self):
        scene = serial_dispute()
        _, trajectory, bindings = world_trajectory(scene)
        target = next(
            t["subject"] for t in scene["evidence_ledger"][
                "reference_wave_trust"
            ]["transitions"] if t["answer_supporting_claim_pair"]
        )
        first = trajectory[0]["result"]["subtask_results"][0]
        first_claim = next(x for x in visible_year_claims(
            first, "report_a",
        ) if x["subject"] == target)
        second = trajectory[1]["result"]["subtask_results"][0]
        summary = second["finish_result"]["summary"]
        import re
        summary = re.sub(
            rf"\b{target}=\d{{4}}\b",
            f"{target}={first_claim['value']}", summary,
        )
        second["finish_result"]["summary"] = summary
        second["finish_result"]["result"] = summary
        state = audit_visible_claim_states(
            scene, trajectory, task_ids_by_round=bindings,
        )
        self.assertEqual(
            year_state(state["rounds"][1], target)["status"],
            "corroborated",
        )
        # This is a surface-only trust annotation, not true ground truth.
        self.assertNotEqual(
            scene["evidence_ledger"]["reference_wave_trust"][
                "transitions"
            ][-1]["after_status"], "fabricated_gold_claim",
        )

    def test_retry_replaces_old_source_report_and_retracts_trust(self):
        scene = serial_dispute()
        _, trajectory, bindings = world_trajectory(scene)
        # A later retry of report B fails: no earlier B claims are retained.
        trajectory.append({
            "attempt": len(trajectory) + 1,
            "action": "delegate_task",
            "result": {"subtask_results": [{
                "finish_result": None,
                "task_instruction": "Retry the second independent report",
            }]},
        })
        retry_round = trajectory[-1]["attempt"]
        bindings[retry_round] = ["work_M2"]
        # Keep the duplicate round out of the standard reference schedule.
        state = audit_visible_claim_states(
            scene, trajectory, task_ids_by_round=bindings,
        )
        target = next(
            t["subject"] for t in scene["evidence_ledger"][
                "reference_wave_trust"
            ]["transitions"] if t["answer_supporting_claim_pair"]
        )
        self.assertEqual(
            year_state(state["rounds"][-1], target)["status"], "unverified",
        )
        self.assertGreater(state["summary"].get("source_report_replacements", 0), 0)

    def test_intra_source_conflict_not_counted_as_independent_support(self):
        scene = serial_dispute()
        _, trajectory, bindings = world_trajectory(scene)
        second = trajectory[1]["result"]["subtask_results"][0]
        examples = visible_year_claims(second, "report_b")
        case = examples[0]["subject"]
        year = examples[0]["value"]
        mismatched = json.dumps([{"values": {
            "case_id": case, "reported_year_b": year + 1,
        }}])
        second["finish_result"]["result"] = mismatched
        state = audit_visible_claim_states(
            scene, trajectory, task_ids_by_round=bindings,
        )
        self.assertEqual(
            year_state(state["rounds"][1], case)["status"],
            "intra_report_conflict",
        )
        self.assertFalse(
            year_state(state["rounds"][1], case)["sufficient_to_accept"],
        )

    def test_reviewed_stage2_report_includes_code_owned_witness(self):
        from environment.implement_subtask.reporting import author_critic_report

        class Author:
            def complete_json(self, system_prompt, user_prompt):
                contract = json.loads(user_prompt)["frozen_task_and_evidence"]
                return {
                    "result": contract["exact_oracle_answer"],
                    "status": "done",
                    "summary": "Read the assigned report and returned the source rows.",
                }

        class Critic:
            def complete_json(self, system_prompt, user_prompt):
                return {
                    "verdict": "pass",
                    "grounded": True,
                    "answer_scope_preserved": True,
                    "evidence_scope_respected": True,
                    "natural": True,
                }

        scene = serial_dispute()
        teacher = realize_scene(scene)
        record = next(x for x in teacher["macro_tasks"]
                      if x["macro_task_id"] == "work_M1")
        reviewed, status = author_critic_report(
            record, question=scene["question"],
            author=Author(), critic=Critic(), max_revisions=0,
        )
        self.assertEqual(status["status"], "accepted")
        digest = witnessed_source_summary(record)
        self.assertIn(digest, reviewed["finish_result"]["summary"])
        self.assertEqual(
            {x["subject"] for x in visible_year_claims(reviewed, "report_a")},
            {x["case_id"] for x in record["answer"]},
        )

    def test_real_stage4_branch_keeps_visible_state_in_private_audit(self):
        from environment.base_orchestration.trajectory_builder import build_episode
        from environment.untrusted_orchestration.trajectory_builder import build_branch
        from environment.untrusted_orchestration.faults import Fault

        scene = serial_dispute()
        records = realize_scene(scene)["macro_tasks"]
        reports = [{
            **record,
            "finish_result": {
                "status": "done",
                "result": json.dumps(record["answer"]),
                "summary": (
                    "Read and returned the assigned source. " +
                    witnessed_source_summary(record)
                ),
            },
            "report_authoring": {"status": "accepted"},
        } for record in records]
        base, _, _, status = build_episode(scene, reports)
        self.assertEqual(status["status"], "accepted")
        ep, turns, audit = build_branch(
            scene, reports, base, branch_round=2,
            faults=[Fault("work_M1", "missing")],
            max_rounds=16,
        )
        self.assertTrue(audit["visible_claim_state"]["applicable"])
        self.assertTrue(audit["visible_claim_state"]["rounds"])
        self.assertNotIn("visible_claim_state", json.dumps(ep))
        self.assertNotIn("visible_claim_state", json.dumps(turns))
        self.assertNotIn("evidence_ledger", json.dumps(turns))

    def test_witness_text_only_contains_returned_source_rows(self):
        scene = serial_dispute()
        teacher = realize_scene(scene)
        record = next(x for x in teacher["macro_tasks"]
                      if x["macro_task_id"] == "work_M2")
        digest = witnessed_source_summary(record)
        claims = visible_year_claims(
            {"finish_result": {
                "status": "done", "result": "Acknowledged.",
                "summary": digest,
            }}, "report_b",
        )
        self.assertEqual(
            {x["subject"] for x in claims},
            {x["case_id"] for x in record["answer"]},
        )
        self.assertTrue(all(
            x["surface"] == "summary" for x in claims
        ))
        self.assertEqual(
            visible_year_claims(
                {"finish_result": {"status": "missing",
                                   "result": digest, "summary": digest}},
                "report_b",
            ), [],
        )


if __name__ == "__main__":
    unittest.main()
