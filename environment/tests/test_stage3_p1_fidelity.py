"""P1A/P1B: no future-gold reasoning and no silent report-fact rewrites."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.base_orchestration.action_author import (
    author_delegate_action, STAGE3_REASONING_AUTHOR_SYSTEM,
    STAGE3_REASONING_CRITIC_SYSTEM,
)
from environment.base_orchestration.reasoning_fidelity import (
    build_round_evidence_contract, check_reasoning_fidelity,
)
from environment.implement_subtask.public_surface import (
    clean_public_prose, reviewed_public_view,
)


def pending(source="index_1", field="record_key_1"):
    return {
        "target_instruction": (
            "Use the previously returned marker key to inspect the "
            "assigned index source, return only the matching "
            + field + " field and explain the local observation."
        ),
        "context": (
            "Previously returned result: Case-008 has marker MK-4059."
        ),
        "source_ids": [source],
        "output_fields": [field],
        "needs_upstream": True,
        "model": "model_1", "tools": [],
    }


def returned():
    return {
        "slot_one": {
            "finish_result": {
                "status": "done",
                "result": "The matching case has marker MK-4059.",
                "summary": "I confirmed the candidate using the returned source rows.",
            }
        }
    }


class ReasoningFidelityTests(unittest.TestCase):
    def test_current_handoff_cannot_be_final_exhibition_year(self):
        contract = build_round_evidence_contract(
            [pending()], returned(), has_later_delegation=True,
        )
        self.assertEqual(
            check_reasoning_fidelity(
                "This lookup will retrieve the exhibition year.",
                contract,
            ),
            ["reasoning_current_output_field_mismatch"],
        )
        self.assertIn(
            "reasoning_premature_final_step",
            check_reasoning_fidelity(
                "This lookup is the final step in the investigation.",
                contract,
            ),
        )
        self.assertEqual(
            check_reasoning_fidelity(
                "This lookup will return record_key_1, so that we can "
                "later retrieve the exhibition year from the archive.",
                contract,
            ),
            [],
        )

    def test_counts_sources_separately_from_subagent_tasks(self):
        contract = build_round_evidence_contract(
            [pending("visual", "candidate_id"),
             pending("context", "case_id")], returned(),
        )
        self.assertIn(
            "reasoning_dispatch_task_count_mismatch",
            check_reasoning_fidelity(
                "I will delegate three separate tasks in parallel.",
                contract,
            ),
        )
        self.assertEqual(
            check_reasoning_fidelity(
                "The two Sub-Agents will examine three evidence sources "
                "while running in parallel.",
                contract,
            ),
            [],
        )
        self.assertEqual(
            check_reasoning_fidelity(
                "I will delegate two independent tasks now.",
                contract,
            ),
            [],
        )

    def test_future_fields_never_enter_author_contract(self):
        contract = build_round_evidence_contract(
            [pending()], returned(), has_later_delegation=True,
        )
        public = contract["public"]
        self.assertEqual(public["current_round_task_count"], 1)
        self.assertEqual(
            public["current_round_tasks"][0]["output_fields"],
            ["record_key_1"],
        )
        self.assertNotIn("has_later_delegation", json.dumps(public))
        self.assertNotIn("gold_answer", json.dumps(public))
        self.assertNotIn("work_M", json.dumps(public))
        self.assertIn("MK-4059", json.dumps(public))
        self.assertEqual(
            check_reasoning_fidelity(
                "This step is the last step.",
                build_round_evidence_contract(
                    [pending()], returned(), has_later_delegation=False,
                ),
            ),
            [],
        )

    def test_author_repairs_only_reasoning_after_guard(self):
        assignments = [pending()]
        contract = build_round_evidence_contract(
            assignments, returned(), has_later_delegation=True,
        )

        class Author:
            def __init__(self):
                self.requests = []
            def complete_json(self, system, prompt):
                request = json.loads(prompt)
                self.requests.append(request)
                if len(self.requests) == 1:
                    return {"reasoning": (
                        "This lookup will retrieve the exhibition year "
                        "to finish the user's investigation."
                    )}
                return {"reasoning": (
                    "The previous report gave marker MK-4059. "
                    "The present lookup will return record_key_1 "
                    "for subsequent evidence retrieval."
                )}

        class Critic:
            def __init__(self):
                self.requests = []
            def complete_json(self, system, prompt):
                request = json.loads(prompt)
                self.requests.append(request)
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        author, critic = Author(), Critic()
        output, audit = author_delegate_action(
            assignments=assignments,
            prompt="==== QUESTION ==== Retrieve a confirmed year from sources."
                   " ==== SUBTASK HISTORY ==== MK-4059 was reported.",
            author=author, critic=critic, max_revisions=2,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            round_evidence_contract=contract,
            author_system=STAGE3_REASONING_AUTHOR_SYSTEM,
            critic_system=STAGE3_REASONING_CRITIC_SYSTEM,
            reasoning_guard=lambda value: check_reasoning_fidelity(
                value, contract,
            ),
        )
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(audit["attempts"], 2)
        self.assertEqual(len(critic.requests), 1)
        self.assertEqual(
            output["params"]["tasks"][0]["task_instruction"],
            assignments[0]["target_instruction"],
        )
        self.assertIn(
            "Only describe this round's output",
            " ".join(author.requests[1].get("repair_guidance", [])),
        )
        self.assertEqual(
            critic.requests[0]["current_round_evidence_contract"],
            contract["public"],
        )
        self.assertNotIn(
            "private_limits", json.dumps(author.requests),
        )


class PublicStage2SurfaceTests(unittest.TestCase):
    def test_copy_preserves_exact_result_status_and_factrefs(self):
        record = {
            "scene_task_id": "question-1",
            "macro_task_id": "work_M1",
            "answer": [{"case_id": "Case-008"}],
            "answer_with_provenance": [
                {"values": {"case_id": "Case-008"},
                 "source_fact_refs": ["context:F002"]}
            ],
            "report_authoring": {"status": "accepted"},
            "finish_result": {
                "status": "done",
                "result": "The exact result is Case-008 (MK-4059).",
                "summary": (
                    "This Macro retrieved Case-008 (MK-4059) from "
                    "context:F002. The macro outputted one record, "
                    "without asserting the exhibition year 1999."
                ),
            },
            "trace_summary": {
                "narrative": (
                    "This stage checked context:F002 and returned "
                    "the 1999 observation."
                ),
                "steps": [{"operation": "scan", "supporting_fact_refs": [
                    "context:F002"
                ]}],
            },
        }
        frozen = deepcopy(record)
        public = reviewed_public_view(record)
        self.assertEqual(record, frozen)
        self.assertEqual(
            public["finish_result"]["result"],
            record["finish_result"]["result"],
        )
        self.assertEqual(public["finish_result"]["status"], "done")
        self.assertEqual(
            public["answer_with_provenance"],
            record["answer_with_provenance"],
        )
        self.assertEqual(
            public["trace_summary"]["steps"],
            record["trace_summary"]["steps"],
        )
        self.assertNotIn("macro", public["finish_result"]["summary"].lower())
        self.assertNotIn("stage", public["trace_summary"]["narrative"].lower())
        self.assertIn("context:F002", public["finish_result"]["summary"])
        self.assertIn("Case-008", public["finish_result"]["summary"])
        self.assertIn("MK-4059", public["finish_result"]["summary"])
        self.assertIn("1999", public["finish_result"]["summary"])
        self.assertEqual(
            reviewed_public_view(public), public,
        )

    def test_no_unsupported_local_report_is_silently_sanitized(self):
        record = {
            "report_authoring": {"status": "failed"},
            "finish_result": {"status": "done", "result": "Case-999",
                              "summary": "The macro fabricated Case-999"},
        }
        with self.assertRaisesRegex(ValueError, "requires_reviewed"):
            reviewed_public_view(record)
        self.assertEqual(
            clean_public_prose(
                "The source observations disagree: Case-002 vs Case-003."
            ),
            "The source observations disagree: Case-002 vs Case-003.",
        )


if __name__ == "__main__":
    unittest.main()
