"""Precomputed correct Subtask Bank -> independent, causal O1 Main decisions."""
from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.tests.test_gold_trajectory import fixture
from environment.trajectory_simulation.backend import (
    PassthroughPlanner, PassthroughExecutor,
)
from environment.trajectory_simulation.eligibility import digest
from environment.trajectory_simulation.exporter import validate_public_episode
from environment.trajectory_simulation.natural_language import realize_main
from environment.trajectory_simulation.rounds import (
    make_round_record, synthesize_rounds_jsonl, o1_input, native_tool_spec,
)
from environment.trajectory_simulation.subtask_bank import build_subtask_bank


def reviewed(gold, record, role, *, review=True):
    return {
        "eligible": True, "reviewed": True,
        "question_sha256": digest(gold["question"]),
        "program_sha256": digest(record["program"]),
        "method": "test_exact_program_review", "issues": [],
    }


class RoundCentricTests(unittest.TestCase):
    def bank(self, *, dependent=False, planner=None, executor=None):
        q, record, rendered = fixture(dependent=dependent)
        with patch(
            "environment.trajectory_simulation.subtask_bank.check_question", reviewed
        ):
            bank = build_subtask_bank(
                q, record, rendered,
                planner=planner or PassthroughPlanner(),
                executor=executor or PassthroughExecutor(),
                schedule="baseline",
            )
        return q,record,rendered,bank

    def test_real_o1_input_uses_native_prompt_and_native_tool_schema(self):
        state = {
            "question": "Who wrote the research report?",
            "subtask_history": "No subtasks completed yet.",
            "available_tools": ["GoogleSearchAction", "ExecuteCodeAction"],
            "evidence_store": {},
        }
        text=o1_input(state,[],question_record={},round_index=1)
        for phrase in ("==== QUESTION ====", "==== SUBTASK HISTORY ====",
                       "==== AVAILABLE TOOLS", "[Attempt 1/10]"):
            self.assertIn(phrase,text)
        self.assertIn("Who wrote the research report?",text)
        spec=native_tool_spec("GoogleSearchAction")
        self.assertIn(spec.description,text)
        self.assertIn('"query"',text)
        self.assertNotIn("gold_supervision",text)
        self.assertNotIn("expected_answer",text)
        from environment.trajectory_simulation.tools import tool_profile
        # Make every synthetic tool card agree with the actual benchmark
        # definition, not a handcrafted substitute.
        for name in tool_profile("omnigaia").available:
            with self.subTest(tool=name):
                self.assertEqual(native_tool_spec(name).name,name)
                self.assertTrue(native_tool_spec(name).parameters)
                self.assertTrue(native_tool_spec(name).description)

    def test_subtask_bank_materializes_all_answers_before_main_sampling(self):
        _,_,_,bank=self.bank(dependent=True)
        self.assertEqual(len(bank["fragments"]),3)
        self.assertEqual(len(bank["waves"]),3)
        self.assertEqual(bank["complete"]["round_index"],4)
        self.assertEqual(len(bank["full_public_history"]),3)
        for fragment in bank["fragments"].values():
            self.assertTrue(fragment["correct_answer"])
            self.assertTrue(fragment["summary"])
            self.assertTrue(fragment["trace_summary"])
            self.assertIn(fragment["natural_feedback"],{True,False})
        results=[w["step"]["result"]["subtask_results"][0] for w in bank["waves"]]
        self.assertEqual(results[0]["finish_result"]["status"],"done")
        self.assertEqual(results[1]["finish_result"]["status"],"done")
        self.assertEqual(results[2]["finish_result"]["status"],"done")
        self.assertEqual(bank["complete"]["action"]["action"],"complete")

    def test_bank_history_is_topological_and_no_same_wave_future_results(self):
        _,_,_,bank=self.bank(dependent=False)
        self.assertEqual(len(bank["waves"]),2)
        first=bank["waves"][0]
        self.assertEqual(len(first["fragment_ids"]),2)
        self.assertEqual(first["state"]["evidence_store"],{})
        self.assertEqual(
            first["state"]["subtask_history"],"No subtasks completed yet."
        )
        second=bank["waves"][1]
        self.assertIn("Attempt 1",second["state"]["subtask_history"])
        self.assertEqual(len(second["state"]["evidence_store"]),2)
        self.assertNotIn("Attempt 2",second["state"]["subtask_history"])
        self.assertEqual(
            set(second["state"]["evidence_store"]),
            {o["resource_id"] for step in [first["step"]]
             for reply in step["result"]["subtask_results"]
             for o in reply["observed_evidence"]},
        )

    def test_full_public_replay_still_validates_precomputed_bank(self):
        _,_,_,bank=self.bank(dependent=True)
        episode={
            "task_id":bank["task_id"],"question":bank["question"],
            "input_eligibility":bank["eligibility"],
            "validation":{"public_evidence_replay_verified":True},
            "trajectory":[*bank["full_public_history"],{
                "attempt":bank["complete"]["round_index"],
                **bank["complete"]["action"],
                "state_before":bank["complete"]["state"],
                "result":{"done":True,"success":True,
                          "answer":bank["complete"]["action"]["params"]["answer"]},
            }],
        }
        validate_public_episode(episode,require_training=False)
        prefix=episode["trajectory"][:-1]
        validate_public_episode(
            {**episode,"trajectory":prefix},
            require_training=False,require_terminal=False,
        )

    def test_rejected_earlier_natural_language_does_not_block_later_target(self):
        class FirstBadThenGood(PassthroughPlanner):
            offline=False
            style_provider="natural"
            attempts=0
            def naturalize_action(self,payload,canonical):
                if canonical["action"]=="complete":
                    return realize_main(payload,canonical,{"reasoning":""})
                self.attempts+=1
                if self.attempts==1:
                    raise ValueError("first_natural_rewrite_rejected")
                natural=[]
                for task in canonical["params"]["tasks"]:
                    instructions=[
                        re.sub(r"^\d+\.\s+", "", line)
                        for line in task["task_instruction"].splitlines()
                        if re.match(r"^\d+\.\s+",line)
                    ]
                    natural.append(
                        "Review the supplied references to determine the answer. "
                        + " ".join(instructions)
                    )
                return realize_main(
                    payload,canonical,
                    {"reasoning":"","task_instructions":natural},
                )

        q,record,rendered,bank=self.bank(
            dependent=True,planner=FirstBadThenGood(),
        )
        self.assertEqual(bank["waves"][0]["naturalization"]["rewritten_tasks"],0)
        self.assertEqual(bank["waves"][1]["naturalization"]["rewritten_tasks"],1)
        self.assertEqual(bank["waves"][2]["naturalization"]["rewritten_tasks"],1)
        # A complete Gold-backed history is available for a later round even
        # though the first natural Main task did not produce an SFT target.
        second=bank["waves"][1]
        row=make_round_record(
            task_id=bank["task_id"],round_index=2,question=bank["question"],
            eligibility=bank["eligibility"],state=second["state"],
            prefix=[bank["waves"][0]["step"]],
            action=second["action"],question_record=q,
            natural_task_count=1,assigned_task_count=1,
            history_origin="teacher_reconstructed",
        )
        self.assertEqual(row["macro_label"],"advance")
        self.assertIn("[Attempt 1]",row["input"])
        self.assertIn("[Attempt 2/10]",row["input"])
        self.assertNotIn("execution_checks",row["target_json"])
        self.assertNotIn("gold_supervision",row["input"])

        # Main SFT accepts Round 2 without any Round-1 natural target:
        # the offline teacher bank still supplies a complete correct history.
        with tempfile.TemporaryDirectory() as tmp, patch(
            "environment.trajectory_simulation.subtask_bank.build_subtask_bank",
            return_value=bank,
        ), patch(
            "environment.trajectory_simulation.pipeline._rows",return_value=[q],
        ), patch(
            "environment.trajectory_simulation.pipeline._world_map",
            return_value={q["task_id"]:(record,rendered)},
        ):
            source=Path(tmp)/"worlds.jsonl"
            realized=Path(tmp)/"realized.jsonl"
            source.write_text("{}\\n")
            realized.write_text("{}\\n")
            output=Path(tmp)/"rounds.jsonl"
            summary=synthesize_rounds_jsonl(
                source,realized,output,
                planner=FirstBadThenGood(),executor=PassthroughExecutor(),
                round_indices={1,2},progress=False,
            )
            self.assertEqual(summary["rounds_skipped"],1)
            self.assertEqual(summary["eligible_rounds"],1)
            rows=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["round_index"],2)
            self.assertEqual(rows[0]["history_origin"],"teacher_reconstructed")

    def test_round_export_refuses_invalid_state_or_corrupted_prefix(self):
        q,_,_,bank=self.bank(dependent=True)
        second=bank["waves"][1]
        prefix=copy.deepcopy([bank["waves"][0]["step"]])
        broken_state=copy.deepcopy(second["state"])
        broken_state["subtask_history"]="This is a forged summary."
        params=dict(
            task_id=bank["task_id"],round_index=2,
            question=bank["question"],eligibility=bank["eligibility"],
            prefix=prefix,action=second["action"],question_record=q,
            natural_task_count=0,assigned_task_count=1,
            history_origin="teacher_reconstructed",
        )
        with self.assertRaisesRegex(ValueError,"round_history_not_equal"):
            make_round_record(state=broken_state,**params)
        prefix[0]["result"]["subtask_results"][0]["result_value"]=["Incorrect"]
        with self.assertRaises(ValueError):
            make_round_record(state=second["state"],**params)

    def test_synthesize_rounds_jsonl_uses_bank_and_selected_rounds(self):
        q,record,rendered=fixture(dependent=True)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "environment.trajectory_simulation.subtask_bank.check_question",reviewed,
        ),patch(
            "environment.trajectory_simulation.pipeline._rows",
            return_value=[q],
        ),patch(
            "environment.trajectory_simulation.pipeline._world_map",
            return_value={q["task_id"]:(record,rendered)},
        ):
            src=Path(tmp)/"worlds.jsonl"
            realized=Path(tmp)/"realized.jsonl"
            src.write_text("{}\n")
            realized.write_text("{}\n")
            out=Path(tmp)/"rounds.jsonl"
            bank_file=Path(tmp)/"bank.jsonl"
            stats=synthesize_rounds_jsonl(
                src,realized,out,planner=PassthroughPlanner(),
                executor=PassthroughExecutor(),round_indices={2,3},
                schedule="baseline",progress=False,bank_output_path=bank_file,
            )
            self.assertEqual(stats["subtask_bank_fragments"],3)
            self.assertEqual(stats["rounds_considered"],2)
            self.assertEqual(stats["eligible_rounds"],2)
            rows=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([r["round_index"] for r in rows],[2,3])
            self.assertTrue(all(r["history_valid"] for r in rows))
            private=json.loads(bank_file.read_text().strip())
            self.assertTrue(private["private_teacher_bank"])
            self.assertEqual(len(private["fragments"]),3)
            self.assertEqual(stats["history_origins"],{
                "teacher_reconstructed":2,
            })

    def test_invalid_stage2_question_is_not_sft_training_data(self):
        q,record,rendered=fixture(dependent=True)
        with patch(
            "environment.trajectory_simulation.subtask_bank.check_question",
            return_value={
                "eligible":False,"reviewed":True,"question_sha256":digest(q["question"]),
                "method":"test_rejection",
            },
        ):
            from environment.trajectory_simulation.pipeline import GoldTrajectoryError
            with self.assertRaisesRegex(GoldTrajectoryError,"question_eligibility"):
                build_subtask_bank(
                    q,record,rendered,planner=PassthroughPlanner(),
                    executor=PassthroughExecutor(),
                )


if __name__=="__main__":
    unittest.main()
