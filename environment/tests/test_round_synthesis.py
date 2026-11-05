"""Two explicit Stage-3 phases: correct Sub-Agent feedback, then Main rounds."""
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
    PassthroughPlanner,PassthroughExecutor,
)
from environment.trajectory_simulation.eligibility import digest
from environment.trajectory_simulation.natural_language import realize_main
from environment.trajectory_simulation.rounds import (
    _read_verified_bank,compose_main_rounds,generate_main_rounds_jsonl,
    native_tool_spec,o1_input,
)
from environment.trajectory_simulation.subtask_bank import (
    build_subtask_bank,generate_subtasks_jsonl,
)


def reviewed(gold,record,role,*,review=True):
    return {
        "eligible":True,"reviewed":True,
        "question_sha256":digest(gold["question"]),
        "program_sha256":digest(record["program"]),
        "method":"test_exact_program_review","issues":[],
    }


class TwoPhaseTests(unittest.TestCase):
    def bank(self,*,dependent=True):
        q,world,rendered=fixture(dependent=dependent)
        with patch(
            "environment.trajectory_simulation.subtask_bank.check_question",reviewed
        ):
            bank=build_subtask_bank(
                q,world,rendered,planner=PassthroughPlanner(),
                executor=PassthroughExecutor(),schedule="baseline",
            )
        return q,world,rendered,bank

    def test_phase1_generates_all_subagent_answers_without_main_language(self):
        class Reviewer(PassthroughPlanner):
            style_provider="natural"
            def naturalize_action(self,*args):
                raise AssertionError("Phase 1 must not call Main writer")

        q,world,rendered=fixture(dependent=True)
        with patch("environment.trajectory_simulation.subtask_bank.check_question",reviewed):
            bank=build_subtask_bank(
                q,world,rendered,planner=Reviewer(),
                executor=PassthroughExecutor(),schedule="baseline",
            )
        self.assertEqual(bank["schema_version"],1)
        self.assertEqual(len(bank["fragments"]),3)
        self.assertEqual(len(bank["waves"]),3)
        self.assertEqual(len(bank["full_public_history"]),3)
        self.assertEqual(bank["complete"]["round_index"],4)
        for item in bank["fragments"].values():
            self.assertTrue(item["correct_answer"])
            self.assertTrue(item["summary"])
            self.assertTrue(item["trace_summary"])
            self.assertFalse(item["natural_feedback"])
            self.assertFalse(item["natural_task"])
        for wave in bank["waves"]:
            self.assertEqual(wave["canonical_action"]["action"],"delegate_task")
            self.assertEqual(len(wave["bound_fragments"]),len(wave["fragment_ids"]))
            self.assertNotIn("naturalization",wave)
            self.assertNotIn("action",wave)
        self.assertEqual(bank["gold_plan"]["task_id"],bank["task_id"])

    def test_precomputed_parallel_wave_hides_sibling_answers(self):
        _,_,_,bank=self.bank(dependent=False)
        first,second=bank["waves"]
        self.assertEqual(len(first["fragment_ids"]),2)
        self.assertEqual(first["state"]["evidence_store"],{})
        self.assertEqual(first["state"]["subtask_history"],"No subtasks completed yet.")
        self.assertNotIn("Attempt 1",first["state"]["subtask_history"])
        self.assertIn("Attempt 1",second["state"]["subtask_history"])
        self.assertNotIn("Attempt 2",second["state"]["subtask_history"])

    def test_two_commands_have_distinct_inputs_without_legacy_flow(self):
        from environment.trajectory_simulation.cli import build_parser
        parser=build_parser()
        commands=next(
            action.choices for action in parser._actions
            if isinstance(action,__import__("argparse")._SubParsersAction)
        )
        self.assertEqual(set(commands),{"subtasks","rounds"})
        phase1=parser.parse_args(["subtasks","--worlds","w","--realized","q","--output","b"])
        self.assertEqual(phase1.worlds,"w")
        phase2=parser.parse_args(["rounds","--bank","b","--output","s"])
        self.assertEqual(phase2.bank,"b")
        self.assertFalse(hasattr(phase2,"worlds"))
        self.assertFalse(hasattr(phase2,"realized"))

    def test_phase1_output_can_be_used_by_separate_phase2_without_world(self):
        q,world,rendered=fixture(dependent=True)
        with tempfile.TemporaryDirectory() as tmp,patch(
            "environment.trajectory_simulation.subtask_bank.check_question",reviewed
        ),patch(
            "environment.trajectory_simulation.pipeline._rows",return_value=[q]
        ),patch(
            "environment.trajectory_simulation.pipeline._world_map",
            return_value={q["task_id"]:(world,rendered)}
        ):
            worlds=Path(tmp)/"worlds.jsonl"
            questions=Path(tmp)/"realized.jsonl"
            worlds.write_text("{}\n")
            questions.write_text("{}\n")
            bankfile=Path(tmp)/"bank.jsonl"
            summary=generate_subtasks_jsonl(
                worlds,questions,bankfile,
                reviewer=PassthroughPlanner(),executor=PassthroughExecutor(),
                schedule="baseline",progress=False,
            )
            self.assertEqual(summary["accepted_banks"],1)
            self.assertEqual(summary["subtasks"],3)
            self.assertEqual(summary["gold_waves"],3)
            record=json.loads(bankfile.read_text().strip())
            bank=_read_verified_bank(record)
            self.assertEqual(len(bank["fragments"]),3)

            # Important: Phase 2 receives BANK ONLY; it must not query any
            # original World/Question input or create Sub-Agent language.
            with patch(
                "environment.trajectory_simulation.pipeline._world_map",
                side_effect=AssertionError("Phase 2 cannot load original Worlds")
            ),patch(
                "environment.trajectory_simulation.subtask_bank.build_subtask_bank",
                side_effect=AssertionError("Phase 2 cannot rebuild Sub-Agent reports")
            ):
                output=Path(tmp)/"main_sft.jsonl"
                result=generate_main_rounds_jsonl(
                    bankfile,output,planner=PassthroughPlanner(),progress=False,
                )
            self.assertEqual(result["banks_processed"],1)
            self.assertEqual(result["eligible_rounds"],4)
            self.assertEqual(result["banks_rejected"],0)
            samples=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([item["round_index"] for item in samples],[1,2,3,4])
            self.assertEqual([item["macro_label"] for item in samples],
                             ["advance","advance","advance","complete"])
            self.assertIn("[Attempt 2/10]",samples[1]["input"])
            self.assertNotIn("gold_plan",samples[1]["input"])
            self.assertNotIn("correct_answer",samples[1]["input"])
            self.assertNotIn("execution_checks",samples[1]["target_json"])

    def test_phase2_late_round_survives_earlier_natural_writer_failure(self):
        class FirstBadThenGood(PassthroughPlanner):
            offline=False
            style_provider="natural"
            count=0
            def naturalize_action(self,payload,canonical):
                if canonical["action"]=="complete":
                    return realize_main(payload,canonical,{"reasoning":""})
                self.count+=1
                if self.count==1:
                    raise ValueError("round1 Main rewrite failed")
                natural=[]
                for task in canonical["params"]["tasks"]:
                    steps=[
                        re.sub(r"^\d+\.\s+","",line)
                        for line in task["task_instruction"].splitlines()
                        if re.match(r"^\d+\.\s+",line)
                    ]
                    natural.append(
                        "Review the supplied records and return the answer. "+" ".join(steps)
                    )
                return realize_main(
                    payload,canonical,{"reasoning":"","task_instructions":natural}
                )

        _,_,_,bank=self.bank(dependent=True)
        model=FirstBadThenGood()
        samples,stats=compose_main_rounds(bank,planner=model)
        self.assertEqual(stats["skipped"],1)
        self.assertEqual([row["round_index"] for row in samples],[2,3,4])
        self.assertEqual(samples[0]["history_origin"],"teacher_reconstructed")
        self.assertIn("Attempt 1",samples[0]["input"])
        self.assertNotIn("private_gold",samples[0]["input"])
        self.assertEqual(stats["rewritten"],2)

    def test_phase2_rejects_tampered_bank_and_future_leakage(self):
        _,_,_,bank=self.bank(dependent=True)
        envelope={"format":"maestro.clean_subtasks.v1","bank":bank}
        envelope["sha256"]=digest(envelope)
        self.assertEqual(_read_verified_bank(envelope)["task_id"],bank["task_id"])
        tampered=copy.deepcopy(envelope)
        tampered["bank"]["waves"][0]["step"]["result"]["subtask_results"][0][
            "finish_result"
        ]["result"] = "forged"
        with self.assertRaisesRegex(ValueError,"integrity"):
            _read_verified_bank(tampered)
        # Recomputed hash alone cannot bypass source-only public replay.
        tampered["sha256"]=digest({
            "format":tampered["format"],"bank":tampered["bank"],
        })
        with self.assertRaises(ValueError):
            _read_verified_bank(tampered)

    def test_main_prompt_uses_actual_orchestra_o1_tool_descriptions(self):
        state={
            "question":"Who wrote the research report?",
            "subtask_history":"No subtasks completed yet.",
            "available_tools":["GoogleSearchAction","ExecuteCodeAction"],
            "evidence_store":{},
        }
        prompt=o1_input(state,[],question_record={},round_index=1)
        self.assertIn("[Attempt 1/10]",prompt)
        self.assertIn("==== SUBTASK HISTORY ====",prompt)
        spec=native_tool_spec("GoogleSearchAction")
        self.assertIn(spec.description,prompt)
        self.assertIn('"query"',prompt)
        self.assertNotIn("expected_answer",prompt)


if __name__=="__main__":
    unittest.main()
