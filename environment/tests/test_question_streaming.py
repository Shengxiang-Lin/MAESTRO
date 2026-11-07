"""No-network regressions for ordered streaming and checkpoint recovery."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from environment.question_generation.pipeline import build_parser, run
from environment.question_generation.streaming import (
    QuestionCheckpointError, QuestionProgress, question_output_lock,
)


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class QuestionProgressTests(unittest.TestCase):
    def test_flush_resume_and_truncate_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            ids=["task-0","task-1","task-2"]
            with QuestionProgress(directory,"stable",ids,flush_every=2) as progress:
                progress.append(0,True,{"task_id":"task-0","question":"Good"},[])
                progress.append(1,False,{"task_id":"task-1","reason":"bad"},[])
                self.assertEqual(progress.completed,2)
            realized=directory/"realized_capability_worlds.jsonl"
            with realized.open("ab") as handle:
                handle.write(b'{"bad": "uncommitted"}\n')
            with QuestionProgress(directory,"stable",ids,flush_every=1) as resumed:
                self.assertEqual(resumed.completed,2)
                resumed.append(2,True,{"task_id":"task-2","question":"Great"},[])
            self.assertEqual([r["task_id"] for r in _rows(realized)],
                             ["task-0","task-2"])
            self.assertEqual([r["index"] for r in _rows(
                directory/"question_review_audit.jsonl")],[0,1,2])
            self.assertEqual([r["task_id"] for r in _rows(
                directory/"question_rejected.jsonl")],["task-1"])

    def test_mismatched_signature_requires_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with QuestionProgress(directory,"original",["task-0"]) as p:
                p.append(0,True,{"task_id":"task-0","question":"good"},[])
            with self.assertRaisesRegex(QuestionCheckpointError,
                                        "--restart-questions"):
                QuestionProgress(directory,"changed",["task-0"])
            with QuestionProgress(directory,"changed",["task-0"],
                                  restart=True) as p:
                self.assertEqual(p.completed,0)

    def test_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with question_output_lock(directory):
                with self.assertRaises(QuestionCheckpointError):
                    with question_output_lock(directory):
                        pass


class QuestionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ids=[f"task-{i}" for i in range(6)]
        self.authoring=self.root/"authoring_inputs.jsonl"
        self.authoring.write_text(
            "".join(json.dumps({"task_id":id_,"draft_question":"Why?"})+"\n"
                    for id_ in self.ids),encoding="utf-8")
        self.manifest=self.root/"manifest.json"
        self.worlds=self.root/"worlds.jsonl"
        self.capabilities=self.root/"capabilities.jsonl"
        for path in (self.manifest,self.worlds,self.capabilities):
            path.write_text("{}\n",encoding="utf-8")
        self.output=self.root/"questions"
        self.args=build_parser().parse_args([
            "--manifest",str(self.manifest),
            "--authoring-inputs",str(self.authoring),
            "--worlds",str(self.worlds),
            "--capabilities",str(self.capabilities),
            "--output-dir",str(self.output),
            "--question-workers","3","--flush-every","2","--no-progress",
        ])

    def fake_backends(self,args):
        args._resolved_models={
            "question_author":{"backend":"vllm","model":"local-test"},
            "question_critic":{"backend":"vllm","model":"local-test"},
        }
        return object(),object()

    def fake_realize(self,payload,*args,**kwargs):
        index=int(payload["task_id"].split("-")[-1])
        time.sleep((5-index)*.006)
        ok=(index%2==0)
        row={"task_id":payload["task_id"]}
        if ok:row["question"]=f"Question {index}"
        else:row["reason"]="quality_rejected"
        return ok,row,[{"author_check":"pass","thread":threading.current_thread().name}]

    def patches(self):
        return (
            patch("environment.question_generation.pipeline.validate_frozen_inputs",
                  return_value={"status":"valid","frozen_count":len(self.ids)}),
            patch("environment.question_generation.pipeline._world_by_task",
                  return_value={id_:{} for id_ in self.ids}),
            patch("environment.question_generation.pipeline._backends",
                  side_effect=self.fake_backends),
        )

    def test_out_of_order_threads_write_in_input_order_and_no_repeated_calls(self):
        a,b,c=self.patches()
        with a,b,c,patch(
            "environment.question_generation.pipeline._realize_one",
            side_effect=self.fake_realize,
        ) as realization:
            first=run(self.args)
            self.assertEqual(realization.call_count,6)
            self.assertEqual(first["requested"],6)
            self.assertEqual(first["accepted"],3)
            self.assertEqual(first["rejected"],3)
            self.assertEqual(first["streaming"]["question_workers"],3)
            audit=_rows(self.output/"question_review_audit.jsonl")
            self.assertEqual([r["task_id"] for r in audit],self.ids)
            self.assertEqual([r["accepted"] for r in audit],
                             [True,False,True,False,True,False])
            self.assertEqual([r["task_id"] for r in _rows(
                self.output/"realized_capability_worlds.jsonl")],
                ["task-0","task-2","task-4"])
            second=run(self.args)
            self.assertEqual(realization.call_count,6)
            self.assertEqual(second["streaming"]["resumed_records"],6)

    def test_interruption_resume_only_missing_frozen_tasks(self):
        a,b,c=self.patches()
        def interrupted(authoring,worlds,capabilities,author,critic,args,progress):
            for index in range(2):
                progress.append(index,True,{"task_id":self.ids[index],
                                            "question":f"Partial {index}"},[])
            raise RuntimeError("simulated interruption")
        with a,b,c:
            with patch("environment.question_generation.pipeline._stream_parallel",
                       side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                    run(self.args)
            with patch("environment.question_generation.pipeline._realize_one",
                       side_effect=self.fake_realize) as realization:
                result=run(self.args)
                self.assertEqual(realization.call_count,4)
                self.assertEqual(result["streaming"]["resumed_records"],2)
                self.assertEqual(result["accepted"],4)
                self.assertEqual(result["rejected"],2)
                self.assertEqual([row["task_id"] for row in _rows(
                    self.output/"question_review_audit.jsonl")],self.ids)

    def test_changed_model_settings_block_resume_without_discarding_outputs(self):
        a,b,c=self.patches()
        with a,b,c:
            with patch("environment.question_generation.pipeline._realize_one",
                       side_effect=self.fake_realize):
                run(self.args)
            self.args.question_temperature=.7
            with self.assertRaisesRegex(QuestionCheckpointError,
                                        "--restart-questions"):
                run(self.args)
            self.assertEqual(len(_rows(self.output/"question_review_audit.jsonl")),6)

    def test_offset_selects_disjoint_inputs_and_preserves_original_slice_indices(self):
        self.args.start_record = 2
        self.args.max_records = 2
        a,b,c=self.patches()
        with a,b,c,patch(
            "environment.question_generation.pipeline._realize_one",
            side_effect=self.fake_realize,
        ) as realization:
            result=run(self.args)
            self.assertEqual([call.args[0]["task_id"] for call in realization.call_args_list],
                             ["task-2","task-3"])
            self.assertEqual(result["frozen_task_range"],
                             {"start_record":2,"end_record_exclusive":4})
            self.assertEqual(result["evaluation_slices"]["previously_debugged_first_10"]["tasks"],2)
            self.assertEqual(result["evaluation_slices"]["holdout_tasks_51_to_100"],{})
            self.args.start_record = 3
            with self.assertRaises(QuestionCheckpointError):
                run(self.args)
            self.assertEqual([row["task_id"] for row in _rows(
                self.output/"question_review_audit.jsonl")],["task-2","task-3"])

    def test_existing_outputs_without_checkpoint_are_not_overwritten(self):
        a,b,c=self.patches()
        self.output.mkdir()
        existing=self.output/"realized_capability_worlds.jsonl"
        existing.write_text('{"previous": true}\n',encoding="utf-8")
        with a,b,c:
            with self.assertRaisesRegex(QuestionCheckpointError,
                                        "--restart-questions"):
                run(self.args)
            self.assertEqual(_rows(existing),[{"previous":True}])


if __name__=="__main__":
    unittest.main()
