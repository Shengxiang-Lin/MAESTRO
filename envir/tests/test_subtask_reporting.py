"""Realistic Sub-Agent finish reporting over frozen Oracle Macro evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from envir.generate_question.spec import make_spec
from envir.generate_question.question import reference_question
from envir.generate_question.quality import compile_and_validate
from envir.implement_subtask.config import (
    DEFAULT_CONFIG, load_reporting_config,
)
from envir.implement_subtask.realize import realize_scene
from envir.implement_subtask.naturalize import naturalize_subtasks
from envir.implement_subtask.reporting import author_critic_report


def scene(i, difficulty="medium"):
    spec = make_spec(
        seed_record={
            "record_id": f"reporting-{i}",
            "topic": f"Fictional Study {i}",
            "domain_hint": "science",
            "source_modalities": ["table", "image", "audio", "text"],
        }, ordinal=i, run_seed=17, difficulty=difficulty,
    )
    result = compile_and_validate(spec, reference_question(spec))
    result["question_authoring"] = {"status": "accepted"}
    result["question_origin"] = "llm_accepted_frozen_spec"
    return result


class ReportAuthor:
    def __init__(self, *, fail_topic=None, forged_tool=False):
        self.calls = []
        self.fail_topic = fail_topic
        self.forged_tool = forged_tool

    def complete_json(self, system_prompt, user_prompt):
        contract = json.loads(user_prompt)["frozen_task_and_evidence"]
        self.calls.append(contract)
        answer_count = len(contract["exact_oracle_answer"])
        if (self.fail_topic is not None
                and self.fail_topic in contract["overall_question_for_context_only"]):
            return {
                "result": "I could not verify the requested records.",
                "summary": "The required source details could not be located in the assigned evidence.",
                "status": "partial",
            }
        return {
            "result": (
                "Checked the assigned source records and identified "
                f"{answer_count} matching row(s); "
                + ("no qualifying records were found." if answer_count == 0
                   else "the exact matching items are included below.")
                + (" Used GoogleSearchAction." if self.forged_tool else "")
            ),
            "summary": (
                "Compared the relevant source observations with upstream "
                "workproducts and applied the recorded criteria. "
                f"The resulting handoff contains {answer_count} row(s); "
                "no additional unsupported eligibility condition was assumed."
            ),
            "status": "done",
        }


class ReportCritic:
    def __init__(self, *, reject_first=False):
        self.calls = []
        self.reject_first = reject_first

    def complete_json(self, system_prompt, user_prompt):
        payload = json.loads(user_prompt)
        self.calls.append(payload)
        passed = not (self.reject_first and len(self.calls) == 1)
        return {
            "verdict": "pass" if passed else "fail",
            "grounded": passed,
            "answer_scope_preserved": True,
            "evidence_scope_respected": True,
            "natural": True,
            "reason": "Supported by frozen evidence." if passed
                      else "Explain source comparison more precisely.",
            "repair_hint": "Mention the actual matching workproduct.",
        }


class ReportingUnitTests(unittest.TestCase):
    def test_real_finish_shape_and_frozen_gold(self):
        frozen = scene(23, "hard")
        realized = realize_scene(frozen)
        record = realized["macro_tasks"][-1]
        original = deepcopy(record)
        author, critic = ReportAuthor(), ReportCritic()
        revised, info = author_critic_report(
            record, question=realized["question"],
            author=author, critic=critic,
            is_terminal=True,
        )
        self.assertEqual(info["status"], "accepted")
        self.assertEqual(info["attempts"], 1)
        self.assertEqual(record, original)
        self.assertEqual(revised["answer"], original["answer"])
        self.assertEqual(revised["answer_with_provenance"],
                         original["answer_with_provenance"])
        self.assertEqual(revised["trace_summary"]["steps"],
                         original["trace_summary"]["steps"])
        self.assertEqual(revised["finish_result"]["status"], "done")
        self.assertEqual(
            set(revised["finish_result"]), {"result", "status", "summary"}
        )
        self.assertIn("Verified result:", revised["finish_result"]["result"])
        self.assertIn(
            json.dumps(record["answer"], ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")),
            revised["finish_result"]["result"],
        )
        self.assertEqual(revised["answer_text"],
                         revised["finish_result"]["result"])
        self.assertEqual(revised["trace_summary"]["narrative"],
                         revised["finish_result"]["summary"])
        self.assertTrue(author.calls[0]["macro_is_terminal"])
        self.assertNotIn("world", author.calls[0])
        self.assertNotIn("gold_program", author.calls[0])
        self.assertEqual(len(critic.calls), 1)

    def test_partial_from_author_never_counts_as_complete(self):
        record = realize_scene(scene(10, "medium"))["macro_tasks"][0]
        author = ReportAuthor(fail_topic="Fictional Study 10")
        reviewed, info = author_critic_report(
            record, question="Fictional Study 10 tasks",
            author=author, critic=ReportCritic(),
            max_revisions=1,
        )
        self.assertIsNone(reviewed)
        self.assertEqual(info["status"], "failed")
        self.assertEqual(info["attempts"], 2)
        self.assertTrue(all(x["stage"] == "guard" for x in info["failures"]))

    def test_unperformed_tools_cannot_be_invented(self):
        record = realize_scene(scene(12, "medium"))["macro_tasks"][0]
        reviewed, info = author_critic_report(
            record, question="Check Fictional Study 12 records",
            author=ReportAuthor(forged_tool=True),
            critic=ReportCritic(), max_revisions=0,
        )
        self.assertIsNone(reviewed)
        self.assertIn("claims_unperformed_tool_or_media_action",
                      " ".join(info["failures"][0]["errors"]))

    def test_independent_critic_uses_targeted_revision(self):
        record = realize_scene(scene(11, "medium"))["macro_tasks"][0]
        author, critic = ReportAuthor(), ReportCritic(reject_first=True)
        reviewed, info = author_critic_report(
            record, question="Check Fictional Study 11 records",
            author=author, critic=critic, max_revisions=2,
        )
        self.assertIsNotNone(reviewed)
        self.assertEqual(info["attempts"], 2)
        self.assertEqual(len(critic.calls), 2)

    def test_config_has_model_roles_and_does_not_break_oracle_mode(self):
        cfg = load_reporting_config(DEFAULT_CONFIG)
        self.assertEqual(cfg["model"], "local-qwen3.5-9b")
        self.assertEqual(cfg["max_revisions"], 2)
        self.assertTrue(str(cfg["source_subtasks"]).endswith("subtasks.jsonl"))


class ReportingBatchTests(unittest.TestCase):
    def _teacher_inputs(self, root, ids):
        teacher_path = root / "oracle.jsonl"
        index_path = root / "index.jsonl"
        with (teacher_path.open("w", encoding="utf-8") as out,
              index_path.open("w", encoding="utf-8") as index):
            for i in ids:
                result = realize_scene(scene(i, "medium"))
                index.write(json.dumps({
                    k: v for k, v in result.items() if k != "macro_tasks"
                }, ensure_ascii=False) + "\n")
                for task in result["macro_tasks"]:
                    out.write(json.dumps(task, ensure_ascii=False) + "\n")
        return teacher_path, index_path

    def test_complete_scene_atomic_acceptance_and_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subtasks, index = self._teacher_inputs(root, [27, 28])
            author = ReportAuthor(fail_topic="Fictional Study 28")
            summary = naturalize_subtasks(
                source_subtasks=subtasks, source_scene_index=index,
                output_dir=root / "reviewed", api_base="http://unused/v1",
                model="mock", workers=3, max_revisions=0,
                no_progress=True,
                clients=(author, ReportCritic()),
            )
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["attempted_scenes"], 2)
            self.assertEqual(summary["accepted_scenes"], 1)
            self.assertEqual(summary["failed_scenes"], 1)
            self.assertGreater(summary["author_critic_failed_macros"], 0)
            rows = [
                json.loads(line) for line in
                (root / "reviewed/subtasks.jsonl").read_text().splitlines()
            ]
            self.assertTrue(rows)
            self.assertEqual({x["scene_task_id"] for x in rows},
                             {scene(27, "medium")["task_id"]})
            self.assertTrue(all("finish_result" in x for x in rows))
            self.assertEqual(len((root / "reviewed/scene_index.jsonl")
                                 .read_text().splitlines()), 1)
            self.assertEqual(len((root / "reviewed/failed.jsonl")
                                 .read_text().splitlines()), 1)
            with self.assertRaises(FileExistsError):
                naturalize_subtasks(
                    source_subtasks=subtasks, source_scene_index=index,
                    output_dir=root / "reviewed",
                    api_base="http://unused/v1", model="mock",
                    clients=(ReportAuthor(), ReportCritic()),
                    no_progress=True,
                )

    def test_successful_batch_retains_one_response_per_oracle_macro(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, idx = self._teacher_inputs(root, [30])
            summary = naturalize_subtasks(
                source_subtasks=source, source_scene_index=idx,
                output_dir=root / "natural", api_base="http://unused/v1",
                model="mock", clients=(ReportAuthor(), ReportCritic()),
                no_progress=True,
            )
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["failed_scenes"], 0)
            self.assertEqual(summary["attempted_macro_tasks"],
                             summary["published_macro_tasks"])
            self.assertEqual(summary["author_critic_failed_macros"], 0)


if __name__ == "__main__":
    unittest.main()
