"""V4: Task-first public Question and complete per-Task evidence (clean only).

Small fixtures mirror KQA-Pro program and MMQA official question-type schema,
not private real-world data and not actual multimedia files.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from envir.macro.priors import load_kqa, load_mmqa
from envir.macro.v4_builder import build_batch, build_one
from envir.macro.v4_cli import main
from envir.macro.v4_evidence import run_clean_tasks, EvidenceExecutionError
from envir.macro.v4_taskfirst import author_question, task_level_brief
from envir.macro.v4_validation import validate_clean_scene
from envir.macro.v4_streaming import (
    V4QuestionCheckpointError, V4QuestionProgress, stream_naturalize_scenes,
)
from envir.macro.v4_question_review import (
    naturalize_one, naturalize_scenes, public_task_contract, surface_issues,
)


class CleanTaskFirstTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp = Path(temp.name)
        self.kqa_file = self.temp / "train.json"
        self.mmqa_file = self.temp / "MMQA_train.jsonl"
        kqa = [
            {
                "question": "Which groups share a member?",
                "program": [
                    {"function": "Find", "inputs": ["A"], "dependencies": []},
                    {"function": "Relate", "inputs": ["member", "forward"], "dependencies": [0]},
                    {"function": "Find", "inputs": ["B"], "dependencies": []},
                    {"function": "Relate", "inputs": ["member", "forward"], "dependencies": [2]},
                    {"function": "And", "inputs": [], "dependencies": [1, 3]},
                    {"function": "Count", "inputs": [], "dependencies": [4]},
                ],
            },
            {
                "question": "How many things match?",
                "program": [
                    {"function": "FindAll", "inputs": [], "dependencies": []},
                    {"function": "FilterConcept", "inputs": ["Item"], "dependencies": [0]},
                    {"function": "FilterNum", "inputs": ["count", ">", "5"], "dependencies": [1]},
                    {"function": "Count", "inputs": [], "dependencies": [2]},
                ],
            },
            {
                "question": "Which is earlier?",
                "program": [
                    {"function": "Find", "inputs": ["A"], "dependencies": []},
                    {"function": "Find", "inputs": ["B"], "dependencies": []},
                    {"function": "SelectBetween", "inputs": ["date", "less"],
                     "dependencies": [0, 1]},
                ],
            },
        ]
        mmqa = [
            {"qid": "a", "question": "Which photograph matches the record?",
             "metadata": {"type": "Compose(ImageQ,TableQ)",
                          "supporting_context": [
                              {"doc_id": "i", "doc_part": "image"},
                              {"doc_id": "t", "doc_part": "table"}]}},
            {"qid": "b", "question": "Which person is in both sources?",
             "metadata": {"type": "Intersect(ImageListQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "j", "doc_part": "image"},
                              {"doc_id": "x", "doc_part": "text"}]}},
            {"qid": "c", "question": "Which has the larger amount?",
             "metadata": {"type": "Compare(TableQ,TextQ)",
                          "supporting_context": [
                              {"doc_id": "k", "doc_part": "table"},
                              {"doc_id": "y", "doc_part": "text"}]}},
        ]
        self.kqa_file.write_text(json.dumps(kqa), encoding="utf-8")
        self.mmqa_file.write_text(
            "\n".join(json.dumps(r) for r in mmqa) + "\n",
            encoding="utf-8",
        )
        self.kqa, _ = load_kqa(self.kqa_file)
        self.mmqa, _ = load_mmqa(self.mmqa_file)

    def build(self, ordinal):
        return build_one(
            ordinal=ordinal, seed=17, kqa=self.kqa, mmqa=self.mmqa)

    def test_100_scenes_have_clean_multi_evidence_feedback(self):
        scenes, stats = build_batch(
            kqa=self.kqa, mmqa=self.mmqa, target=100, seed=17)
        self.assertEqual(stats["generated"], 100)
        self.assertEqual(stats["clean_evidence_coverage"], 100)
        self.assertEqual(stats["fixed_max_concurrency"], 10)
        self.assertFalse(stats["final_gold_answer_generated"])
        self.assertFalse(stats["untrusted_scenarios_generated"])
        self.assertGreater(stats["macro_task_count"], 300)
        self.assertGreater(stats["evidence_node_count"],
                           3 * stats["macro_task_count"] - 1)
        self.assertGreaterEqual(stats["unique_macro_work_types"], 12)
        self.assertGreaterEqual(stats["macro_graph_signatures"], 15)
        self.assertGreaterEqual(stats["unique_question_texts"], 90)
        print("MACRO_V4_STATS", json.dumps({
            k: stats[k] for k in (
                "macro_task_count", "evidence_node_count",
                "mean_evidence_nodes_per_task", "macro_graph_signatures",
                "unique_macro_work_types", "clean_evidence_coverage",
            )
        }, sort_keys=True))
        for s in scenes:
            self.assertEqual(validate_clean_scene(s), [])
            self.assertNotIn("gold_answer", s)
            self.assertNotIn("task_evidence_graphs", s["public"])
            self.assertNotIn("macro_graph", s["public"])
            self.assertTrue(s["goal_evidence_coverage"]["can_finish_cleanly"])
            self.assertTrue(s["goal_evidence_coverage"]["evidence_level"])
            for t in s["macro_graph"]["tasks"]:
                graph = s["task_evidence_graphs"][t["task_id"]]
                feedback = s["feedback_bank"][t["task_id"]]
                self.assertGreaterEqual(len(graph["nodes"]), 3)
                self.assertTrue(any(
                    n["kind"] == "work_synthesis" for n in graph["nodes"]))
                self.assertEqual(len(feedback["summary_trace"]),
                                 len(graph["nodes"]))
                self.assertEqual(feedback["status"], "completed")
                self.assertTrue(feedback["summary"])
                self.assertTrue(feedback["answer"])
                self.assertTrue(all(
                    trace["evidence_id"] in feedback["evidence_refs"]
                    for trace in feedback["summary_trace"]
                ))

    def test_question_determined_by_tasks_not_private_evidence(self):
        scene = self.build(3)
        brief = task_level_brief(scene)
        meta = scene["task_plan_provenance"]
        self.assertEqual(scene["question"], author_question(
            brief, ordinal=meta["ordinal"], seed=meta["seed"]))
        # Private facts can vary while the public Question remains identical.
        changed = copy.deepcopy(scene)
        sid = next(iter(changed["world"]["sources"]))
        changed["world"]["sources"][sid]["facts"][2]["value"] = "updated source fact"
        self.assertEqual(scene["question"], author_question(
            task_level_brief(changed), ordinal=meta["ordinal"],
            seed=meta["seed"]))
        # Changing the actual terminal work goal MUST affect the Question.
        modified_brief = dict(brief)
        modified_brief["final_work_goal"] = "Assess the alternative proposals"
        self.assertNotEqual(scene["question"], author_question(
            modified_brief, ordinal=meta["ordinal"], seed=meta["seed"]))

    def test_evidence_graphs_only_consume_authorized_and_completed_work(self):
        scene = self.build(7)
        for t in scene["macro_graph"]["tasks"]:
            graph = scene["task_evidence_graphs"][t["task_id"]]
            for n in graph["nodes"]:
                if n["kind"] == "source_observation":
                    self.assertIn(
                        n["source_id"],
                        scene["world"]["source_grants"][t["task_id"]],
                    )
                elif n["kind"].startswith("upstream_"):
                    self.assertIn(n["from_task"], t["depends_on"])
        self.assertEqual(
            run_clean_tasks(scene, scene["world"])["feedback_bank"],
            scene["feedback_bank"],
        )

    def test_tampering_and_missing_evidence_detected(self):
        scene = self.build(9)
        modified = copy.deepcopy(scene)
        tid = next(iter(modified["task_evidence_graphs"]))
        modified["task_evidence_graphs"][tid]["nodes"][0]["statement"] = "INVENTED"
        problems = validate_clean_scene(modified)
        self.assertIn(
            "clean_evidence_or_feedback_replay_mismatch:task_evidence_graphs",
            problems,
        )
        modified = copy.deepcopy(scene)
        modified["feedback_bank"][tid]["summary_trace"].append(
            {"evidence_id": "FAKE"}
        )
        self.assertIn(
            "clean_evidence_or_feedback_replay_mismatch:feedback_bank",
            validate_clean_scene(modified),
        )
        modified = copy.deepcopy(scene)
        sid = next(iter(modified["world"]["sources"]))
        del modified["world"]["sources"][sid]
        self.assertTrue(any(
            x.startswith("clean_replay_failed:")
            for x in validate_clean_scene(modified)
        ))
        modified = copy.deepcopy(scene)
        modified["goal_evidence_coverage"]["can_finish_cleanly"] = False
        self.assertIn("evidence_sufficiency_not_proven",
                      validate_clean_scene(modified))
        modified = copy.deepcopy(scene)
        modified["public"]["macro_graph"] = scene["macro_graph"]
        self.assertIn("public_projection_keys",
                      validate_clean_scene(modified))

    def test_llm_author_critic_accepts_natural_question_without_changing_tasks(self):
        scene = self.build(2)
        before = copy.deepcopy(scene)
        class Author:
            def __init__(self):
                self.requests = []
            def complete_json(self, system, user, *, response_schema=None):
                p = json.loads(user)
                self.requests.append(p)
                return {"question": p["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports your conclusion?",
                )}
        class Critic:
            def __init__(self):
                self.requests = []
            def complete_json(self, system, user, *, response_schema=None):
                p = json.loads(user)
                self.requests.append(p)
                return {"verdict": "pass", "naturalness": "acceptable",
                        "summary": "Same task goal and natural wording.",
                        "differences": []}
        author, critic = Author(), Critic()
        stats = naturalize_scenes([scene], author=author, critic=critic)
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["fallback"], 0)
        self.assertEqual(validate_clean_scene(scene), [])
        self.assertNotEqual(scene["question"], before["question"])
        for field in ("macro_graph", "world", "task_evidence_graphs",
                      "feedback_bank", "goal_evidence_coverage", "source_priors"):
            self.assertEqual(scene[field], before[field], field)
        for payload in author.requests + critic.requests:
            flattened = json.dumps(payload)
            self.assertNotIn(before["world"]["case_reference"], flattened)
            self.assertNotIn('"macro_graph"', flattened)
            self.assertNotIn('"task_evidence_graphs"', flattened)
            self.assertNotIn('"feedback_bank"', flattened)
            self.assertNotIn('"source_fact_refs"', flattened)
            self.assertNotIn('"task_id"', flattened)
            self.assertNotIn('"depends_on"', flattened)
        self.assertFalse(
            scene["question_authoring"]["semantic_equivalence_certified"]
        )
        self.assertEqual(
            scene["question_authoring"]["accepted_review"]["verdict"], "pass"
        )
        self.assertEqual(scene["question_authoring"]["author_calls"], 1)
        self.assertEqual(scene["question_authoring"]["critic_calls"], 1)

    def test_critic_revise_local_issues_and_resubmit(self):
        scene = self.build(1)
        class Author:
            def __init__(self):
                self.requests = []
            def complete_json(self, system, user, *, response_schema=None):
                p = json.loads(user)
                self.requests.append(p)
                if "previous_draft" in p:
                    self_test.assertEqual(
                        p["reviewer_feedback"]["differences"][0]["problem"],
                        "The phrase is awkward and repetitive",
                    )
                    question = p["reference_question"].replace(
                        " Explain which documented findings support your assessment.",
                        " Please cite the relevant evidence.",
                    )
                else:
                    question = p["reference_question"].replace(
                        " Explain which documented findings support your assessment.",
                        " What evidence supports your assessment?",
                    )
                return {"question": question}
        class Critic:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, user, *, response_schema=None):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "verdict": "revise",
                        "naturalness": "needs_polish",
                        "summary": "One local style issue",
                        "differences": [{
                            "reference_excerpt": "",
                            "candidate_excerpt": "What evidence",
                            "problem": "The phrase is awkward and repetitive",
                            "minimal_fix": "Use a shorter evidence request",
                        }],
                    }
                return {
                    "verdict": "pass", "naturalness": "acceptable",
                    "summary": "Equivalent and natural", "differences": [],
                }
        self_test = self
        author, critic = Author(), Critic()
        question, meta = naturalize_one(
            scene, author=author, critic=critic, max_revisions=2,
        )
        self.assertNotEqual(scene["question"], question)
        self.assertEqual(meta["status"], "accepted")
        self.assertEqual(meta["author_calls"], 2)
        self.assertEqual(meta["critic_calls"], 2)
        self.assertEqual(meta["revision_rounds"], 1)
        self.assertEqual(critic.calls, 2)
        self.assertIn("previous_draft", author.requests[1])
        scene["question"] = question
        scene["public"]["question"] = question
        scene["question_authoring"] = meta
        self.assertEqual(validate_clean_scene(scene), [])

    def test_semantic_drift_or_uncertainty_falls_back(self):
        scene = self.build(4)
        reference = scene["question"]
        self.assertIn("invented_numeric_constraint", surface_issues(
            scene, reference + " Use the 98765 threshold.",
        ))
        self.assertIn("private_orchestration_plan_leak", surface_issues(
            scene, reference + " Delegate to S1.",
        ))
        class DriftAuthor:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, user, *, response_schema=None):
                self.calls += 1
                return {"question": json.loads(user)["reference_question"]
                        + " Use the 98765 threshold."}
        class ShouldNotCallCritic:
            def complete_json(self, *args, **kwargs):
                raise AssertionError("Guard must reject before critic")
        author = DriftAuthor()
        question, meta = naturalize_one(
            scene, author=author, critic=ShouldNotCallCritic(),
            max_revisions=2,
        )
        self.assertEqual(question, reference)
        self.assertEqual(meta["status"], "fallback")
        self.assertEqual(meta["author_calls"], 3)
        self.assertEqual(meta["critic_calls"], 0)
        self.assertEqual(author.calls, 3)

    def test_critic_uncertain_and_tampered_certificate_rejected(self):
        scene = self.build(6)
        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports your conclusion?",
                )}
        class Uncertain:
            def complete_json(self, system, user, *, response_schema=None):
                return {
                    "verdict": "uncertain", "naturalness": "acceptable",
                    "summary": "I cannot establish equivalence",
                    "differences": [],
                }
        reference = scene["question"]
        question, meta = naturalize_one(
            scene, author=Author(), critic=Uncertain(), max_revisions=2,
        )
        self.assertEqual(question, reference)
        self.assertEqual(meta["status"], "fallback")
        self.assertEqual(meta["reason"], "critic_uncertain")
        self.assertEqual(meta["author_calls"], 1)
        self.assertEqual(meta["critic_calls"], 1)
        scene["question_authoring"] = meta
        self.assertEqual(validate_clean_scene(scene), [])
        scene["question_authoring"]["macro_graph_sha256"] = "not-the-original"
        self.assertIn("review_macro_graph_changed", validate_clean_scene(scene))

    def test_question_only_cli_naturalize_does_not_regenerate_evidence(self):
        originals, _ = build_batch(
            kqa=self.kqa, mmqa=self.mmqa, target=3, seed=17,
        )
        source = self.temp / "bank.jsonl"
        source.write_text(
            "\n".join(json.dumps(r) for r in originals) + "\n",
            encoding="utf-8",
        )
        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports the assessment?",
                )}
        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                return {
                    "verdict": "pass", "naturalness": "acceptable",
                    "summary": "Aligned", "differences": [],
                }
        instances = [Author(), Critic()]
        def fake_backend(**kwargs):
            return instances.pop(0)
        output = self.temp / "naturalized"
        with patch(
            "environment.question_generation.backend.OpenAICompatibleBackend",
            side_effect=fake_backend,
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "naturalize", "--scenes", str(source),
                "--output-dir", str(output),
                "--question-model", "fake-author",
                "--question-api-base", "http://localhost:8807/v1",
                "--question-workers", "2",
                "--max-revisions", "1",
            ]), 0)
            self.assertEqual(main([
                "replay", "--scenes", str(
                    output / "private_clean_scenes_v4.jsonl"),
            ]), 0)
        changed = [
            json.loads(l) for l in
            (output / "private_clean_scenes_v4.jsonl").read_text().splitlines()
        ]
        summary = json.loads((output / "macro_v4_summary.json").read_text())
        self.assertEqual(summary["statistics"]["question_authoring"]["accepted"], 3)
        self.assertEqual(len(
            (output / "question_review_audit_v4.jsonl").read_text().splitlines()
        ), 3)
        for original, rewritten in zip(originals, changed):
            self.assertNotEqual(original["question"], rewritten["question"])
            for key in ("macro_graph", "world", "task_evidence_graphs", "feedback_bank"):
                self.assertEqual(original[key], rewritten[key])

    def test_duplicate_accepted_surface_reverts_later_scene(self):
        first = self.build(8)
        second = copy.deepcopy(first)
        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports the conclusion?",
                )}
        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                return {
                    "verdict": "pass", "naturalness": "acceptable",
                    "summary": "Equivalent user request", "differences": [],
                }
        stats = naturalize_scenes(
            [first, second], author=Author(), critic=Critic(),
        )
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["fallback"], 1)
        self.assertEqual(
            stats["failure_reasons"]["duplicate_naturalized_question"], 1,
        )
        self.assertNotEqual(first["question"], second["question"])
        self.assertEqual(validate_clean_scene(first), [])
        self.assertEqual(validate_clean_scene(second), [])

    def test_critic_format_retry_uses_same_draft(self):
        scene = self.build(7)
        class Author:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, user, *, response_schema=None):
                self.calls += 1
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports your conclusion?",
                )}
        class Critic:
            def __init__(self):
                self.drafts = []
            def complete_json(self, system, user, *, response_schema=None):
                request = json.loads(user)
                self.drafts.append(request["candidate_question"])
                if len(self.drafts) == 1:
                    return {"verdict": "pass"}  # Malformed response.
                return {
                    "verdict": "pass", "naturalness": "acceptable",
                    "summary": "Equivalent", "differences": [],
                }
        author, critic = Author(), Critic()
        question, info = naturalize_one(
            scene, author=author, critic=critic, max_revisions=2,
        )
        self.assertEqual(info["status"], "accepted", info)
        self.assertEqual(info["author_calls"], 1)
        self.assertEqual(info["critic_calls"], 2)
        self.assertEqual(critic.drafts, [question, question])


    @staticmethod
    def stream_args(**updates):
        settings = {
            "question_workers": 2, "flush_every": 1,
            "no_progress": True, "max_revisions": 1,
            "question_model": "fake-author",
            "question_critic_model": None,
            "question_api_base": "http://localhost:8807/v1",
            "question_critic_api_base": None,
            "question_temperature": 0.55,
            "question_critic_temperature": 0.0,
            "question_max_tokens": 1400,
            "question_critic_max_tokens": 2000,
            "question_timeout": 180,
            "question_no_json_mode": False,
        }
        settings.update(updates)
        return SimpleNamespace(**settings)

    def test_stream_checkpoint_resume_and_truncate_uncommitted_tail(self):
        originals = [self.build(i) for i in range(6)]
        args = self.stream_args()
        output = self.temp / "stream"
        class Author:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, user, *, response_schema=None):
                self.calls += 1
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports the conclusion?",
                )}
        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                return {"verdict": "pass", "naturalness": "acceptable",
                        "summary": "No semantic change", "differences": []}
        writer = V4QuestionProgress.append
        def interrupted(self_writer, index, scene):
            if index == 2:
                raise RuntimeError("simulated process interruption")
            return writer(self_writer, index, scene)
        with patch.object(V4QuestionProgress, "append", interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated process"):
                stream_naturalize_scenes(
                    copy.deepcopy(originals), args=args,
                    output_dir=output, author=Author(), critic=Critic(),
                )
        state = json.loads((output / "question_progress_state_v4.json").read_text())
        self.assertEqual(state["completed"], 2)
        for file in (
            "private_clean_scenes_v4.jsonl",
            "public_questions_v4.jsonl",
            "question_review_audit_v4.jsonl",
        ):
            self.assertEqual(
                len((output / file).read_text(encoding="utf-8").splitlines()), 2
            )
        # Simulate a crash midway through the NEXT JSONL write.
        with (output / "public_questions_v4.jsonl").open("ab") as handle:
            handle.write(b'{"half_written":')
        author = Author()
        stats, unique = stream_naturalize_scenes(
            copy.deepcopy(originals), args=args,
            output_dir=output, author=author, critic=Critic(),
        )
        self.assertEqual(stats["resumed_records"], 2)
        self.assertEqual(stats["streaming"]["completed"], 6)
        self.assertEqual(stats["streaming"]["total"], 6)
        self.assertEqual(author.calls, 4)  # no model calls for committed records
        self.assertGreaterEqual(unique, 1)
        for file in (
            "private_clean_scenes_v4.jsonl",
            "public_questions_v4.jsonl",
            "question_review_audit_v4.jsonl",
        ):
            self.assertEqual(
                len((output / file).read_text(encoding="utf-8").splitlines()), 6
            )
        questions = [
            json.loads(line) for line in
            (output / "private_clean_scenes_v4.jsonl").read_text().splitlines()
        ]
        self.assertEqual([x["task_id"] for x in questions],
                         [x["task_id"] for x in originals])
        for scene in questions:
            self.assertEqual(validate_clean_scene(scene), [])
        # Completed resumption is a no-op (including no Author calls).
        no_op_author = Author()
        stats2, _ = stream_naturalize_scenes(
            copy.deepcopy(originals), args=args,
            output_dir=output, author=no_op_author, critic=Critic(),
        )
        self.assertEqual(no_op_author.calls, 0)
        self.assertEqual(stats2["resumed_records"], 6)

    def test_stream_detects_tampering_and_settings_change(self):
        originals = [self.build(i) for i in range(3)]
        args = self.stream_args()
        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports the conclusion?",
                )}
        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                return {"verdict": "pass", "naturalness": "acceptable",
                        "summary": "Equivalent", "differences": []}
        path = self.temp / "tamper"
        stream_naturalize_scenes(
            copy.deepcopy(originals), args=args, output_dir=path,
            author=Author(), critic=Critic(),
        )
        with self.assertRaisesRegex(
            V4QuestionCheckpointError, "signature differs"
        ):
            stream_naturalize_scenes(
                copy.deepcopy(originals),
                args=self.stream_args(question_model="different-model"),
                output_dir=path, author=Author(), critic=Critic(),
            )
        private_file = path / "private_clean_scenes_v4.jsonl"
        with private_file.open("r+b") as handle:
            handle.write(b"X")
        with self.assertRaisesRegex(
            V4QuestionCheckpointError, "checksum mismatch"
        ):
            stream_naturalize_scenes(
                copy.deepcopy(originals), args=args, output_dir=path,
                author=Author(), critic=Critic(),
            )

    def test_stream_incremental_progress_updates(self):
        originals = [self.build(i) for i in range(4)]
        args = self.stream_args(no_progress=False, question_workers=2)
        calls = []
        class FakeBar:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs["initial"], kwargs["total"]))
            def update(self, n):
                calls.append(("update", n))
            def set_postfix(self, **kwargs):
                calls.append(("postfix", kwargs["accepted"], kwargs["fallback"]))
            def close(self):
                calls.append(("close",))
        class Author:
            def complete_json(self, system, user, *, response_schema=None):
                return {"question": json.loads(user)["reference_question"].replace(
                    " Explain which documented findings support your assessment.",
                    " What evidence supports the conclusion?",
                )}
        class Critic:
            def complete_json(self, system, user, *, response_schema=None):
                return {"verdict": "pass", "naturalness": "acceptable",
                        "summary": "Same goal", "differences": []}
        with patch("envir.macro.v4_streaming.tqdm", FakeBar):
            stats, _ = stream_naturalize_scenes(
                copy.deepcopy(originals), args=args,
                output_dir=self.temp / "progress",
                author=Author(), critic=Critic(),
            )
        self.assertIn(("init", 0, 4), calls)
        self.assertEqual(calls.count(("update", 1)), 4)
        self.assertIn(("close",), calls)
        self.assertEqual(stats["streaming"]["flush_every"], 1)

    def test_cli_build_replay_and_no_overwrite(self):
        path = self.temp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([
                "build", "--kqa", str(self.kqa_file),
                "--mmqa", str(self.mmqa_file),
                "--target", "24", "--output-dir", str(path),
            ]), 0)
            self.assertEqual(main([
                "replay", "--scenes", str(
                    path / "private_clean_scenes_v4.jsonl"),
            ]), 0)
        self.assertEqual(len(
            (path / "public_questions_v4.jsonl").read_text().splitlines()), 24)
        with self.assertRaises(FileExistsError):
            with contextlib.redirect_stdout(io.StringIO()):
                main([
                    "build", "--kqa", str(self.kqa_file),
                    "--mmqa", str(self.mmqa_file),
                    "--target", "24", "--output-dir", str(path),
                ])


if __name__ == "__main__":
    unittest.main()
