"""Stage 3 O1 per-round training data: actual visibility and legal waves."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from envir.generate_question.spec import make_spec
from envir.generate_question.question import reference_question
from envir.generate_question.quality import compile_and_validate
from envir.implement_subtask.realize import realize_scene
from envir.stage3.trajectory_builder import build_episode
from envir.stage3.generate_turns import generate_turns
from envir.stage3.o1_compat import o1_prompt, o1_history
from envir.stage3.action_author import action_issues


def fixture(i=0):
    spec = make_spec(
        seed_record={"record_id": "stage3-turn-"+str(i),
                     "topic": "Premier League", "domain_hint": "sports"},
        ordinal=i, run_seed=97, difficulty="hard",
        macro_first_v2_share=100,
    )
    return compile_and_validate(spec, reference_question(spec))


def reviewed_rows(scene):
    return [
        {
            **deepcopy(row),
            "finish_result": {
                "status": "done",
                "result": (
                    "Visible result from the assigned evidence: " +
                    json.dumps(row["answer"], ensure_ascii=False)
                ),
                "summary": (
                    "Recorded source observations and their scoped "
                    "intermediate output. PUBLIC_SUMMARY_"
                    + row["macro_task_id"].replace("work_M", "S")
                ),
            },
            "report_authoring": {"status": "accepted"},
        }
        for row in realize_scene(scene)["macro_tasks"]
    ]


class MainTurnDatasetTests(unittest.TestCase):
    def test_one_training_example_per_o1_main_decision(self):
        for i in range(15):
            scene = fixture(i)
            records = reviewed_rows(scene)
            episode, turns, private, status = build_episode(scene, records)
            self.assertEqual(status["status"], "accepted")
            self.assertEqual(len(turns), len(episode["trajectory"]))
            self.assertEqual(len(turns), len(private))
            self.assertEqual(turns[-1]["output"]["action"], "complete")
            self.assertEqual([t["round"] for t in turns],
                             list(range(1, len(turns)+1)))
            self.assertEqual(sum(
                t["output"]["action"] == "delegate_task" for t in turns
            ), status["parallel_delegation_rounds"])
            self.assertEqual(sum(
                len(t["output"]["params"]["tasks"])
                for t in turns if t["output"]["action"] == "delegate_task"
            ), scene["macro_graph"]["macro_task_count"] -
                int(status["main_local_goal_resolution"]))
            for turn in turns:
                prompt = turn["input"]["prompt"]
                action = turn["output"]
                self.assertIn("==== QUESTION ====", prompt)
                self.assertIn("==== SUBTASK HISTORY ====", prompt)
                self.assertIn("==== AVAILABLE TOOLS", prompt)
                self.assertIn(scene["question"], prompt)
                self.assertNotIn("macro_task_id", prompt)
                self.assertNotIn("gold_program", prompt)
                self.assertNotIn("work_M", json.dumps(action))
                self.assertEqual(
                    json.loads(turn["output_json"]), action,
                )
            self.assertTrue(all(x["private_only"] for x in private))
            self.assertNotIn("private_macro_alignment", json.dumps(turns))

    def test_information_arrives_after_the_round_not_before(self):
        scene = fixture(2)
        rows = reviewed_rows(scene)
        by_id = {r["macro_task_id"]: r for r in rows}
        _, turns, audit, _ = build_episode(scene, rows)
        first_prompt = turns[0]["input"]["prompt"]
        self.assertIn("No subtasks completed yet.", first_prompt)
        self.assertNotIn("PUBLIC_SUMMARY_", first_prompt)
        for turn, record in zip(turns, audit):
            prompt = turn["input"]["prompt"]
            visible = set(record["visible_macros_before_round"])
            for tid, row in by_id.items():
                marker = "PUBLIC_SUMMARY_" + tid.replace("work_M", "S")
                if tid in visible:
                    self.assertIn(marker, prompt)
                else:
                    self.assertNotIn(marker, prompt)
        self.assertIn("Parallel Batch", turns[1]["input"]["prompt"])
        self.assertTrue(any(
            len(t["output"]["params"]["tasks"]) > 1
            for t in turns if t["output"]["action"] == "delegate_task"
        ))

    def test_frozen_goal_resolution_may_finish_from_known_counts(self):
        matched = 0
        for i in range(40):
            scene = fixture(i)
            kind = scene["problem_spec"]["macro_plan"][
                "goal_resolution"]["kind"]
            if kind == "direct_lookup":
                continue
            episode, turns, audit, status = build_episode(
                scene, reviewed_rows(scene)
            )
            self.assertTrue(status["main_local_goal_resolution"])
            self.assertTrue(audit[-1]["main_locally_computed"])
            self.assertEqual(
                sum(len(t["output"]["params"]["tasks"]) for t in turns
                    if t["output"]["action"] == "delegate_task"),
                len(scene["macro_graph"]["tasks"])-1,
            )
            gold_val = next(iter(scene["gold_answer"][0].values()))
            expected = (
                str(gold_val) + "%" if kind == "calculation"
                else str(gold_val)
            )
            self.assertEqual(turns[-1]["output"]["params"]["answer"], expected)
            matched += 1
        self.assertGreater(matched, 10)

    def test_missing_tampered_unreviewed_records_fail_closed(self):
        scene = fixture(1)
        rows = reviewed_rows(scene)
        with self.assertRaisesRegex(ValueError, "complete_accepted_scene"):
            build_episode(scene, rows[:-1])
        altered = deepcopy(rows)
        altered[0]["answer"] = [{"fabricated": 1}]
        with self.assertRaisesRegex(ValueError, "not_grounded"):
            build_episode(scene, altered)
        altered = deepcopy(rows)
        altered[0]["report_authoring"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "not_reviewed"):
            build_episode(scene, altered)

    def test_dependent_instruction_must_reuse_visible_upstream_identity(self):
        from envir.stage3.action_author import (
            _assignment, proposed_template_action,
        )
        from envir.stage3.o1_compat import source_catalog
        scene = fixture(0)
        graph = scene["macro_graph"]
        target = next(t for t in graph["tasks"] if t["depends_on"])
        rows = {r["macro_task_id"]: r for r in reviewed_rows(scene)}
        expected = _assignment(target, source_catalog(scene), rows)
        action = proposed_template_action([expected])
        self.assertEqual(action_issues(action, [expected], prompt=""), [])
        action["params"]["tasks"][0]["context"] = (
            "Please proceed based on unspecified earlier findings."
        )
        issues = action_issues(action, [expected], prompt="")
        self.assertTrue(any(
            "upstream_" in issue for issue in issues
        ), issues)

    def test_llm_author_and_critic_are_checked_not_replaced(self):
        scene = fixture(3)
        rows = reviewed_rows(scene)
        class Author:
            def __init__(self, bad=False):
                self.bad, self.calls = bad, 0
            def complete_json(self, system, request):
                self.calls += 1
                body = json.loads(request)
                action = body["reference_action_for_structure_only"]
                if self.bad:
                    action = deepcopy(action)
                    action["params"]["tasks"][0]["task_instruction"] = (
                        "Use the private oracle to execute work_M1."
                    )
                return action
        class Critic:
            def __init__(self):
                self.calls = 0
            def complete_json(self, system, request):
                self.calls += 1
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }
        author, critic = Author(), Critic()
        episode, turns, audit, status = build_episode(
            scene, rows, author=author, critic=critic, max_revisions=0,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(author.calls, critic.calls)
        self.assertGreater(author.calls, 0)
        self.assertTrue(all(
            t["generation_mode"] in {
                "llm_reviewed", "teacher_verified_completion",
            } for t in turns
        ))
        bad, critic = Author(bad=True), Critic()
        rejected, turns, audit, status = build_episode(
            scene, rows, author=bad, critic=critic, max_revisions=0,
        )
        self.assertIsNone(rejected)
        self.assertEqual(status["status"], "rejected")
        self.assertEqual(critic.calls, 0)

    def test_atomic_journal_resume_and_public_private_outputs(self):
        scene = fixture(8)
        rows = reviewed_rows(scene)
        with TemporaryDirectory() as td:
            folder = Path(td)
            scenes_path, reports_path = folder/"scenes.jsonl", folder/"reports.jsonl"
            scenes_path.write_text(
                json.dumps(scene, ensure_ascii=False)+"\n",
                encoding="utf-8",
            )
            reports_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False)+"\n"
                        for row in rows), encoding="utf-8",
            )
            output = folder/"stage3"
            first = generate_turns(
                scenes_path=scenes_path, reports_path=reports_path,
                output_dir=output,
            )
            self.assertEqual(first["accepted_scenes"], 1)
            self.assertGreater(first["main_training_turns"], 1)
            second = generate_turns(
                scenes_path=scenes_path, reports_path=reports_path,
                output_dir=output,
            )
            self.assertEqual(first, second)
            public = [
                json.loads(row) for row in
                (output/"main_turns.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(public), first["main_training_turns"])
            self.assertTrue((output/"private"/"teacher_alignment.jsonl").is_file())
            self.assertNotIn("private_macro_alignment", json.dumps(public))
            with self.assertRaisesRegex(ValueError, "signature_mismatch"):
                generate_turns(
                    scenes_path=scenes_path, reports_path=reports_path,
                    output_dir=output, max_attempts=30,
                )


    def test_round_journal_resumes_without_reauthoring_completed_round(self):
        """A process crash after round 1 must preserve round-1 O1 labels."""
        scene = fixture(11)
        rows = reviewed_rows(scene)

        class Author:
            def __init__(self, interrupt_at=None):
                self.calls = 0
                self.interrupt_at = interrupt_at

            def complete_json(self, system, request):
                self.calls += 1
                if self.calls == self.interrupt_at:
                    raise KeyboardInterrupt("simulated interruption")
                return json.loads(request)["reference_action_for_structure_only"]

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        with TemporaryDirectory() as td:
            folder = Path(td)
            scenes = folder / "scenes.jsonl"
            reports = folder / "reports.jsonl"
            scenes.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            reports.write_text(
                "".join(json.dumps(x) + "\n" for x in rows),
                encoding="utf-8",
            )
            output = folder / "interrupted"
            common = {
                "scenes_path": scenes, "reports_path": reports,
                "critic": Critic(), "author_model": "test-author",
                "critic_model": "test-critic", "no_progress": True,
            }
            with self.assertRaises(KeyboardInterrupt):
                generate_turns(
                    **common, author=Author(interrupt_at=2),
                    output_dir=output,
                )
            round_dir = output / "journal" / "000000.rounds"
            first_round = round_dir / "round-0001.json"
            self.assertTrue(first_round.is_file())
            self.assertFalse((round_dir / "round-0002.json").exists())
            self.assertFalse((output / "journal" / "000000.json").exists())
            self.assertEqual(
                (output / "main_turns.jsonl").read_text(), "",
            )
            first_before = first_round.read_bytes()
            resumed_author = Author()
            resumed = generate_turns(
                **common, author=resumed_author, output_dir=output,
            )
            self.assertEqual(resumed["accepted_scenes"], 1)
            self.assertEqual(first_before, first_round.read_bytes())
            self.assertEqual(
                resumed_author.calls,
                resumed["actions"]["delegate_task"] - 1,
            )
            fresh = generate_turns(
                **common, author=Author(), output_dir=folder / "fresh",
            )
            self.assertEqual(resumed["main_training_turns"],
                             fresh["main_training_turns"])
            self.assertEqual(
                (output / "main_turns.jsonl").read_text(),
                (folder / "fresh" / "main_turns.jsonl").read_text(),
            )
            # Repeating the command does not double-publish accepted turns.
            again = generate_turns(
                **common, author=Author(), output_dir=output,
            )
            self.assertEqual(resumed, again)
            self.assertEqual(
                len((output / "main_turns.jsonl").read_text().splitlines()),
                resumed["main_training_turns"],
            )

    def test_only_finished_episodes_are_published_while_running(self):
        first, second = fixture(13), fixture(14)
        self.assertNotEqual(first["question"], second["question"])
        class Author:
            def complete_json(self, system, request):
                data = json.loads(request)
                if second["question"] in data["current_main_visible_prompt"]:
                    raise KeyboardInterrupt("second scene interrupted")
                return data["reference_action_for_structure_only"]

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        with TemporaryDirectory() as td:
            root = Path(td)
            scenes = root / "scenes.jsonl"
            reports = root / "reports.jsonl"
            scenes.write_text("".join(
                json.dumps(x) + "\n" for x in (first, second)
            ), encoding="utf-8")
            reports.write_text("".join(
                json.dumps(r) + "\n"
                for sc in (first, second) for r in reviewed_rows(sc)
            ), encoding="utf-8")
            out = root / "output"
            with self.assertRaises(KeyboardInterrupt):
                generate_turns(
                    scenes_path=scenes, reports_path=reports,
                    output_dir=out, author=Author(), critic=Critic(),
                    author_model="test-author", critic_model="test-critic",
                    no_progress=True,
                )
            self.assertTrue((out / "journal" / "000000.json").is_file())
            self.assertFalse((out / "journal" / "000001.json").exists())
            turns = [
                json.loads(x) for x in
                (out / "main_turns.jsonl").read_text().splitlines()
            ]
            self.assertGreater(len(turns), 1)
            self.assertEqual({row["task_id"] for row in turns},
                             {first["task_id"]})
            self.assertEqual(
                len((out / "episodes.jsonl").read_text().splitlines()), 1,
            )

    def test_corrupt_partial_round_checkpoint_fails_closed(self):
        scene = fixture(15)

        class Interrupt:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, request):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt("stop")
                return json.loads(request)["reference_action_for_structure_only"]

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        with TemporaryDirectory() as td:
            root = Path(td)
            scenes, reports = root / "scenes.jsonl", root / "reports.jsonl"
            scenes.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            reports.write_text("".join(
                json.dumps(row) + "\n" for row in reviewed_rows(scene)
            ), encoding="utf-8")
            args = dict(scenes_path=scenes, reports_path=reports,
                        output_dir=root / "out", critic=Critic(),
                        author_model="test-author", critic_model="test-critic",
                        no_progress=True)
            with self.assertRaises(KeyboardInterrupt):
                generate_turns(**args, author=Interrupt())
            checkpoint = root / "out" / "journal" / "000000.rounds" / "round-0001.json"
            content = json.loads(checkpoint.read_text())
            content["turn"]["input"]["prompt"] = "tampered"
            checkpoint.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_prompt_mismatch"):
                generate_turns(**args, author=Interrupt())


if __name__ == "__main__":
    unittest.main()
