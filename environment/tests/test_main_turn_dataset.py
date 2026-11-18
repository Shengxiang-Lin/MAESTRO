"""Stage 3 O1 per-round training data: actual visibility and legal waves."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from threading import Barrier, BrokenBarrierError, Lock
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode
from environment.base_orchestration.generate_turns import generate_turns
from environment.base_orchestration.o1_compat import o1_prompt, o1_history, o1_prompt_from_turn
from environment.base_orchestration.action_author import action_issues


def fixture(i=0):
    return current_scene(i)


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

    def test_v14_critic_receives_current_intermediate_lookup_scope(self):
        from environment.base_orchestration.action_author import (
            author_delegate_action,
            STAGE3_REASONING_AUTHOR_SYSTEM,
            STAGE3_REASONING_CRITIC_SYSTEM,
        )
        # Final question asks to compare two metric tables, but this
        # current round is only the first dependency-key lookup.
        assignment = {
            "target_instruction": (
                "Use the visible marker code MK-6787 to query index_1 "
                "for its matching first record key. Return record_key_1 "
                "without claiming the final comparison is complete."
            ),
            "context": (
                "Previously returned result: the unique matching marker "
                "is MK-6787, associated with Case-005."
            ),
            "source_ids": ["index_1"],
            "output_fields": ["record_key_1"],
            "needs_upstream": True,
            "model": "model_1",
            "tools": [],
        }
        review_requests = []

        class Author:
            def complete_json(self, system, payload):
                request = json.loads(payload)
                assert "reference_action_for_structure_only" not in request
                return {"reasoning": (
                    "The previous report identified marker code MK-6787. "
                    "I will resolve the corresponding first index key "
                    "before querying dependent sources for the final count "
                    "comparison."
                )}

        class Critic:
            def complete_json(self, system, payload):
                request = json.loads(payload)
                review_requests.append(request)
                current = request["current_round_task_contract"]
                scoped = (
                    len(current) == 1
                    and current[0]["source_ids"] == ["index_1"]
                    and current[0]["output_fields"] == ["record_key_1"]
                    and "stat_a" not in current[0]["source_ids"]
                )
                return {
                    "verdict": "pass" if scoped else "fail",
                    "grounded": scoped,
                    "assignment_fidelity": scoped,
                    "o1_style": scoped,
                }

        action, review = author_delegate_action(
            assignments=[assignment],
            prompt=(
                "==== QUESTION ==== Which count is higher, stat_a or stat_b?\n"
                "==== SUBTASK HISTORY ==== A returned report identifies "
                "Case-005 and marker MK-6787."
            ),
            author=Author(), critic=Critic(), max_revisions=0,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            author_system=STAGE3_REASONING_AUTHOR_SYSTEM,
            critic_system=STAGE3_REASONING_CRITIC_SYSTEM,
        )
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(review["mode"], "llm_reviewed_contract_frozen")
        self.assertEqual(
            action["params"]["tasks"][0]["task_instruction"],
            assignment["target_instruction"],
        )
        self.assertTrue(review_requests)
        self.assertIn("current_round_task_contract", review_requests[0])
        self.assertNotIn(
            "locked_main_visible_decision_facts", review_requests[0],
        )
        self.assertIn(
            "First compare", review_requests[0]["review_scope_priority"],
        )

    def test_v14_reasoning_guard_uses_prior_visible_status_only(self):
        from environment.base_orchestration.action_author import (
            stage3_reasoning_issues,
        )
        delivered = {
            "public_slot": {"finish_result": {
                "status": "done", "result": "Case-016",
                "summary": "The assigned record was found.",
            }}
        }
        self.assertIn(
            "invented_prior_subagent_failure",
            stage3_reasoning_issues(
                "The previous retrieval failed, so I must repeat the search.",
                delivered,
            ),
        )
        self.assertEqual(
            stage3_reasoning_issues(
                "Both independent sources disagree; I will compare them "
                "before choosing the final case.", delivered,
            ), [],
        )
        self.assertIn(
            "internal_reference_action_leak",
            stage3_reasoning_issues(
                "Follow reference_action_for_structure_only for this round.",
                delivered,
            ),
        )

    def test_one_training_example_per_o1_main_decision(self):
        for i in range(15):
            scene = fixture(i)
            records = reviewed_rows(scene)
            episode, turns, private, status = build_episode(scene, records)
            self.assertEqual(status["status"], "accepted")
            self.assertEqual(len(turns), len(episode["trajectory"]))
            self.assertEqual(len(turns), len(private))
            self.assertEqual(turns[-1]["output"]["action"], "complete")
            self.assertEqual(episode["format"],
                             "orchestra_o1_main_trajectory.v3")
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
                self.assertEqual(set(turn["input"]), {"subtask_history"})
                self.assertNotIn("prompt", turn["input"])
                prompt = o1_prompt_from_turn(
                    scene, turn, max_attempts=16, sub_model="qwen3-8b",
                )
                action = turn["output"]
                self.assertIn("==== QUESTION ====", prompt)
                self.assertIn("==== SUBTASK HISTORY ====", prompt)
                self.assertIn("==== AVAILABLE TOOLS", prompt)
                self.assertIn(scene["question"], prompt)
                self.assertNotIn("macro_task_id", prompt)
                self.assertNotIn("gold_program", prompt)
                self.assertNotIn("work_M", json.dumps(action))
                self.assertNotIn("output_json", turn)
            self.assertTrue(all(x["private_only"] for x in private))
            self.assertNotIn("private_macro_alignment", json.dumps(turns))

    def test_information_arrives_after_the_round_not_before(self):
        scene = fixture(2)
        rows = reviewed_rows(scene)
        by_id = {r["macro_task_id"]: r for r in rows}
        _, turns, audit, _ = build_episode(scene, rows)
        first_prompt = o1_prompt_from_turn(
            scene, turns[0], max_attempts=16, sub_model="qwen3-8b",
        )
        self.assertIn("No subtasks completed yet.", first_prompt)
        self.assertNotIn("PUBLIC_SUMMARY_", first_prompt)
        for turn, record in zip(turns, audit):
            prompt = o1_prompt_from_turn(
                scene, turn, max_attempts=16, sub_model="qwen3-8b",
            )
            visible = set(record["visible_macros_before_round"])
            for tid, row in by_id.items():
                marker = "PUBLIC_SUMMARY_" + tid.replace("work_M", "S")
                if tid in visible:
                    self.assertIn(marker, prompt)
                else:
                    self.assertNotIn(marker, prompt)
        self.assertIn("Parallel Batch", turns[1]["input"]["subtask_history"])
        self.assertTrue(any(
            len(t["output"]["params"]["tasks"]) > 1
            for t in turns if t["output"]["action"] == "delegate_task"
        ))

    def test_every_main_round_sees_all_previous_subagent_feedback(self):
        scene = fixture(2)
        rows = reviewed_rows(scene)
        episode, turns, audits, status = build_episode(scene, rows)
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(turns[0]["input"]["subtask_history"],
                         "No subtasks completed yet.")
        prior_feedback = []
        for i, turn in enumerate(turns):
            history = turn["input"]["subtask_history"]
            # Each turn's history contains *every previous* Sub-Agent
            # result (including all parallel subtasks), but never the current
            # round's future Sub-Agent results.
            for previous in prior_feedback:
                self.assertIn(previous, history)
            if i == 0:
                self.assertNotIn("PUBLIC_SUMMARY_", history)
            self.assertEqual(turn["round"], i + 1)
            if i < len(episode["trajectory"]):
                trajectory_round = episode["trajectory"][i]
                if trajectory_round["action"] == "delegate_task":
                    for result in trajectory_round["result"]["subtask_results"]:
                        marker = result["finish_result"]["summary"]
                        self.assertNotIn(marker, history)
                        prior_feedback.append(marker)
        self.assertTrue(prior_feedback)
        self.assertEqual(audits[-1]["action"], "complete")
        for marker in prior_feedback:
            self.assertIn(marker, turns[-1]["input"]["subtask_history"])

    def test_compact_input_renders_identical_o1_prompts(self):
        scene = fixture(7)
        _, turns, private, status = build_episode(
            scene, reviewed_rows(scene), max_attempts=18,
            sub_model="other-model", model_alias="model_custom",
        )
        self.assertEqual(status["status"], "accepted")
        import hashlib
        for turn, audit in zip(turns, private):
            rendered = o1_prompt_from_turn(
                scene, turn, max_attempts=18,
                sub_model="other-model", model_alias="model_custom",
            )
            actual_hash = hashlib.sha256(
                json.dumps(rendered, sort_keys=True, ensure_ascii=False)
                .encode("utf-8")
            ).hexdigest()
            self.assertEqual(actual_hash, audit["prompt_sha256"])
            self.assertIn(f"[Attempt {turn['round']}/18]", rendered)
            self.assertIn(turn["input"]["subtask_history"], rendered)
            self.assertNotIn("prompt", turn["input"])

    def test_critic_reviews_prose_without_second_guessing_frozen_tools(self):
        scene = fixture(4)
        rows = reviewed_rows(scene)
        seen = []

        class Author:
            def complete_json(self, system, request):
                payload = json.loads(request)
                result = deepcopy(payload["reference_action_for_structure_only"])
                for task in result["params"]["tasks"]:
                    # Deliberately omit required source IDs and previous
                    # handoff marker; the deterministic compiler restores
                    # the already-visible public evidence context.
                    task["task_instruction"] = (
                        "Review the assigned evidence carefully, identify "
                        "the applicable records, and report the verified "
                        "intermediate result with concrete supporting details."
                    )
                    task["context"] = "Use already available prior findings."
                return result

        class Critic:
            def complete_json(self, system, request):
                payload = json.loads(request)
                self_is_prose = payload["candidate_prose_only"]
                self_assignment = payload["assignment_prose_goals_in_required_order"]
                seen.append((self_is_prose, self_assignment, payload))
                return {"verdict": "pass", "grounded": True,
                        "assignment_fidelity": True, "o1_style": True}

        critic = Critic()
        episode, turns, _, status = build_episode(
            scene, rows, author=Author(), critic=critic,
            max_revisions=0,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertTrue(seen)
        self.assertTrue(all(
            "tools" not in task and "model" not in task
            for prose, _, _ in seen for task in prose["tasks"]
        ))
        self.assertTrue(all(
            item["fixed_tool_and_model_fields_checked_by_code"]
            for _, _, item in seen
        ))
        self.assertTrue(any(
            "Previously returned result:" in task["context"]
            for turn in turns
            if turn["output"]["action"] == "delegate_task"
            for task in turn["output"]["params"]["tasks"]
        ))

    def test_metadata_leak_rejected_and_tool_only_critic_gets_second_review(self):
        from environment.base_orchestration.action_author import (
            _assignment, proposed_template_action, author_delegate_action,
            completion_issues,
        )
        from environment.base_orchestration.o1_compat import source_catalog
        scene = fixture(1)
        target = scene["macro_graph"]["tasks"][0]
        rows = {r["macro_task_id"]: r for r in reviewed_rows(scene)}
        assignment = _assignment(target, source_catalog(scene), rows)
        action = proposed_template_action([assignment])
        action["reasoning"] += (
            " In my previous draft I chose the wrong tool."
        )
        self.assertIn(
            "generation_process_leak",
            action_issues(action, [assignment], prompt=""),
        )
        action["reasoning"] = (
            "I will investigate the assigned source independently, "
            "then use the evidence to guide the next decision."
        )

        class Author:
            def complete_json(self, system, payload):
                return json.loads(payload)["reference_action_for_structure_only"]

        class Critic:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, payload):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "verdict": "fail", "grounded": False,
                        "assignment_fidelity": False, "o1_style": True,
                        "reason": (
                            "Using GoogleSearchAction for simulated workspace "
                            "sources is invalid; replace the tool choice."
                        ),
                        "repair_hint": "Use a different tool instead.",
                    }
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        critic = Critic()
        result, info = author_delegate_action(
            assignments=[assignment], prompt="==== QUESTION ==== Find record",
            author=Author(), critic=critic, max_revisions=0,
        )
        self.assertEqual(info["status"], "accepted")
        self.assertEqual(critic.calls, 2)
        self.assertEqual(result["params"]["tasks"][0]["tools"],
                         assignment["tools"])
        class StillRejectingCritic:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, payload):
                self.calls += 1
                return {
                    "verdict": "fail", "grounded": False,
                    "assignment_fidelity": False, "o1_style": False,
                    "reason": (
                        "GoogleSearchAction is not valid for workspace:// "
                        "sources; the tool must be replaced."
                    ),
                    "repair_hint": "Replace the tool.",
                }

        rejecting = StillRejectingCritic()
        refused, info = author_delegate_action(
            assignments=[assignment], prompt="==== QUESTION ==== Find record",
            author=Author(), critic=rejecting, max_revisions=0,
        )
        self.assertIsNone(refused)
        self.assertEqual(info["status"], "failed")
        self.assertEqual(rejecting.calls, 2)
        completion = {
            "action": "complete", "reasoning": "Generic statement without evidence.",
            "params": {"answer": "wrong"},
        }
        self.assertIn(
            "completion_answer_not_teacher_verified",
            completion_issues(
                completion, expected_answer="correct",
                delivered={target["task_id"]: rows[target["task_id"]]},
            ),
        )

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
        from environment.base_orchestration.action_author import (
            _assignment, proposed_template_action,
        )
        from environment.base_orchestration.o1_compat import source_catalog
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
                "llm_reviewed", "llm_reviewed_completion",
                "teacher_grounded_completion_fallback",
                "teacher_verified_completion",
            } for t in turns
        ))
        bad, critic = Author(bad=True), Critic()
        rejected, turns, audit, status = build_episode(
            scene, rows, author=bad, critic=critic, max_revisions=0,
        )
        self.assertIsNone(rejected)
        self.assertEqual(status["status"], "rejected")
        self.assertEqual(critic.calls, 0)

    def test_completion_author_review_and_explicit_fallback(self):
        scene = fixture(4)
        rows = reviewed_rows(scene)

        class TemplateAuthor:
            def complete_json(self, system, request):
                return json.loads(request)["reference_action_for_structure_only"]

        class GoodCritic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        _, turns, audit, status = build_episode(
            scene, rows, author=TemplateAuthor(), critic=GoodCritic(),
            max_revisions=0,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(status["completion_generation_mode"],
                         "llm_reviewed_completion")
        self.assertEqual(turns[-1]["generation_mode"],
                         "llm_reviewed_completion")
        self.assertEqual(audit[-1]["authoring"]["status"], "accepted")
        self.assertIn("The returned finding states:",
                      turns[-1]["output"]["reasoning"])

        class BadFinalAuthor:
            def complete_json(self, system, request):
                data = json.loads(request)
                template = data["reference_action_for_structure_only"]
                if template["action"] == "complete":
                    return {"reasoning":
                            "As the Critic said about my previous draft, "
                            "I invented future evidence."}
                return template

        _, turns, audit, status = build_episode(
            scene, rows, author=BadFinalAuthor(), critic=GoodCritic(),
            max_revisions=1,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertEqual(
            turns[-1]["generation_mode"],
            "teacher_grounded_completion_fallback",
        )
        self.assertFalse(audit[-1]["authoring"]["reviewed"])
        self.assertNotIn("previous draft",
                         turns[-1]["output"]["reasoning"])
        self.assertTrue(audit[-1]["authoring"]["failures"])

    def test_completion_critic_reviews_only_new_prose_not_frozen_answer(self):
        from environment.base_orchestration.action_author import (
            author_complete_action,
        )
        delivered = {
            "metric_date": {
                "finish_result": {
                    "result": "2018-12-29",
                    "summary": (
                        "The reported midpoint between the two recorded "
                        "metric dates is 2018-12-29."
                    ),
                },
            },
        }
        template = {
            "action": "complete",
            "reasoning": (
                'The returned finding states: "2018-12-29". '
                "The requested date is available from the previously "
                "returned evidence, so no further delegation is needed."
            ),
            "params": {"answer": "2018-12-29"},
        }

        class CompletingAuthor:
            def complete_json(self, system, request):
                return {"reasoning": (
                    "The available public record establishes the requested "
                    "date and no more source lookups are required."
                )}

        class FrozenAnswerCritic:
            def __init__(self, allow_second):
                self.requests = []
                self.allow_second = allow_second

            def complete_json(self, system, request):
                self.requests.append((system, json.loads(request)))
                if len(self.requests) == 1 or not self.allow_second:
                    return {
                        "verdict": "fail", "grounded": False,
                        "assignment_fidelity": False, "o1_style": True,
                        "reason": (
                            "The final answer 2018-12-29 is mathematically "
                            "incorrect; recalculate the midpoint."
                        ),
                        "repair_hint": "Replace the answer with a new date.",
                    }
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        critic = FrozenAnswerCritic(allow_second=True)
        result, status = author_complete_action(
            template=template, prompt="Find the supported midpoint date.",
            delivered=delivered, author=CompletingAuthor(),
            critic=critic, max_revisions=0,
        )
        self.assertEqual(status["mode"], "llm_reviewed_completion")
        self.assertTrue(status["completion_answer_scope_rereviewed"])
        self.assertEqual(result["params"]["answer"], "2018-12-29")
        self.assertEqual(len(critic.requests), 2)
        reviewed_text = critic.requests[0][1][
            "candidate_prose_only"
        ]["reasoning"]
        self.assertNotIn("The returned finding states:", reviewed_text)
        self.assertIn("The returned finding states:",
                      result["reasoning"])
        self.assertIn("SECOND INDEPENDENT REVIEW",
                      critic.requests[1][0])

        stubborn = FrozenAnswerCritic(allow_second=False)
        fallback, info = author_complete_action(
            template=template, prompt="Find the supported midpoint date.",
            delivered=delivered, author=CompletingAuthor(),
            critic=stubborn, max_revisions=0,
        )
        self.assertEqual(info["mode"],
                         "teacher_grounded_completion_fallback")
        self.assertFalse(info["reviewed"])
        self.assertEqual(fallback, template)
        self.assertEqual(len(stubborn.requests), 2)

        class EvidenceErrorCritic:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, request):
                self.calls += 1
                return {
                    "verdict": "fail", "grounded": False,
                    "assignment_fidelity": False, "o1_style": True,
                    "reason": "Authored reasoning invents a separate source.",
                    "repair_hint": "Remove the unsupported evidence claim.",
                }
        evidence_critic = EvidenceErrorCritic()
        _, info = author_complete_action(
            template=template, prompt="Find the supported midpoint date.",
            delivered=delivered, author=CompletingAuthor(),
            critic=evidence_critic, max_revisions=0,
        )
        self.assertEqual(info["mode"],
                         "teacher_grounded_completion_fallback")
        self.assertEqual(evidence_critic.calls, 1)

    def test_all_completions_use_their_own_visible_evidence(self):
        reasons = set()
        for i in (0, 1, 2, 3, 4):
            scene = fixture(i)
            episode, turns, _, status = build_episode(
                scene, reviewed_rows(scene),
            )
            self.assertEqual(status["status"], "accepted")
            reason = turns[-1]["output"]["reasoning"]
            previous_history = turns[-1]["input"]["subtask_history"]
            # No private or future outcome is used to create the rationale.
            self.assertIn("The returned finding states:", reason)
            self.assertIn("PUBLIC_SUMMARY_", previous_history)
            reasons.add(reason)
        self.assertGreaterEqual(len(reasons), 4)

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


    def test_optional_export_rebuilds_exact_full_prompts_and_checks_identity(self):
        from environment.base_orchestration.rebuild_prompts import export_full_prompts
        from hashlib import sha256

        scene = fixture(9)
        with TemporaryDirectory() as td:
            folder = Path(td)
            scenes = folder / "source_scenes.jsonl"
            reports = folder / "stage2_reports.jsonl"
            scenes.write_text(json.dumps(scene, ensure_ascii=False) + "\n",
                              encoding="utf-8")
            reports.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in reviewed_rows(scene)), encoding="utf-8",
            )
            output = folder / "stage3"
            summary = generate_turns(
                scenes_path=scenes, reports_path=reports,
                output_dir=output, no_progress=True,
            )
            self.assertEqual(summary["input_storage"],
                             "cumulative_o1_subtask_history_only")
            compact_path = output / "main_turns.jsonl"
            original_compact = compact_path.read_bytes()
            exported_path = folder / "rehydrated.jsonl"
            exported = export_full_prompts(
                scenes_path=scenes, turns_path=compact_path,
                manifest_path=output / "manifest.json",
                output_path=exported_path,
            )
            self.assertEqual(exported["exported_turns"],
                             summary["main_training_turns"])
            compact = [json.loads(line) for line in
                       original_compact.decode("utf-8").splitlines()]
            reconstructed = [json.loads(line) for line in
                             exported_path.read_text().splitlines()]
            audits = [json.loads(line) for line in
                      (output / "private" / "teacher_alignment.jsonl")
                      .read_text().splitlines()]
            self.assertEqual(len(reconstructed), len(compact))
            for short, full, audit in zip(compact, reconstructed, audits):
                self.assertEqual(set(short["input"]), {"subtask_history"})
                prompt = full["input"]["prompt"]
                self.assertIn(short["input"]["subtask_history"], prompt)
                self.assertEqual(full["output"], short["output"])
                self.assertEqual(
                    sha256(json.dumps(prompt, sort_keys=True,
                                      ensure_ascii=False).encode("utf-8"))
                    .hexdigest(),
                    audit["prompt_sha256"],
                )
            # The exporter never mutates the compact source.
            self.assertEqual(compact_path.read_bytes(), original_compact)
            with self.assertRaisesRegex(ValueError, "must_not_replace_inputs"):
                export_full_prompts(
                    scenes_path=scenes, turns_path=compact_path,
                    manifest_path=output / "manifest.json",
                    output_path=compact_path,
                )
            # The manifest binds a particular frozen Stage-1 source.
            scenes.write_text(scenes.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scene_manifest_mismatch"):
                export_full_prompts(
                    scenes_path=scenes, turns_path=compact_path,
                    manifest_path=output / "manifest.json",
                    output_path=folder / "should_not_exist.jsonl",
                )

    def test_stage3_max_scenes_selects_from_accepted_stage2_not_stage1_prefix(self):
        """Stage-2 failures do not consume max_scenes or Stage-3 attempts."""
        scenes = [fixture(i) for i in (0, 1, 2, 3, 4)]
        # Source order: no reports, complete, partial, complete, complete.
        # max_scenes=2 must choose positions 1 and 3 (not 0 and 1).
        reports = (
            reviewed_rows(scenes[1])
            + reviewed_rows(scenes[2])[:-1]
            + reviewed_rows(scenes[3])
            + reviewed_rows(scenes[4])
        )
        with TemporaryDirectory() as td:
            folder = Path(td)
            scenes_path = folder / "scenes.jsonl"
            reports_path = folder / "reports.jsonl"
            scenes_path.write_text(
                "".join(json.dumps(sc, ensure_ascii=False) + "\n"
                        for sc in scenes), encoding="utf-8",
            )
            reports_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in reports), encoding="utf-8",
            )
            out = folder / "stage3"
            kwargs = dict(
                scenes_path=scenes_path, reports_path=reports_path,
                output_dir=out, max_scenes=2, no_progress=True,
                workers=2, max_attempts=16,
            )
            first = generate_turns(**kwargs)
            self.assertEqual(first["status"], "done", first)
            self.assertEqual(first["requested_scenes"], 2)
            self.assertEqual(first["accepted_scenes"], 2)
            self.assertEqual(first["rejected_scenes"], 0)
            stats = first["stage2_selection"]
            self.assertEqual(stats["stage1_scenes"], 5)
            self.assertEqual(stats["eligible_stage2_scenes"], 3)
            self.assertEqual(stats["selected_stage3_scenes"], 2)
            self.assertEqual(stats["no_published_stage2_reports"], 1)
            self.assertEqual(stats["incomplete_stage2_reports"], 1)
            self.assertEqual(stats["skipped_without_accepted_stage2"], 2)
            published = [
                json.loads(row) for row in
                (out / "main_turns.jsonl").read_text(
                    encoding="utf-8",
                ).splitlines()
            ]
            self.assertEqual(
                {row["task_id"] for row in published},
                {scenes[1]["task_id"], scenes[3]["task_id"]},
            )
            self.assertEqual(
                [json.loads(row)["task_id"] for row in
                 (out / "episodes.jsonl").read_text().splitlines()],
                [scenes[1]["task_id"], scenes[3]["task_id"]],
            )
            self.assertEqual((out / "failed.jsonl").read_text(), "")
            resumed = generate_turns(**{**kwargs, "workers": 1})
            self.assertEqual(resumed["stage2_selection"], stats)
            self.assertEqual(resumed["main_training_turns"],
                             first["main_training_turns"])
            self.assertEqual(resumed["rejected_scenes"], 0)

    def test_stage3_filters_unreviewed_reports_and_rejects_duplicate_macros(self):
        from environment.base_orchestration.generate_turns import (
            _select_stage2_accepted_scenes,
        )
        scenes = [fixture(5), fixture(6)]
        unreviewed = reviewed_rows(scenes[0])
        unreviewed[0]["report_authoring"]["status"] = "failed"
        good = reviewed_rows(scenes[1])
        selected, stats = _select_stage2_accepted_scenes(
            scenes, {
                scenes[0]["task_id"]: unreviewed,
                scenes[1]["task_id"]: good,
            }, max_scenes=1,
        )
        self.assertEqual([x["task_id"] for x in selected],
                         [scenes[1]["task_id"]])
        self.assertEqual(stats["unreviewed_stage2_reports"], 1)
        self.assertEqual(stats["selected_stage3_scenes"], 1)
        with self.assertRaisesRegex(
            ValueError, "stage3_duplicate_stage2_macro_report",
        ):
            _select_stage2_accepted_scenes(
                scenes, {scenes[1]["task_id"]: good + [good[0]]},
                max_scenes=1,
            )

    def test_stage3_refuses_empty_accepted_stage2_bank(self):
        scene = fixture(7)
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes, reports = root / "scenes.jsonl", root / "reports.jsonl"
            scenes.write_text(
                json.dumps(scene, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            reports.write_text("", encoding="utf-8")
            out = root / "stage3"
            with self.assertRaisesRegex(
                ValueError, "stage3_no_complete_accepted_stage2_scenes",
            ):
                generate_turns(
                    scenes_path=scenes, reports_path=reports,
                    output_dir=out, no_progress=True,
                )
            self.assertFalse(out.exists())

    def test_stage3_live_progress_counts_and_resume(self):
        """Scene-level totals and SFT turn counts stay correct on restart."""
        from unittest.mock import patch

        scenes = [fixture(17), fixture(18)]
        # First Scene has a complete reviewed Stage-2 bank and is selected;
        # second has no published reports and must be skipped before Stage 3.
        reports = reviewed_rows(scenes[0])
        bars = []

        class RecordingBar:
            def __init__(self, *args, **kwargs):
                self.initial = kwargs.get("initial", 0)
                self.total = kwargs["total"]
                self.states = []
                self.completed = self.initial
                bars.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def update(self, count):
                self.completed += count

            def set_postfix(self, values):
                self.states.append(dict(values))

        with TemporaryDirectory() as td:
            root = Path(td)
            scenes_path, reports_path = root / "scenes.jsonl", root / "reports.jsonl"
            scenes_path.write_text(
                "".join(json.dumps(scene) + "\n" for scene in scenes),
                encoding="utf-8",
            )
            reports_path.write_text(
                "".join(json.dumps(row) + "\n" for row in reports),
                encoding="utf-8",
            )
            kwargs = {
                "scenes_path": scenes_path, "reports_path": reports_path,
                "output_dir": root / "out", "workers": 1,
                "max_attempts": 16,  # Current 12-Macro fixture can exceed 10 waves.
            }
            with patch("environment.base_orchestration.generate_turns.tqdm", RecordingBar):
                summary = generate_turns(**kwargs)
                resumed = generate_turns(**kwargs)

            self.assertEqual(summary, resumed)
            self.assertEqual(summary["accepted_scenes"], 1)
            self.assertEqual(summary["rejected_scenes"], 0)
            self.assertEqual(summary["status"], "done")
            self.assertEqual(summary["stage2_selection"]["stage1_scenes"], 2)
            self.assertEqual(
                summary["stage2_selection"]["no_published_stage2_reports"], 1,
            )
            self.assertEqual(summary["stage2_selection"]["selected_stage3_scenes"], 1)
            self.assertGreater(summary["main_training_turns"], 1)
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars[0].initial, 0)
            self.assertEqual(bars[0].total, 1)
            self.assertEqual(bars[0].completed, 1)
            self.assertEqual(bars[0].states[0], {
                "accepted": 0, "rejected": 0,
                "written": 0, "turns": 0,
            })
            self.assertEqual(bars[0].states[-1], {
                "accepted": 1, "rejected": 0,
                "written": 1, "turns": summary["main_training_turns"],
            })
            self.assertEqual(bars[1].initial, 1)
            self.assertEqual(bars[1].states[0], bars[0].states[-1])
            saved = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(saved["main_training_turns"], summary["main_training_turns"])
            self.assertEqual(saved["accepted_scenes"] + saved["rejected_scenes"], 1)
            self.assertEqual(saved["stage2_selection"], summary["stage2_selection"])

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
                resumed["actions"]["delegate_task"],
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
                    no_progress=True, workers=1,
                )
            self.assertTrue((out / "journal" / "000000.json").is_file())
            self.assertFalse((out / "journal" / "000001.json").exists())
            live = json.loads((out / "summary.json").read_text())
            self.assertEqual(live["status"], "running")
            self.assertEqual(live["processed_scenes"], 1)
            self.assertEqual(live["accepted_scenes"], 1)
            self.assertEqual(live["rejected_scenes"], 0)
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
            content["turn"]["input"]["subtask_history"] = "tampered"
            checkpoint.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_prompt_mismatch"):
                generate_turns(**args, author=Interrupt())


    def test_four_workers_overlap_and_reconstruct_stable_order(self):
        """Different scenes overlap; their Main rounds stay sequential."""
        scenes = [fixture(i) for i in (21, 22, 23, 24)]

        class Author:
            def __init__(self, parallel=False):
                self.lock = Lock()
                self.barrier = Barrier(2) if parallel else None
                self.calls = 0
                self.overlap_observed = False

            def complete_json(self, system, request):
                with self.lock:
                    self.calls += 1
                    number = self.calls
                if self.barrier is not None and number <= 2:
                    try:
                        self.barrier.wait(timeout=8)
                        self.overlap_observed = True
                    except BrokenBarrierError as exc:
                        raise AssertionError(
                            "independent Scene Author calls did not overlap"
                        ) from exc
                return json.loads(request)["reference_action_for_structure_only"]

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        with TemporaryDirectory() as td:
            root = Path(td)
            scenes_path = root / "scenes.jsonl"
            reports_path = root / "reports.jsonl"
            scenes_path.write_text(
                "".join(json.dumps(x) + "\n" for x in scenes),
                encoding="utf-8",
            )
            reports_path.write_text(
                "".join(
                    json.dumps(r) + "\n"
                    for sc in scenes for r in reviewed_rows(sc)
                ), encoding="utf-8",
            )
            common = {
                "scenes_path": scenes_path, "reports_path": reports_path,
                "critic": Critic(), "no_progress": True,
                "author_model": "test", "critic_model": "test",
            }
            author = Author(parallel=True)
            parallel = generate_turns(
                **common, author=author, workers=4,
                output_dir=root / "parallel",
            )
            self.assertTrue(author.overlap_observed)
            self.assertEqual(parallel["accepted_scenes"], 4)
            self.assertEqual(parallel["workers"], 4)
            serial = generate_turns(
                **common, author=Author(), workers=1,
                output_dir=root / "serial",
            )
            self.assertEqual(serial["workers"], 1)
            for filename in ("main_turns.jsonl", "episodes.jsonl",
                             "private/teacher_alignment.jsonl"):
                self.assertEqual(
                    (root / "parallel" / filename).read_bytes(),
                    (root / "serial" / filename).read_bytes(),
                    filename,
                )
            rows = [
                json.loads(line) for line in
                (root / "parallel" / "main_turns.jsonl").read_text().splitlines()
            ]
            for scene in scenes:
                rounds = [row["round"] for row in rows
                          if row["task_id"] == scene["task_id"]]
                self.assertEqual(rounds, list(range(1, len(rounds) + 1)))
            # Worker count is orchestration-only, not a dataset signature.
            resumed = generate_turns(
                **common, author=Author(), workers=1,
                output_dir=root / "parallel",
            )
            self.assertEqual(resumed["accepted_scenes"], 4)
            self.assertEqual(
                (root / "parallel" / "main_turns.jsonl").read_bytes(),
                (root / "serial" / "main_turns.jsonl").read_bytes(),
            )

    def test_parallel_selection_skips_stage2_incomplete_scenes(self):
        scenes = [fixture(i) for i in (25, 26, 27)]
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes_path = root / "scenes.jsonl"
            reports_path = root / "reports.jsonl"
            scenes_path.write_text(
                "".join(json.dumps(sc) + "\n" for sc in scenes),
                encoding="utf-8",
            )
            reports_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for sc in (scenes[0], scenes[2])
                    for row in reviewed_rows(sc)
                ), encoding="utf-8",
            )
            kwargs = {
                "scenes_path": scenes_path,
                "reports_path": reports_path,
                "output_dir": root / "output",
                "no_progress": True,
            }
            result = generate_turns(**kwargs, workers=3)
            self.assertEqual(result["accepted_scenes"], 2)
            self.assertEqual(result["rejected_scenes"], 0)
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["requested_scenes"], 2)
            self.assertEqual(
                result["stage2_selection"]["no_published_stage2_reports"], 1,
            )
            rows = [
                json.loads(x) for x in
                (root / "output" / "main_turns.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["task_id"] for row in rows},
                {scenes[0]["task_id"], scenes[2]["task_id"]},
            )
            rerun = generate_turns(**kwargs, workers=1)
            self.assertEqual(rerun["main_training_turns"],
                             result["main_training_turns"])
            self.assertEqual(
                len((root / "output" / "main_turns.jsonl").read_text().splitlines()),
                len(rows),
            )

    def test_invalid_worker_counts_fail_before_output_creation(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            scenes_path = root / "scenes.jsonl"
            reports_path = root / "reports.jsonl"
            scenes_path.write_text(json.dumps(fixture(28)) + "\n")
            reports_path.write_text("")
            for value in (0, -1, 65, 2.0, True):
                with self.subTest(workers=value):
                    with self.assertRaisesRegex(
                        ValueError, "stage3_workers_must_be_between"
                    ):
                        generate_turns(
                            scenes_path=scenes_path,
                            reports_path=reports_path,
                            output_dir=root / "output",
                            workers=value,
                        )
                    self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
