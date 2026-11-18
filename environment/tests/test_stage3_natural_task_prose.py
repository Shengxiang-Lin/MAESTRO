"""Regression tests: Stage 3 writes task prose but never schedules new work."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.generate_question.execution.evidence_tools import (
    legacy_tools_from_task,
)
from environment.base_orchestration.action_author import (
    _assignment,
    _stage3_task_prose_issues,
    _stage3_compile_task_prose,
    _stage3_task_prose_repair_guidance, author_delegate_action,
    stage3_reasoning_issues, STAGE3_TASK_PROSE_AUTHOR_SYSTEM,
    proposed_template_action,
)
from environment.base_orchestration.o1_compat import (
    source_catalog, o1_prompt_from_turn, stage3_author_context,
)
from environment.base_orchestration.trajectory_builder import build_episode
from environment.tests.current_stage1 import current_scene
from environment.tests.test_main_turn_dataset import reviewed_rows


def author_task_prose(assignments):
    result = []
    for item in assignments:
        sources = item.get("allowed_source_ids", [])
        names = item.get("public_source_descriptions", [])
        if sources:
            named = "; ".join(
                obj["source_id"] + " (" + obj["title"] + ")"
                for obj in names
            )
            if not named:
                named = ", ".join(sources)
            task = (
                "Examine " + named + " for the documented observations "
                "relevant to the current investigation. Establish the "
                "requested evidence with exact identifying details and "
                "preserve the associations for subsequent checks."
            )
        else:
            task = (
                "Cross-check the already returned observations, using "
                "the matching identifiers to derive the specifically "
                "requested result without inventing new source material."
            )
        context = (
            "This follows the reported findings from the earlier Sub-Agent "
            "handoff. Use the returned records as evidence, not as a "
            "justification for unperformed future checks."
            if item.get("requires_upstream_result") else
            "Use the user-provided clues and the assigned evidence "
            "descriptions. No earlier Sub-Agent retrieval is assumed."
        )
        result.append({"task_instruction": task, "context": context})
    return result


class TaskProseAuthor:
    def __init__(self):
        self.requests = []

    def complete_json(self, system, payload):
        body = json.loads(payload)
        self.requests.append(body)
        if "pending_investigations" not in body:
            return {"reasoning": (
                "The visible Sub-Agent findings provide the requested "
                "answer and its supporting source evidence."
            )}
        return {
            "reasoning": (
                "The required source investigations are ready to run "
                "using only the information currently available. I will "
                "delegate the assigned evidence work and use the reported "
                "findings for the next dependency-safe step."
            ),
            "tasks": author_task_prose(body["pending_investigations"]),
        }


class PassingCritic:
    def __init__(self):
        self.requests = []

    def complete_json(self, system, payload):
        self.requests.append(json.loads(payload))
        return {
            "verdict": "pass", "grounded": True,
            "assignment_fidelity": True, "o1_style": True,
        }


class CriticDispatchBoundaryTests(unittest.TestCase):
    """Do not let a Critic redesign a code-frozen Stage-3 investigation."""

    @staticmethod
    def _assignment_and_prompt():
        scene = current_scene(0)
        target = next(
            item for item in scene["macro_graph"]["tasks"]
            if item["source_ids"] and not item["depends_on"]
        )
        return _assignment(target, source_catalog(scene), {}), scene["question"]

    def test_out_of_scope_split_gets_second_review_not_auto_accept(self):
        assignment, prompt = self._assignment_and_prompt()
        class InitiallyOutOfScopeCritic:
            def __init__(self, second_pass=True):
                self.requests = []
                self.second_pass = second_pass

            def complete_json(self, system, payload):
                self.requests.append((system, json.loads(payload)))
                if len(self.requests) == 1 or not self.second_pass:
                    return {
                        "verdict": "fail", "grounded": False,
                        "assignment_fidelity": False, "o1_style": True,
                        "reason": "Split the task into two independent rounds.",
                        "repair_hint": "Split the task; the current lookup "
                                       "cannot be done in one task.",
                    }
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        critic = InitiallyOutOfScopeCritic()
        action, status = author_delegate_action(
            assignments=[assignment], prompt=prompt,
            author=TaskProseAuthor(), critic=critic,
            freeze_assignment_prose=True,
            reasoning_only_stage3=True,
            naturalize_assignment_prose=True, max_revisions=0,
            author_system=STAGE3_TASK_PROSE_AUTHOR_SYSTEM,
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertTrue(status["critic_dispatch_scope_rereviewed"])
        self.assertEqual(len(critic.requests), 2)
        self.assertEqual(len(action["params"]["tasks"]), 1)
        self.assertEqual(action["params"]["tasks"][0]["model"],
                         assignment["model"])
        self.assertEqual(action["params"]["tasks"][0]["tools"],
                         assignment["tools"])
        self.assertEqual(
            critic.requests[1][1]["current_round_task_contract"][0][
                "output_fields"
            ], assignment["output_fields"],
        )

        stubborn = InitiallyOutOfScopeCritic(second_pass=False)
        rejected, status = author_delegate_action(
            assignments=[assignment], prompt=prompt,
            author=TaskProseAuthor(), critic=stubborn,
            freeze_assignment_prose=True,
            reasoning_only_stage3=True,
            naturalize_assignment_prose=True, max_revisions=0,
        )
        self.assertIsNone(rejected)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(len(stubborn.requests), 2)

    def test_legitimate_factual_error_has_no_out_of_scope_rereview(self):
        assignment, prompt = self._assignment_and_prompt()
        class EvidenceCritic:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system, payload):
                self.calls += 1
                return {
                    "verdict": "fail", "grounded": False,
                    "assignment_fidelity": False, "o1_style": True,
                    "reason": "The authored prose claims a nationality "
                              "was identified from only a reference key.",
                    "repair_hint": "Remove the unsupported nationality claim.",
                }
        critic = EvidenceCritic()
        action, status = author_delegate_action(
            assignments=[assignment], prompt=prompt,
            author=TaskProseAuthor(), critic=critic,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            naturalize_assignment_prose=True, max_revisions=0,
        )
        self.assertIsNone(action)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(critic.calls, 1)

    def test_stage3_delegate_prompt_returns_to_stable_o1_scope(self):
        """Avoid the v3 instruction inflation and primed boilerplate."""
        prompt = STAGE3_TASK_PROSE_AUTHOR_SYSTEM
        self.assertIn("an actionable request", prompt)
        self.assertIn("which specific clue/identifier/date/relationship", prompt)
        self.assertIn("Mention each permitted source ID literally", prompt)
        self.assertIn("Parallel investigations need distinct", prompt)
        self.assertIn("one natural handoff of 25–700 characters", prompt)
        self.assertIn("Do not write tool names, models or internal task identifiers", prompt)
        self.assertIn("An earlier successful intermediate lookup is not a failed task", prompt)
        self.assertNotIn("ALREADY ASSIGNED", prompt)
        self.assertNotIn("concrete evidence action", prompt)
        self.assertNotIn("next logical step", prompt)
        self.assertNotIn("the investigation has successfully", prompt)
        self.assertNotIn("fixed", prompt.casefold())
        self.assertNotIn("frozen", prompt.casefold())

    def test_stage3_completion_review_improvements_are_retained(self):
        """Prompt rollback must NOT roll back P1 or the safe fallback."""
        from environment.base_orchestration.action_author import (
            _COMPLETE_CRITIC_SYSTEM, author_complete_action,
        )
        self.assertIn("Main-Agent's AUTHORED", _COMPLETE_CRITIC_SYSTEM)
        self.assertIn("final answer", _COMPLETE_CRITIC_SYSTEM)
        self.assertIn("not submitted for your review", _COMPLETE_CRITIC_SYSTEM)
        self.assertIn("AUTHOR'S NEW REASONING", _COMPLETE_CRITIC_SYSTEM)
        self.assertTrue(callable(author_complete_action))


class FrozenTaskProseTests(unittest.TestCase):
    def test_rejects_internal_scheduler_language_not_grounded_domain_terms(self):
        """Actual 8/10 pilot leaked 'fixed task graph' into published reasoning."""
        observed = (
            "This batch executes the first parallel layer of the fixed "
            "dependency graph to retrieve case identifiers.",
            "The current fixed slot requires looking up the confirmed "
            "project key from the index.",
            "The fixed task graph requires a final synthesis step.",
            "The next fixed step is to examine the record ledger.",
            "The three frozen investigation slots can run together.",
        )
        for reason in observed:
            with self.subTest(reason=reason):
                self.assertIn(
                    "internal_orchestration_language",
                    stage3_reasoning_issues(reason, {}),
                )
        self.assertEqual(stage3_reasoning_issues(
            "The fixed deadline is 2025-01-01, so I will inspect the "
            "register to verify the dates.", {},
        ), [])
        self.assertEqual(stage3_reasoning_issues(
            "The observation points to a geographic dependency; I will "
            "check the regional register.", {},
        ), [])
        # The Stage-3 Author should no longer be primed by the terms
        # that caused the repeated public reasoning leak.
        self.assertNotIn("fixed", STAGE3_TASK_PROSE_AUTHOR_SYSTEM.casefold())
        self.assertNotIn("frozen", STAGE3_TASK_PROSE_AUTHOR_SYSTEM.casefold())

    def test_specific_task_with_generic_clause_is_not_a_false_rejection(self):
        """A copied clause is an issue only without additional real clues."""
        scene = current_scene(0)
        task = next(
            t for t in scene["macro_graph"]["tasks"]
            if t["source_ids"] and not t["depends_on"]
        )
        assignment = _assignment(task, source_catalog(scene), {})
        sources = ", ".join(assignment["source_ids"])
        output_field = assignment["output_fields"][0]
        base = (
            "Examine " + sources + ". Identify the candidates meeting "
            "the evidence described in the user question. "
        )
        context = (
            "The initial question supplies a valley landmark and a "
            "shoreline annotation; use these as the matching clues."
        )
        plain = {
            "reasoning": "I will check the available source observations.",
            "tasks": [{
                "task_instruction": (
                    base + "Return the requested " + output_field +
                    " from the assigned records."
                ),
                "context": context,
            }],
        }
        prompt = (
            scene["question"] + " The valley landmark and shoreline "
            "annotation are visible to the investigator."
        )
        errors = _stage3_task_prose_issues(
            plain, [assignment], prompt=prompt,
        )
        self.assertIn("task_0_generic_template_repeated", errors)
        grounded = deepcopy(plain)
        grounded["tasks"][0]["task_instruction"] = (
            base + "Cross-reference the valley landmark with the shoreline "
            "annotation to identify the matching " + output_field + "."
        )
        errors = _stage3_task_prose_issues(
            grounded, [assignment], prompt=prompt,
        )
        self.assertNotIn("task_0_generic_template_repeated", errors)
        self.assertNotIn("task_0_source_scope_missing:" +
                         assignment["source_ids"][0], errors)

    def test_meta_reasoning_retries_without_leaking_internal_dispatch(self):
        scene = current_scene(0)
        target = next(
            t for t in scene["macro_graph"]["tasks"]
            if t["source_ids"] and not t["depends_on"]
        )
        assignment = _assignment(target, source_catalog(scene), {})

        class MetaThenGroundedAuthor:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, payload):
                request = json.loads(payload)
                self.requests.append(request)
                return {
                    "reasoning": (
                        "The current fixed slot requires a lookup in the "
                        "available source records."
                        if not request.get("repair_guidance") else
                        "The named sources contain the observations needed "
                        "to identify the relevant case before checking "
                        "its linked records."
                    ),
                    "tasks": author_task_prose(
                        request["pending_investigations"]
                    ),
                }

        author, critic = MetaThenGroundedAuthor(), PassingCritic()
        action, status = author_delegate_action(
            assignments=[assignment], prompt=scene["question"],
            author=author, critic=critic, max_revisions=1,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            naturalize_assignment_prose=True,
            reasoning_guard=lambda value: stage3_reasoning_issues(value, {}),
            author_system=STAGE3_TASK_PROSE_AUTHOR_SYSTEM,
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertEqual(status["attempts"], 2)
        self.assertEqual(len(author.requests), 2)
        self.assertEqual(len(critic.requests), 1)
        self.assertEqual(
            stage3_reasoning_issues(action["reasoning"], {}), [],
        )
        self.assertTrue(any(
            "visible source clues" in guidance
            for guidance in author.requests[1]["repair_guidance"]
        ))
        template = proposed_template_action([assignment])
        self.assertEqual(action["params"]["tasks"][0]["model"],
                         template["params"]["tasks"][0]["model"])
        self.assertEqual(action["params"]["tasks"][0]["tools"],
                         template["params"]["tasks"][0]["tools"])

    def test_retry_hints_target_context_length_and_generic_prose(self):
        scene = current_scene(0)
        target = next(
            t for t in scene["macro_graph"]["tasks"]
            if t["source_ids"] and not t["depends_on"]
        )
        assignment = _assignment(target, source_catalog(scene), {})
        failures = [{
            "stage": "stage3_task_prose_guard",
            "errors": [
                "task_0_authored_context_invalid",
                "task_0_generic_template_repeated",
            ],
        }]
        guidance = _stage3_task_prose_repair_guidance(
            failures, [assignment],
        )
        self.assertEqual(len(guidance), 2)
        self.assertTrue(any("25–700" in item for item in guidance))
        self.assertTrue(any(
            assignment["source_ids"][0] in item for item in guidance
        ))
        self.assertTrue(any(
            assignment["output_fields"][0] in item for item in guidance
        ))
        self.assertFalse(any("work_M" in item for item in guidance))

        # Both errors must be collected from a single invalid draft.
        # Previously the too-short context caused an early continue,
        # hiding the generic instruction until a later LLM retry.
        simultaneous = _stage3_task_prose_issues({
            "reasoning": (
                "The current source investigation can establish "
                "the requested evidence using the provided clues."
            ),
            "tasks": [{
                "task_instruction": (
                    "Identify the candidates meeting the evidence "
                    "described in the user question for the assigned "
                    "investigation, without checking later sources."
                ),
                "context": "Follow up.",
            }],
        }, [assignment], prompt=scene["question"])
        self.assertIn("task_0_authored_context_invalid", simultaneous)
        self.assertIn("task_0_generic_template_repeated", simultaneous)

        class RepairingAuthor:
            def __init__(self):
                self.requests = []

            def complete_json(self, system, payload):
                request = json.loads(payload)
                self.requests.append(request)
                if not request.get("repair_guidance"):
                    return {
                        "reasoning": (
                            "The current source investigation can establish "
                            "the requested evidence using the provided clues."
                        ),
                        "tasks": [{
                            "task_instruction": (
                                "Identify the candidates meeting the evidence "
                                "described in the user question for the assigned "
                                "investigation, without checking later sources."
                            ),
                            "context": "Follow up.",
                        }],
                    }
                slot = request["pending_investigations"][0]
                return {
                    "reasoning": (
                        "The current source investigation can establish "
                        "the requested evidence using the provided clues."
                    ),
                    "tasks": author_task_prose([slot]),
                }

        author = RepairingAuthor()
        action, status = author_delegate_action(
            assignments=[assignment], prompt=scene["question"],
            author=author, critic=PassingCritic(), max_revisions=1,
            freeze_assignment_prose=True, reasoning_only_stage3=True,
            naturalize_assignment_prose=True,
        )
        self.assertIsNotNone(action, status)
        self.assertEqual(status["status"], "accepted", status)
        self.assertEqual(status["attempts"], 2)
        self.assertEqual(len(author.requests), 2)
        self.assertTrue(any(
            "25–700" in text
            for text in author.requests[1]["repair_guidance"]
        ))
        self.assertTrue(any(
            assignment["source_ids"][0] in text
            for text in author.requests[1]["repair_guidance"]
        ))
        self.assertEqual(action["params"]["tasks"][0]["model"],
                         assignment["model"])
        self.assertEqual(action["params"]["tasks"][0]["tools"],
                         assignment["tools"])

    def test_guard_blocks_invented_identifiers_or_missing_sources(self):
        scene = current_scene(0)
        graph = scene["macro_graph"]
        target = next(
            t for t in graph["tasks"] if t["source_ids"]
        )
        assignment = _assignment(target, source_catalog(scene), {})
        slot = {
            "reference_instruction": assignment["target_instruction"],
            "prior_visible_context": assignment["context"],
            "public_source_descriptions": assignment[
                "public_source_descriptions"
            ],
            "allowed_source_ids": assignment["source_ids"],
            "required_output_fields": assignment["output_fields"],
            "requires_upstream_result": False,
        }
        draft = {
            "reasoning": "I will retrieve the specified local evidence.",
            "tasks": author_task_prose([slot]),
        }
        prompt = scene["question"]
        self.assertEqual(
            _stage3_task_prose_issues(draft, [assignment], prompt=prompt),
            [],
        )
        bad_id = deepcopy(draft)
        bad_id["tasks"][0]["context"] += (
            " The final answer is Dossier-NOTREPORTED999."
        )
        self.assertIn(
            "task_0_invented_visible_identifier",
            _stage3_task_prose_issues(
                bad_id, [assignment], prompt=prompt,
            ),
        )
        bad_source = deepcopy(draft)
        bad_source["tasks"][0]["task_instruction"] = (
            "Examine an entirely different source collection and report "
            "the requested evidence with complete documentation."
        )
        self.assertTrue(any(
            "source_scope_missing" in issue
            for issue in _stage3_task_prose_issues(
                bad_source, [assignment], prompt=prompt,
            )
        ))
        improper = deepcopy(draft)
        improper["tasks"][0]["tools"] = ["unpermitted_tool"]
        self.assertIn(
            "task_0_prose_shape",
            _stage3_task_prose_issues(
                improper, [assignment], prompt=prompt,
            ),
        )
        template = proposed_template_action([assignment])
        result = _stage3_compile_task_prose(draft, [assignment], template)
        self.assertEqual(
            result["params"]["tasks"][0]["tools"],
            template["params"]["tasks"][0]["tools"],
        )
        self.assertEqual(
            result["params"]["tasks"][0]["model"],
            template["params"]["tasks"][0]["model"],
        )
        self.assertIn(
            "Report ",
            result["params"]["tasks"][0]["task_instruction"],
        )

    def test_frozen_dag_and_model_tools_survive_real_prose_authorship(self):
        scene = current_scene(0)
        author, critic = TaskProseAuthor(), PassingCritic()
        episode, turns, audits, status = build_episode(
            scene, reviewed_rows(scene),
            author=author, critic=critic,
            freeze_dispatch=True, max_revisions=0,
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertEqual(episode["trajectory"][-1]["action"], "complete")
        reference_waves = scene["macro_graph"]["reference_waves"]
        private_waves = [
            item["private_macro_alignment"] for item in audits
            if item["action"] == "delegate_task"
        ]
        self.assertEqual(
            private_waves, reference_waves[:len(private_waves)]
        )
        task_map = {
            task["task_id"]: task
            for task in scene["macro_graph"]["tasks"]
        }
        for turn, audit in zip(turns, audits):
            if turn["output"]["action"] != "delegate_task":
                continue
            self.assertEqual(
                audit["authoring"]["authored_fields"],
                ["reasoning", "task_instruction", "context"],
            )
            self.assertTrue(
                audit["authoring"]["naturalized_task_prose"]
            )
            actions = turn["output"]["params"]["tasks"]
            for task, tid in zip(actions, audit["private_macro_alignment"]):
                self.assertEqual(task["model"], "model_1")
                expected_tools = task_map[tid].get("required_tools")
                if expected_tools is None:
                    expected_tools = legacy_tools_from_task(
                        task_map[tid], source_catalog(scene),
                    )
                self.assertEqual(task["tools"], expected_tools)
                if task_map[tid]["source_ids"]:
                    self.assertIn(
                        "documented observations",
                        task["task_instruction"],
                    )
                if task_map[tid]["depends_on"]:
                    self.assertIn(
                        "Previously returned result:", task["context"]
                    )
        natural_author_calls = [
            request for request in author.requests
            if "pending_investigations" in request
        ]
        self.assertTrue(natural_author_calls)
        for request in natural_author_calls:
            self.assertNotIn(
                "reference_action_for_structure_only", request
            )
            self.assertNotIn("gold_answer", json.dumps(request))
            self.assertNotIn("private_macro_alignment", json.dumps(request))
            self.assertEqual(
                len(request["pending_investigations"]),
                len(request["output_format"]["tasks"]),
            )
        reviewer_calls = [
            request for request in critic.requests
            if "candidate_task_prose_only" in request
        ]
        self.assertEqual(len(reviewer_calls), len(private_waves))
        self.assertTrue(all(
            all(set(task) == {"task_instruction", "context"}
                for task in request["candidate_task_prose_only"])
            for request in reviewer_calls
        ))
        self.assertTrue(all(
            request["assignment_contract_validated_by_code"]
            for request in reviewer_calls
        ))

        # Generation-only prompt is compact, but the stored turn still
        # reconstructs the original byte-for-byte O1 training prompt.
        self.assertEqual(len(author.requests), len(turns))
        self.assertEqual(len(critic.requests), len(turns))
        for turn, audit, authored, reviewed in zip(
                turns, audits, author.requests, critic.requests):
            context = authored.get(
                "question_and_subagent_history",
                authored.get("current_main_visible_prompt"),
            )
            self.assertEqual(context, reviewed["current_main_visible_prompt"])
            self.assertIn(scene["question"], context)
            self.assertIn(
                "==== SUBTASK HISTORY ====\n" +
                turn["input"]["subtask_history"], context,
            )
            self.assertNotIn("==== AVAILABLE TOOLS (for SubAgents) ====", context)
            self.assertNotIn("==== OUTPUT FORMAT ====", context)
            self.assertNotIn("gold_answer", context)
            self.assertNotIn("private_macro_alignment", context)

            full_prompt = o1_prompt_from_turn(
                scene, turn, max_attempts=16,
                sub_model="qwen3-8b", model_alias="model_1",
            )
            self.assertIn("==== AVAILABLE TOOLS (for SubAgents) ====", full_prompt)
            self.assertIn("==== OUTPUT FORMAT ====", full_prompt)
            self.assertLess(len(context), len(full_prompt))
            self.assertEqual(audit["full_o1_prompt_chars"], len(full_prompt))
            self.assertEqual(audit["author_context_chars"], len(context))

    def test_compact_context_includes_legacy_public_sources(self):
        scene = deepcopy(current_scene(0))
        scene["public"].pop("task_input", None)
        history = "[Attempt 1] A previously completed public result."
        context = stage3_author_context(
            scene, history, attempt=2, max_attempts=10,
        )
        self.assertIn(scene["question"], context)
        self.assertIn(history, context)
        self.assertIn("Attempt 2/10; remaining 8", context)
        self.assertIn("==== AVAILABLE EVIDENCE SOURCES ====", context)
        for entry in scene["public"]["available_source_descriptions"]:
            self.assertIn(entry["source_id"], context)
            self.assertIn(entry["title"], context)
            if entry.get("resource_url"):
                self.assertIn(entry["resource_url"], context)


if __name__ == "__main__":
    unittest.main()
