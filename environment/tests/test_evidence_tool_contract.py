"""Regression tests for primitive-first O1 tool contracts."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.generate_question.spec import make_spec
from environment.generate_question.question import reference_question
from environment.generate_question.quality import compile_and_validate, validate_replay
from environment.generate_question.evidence_tools import (
    EVIDENCE_TOOL_CONTRACT_VERSION,
    acquisition_for_operator, legacy_tools_from_task, o1_tool_definitions,
    registered_o1_tool_names, source_acquisition_policy,
    tools_from_primitive_steps, validate_scene_evidence_tools,
)
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.o1_compat import o1_prompt, source_catalog
from environment.base_orchestration.action_author import (
    _assignment, author_delegate_action, proposed_template_action,
)
from environment.base_orchestration.trajectory_builder import build_episode


def fresh_scene(seed=0):
    spec = make_spec(
        seed_record={
            "record_id": "o1-evidence-contract-" + str(seed),
            "topic": "Premier League",
            "domain_hint": "sports",
        },
        ordinal=seed, run_seed=78, difficulty="hard",
        macro_first_v2_share=100,
    )
    return compile_and_validate(spec, reference_question(spec))


def reviewed_rows(scene):
    return [{
        **deepcopy(row),
        "finish_result": {
            "result": "Verified scoped result: " + json.dumps(row["answer"]),
            "summary": "Checked only the recorded local evidence and returned"
                       " the verified findings to Main.",
            "status": "done",
        },
        "report_authoring": {"status": "accepted"},
    } for row in realize_scene(scene)["macro_tasks"]]


class FrozenO1EvidenceTests(unittest.TestCase):
    def test_baseline_action_schemas_and_names_are_actual(self):
        tools = {tool["name"]: tool for tool in o1_tool_definitions()}
        self.assertEqual(set(tools), {
            "GoogleSearchAction", "ExtractUrlContentAction",
            "ImageAnalysisAction", "VideoAnalysisAction",
            "ParseAudioAction", "ExecuteCodeAction",
        })
        self.assertIn("image_path", tools["ImageAnalysisAction"]["parameters"]["properties"])
        self.assertIn("video_path", tools["VideoAnalysisAction"]["parameters"]["properties"])
        self.assertIn("audio_path", tools["ParseAudioAction"]["parameters"]["properties"])
        self.assertIn("url", tools["ExtractUrlContentAction"]["parameters"]["properties"])
        self.assertIn("query", tools["GoogleSearchAction"]["parameters"]["properties"])
        self.assertNotIn("WorkspaceSearchAction", registered_o1_tool_names())
        self.assertNotIn("TableLookupAction", registered_o1_tool_names())

    def test_every_v2_source_operator_and_macro_is_frozen(self):
        seen_media = set()
        seen_tools = set()
        seen_empty_macro = False
        for seed in range(12):
            scene = fresh_scene(seed)
            self.assertEqual(
                scene["problem_spec"]["evidence_tool_contract_version"],
                EVIDENCE_TOOL_CONTRACT_VERSION,
            )
            self.assertEqual(validate_replay(scene)["status"], "valid")
            self.assertEqual(validate_scene_evidence_tools(scene), [])
            sources = {x["source_id"]: x for x in scene["problem_spec"]["sources"]}
            stage2 = {x["macro_task_id"]: x
                      for x in realize_scene(scene)["macro_tasks"]}
            for macro in scene["macro_graph"]["tasks"]:
                self.assertEqual(
                    macro["required_tools"],
                    tools_from_primitive_steps(macro["operator_chain"]),
                )
                self.assertEqual(
                    macro["required_tools"], stage2[macro["task_id"]]["required_tools"],
                )
                if not macro["required_tools"]:
                    seen_empty_macro = True
                for actual, node in zip(
                    stage2[macro["task_id"]]["trace_summary"]["steps"],
                    macro["operator_chain"],
                ):
                    expected = None
                    if node["op"] in {"scan", "lookup_by_key"}:
                        source = sources[node["source_id"]]
                        expected = acquisition_for_operator(node, source)
                        seen_tools.add(expected["tool"])
                        seen_media.add(expected["modality"])
                        self.assertEqual(source["evidence_access"],
                                         source_acquisition_policy(source))
                        self.assertEqual(actual["tool_executed"], False)
                    else:
                        self.assertNotIn("tool_executed", actual)
                    self.assertEqual(actual.get("evidence_acquisition"), expected)
        self.assertGreater(len(seen_media), 2)
        self.assertIn("GoogleSearchAction", seen_tools)
        self.assertTrue(seen_empty_macro)

    def test_manipulated_acquisition_or_macro_tools_fail_closed(self):
        scene = fresh_scene(2)
        corrupt = deepcopy(scene)
        target = next(x for x in corrupt["macro_graph"]["tasks"]
                      if x["required_tools"])
        target["required_tools"] = ["WorkspaceSearchAction"]
        self.assertTrue(any("aggregation_mismatch" in msg
                            for msg in validate_scene_evidence_tools(corrupt)))
        self.assertEqual(validate_replay(corrupt)["status"], "invalid")

        corrupt = deepcopy(scene)
        target = next(
            step for task in corrupt["macro_graph"]["tasks"]
            for step in task["operator_chain"]
            if "evidence_acquisition" in step
        )
        target["evidence_acquisition"]["tool"] = "TableLookupAction"
        self.assertTrue(any("evidence_step_contract_mismatch" in msg
                            for msg in validate_scene_evidence_tools(corrupt)))
        self.assertEqual(validate_replay(corrupt)["status"], "invalid")

        corrupt = deepcopy(scene)
        source = corrupt["problem_spec"]["sources"][0]
        source["evidence_access"]["tool"] = "TableLookupAction"
        self.assertTrue(any("source_policy_mismatch" in msg
                            for msg in validate_scene_evidence_tools(corrupt)))

    def test_stage3_prompt_and_output_use_frozen_o1_tools(self):
        scene = fresh_scene(3)
        prompt = o1_prompt(scene, [], attempt=1, max_attempts=16,
                           sub_model="qwen3-8b")
        self.assertIn("GoogleSearchAction", prompt)
        self.assertIn("ExtractUrlContentAction", prompt)
        self.assertIn('"image_path"', prompt)
        self.assertNotIn("WorkspaceSearchAction", prompt)
        self.assertNotIn("TableLookupAction", prompt)
        graph = scene["macro_graph"]
        task = next(t for t in graph["tasks"] if t["required_tools"])
        assignment = _assignment(task, source_catalog(scene), {})
        self.assertEqual(assignment["tools"], task["required_tools"])

        class MisroutedAuthor:
            def complete_json(self, system, request):
                action = deepcopy(json.loads(request)["reference_action_for_structure_only"])
                for slot in action["params"]["tasks"]:
                    slot["tools"] = ["TableLookupAction"]
                    slot["model"] = "fabricated_backend"
                return action

        class Critic:
            def complete_json(self, system, request):
                return {
                    "verdict": "pass", "grounded": True,
                    "assignment_fidelity": True, "o1_style": True,
                }

        action, review = author_delegate_action(
            assignments=[assignment], prompt=prompt,
            author=MisroutedAuthor(), critic=Critic(),
            max_revisions=0,
        )
        self.assertEqual(review["status"], "accepted")
        self.assertEqual(action["params"]["tasks"][0]["tools"],
                         task["required_tools"])
        self.assertEqual(action["params"]["tasks"][0]["model"], "model_1")
        self.assertEqual(
            proposed_template_action([assignment])["params"]["tasks"][0]["tools"],
            task["required_tools"],
        )

    def test_stage3_requires_stage2_tool_trace_fidelity(self):
        scene = fresh_scene(4)
        reports = reviewed_rows(scene)
        episode, turns, _, result = build_episode(scene, reports)
        self.assertEqual(result["status"], "accepted")
        self.assertGreater(len(turns), 2)
        corrupt = deepcopy(reports)
        target = next(row for row in corrupt if row["required_tools"])
        target["required_tools"] = ["TableLookupAction"]
        with self.assertRaisesRegex(ValueError, "not_grounded"):
            build_episode(scene, corrupt)
        corrupt = deepcopy(reports)
        target = next(
            step for row in corrupt
            for step in row["trace_summary"]["steps"]
            if "evidence_acquisition" in step
        )
        target["tool_executed"] = True
        with self.assertRaisesRegex(ValueError, "acquisition_tool_drift"):
            build_episode(scene, corrupt)

    def test_pre_contract_scene_remains_replay_compatible(self):
        spec = deepcopy(fresh_scene(5)["problem_spec"])
        spec.pop("evidence_tool_contract_version")
        for source in spec["sources"]:
            source.pop("evidence_access")
        legacy = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(legacy)["status"], "valid")
        self.assertNotIn("required_tools", legacy["macro_graph"]["tasks"][0])
        catalog = source_catalog(legacy)
        for task in legacy["macro_graph"]["tasks"]:
            assignment = _assignment(task, catalog, {})
            self.assertEqual(
                assignment["tools"], legacy_tools_from_task(task, catalog),
            )
            break


    def test_all_web_routes_and_search_to_extract_url_handoff(self):
        covered = set()
        seen_search_extract = False
        for seed in range(20):
            scene = fresh_scene(seed)
            sources = {x["source_id"]: x for x in scene["problem_spec"]["sources"]}
            public = source_catalog(scene)
            reports = {row["macro_task_id"]: row for row
                       in realize_scene(scene)["macro_tasks"]}
            self.assertEqual(validate_replay(scene)["status"], "valid")
            for macro in scene["macro_graph"]["tasks"]:
                trace = reports[macro["task_id"]]["trace_summary"]["steps"]
                for step, observed in zip(macro["operator_chain"], trace):
                    if "evidence_acquisition" not in step:
                        continue
                    source = sources[step["source_id"]]
                    if source["modality"] not in {"text", "table"}:
                        continue
                    route = step["evidence_acquisition"]["strategy"]
                    covered.add(route)
                    calls = observed["simulated_tool_calls"]
                    self.assertEqual(
                        [call["tool"] for call in calls],
                        [req["tool"] for req
                         in step["evidence_acquisition"]["tool_steps"]],
                    )
                    self.assertTrue(all(
                        call["tool_executed"] is False
                        and call["execution_mode"] == "simulated_no_tool_execution"
                        for call in calls
                    ))
                    self.assertEqual(
                        macro["required_tools"],
                        tools_from_primitive_steps(macro["operator_chain"]),
                    )
                    if route == "search_only":
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0]["tool"], "GoogleSearchAction")
                        self.assertNotIn("resource_url", public[step["source_id"]])
                        self.assertEqual(
                            calls[0]["observation"]["source_fact_refs"],
                            observed["supporting_fact_refs"],
                        )
                    elif route == "direct_extract":
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0]["tool"], "ExtractUrlContentAction")
                        self.assertEqual(
                            calls[0]["params"]["url"],
                            public[step["source_id"]]["resource_url"],
                        )
                        self.assertIn("resource_url", scene["public"]["available_source_descriptions"][
                            next(i for i, x in enumerate(
                                scene["public"]["available_source_descriptions"]
                            ) if x["source_id"] == step["source_id"])
                        ])
                    else:
                        seen_search_extract = True
                        self.assertEqual(route, "search_then_extract")
                        self.assertEqual(
                            [x["tool"] for x in calls],
                            ["GoogleSearchAction", "ExtractUrlContentAction"],
                        )
                        self.assertNotIn("resource_url", public[step["source_id"]])
                        discovered = calls[0]["observation"]["selected_result"]["url"]
                        self.assertEqual(calls[1]["params"]["url"], discovered)
                        self.assertEqual(calls[1]["depends_on"], [calls[0]["call_id"]])
                        self.assertNotIn("source_fact_refs", calls[0]["observation"])
                        self.assertEqual(
                            calls[1]["observation"]["source_fact_refs"],
                            observed["supporting_fact_refs"],
                        )
                        self.assertTrue(discovered.endswith(
                            "/" + step["source_id"] + "/record"
                        ))
                        self.assertIn(".example.invalid/", discovered)
        self.assertEqual(
            covered, {"search_only", "direct_extract", "search_then_extract"}
        )
        self.assertTrue(seen_search_extract)

    def test_direct_url_visible_but_unsearched_url_not_visible_to_main(self):
        scene = next(
            fresh_scene(seed) for seed in range(20)
            if any(x.get("acquisition_route") == "search_then_extract"
                   for x in fresh_scene(seed)["problem_spec"]["sources"])
        )
        public = scene["public"]["available_source_descriptions"]
        public_catalog = source_catalog(scene)
        prompt = o1_prompt(scene, [], attempt=1, max_attempts=16,
                           sub_model="qwen3-8b")
        for source in scene["problem_spec"]["sources"]:
            url = source.get("simulated_page_url")
            if source.get("acquisition_route") == "direct_extract":
                self.assertEqual(public_catalog[source["source_id"]]["resource_url"],
                                 url)
                self.assertIn(url, prompt)
            if source.get("acquisition_route") == "search_then_extract":
                self.assertNotIn("resource_url", public_catalog[source["source_id"]])
                self.assertNotIn(url, prompt)
        self.assertTrue(public)

    def test_tampered_search_extract_observation_rejected(self):
        scene = next(
            fresh_scene(seed) for seed in range(20)
            if any(source.get("acquisition_route") == "search_then_extract"
                   for source in fresh_scene(seed)["problem_spec"]["sources"])
        )
        reports = reviewed_rows(scene)
        target = next(
            step for record in reports
            for step in record["trace_summary"]["steps"]
            if len(step.get("simulated_tool_calls", [])) == 2
        )
        calls = target["simulated_tool_calls"]
        calls[1]["params"]["url"] = "https://unrelated.example.invalid/"
        with self.assertRaisesRegex(ValueError, "acquisition_tool_drift"):
            build_episode(scene, reports)

    def test_v1_contract_remains_regenerable(self):
        from environment.generate_question.macro_planning import sample_macro_plan
        from environment.generate_question.evidence_tools import (
            LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION,
        )
        base = fresh_scene(8)
        spec = deepcopy(base["problem_spec"])
        plan, roles = sample_macro_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            record_id=spec["source_anchor"]["source_record_id"],
            topic=spec["scenario"]["topic"],
            difficulty=spec["difficulty_target"],
            domain=spec["macro_plan"]["domain"],
            information_need=True, realism_v3=True,
            extended_grammar=True, goal_resolution_enabled=True,
            evidence_tools_enabled=True,
            evidence_tools_version=LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION,
        )
        spec["macro_plan"] = plan
        spec["evidence_tool_contract_version"] = (
            LEGACY_EVIDENCE_TOOL_CONTRACT_VERSION
        )
        from environment.generate_question.source_title_contract import expected_source_titles
        titles = expected_source_titles(spec)
        for role in roles:
            role["title"] = titles[role["source_id"]]
        spec["sources"] = roles
        legacy = compile_and_validate(spec, reference_question(spec))
        self.assertEqual(validate_replay(legacy)["status"], "valid")
        self.assertTrue(all(
            "tool_steps" not in op["evidence_acquisition"]
            for task in legacy["macro_graph"]["tasks"]
            for op in task["operator_chain"]
            if "evidence_acquisition" in op
        ))


if __name__ == "__main__":
    unittest.main()
