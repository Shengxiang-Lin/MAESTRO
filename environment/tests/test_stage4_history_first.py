"""History-first Stage-4 keyed handoff integration, independent of an LLM."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.tests.test_untrusted_orchestration import sample
from environment.untrusted_orchestration.runtime.atomic_faults import AtomicFault
from environment.untrusted_orchestration.runtime.faults import relation_rows
from environment.untrusted_orchestration.scenario.candidate_exploration import (
    _clean_products, enumerate_round_candidates,
)
from environment.untrusted_orchestration.scenario.handoff_conflicts import (
    discover_handoffs, expose_prior_handoff, handoff_disagreements,
)
from environment.untrusted_orchestration.state_machine import (
    ExecutionState, deterministic_report, public_history_item,
)


def _scene():
    scene = deepcopy(sample(0)[0])
    source_a = {
        "source_id": "parent_src", "access_policy": {"mode": "browse"},
        "schema": ["case_id", "entity_final"],
        "observations": [{
            "values": {"case_id": "Case-001",
                       "entity_final": "Lakehurst District"},
            "fact_refs": ["parent_src:F001"],
            "observation_id": "parent_src.OBS001",
            "evidence_origin_id": "synthetic:parent001",
        }],
    }
    source_b = {
        "source_id": "lookup_src",
        "access_policy": {"mode": "keyed_lookup",
                          "key_field": "entity_final"},
        "schema": ["entity_final", "observation_year"],
        "observations": [{
            "values": {"entity_final": "Lakehurst District",
                       "observation_year": 2025},
            "fact_refs": ["lookup_src:F001"],
            "observation_id": "lookup_src.OBS001",
            "evidence_origin_id": "synthetic:lookup001",
        }, {
            "values": {"entity_final": "Dunhaven District",
                       "observation_year": 2024},
            "fact_refs": ["lookup_src:F002"],
            "observation_id": "lookup_src.OBS002",
            "evidence_origin_id": "synthetic:lookup002",
        }],
    }
    parent = {
        "task_id": "parent", "source_ids": ["parent_src"],
        "depends_on": [], "external_inputs": [], "input_schema": {},
        "operator_chain": [
            {"id": "parent_scan", "op": "scan", "inputs": [],
             "source_id": "parent_src"},
            {"id": "parent_project", "op": "project",
             "inputs": ["parent_scan"], "fields": ["entity_final"]},
        ],
        "program_node_id": "parent_project",
        "outputs": {"typed_relation": ["entity_final"]},
    }
    child = {
        "task_id": "child", "source_ids": ["lookup_src"],
        "depends_on": ["parent"], "external_inputs": ["parent_project"],
        "input_schema": {"input_0": ["entity_final"]},
        "operator_chain": [
            {"id": "child_lookup", "op": "lookup_by_key",
             "inputs": ["parent_project"], "source_id": "lookup_src",
             "input_key_field": "entity_final",
             "source_key_field": "entity_final"},
            {"id": "child_project", "op": "project",
             "inputs": ["child_lookup"],
             "fields": ["entity_final", "observation_year"]},
        ],
        "program_node_id": "child_project",
        "outputs": {"typed_relation": ["entity_final", "observation_year"]},
    }
    scene["world"]["sources"] = {
        "parent_src": source_a, "lookup_src": source_b,
    }
    scene["macro_graph"] = {
        **scene["macro_graph"],
        "tasks": [parent, child],
        "reference_waves": [["parent"], ["child"]],
        "terminal_task_id": "child",
    }
    scene["gold_answer"] = [{
        "entity_final": "Lakehurst District",
        "observation_year": 2025,
    }]
    return scene, parent, child


def _round_plan(scene):
    return {
        "sample_id": "history-first-test", "task_id": scene["task_id"],
        "scene_index": 0, "branch_round": 3, "index": 0,
        "incoming_task_ids": ["child"],
        "faults": [], "skip_reason": None,
    }


class HistoryFirstHandoffTests(unittest.TestCase):
    def test_source_key_is_available_before_latest_fault(self):
        scene, parent, child = _scene()
        products = _clean_products(scene)
        base = {"trajectory": [{
            "action": "delegate_task", "attempt": 1,
            "result": {"subtask_results": [{
                "finish_result": {"status": "done",
                                  "result": "Lakehurst District",
                                  "summary": "Identified Lakehurst District."},
            }]},
        }]}
        paths = discover_handoffs(
            scene, _round_plan(scene), products, base_episode=base,
        )
        self.assertEqual(paths["child"]["older_task"], "parent")
        self.assertEqual(paths["child"]["key_field"], "entity_final")
        expanded = enumerate_round_candidates(
            scene, _round_plan(scene),
            atomic_fault_kinds=["no_result", "wrong_retrieval",
                                "fabricated_claim"],
            max_candidates=3, max_distractors=1,
            base_episode=base,
        )
        marked = [
            item for item in expanded
            if item["scenario_plan"].get("history_first_plan")
        ]
        self.assertTrue(marked)
        self.assertIn(marked[0]["fault_mode"],
                      {"wrong_retrieval", "fabricated_claim"})
        self.assertEqual(marked[0]["scenario_plan"]["history_first_plan"],
                         paths["child"])

    def test_public_handoff_conflict_is_rechecked_and_resolved(self):
        scene, parent, child = _scene()
        state = ExecutionState(scene, {})
        previous = state.register("parent")
        earlier = deterministic_report(
            parent, previous, instruction="Find the entity.",
            model_alias="model_1",
        )
        # Stage-3-style plain text already states the real value.
        earlier["finish_result"]["result"] = "Lakehurst District"
        state.set_public_report("parent", earlier, round_number=1)
        fault = AtomicFault("wrong_retrieval", "child", 2).as_legacy_fault()
        recent = state.register("child", fault=fault)
        latest = deterministic_report(
            child, recent,
            instruction="Lookup observation_log by entity_final.",
            model_alias="model_1",
        )
        state.set_public_report("child", latest, round_number=2)
        self.assertEqual(relation_rows(recent.product)[0]["values"][
            "entity_final"], "Dunhaven District")

        task_entries = [{
            "subtask_results": [public_history_item(earlier, 0)],
            "total_steps": earlier["steps_taken"],
        }, {
            "subtask_results": [public_history_item(latest, 0)],
            "total_steps": latest["steps_taken"],
        }]
        episode = [{
            "result": {"subtask_results": [earlier]},
        }, {
            "result": {"subtask_results": [latest]},
        }]
        plan = {"older_task": "parent", "latest_task": "child",
                "key_field": "entity_final"}
        result = expose_prior_handoff(state, task_entries, episode, plan)
        self.assertEqual(result["status"], "generated")
        self.assertEqual(
            state.public_reports["parent"]["finish_result"]["result"],
            episode[0]["result"]["subtask_results"][0][
                "finish_result"]["result"],
        )
        self.assertEqual(len(handoff_disagreements(state)), 1)
        state.cross_source_checks_enabled = True
        scheduling = state.scheduling_state()
        self.assertIn("child", scheduling["repair"])
        self.assertEqual(
            state.detected_conflicts[-1]["comparison_kind"],
            "earlier_keyed_handoff",
        )
        self.assertTrue(state.unresolved_cited_conflicts())
        fixed = state.register("child")
        corrected = deterministic_report(
            child, fixed, instruction="Recheck the source and key.",
            model_alias="model_1",
        )
        state.set_public_report("child", corrected, round_number=3)
        self.assertFalse(state.unresolved_cited_conflicts())
        self.assertEqual(handoff_disagreements(state), [])

    def test_real_stage1_scene_produces_cross_round_handoff_conflict(self):
        """End-to-end 4A test on a real Stage-1/2/3 Task Plan, without LLM."""
        from pathlib import Path
        import tempfile

        from environment.tests.current_stage1 import current_scene
        from environment.implement_subtask.realize import realize_scene
        from environment.implement_subtask.reporting import code_owned_result
        from environment.base_orchestration.trajectory_builder import (
            build_episode,
        )
        from environment.untrusted_orchestration.generate_turns import (
            generate_turns,
        )

        scene = current_scene(0, "medium")
        teacher = realize_scene(scene)
        reports = []
        for item in teacher["macro_tasks"]:
            reports.append({
                **deepcopy(item),
                "finish_result": {
                    "status": "done",
                    "result": code_owned_result(item),
                    "summary": "Returned the exact delegated Macro result.",
                },
                "report_authoring": {
                    "status": "accepted",
                    "mode": "pilot_deterministic_not_llm_reviewed",
                },
            })
        clean, _turns, _audit, status = build_episode(
            scene, reports, freeze_dispatch=True, max_attempts=16,
        )
        self.assertEqual(status["status"], "accepted")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name, records in (
                ("scenes.jsonl", [scene]),
                ("subtasks.jsonl", reports),
                ("clean_episodes.jsonl", [clean]),
            ):
                (output / name).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in records),
                    encoding="utf-8",
                )
            summary = generate_turns(
                scenes_path=output / "scenes.jsonl",
                reports_path=output / "subtasks.jsonl",
                clean_episodes_path=output / "clean_episodes.jsonl",
                output_dir=output / "stage4",
                max_scenes=1, branches_per_scene=None, seed=17,
                scenario_mode="event_driven",
                atomic_fault_kinds=["wrong_retrieval", "fabricated_claim"],
                candidate_strategy="enumerate", max_candidates_per_round=2,
                max_distractors_per_fault=1,
                max_selected_per_round=1,
                max_language_attempts_per_round=1,
                max_rounds=16, max_retries=1, max_parallel_tasks=4,
                workers=2, no_progress=True, overwrite=True,
                main_mode="template", report_mode="template",
            )
            print("HISTORY_FIRST_4A_COUNTS=" + json.dumps({
                "anchors": summary.get("historical_anchor_status_distribution", {}),
                "reasons": summary.get("historical_anchor_reason_distribution", {}),
                "detected_conflicts": summary.get("cross_round_report_conflicts_detected"),
                "outcomes": summary.get("execution_outcomes", {}),
                "reviewed_positives": summary.get("outcomes", {}),
                "main_training_turns": summary.get("main_training_turns"),
            }, ensure_ascii=False, sort_keys=True), flush=True)
            self.assertGreater(
                summary.get("historical_anchor_status_distribution", {})
                       .get("generated", 0), 0,
                summary.get("historical_anchor_reason_distribution", {}),
            )
            self.assertGreater(
                summary.get("cross_round_report_conflicts_detected", 0), 0,
                summary.get("historical_anchor_reason_distribution", {}),
            )


    def test_handoff_discovery_is_not_limited_to_adjacent_waves(self):
        """A keyed handoff in wave 1 is usable after any intervening waves."""
        for intervening in (0, 1, 3):
            with self.subTest(intervening=intervening):
                scene, parent, child = _scene()
                filler_tasks = []
                for index in range(intervening):
                    filler = deepcopy(parent)
                    filler["task_id"] = f"unrelated_{index}"
                    filler_tasks.append(filler)
                scene["macro_graph"]["tasks"] = [
                    parent, *filler_tasks, child,
                ]
                scene["macro_graph"]["reference_waves"] = [
                    ["parent"],
                    *[[filler["task_id"]] for filler in filler_tasks],
                    ["child"],
                ]
                plan = _round_plan(scene)
                plan["branch_round"] = 3 + intervening
                prefix = [
                    {"attempt": index + 1, "action": "delegate_task",
                     "result": {"subtask_results": [{
                         "finish_result": {
                             "status": "done",
                             "result": "Lakehurst District",
                         },
                     }]}}
                    for index in range(intervening + 1)
                ]
                paths = discover_handoffs(
                    scene, plan, _clean_products(scene),
                    base_episode={"trajectory": prefix},
                )
                self.assertEqual(paths["child"]["older_task"], "parent")
                self.assertEqual(paths["child"]["parent_wave"], 1)

    def test_latest_report_can_disagree_with_multiple_older_waves(self):
        """Recording one conflict must not suppress other historical pairs."""
        from types import SimpleNamespace
        from unittest.mock import patch

        state = SimpleNamespace(
            current={"latest": SimpleNamespace(
                status="done", version=1, runtime={},
            )},
            detected_conflicts=[],
            public_status=lambda _task_id: "done",
        )
        earlier_rounds = (1, 3, 5)
        conflicts = [{
            "older_task": f"older_{round_number}",
            "latest_task": "latest",
            "subject": "Case-001",
            "attribute": "marker_code",
            "earlier_value": f"CODE-{round_number}",
            "latest_value": "WRONG",
            "older_round": round_number,
            "latest_round": 6,
            "older_fact_refs": [f"source_{round_number}:F001"],
            "latest_fact_refs": ["latest_source:F002"],
        } for round_number in earlier_rounds]
        with (
            patch("environment.untrusted_orchestration.scenario.history_conflicts.latest_history_disagreements",
                  return_value=conflicts),
            patch("environment.untrusted_orchestration.scenario.handoff_conflicts.handoff_disagreements",
                  return_value=[]),
            patch("environment.untrusted_orchestration.scenario.prior_observation.comparison_rows",
                  return_value=[]),
            patch("environment.untrusted_orchestration.scenario.history_conflicts.same_query_disagreements",
                  return_value=[]),
        ):
            ExecutionState._detect_history_report_conflicts(state)
            recorded = [
                event for event in state.detected_conflicts
                if event["kind"] == "cross_round_report_disagreement"
            ]
            self.assertEqual(
                {event["older_round"] for event in recorded},
                set(earlier_rounds),
            )
            self.assertEqual(len(recorded), 3)
            self.assertEqual(state.current["latest"].status, "disputed")
            self.assertTrue(
                state.current["latest"].runtime[
                    "history_conflict_recheck_requested"
                ],
            )
            # Re-observation without a newer report version must not
            # endlessly append the same three historical conflicts.
            ExecutionState._detect_history_report_conflicts(state)
            self.assertEqual(len(state.detected_conflicts), 3)

    def test_later_same_source_recheck_is_a_real_delegated_fault_candidate(self):
        """Method B augments the latest Main assignment, not just its report."""
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            discover_same_source_rechecks, patch_clean_episode, patch_scene,
        )
        from environment.untrusted_orchestration.reporting.result_text import (
            render_records,
        )
        scene, parent, child = _scene()
        scene["world"]["sources"]["parent_src"]["observations"].append({
            "values": {"case_id": "Case-002",
                       "entity_final": "Dunhaven District"},
            "fact_refs": ["parent_src:F002"],
            "observation_id": "parent_src.OBS002",
            "evidence_origin_id": "synthetic:parent002",
        })
        parent["operator_chain"] = [
            {"id": "parent_scan", "op": "scan", "inputs": [],
             "source_id": "parent_src"},
            {"id": "parent_filter", "op": "filter_eq",
             "inputs": ["parent_scan"], "field": "case_id",
             "value": "Case-001"},
            {"id": "parent_project", "op": "project",
             "inputs": ["parent_filter"],
             "fields": ["case_id", "entity_final"]},
        ]
        parent["outputs"] = {"typed_relation": ["case_id", "entity_final"]}
        child["input_schema"] = {"input_0": ["case_id", "entity_final"]}
        from environment.untrusted_orchestration.scenario.candidate_exploration import (
            _clean_products,
        )
        from environment.untrusted_orchestration.runtime.faults import relation_rows
        products = _clean_products(scene)
        old_report = {
            "task_instruction": "Find Case-001 in the image records.",
            "model": "model_1", "steps_taken": 3,
            "finish_result": {
                "status": "done",
                "result": render_records(relation_rows(products["parent"])),
                "summary": "Case-001: Lakehurst District.",
            },
        }
        new_report = {
            "task_instruction": "Find its year.",
            "model": "model_1", "steps_taken": 2,
            "finish_result": {
                "status": "done",
                "result": render_records(relation_rows(products["child"])),
                "summary": "Observed the requested year.",
            },
        }
        assignments = [
            {"task_instruction": "Find Case-001.", "context": "Inspect source.",
             "model": "model_1", "tools": ["Inspect"]},
        ]
        base = {"trajectory": [{
            "attempt": 1, "action": "delegate_task",
            "params": {"tasks": deepcopy(assignments)},
            "result": {"subtask_results": [old_report],
                       "total_subtasks": 1, "total_steps": 3},
        }, {
            "attempt": 2, "action": "delegate_task",
            "params": {"tasks": deepcopy(assignments)},
            "result": {"subtask_results": [new_report],
                       "total_subtasks": 1, "total_steps": 2},
        }]}
        plan = _round_plan(scene)
        discovered = discover_same_source_rechecks(
            scene, plan, products, base_episode=base,
            atomic_fault_kinds=["fabricated_claim"],
            max_parallel_tasks=4,
        )
        self.assertTrue(discovered, "No real repeated-source contradiction")
        patch = discovered[0]
        self.assertEqual(patch["older_wave_index"], 0)
        self.assertEqual(patch["new_wave_index"], 1)
        self.assertEqual(
            patch["new_task"]["source_ids"], parent["source_ids"],
        )
        augmented_scene = patch_scene(scene, patch)
        augmented_base = patch_clean_episode(
            base, scene, patch, max_parallel_tasks=4,
        )
        self.assertEqual(
            augmented_scene["macro_graph"]["reference_waves"][1],
            ["child", patch["new_task"]["task_id"]],
        )
        self.assertEqual(
            len(augmented_base["trajectory"][1]["params"]["tasks"]), 2,
        )
        self.assertEqual(
            len(augmented_base["trajectory"][1]["result"]["subtask_results"]), 2,
        )
        self.assertIn(
            "Repeat the same source query",
            augmented_base["trajectory"][1]["params"]["tasks"][1][
                "task_instruction"],
        )
        # A same-source disagreement is observable, but is not an
        # independent-source vote for either side.
        from environment.untrusted_orchestration.reporting.result_text import (
            render_records,
        )
        from environment.untrusted_orchestration.runtime.faults import relation_rows
        state = ExecutionState(augmented_scene, {})
        attempt = state.register("parent")
        earlier = deterministic_report(
            parent, attempt, instruction="Find the case",
            model_alias="model_1",
        )
        state.set_public_report("parent", earlier, round_number=1)
        state.register("child")
        new_task = patch["new_task"]
        fault = AtomicFault(
            patch["fault_kind"], new_task["task_id"], 2,
            variant_index=patch["fault_variant"],
        ).as_legacy_fault()
        faulted = state.register(new_task["task_id"], fault=fault)
        last = deterministic_report(
            new_task, faulted, instruction="Recheck the old case.",
            model_alias="model_1",
        )
        changed = faulted.runtime.get("communicated_rows")
        self.assertIsNotNone(changed)
        last["finish_result"]["result"] = render_records(changed)
        state.set_public_report(new_task["task_id"], last, round_number=2)
        state.cross_source_checks_enabled = True
        state.scheduling_state()
        found = [
            event for event in state.detected_conflicts
            if event.get("kind") == "cross_round_report_disagreement"
        ]
        self.assertTrue(found)
        self.assertFalse(found[0]["empirically_independent"])
        self.assertEqual(found[0]["older_round"], 1)
        self.assertEqual(found[0]["latest_round"], 2)

    def test_same_query_identity_change_across_long_history_is_repaired(self):
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            _clone_source_query, _public_result_set_differs,
        )
        from environment.untrusted_orchestration.scenario.history_conflicts import (
            same_query_disagreements,
        )
        scene, parent, _child = _scene()
        source = scene["world"]["sources"]["parent_src"]
        source["observations"].append({
            "values": {"case_id": "Case-002", "entity_final": "Other District"},
            "fact_refs": ["parent_src:F002"], "observation_id": "parent_src.OBS002",
            "evidence_origin_id": "synthetic:parent002",
        })
        parent["operator_chain"] = [
            {"id": "parent_scan", "op": "scan", "inputs": [], "source_id": "parent_src"},
            {"id": "parent_filter", "op": "filter_eq", "inputs": ["parent_scan"],
             "field": "case_id", "value": "Case-001"},
            {"id": "parent_project", "op": "project", "inputs": ["parent_filter"],
             "fields": ["case_id"]},
        ]
        parent["outputs"] = {"typed_relation": ["case_id"]}
        repeated = _clone_source_query(parent, 5)
        scene["macro_graph"]["tasks"].append(repeated)
        state = ExecutionState(scene, {})
        original = state.register("parent")
        old = deterministic_report(parent, original, instruction="Find the matching case.",
                                   model_alias="model_1")
        state.set_public_report("parent", old, round_number=1)
        wrong = state.register(
            repeated["task_id"],
            fault=AtomicFault("wrong_retrieval", repeated["task_id"], 4).as_legacy_fault(),
        )
        new = deterministic_report(repeated, wrong, instruction="Repeat the matching query.",
                                   model_alias="model_1")
        state.set_public_report(repeated["task_id"], new, round_number=4)
        self.assertTrue(_public_result_set_differs(
            relation_rows(original.product), relation_rows(wrong.product), old,
        ))
        events = same_query_disagreements(state)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["latest_round"] - events[0]["older_round"], 3)
        self.assertEqual(events[0]["comparison_kind"], "same_query_result_set")
        state.cross_source_checks_enabled = True
        self.assertIn(repeated["task_id"], state.scheduling_state()["repair"])
        self.assertTrue(state.unresolved_cited_conflicts())
        fixed = state.register(repeated["task_id"])
        state.set_public_report(
            repeated["task_id"],
            deterministic_report(repeated, fixed, instruction="Re-read the query.",
                                 model_alias="model_1"),
            round_number=5,
        )
        self.assertFalse(state.unresolved_cited_conflicts())
        self.assertEqual(same_query_disagreements(state), [])
        unknown = deepcopy(old)
        unknown["finish_result"]["result"] = "No usable candidate was identified."
        self.assertFalse(_public_result_set_differs(
            relation_rows(original.product), relation_rows(wrong.product), unknown,
        ))

    def test_recheck_result_is_consumed_and_stale_descendant_is_recomputed(self):
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            _clone_source_query, patch_scene,
        )
        scene, parent, child = _scene()
        scene["world"]["sources"]["parent_src"]["observations"].append({
            "values": {"case_id": "Case-002", "entity_final": "Dunhaven District"},
            "fact_refs": ["parent_src:F002"], "observation_id": "parent_src.OBS002",
            "evidence_origin_id": "synthetic:parent002",
        })
        parent["operator_chain"].insert(1, {
            "id": "parent_filter", "op": "filter_eq", "inputs": ["parent_scan"],
            "field": "case_id", "value": "Case-001",
        })
        parent["operator_chain"][-1]["inputs"] = ["parent_filter"]
        repeated = _clone_source_query(parent, 3)
        scene["macro_graph"]["reference_waves"] = [["parent"], [], ["child"]]
        patch = {
            "new_task": repeated, "older_task_id": "parent",
            "older_wave_index": 0, "new_wave_index": 1,
            "feeds_future_tasks": ["child"],
        }
        augmented = patch_scene(scene, patch)
        self.assertEqual(child["depends_on"], ["parent"])
        current_child = next(t for t in augmented["macro_graph"]["tasks"]
                             if t["task_id"] == "child")
        self.assertEqual(current_child["depends_on"], [repeated["task_id"]])
        self.assertEqual(current_child["operator_chain"], child["operator_chain"])
        state = ExecutionState(augmented, {})
        state.register("parent")
        state.register(
            repeated["task_id"],
            fault=AtomicFault("wrong_retrieval", repeated["task_id"], 2).as_legacy_fault(),
        )
        state.register("child")
        self.assertFalse(state.answer_matches_gold())
        state.register(repeated["task_id"])
        self.assertEqual(state.current["child"].status, "stale")
        self.assertTrue(state.invalidations)
        state.register("child")
        self.assertTrue(state.answer_matches_gold())

    def test_delayed_fault_is_consumed_before_late_clean_recheck(self):
        """One early error, real downstream use, late discovery, two-sided
        public verification, actual stale descendant and recomputation.
        No hidden fault label tells the Main which report to trust.
        """
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            _clone_source_query, patch_scene,
        )
        from environment.untrusted_orchestration.scenario.history_conflicts import (
            same_query_disagreements,
        )
        scene, parent, child = _scene()
        scene["world"]["sources"]["parent_src"]["observations"].append({
            "values": {"case_id": "Case-002", "entity_final": "Dunhaven District"},
            "fact_refs": ["parent_src:F002"],
            "observation_id": "parent_src.OBS002",
            "evidence_origin_id": "synthetic:parent002",
        })
        parent["operator_chain"].insert(1, {
            "id": "parent_filter", "op": "filter_eq",
            "inputs": ["parent_scan"], "field": "case_id",
            "value": "Case-001",
        })
        parent["operator_chain"][-1]["inputs"] = ["parent_filter"]
        recheck = _clone_source_query(parent, 4)
        scene["macro_graph"]["reference_waves"] = [
            ["parent"], ["child"], [],
        ]
        patch = {
            "new_task": recheck, "older_task_id": "parent",
            "older_wave_index": 0, "new_wave_index": 2,
            "feeds_future_tasks": [],
            "consumers_before_discovery": ["child"],
            "construction": "delayed_consumed_error_recovery",
        }
        scene = patch_scene(scene, patch)
        tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
        self.assertEqual(tasks["child"]["depends_on"], ["parent"])

        state = ExecutionState(scene, {})
        bad = state.register(
            "parent",
            fault=AtomicFault(
                "wrong_retrieval", "parent", 1,
            ).as_legacy_fault(),
        )
        state.set_public_report(
            "parent", deterministic_report(
                parent, bad, instruction="Find the source record.",
                model_alias="model_1",
            ), round_number=1,
        )
        consumer = state.register("child")
        self.assertEqual(
            relation_rows(consumer.product)[0]["values"]["observation_year"],
            2024,
        )
        state.set_public_report(
            "child", deterministic_report(
                child, consumer, instruction="Use the returned key.",
                model_alias="model_1",
            ), round_number=2,
        )
        witness = state.register(recheck["task_id"])
        state.set_public_report(
            recheck["task_id"], deterministic_report(
                recheck, witness, instruction="Repeat the same source query.",
                model_alias="model_1",
            ), round_number=3,
        )
        state.cross_source_checks_enabled = True
        self.assertIn(recheck["task_id"], state.scheduling_state()["repair"])
        conflicts = [
            e for e in state.detected_conflicts
            if e["kind"] == "cross_round_report_disagreement"
        ]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["older_round"], 1)
        self.assertEqual(conflicts[0]["latest_round"], 3)
        self.assertTrue(conflicts[0]["downstream_previously_executed"] is False)

        # First same-query retry confirms the clean witness. The
        # disagreement persists, so the earlier consumed side is checked.
        checked = state.register(recheck["task_id"])
        state.set_public_report(
            recheck["task_id"], deterministic_report(
                recheck, checked, instruction="Recheck newest.",
                model_alias="model_1",
            ), round_number=4,
        )
        self.assertIn("parent", state.scheduling_state()["repair"])
        self.assertEqual(len(state.detected_conflicts), 2)
        self.assertTrue(state.unresolved_cited_conflicts())

        corrected = state.register("parent")
        state.set_public_report(
            "parent", deterministic_report(
                parent, corrected, instruction="Verify old source value.",
                model_alias="model_1",
            ), round_number=5,
        )
        self.assertTrue(state.invalidations)
        self.assertEqual(state.current["child"].status, "stale")
        self.assertFalse(state.unresolved_cited_conflicts())
        updated = state.register("child")
        self.assertEqual(
            relation_rows(updated.product)[0]["values"]["observation_year"],
            2025,
        )
        self.assertEqual(len(state.attempts["child"]), 2)

    def test_poisoned_prefix_replays_real_descendant_and_rejects_clean_leak(self):
        """The intermediate Stage-3 dispatch may be reused only while its
        source values are grounded in the ACTUAL earlier public history.
        """
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            _clone_source_query, patch_clean_episode, patch_scene,
        )
        from environment.untrusted_orchestration.trajectory_builder import (
            _replay_observed_prefix,
        )
        scene, parent, child = _scene()
        source = scene["world"]["sources"]["parent_src"]
        source["observations"].append({
            "values": {"case_id": "Case-002", "entity_final": "Dunhaven District"},
            "fact_refs": ["parent_src:F002"],
            "observation_id": "parent_src.OBS002",
            "evidence_origin_id": "synthetic:parent002",
        })
        parent["operator_chain"].insert(1, {
            "id": "parent_filter", "op": "filter_eq",
            "inputs": ["parent_scan"], "field": "case_id",
            "value": "Case-001",
        })
        parent["operator_chain"][-1]["inputs"] = ["parent_filter"]
        filler = _clone_source_query(parent, 99)
        witness = _clone_source_query(parent, 4)
        scene["macro_graph"]["tasks"].append(filler)
        scene["macro_graph"]["reference_waves"] = [
            ["parent"], ["child"], [filler["task_id"]],
        ]
        clean_state = ExecutionState(scene, {})
        results = []
        for index, wave in enumerate(scene["macro_graph"]["reference_waves"]):
            batch = []
            for task_id in wave:
                task = clean_state.tasks[task_id]
                attempt = clean_state.register(task_id)
                batch.append(deterministic_report(
                    task, attempt,
                    instruction="Inspect the requested source.",
                    model_alias="model_1",
                ))
            results.append({
                "attempt": index + 1,
                "action": "delegate_task",
                "reasoning": "Collect authorized evidence.",
                "params": {"tasks": [{
                    "task_instruction": "Inspect the requested source.",
                    "context": "Use the authorized source.",
                    "model": "model_1", "tools": ["Inspect"],
                } for _ in wave]},
                "result": {
                    "subtask_results": batch,
                    "total_subtasks": len(batch),
                    "total_steps": sum(x["steps_taken"] for x in batch),
                    "all_done": True,
                },
            })
        base = {"task_id": scene["task_id"], "trajectory": results}
        patch = {
            "new_task": witness,
            "older_task_id": "parent", "older_wave_index": 0,
            "new_wave_index": 2, "feeds_future_tasks": [],
            "consumers_before_discovery": ["child"],
            "construction": "delayed_consumed_error_recovery",
        }
        augmented_base = patch_clean_episode(
            base, scene, patch, max_parallel_tasks=4,
        )
        augmented_scene = patch_scene(scene, patch)
        review_rows = []
        fault = AtomicFault(
            "wrong_retrieval", "parent", 1,
        ).as_legacy_fault()
        state, history, prefix, log = _replay_observed_prefix(
            augmented_scene, {}, augmented_base,
            branch_round=4, faults=[fault],
            fault_origin_wave_index=0,
            prefix_report_reviews=review_rows,
        )
        self.assertEqual(len(log), 1)
        self.assertEqual({r["task_id"] for r in review_rows},
                         {"parent", "child", witness["task_id"]})
        bad_child = relation_rows(state.current["child"].product)
        self.assertEqual(bad_child[0]["values"]["observation_year"], 2024)
        self.assertIn(
            "2024", str(prefix[1]["result"]["subtask_results"][0][
                "finish_result"]["result"]),
        )
        self.assertEqual(
            relation_rows(clean_state.current["child"].product)[0][
                "values"]["observation_year"], 2025,
        )
        self.assertEqual(history[0]["subtask_results"][0]["status"], "done")
        # A plan that still uses the clean-only name after the early fault
        # is *not* an acceptable delayed training history.
        leaked_base = deepcopy(augmented_base)
        leaked_base["trajectory"][1]["params"]["tasks"][0][
            "task_instruction"] = "Inspect Lakehurst District by name."
        with self.assertRaisesRegex(
            ValueError, "delayed_prefix_dispatch_references_unseen_clean_value",
        ):
            _replay_observed_prefix(
                augmented_scene, {}, leaked_base,
                branch_round=4, faults=[fault],
                fault_origin_wave_index=0,
            )

    def test_delayed_fault_candidate_keeps_old_dependency_edges(self):
        """Fault only the old task; a clean new witness is genuinely delegated."""
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            _clone_source_query, patch_scene,
        )
        scene, parent, child = _scene()
        patch = {
            "new_task": _clone_source_query(parent, 4),
            "older_task_id": "parent", "older_wave_index": 0,
            "new_wave_index": 2,
            "feeds_future_tasks": [],
            "consumers_before_discovery": ["child"],
            "construction": "delayed_consumed_error_recovery",
        }
        scene["macro_graph"]["reference_waves"] = [
            ["parent"], ["child"], [],
        ]
        new_scene = patch_scene(scene, patch)
        descendants = {
            t["task_id"]: t["depends_on"]
            for t in new_scene["macro_graph"]["tasks"]
        }
        self.assertEqual(descendants["child"], ["parent"])
        self.assertEqual(new_scene["macro_graph"]["reference_waves"][2],
                         [patch["new_task"]["task_id"]])

    def test_public_measurement_map_never_fills_omitted_rows(self):
        from environment.untrusted_orchestration.scenario.same_source_recheck import (
            public_query_rows,
        )
        task = {"outputs": {"typed_relation": ["entity_final", "metric_a"]}}
        report = {"finish_result": {
            "status": "done", "result": "Alderwick: 87, Lakehurst: 55.",
        }}
        self.assertEqual(public_query_rows(task, report), [
            {"entity_final": "Alderwick", "metric_a": 87},
            {"entity_final": "Lakehurst", "metric_a": 55},
        ])
        task["outputs"]["typed_relation"].reverse()
        self.assertEqual(public_query_rows(task, report)[0], {
            "entity_final": "Alderwick", "metric_a": 87,
        })
        report["finish_result"]["result"] = "Several cities have measurements."
        self.assertIsNone(public_query_rows(task, report))

    def test_report_fault_can_misassociate_existing_values_with_another_row(self):
        from environment.untrusted_orchestration.runtime.faults import run_attempt
        scene, parent, _child = _scene()
        scene["world"]["sources"]["parent_src"]["observations"].append({
            "values": {"case_id": "Case-002", "entity_final": "Dunhaven District"},
            "fact_refs": ["parent_src:F002"], "observation_id": "parent_src.OBS002",
            "evidence_origin_id": "synthetic:parent002",
        })
        parent["operator_chain"][-1]["fields"] = ["case_id", "entity_final"]
        parent["outputs"] = {"typed_relation": ["case_id", "entity_final"]}
        actual, runtime = run_attempt(
            parent, parents=[],
            sources={"parent_src": scene["world"]["sources"]["parent_src"]},
            fault=AtomicFault("fabricated_claim", "parent", 1).as_legacy_fault(),
        )
        self.assertEqual(actual.rows[0]["values"]["case_id"], "Case-001")
        communicated = runtime["communicated_rows"]
        self.assertNotEqual(communicated, relation_rows(actual))
        self.assertIn(runtime["report_fault_payload"]["replacement"], {
            value for row in actual.rows for value in row["values"].values()
        })
        second, changed = run_attempt(
            parent, parents=[],
            sources={"parent_src": scene["world"]["sources"]["parent_src"]},
            fault=AtomicFault("fabricated_claim", "parent", 1,
                              variant_index=1).as_legacy_fault(),
        )
        self.assertEqual(changed["report_fault_payload"]["row_index"], 1)
        self.assertEqual(changed["communicated_rows"][0], relation_rows(second)[0])
        self.assertNotEqual(changed["communicated_rows"][1], relation_rows(second)[1])


    def test_absent_parent_visible_key_is_not_a_candidate(self):
        scene, _, _ = _scene()
        base = {"trajectory": [{
            "action": "delegate_task", "attempt": 1,
            "result": {"subtask_results": [{
                "finish_result": {"status": "done",
                                  "result": "No matching observation.",
                                  "summary": "Nothing usable was identified."}
            }]},
        }]}
        self.assertEqual(discover_handoffs(
            scene, _round_plan(scene), _clean_products(scene),
            base_episode=base,
        ), {})


if __name__ == "__main__":
    unittest.main()
