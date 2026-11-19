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
                    "".join(json.dumps(row, ensure_ascii=False) + "\\n"
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
            self.assertGreater(
                summary.get("historical_anchor_status_distribution", {})
                       .get("generated", 0), 0,
                summary.get("historical_anchor_reason_distribution", {}),
            )
            self.assertGreater(
                summary.get("cross_round_report_conflicts_detected", 0), 0,
                summary.get("historical_anchor_reason_distribution", {}),
            )

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
