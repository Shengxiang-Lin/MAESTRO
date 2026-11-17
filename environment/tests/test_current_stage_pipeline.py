"""Regression: current Stage-1 Task Plan must feed Stage 2, 3 and 4.

All evidence is synthetic and locally replayed; no remote LLM or YAGO dump.
"""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from environment.tests.current_stage1 import current_scene
from environment.generate_question.planning.spec import spec_issues
from environment.generate_question.auditing.quality import validate_replay
from environment.implement_subtask.realize import realize_scene
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.faults import Fault


def accepted_reports(scene):
    rows = realize_scene(scene)["macro_tasks"]
    return [{
        **deepcopy(row),
        "finish_result": {
            "status": "done",
            "result": "Observed: " + json.dumps(row["answer"], ensure_ascii=False),
            "summary": "Grounded observations " + json.dumps(row["answer"], ensure_ascii=False),
        },
        "report_authoring": {"status": "accepted"},
    } for row in rows]


class CurrentStagePipelineTests(unittest.TestCase):
    def test_no_dangling_stage1_imports_in_any_current_environment_module(self):
        """Deleting old modules must not strand less frequently used consumers."""
        import ast
        from pathlib import Path
        env = Path(__file__).resolve().parents[1]
        prefix = "environment.generate_question."
        failures = []
        for file in env.rglob("*.py"):
            if "tests" in file.relative_to(env).parts:
                continue
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append((node.lineno, node.module))
                elif isinstance(node, ast.Import):
                    imports.extend((node.lineno, alias.name) for alias in node.names)
            for line, name in imports:
                if name.startswith(prefix):
                    path = env / "generate_question" / name[len(prefix):].replace(".", "/")
                    if not path.with_suffix(".py").is_file() and not (
                        path / "__init__.py"
                    ).is_file():
                        failures.append((str(file.relative_to(env)), line, name))
        self.assertEqual(failures, [])

    def test_only_task_plan_is_supported(self):
        scene = current_scene(0)
        self.assertEqual(scene["problem_spec"]["reasoning_pattern"], "task_plan")
        self.assertEqual(validate_replay(scene)["status"], "valid")
        retired = deepcopy(scene["problem_spec"])
        retired["reasoning_pattern"] = "typed_knowledge_v4"
        self.assertIn("unsupported_stage1_reasoning_pattern", spec_issues(retired))

    def test_stage2_exposes_source_scoped_workproducts(self):
        scene = current_scene(1, "medium")
        before = deepcopy(scene)
        teacher = realize_scene(scene)
        self.assertEqual(scene, before)
        self.assertEqual(len(teacher["macro_tasks"]), scene["macro_graph"]["macro_task_count"])
        self.assertTrue(all(row["trace_summary"]["steps"] for row in teacher["macro_tasks"]))
        self.assertTrue(all(row["answer_with_provenance"] for row in teacher["macro_tasks"]))

    def test_stage3_and_stage4_consume_frozen_current_scene(self):
        scene = current_scene(3)
        reports = accepted_reports(scene)
        episode, turns, private, status = build_episode(
            scene, reports, freeze_dispatch=True,
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertTrue(turns)
        self.assertTrue(private)
        self.assertEqual(episode["trajectory"][-1]["action"], "complete")
        target = scene["macro_graph"]["reference_waves"][0][0]
        recovered, damaged_turns, audit = build_branch(
            scene, reports, episode, branch_round=2,
            faults=[Fault(target, "missing")],
        )
        self.assertEqual(audit["outcome"], "recovered", audit)
        self.assertTrue(damaged_turns)
        self.assertEqual(recovered["trajectory"][-1]["action"], "complete")


    def test_many_row_code_owned_delivery_never_truncates(self):
        from environment.implement_subtask.reporting import code_owned_result
        from environment.untrusted_orchestration.report_contract import (
            render_delivered_result,
        )
        rows = [{
            "values": {"entity_final": "Entity-" + str(i),
                       "observation_year": 2026},
            "source_fact_refs": ["observation_log:F" + str(i).zfill(5)],
        } for i in range(220)]
        record = {
            "result_encoding": "typed_evidence_json_v1",
            "answer": [row["values"] for row in rows],
            "answer_with_provenance": rows,
            "output_schema": ["entity_final", "observation_year"],
        }
        first = code_owned_result(record)
        second = render_delivered_result({
            "result_encoding": "typed_evidence_json_v1",
            "actual_delivered_rows": rows,
        })
        self.assertGreater(len(first), 10000)
        self.assertEqual(json.loads(first), rows)
        self.assertEqual(json.loads(second), rows)

    def test_stage2_current_result_contains_all_rows_and_fact_refs(self):
        from environment.implement_subtask.reporting import code_owned_result
        scene = current_scene(4, "medium")
        rows = realize_scene(scene)["macro_tasks"]
        self.assertTrue(all(
            row.get("result_encoding") == "typed_evidence_json_v1"
            for row in rows
        ))
        bounded = next(
            row for row in rows if len(code_owned_result(row)) < 5000
        )
        result = json.loads(code_owned_result(bounded))
        self.assertEqual(len(result), len(bounded["answer"]))
        self.assertEqual(
            [item["values"] for item in result],
            bounded["answer"],
        )
        self.assertEqual(
            [item["source_fact_refs"] for item in result],
            [item["source_fact_refs"]
             for item in bounded["answer_with_provenance"]],
        )

    def test_structured_stage2_reports_feed_validated_clean_stage3(self):
        from environment.implement_subtask.reporting import code_owned_result
        from environment.untrusted_orchestration.trusted_prefix import (
            compile_trusted_prefixes,
        )
        scene = current_scene(8, "medium")
        reports = []
        for item in realize_scene(scene)["macro_tasks"]:
            reports.append({
                **deepcopy(item),
                "finish_result": {
                    "status": "done",
                    "result": code_owned_result(item),
                    "summary": "Reported only the source-scoped finding.",
                },
                "report_authoring": {"status": "accepted"},
            })
        expected, turns, _, status = build_episode(
            scene, reports, freeze_dispatch=True,
        )
        self.assertEqual(status["status"], "accepted", status)
        self.assertTrue(turns)
        compiled, review = compile_trusted_prefixes(
            [scene], reports, max_rounds=16,
            sub_model="qwen3-8b", model_alias="model_1",
            clean_episodes={scene["task_id"]: expected},
        )
        self.assertEqual(review[0]["status"], "accepted", review)
        self.assertEqual(
            compiled[scene["task_id"]]["clean_reference_id"],
            expected["clean_reference_id"],
        )

    def test_stage4_retry_contract_keeps_stage2_citations(self):
        from environment.untrusted_orchestration.state_machine import (
            ExecutionState,
        )
        from environment.untrusted_orchestration.report_contract import (
            build_report_contract, render_delivered_result,
        )
        from environment.untrusted_orchestration.faults import relation_rows
        scene = current_scene(7, "medium")
        rows = accepted_reports(scene)
        reviewed = {r["macro_task_id"]: r for r in rows}
        runtime = ExecutionState(scene, reviewed)
        tid = scene["macro_graph"]["reference_waves"][0][0]
        attempt = runtime.register(tid)
        contract = build_report_contract(
            runtime.tasks[tid], attempt, "Read the assigned source.",
            result_encoding=reviewed[tid]["result_encoding"],
        )
        self.assertEqual(
            json.loads(render_delivered_result(contract)),
            relation_rows(attempt.product),
        )
        self.assertEqual(
            json.loads(contract["fact_spine"]["delivery"]["result_text"]),
            relation_rows(attempt.product),
        )

    def test_clean_reference_is_stable_and_fails_closed_when_tampered(self):
        from environment.untrusted_orchestration.trusted_prefix import (
            validate_clean_episode,
        )
        scene = current_scene(5, "medium")
        reports = accepted_reports(scene)
        expected, _turns, _audit, status = build_episode(
            scene, reports, freeze_dispatch=True,
        )
        self.assertEqual(status["status"], "accepted")
        self.assertTrue(expected["clean_reference_id"].startswith("clean-"))
        duplicate, _, _, _ = build_episode(
            scene, reports, freeze_dispatch=True,
        )
        self.assertEqual(
            expected["clean_reference_id"], duplicate["clean_reference_id"],
        )
        validate_clean_episode(expected, duplicate)
        altered = deepcopy(duplicate)
        altered["trajectory"][0]["result"]["subtask_results"][0][
            "finish_result"]["result"] = "untrusted replacement"
        with self.assertRaisesRegex(ValueError, "clean_report_mismatch"):
            validate_clean_episode(expected, altered)



if __name__ == "__main__":
    unittest.main()
