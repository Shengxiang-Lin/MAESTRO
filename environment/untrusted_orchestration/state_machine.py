"""Versioned WorkProducts and O1-visible feedback in a branched episode.

This runtime never modifies Scene/World/Gold. It executes actual Macro
operator chains with the *latest parent Relations*. An upstream correction
invalidates all already-executed descendants and keeps their older reports
in history for causal audit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re

from environment.generate_question.program_ir import Relation
from .faults import Fault, relation_rows, run_attempt


@dataclass
class Attempt:
    task_id: str
    version: int
    status: str
    product: Relation | None
    available_fields: tuple[str, ...]
    used_parent_versions: dict[str, int]
    runtime: dict
    cost: int = 0


class ExecutionState:
    def __init__(self, scene: dict, reviewed_reports: dict[str, dict]):
        self.scene = scene
        self.tasks = {
            task["task_id"]: task for task in scene["macro_graph"]["tasks"]
        }
        self.reports = reviewed_reports
        self.current: dict[str, Attempt] = {}
        self.attempts: dict[str, list[Attempt]] = {
            tid: [] for tid in self.tasks
        }
        self.public_reports: dict[str, dict] = {}
        self.total_cost = 0
        self.invalidations: list[dict] = []

    def _dependents(self, task_id: str) -> set[str]:
        affected, queue = set(), [task_id]
        while queue:
            parent = queue.pop(0)
            for task in self.tasks.values():
                tid = task["task_id"]
                if parent in task["depends_on"] and tid not in affected:
                    affected.add(tid)
                    queue.append(tid)
        return affected

    def required_parent_fields(self, task: dict, parent: str) -> set[str]:
        """Conservative field-level readiness inferred from primitive ops."""
        parent_idx = task["depends_on"].index(parent)
        visible_schema = set(task["input_schema"]["input_" + str(parent_idx)])
        referenced = set()
        for op in task["operator_chain"]:
            for name in ("field", "left_field", "right_field",
                         "left_on", "right_on", "input_key_field"):
                if isinstance(op.get(name), str):
                    referenced.add(op[name])
            referenced.update(op.get("fields") or [])
        # A pure project with no directly mentioned fields on this parent
        # may still rely on its full upstream row. Safe fallback: block.
        return referenced & visible_schema or visible_schema

    def usable(self, task_id: str, *, required: set[str] | None = None) -> bool:
        current = self.current.get(task_id)
        if current is None or current.product is None:
            return False
        if current.status in {"missing", "disputed", "stale"}:
            return False
        if current.status == "partial":
            return bool(required is not None and
                        required <= set(current.available_fields))
        return True

    def ready(self, task_id: str) -> bool:
        task = self.tasks[task_id]
        return all(
            self.usable(parent, required=self.required_parent_fields(
                task, parent,
            )) for parent in task["depends_on"]
        )

    def register(
        self, task_id: str, *, fault: Fault | None = None,
        public_report: dict | None = None,
    ) -> Attempt:
        if task_id not in self.tasks:
            raise KeyError("unknown_runtime_macro")
        task = self.tasks[task_id]
        if not self.ready(task_id):
            raise ValueError("execution_requires_valid_parent_fields:" + task_id)
        parents = [self.current[parent].product
                   for parent in task["depends_on"]]
        versions = {
            parent: self.current[parent].version
            for parent in task["depends_on"]
        }
        # Privately read only sources within this task's frozen scope.
        sources = {
            sid: self.scene["world"]["sources"][sid]
            for sid in task["source_ids"]
        }
        product, metadata = run_attempt(
            task, parents=parents, sources=sources, fault=fault,
        )
        previous = self.current.get(task_id)
        version = (previous.version + 1 if previous else 1)
        cost = 1 + 2 * len(task["source_ids"])
        attempt = Attempt(
            task_id=task_id, version=version,
            status=metadata["status"], product=product,
            available_fields=tuple(metadata["available_fields"]),
            used_parent_versions=versions, runtime=metadata,
            cost=cost,
        )
        self.current[task_id] = attempt
        self.attempts[task_id].append(attempt)
        self.total_cost += cost
        if previous is not None:
            for child in sorted(self._dependents(task_id)):
                existing = self.current.get(child)
                if existing is not None and existing.status != "stale":
                    existing.status = "stale"
                    self.invalidations.append({
                        "replaced_task": task_id,
                        "stale_task": child,
                        "old_parent_version": previous.version,
                        "new_parent_version": version,
                    })
        if public_report is not None:
            self.public_reports[task_id] = public_report
        return attempt

    def set_public_report(self, task_id: str, report: dict) -> None:
        if task_id not in self.current:
            raise ValueError("report_for_unexecuted_task")
        self.public_reports[task_id] = report

    def public_status(self, task_id: str) -> str:
        """The *delivered report* can be partial even when outer done=true."""
        report = self.public_reports.get(task_id)
        if report is None:
            return "not_returned"
        finish = report.get("finish_result")
        if not isinstance(finish, dict):
            return "missing"
        return str(finish.get("status", "unknown"))

    def visible_discrepancy(self, task_id: str) -> str | None:
        """Find a contradiction using only facts *reported* to Main.

        No Gold, hidden fault flag, unreported source observations, or
        private typed value is used to force a verify label.
        """
        current = self.current.get(task_id)
        if current is None:
            return None
        if current.product is None or self.public_status(task_id) == "missing":
            return (
                "The returned Sub-Agent message did not contain a usable "
                "Finish Result for the assigned investigation."
            )
        if self.public_status(task_id) == "partial":
            report = self.public_reports[task_id]["finish_result"]
            return (
                "The returned investigation is only partial: " +
                report.get("summary", "Some required evidence is missing.")
            )
        task = self.tasks[task_id]
        if not any(step["op"] == "join" for step in task["operator_chain"]):
            return None
        parents = task["depends_on"]
        if len(parents) != 2 or not all(p in self.public_reports
                                        for p in parents):
            return None
        left = self.current[parents[0]].product
        right = self.current[parents[1]].product
        if left is None or right is None:
            return None
        join = next(step for step in task["operator_chain"]
                    if step["op"] == "join")
        key_left, key_right = join["left_on"], join["right_on"]
        if key_left not in left.columns or key_right not in right.columns:
            return None

        # Compare the newly returned marker/ID with the *visible* parent
        # candidates, including a marker-to-candidate mapping if the joined
        # WorkProduct projected away the join key.
        visible_parent_text = {
            tid: " ".join(str(self.public_reports[tid]
                              .get("finish_result", {}).get(k, ""))
                          for k in ("result", "summary"))
            for tid in parents
        }
        left_keys = {r["values"][key_left] for r in left.rows}
        right_keys = {r["values"][key_right] for r in right.rows}
        shared = left_keys & right_keys
        result_text = " ".join(
            str(self.public_reports.get(task_id, {})
                .get("finish_result", {}).get(k, ""))
            for k in ("result", "summary")
        )
        if not shared or not result_text:
            return None
        for row in current.product.rows:
            returned_keys = []
            if key_left in row["values"]:
                returned_keys.append(row["values"][key_left])
            elif "marker_code" in row["values"] and "marker_code" in left.columns:
                returned_keys.extend(
                    item["values"][key_left] for item in left.rows
                    if item["values"]["marker_code"] == row["values"]["marker_code"]
                )
            if not returned_keys:
                continue
            for value in returned_keys:
                if value in shared:
                    continue
                marker = str(row["values"].get("marker_code", value))
                if (marker not in result_text
                        or str(value) not in visible_parent_text[parents[0]]
                        or not any(str(k) in visible_parent_text[parents[1]]
                                   for k in shared)):
                    # No teacher-only discrepancy: the identifier must be
                    # inferable from earlier Main-visible Sub-Agent reports.
                    continue
                return (
                    "The new joined result reports " + marker +
                    ", but earlier Sub-Agent findings associate that "
                    "candidate with a different set of records. Reconcile "
                    "the already returned candidate sets before keyed lookup."
                )
        return None

    def scheduling_state(self) -> dict:
        disputed = []
        for tid, item in self.current.items():
            discrepancy = self.visible_discrepancy(tid)
            if discrepancy and item.status == "done":
                # Only *observable* contradictions block descendants.
                item.status = "disputed"
                item.runtime["visible_issue"] = discrepancy
            if item.status in {"stale", "disputed", "missing", "partial"}:
                disputed.append(tid)
        ready, blocked = [], []
        for tid in self.tasks:
            if tid in self.current:
                continue
            if self.ready(tid):
                ready.append(tid)
            else:
                blocked.append(tid)
        return {
            "repair": disputed,
            "ready": ready,
            "blocked": blocked,
        }

    def effective_answer(self) -> list[dict] | None:
        tid = self.scene["macro_graph"]["terminal_task_id"]
        attempt = self.current.get(tid)
        if attempt is None or attempt.product is None or (
            not self.usable(tid, required=set(
                self.tasks[tid]["outputs"]["typed_relation"]
            ))
        ):
            return None
        return [row["values"] for row in attempt.product.rows]

    def answer_matches_gold(self) -> bool:
        actual = self.effective_answer()
        if actual is None:
            return False
        def norm(rows):
            return sorted(json.dumps(x, ensure_ascii=False, sort_keys=True)
                          for x in rows)
        return norm(actual) == norm(self.scene["gold_answer"])

    def audit(self) -> dict:
        return {
            "attempt_versions": {
                tid: [{
                    "version": a.version,
                    "status": a.status,
                    "used_parent_versions": a.used_parent_versions,
                    "runtime": a.runtime, "cost": a.cost,
                    "rows": relation_rows(a.product),
                } for a in attempts]
                for tid, attempts in self.attempts.items() if attempts
            },
            "invalidations": list(self.invalidations),
            "total_cost": self.total_cost,
        }


def deterministic_report(
    task: dict, attempt: Attempt, *,
    instruction: str, model_alias: str,
    reviewed: dict | None = None,
) -> dict:
    """Sub-Agent feedback from THIS attempt, never future/clean replay."""
    actual_rows = relation_rows(attempt.product)
    original_rows = (reviewed or {}).get("answer_with_provenance")
    if (reviewed is not None and attempt.status == "done"
            and actual_rows == original_rows):
        finish = deepcopy(reviewed["finish_result"])
        narrative = reviewed.get("trace_summary", {}).get("narrative") or (
            finish["summary"]
        )
        mode = "reviewed_clean"
    elif attempt.status == "missing":
        finish = None
        narrative = "The assigned evidence acquisition did not return an inspectable deliverable."
        mode = "executed_missing"
    else:
        result = json.dumps(
            [{"values": row["values"], "source_fact_refs":
              row["source_fact_refs"]} for row in actual_rows],
            ensure_ascii=False,
        )
        available = ", ".join(attempt.available_fields) or "(none)"
        unresolved = ", ".join(attempt.runtime.get("missing_fields", []))
        summary = (
            "The current investigation returned " +
            str(len(actual_rows)) + " record(s) for " +
            available + "."
        )
        if unresolved:
            summary += " Still unresolved: " + unresolved + "."
        finish = {
            "status": attempt.status,
            "result": result,
            "summary": summary,
        }
        narrative = (
            "Inspected assigned source/preceding records and produced "
            "the above field values with the attached evidence references."
        )
        mode = "deterministic_executed"
    return {
        "subtask_index": 0,
        "task_instruction": instruction,
        "model": model_alias,
        "done": True,         # execution terminated, not evidence correctness
        "steps_taken": len(task["operator_chain"]),
        "statistics": {"max_steps": 30},
        "finish_result": finish,
        "trace_summary": narrative,
        "cost": 0.0,          # offline model/tool costs unknown
        "_report_mode": mode, # stripped before Main-visible publication
    }


def public_history_item(result: dict, index: int) -> dict:
    finish = result.get("finish_result") or {}
    return {
        "status": finish.get("status", "missing"),
        "instruction": result["task_instruction"],
        "model": result["model"], "steps_taken": result["steps_taken"],
        "max_steps": result.get("statistics", {}).get("max_steps", 30),
        "cost": result.get("cost", 0.0),
        "result": finish.get("result", ""),
        "summary": finish.get("summary", ""),
        "trace_summary": result.get("trace_summary", ""),
    }
