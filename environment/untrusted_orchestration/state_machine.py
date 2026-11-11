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
from environment.implement_subtask.realize import _fact_index


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
        self.fact_index = _fact_index(scene["world"])
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
        """Back-propagate field liveness through the operator chain.

        A downstream take/project of an *already sorted* relation may need
        only the projected label, not the parent's no-longer-used sort key.
        In contrast, join, filter, sort and aggregation consume additional
        fields even when those fields are absent from the final output.
        """
        parent_idx = task["depends_on"].index(parent)
        # In the frozen Macro IR, external_inputs contains the *actual*
        # upstream program node ID (e.g. "ordered" or "m3_op02").
        # input_schema uses "input_0" only as the typed contract label.
        external = task["external_inputs"][parent_idx]
        declared = set(task["input_schema"]["input_" + str(parent_idx)])
        steps = task["operator_chain"]
        needed = {task["program_node_id"]:
                  set(task["outputs"]["typed_relation"])}
        def add(node, fields):
            needed.setdefault(node, set()).update(fields)
        for step in reversed(steps):
            output_need = needed.get(step["id"], set())
            op = step["op"]
            inputs = step["inputs"]
            if not inputs:
                continue
            if op in {"project", "take", "sort", "filter_eq", "filter_gte",
                      "unique", "compare_fields", "lookup_by_key",
                      "attach_evidence", "derive_comparison",
                      "derive_percentage", "group_count", "count"}:
                used = set(output_need)
                used.update(
                    step[k] for k in ("field", "left_field", "right_field",
                                      "input_key_field")
                    if isinstance(step.get(k), str)
                )
                if op == "sort":
                    used.add(step["field"])
                if op in {"derive_comparison", "derive_percentage"}:
                    used.update((step["left_field"], step["right_field"]))
                if op == "group_count":
                    used.add(step["field"])
                if op == "unique":
                    # A unique operation following project deduplicates
                    # only projected columns, not every original input field.
                    # If unique reads the full external relation directly,
                    # all its columns are semantically relevant.
                    if inputs[0] == external:
                        used.update(declared)
                    else:
                        producer = next(
                            (prior for prior in steps
                             if prior["id"] == inputs[0]),
                            None,
                        )
                        if producer and producer["op"] == "project":
                            used.update(producer["fields"])
                if op == "count":
                    # Count depends on row-set completeness, NOT values of
                    # unrelated columns. rows_complete is checked in usable().
                    used.clear()
                if op == "project":
                    used &= set(step["fields"])
                if op == "lookup_by_key":
                    # New source columns are not inputs; preserve only
                    # existing upstream fields plus the key.
                    used = {step["input_key_field"]} | output_need
                if op == "attach_evidence":
                    used.discard("evidence_refs")
                add(inputs[0], used)
            elif op == "join":
                # Both inputs are needed for the equality predicate; each
                # passes through only its own fields.
                add(inputs[0], output_need | {step["left_on"]})
                add(inputs[1], output_need | {step["right_on"]})
            else:
                # Fail closed for future operators we cannot analyze.
                for dep in inputs:
                    add(dep, declared)
        return needed.get(external, declared) & declared

    def usable(self, task_id: str, *, required: set[str] | None = None) -> bool:
        current = self.current.get(task_id)
        if current is None or current.product is None:
            return False
        if current.status in {"missing", "disputed", "stale"} or (
            current.runtime.get("report_fault_kind") == "report_missing"
        ):
            return False
        if current.status == "partial":
            return (
                bool(current.runtime.get("rows_complete", True))
                and required is not None
                and required <= set(current.available_fields)
            )
        return True

    def repair_required(self, task_id: str) -> bool:
        """A partial handoff is repaired only when needed downstream."""
        item = self.current[task_id]
        if item.runtime.get("report_fault_kind") == "report_missing":
            return True
        if item.status != "partial":
            return item.status in {"missing", "disputed", "stale"}
        if task_id == self.scene["macro_graph"]["terminal_task_id"]:
            return not self.usable(
                task_id,
                required=set(self.tasks[task_id]["outputs"]["typed_relation"]),
            )
        for dependent in self.tasks.values():
            if task_id in dependent["depends_on"] and (
                dependent["task_id"] not in self.current
                or self.current[dependent["task_id"]].status == "stale"
            ):
                if not self.usable(
                    task_id,
                    required=self.required_parent_fields(dependent, task_id),
                ):
                    return True
        return False

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
            evidence_index=self.fact_index,
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

    @staticmethod
    def _public_text(report: dict) -> str:
        finish = (report or {}).get("finish_result") or {}
        return " ".join(
            str(finish.get(field, "")) for field in ("result", "summary")
        )

    @staticmethod
    def _public_field_values(report: dict, field: str) -> set[str]:
        """Read a field only from the *delivered* result, never Oracle rows.

        Natural Sub-Agent prose often mentions KEY1-94705, while poisoned
        results contain a JSON list of {values: {...}}. We require explicit,
        unique identifiers and avoid inventing links from unreported data.
        """
        finish = (report or {}).get("finish_result") or {}
        body = str(finish.get("result", ""))
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            rows = [(row.get("values") if isinstance(row, dict)
                     and "values" in row else row) for row in parsed]
            values = {
                str(row[field]) for row in rows if isinstance(row, dict)
                and field in row and row[field] is not None
            }
            if values:
                return values
        text = ExecutionState._public_text(report)
        patterns = {
            "record_key_1": r"\bKEY1-\d+\b",
            "record_key_2": r"\bKEY2-\d+\b",
            "marker_code": r"\bMK-\d+\b",
            "candidate_id": r"\bCase-\d+\b",
            "entry_label": r"\bEntry-\d+\b",
        }
        pattern = patterns.get(field)
        return set(re.findall(pattern, text)) if pattern else set()

    def _visible_lookup_issue(self, task_id: str) -> str | None:
        task = self.tasks[task_id]
        lookups = [step for step in task["operator_chain"]
                   if step["op"] == "lookup_by_key"]
        if not lookups or len(task["depends_on"]) != 1:
            return None
        parent_id = task["depends_on"][0]
        if parent_id not in self.public_reports:
            return None
        result = self.public_reports.get(task_id)
        for op in lookups:
            parent_keys = self._public_field_values(
                self.public_reports[parent_id], op["input_key_field"],
            )
            output_keys = self._public_field_values(
                result, op["source_key_field"],
            )
            # When two different field names represent the same key value,
            # the returned result may expose only the upstream field.
            if not output_keys:
                output_keys = self._public_field_values(
                    result, op["input_key_field"],
                )
            if (len(parent_keys) == 1 and len(output_keys) == 1
                    and parent_keys != output_keys):
                return (
                    "The returned keyed lookup identifies " +
                    next(iter(output_keys)) +
                    ", while the earlier Sub-Agent supplied the lookup "
                    "key " + next(iter(parent_keys)) +
                    ". Verify the record identity before using this result."
                )
        return None

    def _visible_passthrough_issue(self, task_id: str) -> str | None:
        """Detect changed identifiers through public one-row transformations.

        Only applies to source-free transforms (project/filter/take/unique
        and sorting), and only if both reports explicitly have exactly one
        value for the same identifier field.
        """
        task = self.tasks[task_id]
        if (task["source_ids"] or len(task["depends_on"]) != 1 or
                any(step["op"] not in {
                    "project", "filter_eq", "filter_gte", "take",
                    "sort", "unique", "attach_evidence",
                } for step in task["operator_chain"])):
            return None
        parent = task["depends_on"][0]
        if parent not in self.public_reports:
            return None
        for field in ("record_key_1", "record_key_2", "marker_code",
                      "candidate_id", "entry_label"):
            incoming = self._public_field_values(
                self.public_reports[parent], field,
            )
            outgoing = self._public_field_values(
                self.public_reports[task_id], field,
            )
            if len(incoming) == len(outgoing) == 1 and incoming != outgoing:
                return (
                    "The derived report changed " + field + " from " +
                    next(iter(incoming)) + " to " + next(iter(outgoing)) +
                    " without a source lookup or transformation that "
                    "can justify that change."
                )
        return None

    def _visible_report_internal_conflict(self, task_id: str) -> str | None:
        """Compare RESULT with SUMMARY using ONLY the delivered report.

        The clean private WorkProduct cannot be used to label a report
        erroneous, since the Main did not observe that private result.
        """
        finish = (self.public_reports.get(task_id) or {}).get(
            "finish_result"
        ) or {}
        try:
            rows = json.loads(finish.get("result"))
        except (TypeError, ValueError):
            return None
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not rows:
            return None
        if not isinstance(rows[0], dict):
            return None
        if "values" in rows[0] and isinstance(rows[0]["values"], dict):
            rows = [row["values"] for row in rows
                    if isinstance(row, dict) and "values" in row]
        summary = str(finish.get("summary") or "")
        for match in re.finditer(
            r"\bsummary records?\s+([A-Za-z][A-Za-z0-9_ ]+)"
            r"\s*=\s*([^.;,]+)", summary, flags=re.I,
        ):
            field = match.group(1).strip().replace(" ", "_")
            claimed = match.group(2).strip()
            values = {str(row[field]) for row in rows
                      if isinstance(row, dict) and field in row}
            if values and claimed not in values:
                return (
                    "The delivered result and summary disagree about "
                    + field.replace("_", " ") + ": result lists "
                    + ", ".join(sorted(values)[:4])
                    + ", but the summary says " + claimed
                    + ". Check the field against its source."
                )
        return None

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
        report_issue = self._visible_report_internal_conflict(task_id)
        if report_issue:
            return report_issue
        lookup_issue = self._visible_lookup_issue(task_id)
        if lookup_issue:
            return lookup_issue
        passthrough_issue = self._visible_passthrough_issue(task_id)
        if passthrough_issue:
            return passthrough_issue
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
            if self.repair_required(tid):
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
