"""Versioned WorkProducts and O1-visible feedback in a branched episode.

This runtime never modifies Scene/World/Gold. It executes actual Macro
operator chains with the *latest parent Relations*. An upstream correction
invalidates all already-executed descendants and keeps their older reports
in history for causal audit.
"""
from __future__ import annotations

from copy import deepcopy
from environment.implement_subtask.public_surface import reviewed_public_view
from dataclasses import dataclass
import json
import re

from environment.generate_question.execution.program_ir import Relation
from .faults import Fault, relation_rows, run_attempt
from .visible_claim_extraction import extract_visible_claims
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
        # Source origins are *metadata only*: only FactRefs literally cited
        # in returned Main-visible Result rows may be mapped to these roots.
        self.origin_by_ref = {
            ref: observation["evidence_origin_id"]
            for source in scene["world"]["sources"].values()
            for observation in source.get("observations", [])
            if isinstance(observation.get("evidence_origin_id"), str)
            for ref in observation.get("fact_refs", [])
            if isinstance(ref, str)
        }
        self.tasks = {
            task["task_id"]: task for task in scene["macro_graph"]["tasks"]
        }
        self.reports = reviewed_reports
        self.current: dict[str, Attempt] = {}
        self.attempts: dict[str, list[Attempt]] = {
            tid: [] for tid in self.tasks
        }
        self.public_reports: dict[str, dict] = {}
        self.public_report_rounds: dict[str, int] = {}
        self.cross_source_checks_enabled = False
        self.detected_conflicts: list[dict] = []
        self.checked_explicit_claim_pairs: set[tuple] = set()
        self.total_cost = 0
        self.invalidations: list[dict] = []
        self.revalidations: list[dict] = []

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

    def communicated_product(self, task_id: str) -> Relation | None:
        """Relation actually passed to a downstream Sub-Agent.

        An intentionally fabricated Sub-Agent result can disagree with
        what its own tool really executed. Downstream consumers use the
        communicated value, NEVER a hidden clean executor result.
        """
        item = self.current[task_id]
        if item.runtime.get("report_fault_kind") == "report_missing":
            return None
        delivered = item.runtime.get("communicated_rows")
        if delivered is None:
            return item.product
        if item.product is None:
            raise ValueError("communicated_rows_without_execution")
        return Relation(
            item.product.columns,
            tuple({
                "values": deepcopy(row["values"]),
                "support": tuple(row["source_fact_refs"]),
            } for row in delivered),
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
        parents = [self.communicated_product(parent)
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
        previous_delivered = (
            self.communicated_product(task_id)
            if previous is not None else None
        )
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
            current_delivered = self.communicated_product(task_id)
            previous_rows = relation_rows(previous_delivered)
            current_rows = relation_rows(current_delivered)
            changed = previous_rows != current_rows
            affected = sorted(self._dependents(task_id))
            if not changed:
                # A disputed but reconfirmed value should NOT cause a
                # cascade of redundant lookups. Preserve prior executed
                # attempts and record semantic equivalence privately.
                self.revalidations.append({
                    "task_id": task_id,
                    "old_version": previous.version,
                    "new_version": version,
                    "downstream_preserved": [
                        child for child in affected
                        if child in self.current
                        and self.current[child].status != "stale"
                    ],
                })
            else:
                for child in affected:
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

    def set_public_report(self, task_id: str, report: dict, *,
                          round_number: int | None = None) -> None:
        if task_id not in self.current:
            raise ValueError("report_for_unexecuted_task")
        self.public_reports[task_id] = report
        self.public_report_rounds[task_id] = (
            round_number if round_number is not None
            else len(self.public_reports)
        )

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
    def _public_result_rows(report: dict) -> list[dict] | None:
        """Return rows only when the *delivered Result* explicitly has them.

        Do not reconstruct a cross-task join from private WorkProducts just
        because some identifiers happen to occur in public free-form prose.
        """
        finish = (report or {}).get("finish_result") or {}
        try:
            rows = json.loads(finish.get("result", ""))
        except (TypeError, ValueError):
            return None
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            return None
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            values = row.get("values", row)
            if not isinstance(values, dict):
                return None
            normalized.append(values)
        return normalized

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

    def _visible_done_field_gap(self, task_id: str) -> str | None:
        """A DONE report can still visibly omit a required typed field.

        Completeness is determined from the *delivered* Result and frozen
        public field contract; never compare with private clean rows. Only
        fields actually required by a pending dependent (or terminal goal)
        create a repair obligation.
        """
        if self.public_status(task_id) != "done":
            return None
        result_rows = self._public_result_rows(
            self.public_reports.get(task_id)
        )
        if not result_rows:
            return None
        required = set()
        for child in self.tasks.values():
            if (task_id in child["depends_on"]
                    and {"external_inputs", "input_schema", "operator_chain",
                         "outputs"} <= set(child)
                    and (child["task_id"] not in self.current
                         or self.current[child["task_id"]].status == "stale")):
                required.update(self.required_parent_fields(
                    child, task_id,
                ))
        if task_id == self.scene["macro_graph"]["terminal_task_id"]:
            required.update(self.tasks[task_id]["outputs"]["typed_relation"])
        missing = sorted(
            field for field in required
            if all(row.get(field) is None for row in result_rows)
        )
        if not missing:
            return None
        return (
            "The delivered Finish Result is marked done, but it has no "
            "usable values for the required fields: "
            + ", ".join(missing) + ". The missing evidence is needed "
            "before the next dependent investigation can proceed."
        )

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

    def _visible_scope_issue(self, task_id: str) -> str | None:
        """Check explicit task-specific filters against a delivered row.

        Scope/edition/year conditions must be recognizable in the ACTUAL
        Main-facing task instruction. A private operator alone cannot
        create a claim that Main knew the expected year or source policy.
        """
        if self.public_status(task_id) != "done":
            return None
        task = self.tasks[task_id]
        report = self.public_reports.get(task_id, {})
        instruction = str(report.get("task_instruction") or "").casefold()
        if not instruction:
            return None
        rows = self._public_result_rows(report)
        if not rows:
            return None
        for step in task.get("operator_chain", []):
            if step.get("op") not in {"filter_eq", "filter_gte"}:
                continue
            field, limit = step.get("field"), step.get("value")
            if (not isinstance(field, str)
                    or type(limit) not in {str, int, float}
                    or field.casefold().replace("_", " ") not in
                       instruction.replace("_", " ")
                    or str(limit).casefold() not in instruction):
                continue
            for row in rows:
                actual = row.get(field)
                if actual is None:
                    continue
                if step["op"] == "filter_eq":
                    valid = actual == limit
                else:
                    valid = (
                        type(actual) in (int, float)
                        and type(limit) in (int, float)
                        and actual >= limit
                    )
                if not valid:
                    return (
                        "The delivered record reports "
                        + field.replace("_", " ") + " = " + str(actual)
                        + ", although the assigned investigation explicitly "
                        "requires " + step["op"].replace("_", " ")
                        + " " + str(limit)
                        + ". Recheck the returned record's requested scope."
                    )
        return None

    def _visible_arithmetic_issue(self, task_id: str) -> str | None:
        """Check one source-free derivation against explicitly delivered inputs.

        No private evaluator input, hidden Gold or unreported intermediate
        output may be used. Skip aggregation/count (unknown completeness)
        and multi-step chains whose private intermediate values are not
        reconstructible from Main's received reports.
        """
        task = self.tasks[task_id]
        chain = task.get("operator_chain") or []
        parents = task.get("depends_on") or []
        if (task.get("source_ids") or len(parents) != 1
                or len(chain) != 1
                or chain[0].get("op") != "derive_arithmetic"
                or chain[0].get("inputs") != task.get("external_inputs")
                or parents[0] not in self.public_reports):
            return None
        if self.public_status(parents[0]) != "done":
            return None
        parent_rows = self._public_result_rows(self.public_reports[parents[0]])
        result_rows = self._public_result_rows(self.public_reports.get(task_id))
        if (not parent_rows or not result_rows
                or len(parent_rows) != len(result_rows)):
            return None
        step = chain[0]
        left_name, right_name = step["left_field"], step["right_field"]
        output_field = step["output_field"]
        operation = step["operation"]
        for parent_row, returned in zip(parent_rows, result_rows):
            left, right = parent_row.get(left_name), parent_row.get(right_name)
            observed = returned.get(output_field)
            if not all(type(v) in (int, float) for v in (
                left, right, observed
            )):
                continue
            if operation == "absolute_difference":
                expected = abs(left - right)
            elif operation == "ratio" and right != 0:
                expected = round(left / right, step["round_digits"])
            else:
                continue
            if abs(float(expected) - float(observed)) > 1e-9:
                return (
                    "The delivered arithmetic result for " + output_field
                    + " is " + str(observed) + ", but the returned inputs "
                    + left_name + " = " + str(left) + " and "
                    + right_name + " = " + str(right)
                    + " imply " + str(expected)
                    + " under the requested " + operation.replace("_", " ")
                    + ". Recompute the derivation using these inputs."
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
        # An inaccessible *private* WorkProduct is not evidence that
        # the Main saw a missing Result. Only delivered feedback counts.
        if self.public_status(task_id) == "missing":
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
        field_gap = self._visible_done_field_gap(task_id)
        if field_gap:
            return field_gap
        report_issue = self._visible_report_internal_conflict(task_id)
        if report_issue:
            return report_issue
        scope_issue = self._visible_scope_issue(task_id)
        if scope_issue:
            return scope_issue
        lookup_issue = self._visible_lookup_issue(task_id)
        if lookup_issue:
            return lookup_issue
        passthrough_issue = self._visible_passthrough_issue(task_id)
        if passthrough_issue:
            return passthrough_issue
        arithmetic_issue = self._visible_arithmetic_issue(task_id)
        if arithmetic_issue:
            return arithmetic_issue
        if not any(step["op"] == "join" for step in task["operator_chain"]):
            return None
        parents = task["depends_on"]
        if len(parents) != 2 or not all(p in self.public_reports
                                        for p in parents):
            return None
        left = self._public_result_rows(self.public_reports[parents[0]])
        right = self._public_result_rows(self.public_reports[parents[1]])
        joined = self._public_result_rows(self.public_reports.get(task_id))
        if left is None or right is None or joined is None:
            return None
        join = next(step for step in task["operator_chain"]
                    if step["op"] == "join")
        key_left, key_right = join["left_on"], join["right_on"]
        if not left or not right or not all(
            key_left in row for row in left
        ) or not all(key_right in row for row in right):
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
        left_keys = {r[key_left] for r in left}
        right_keys = {r[key_right] for r in right}
        shared = left_keys & right_keys
        result_text = " ".join(
            str(self.public_reports.get(task_id, {})
                .get("finish_result", {}).get(k, ""))
            for k in ("result", "summary")
        )
        if not shared or not result_text:
            return None
        for row in joined:
            returned_keys = []
            if key_left in row:
                returned_keys.append(row[key_left])
            elif "marker_code" in row and all("marker_code" in x for x in left):
                returned_keys.extend(
                    item[key_left] for item in left
                    if item["marker_code"] == row["marker_code"]
                )
            if not returned_keys:
                continue
            for value in returned_keys:
                if value in shared:
                    continue
                marker = str(row.get("marker_code", value))
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

    def _explicit_report_claims(self) -> list[dict]:
        """Parse only delivered Result claims with a single known origin.

        Multi-origin Join outputs and unknown provenance are *not*
        independent witnesses, even if they cite several source records.
        """
        rows = []
        for tid, report in self.public_reports.items():
            if tid not in self.tasks or self.public_status(tid) != "done":
                continue
            task = self.tasks[tid]
            scopes = task.get("source_ids", [])
            fallback = scopes[0] if len(scopes) == 1 else None
            claims, _ = extract_visible_claims(
                report,
                output_fields=task.get("outputs", {}).get("typed_relation", []),
                source_id=fallback,
            )
            for claim in claims:
                if (claim["surface"] != "result"
                        or not claim["declared_output_field"]
                        or claim["subject"].startswith("source:")):
                    continue
                refs = claim.get("fact_refs") or []
                origins = {self.origin_by_ref.get(ref) for ref in refs}
                if len(origins) != 1 or None in origins:
                    continue
                rows.append({
                    **claim, "task_id": tid,
                    "evidence_origin_id": next(iter(origins)),
                    "observed_round": self.public_report_rounds.get(tid, 0),
                })
        return rows

    def _detect_explicit_cross_report_disagreements(self) -> None:
        """Late, cited, independent claims can justify one targeted recheck.

        No Gold value is consulted, and no side is declared correct.
        Independent sources merely having different candidates or a
        complementary Join are *not* enough to trigger this detector.
        """
        if not self.cross_source_checks_enabled:
            return
        by_claim = {}
        for claim in self._explicit_report_claims():
            key = (claim["subject"], claim["attribute"], claim["valid_scope"])
            by_claim.setdefault(key, []).append(claim)
        for (subject, field, scope), items in sorted(by_claim.items()):
            ordered = sorted(items, key=lambda row: (
                row["observed_round"], row["task_id"],
            ))
            for i, earlier in enumerate(ordered):
                for later in ordered[i + 1:]:
                    left, right = earlier["task_id"], later["task_id"]
                    if (left == right
                            or earlier["evidence_origin_id"]
                               == later["evidence_origin_id"]
                            or left in self._dependents(right)
                            or right in self._dependents(left)
                            or (type(earlier["value"]), earlier["value"])
                               == (type(later["value"]), later["value"])):
                        continue
                    # Same-round independent reports are actionable as
                    # soon as both arrive. Delayed disagreements are also
                    # actionable; only ACTUAL prior downstream consumers
                    # may later be invalidated. Neither side is named wrong
                    # from private Gold.
                    prev = self.current.get(left)
                    if prev is None or prev.status not in {"done", "disputed"}:
                        continue
                    consumed = any(
                        child in self.current
                        and self.current[child].status != "stale"
                        and self.current[child].used_parent_versions.get(left)
                            == prev.version
                        for child in self._dependents(left)
                    )
                    pair = (left, right, subject, field, scope)
                    if pair in self.checked_explicit_claim_pairs:
                        continue
                    self.checked_explicit_claim_pairs.add(pair)
                    message = (
                        "Two separately cited sources report different "
                        + field.replace("_", " ") + " values for "
                        + subject + ": " + str(earlier["value"])
                        + " versus " + str(later["value"])
                        + (". The earlier value was already used "
                           "downstream; recheck its cited source record."
                           if consumed else
                           ". Reconcile these cited reports before using "
                           "either conflicting claim downstream.")
                    )
                    prev.status = "disputed"
                    prev.runtime["visible_issue"] = message
                    prev.runtime["cross_source_recheck_requested"] = True
                    self.detected_conflicts.append({
                        "kind": "explicit_independent_claim_disagreement",
                        "compared_parent_tasks": [left, right],
                        "recheck_task_id": left,
                        "claim": {"subject": subject, "attribute": field,
                                  "valid_scope": scope},
                        "observed_values": [
                            earlier["value"], later["value"],
                        ],
                        "cited_fact_refs": [
                            earlier["fact_refs"], later["fact_refs"],
                        ],
                        "at_public_round": later["observed_round"],
                        "conflict_timing": (
                            "simultaneous" if
                            earlier["observed_round"] ==
                            later["observed_round"] else "delayed"
                        ),
                        "downstream_previously_executed": consumed,
                    })
                    return

    def unresolved_cited_conflicts(self) -> bool:
        """After rechecking, do two PUBLIC independent claims still disagree?

        A re-execution that returns exactly the same value is *not*
        automatically a resolution. Neither source's hidden Gold is used.
        """
        relevant = [
            row for row in self.detected_conflicts
            if row.get("kind") == "explicit_independent_claim_disagreement"
        ]
        if not relevant:
            return False
        claims = self._explicit_report_claims()
        for event in relevant:
            left, right = event["compared_parent_tasks"]
            wanted = event["claim"]
            matching = {}
            for claim in claims:
                if (claim["task_id"] in (left, right)
                        and all(claim[key] == value
                                for key, value in (
                                    ("subject", wanted["subject"]),
                                    ("attribute", wanted["attribute"]),
                                    ("valid_scope", wanted["valid_scope"]),
                                ))):
                    matching.setdefault(claim["task_id"], set()).add(
                        (type(claim["value"]).__name__, repr(claim["value"]))
                    )
            if (left not in matching or right not in matching
                    or matching[left] != matching[right]):
                return True
        return False

    def observable_issues(self) -> list[dict]:
        """Current decision's report-grounded anomalies, not fault labels.

        The simulator retains these as PRIVATE risk-supervision diagnostics.
        The Main still receives only the original O1 Sub-Agent history.
        """
        issues = []
        for tid in sorted(self.public_reports):
            if tid not in self.current:
                continue
            status = self.public_status(tid)
            item = self.current[tid]
            if status == "missing":
                kind = "missing_delivery"
                reason = self.visible_discrepancy(tid)
            elif status == "partial":
                kind = "partial_delivery"
                reason = self.visible_discrepancy(tid)
            elif item.status == "stale":
                # A previous public parent report was superseded; the old
                # derived result is not newly verified by that replacement.
                kind = "stale_downstream_result"
                reason = "A previously used upstream report was replaced."
            elif item.runtime.get("cross_source_recheck_requested"):
                kind = "cross_source_conflict"
                reason = item.runtime.get("visible_issue")
            elif self._visible_done_field_gap(tid):
                kind = "done_but_incomplete"
                reason = self._visible_done_field_gap(tid)
            elif self._visible_report_internal_conflict(tid):
                kind = "intra_report_conflict"
                reason = self._visible_report_internal_conflict(tid)
            elif self._visible_scope_issue(tid):
                kind = "scope_constraint_mismatch"
                reason = self._visible_scope_issue(tid)
            elif self._visible_lookup_issue(tid):
                kind = "lookup_key_disagreement"
                reason = self._visible_lookup_issue(tid)
            elif self._visible_passthrough_issue(tid):
                kind = "derived_identifier_disagreement"
                reason = self._visible_passthrough_issue(tid)
            elif self._visible_arithmetic_issue(tid):
                kind = "arithmetic_inconsistency"
                reason = self._visible_arithmetic_issue(tid)
            else:
                # An unexplained discrepancy is not a license to read
                # the private Attempt or invent an outside verifier.
                continue
            if reason:
                issues.append({
                    "task_id": tid, "kind": kind, "evidence": reason,
                    "reported_round": self.public_report_rounds.get(tid),
                    "delivery_status": status,
                })
        return issues

    def scheduling_state(self) -> dict:
        self._detect_explicit_cross_report_disagreements()
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
            "revalidations": list(self.revalidations),
            "detected_conflicts": list(self.detected_conflicts),
            "public_report_rounds": dict(self.public_report_rounds),
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
        # Shared Stage-3/4 clean presentation; never sanitize a
        # faulted or adversarially altered report here.
        view = reviewed_public_view(reviewed)
        finish = deepcopy(view["finish_result"])
        narrative = view.get("trace_summary", {}).get("narrative") or (
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
