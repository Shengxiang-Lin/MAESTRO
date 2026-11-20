"""Stage-4B-style conflict seeds from genuinely earlier keyed handoffs.

Discovery happens *before* selecting an atomic fault. An upstream Sub-Agent's
publicly available, cited lookup key is the anchor. A later keyed lookup
promises to preserve that key; a real erroneous record can contradict it.
No prior-wave source scan, invented source or future key is needed.
"""
from __future__ import annotations

from copy import deepcopy

from environment.untrusted_orchestration.runtime.faults import relation_rows, run_attempt
from ..reporting.result_text import delivered_records, render_records


FAULTS = {"wrong_retrieval", "wrong_entity_binding", "fabricated_claim",
          "legacy_wrong_evidence"}


def _scalar(value):
    return type(value) in (str, int, float, bool) and (
        not isinstance(value, str) or bool(value.strip())
    )


def discover_handoffs(scene: dict, round_plan: dict, products: dict,
                      base_episode: dict | None = None) -> dict[str, dict]:
    """Identify previous *legally returned* facts independently of any fault.

    Require one public parent key, carried by a cited executed relation,
    and an ordinary keyed lookup whose output preserves that exact key.
    Never compare arbitrary fields or infer identity from a future result.
    """
    graph = scene["macro_graph"]
    tasks = {task["task_id"]: task for task in graph["tasks"]}
    earlier = {
        tid: index
        for index, wave in enumerate(graph["reference_waves"])
        if index < round_plan["branch_round"] - 2
        for tid in wave
    }
    visible = {}
    if base_episode is not None:
        for action in base_episode["trajectory"]:
            if action.get("action") != "delegate_task":
                continue
            wave = action["attempt"] - 1
            if wave not in range(round_plan["branch_round"] - 2):
                continue
            for tid, report in zip(graph["reference_waves"][wave],
                                   action["result"]["subtask_results"]):
                visible[tid] = report
    found = {}
    for target_id in round_plan["incoming_task_ids"]:
        task = tasks[target_id]
        parents = task.get("depends_on", [])
        for step in task.get("operator_chain", []):
            if step.get("op") != "lookup_by_key":
                continue
            source_id = step["source_id"]
            source = scene["world"]["sources"][source_id]
            if (source.get("access_policy", {}).get("mode") != "keyed_lookup"
                    or source.get("access_policy", {}).get("key_field")
                    != step.get("source_key_field")):
                continue
            key = step["input_key_field"]
            if (key != step.get("source_key_field")
                    or key not in task["outputs"]["typed_relation"]):
                continue
            external = step["inputs"][0] if step["inputs"] else None
            indices = [
                index for index, name in enumerate(task["external_inputs"])
                if name == external
            ]
            if len(indices) != 1:
                continue
            parent_id = parents[indices[0]]
            if parent_id not in earlier:
                continue
            rows = relation_rows(products[parent_id])
            refs = {ref for row in rows for ref in row["source_fact_refs"]}
            values = {row["values"].get(key) for row in rows
                      if _scalar(row["values"].get(key))}
            if len(values) != 1 or not refs:
                continue
            expected = next(iter(values))
            if not rows or any(row["values"].get(key) != expected for row in rows):
                continue
            if base_episode is not None:
                report = visible.get(parent_id, {})
                finish = report.get("finish_result") or {}
                if (finish.get("status") != "done"
                        or not any(str(expected) in str(part) for part in (
                            finish.get("result"), finish.get("summary"),
                            report.get("trace_summary")))):
                    continue
            clean = relation_rows(products[target_id])
            if not clean or any(
                row["values"].get(key) != expected for row in clean
            ):
                continue
            found[target_id] = {
                "kind": "earlier_keyed_handoff",
                "older_task": parent_id, "latest_task": target_id,
                "key_field": key,
                "source_id": source_id,
                "parent_wave": earlier[parent_id] + 1,
            }
            break
    return found


def fault_breaks_handoff(scene: dict, plan: dict, fault, products: dict) -> bool:
    """Offline preflight only: does the actual injected result break the key?"""
    tasks = {task["task_id"]: task for task in scene["macro_graph"]["tasks"]}
    tid = plan["latest_task"]
    task = tasks[tid]
    if fault.atom_kind not in FAULTS:
        return False
    field = plan["key_field"]
    older = relation_rows(products[plan["older_task"]])
    expected = {row["values"].get(field) for row in older}
    if len(expected) != 1:
        return False
    try:
        altered, runtime = run_attempt(
            task, parents=[products[parent] for parent in task["depends_on"]],
            sources={sid: scene["world"]["sources"][sid]
                     for sid in task["source_ids"]}, fault=fault,
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return False
    delivered = runtime.get("communicated_rows")
    rows = delivered if delivered is not None else relation_rows(altered)
    # One off-key delivered row is enough; a report-only summary
    # disagreement is not masqueraded as a changed WorkProduct.
    return bool(rows) and any(
        field in row["values"] and row["values"][field] != next(iter(expected))
        for row in rows
    )


def expose_prior_handoff(state, task_entries: list[dict],
                         episode_prefix: list[dict], plan: dict) -> dict:
    """Make an already reported parent value explicitly parseable to Main.

    Preserve the old Sub-Agent's real executed values/citations, not a
    constructed verifier. The original report must already state that value.
    """
    older, latest, field = (plan["older_task"], plan["latest_task"],
                            plan["key_field"])
    old = state.public_reports.get(older)
    parent = state.current.get(older)
    if old is None or parent is None or parent.product is None:
        return {"status": "ineligible", "reason": "no_executed_prior_handoff"}
    rows = relation_rows(parent.product)
    values = {row["values"].get(field) for row in rows}
    if (len(values) != 1 or not all(row["source_fact_refs"] for row in rows)
            or not all(_scalar(v) for v in values)):
        return {"status": "ineligible", "reason": "prior_handoff_not_unique_or_cited"}
    expected = next(iter(values))
    finish = old.get("finish_result") or {}
    if (finish.get("status") != "done"
            or not any(str(expected) in str(part) for part in (
                finish.get("result"), finish.get("summary"),
                old.get("trace_summary")))):
        return {"status": "ineligible", "reason": "prior_key_not_public"}
    prior_round = state.public_report_rounds[older]
    if prior_round >= state.public_report_rounds.get(latest, 0):
        return {"status": "ineligible", "reason": "prior_handoff_not_earlier"}
    # The previous report already declared this value. Only its formatting
    # changes so a Main-visible comparison is possible.
    rendered = deepcopy(old)
    rendered["finish_result"]["result"] = render_records(rows)
    state.set_public_report(older, rendered, round_number=prior_round)
    action = episode_prefix[prior_round - 1]
    ids = [tid for tid, round_no in state.public_report_rounds.items()
           if round_no == prior_round]
    slot = ids.index(older)
    action["result"]["subtask_results"][slot] = rendered
    from ..state_machine import public_history_item
    task_entries[prior_round - 1]["subtask_results"][slot] = (
        public_history_item(rendered, slot)
    )
    state.tasks[latest] = {
        **state.tasks[latest],
        "history_handoff_parent_task_id": older,
        "history_handoff_key_field": field,
    }
    return {
        "status": "generated", "reason": "previous_public_keyed_handoff",
        "task_id": older, "target_task_id": latest,
        "prior_round": prior_round, "field": field,
        "construction": "earlier_executed_keyed_handoff",
    }


def restore_handoff_contract(scene: dict, anchor: dict | None) -> dict:
    """Restore a public keyed-query relationship, never an expected answer.

    expose_prior_handoff establishes this link on the runtime task copy.
    Offline replay must restore the same link; its claims are still checked
    against the reports actually received at the replay decision.
    """
    if not anchor or anchor.get("construction") != "earlier_executed_keyed_handoff":
        return scene
    old, target, field = (anchor["task_id"], anchor["target_task_id"], anchor["field"])
    tasks = {task["task_id"]: task for task in scene["macro_graph"]["tasks"]}
    if (old not in tasks or target not in tasks
            or old not in tasks[target]["depends_on"]
            or field not in tasks[old]["outputs"]["typed_relation"]
            or field not in tasks[target]["outputs"]["typed_relation"]
            or not any(
                node["op"] == "lookup_by_key"
                and node["input_key_field"] == field
                and node["source_key_field"] == field
                for node in tasks[target]["operator_chain"]
            )):
        raise ValueError("invalid_public_handoff_contract")
    result = deepcopy(scene)
    for task in result["macro_graph"]["tasks"]:
        if task["task_id"] == target:
            task["history_handoff_parent_task_id"] = old
            task["history_handoff_key_field"] = field
    return result


def handoff_disagreements(state) -> list[dict]:
    """Read only public cited Results, with a pre-existing keyed lookup contract."""
    conflicts = []
    for tid, task in state.tasks.items():
        parent = task.get("history_handoff_parent_task_id")
        field = task.get("history_handoff_key_field")
        if not parent or not field:
            continue
        if (state.public_status(parent) != "done"
                or state.public_status(tid) != "done"
                or state.public_report_rounds[parent]
                    >= state.public_report_rounds[tid]):
            continue
        old = delivered_records(
            (state.public_reports[parent].get("finish_result") or {}).get("result")
        )
        new = delivered_records(
            (state.public_reports[tid].get("finish_result") or {}).get("result")
        )
        if not old or not new:
            continue
        old_values = {r["values"].get(field) for r in old}
        if len(old_values) != 1 or not all(
            row.get("source_fact_refs") for row in old + new
        ):
            continue
        expected = next(iter(old_values))
        if not _scalar(expected):
            continue
        # Every returned row is addressed by the upstream single key.
        # The report must explicitly claim a different value to conflict.
        mismatched = [r for r in new if (
            field in r["values"]
            and type(r["values"][field]) is type(expected)
            and r["values"][field] != expected
        )]
        if not mismatched:
            continue
        item = mismatched[0]
        conflicts.append({
            "older_task": parent, "latest_task": tid,
            "subject": "keyed_handoff:" + field,
            "attribute": field, "earlier_value": expected,
            "latest_value": item["values"][field],
            "older_round": state.public_report_rounds[parent],
            "latest_round": state.public_report_rounds[tid],
            "older_fact_refs": old[0]["source_fact_refs"],
            "latest_fact_refs": item["source_fact_refs"],
            "comparison_kind": "earlier_keyed_handoff",
        })
    return conflicts
