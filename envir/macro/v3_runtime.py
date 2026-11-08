"""Source-scoped reliable Sub-Agent execution for a controlled synthetic world.

Every feedback packet is *derived* from accessible simulated observations or
previously completed task outputs. No private world, hidden Gold, or future
subtask result is passed to an executor. Summary Trace is an event summary,
never fabricated model chain-of-thought.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

from .graph import compile_waves


class MissingEvidence(ValueError):
    pass


def _select(rows: list[dict], scope: str) -> list[dict]:
    return [dict(r) for r in rows if scope == "all" or r["case_id"] == scope]


def _join(events: list[dict], records: list[dict]) -> list[dict]:
    by_id = {r["case_id"]: r for r in records}
    result = []
    for e in events:
        if e["case_id"] not in by_id:
            raise MissingEvidence("unmatched_case:" + e["case_id"])
        row = {**e, **by_id[e["case_id"]]}
        result.append(row)
    if not result:
        raise MissingEvidence("empty_case_set")
    return result


def _result(intent: str, inputs: dict) -> dict:
    if intent == "comparison":
        a = inputs["case_a"]["cases"]
        b = inputs["case_b"]["cases"]
        if len(a) != 1 or len(b) != 1:
            raise MissingEvidence("comparison_needs_one_scoped_case_per_operand")
        aa, bb = a[0], b[0]
        winner = aa if aa["units"] > bb["units"] else bb
        return {
            "higher_workload_case_id": winner["case_id"],
            "difference_units": abs(aa["units"] - bb["units"]),
            "compared_case_ids": [aa["case_id"], bb["case_id"]],
        }
    if intent in {"timeline", "reconciliation"}:
        events = inputs["event"]["events"]
        reports = {r["case_id"]: r for r in
                   inputs["independent"]["observations"]}
        differences = []
        for e in events:
            report = reports.get(e["case_id"])
            if report is None:
                raise MissingEvidence("missing_independent_case")
            differences.append({
                "case_id": e["case_id"],
                "difference_minutes": abs(e["event_minute"] -
                                          report["reported_minute"]),
            })
        if intent == "timeline":
            winner = max(differences,
                         key=lambda d: (d["difference_minutes"], d["case_id"]))
            return {
                "largest_gap_case_id": winner["case_id"],
                "gap_minutes": winner["difference_minutes"],
            }
        else:
            return {
                "discrepant_cases": [
                    d for d in differences if d["difference_minutes"] > 10
                ],
                "tolerance_minutes": 10,
            }
    cases = inputs["matched_cases"]["cases"]
    rules = inputs.get("rules", {}).get("rule")
    capacity = {r["case_id"]: r["available_units"] for r in
                inputs.get("capacity", {}).get("capacities", [])}
    if intent in {"compliance", "impact", "feasibility"} and rules is None:
        raise MissingEvidence("missing_rules")
    if intent in {"prioritization", "mitigation", "feasibility"} and (
        not {c["case_id"] for c in cases} <= set(capacity)
    ):
        raise MissingEvidence("missing_capacity_for_cases")
    if intent == "compliance":
        violations = []
        for case in cases:
            reasons = []
            if case["status"] != rules["valid_status"]:
                reasons.append("registration_status")
            if case["event_minute"] > rules["closing_minute"]:
                reasons.append("time_window")
            if reasons:
                violations.append({"case_id": case["case_id"], "reasons": reasons})
        return {"violations": violations, "noncompliant_count": len(violations)}
    if intent == "impact":
        affected = [c for c in cases
                    if c["status"] != rules["valid_status"]
                    and c["units"] >= rules["min_affected_units"]]
        return {
            "affected_case_ids": [c["case_id"] for c in affected],
            "affected_case_count": len(affected),
            "total_affected_units": sum(c["units"] for c in affected),
        }
    if intent == "feasibility":
        eligible = [
            c["case_id"] for c in cases if (
                c["status"] == rules["valid_status"]
                and c["event_minute"] <= rules["closing_minute"]
                and capacity[c["case_id"]] >= c["units"]
            )
        ]
        return {"feasible_case_ids": eligible, "feasible_count": len(eligible)}
    if intent == "prioritization":
        scores = [{
            "case_id": c["case_id"],
            "score": c["units"] * c["urgency"] - capacity[c["case_id"]],
        } for c in cases]
        best = max(scores, key=lambda p: (p["score"], p["case_id"]))
        return {"first_case_id": best["case_id"], "priority_score": best["score"]}
    if intent == "mitigation":
        shortfalls = [{
            "case_id": c["case_id"],
            "shortfall_units": max(0, c["units"] - capacity[c["case_id"]]),
        } for c in cases]
        top = max(shortfalls, key=lambda p: (p["shortfall_units"], p["case_id"]))
        return {"highest_shortfall_case_id": top["case_id"],
                "shortfall_units": top["shortfall_units"],
                "action": "review_capacity_reallocation"}
    raise ValueError("unsupported final intent: " + intent)


def _read_source(task: dict, source_map: dict, events: list) -> tuple[str, dict]:
    ids = task["authorized_source_ids"]
    if len(ids) != 1:
        raise MissingEvidence("source_task_has_no_unique_authorized_source")
    source_id = ids[0]
    if source_id not in source_map:
        raise MissingEvidence("source_unavailable:" + source_id)
    record = source_map[source_id]
    if record["source_id"] != source_id:
        raise MissingEvidence("source_identity_mismatch")
    events.append({
        "action": "ReadSimulatedSource",
        "source_id": source_id,
        "title": record["title"],
        "modality": record["modality"],
        "result": "source_observations_read",
    })
    return source_id, copy.deepcopy(record["observations"])



def _summary(kind: str, body: dict, task: dict) -> str:
    """Describe *actual produced data*, never just restate a work goal."""
    if kind == "event_investigation":
        return f"Established {len(body['events'])} scoped event observations."
    if kind == "records_audit":
        return f"Audited {len(body['records'])} registered case entries."
    if kind == "rules_review":
        rule = body["rule"]
        return (f"Confirmed accepted status '{rule['valid_status']}', "
                f"closing minute {rule['closing_minute']}, and minimum "
                f"affected workload {rule['min_affected_units']}.")
    if kind == "capacity_review":
        return f"Established available capacity for {len(body['capacities'])} entries."
    if kind == "independent_review":
        return f"Collected {len(body['observations'])} independent time observations."
    if kind in {"case_matching", "integrated_case_audit"}:
        return f"Matched {len(body['cases'])} observed cases to registered records."
    if kind == "final_assessment":
        return "Completed the requested assessment: " + json.dumps(
            body, ensure_ascii=False, sort_keys=True)
    raise ValueError("unknown summary work kind")


def evaluate(plan: dict, world: dict, *,
             hidden_sources: frozenset[str] = frozenset()) -> dict:
    graph = plan["macro_graph"]
    tasks = {t["task_id"]: t for t in graph["tasks"]}
    if compile_waves(list(tasks.values())) != graph["reference_waves"]:
        raise MissingEvidence("invalid_reference_schedule")
    provided = {k: v for k, v in world["sources"].items()
                if k not in hidden_sources}
    completed, feedback = {}, {}
    for wave in graph["reference_waves"]:
        if len(wave) > 10:
            raise ValueError("max_concurrency_exceeded")
        # A wave sees *only* completed earlier waves. Cross-wave results
        # are transferred by named typed input/output contracts.
        state_before = dict(completed)
        for tid in wave:
            task = tasks[tid]
            inputs, proof = {}, set()
            events = []
            for slot, binding in task["inputs"].items():
                parent = binding["from_task"]
                if parent not in state_before:
                    raise MissingEvidence("future_or_unavailable_upstream_output")
                expected_type = binding["type"]
                parent_field = binding["output_slot"]
                if tasks[parent]["outputs"].get(parent_field) != expected_type:
                    raise MissingEvidence("cross_task_output_type_drift")
                inputs[slot] = state_before[parent]["answer"][parent_field]
                proof.update(state_before[parent]["evidence_ids"])
                events.append({
                    "action": "UseCompletedWorkProduct",
                    "from_task": parent,
                    "input_slot": slot,
                    "type": expected_type,
                    "result": "named_work_product_bound",
                })
            kind = task["work_type"]
            scope = task["scope_key"]
            own_source = None
            if task["authorized_source_ids"]:
                own_source, obs = _read_source(task, provided, events)
                proof.add(own_source)
            else:
                obs = None
            if kind == "event_investigation":
                body = {"events": _select(obs["events"], scope)}
            elif kind == "records_audit":
                records = _select(obs["records"], scope)
                if "event_scope" in inputs:
                    ids = {e["case_id"] for e in inputs["event_scope"]["events"]}
                    records = [r for r in records if r["case_id"] in ids]
                body = {"records": records}
            elif kind == "rules_review":
                body = {"rule": obs["rule"]}
            elif kind == "capacity_review":
                items = obs["capacities"]
                if "matched_cases" in inputs:
                    scopes = {c["case_id"] for c in inputs["matched_cases"]["cases"]}
                    items = [r for r in items if r["case_id"] in scopes]
                body = {"capacities": items}
            elif kind == "independent_review":
                items = obs["observations"]
                if "event" in inputs:
                    scopes = {e["case_id"] for e in inputs["event"]["events"]}
                    items = [r for r in items if r["case_id"] in scopes]
                body = {"observations": items}
            elif kind == "case_matching":
                body = {
                    "cases": _join(inputs["event"]["events"],
                                   inputs["records"]["records"])
                }
            elif kind == "integrated_case_audit":
                body = {
                    "cases": _join(inputs["event"]["events"],
                                   _select(obs["records"], scope))
                }
            elif kind == "final_assessment":
                body = _result(plan["information_need"]["intent"], inputs)
            else:
                raise ValueError("unsupported macro executor: " + str(kind))
            field, expected_type = next(iter(task["outputs"].items()))
            events.append({
                "action": "ProduceWorkProduct",
                "output_slot": field,
                "type": expected_type,
                "result": copy.deepcopy(body),
            })
            # Summary trace derives directly from recorded actions, not a
            # made-up reasoning narrative or private teacher solution.
            trace = []
            for item in events:
                if item["action"] == "ReadSimulatedSource":
                    trace.append(
                        f"Read simulated {item['modality']} source "
                        f"'{item['title']}' ({item['source_id']})."
                    )
                elif item["action"] == "UseCompletedWorkProduct":
                    trace.append(
                        f"Used completed {item['type']} from "
                        f"{item['from_task']} as input '{item['input_slot']}'."
                    )
                elif item["action"] == "ProduceWorkProduct":
                    trace.append(
                        f"Produced the {item['type']} deliverable "
                        f"'{item['output_slot']}'."
                    )
            answer = {field: body}
            packet = {
                "subtask_id": tid,
                "answer": answer,
                "summary": _summary(kind, body, task),
                "summary_trace": trace,
                "tool_events": events,
                "evidence_ids": sorted(proof),
                "verification": "deterministic_simulated_source_replay",
                "real_media_perception": False,
            }
            feedback[tid] = packet
            completed[tid] = packet
    terminal = graph["terminal_handoffs"][0]
    final = completed[terminal["from_task"]]["answer"][terminal["output_slot"]]
    return {"gold_answer": final, "feedback_bank": feedback,
            "used_sources": sorted({sid for p in feedback.values()
                                    for sid in p["evidence_ids"]}),
            "verification_policy": (
                "controlled synthetic source replay, not actual media perception "
                "or real-world factual verification"
            )}


def validate_grounded(scene: dict) -> list[str]:
    issues = []
    try:
        derived = evaluate(scene, scene["world"])
        if derived["gold_answer"] != scene["gold_answer"]:
            issues.append("gold_answer_replay_mismatch")
        if derived["feedback_bank"] != scene["feedback_bank"]:
            issues.append("feedback_trace_or_binding_drift")
        source_ids = set(scene["world"]["sources"])
        if not source_ids or set(derived["used_sources"]) != source_ids:
            issues.append("unused_or_missing_source")
        for sid in source_ids:
            try:
                evaluate(scene, scene["world"], hidden_sources=frozenset((sid,)))
                issues.append("source_ablation_did_not_block_execution:" + sid)
            except MissingEvidence:
                pass
        if scene["question"] != scene["public"]["question"]:
            issues.append("public_question_drift")
        pub = scene["public"]
        if set(pub) != {
            "task_id", "question", "provided_sources", "max_concurrent_subagents",
            "media_policy",
        }:
            issues.append("public_projection_keys")
        if pub["max_concurrent_subagents"] != 10:
            issues.append("wrong_concurrency")
        if any(key in json.dumps(pub) for key in (
            '"gold_answer"', '"feedback_bank"', '"world"', '"macro_graph"',
            '"source_priors"',
        )):
            issues.append("private_data_leaked_to_public_task")
        ids = scene["information_need"]["case_ids"]
        if any(cid not in scene["question"] for cid in ids):
            issues.append("missing_concrete_case_scope")
        if scene["information_need"]["as_of"] not in scene["question"]:
            issues.append("missing_case_date")
        if len({t["task_id"] for t in scene["macro_graph"]["tasks"]}) != len(
            scene["macro_graph"]["tasks"]):
            issues.append("duplicate_macro_tasks")
    except (MissingEvidence, KeyError, ValueError, TypeError) as exc:
        issues.append("runtime_failed:" + type(exc).__name__ + ":" + str(exc)[:120])
    return issues
