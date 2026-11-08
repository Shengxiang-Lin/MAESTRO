"""Compile macro work handoffs from a concrete information need.

Case scopes are keyed by (artifact type, subject ID, work scope). Two cases
may each generate a CaseSet/RecordBundle; only same-key work is reusable.
The coarsener ACTUALLY merges a redundant scoped record lookup + matching
task when they serve the same evidence-bound work goal.
"""
from __future__ import annotations

from collections import Counter

from .graph import (MAX_CONCURRENCY, build_atomic_draft, coarsen,
                    compile_contracts, compile_waves, complexity)


OUTPUTS = {
    "event_investigation": ("event_brief", "EventBundle"),
    "records_audit": ("record_audit", "RecordBundle"),
    "rules_review": ("rules_brief", "RuleBrief"),
    "capacity_review": ("capacity_brief", "CapacityBundle"),
    "independent_review": ("independent_brief", "IndependentBundle"),
    "case_matching": ("matched_cases", "CaseSet"),
    "integrated_case_audit": ("matched_cases", "CaseSet"),
    "final_assessment": ("report", "FinalReport"),
}


def _work_goal(kind: str, term: str, scope: str, intent: str):
    qualifier = f" for {term} {scope}" if scope != "all" else " for the specified cases"
    return {
        "event_investigation": "Investigate the case observations and establish the relevant event dates and case identities" + qualifier,
        "records_audit": "Audit the registered case records, reported status and measured workload" + qualifier,
        "rules_review": "Investigate the applicable status, time-window and scope requirements",
        "capacity_review": "Investigate the operating resources available to the identified cases",
        "independent_review": "Investigate independent event accounts and their reported timing",
        "case_matching": "Reconcile observed case identities with the corresponding registered records" + qualifier,
        "integrated_case_audit": "Audit the registered records for the identified observations and return reconciled case findings" + qualifier,
        "final_assessment": {
            "compliance": "Assess which cases satisfy the applicable registration and time requirements",
            "impact": "Assess affected workload using matched case records and applicable thresholds",
            "comparison": "Compare the recorded workload for the two investigated cases",
            "timeline": "Reconstruct which reported event has the greatest time difference",
            "reconciliation": "Reconcile conflicting event timestamps across primary and independent sources",
            "feasibility": "Evaluate which cases meet registration, time and capacity conditions",
            "prioritization": "Rank case review priority using workload, urgency and capacity",
            "mitigation": "Identify the case with the most urgent unmet workload capacity",
        }[intent],
    }[kind]


def build_graph(need: dict, world: dict) -> dict:
    packages, edges = [], []
    source_ids = world["source_ids"]
    n = Counter()
    indexes = {}

    def add(kind: str, *, scope: str = "all", source_role: str | None = None,
            dependencies: tuple[tuple[str, str, str, str], ...] = ()):
        key = (kind, scope, source_role, tuple(dependencies))
        if key in indexes:
            return indexes[key]
        n[kind] += 1
        tid = "S" + str(len(packages) + 1)
        field, typ = OUTPUTS[kind]
        deps = []
        for consumer_field, upstream, producer_field, required_type in dependencies:
            if not any(p["task_id"] == upstream and
                       p["outputs"].get(producer_field) == required_type
                       for p in packages):
                raise ValueError("missing or mistyped upstream work output")
            deps.append({
                "from": upstream, "to": tid, "input_slot": consumer_field,
                "output_slot": producer_field, "type": required_type,
                "kind": "required_data",
                "activation": "producer_completed",
                "handoff_reason": {
                    "event_scope": "Identified event cases define the record-audit scope.",
                    "event": "The primary investigation supplies observed case identifiers and timing.",
                    "records": "Audited records supply registered status and measured workload.",
                    "matched_cases": "Matched case findings are needed to evaluate the requested result.",
                    "case_a": "The first case provides a separately grounded comparison operand.",
                    "case_b": "The second case provides a separately grounded comparison operand.",
                    "rules": "Applicable rules supply the threshold and authorization conditions.",
                    "capacity": "Operating capacity constrains which cases or actions are feasible.",
                    "independent": "Independent observations provide cross-source timing evidence.",
                }[consumer_field],
            })
        private_actions = {
            "event_investigation": [
                "review the available primary incident material",
                "identify the relevant case observations",
                "confirm observation timestamps",
            ],
            "records_audit": [
                "locate the relevant record register",
                "check case identifiers",
                "extract registered workload, status and urgency",
            ],
            "rules_review": [
                "locate applicable requirements",
                "verify scope and effective date",
                "record status, time and workload conditions",
            ],
            "capacity_review": [
                "review available operational resources",
                "match capacity to the relevant cases",
                "summarize usable capacity",
            ],
            "independent_review": [
                "review independent reports",
                "establish reported case identities",
                "extract independent event timestamps",
            ],
            "case_matching": [
                "align observed case identities with record keys",
                "validate matching time scope",
                "compile corroborated case profiles",
            ],
            "integrated_case_audit": [
                "locate matched registration records",
                "cross-check each observed case",
                "compile reconciled event and registration details",
            ],
            "final_assessment": [
                "collect the completed work products",
                "apply the requested comparison or evaluation criteria",
                "prepare a source-supported answer",
            ],
        }[kind]
        packages.append({
            "task_id": tid,
            "work_type": kind,
            "scope_key": scope,
            "goal": _work_goal(kind, need["case_term"], scope, need["intent"]),
            "deliverable": {
                "event_investigation": "An event investigation brief.",
                "records_audit": "A scoped register audit.",
                "rules_review": "An applicable requirements brief.",
                "capacity_review": "A capacity assessment.",
                "independent_review": "An independent source report.",
                "case_matching": "A set of matched case records.",
                "integrated_case_audit": "An integrated investigation and record audit.",
                "final_assessment": "An answer to the requested case-scoped question.",
            }[kind],
            "outputs": {field: typ},
            "input_schema": {e["input_slot"]: e["type"] for e in deps},
            "authorized_source_ids": [source_ids[source_role]] if source_role else [],
            "capability_hints": {
                "event_investigation": ["MediaAnalysis", "Investigation"],
                "records_audit": ["TableAnalysis", "Research"],
                "rules_review": ["Research", "PolicyAnalysis"],
                "capacity_review": ["TableAnalysis", "OperationsResearch"],
                "independent_review": ["Research", "MediaAnalysis"],
                "case_matching": ["Research", "TableAnalysis"],
                "integrated_case_audit": ["Research", "TableAnalysis"],
                "final_assessment": ["Analysis", "Reasoning"],
            }[kind],
            "internal_actions": private_actions,
        })
        edges.extend(deps)
        indexes[key] = tid
        return tid

    def event(scope):
        return add("event_investigation", scope=scope, source_role="event")

    def records(scope, event_task=None):
        deps = (
            (("event_scope", event_task, "event_brief", "EventBundle"),)
            if event_task and need["record_scope"] == "event_scoped"
            else ()
        )
        return add("records_audit", scope=scope, source_role="records",
                   dependencies=deps)

    def matched(scope):
        e = event(scope)
        r = records(scope, e)
        return add("case_matching", scope=scope, dependencies=(
            ("event", e, "event_brief", "EventBundle"),
            ("records", r, "record_audit", "RecordBundle"),
        ))

    def rules():
        return add("rules_review", source_role="rules")

    def capacity():
        return add("capacity_review", source_role="capacity")

    def independent():
        return add("independent_review", source_role="independent")

    intent = need["intent"]
    links = []
    if intent == "comparison":
        # Two distinct CaseSet work products; a type-only cache would
        # conflate these and destroy the actual comparison.
        a = matched(need["case_ids"][0])
        b = matched(need["case_ids"][1])
        links = [("case_a", a, "matched_cases", "CaseSet"),
                 ("case_b", b, "matched_cases", "CaseSet")]
    elif intent in {"timeline", "reconciliation"}:
        a = event("all")
        b = independent()
        links = [("event", a, "event_brief", "EventBundle"),
                 ("independent", b, "independent_brief", "IndependentBundle")]
    else:
        m = matched("all")
        links = [("matched_cases", m, "matched_cases", "CaseSet")]
        if intent in {"compliance", "impact", "feasibility"}:
            links.append(("rules", rules(), "rules_brief", "RuleBrief"))
        if intent in {"prioritization", "mitigation", "feasibility"}:
            links.append(("capacity", capacity(), "capacity_brief", "CapacityBundle"))
    final_id = add("final_assessment", dependencies=tuple(links))
    # A genuine coarsening opportunity: if record auditing is event-scoped,
    # its only consumer is matching, and both actions form one work product,
    # merge them into an integrated investigation, removing the redundant
    # boundary while preserving event -> matched-cases handoff.
    merged = []
    by_id = {p["task_id"]: p for p in packages}
    for match in list(packages):
        if match["work_type"] != "case_matching":
            continue
        edge_record = next((edge for edge in edges
                            if edge["to"] == match["task_id"]
                            and edge["input_slot"] == "records"), None)
        if edge_record is None:
            continue
        r = by_id[edge_record["from"]]
        if r["work_type"] != "records_audit":
            continue
        rec_inputs = [e for e in edges if e["to"] == r["task_id"]]
        consumers = [e for e in edges if e["from"] == r["task_id"]]
        if len(rec_inputs) != 1 or rec_inputs[0]["input_slot"] != "event_scope":
            continue
        if len(consumers) != 1 or consumers[0] is not edge_record:
            continue
        match["work_type"] = "integrated_case_audit"
        match["goal"] = _work_goal("integrated_case_audit", need["case_term"],
                                  match["scope_key"], intent)
        match["deliverable"] = "An integrated scoped incident and registration audit."
        match["authorized_source_ids"] = r["authorized_source_ids"]
        match["capability_hints"] = ["TableAnalysis", "Investigation"]
        match["internal_actions"] = (
            r["internal_actions"] + match["internal_actions"][:2]
        )
        match["input_schema"].pop("records")
        edges.remove(edge_record)
        edges.remove(rec_inputs[0])
        merged.append({"removed": r["task_id"], "into": match["task_id"],
                       "reason": "only_consumer_and_same_record_audit_deliverable"})
    if merged:
        remove_ids = {x["removed"] for x in merged}
        packages = [p for p in packages if p["task_id"] not in remove_ids]
    atomic = build_atomic_draft(packages, edges)
    tasks, stats = coarsen(atomic)
    compile_contracts(tasks, edges)
    waves = compile_waves(tasks, max_concurrency=MAX_CONCURRENCY)
    metrics = complexity(tasks, waves, stats)
    metrics["actual_work_boundary_merges"] = len(merged)
    metrics["scoped_work_keys"] = len({(t["work_type"], t["scope_key"]) for t in tasks})
    metrics["parallel_waves"] = sum(len(w) > 1 for w in waves)
    metrics["root_tasks"] = sum(not t["depends_on"] for t in tasks)
    return {
        "schema": "maestro.macro_graph.v3",
        "tasks": tasks, "edges": edges,
        "terminal_handoffs": [
            {"from_task": final_id, "output_slot": "report", "type": "FinalReport"}
        ],
        "reference_waves": waves,
        "coarsening": {**stats, "actual_merges": merged},
        "complexity": metrics,
        "selection_policy": "semantic_information_requirements_not_prefixed_DAG",
        "max_concurrent_subagents": MAX_CONCURRENCY,
    }
