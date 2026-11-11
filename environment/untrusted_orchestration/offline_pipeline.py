"""Offline Stage-4B/4C: review execution reports first, then Main actions.

4A freezes evidence requests, immutable actual Attempt traces, batch ordering,
and a public-history-grounded deterministic reference schedule. 4B may change
only report language. 4C sees ONLY the finalized report history and prior Main
actions, never the private Attempt ledger or Gold. Evidence-bound assignments
remain constant in scope/tools but can reuse the newly authored public context.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Callable

from environment.base_orchestration.action_author import (
    _assignment, author_delegate_action, author_complete_action,
)
from environment.base_orchestration.o1_compat import (
    o1_prompt, source_catalog,
)
from environment.base_orchestration.trajectory_builder import _strict_reports
from environment.generate_question.program_ir import Relation
from .report_author import naturalize_report
from .state_machine import Attempt
from .trajectory_builder import (
    _MAIN_AUTHOR_SYSTEM, _MAIN_CRITIC_SYSTEM, _history_batch,
    _as_public,
)


def _attempt_from_audit(task: dict, entry: dict) -> Attempt:
    runtime = entry["runtime"]
    # A previously executed attempt can become 'stale' or 'disputed' later.
    # Report the original delivery outcome rather than the *later* plan state.
    status = runtime.get("status", entry["status"])
    rows = entry["rows"]
    relation = None if status == "missing" else Relation(
        tuple(task["outputs"]["typed_relation"]), tuple({
            "values": deepcopy(row["values"]),
            "support": tuple(row["source_fact_refs"]),
        } for row in rows),
    )
    return Attempt(
        task_id=task["task_id"], version=entry["version"], status=status,
        product=relation,
        available_fields=tuple(runtime.get("available_fields", [])),
        used_parent_versions=entry["used_parent_versions"],
        runtime=deepcopy(runtime), cost=entry["cost"],
    )


def author_branch_reports(scene: dict, stage2_rows: list[dict],
                          episode: dict, audit: dict, *,
                          author=None, critic=None, max_revisions: int = 2,
                          on_report=None) -> tuple[dict, dict]:
    """4B authors *only* actual previously simulated Attempt returns.

    The original trusted prefix reports are reused verbatim. 4B does not
    schedule tasks, invoke Main, or redo the underlying poisoned program.
    """
    episode, audit = deepcopy(episode), deepcopy(audit)
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    reviewed = _strict_reports(scene, stage2_rows)
    versions = audit["runtime"]["attempt_versions"]
    decision_ids = {
        d["round"]: d["task_ids"]
        for d in audit["decision_audit"] if d.get("decision_kind") == "delegate"
    }
    waves = scene["macro_graph"]["reference_waves"]
    fork = audit["branch_round"]
    review_rows = []
    for scheduled in audit["subagent_report_reviews"]:
        tid, version, round_number = (
            scheduled["task_id"], scheduled["attempt_version"],
            scheduled["round"],
        )
        wave = (waves[round_number - 1] if round_number < fork
                else decision_ids[round_number])
        slot = wave.index(tid)
        step = next(
            x for x in episode["trajectory"]
            if x["action"] == "delegate_task" and x["attempt"] == round_number
        )
        old = step["result"]["subtask_results"][slot]
        record = next(
            v for v in versions[tid] if v["version"] == version
        )
        actual = _attempt_from_audit(tasks[tid], record)
        report, status = naturalize_report(
            tasks[tid], actual,
            instruction=old["task_instruction"], model_alias=old["model"],
            reviewed=reviewed.get(tid), author=author, critic=critic,
            max_revisions=max_revisions,
        )
        report["subtask_index"] = slot
        step["result"]["subtask_results"][slot] = _as_public(report)
        review_rows.append({
            **scheduled, "authoring": deepcopy(status),
        })
        if round_number == fork - 1:
            for fault in audit["faults"]:
                if fault["task_id"] == tid:
                    fault["report_mode"] = deepcopy(status)
        if on_report is not None:
            on_report(status)
    audit["subagent_report_reviews"] = review_rows
    return episode, audit


def _visibility_issues(requests: list[dict], public_history: str) -> list[str]:
    """Fail closed if the planner relied on an identifier not in reports."""
    visible = public_history.casefold()
    missing = []
    for request in requests:
        for identifier in request.get("known_identifiers", []):
            if identifier.casefold() not in visible:
                missing.append(request["request_id"] + ":" + identifier)
        if request["operation"] in {
            "retry_missing_evidence", "complete_partial_evidence",
        }:
            if not any(x in visible for x in (
                "missing", "partial", "no result", "unresolved",
                "not delivered", "incomplete", "did not return",
                "did not produce",
            )):
                missing.append(request["request_id"] + ":missing_status")
    return missing


def _completion_from_visible(step: dict, latest: dict,
                             scene: dict) -> dict:
    terminal = scene["macro_graph"]["terminal_task_id"]
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    refs = ([terminal] if terminal in latest else
            tasks[terminal]["depends_on"])
    quotes = []
    for tid in refs[-2:]:
        finish = (latest.get(tid) or {}).get("finish_result") or {}
        value = str(finish.get("result") or "").strip().replace("\n", " ")
        if value:
            quotes.append(value[:360].rstrip())
    if not quotes:
        raise ValueError("stage4c_completion_has_no_visible_result")
    reason = " ".join(
        'The returned finding states: "' + quote + '".'
        for quote in quotes
    ) + (
        " These returned findings provide the required evidence "
        "for the final answer."
    )
    return {
        "action": "complete", "reasoning": reason,
        "params": deepcopy(step["params"]),
    }


def author_branch_main(scene: dict, episode: dict, audit: dict, *,
                       author=None, critic=None,
                       max_revisions: int = 2,
                       max_rounds: int = 16,
                       sub_model: str = "qwen3-8b",
                       model_alias: str = "model_1",
                       on_main=None) -> tuple[dict, list[dict], dict]:
    """4C writes all Main turns after 4B has completed, from public history."""
    episode, audit = deepcopy(episode), deepcopy(audit)
    fork = audit["branch_round"]
    plans = {p["round"]: p for p in audit.get("evidence_plan", [])}
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    catalog = source_catalog(scene)
    entries, latest, turns, reviews = [], {}, [], []
    failure = None
    for step in episode["trajectory"]:
        round_number = step["attempt"]
        if round_number >= fork and step["action"] in {
            "delegate_task", "complete",
        }:
            prompt = o1_prompt(
                scene, entries, attempt=round_number,
                max_attempts=max_rounds, sub_model=sub_model,
                model_alias=model_alias,
            )
            if step["action"] == "delegate_task":
                plan = plans[round_number]
                history = json.dumps(entries, ensure_ascii=False)
                issues = _visibility_issues(plan["evidence_requests"], history)
                if issues:
                    failure = "stage4c_evidence_not_main_visible:" + ",".join(issues)
                    break
                assignments = []
                for dispatch in plan["dispatch"]:
                    tid = dispatch["execution_binding"]["macro_task_id"]
                    planned = dispatch["assignment_contract"]
                    # The fixed semantics and tool scope come from 4A.
                    # Stage-4B's actual report language supplies up-to-date
                    # parent context; never copy future or private records.
                    current = _assignment(
                        tasks[tid], catalog, latest, model_alias=model_alias,
                    )
                    current["target_instruction"] = planned["target_instruction"]
                    current["context"] = (
                        current["context"] + " Current evidence need: "
                        + next(req["main_visible_justification"]
                               for req in plan["evidence_requests"]
                               if req["request_id"] in
                               dispatch["evidence_request_ids"])
                    )[:3500]
                    current["evidence_request_id"] = planned["evidence_request_id"]
                    assignments.append(current)
                action, status = author_delegate_action(
                    assignments=assignments, prompt=prompt,
                    author=author, critic=critic,
                    max_revisions=max_revisions,
                    author_system=_MAIN_AUTHOR_SYSTEM,
                    critic_system=_MAIN_CRITIC_SYSTEM,
                )
            else:
                expected = _completion_from_visible(step, latest, scene)
                action, status = author_complete_action(
                    template=expected, prompt=prompt,
                    delivered={
                        k: v for k, v in latest.items()
                        if isinstance(v.get("finish_result"), dict)
                    },
                    author=author, critic=critic,
                    max_revisions=max_revisions,
                )
            reviews.append({
                "round": round_number,
                "kind": ("delegate" if step["action"] == "delegate_task"
                         else "complete"),
                "authoring": deepcopy(status),
            })
            if on_main is not None:
                on_main(status)
            if action is None:
                failure = "main_action_not_accepted"
                break
            step["reasoning"] = action["reasoning"]
            step["params"] = deepcopy(action["params"])
            turns.append({
                "sample_id": (scene["task_id"] +
                              f"-branch-{fork:02d}-round-{round_number:02d}"),
                "task_id": scene["task_id"], "round": round_number,
                "input": {
                    "subtask_history": __import__(
                        "environment.base_orchestration.o1_compat",
                        fromlist=["o1_history"],
                    ).o1_history(entries),
                },
                "output": deepcopy(action),
                "generation_mode": status["mode"],
            })
        if step["action"] == "delegate_task":
            ids = (scene["macro_graph"]["reference_waves"][round_number - 1]
                   if round_number < fork else
                   [d["execution_binding"]["macro_task_id"]
                    for d in plans[round_number]["dispatch"]])
            returned = step["result"]["subtask_results"]
            for tid, report in zip(ids, returned):
                latest[tid] = report
            entries.append(_history_batch(round_number, returned))
    execution_outcome = audit["execution_outcome"]
    had_report_fallback = any(
        row["authoring"].get("mode") == "grounded_template_fallback"
        for row in audit["subagent_report_reviews"]
    )
    had_main_fallback = any(
        row["authoring"].get("mode") ==
        "teacher_grounded_completion_fallback"
        for row in reviews
    )
    if failure is not None:
        outcome = ("language_review_unaccepted"
                   if failure.startswith("stage4c_evidence_not_main_visible")
                   else failure)
    elif execution_outcome == "recovered" and (
        had_report_fallback or had_main_fallback
    ):
        outcome = "language_review_unaccepted"
    else:
        outcome = execution_outcome
    audit["main_action_reviews"] = reviews
    audit["outcome"] = outcome
    audit["language_review_accepted"] = (
        failure is None and not had_report_fallback and not had_main_fallback
    )
    audit["language_evidence_issues"] = [failure] if failure else []
    audit["accepted_turns"] = len(turns) if outcome == "recovered" else 0
    episode["outcome"] = outcome
    episode["recovered"] = outcome == "recovered"
    return episode, turns if outcome == "recovered" else [], audit
