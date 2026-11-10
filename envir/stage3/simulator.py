"""Stage 3 private, deterministic simulator for incremental Macro orchestration.

This MVP models a *trusted* Sub-Agent channel. The underlying World,
Gold Program, DAG edges, future reports and correctness oracle remain
private. A Main-Agent sees only JSON returned by the session methods.

This is an API/prompt boundary, not an OS-level security sandbox.
"""
from __future__ import annotations

from copy import deepcopy
import json

from envir.generate_question.quality import validate_replay
from envir.implement_subtask.realize import realize_scene


COSTS = {
    "delegate": 1.0,
    "accept": 0.1,
    "hold": 0.1,
    "request_evidence": 0.5,
    "retry_same": 1.0,
    "terminate": 0.0,
}
PUBLIC_ACTIONS = tuple(COSTS)


def _same_rows(left, right) -> bool:
    canonical = lambda rows: sorted(
        json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows
    )
    return canonical(left) == canonical(right)


class TrustedSimulator:
    """Private environment engine. Never provide this object as an LLM tool."""

    def __init__(self, scene: dict, reports: list[dict] | None = None, *,
                 mode: str = "reviewed", budget: float = 30.0,
                 max_turns: int = 128, costs: dict | None = None):
        if validate_replay(scene)["status"] != "valid":
            raise ValueError("stage3_scene_replay_invalid")
        if mode not in {"reviewed", "oracle_baseline"}:
            raise ValueError("stage3_unknown_report_mode")
        if mode == "reviewed" and reports is None:
            raise ValueError("reviewed_mode_requires_stage2_reports")
        if type(budget) not in (int, float) or budget <= 0:
            raise ValueError("stage3_invalid_budget")
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("stage3_invalid_max_turns")
        self._scene = deepcopy(scene)
        self._graph = self._scene["macro_graph"]
        self._tasks = {x["task_id"]: x for x in self._graph["tasks"]}
        self._teacher = {
            row["macro_task_id"]: row
            for row in realize_scene(self._scene)["macro_tasks"]
        }
        if len(self._tasks) != len(self._teacher):
            raise ValueError("stage3_oracle_graph_mismatch")
        supplied = {}
        for row in reports or []:
            if row.get("scene_task_id") != self._scene["task_id"]:
                raise ValueError("stage3_foreign_report_in_scene")
            tid = row.get("macro_task_id")
            if tid in supplied or tid not in self._tasks:
                raise ValueError("stage3_duplicate_or_unknown_report")
            if row.get("report_authoring", {}).get("status") != "accepted":
                raise ValueError("stage3_unapproved_report")
            teacher = self._teacher[tid]
            if (not _same_rows(row.get("answer", []), teacher["answer"])
                    or row.get("answer_with_provenance")
                    != teacher["answer_with_provenance"]):
                raise ValueError("stage3_report_not_oracle_grounded")
            finish = row.get("finish_result")
            if (not isinstance(finish, dict)
                    or finish.get("status") != "done"
                    or not isinstance(finish.get("summary"), str)
                    or not isinstance(finish.get("result"), (str, list, dict))):
                raise ValueError("stage3_invalid_finish_result")
            supplied[tid] = {
                "result": deepcopy(finish["result"]),
                "summary": finish["summary"],
                "status": "done",
            }
        if mode == "reviewed" and set(supplied) != set(self._tasks):
            raise ValueError("stage3_requires_complete_reviewed_scene")
        self._report_payloads = (
            supplied if mode == "reviewed" else {
                tid: {
                    "result": deepcopy(row["answer"]),
                    "summary": row["trace_summary"]["summary"],
                    "status": "done",
                } for tid, row in self._teacher.items()
            }
        )
        self._budget_limit = float(budget)
        self._max_turns = max_turns
        self._costs = dict(COSTS)
        if costs is not None:
            if set(costs) != set(COSTS) or any(
                type(v) not in (int, float) or v < 0 for v in costs.values()
            ):
                raise ValueError("stage3_invalid_action_costs")
            self._costs = {key: float(value) for key, value in costs.items()}
        self.reset()

    def public_session(self) -> "MainAgentSession":
        """Only this wrapper is intended for public model tool wiring."""
        return MainAgentSession(self)

    def reset(self) -> dict:
        self._state = {tid: "pending" for tid in self._tasks}
        self._public_reports = {}
        self._evidence_published = set()
        self._history = []
        self._spent = 0.0
        self._done = False
        self._termination = "running"
        self._terminal_submitted_answer = None
        return self.observe()

    def _ready(self) -> list[dict]:
        return [
            task for task in self._graph["tasks"]
            if self._state[task["task_id"]] == "pending"
            and all(self._state[parent] == "accepted"
                    for parent in task["depends_on"])
        ]

    def observe(self) -> dict:
        """Strictly whitelisted initial and acquired public information."""
        available_sources = self._scene.get("public", {}).get(
            "available_source_descriptions", []
        )
        return deepcopy({
            "task_id": self._scene["task_id"],
            "question": self._scene["question"],
            "available_source_descriptions": available_sources,
            "ready_tasks": [{
                "macro_task_id": task["task_id"],
                "objective": "Obtain " + ", ".join(
                    task["outputs"]["typed_relation"]
                ),
                "authorized_source_ids": list(task["source_ids"]),
            } for task in self._ready()] if not self._done else [],
            "reports": [
                {"macro_task_id": tid,
                 "state": self._state[tid],
                 "finish_result": self._public_reports[tid]}
                for tid in self._tasks if tid in self._public_reports
            ],
            "accepted_macro_task_ids": [
                tid for tid in self._tasks if self._state[tid] == "accepted"
            ],
            "remaining_budget": round(
                max(0.0, self._budget_limit - self._spent), 4
            ),
            "total_cost": round(self._spent, 4),
            "action_history": list(self._history),
            "allowed_actions": list(PUBLIC_ACTIONS) if not self._done else [],
            "done": self._done,
            "status": self._termination,
        })

    def _charge(self, verb: str) -> float:
        cost = self._costs[verb]
        if self._spent + cost > self._budget_limit + 1e-9:
            raise ValueError("stage3_insufficient_budget")
        self._spent += cost
        return cost

    def step(self, action: dict) -> dict:
        if self._done:
            raise ValueError("stage3_episode_already_finished")
        if not isinstance(action, dict):
            raise ValueError("stage3_action_must_be_object")
        verb = action.get("type")
        if verb not in PUBLIC_ACTIONS:
            raise ValueError("stage3_invalid_action")
        if len(self._history) >= self._max_turns and verb != "terminate":
            raise ValueError("stage3_max_turns_exhausted")
        tid = action.get("macro_task_id")
        if verb != "terminate":
            if not isinstance(tid, str) or tid not in self._tasks:
                raise ValueError("stage3_unknown_macro_task")
        if verb == "delegate" and tid not in {
            x["task_id"] for x in self._ready()
        }:
            raise ValueError("stage3_dependency_not_unlocked")
        if verb in {"accept", "hold", "request_evidence", "retry_same"} and (
            self._state[tid] not in {"delivered", "held"}
        ):
            raise ValueError("stage3_requires_delivered_report")
        if verb == "request_evidence" and tid in self._evidence_published:
            raise ValueError("stage3_evidence_already_requested")
        cost = self._charge(verb)
        event = {"type": verb, "macro_task_id": tid}
        if verb == "delegate":
            self._state[tid] = "delivered"
            self._public_reports[tid] = deepcopy(self._report_payloads[tid])
            event["report_delivered"] = True
        elif verb == "request_evidence":
            record = self._teacher[tid]
            # All observations are from this executed Macro's reachable
            # evidence. No future task report or unrelated source is revealed.
            allowed = set(record["trace_summary"][
                "available_evidence_source_ids"
            ])
            evidence = []
            for row in record["trace_summary"]["steps"]:
                for atom in row["evidence_atoms"]:
                    if atom["source_id"] in allowed:
                        evidence.append({
                            "fact_ref": atom["fact_ref"],
                            "source_id": atom["source_id"],
                            "values": deepcopy(atom["values"]),
                        })
            event["requested_evidence"] = evidence
            self._evidence_published.add(tid)
        elif verb == "retry_same":
            # Trusted baseline: retrying does not magically generate new
            # evidence; attack/backend persistence belongs to next phase.
            event["report_delivered"] = True
            event["same_report_as_before"] = True
        elif verb == "hold":
            self._state[tid] = "held"
        elif verb == "accept":
            self._state[tid] = "accepted"
            if tid == self._graph["terminal_task_id"]:
                self._done = True
                self._termination = "completed"
                declared = action.get("declared_answer")
                self._terminal_submitted_answer = deepcopy(
                    self._teacher[tid]["answer"] if declared is None
                    else declared
                )
                # Do not disclose whether the completed answer is correct
                # until the PRIVATE teacher evaluates the whole trajectory.
                event["episode_completed"] = True
        elif verb == "terminate":
            self._done = True
            self._termination = "terminated"
            event["episode_completed"] = True
        self._history.append({
            "turn": len(self._history) + 1,
            "type": verb, "macro_task_id": tid, "cost": cost,
        })
        return {
            "event": deepcopy(event),
            "observation": self.observe(),
            "cost": cost,
            "done": self._done,
        }

    def evaluate(self) -> dict:
        """PRIVATE teacher evaluation. Never include this in model input."""
        successful = (
            self._termination == "completed"
            and _same_rows(
                self._terminal_submitted_answer or [],
                self._scene["gold_answer"],
            )
            and all(v == "accepted" for v in self._state.values())
        )
        return {
            "task_id": self._scene["task_id"],
            "success": successful,
            "terminal_state": self._termination,
            "accepted_macro_count": sum(
                status == "accepted" for status in self._state.values()
            ),
            "total_macro_count": len(self._state),
            "total_cost": round(self._spent, 4),
            "action_count": len(self._history),
            "trusted_feedback": True,
            "private_evaluation_only": True,
        }


class MainAgentSession:
    """Public forwarding surface without evaluate() or private scene access."""

    def __init__(self, engine: TrustedSimulator):
        self._engine = engine

    def reset(self) -> dict:
        return self._engine.reset()

    def observe(self) -> dict:
        return self._engine.observe()

    def step(self, action: dict) -> dict:
        return self._engine.step(action)
