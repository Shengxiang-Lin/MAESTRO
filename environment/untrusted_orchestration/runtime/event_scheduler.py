"""Causally ordered evidence events for the Stage-4A executor.

Events schedule *when an existing, authorized investigation may run*. They
never fabricate Source Observations and never expose hidden World/Gold to
the Main prompt. A delayed independent assignment is distinct from a
transport-delayed already executed Sub-Agent message; the latter requires
an asynchronous O1 message adapter and is explicitly not simulated here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


SUPPORTED_SCHEDULE_EVENTS = {"hold_independent_evidence"}
DOCUMENTED_FUTURE_EVENTS = {
    "message_delayed", "report_arrived", "source_reconfirmed",
}


@dataclass(frozen=True)
class EvidenceEvent:
    kind: str
    task_id: str
    earliest_round: int
    after_task_id: str | None = None
    event_id: str = "E001"

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_SCHEDULE_EVENTS:
            raise ValueError("unsupported_executable_evidence_event:" + self.kind)
        if not self.task_id:
            raise ValueError("evidence_event_missing_task")
        if type(self.earliest_round) is not int or self.earliest_round < 1:
            raise ValueError("invalid_evidence_event_round")
        if self.after_task_id == self.task_id:
            raise ValueError("evidence_event_self_dependency")

    def to_private_dict(self) -> dict:
        return asdict(self)


class EventScheduler:
    """Gate independent *future* dispatches until a real causal event.

    No task is hidden after its result has already been delivered; that
    would retroactively edit Main history. A held task must be unexecuted
    when the scenario forks, and the named release prerequisite must be
    independent of that task (validated against the frozen DAG).
    """

    def __init__(self, events: list[dict] | tuple[dict, ...],
                 *, state, fork_round: int):
        self.events = [EvidenceEvent(**data) for data in events]
        self.fork_round = fork_round
        self.delivered_rounds = {
            task_id: fork_round - 1 for task_id in state.current
        }
        self.log: list[dict] = []
        for event in self.events:
            if event.task_id not in state.tasks:
                raise ValueError("evidence_event_unknown_task")
            if event.task_id in state.current:
                raise ValueError("event_cannot_delay_already_returned_task")
            if event.after_task_id is not None:
                if event.after_task_id not in state.tasks:
                    raise ValueError("evidence_event_unknown_release_anchor")
                descendants = state._dependents(event.task_id)
                if event.after_task_id in descendants:
                    raise ValueError("event_release_anchor_depends_on_held_task")
            self.log.append({
                "event_id": event.event_id,
                "kind": event.kind,
                "task_id": event.task_id,
                "release_after_task_id": event.after_task_id,
                "earliest_round": event.earliest_round,
                "status": "scheduled",
            })

    def hold(self, task_id: str, round_number: int) -> bool:
        for event in self.events:
            if event.task_id != task_id:
                continue
            anchor_round = self.delivered_rounds.get(event.after_task_id)
            anchor_released = (
                event.after_task_id is None or
                anchor_round is not None and round_number > anchor_round
            )
            if round_number >= event.earliest_round and anchor_released:
                entry = next(row for row in self.log
                             if row["event_id"] == event.event_id)
                if entry["status"] == "scheduled":
                    entry["status"] = "released"
                    entry["release_round"] = round_number
                return False
            return True
        return False

    def available(self, task_ids: list[str], round_number: int) -> list[str]:
        return [tid for tid in task_ids if not self.hold(tid, round_number)]

    def note_execution(self, task_id: str, round_number: int) -> None:
        self.delivered_rounds[task_id] = round_number

    def audit(self) -> list[dict]:
        return [dict(row) for row in self.log]
