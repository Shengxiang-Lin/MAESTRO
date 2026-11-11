"""Atomic fault ontology and executable, provenance-grounded candidate probing.

An atomic fault describes ONE mutated execution/report surface. A scenario
can schedule several atoms at different genuine execution boundaries, while
the legacy latest-feedback fork remains a one-atom special case.

The compiler never invents a replacement value: primitive faults are probed
against the frozen World using the ordinary executor. Report-layer atoms
must also have an actual same-schema distractor from an authorized source.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from environment.generate_question.program_ir import ProgramError
from .faults import Fault, FaultUnsupported, relation_rows, run_attempt


# Public names intentionally distinguish *where* the fault occurred from
# the Main-visible symptom (which may be "done" in multiple categories).
ATOM_SPECS = {
    "no_result": ("acquisition", "missing"),
    "incomplete_retrieval": ("acquisition", "partial"),
    "wrong_retrieval": ("processing", "wrong"),
    "wrong_entity_binding": ("processing", "wrong"),
    "report_missing": ("report", "report"),
    "fabricated_claim": ("report", "report"),
    "result_summary_conflict": ("report", "report"),
}

# Delayed delivery is an EVENT, not a WorkProduct mutation. It is not
# sampled as a fault until an explicit legal arrival/dispatch contract
# is supplied by the event scheduler.
EVENT_KINDS = {
    "hold_independent_evidence",
    "release_independent_evidence",
    "message_delayed",
    "report_arrived",
}
LEGACY_MODES = {"missing", "partial", "wrong"}
LEGACY_TO_ATOM = {
    "missing": "no_result",
    "partial": "incomplete_retrieval",
    "wrong": "legacy_wrong_evidence",
}
ATOM_TO_MODE = {key: value[1] for key, value in ATOM_SPECS.items()}
ATOM_TO_MODE["legacy_wrong_evidence"] = "wrong"


@dataclass(frozen=True)
class AtomicFault:
    kind: str
    task_id: str
    origin_round: int
    seed: int = 0
    persistence: str = "once"
    atom_id: str = "F001"

    def __post_init__(self) -> None:
        if self.kind not in ATOM_TO_MODE:
            raise ValueError("unknown_atomic_fault_kind:" + self.kind)
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("atomic_fault_requires_task")
        if type(self.origin_round) is not int or self.origin_round < 1:
            raise ValueError("invalid_atomic_fault_origin")
        if self.persistence not in {"once", "same_path"}:
            raise ValueError("invalid_atomic_fault_persistence")
        if type(self.seed) is not int:
            raise ValueError("invalid_atomic_fault_seed")

    @property
    def surface(self) -> str:
        if self.kind == "legacy_wrong_evidence":
            return "processing"
        return ATOM_SPECS[self.kind][0]

    def as_legacy_fault(self) -> Fault:
        return Fault(
            self.task_id, ATOM_TO_MODE[self.kind],
            seed=self.seed, persistence=self.persistence,
            atom_kind=self.kind,
        )

    def to_private_dict(self) -> dict:
        return {**asdict(self), "surface": self.surface,
                "executor_mode": ATOM_TO_MODE[self.kind]}


def legacy_atom(fault: Fault, *, origin_round: int,
                atom_id: str = "F001") -> AtomicFault:
    return AtomicFault(
        kind=(fault.atom_kind or LEGACY_TO_ATOM[fault.mode]),
        task_id=fault.task_id, origin_round=origin_round,
        seed=fault.seed, persistence=fault.persistence,
        atom_id=atom_id,
    )


def probe_atoms(scene: dict, kinds: Iterable[str]) -> dict[str, list[str]]:
    """Probe *actual execution*, not textual plausibility, for each kind.

    This is private preparatory data. A failed probe does not become a
    synthetic "successful" example; unsupported cases are filtered out.
    """
    kinds = list(kinds)
    if any(kind not in ATOM_TO_MODE for kind in kinds):
        raise ValueError("unsupported_atomic_probe_kind")
    tasks = {task["task_id"]: task for task in scene["macro_graph"]["tasks"]}
    clean = {}
    eligible = {}
    for wave in scene["macro_graph"]["reference_waves"]:
        for tid in wave:
            task = tasks[tid]
            parents = [clean[parent] for parent in task["depends_on"]]
            sources = {
                sid: scene["world"]["sources"][sid]
                for sid in task["source_ids"]
            }
            clean[tid], _ = run_attempt(
                task, parents=parents, sources=sources,
            )
            eligible[tid] = []
            for kind in kinds:
                try:
                    altered, metadata = run_attempt(
                        task, parents=parents, sources=sources,
                        fault=AtomicFault(
                            kind, tid, 1,
                        ).as_legacy_fault(),
                    )
                except (FaultUnsupported, ProgramError, ValueError,
                        TypeError, KeyError, IndexError, ZeroDivisionError):
                    continue
                if kind in {"no_result", "incomplete_retrieval"}:
                    if (kind == "no_result" and altered is not None
                            or kind == "incomplete_retrieval"
                            and metadata.get("status") != "partial"):
                        continue
                elif kind in {"wrong_retrieval", "wrong_entity_binding",
                              "legacy_wrong_evidence"}:
                    if (altered is None
                            or relation_rows(altered) ==
                            relation_rows(clean[tid])):
                        continue
                elif metadata.get("report_fault_kind") != kind:
                    continue
                eligible[tid].append(kind)
    return eligible
