"""Build per-decision GLOBAL reward-only context from MAESTRO evidence artifacts.

The student sees the unchanged O1 prompt. This module is used ONLY by the
remote reward evaluator and never exported to GRPO's prompt column.

Two evidence lanes are deliberately separate:
  - actual Sub-Agent claims delivered BEFORE decision t;
  - evaluator-only source World facts (may include unreported rows).
Never present the second lane as observations available to the student.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

SCHEMA = "maestro.global_evidence_context.v1"
CLAIM_SCHEMA = "claim_source_dependency"
_CLAIM_FIELDS = (
    "claim_id", "report_task_id", "observed_in_round", "source_id",
    "evidence_origin_id", "subject", "attribute", "value", "valid_scope",
    "surface", "delivery_status", "revision_seq", "revision_of",
)
_STATE_FIELDS = (
    "subject", "attribute", "valid_scope", "status", "sufficient_to_accept",
    "independent_origins", "report_sources", "internal_disagreement",
    "evidence_count", "supporting_claim_ids", "missing_evidence",
    "evidence_posture",
)
_ACTION_FIELDS = (
    "task_id", "operation", "kind", "target_fields", "source_ids",
    "known_identifiers", "required_parent_fields", "depends_on",
    "descendants", "tools", "model", "priority", "evidence_units",
)
_GRAPH_FIELDS = (
    "task_id", "depends_on", "source_ids", "outputs", "operator_chain",
)
_REL_FIELDS = ("from", "to", "relation")
_DEP_FIELDS = ("parent_claim_id", "child_claim_id", "relation")


def _projection(obj: dict, fields: tuple[str, ...]) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("Global Judge context expects a structured record")
    return {key: obj[key] for key in fields if key in obj}


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Global Judge evidence file: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_claim_snapshots(path: str | Path) -> dict[tuple[str, int], dict]:
    """Exact branch+next-round join. Never use a previous/future snapshot."""
    index = {}
    for row in _jsonl(Path(path)):
        if row.get("schema") != CLAIM_SCHEMA or row.get("private_only") is not True:
            raise ValueError("Invalid Stage4 private claim-state schema")
        sample = row.get("round_sample_id")
        nxt = row.get("main_can_act_from_round")
        if not isinstance(sample, str) or type(nxt) is not int or nxt < 2:
            raise ValueError("Invalid claim snapshot identity/round")
        key = (sample, nxt)
        if key in index:
            raise ValueError("Duplicate Global Judge claim snapshot: " + str(key))
        # A claim extracted after the decision is future information leakage.
        for claim in row.get("claim_nodes", []):
            if type(claim.get("observed_in_round")) is not int or not (
                1 <= claim["observed_in_round"] < nxt
            ):
                raise ValueError("Global Judge claim from current/future round")
        index[key] = row
    return index


def _branch_and_round(sample_id: str) -> tuple[str, int]:
    if not isinstance(sample_id, str) or "-decision-round-" not in sample_id:
        raise ValueError("Invalid Stage4 global-context sample_id")
    branch, digit = sample_id.rsplit("-decision-round-", 1)
    if not digit.isdigit() or int(digit) < 1:
        raise ValueError("Invalid Stage4 global-context decision round")
    return branch, int(digit)


def _scene_for(branch: str, scenes: dict[str, dict]) -> dict:
    names = [task_id for task_id in scenes if branch.startswith(task_id + "-")]
    if len(names) != 1:
        raise ValueError("Missing or ambiguous global task Scene for " + branch)
    return scenes[names[0]]


def _observed_evidence(snapshot: dict | None, round_number: int) -> dict:
    if snapshot is None:
        return {
            "claim_nodes": [], "claim_states": [], "claim_relations": [],
            "claim_dependency_edges": [], "task_dependency_edges": [],
            "report_events": [], "transitions": [],
        }
    nodes = snapshot.get("claim_nodes", [])
    for claim in nodes:
        if not 1 <= claim["observed_in_round"] < round_number:
            raise ValueError("Future claim used in Global Judge decision")
    states = snapshot.get("claim_states", {})
    return {
        "claim_nodes": [_projection(v, _CLAIM_FIELDS) for v in nodes],
        "claim_states": [_projection(v, _STATE_FIELDS) for v in states.values()],
        "claim_relations": [_projection(v, _REL_FIELDS) for v in snapshot.get("claim_relations", [])],
        "claim_dependency_edges": [_projection(v, _DEP_FIELDS) for v in snapshot.get("claim_dependency_edges", [])],
        "task_dependency_edges": snapshot.get("task_dependency_edges", []),
        "report_events": [
            _projection(v, ("macro_task_id", "source_id", "status",
                            "parse_mode", "visible_claim_count",
                            "replaced_earlier_report", "superseded_claim_ids"))
            for v in snapshot.get("report_events", [])
        ],
        "transitions": [
            _projection(v, ("claim_key", "subject", "before", "after",
                            "next_main_decision_round"))
            for v in snapshot.get("transitions", [])
            if v.get("next_main_decision_round", round_number) <= round_number
        ],
    }


def _source_reference(scene: dict, evidence: dict, *, max_rows: int = 12) -> list[dict]:
    """Judge-only true source observations for sources already reported.

    Reference facts are intentionally *not* claimed to have been read by Main.
    Full source facts can include unretrieved rows; never use them to demand
    the student identify an invisible error.
    """
    seen = {c["source_id"] for c in evidence["claim_nodes"] if c.get("source_id")}
    seen.update(e["source_id"] for e in evidence["report_events"] if e.get("source_id"))
    sources = (scene.get("world") or {}).get("sources") or {}
    out = []
    for source_id in sorted(seen):
        source = sources.get(source_id)
        if source is None:
            continue
        observations = source.get("observations", [])
        out.append({
            "source_id": source_id,
            "modality": source.get("modality"),
            "title": source.get("title"),
            "reference_observations": [
                _projection(r, ("observation_id", "fact_refs", "values"))
                for r in observations[:max_rows]
            ],
            "reference_rows_truncated": len(observations) > max_rows,
            "visibility_warning": (
                "This is evaluator-only source truth, NOT necessarily evidence "
                "actually returned to Main; only observed_claims are Main-visible."
            ),
        })
    return out


def build_global_context(scene: dict, state: dict, snapshot: dict | None,
                         *, round_number: int) -> dict:
    """Build immutable, compact reference, without teacher decision or fault labels."""
    if state.get("decision_round") != round_number:
        raise ValueError("Decision state / Global Judge round mismatch")
    graph = scene.get("macro_graph") or {}
    tasks = graph.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Global Judge requires a full Macro DAG")
    evidence = _observed_evidence(snapshot, round_number)
    observable_issues = [
        {"subject": s.get("subject"), "attribute": s.get("attribute"),
         "status": s.get("status"), "missing_evidence": s.get("missing_evidence", [])}
        for s in evidence["claim_states"]
        if s.get("sufficient_to_accept") is False
    ]
    return {
        "schema": SCHEMA,
        "decision_round": round_number,
        "task_graph": {
            "tasks": [_projection(task, _GRAPH_FIELDS) for task in tasks],
            "terminal_task_id": graph.get("terminal_task_id"),
        },
        "source_catalog": [
            _projection(s, ("source_id", "title", "purpose", "modality_hint",
                            "description", "access_mode"))
            for s in scene.get("public", {}).get("available_source_descriptions", [])
        ],
        "scheduling_reference": {
            "max_parallel_tasks": state.get("max_parallel_tasks"),
            "can_complete": state.get("can_complete"),
            "currently_feasible_goals": [
                _projection(goal, _ACTION_FIELDS) for goal in state.get("actionable", [])
            ],
            "blocked_task_ids": state.get("blocked_task_ids", []),
            "completed_task_ids": state.get("completed_task_ids", []),
        },
        "observed_evidence": evidence,
        "recovery_reference": {
            "observable_issues": observable_issues,
            "candidate_repair_goals": [
                _projection(goal, _ACTION_FIELDS) for goal in state.get("actionable", [])
                if goal.get("kind") == "repair"
            ],
        },
        "evaluator_only_reference": {
            "reference_source_facts_for_reported_sources":
                _source_reference(scene, evidence),
            "reference_final_answer": scene.get("gold_answer"),
            "visibility_warning": (
                "The reference World, final answer, and global graph are reward-only. "
                "Never penalize Main for failing to know unobserved facts or fault identity. "
                "Use actual returned evidence to decide whether an action was justified."
            ),
        },
    }


def load_global_contexts(*, scenes_path: str | Path,
                         claim_states_path: str | Path,
                         private_states: dict[str, dict],
                         rows: list[dict]) -> tuple[dict[str, dict], dict]:
    scenes_rows = _jsonl(Path(scenes_path))
    scenes = {x["task_id"]: x for x in scenes_rows}
    if len(scenes) != len(scenes_rows):
        raise ValueError("Duplicate Scene in Global Judge source")
    snapshots = index_claim_snapshots(claim_states_path)
    contexts, count = {}, Counter()
    for row in rows:
        meta = json.loads(row["extra_info_json"])
        sample = meta["sample_id"]
        branch, rnd = _branch_and_round(sample)
        if meta.get("step_index") != rnd:
            raise ValueError("Global Judge Step ID does not match Stage4 round")
        if sample not in private_states:
            raise ValueError("Global Judge missing pre-action decision state")
        scene = _scene_for(branch, scenes)
        snap = snapshots.get((branch, rnd))
        if rnd > 1 and snap is None:
            raise ValueError("Missing pre-decision visible claim snapshot for " + sample)
        if snap is None:
            count["round_one_no_report"] += 1
        else:
            count["with_observed_evidence"] += 1
        contexts[sample] = build_global_context(
            scene, private_states[sample], snap, round_number=rnd,
        )
    if len(contexts) != len(rows):
        raise ValueError("Duplicate Global Judge sample_id")
    return contexts, {
        "rows": len(contexts), "claim_snapshots": len(snapshots),
        "source_scenes": len(scenes), **dict(count),
    }
