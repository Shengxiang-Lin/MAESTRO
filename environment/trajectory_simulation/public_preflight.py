"""Answer-blind, model-free public executability audit for Stage-3 Gold plans.

A graph having a private symbolic answer does not imply that a real O1
Sub-Agent can execute a public delegation. Dry-run exactly the bounded
public task instructions, published sources and predecessor bindings that
synthesis/export will use. Never insert an oracle-only entity or fact.
"""
from __future__ import annotations

from typing import Any, Mapping

from .compiler import json_value, ready_fragments
from .contract import answer_string, answers_equal
from .evidence import bind_ready, public_record, verify_known_packet
from .realization import task_variants
from .replay import replay_task


def _reason(error: Exception) -> str:
    message = str(error)
    if "missing concrete identification clue" in message:
        return "latent_find_missing_public_clue"
    if "identification clue is not uniquely supported" in message:
        return "latent_find_not_uniquely_supported"
    if "unavailable completed input" in message or "missing completed input" in message:
        return "unavailable_predecessor_result"
    if "unknown" in message and "evidence" in message:
        return "unpublished_evidence_reference"
    if "unrecognized public instruction" in message or "unexecutable identification clue" in message:
        return "public_instruction_not_executable"
    return "public_replay_failed"


def audit_public_executability(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Replay every Gold wave without an LLM or privileged answer injection.

    The simulated public cache is updated *after* all tasks in a wave pass,
    making same-round sibling data and future evidence inaccessible. This
    function is a preflight, NOT an eligibility approval or a real tool run.
    """
    completed: set[str] = set()
    previous: dict[str, dict[str, Any]] = {}
    cache: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    fragments = plan["fragments"]
    for round_no in range(1, len(fragments) + 2):
        ready = bind_ready(ready_fragments(plan, completed), cache)
        if not ready:
            if len(completed) == len(fragments):
                return {
                    "publicly_executable": True, "checked_rounds": round_no - 1,
                    "checked_fragments": len(checks),
                }
            return {
                "publicly_executable": False, "reason": "unresolved_fragment_dependencies",
                "checked_rounds": round_no - 1, "checked_fragments": len(checks),
                "completed_fragment_ids": sorted(completed),
            }
        pending = []
        for task_index, fragment in enumerate(ready):
            fragment_id = fragment["fragment_id"]
            try:
                # Exactly the first permitted O1 variant; changing the
                # language-style prefix cannot alter its numbered operations.
                task = task_variants(
                    fragment, previous, plan.get("provided_resources", ()), cache
                )[0]
                verify_known_packet(task, cache)
                observed = [public_record(o) for o in fragment["observations"]]
                replay = replay_task(task, observed, public_cache=cache)
                mode = "multiset" if fragment["operations"][-1]["op"] == "QUERY_ATTR" else "set"
                if not answers_equal(
                    json_value(replay["result"]), fragment["expected_answer"],
                    {"collection_mode": mode},
                ):
                    raise ValueError("public fragment answer differs from frozen oracle")
                if len(replay["step_values"]) != len(fragment["operation_results"]):
                    raise ValueError("public instruction count does not match Gold operations")
                for index, (operation, result) in enumerate(zip(
                    fragment["operations"], fragment["operation_results"],
                )):
                    step_mode = "multiset" if operation["op"] == "QUERY_ATTR" else "set"
                    if not answers_equal(
                        json_value(replay["step_values"][index]), result["answer"],
                        {"collection_mode": step_mode},
                    ):
                        raise ValueError(
                            f"public intermediate step {index+1} differs from frozen oracle"
                        )
                pending.append((fragment, replay, observed, task_index))
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                return {
                    "publicly_executable": False,
                    "reason": _reason(exc),
                    "detail": str(exc)[:400],
                    "round": round_no, "task_index": task_index,
                    "fragment_id": fragment_id,
                    "output_node_id": fragment["output_node_id"],
                    "operations": [op["op"] for op in fragment["operations"]],
                    "new_source_ids": [o["resource_id"] for o in fragment["observations"]],
                    "completed_fragment_ids": sorted(completed),
                    "checked_fragments": len(checks),
                }
        # Commit only after EVERY independent task has passed public replay.
        for index, (fragment, replay, observed, _) in enumerate(pending):
            fid = fragment["fragment_id"]
            previous[fid] = {
                "fragment_id": fid,
                "public_ref": f"Attempt {round_no}, subtask {index+1}",
                "answer": answer_string(fragment["expected_answer"]),
                "answer_kind": fragment["answer_kind"],
                "value": json_value(replay["result"]),
            }
            for record in observed:
                sid = record["resource_id"]
                if sid in cache and cache[sid] != record:
                    return {
                        "publicly_executable": False,
                        "reason": "contradictory_public_source_records",
                        "fragment_id": fid, "resource_id": sid,
                        "round": round_no, "checked_fragments": len(checks),
                    }
                cache[sid] = record
            completed.add(fid)
            checks.append({"fragment_id": fid, "round": round_no})
    return {
        "publicly_executable": False, "reason": "public_round_budget_exhausted",
        "checked_fragments": len(checks),
    }


__all__ = ["audit_public_executability"]
