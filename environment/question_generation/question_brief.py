"""Compile a compact Stage-2B Question Brief from PUBLIC Stage-2A Blueprints.

A Question Author receives one goal per linear workstream, the public
constraints on that workstream, exact input dependencies and a final result
contract. It does NOT receive the full mechanical reference, the recursive
semantic tree, tools, original private ProgramIR, or each redundant fragment
goal. The independent Critic still reviews the full scoped reference.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from environment.delegation_blueprint import _terminal_result_contract
from .public_semantics import public_question_contract


_STEP_FIELDS = (
    "operation", "attribute", "comparison", "public_value",
    "relation", "direction", "entity_type", "output_attribute",
    "ranking", "rank_attribute", "inputs_from",
)


def _compact_step(step):
    """Do not repeat synthetic prose/LLM hints or tool metadata."""
    return {
        key: step[key] for key in _STEP_FIELDS
        if key in step and step[key] is not None
        and (key != "inputs_from" or len(step[key]) != 1)
    }


def _public_anchor_map(world, canonical, gold):
    """FIND labels from the SAME public rendering, bound to program steps.

    No raw FIND inputs, Gold answers, or private evidence sources may enter
    the brief. A concealed identity remains unspecified.
    """
    contract = public_question_contract(world, canonical, gold)
    hints = {}
    for scope in contract["scopes"]:
        if scope.get("definition") != "FIND":
            continue
        clue = {
            key: scope[key] for key in ("public_anchor", "type_anchor")
            if isinstance(scope.get(key), str) and scope[key]
        }
        if not clue:
            continue
        for node in scope.get("program_steps", []):
            hints["S" + str(node)] = clue
    return hints


def compile_question_brief(blueprint, world, canonical, gold):
    """Return a compact, topology-preserving public Question Author brief.

    Collapse only non-branching serial chains. A fork, merge or separate
    parallel branch remains explicit, preserving independent candidate
    pools and the join operator that combines them.
    """
    if not isinstance(blueprint, Mapping):
        raise ValueError("missing_certified_blueprint")
    outline = blueprint["outline"]
    tasks = outline["tasks"]
    goals = {g["fragment_id"]: g["goal"] for g in blueprint["goals"]}
    by_id = {task["fragment_id"]: task for task in tasks}
    if len(goals) != len(tasks) or len(by_id) != len(tasks):
        raise ValueError("invalid_question_brief_dag")
    consumers = defaultdict(list)
    for task in tasks:
        for dep in task["depends_on"]:
            if dep not in by_id:
                raise ValueError("unknown_question_brief_dependency")
            consumers[dep].append(task["fragment_id"])

    # A maximal nonbranching chain is a single information-seeking path.
    # In particular, a shared ancestor never gets copied into two branches.
    visited, streams, group_for = set(), [], {}
    for task in tasks:
        fid = task["fragment_id"]
        if fid in visited:
            continue
        chain = [fid]
        visited.add(fid)
        while len(consumers[chain[-1]]) == 1:
            child_id = consumers[chain[-1]][0]
            if (child_id in visited
                    or len(by_id[child_id]["depends_on"]) != 1):
                break
            chain.append(child_id)
            visited.add(child_id)
        group_id = "B" + str(len(streams) + 1)
        for member in chain:
            group_for[member] = group_id
        streams.append((group_id, chain))
    if len(visited) != len(tasks):
        raise ValueError("uncovered_question_brief_fragments")

    anchors = _public_anchor_map(world, canonical, gold)
    workstreams = []
    for group_id, chain in streams:
        first = by_id[chain[0]]
        upstream = list(dict.fromkeys(
            group_for[dep] for dep in first["depends_on"]
        ))
        details = []
        for fid in chain:
            for step in by_id[fid]["public_steps"]:
                item = _compact_step(step)
                if step["operation"] == "FIND":
                    item.update(anchors.get(step["id"], {}))
                details.append(item)
        workstreams.append({
            "id": group_id,
            "goal": goals[chain[-1]],
            "depends_on": upstream,
            "precise_public_steps": details,
            "output_kind": by_id[chain[-1]]["output_kind"],
        })

    terminal = tasks[-1]
    result = _terminal_result_contract(outline, terminal)
    if result is None:
        # Some final operations are VERIFY/SELECT and have no terminal
        # anchor type; still preserve the exact final operator.
        final_step = terminal["public_steps"][-1]
        result = {
            "action": final_step["operation"],
            "output_kind": terminal["output_kind"],
        }
        for key in ("output_attribute", "ranking", "rank_attribute",
                    "comparison", "public_value"):
            if final_step.get(key) is not None:
                result[key] = final_step[key]
    # Unambiguous AND/OR is explicit in precise_public_steps. Do not
    # ask the LLM to independently infer it from a prose reference.
    return {
        "schema": "maestro.question_brief.branch_v1",
        "task": "Ask one natural question which preserves ALL public constraints.",
        "final_result": result,
        "workstreams": workstreams,
        "public_modalities": outline["public_modalities"],
        "authoring_rules": (
            "Each workstream goal is an INTENT, not a complete specification. "
            "Preserve every precise_public_step, its owner and value; "
            "INTERSECT means same entities in ALL inputs, UNION means "
            "entities in ANY input. depends_on is a data dependency, not "
            "a requirement to narrate internal task order. "
            "Keep source candidates independent until the documented join. "
            "Use a coherent user question, not group/step instructions. "
            "Unspecified public labels must not be guessed."
        ),
    }


__all__ = ["compile_question_brief"]
