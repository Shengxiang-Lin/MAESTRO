"""Goal-conditioned, typed producer composition for Stage-1 Task Plans.

Only supported source-scoped operators are composed.  Dependencies on
candidate sets use *keyed* access to downstream stores, rather than adding
decorative DAG edges.  This is deliberately bounded to the current entity
answer contract; other Goal output kinds require separate executable solvers.
"""
from __future__ import annotations

import copy
import random

from environment.schema import digest

VERSION = "goal_conditioned_typed_v2"

# These are local ways to produce an initial entity reference, not full DAG
# templates. The required information ports are satisfied before the producer
# is emitted; final relationship needs are planned independently.
ENTITY_PRODUCERS = (
    "project_reference",
    "qualified_event_reference",
    "corroborated_reference",
)


def compose_plan(*, seed: int, ordinal: int, difficulty: str, hops: int,
                 goal_keys: tuple[str, ...] | list[str]) -> dict:
    if difficulty not in {"medium", "hard"}:
        raise ValueError("unsupported_composed_difficulty")
    if not 2 <= hops <= 3 or len(goal_keys) != hops:
        raise ValueError("invalid_composed_goal_depth")
    rng = random.Random(int(digest(
        seed, ordinal, tuple(goal_keys), VERSION
    )[:15], 16))
    # Select evidence acquisition capabilities AFTER the semantic Goal.
    # All producers are usable for every supported singleton entity Goal.
    producer = rng.choice(ENTITY_PRODUCERS)
    mode = {
        "project_reference": "trace",
        "qualified_event_reference": "event",
        "corroborated_reference": "verify",
    }[producer]
    condition = None if mode == "trace" else (
        {"field": "event_status", "op": "filter_eq", "value": "completed"}
        if rng.randrange(2) == 0 else
        {"field": "event_year", "op": "filter_gte", "value": 2025}
    )
    discoveries = 2 if difficulty == "medium" or mode == "verify" else 3
    tasks = []

    def add(role, op, dependencies, output, sources, need):
        prior = {row["macro_id"]: row for row in tasks}
        if any(mid not in prior for mid in dependencies):
            raise ValueError("composed_non_topological_dependency")
        mid = f"M{len(tasks) + 1}"
        tasks.append({
            "macro_id": mid, "role": role, "operation": op,
            "depends_on": list(dependencies),
            "input_contract": {
                parent: copy.deepcopy(prior[parent]["output_schema"])
                for parent in dependencies
            },
            "output_schema": dict(output),
            "source_scope": list(sources),
            "solver_template": op, "relation_need": need,
            "evidence_plan": {
                "description": need,
                "source_observation_targets": {sid: 1 for sid in sources},
            },
        })
        return mid

    def produce_candidates():
        # A branch is independently discoverable only if its sources are
        # browsable. Refinement branches require a proven upstream case_id;
        # their source permissions become keyed_lookup in _bind_sources.
        tips = []
        for i in range(discoveries):
            parent = rng.choice(tips) if tips and rng.random() < 0.58 else None
            deps = [parent] if parent else []
            mid = f"M{len(tasks) + 1}"
            tips = [x for x in tips if x != parent]
            tips.append(add(
                "discovery", "match_clues", deps, {"case_id": "str"},
                [mid + "_a", mid + "_b"],
                ("Refine the upstream candidates using case-keyed records"
                 if deps else "Discover candidates from complementary clues"),
            ))
        return tips

    produced = {}

    def require(port):
        if port in produced:
            return produced[port]
        if port == "project_id":
            leaves = produce_candidates()
            produced[port] = add(
                "identification", "identify_project", leaves,
                {"project_id": "str"}, ["entity_index"],
                "Resolve the uniquely identified project",
            )
        elif port == "entity_0":
            project = require("project_id")
            if producer == "project_reference":
                produced[port] = add(
                    "investigation", "reference", [project],
                    {"entity_0": "str"}, ["project_ref"],
                    "Look up the project subject",
                )
            elif producer == "qualified_event_reference":
                event = add(
                    "investigation", "qualifying_event", [project],
                    {"event_id": "str"}, ["event_ledger"],
                    "Find the project event meeting the required condition",
                )
                produced[port] = add(
                    "association", "event_reference", [event],
                    {"entity_0": "str"}, ["event_ref"],
                    "Resolve the eligible event's subject",
                )
            else:
                eligible = add(
                    "investigation", "event_eligibility", [project],
                    {"motif_object": "str"}, ["event_ledger"],
                    "Find the eligible relation object",
                )
                a = add(
                    "investigation", "directory_a", [project],
                    {"route_key": "str", "entity_0": "str"}, ["link_a"],
                    "Find candidate references in directory A",
                )
                b = add(
                    "investigation", "directory_b", [project],
                    {"route_key": "str", "entity_0": "str"}, ["link_b"],
                    "Find candidate references in directory B",
                )
                association = add(
                    "association", "intersect_references", [a, b],
                    {"entity_0": "str"}, [],
                    "Join independently observed reference candidates",
                )
                produced[port] = add(
                    "verification", "verify_real_fork",
                    [association, eligible], {"entity_0": "str"}, ["real_fork"],
                    "Check the candidate's recorded eligible relation",
                )
        else:
            raise ValueError("unsupported_information_need:" + port)
        return produced[port]

    previous = require("entity_0")
    # Partition the semantic relation path at real keyed information-handoff
    # boundaries; do not split a verification route beyond the Macro budget.
    if mode == "verify":
        groups = [list(range(1, hops + 1))]
    else:
        groups, position = [], 1
        budget = 9 - len(tasks)
        while position <= hops:
            remaining = hops - position + 1
            # Reserve room for later hops. Every produced boundary yields an
            # actual EntityRef key for subsequent source access.
            if budget > 1 and remaining > 1 and rng.random() < 0.60:
                width = 1
            else:
                width = remaining
            groups.append(list(range(position, position + width)))
            position += width
            budget -= 1
    for group in groups:
        tail = group[-1]
        previous = add(
            "synthesis" if tail == hops else "investigation",
            "follow_yago", [previous],
            {"answer" if tail == hops else f"entity_{tail}": "str"},
            [f"relation_{i}" for i in group],
            "Follow the goal's documented relationship information",
        )
    if len(tasks) > 9:
        raise ValueError("composed_macro_budget_exceeded")
    return {
        "version": "maestro.task_plan.v1",
        "composer_policy": VERSION,
        "goal_recipe_keys": list(goal_keys),
        "mode": mode, "condition": condition, "hops": hops,
        "difficulty_sampling_policy": "calibrated_structural_difficulty_v1",
        "evidence_sampling_policy": "eligibility_blind_event_names_v1",
        "question_sampling_policy": "grounded_relation_goal_question_v1",
        "macro_tasks": tasks, "terminal_macro_id": previous,
        "goal": {
            "answer_field": "answer", "answer_type": "str",
            "public_label": "related entity", "cardinality": "exactly_one",
            "meaning": "Determine the entity at the end of the documented relation chain",
        },
        "information_necessity_policy": "single_task_plan_counterfactual_v1",
    }
