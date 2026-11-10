"""Macro-FIRST task grammar, frozen contracts, and local Gold Solvers.

Macro task boundaries are sampled *before* primitive operations. Each Macro
has a typed handoff contract and its own executable local solver. The global
Program is only the topologically flattened replay of those local solvers.
"""
from __future__ import annotations

import copy
import random

from environment.schema import digest

VERSION = "maestro.macro_plan.v1"
PATTERNS = ("serial_discovery", "parallel_resolution", "conflict_reconciliation")
YEARS = ("record_year", "filing_year", "review_year")


def _source(sid, topic, modality, schema, *, key=None, purpose):
    return ({
        "source_id": sid, "title": f"{topic}: {sid.replace('_', ' ')} records",
        "modality": modality, "purpose": purpose,
        "access_mode": "keyed_lookup" if key else "browse",
        **({"key_field": key} if key else {}),
    }, schema)


def sample_macro_plan(*, seed: int, ordinal: int, record_id: str,
                      topic: str, difficulty: str) -> tuple[dict, list[dict]]:
    """First sample goal and Macro DAG; no Primitive program is available yet."""
    if difficulty not in {"medium", "hard"}:
        raise ValueError("macro_first_requires_medium_or_hard")
    rng = random.Random(int(digest(
        seed, ordinal, record_id, "macro-first-sampler-v1"
    )[:15], 16))
    pattern = rng.choice(PATTERNS)
    answer_field = rng.choice(YEARS)
    color = rng.choice(("crimson", "azure", "amber", "emerald", "violet"))
    motif = rng.choice(("crescent", "spiral", "trident", "sunburst", "twin-wing"))
    sources, schemas, tasks = [], {}, []

    def add_source(sid, modality, schema, *, purpose, key=None):
        row, fields = _source(sid, topic, modality, schema,
                              purpose=purpose, key=key)
        sources.append(row)
        schemas[sid] = fields

    def macro(role, depends_on, output, source_scope=()):
        number = len(tasks) + 1
        mid = f"M{number}"
        inputs = {parent: next(item["output_schema"]
                               for item in tasks if item["macro_id"] == parent)
                  for parent in depends_on}
        tasks.append({
            "macro_id": mid, "role": role, "depends_on": list(depends_on),
            "input_contract": inputs, "output_schema": dict(output),
            "source_scope": list(source_scope),
            "solver_template": role,
        })
        return mid

    if pattern == "serial_discovery":
        add_source("visual", rng.choice(("image", "video")),
                   {"candidate_id": "str", "emblem_color": "str",
                    "emblem_shape": "str", "marker_code": "str"},
                   purpose="independently identify an object from two visual clues")
        prior = macro("visual_identity", [], {"marker_code": "str"}, ["visual"])
        key_field = "marker_code"
    elif pattern == "parallel_resolution":
        add_source("visual", rng.choice(("image", "video")),
                   {"candidate_id": "str", "emblem_color": "str",
                    "marker_code": "str"},
                   purpose="identify a first set of candidate objects")
        add_source("caption", rng.choice(("text", "audio")),
                   {"candidate_id": "str", "emblem_shape": "str"},
                   purpose="independently identify a second set of candidates")
        left = macro("visual_candidates", [],
                     {"candidate_id": "str", "marker_code": "str"}, ["visual"])
        right = macro("caption_candidates", [],
                      {"candidate_id": "str"}, ["caption"])
        prior = macro("candidate_fusion", [left, right],
                      {"marker_code": "str"})
        key_field = "marker_code"
    else:
        add_source("report_a", "text",
                   {"case_id": "str", "reported_year_a": "int"},
                   purpose="provide one independent dated case report")
        add_source("report_b", "table",
                   {"case_id": "str", "reported_year_b": "int"},
                   purpose="provide a second independent dated case report")
        left = macro("first_report", [],
                     {"case_id": "str", "reported_year_a": "int"}, ["report_a"])
        right = macro("second_report", [],
                      {"case_id": "str", "reported_year_b": "int"}, ["report_b"])
        prior = macro("dispute_resolution", [left, right],
                      {"case_id": "str"})
        key_field = "case_id"

    # Hard tasks require an additional dependent Macro, not more trivial
    # primitive operations inside an otherwise identical Macro DAG.
    index_hops = 2 if pattern == "serial_discovery" and difficulty == "hard" else (
        1 if difficulty == "hard" or pattern == "serial_discovery" else 0
    )
    for i in range(1, index_hops + 1):
        new_key = f"record_key_{i}"
        sid = f"index_{i}"
        add_source(sid, "text", {key_field: "str", new_key: "str"},
                   purpose="resolve a discovered upstream identifier",
                   key=key_field)
        prior = macro(f"key_resolution_{i}", [prior], {new_key: "str"}, [sid])
        key_field = new_key

    add_source("archive", "table",
               {key_field: "str", answer_field: "int"},
               purpose="return the authoritative value for a proven identifier",
               key=key_field)
    terminal = macro("authoritative_result", [prior],
                     {answer_field: "int"}, ["archive"])
    plan = {
        "version": VERSION,
        "pattern": pattern,
        "difficulty": difficulty,
        "goal": {"answer_field": answer_field, "answer_type": "int",
                 "cardinality": "exactly_one"},
        "clues": ({"emblem_color": color, "emblem_shape": motif}
                  if pattern != "conflict_reconciliation" else None),
        "macro_tasks": tasks,
        "terminal_macro_id": terminal,
        "source_schemas": schemas,
        "index_hops": index_hops,
    }
    return plan, sources


def make_macro_first_spec(*, seed_record: dict, ordinal: int, run_seed: int,
                          difficulty: str, topic: str, record_id: str,
                          version: str) -> dict:
    plan, sources = sample_macro_plan(
        seed=run_seed, ordinal=ordinal, record_id=record_id,
        topic=topic, difficulty=difficulty,
    )
    return {
        "version": version,
        "task_id": f"question-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
        "scene_id": record_id,
        "source_anchor": {
            "dataset": "MultiModalQA_topic_only",
            "source_record_id": record_id,
            "original_question_reused": False,
            "original_answer_reused": False,
            "original_media_used": False,
        },
        "difficulty_target": difficulty,
        "family": "macro_first",
        "reasoning_pattern": "macro_first",
        "scenario": {
            "topic": topic,
            "setting": f"a fictional reference collection about {topic}",
            "user_role": "records investigator",
            "purpose": "resolve the authoritative record using independent clues",
            "motivation": "avoid choosing an unsupported identity or conflicting record",
            "entity_kind": "reference records",
        },
        "sources": sources,
        "macro_plan": plan,
        "requirements": {
            "must_resolve_aliases": False,
            "must_match_independent_reviews": False,
            "must_check_independent_clearance": False,
            "answer_kind": "macro_value",
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "seed": run_seed,
        "ordinal": ordinal,
    }


def validate_macro_first_spec(spec: dict) -> list[str]:
    """Fail closed on altered task boundaries, edges, source policies or types."""
    plan = spec.get("macro_plan")
    if not isinstance(plan, dict) or plan.get("version") != VERSION:
        return ["invalid_macro_plan"]
    try:
        expected, roles = sample_macro_plan(
            seed=spec["seed"], ordinal=spec["ordinal"],
            record_id=spec["source_anchor"]["source_record_id"],
            topic=spec["scenario"]["topic"],
            difficulty=spec["difficulty_target"],
        )
        errors = []
        if plan != expected:
            errors.append("macro_plan_not_reproducible_from_frozen_seed")
        source_contract = lambda x: (
            x.get("source_id"), x.get("modality"),
            x.get("access_mode"), x.get("key_field"),
        )
        if ([source_contract(x) for x in spec["sources"]]
                != [source_contract(x) for x in roles]):
            errors.append("macro_plan_source_scope_or_policy_modified")
        if plan.get("terminal_macro_id") != plan["macro_tasks"][-1]["macro_id"]:
            errors.append("macro_plan_terminal_mismatch")
        return errors
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return ["invalid_macro_contract:" + str(exc)[:100]]


def compile_macro_solvers(spec: dict) -> dict:
    """Construct local primitive solvers *inside* each frozen Macro boundary."""
    errors = validate_macro_first_spec(spec)
    if errors:
        raise ValueError("invalid_macro_first_spec:" + ",".join(errors))
    plan = spec["macro_plan"]
    nodes, groups, output_by_macro = [], [], {}
    color = (plan["clues"] or {}).get("emblem_color")
    motif = (plan["clues"] or {}).get("emblem_shape")
    answer_field = plan["goal"]["answer_field"]
    prior_key = ("case_id" if plan["pattern"] == "conflict_reconciliation"
                 else "marker_code")
    local_schema = {}

    for task in plan["macro_tasks"]:
        mid, role = task["macro_id"], task["role"]
        local = []

        def add(op, *inputs, **fields):
            nid = f"{mid.lower()}_op{len(local)+1:02d}"
            entry = {"id": nid, "op": op, "inputs": list(inputs), **fields}
            local.append(entry)
            return nid

        parents = [output_by_macro[pid] for pid in task["depends_on"]]
        if role in {"visual_identity", "visual_candidates"}:
            src = add("scan", source_id="visual")
            selected = add("filter_eq", src,
                           field="emblem_color", value=color)
            if role == "visual_identity":
                selected = add("filter_eq", selected,
                               field="emblem_shape", value=motif)
                last = add("project", selected, fields=["marker_code"])
            else:
                last = add("project", selected,
                           fields=["candidate_id", "marker_code"])
        elif role == "caption_candidates":
            src = add("scan", source_id="caption")
            matched = add("filter_eq", src,
                          field="emblem_shape", value=motif)
            last = add("project", matched, fields=["candidate_id"])
        elif role in {"first_report", "second_report"}:
            sid = "report_a" if role == "first_report" else "report_b"
            src = add("scan", source_id=sid)
            last = add("project", src, fields=[
                "case_id", ("reported_year_a" if sid == "report_a"
                            else "reported_year_b"),
            ])
        elif role == "candidate_fusion":
            aligned = add("join", *parents,
                          left_on="candidate_id", right_on="candidate_id")
            last = add("project", aligned, fields=["marker_code"])
        elif role == "dispute_resolution":
            aligned = add("join", *parents,
                          left_on="case_id", right_on="case_id")
            disagreed = add("compare_fields", aligned,
                            left_field="reported_year_a",
                            right_field="reported_year_b", comparison="ne")
            last = add("project", disagreed, fields=["case_id"])
        elif role.startswith("key_resolution_"):
            i = int(role.split("_")[-1])
            output_key = f"record_key_{i}"
            sid = f"index_{i}"
            lookup = add("lookup_by_key", *parents,
                         source_id=sid, input_key_field=prior_key,
                         source_key_field=prior_key)
            last = add("project", lookup, fields=[output_key])
            prior_key = output_key
        elif role == "authoritative_result":
            lookup = add("lookup_by_key", *parents,
                         source_id="archive", input_key_field=prior_key,
                         source_key_field=prior_key)
            projected = add("project", lookup, fields=[answer_field])
            last = add("unique", projected)
        else:
            raise ValueError("unknown_macro_solver_role:" + role)
        nodes.extend(local)
        groups.append({
            "macro_id": mid, "node_ids": [node["id"] for node in local],
            "output_node_id": last, "external_input_nodes": parents,
        })
        output_by_macro[mid] = last
        local_schema[mid] = task["output_schema"]

    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": "macro_first",
        "nodes": nodes,
        "output": output_by_macro[plan["terminal_macro_id"]],
        "authorized_source_ids": [
            row["source_id"] for row in spec["sources"]
        ],
        "frozen_problem_spec_id": spec["task_id"],
        "macro_solver_groups": groups,
        "macro_plan_version": VERSION,
    }


def validate_handoffs(scene: dict) -> dict:
    """Prove each frozen edge matters under a controlled empty handoff.

    Removing the *edge input product* differs from removing a source. Every
    edge must affect the terminal Gold Answer, not merely the execution graph.
    """
    from .compiler import execute_task
    from .program_ir import Relation
    from .dag import compile_waves

    graph = scene["macro_graph"]
    tasks = {row["task_id"]: row for row in graph["tasks"]}
    world = scene["world"]["sources"]
    expected = scene["gold_answer"]
    passed, edges = 0, []
    for child in graph["tasks"]:
        for parent_id in child["depends_on"]:
            products = {}
            for wave in graph["reference_waves"]:
                for tid in wave:
                    task = tasks[tid]
                    inputs = [
                        Relation(products[p].columns, tuple())
                        if tid == child["task_id"] and p == parent_id
                        else products[p]
                        for p in task["depends_on"]
                    ]
                    products[tid] = execute_task(
                        task, input_products=inputs,
                        visible_sources={sid: world[sid]
                                         for sid in task["source_ids"]},
                    )
            terminal = products[graph["terminal_task_id"]]
            got = sorted([row["values"] for row in terminal.rows],
                         key=lambda x: str(sorted(x.items())))
            sensitive = got != expected
            passed += int(sensitive)
            edges.append({"upstream": parent_id,
                          "downstream": child["task_id"],
                          "blank_handoff_changes_answer": sensitive})
    return {"edge_count": len(edges), "sensitive_edges": passed,
            "all_handoffs_necessary": passed == len(edges),
            "edges": edges}
