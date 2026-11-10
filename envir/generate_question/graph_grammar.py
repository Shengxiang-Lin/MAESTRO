"""Typed, seed-reproducible Evidence Acquisition Graph grammar.

This is an opt-in *semantic* graph sampler, not a random shuffle of Python
operators. Each plan is frozen before World synthesis, and every keyed access
must consume a typed upstream WorkProduct. No real media is decoded.
"""
from __future__ import annotations

import copy
import random

from envir.schema import digest

ARCHETYPES = ("chain", "parallel", "disambiguation", "conflict")
GRAPH_VERSION = "maestro.typed_evidence_graph.v1"
COLORS = ("crimson", "azure", "amber", "emerald", "violet", "silver")
MOTIFS = ("twin-wing", "crescent", "spiral", "trident", "sunburst", "chevron")


def _node(name: str, op: str, *inputs: str, **fields) -> dict:
    return {"id": name, "op": op, "inputs": list(inputs), **fields}


def _roles(topic: str, archetype: str, hops: int) -> tuple[list[dict], dict]:
    roles = []
    schemas = {}

    def add(sid, modality, fields, title, purpose, keyed=None):
        role = {"source_id": sid, "modality": modality,
                "title": f"{topic} {title}", "purpose": purpose,
                "access_mode": "keyed_lookup" if keyed else "browse"}
        if keyed:
            role["key_field"] = keyed
        roles.append(role)
        schemas[sid] = dict(fields)

    if archetype == "chain":
        add("illustration", "image",
            {"candidate_id": "str", "emblem_color": "str",
             "emblem_shape": "str", "marker_code": "str"},
            "illustrated exhibits", "pictured emblems and visible marker codes")
    elif archetype == "parallel":
        add("illustration", "image",
            {"candidate_id": "str", "emblem_color": "str"},
            "illustrated emblems", "pictured exhibit colors and identifiers")
        add("caption", "text",
            {"candidate_id": "str", "emblem_shape": "str",
             "marker_code": "str"},
            "caption notes", "independent motif descriptions and marker codes")
    elif archetype == "disambiguation":
        add("illustration", "image",
            {"candidate_id": "str", "emblem_color": "str",
             "marker_code": "str"},
            "illustrated exhibits", "ambiguous pictured colors and marker codes")
    else:
        add("report_a", "text",
            {"case_id": "str", "reported_year_a": "int"},
            "first inventory", "one report of each case year")
        add("report_b", "table",
            {"case_id": "str", "reported_year_b": "int"},
            "second inventory", "independently reported case years")

    if archetype == "disambiguation":
        # The second browse source is independent and can be dispatched
        # in parallel with the first image -> index chain.
        add("notes", "text",
            {"candidate_id": "str", "emblem_shape": "str"},
            "identification notes", "separate motif descriptions")

    key = "case_id" if archetype == "conflict" else "marker_code"
    for i in range(1, hops):
        next_key = f"key_{i}"
        fields = {key: "str", next_key: "str"}
        if archetype == "disambiguation" and i == 1:
            fields["candidate_id"] = "str"
        add(f"index_{i}", "text", fields,
            f"keyed index {i}", f"derive {next_key} from {key}", keyed=key)
        key = next_key
    add("archive", "table", {key: "str", "accession_year": "int"},
        "accession archive", "authoritative accession year by discovered key",
        keyed=key)
    return roles, schemas


def _program(archetype: str, hops: int, color: str, motif: str) -> tuple[list[dict], str]:
    nodes = []
    if archetype == "chain":
        nodes = [
            _node("illustration_rows", "scan", source_id="illustration"),
            _node("color_candidates", "filter_eq", "illustration_rows",
                  field="emblem_color", value=color),
            _node("pictured_exhibit", "filter_eq", "color_candidates",
                  field="emblem_shape", value=motif),
            _node("start_key", "project", "pictured_exhibit",
                  fields=["marker_code"]),
        ]
        previous = "start_key"
        key = "marker_code"
    elif archetype == "parallel":
        nodes = [
            _node("illustration_rows", "scan", source_id="illustration"),
            _node("color_candidates", "filter_eq", "illustration_rows",
                  field="emblem_color", value=color),
            _node("caption_rows", "scan", source_id="caption"),
            _node("motif_candidates", "filter_eq", "caption_rows",
                  field="emblem_shape", value=motif),
            _node("resolved_identity", "join", "color_candidates",
                  "motif_candidates", left_on="candidate_id",
                  right_on="candidate_id"),
            _node("start_key", "project", "resolved_identity",
                  fields=["marker_code"]),
        ]
        previous = "start_key"
        key = "marker_code"
    elif archetype == "disambiguation":
        nodes = [
            _node("illustration_rows", "scan", source_id="illustration"),
            _node("color_candidates", "filter_eq", "illustration_rows",
                  field="emblem_color", value=color),
            _node("start_key", "project", "color_candidates",
                  fields=["marker_code"]),
        ]
        previous = "start_key"
        key = "marker_code"
    elif archetype == "conflict":
        nodes = [
            _node("report_a_rows", "scan", source_id="report_a"),
            _node("report_b_rows", "scan", source_id="report_b"),
            _node("aligned_reports", "join", "report_a_rows", "report_b_rows",
                  left_on="case_id", right_on="case_id"),
            _node("disputed_cases", "compare_fields", "aligned_reports",
                  left_field="reported_year_a", right_field="reported_year_b",
                  comparison="ne"),
            _node("start_key", "project", "disputed_cases",
                  fields=["case_id"]),
        ]
        previous = "start_key"
        key = "case_id"
    else:
        raise ValueError("invalid_evidence_graph_archetype")

    for i in range(1, hops):
        sid = f"index_{i}"
        lookup = f"lookup_index_{i}"
        next_key = f"key_{i}"
        nodes.append(_node(lookup, "lookup_by_key", previous,
                           source_id=sid, input_key_field=key,
                           source_key_field=key))
        previous = lookup
        key = next_key

    if archetype == "disambiguation":
        # Select from multiple keyed candidates via an independent source.
        nodes.extend([
            _node("notes_rows", "scan", source_id="notes"),
            _node("motif_candidates", "filter_eq", "notes_rows",
                  field="emblem_shape", value=motif),
            _node("resolved_identity", "join", previous, "motif_candidates",
                  left_on="candidate_id", right_on="candidate_id"),
        ])
        previous = "resolved_identity"

    nodes.append(_node("archive_key", "project", previous, fields=[key]))
    nodes.append(_node("archive_lookup", "lookup_by_key", "archive_key",
                       source_id="archive", input_key_field=key,
                       source_key_field=key))
    nodes.append(_node("answer_year", "project", "archive_lookup",
                       fields=["accession_year"]))
    nodes.append(_node("unique_answer", "unique", "answer_year"))
    return nodes, "unique_answer"


def sample_graph(*, seed: int, ordinal: int, record_id: str,
                 difficulty: str, topic: str) -> dict:
    if difficulty not in ("medium", "hard"):
        raise ValueError("evidence_graph_requires_medium_or_hard")
    rng = random.Random(int(digest(
        seed, ordinal, record_id, "evidence-graph-grammar-v1"
    )[:15], 16))
    archetype = rng.choice(ARCHETYPES)
    if archetype == "chain":
        hops = rng.choice((2, 3)) if difficulty == "medium" else 3
    elif archetype == "disambiguation":
        hops = 2
    else:
        hops = 1 if difficulty == "medium" else 2
    color, motif = rng.choice(COLORS), rng.choice(MOTIFS)
    roles, schemas = _roles(topic, archetype, hops)
    nodes, output = _program(archetype, hops, color, motif)
    return {
        "version": GRAPH_VERSION, "archetype": archetype,
        "lookup_hops": hops,
        "visual_features": (
            {"emblem_color": color, "emblem_shape": motif}
            if archetype != "conflict" else None
        ),
        "source_schemas": schemas, "nodes": nodes, "output": output,
        "sources": roles,
        "answer_field": "accession_year",
    }


def make_evidence_spec(*, seed_record: dict, ordinal: int,
                       run_seed: int, difficulty: str,
                       topic: str, record_id: str, version: str) -> dict:
    graph = sample_graph(seed=run_seed, ordinal=ordinal,
                         record_id=record_id, difficulty=difficulty,
                         topic=topic)
    return {
        "version": version,
        "task_id": f"question-{ordinal:06d}-{digest(record_id, run_seed)[:8]}",
        "scene_id": record_id,
        "source_anchor": {
            "dataset": ("MultiModalQA_topic_only" if seed_record.get("record_id")
                        else "scene_bank_topic_only"),
            "source_record_id": record_id,
            "original_question_reused": False, "original_answer_reused": False,
            "original_media_used": False,
        },
        "difficulty_target": difficulty,
        "family": "evidence_answer",
        "reasoning_pattern": "evidence_graph",
        "macro_plan_variant": 0,
        "scenario": {
            "topic": topic, "setting": f"a fictional {topic} exhibit collection",
            "user_role": "collection researcher",
            "purpose": "determine a record's authoritative accession year",
            "motivation": "locate and reconcile clues across partial sources",
            "entity_kind": "exhibits",
        },
        "sources": graph.pop("sources"),
        "evidence_graph": graph,
        "requirements": {
            "must_resolve_aliases": False,
            "must_match_independent_reviews": False,
            "must_check_independent_clearance": False,
            "answer_kind": "accession_year",
            "must_preserve_source_provenance": True,
            "simulation_only": True,
        },
        "seed": run_seed, "ordinal": ordinal,
    }


def validate_evidence_spec(spec: dict) -> list[str]:
    """Static type-check the frozen plan without executing private World."""
    errors = []
    graph = spec.get("evidence_graph")
    if not isinstance(graph, dict) or graph.get("version") != GRAPH_VERSION:
        return ["invalid_evidence_graph"]
    if graph.get("archetype") not in ARCHETYPES:
        errors.append("unknown_evidence_graph_archetype")
    if graph.get("answer_field") != "accession_year":
        errors.append("wrong_evidence_graph_output")
    # Freeze the sampled grammar program itself, not only its type shape.
    # Scenario Author may rename human-facing titles but may never replace
    # an approved acquisition plan, key field or source topology.
    archetype = graph.get("archetype")
    hops = graph.get("lookup_hops")
    difficulty = spec.get("difficulty_target")
    allowed_hops = {
        "chain": (2, 3) if difficulty == "medium" else (3,),
        "parallel": (1,) if difficulty == "medium" else (2,),
        "disambiguation": (2,),
        "conflict": (1,) if difficulty == "medium" else (2,),
    }.get(archetype, ())
    if type(hops) is not int or hops not in allowed_hops:
        errors.append("invalid_evidence_graph_hop_count")
    visual = graph.get("visual_features")
    if archetype == "conflict":
        if visual is not None:
            errors.append("unexpected_conflict_visual_clue")
    elif (not isinstance(visual, dict)
          or visual.get("emblem_color") not in COLORS
          or visual.get("emblem_shape") not in MOTIFS):
        errors.append("invalid_evidence_visual_clue")
    if not errors:
        expected_nodes, expected_output = _program(
            archetype, hops,
            (visual or {}).get("emblem_color", ""),
            (visual or {}).get("emblem_shape", ""),
        )
        if (graph.get("nodes") != expected_nodes
                or graph.get("output") != expected_output):
            errors.append("evidence_graph_program_not_from_grammar")
        expected_roles, expected_schemas = _roles(
            spec["scenario"]["topic"], archetype, hops)
        if graph.get("source_schemas") != expected_schemas:
            errors.append("evidence_graph_source_schema_not_from_grammar")
        supplied_roles = spec.get("sources", [])
        source_contract = lambda r: (
            r.get("source_id"), r.get("modality"),
            r.get("access_mode"), r.get("key_field")
        )
        if ([source_contract(role) for role in supplied_roles]
                != [source_contract(role) for role in expected_roles]):
            errors.append("evidence_graph_source_policy_not_from_grammar")
    sources = spec.get("sources", [])
    schemas = graph.get("source_schemas", {})
    ids = [s.get("source_id") for s in sources if isinstance(s, dict)]
    if not isinstance(schemas, dict) or set(ids) != set(schemas):
        return ["evidence_graph_source_schema_mismatch"]
    roles = {row["source_id"]: row for row in sources}
    if (any(not isinstance(fields, dict) or
            not fields or set(fields.values()) - {"str", "int", "bool"}
            for fields in schemas.values())):
        return ["evidence_graph_invalid_source_types"]
    try:
        from .program_ir import validate_program
        program = compile_evidence_program(spec)
        errors.extend(validate_program(program))
        schemas_at, visited_sources = {}, set()
        for node in graph["nodes"]:
            op, parents, nid = node["op"], node["inputs"], node["id"]
            if any(parent not in schemas_at for parent in parents):
                raise ValueError("untyped_upstream")
            if op == "scan":
                sid = node["source_id"]
                if roles[sid]["access_mode"] != "browse":
                    raise ValueError("unlicensed_full_scan")
                result = dict(schemas[sid])
                visited_sources.add(sid)
            elif op == "lookup_by_key":
                sid = node["source_id"]
                source = schemas[sid]
                upstream = schemas_at[parents[0]]
                a, b = node["input_key_field"], node["source_key_field"]
                if (roles[sid]["access_mode"] != "keyed_lookup"
                        or roles[sid].get("key_field") != b
                        or upstream.get(a) != source.get(b)
                        or a not in upstream):
                    raise ValueError("incompatible_or_unlicensed_lookup_key")
                result = dict(upstream)
                for field, kind in source.items():
                    if field in result and result[field] != kind:
                        raise ValueError("lookup_column_type_conflict")
                    result[field] = kind
                visited_sources.add(sid)
            elif op == "join":
                left, right = (schemas_at[p] for p in parents)
                if left.get(node["left_on"]) != right.get(node["right_on"]) or (
                    node["left_on"] not in left
                ):
                    raise ValueError("incompatible_join_key")
                result = dict(left)
                for field, kind in right.items():
                    if field in result and result[field] != kind:
                        raise ValueError("join_column_type_conflict")
                    result[field] = kind
            elif op in {"filter_eq", "compare_fields", "project", "unique"}:
                upstream = schemas_at[parents[0]]
                if op == "filter_eq" and node["field"] not in upstream:
                    raise ValueError("missing_filter_field")
                if op == "compare_fields" and (
                    upstream.get(node["left_field"]) != upstream.get(
                        node["right_field"]) or
                    node["left_field"] not in upstream
                ):
                    raise ValueError("incompatible_comparison_types")
                if op == "project":
                    if not all(field in upstream for field in node["fields"]):
                        raise ValueError("missing_projected_field")
                    result = {field: upstream[field] for field in node["fields"]}
                else:
                    result = dict(upstream)
            else:
                raise ValueError("unsupported_evidence_graph_operation")
            schemas_at[nid] = result
        if set(visited_sources) != set(ids):
            errors.append("unused_or_unlicensed_evidence_source")
        if schemas_at[graph["output"]] != {"accession_year": "int"}:
            errors.append("evidence_graph_answer_type_mismatch")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        errors.append("evidence_graph_type_error:" + str(exc)[:90])
    return sorted(set(errors))


def compile_evidence_program(spec: dict) -> dict:
    graph = spec["evidence_graph"]
    return {
        "version": "maestro.reasoning_program.v4",
        "semantic_operator_pattern": "evidence_graph",
        "nodes": copy.deepcopy(graph["nodes"]),
        "output": graph["output"],
        "authorized_source_ids": [s["source_id"] for s in spec["sources"]],
        "frozen_problem_spec_id": spec["task_id"],
    }
