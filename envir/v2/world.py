"""General V2 synthetic world factory and executable reasoning program compiler.

Every generated fact is fictional. Image/video/audio roles yield *structured
simulated annotations*, not analysis of original real media. Teacher entity
identity and gold program never appear in the initial public view.
"""
from __future__ import annotations

import random
from envir.schema import digest

from envir.macro.program_ir import answer_values, evaluate_program
from .spec import spec_issues

VERSION = "maestro.source_observation_world.v2"


def _rows(sid: str, modality: str, title: str, description: str,
          data: list[dict], *, original_handle: str | None = None) -> dict:
    if not data:
        raise ValueError("empty synthetic observation source")
    keys = list(data[0])
    if any(set(x) != set(keys) for x in data):
        raise ValueError("inconsistent_source_schema")
    return {
        "source_id": sid,
        "modality": modality,
        "title": title,
        "description": description,
        "representation": "synthetic_structured_observations_no_media_decoding",
        "original_mmqa_handle_used": False,
        "schema": keys,
        "observations": [{
            "observation_id": f"{sid}.OBS{i:03d}",
            "fact_refs": [f"{sid}:F{i:03d}"],
            "values": dict(entry),
        } for i, entry in enumerate(data, 1)],
    }


def _node(node_id: str, operation: str, *parents, **args) -> dict:
    return {"id": node_id, "op": operation, "inputs": list(parents), **args}


def compile_program(spec: dict) -> dict:
    """Compile multi-stage information requirements BEFORE any World sampling."""
    issues = spec_issues(spec)
    if issues:
        raise ValueError("invalid_spec:" + ",".join(issues))
    difficulty, family = spec["difficulty_target"], spec["family"]
    nodes = [
        _node("register_rows", "scan", source_id="register"),
        _node("active_entries", "filter_eq", "register_rows",
              field="active", value=True),
    ]
    final = "active_entries"
    if difficulty != "easy":
        nodes += [
            _node("glossary_rows", "scan", source_id="glossary"),
            _node("resolved_categories", "join", final, "glossary_rows",
                  left_on="label_code", right_on="label_code"),
            _node("annotation_rows", "scan", source_id="annotations"),
            _node("approved_annotations", "filter_eq", "annotation_rows",
                  field="approved", value=True),
            _node("approved_entries", "join",
                  "resolved_categories", "approved_annotations",
                  left_on="category_name", right_on="category_name"),
        ]
        final = "approved_entries"
    if difficulty == "hard":
        nodes += [
            _node("clearance_rows", "scan", source_id="clearance"),
            _node("cleared_entries", "filter_eq", "clearance_rows",
                  field="cleared", value=True),
            _node("fully_cleared", "join", final, "cleared_entries",
                  left_on="entry_label", right_on="entry_label"),
        ]
        final = "fully_cleared"
    if family == "priority":
        nodes += [
            _node("ranked", "sort", final, field="recorded_priority",
                  descending=True),
            _node("top_one", "take", "ranked", limit=1),
        ]
        final = "top_one"
    elif family == "chronology":
        nodes += [
            _node("ordered", "sort", final, field="relative_step",
                  descending=False),
            _node("earliest", "take", "ordered", limit=1),
        ]
        final = "earliest"
    if family == "category_distribution":
        nodes.append(_node("category_summary", "group_count", final,
                           field="category_name"))
        output = "category_summary"
    elif family == "coverage_count":
        nodes.append(_node("answer_count", "count", final))
        output = "answer_count"
    else:
        nodes.extend([
            _node("answer_entries", "project", final, fields=["entry_label"]),
            _node("distinct_answer", "unique", "answer_entries"),
        ])
        output = "distinct_answer"
    return {
        "version": "maestro.reasoning_program.v2",
        "nodes": nodes,
        "output": output,
        "authorized_source_ids": [
            source["source_id"] for source in spec["sources"]
        ],
        "frozen_problem_spec_id": spec["task_id"],
    }


def _field_for_modality(modality: str, index: int) -> tuple[str, object]:
    if modality == "image":
        return "region_label", f"region-{index:02d}"
    if modality == "video":
        return "segment_start_second", 15 * index
    if modality == "audio":
        return "utterance_id", f"utterance-{index:02d}"
    if modality == "text":
        return "text_line", 10 * index
    return "table_row", index


def generate_world(spec: dict) -> dict:
    """Sample values from independent source roles after freezing the Program."""
    issues = spec_issues(spec)
    if issues:
        raise ValueError("invalid_spec:" + ",".join(issues))
    rng = random.Random(int(digest(spec["seed"], spec["task_id"],
                                   "fact-world-v2")[:15], 16))
    codes = [f"{rng.choice('KLMNPRST')}{i+1}" for i in range(4)]
    if len(set(codes)) != len(codes):
        codes = [f"Q{i+1}" for i in range(4)]
    categories = [
        f"{spec['scenario']['entity_kind'].rstrip('s')} category {i+1}"
        for i in range(4)
    ]
    # Synthetically coherent but nonidentical records. The first two share
    # an alias. Other records serve as valid hard negatives, not fake steps.
    indexed = [0, 0, 1, 2, 3, 1, 2, 0]
    active = [True, True, True, False, True, False, True, False]
    relative_steps = [9, 4, 1, 2, 11, 7, 14, 18]
    priority = [84, 63, 95, 38, 77, 10, 72, 30]
    entries = [{
        "entry_label": f"Entry-{i+1:02d}",
        "label_code": codes[position],
        "active": active[i],
        "recorded_priority": priority[i] + rng.randrange(-3, 4),
        "relative_step": relative_steps[i],
    } for i, position in enumerate(indexed)]
    glossary = [{
        "label_code": code,
        "category_name": name,
    } for code, name in zip(codes, categories)]
    modality = next((x["modality"] for x in spec["sources"]
                     if x["source_id"] == "annotations"), "text")
    annotations = []
    for i, name in enumerate(categories, 1):
        display_field, display_value = _field_for_modality(modality, i)
        annotations.append({
            "category_name": name,
            "approved": i == 1,
            display_field: display_value,
        })
    clearance = [{
        "entry_label": f"Entry-{i+1:02d}",
        "cleared": i == 0,
    } for i in range(8)]
    data = {
        "register": entries,
        "glossary": glossary,
        "annotations": annotations,
        "clearance": clearance,
    }
    sources = {
        role["source_id"]: _rows(
            role["source_id"], role["modality"], role["title"],
            role["purpose"], data[role["source_id"]]
        ) for role in spec["sources"]
    }
    registry = [{
        "hidden_entity_id": f"fictional-{i+1:03d}",
        "entry_label": item["entry_label"],
        "category_name": categories[indexed[i]],
        "identity_source": "synthetic_generation_only",
    } for i, item in enumerate(entries)]
    return {
        "version": VERSION,
        "simulated_only": True,
        "sources": sources,
        "entities": registry,
        "generation_seed": spec["seed"],
        "value_sampling_contract": "deterministic_after_frozen_problem_spec",
    }


def initial_public_view(spec: dict, world: dict, question: str) -> dict:
    """Initial Main-Agent view; no original observations, answers, or program."""
    return {
        "task_id": spec["task_id"],
        "question": question,
        "available_source_descriptions": [{
            "source_id": sid,
            "title": item["title"],
            "modality_hint": item["modality"],
            "representation": item["representation"],
        } for sid, item in sorted(world["sources"].items())],
        "max_concurrent_subagents": 10,
        "status": "fictional_structured_multisource_world",
    }


def build_scene(spec: dict, question: str) -> dict:
    """Full teacher scene; private gold execution and public projection."""
    program = compile_program(spec)
    world = generate_world(spec)
    execution = evaluate_program(program, sources=world["sources"])
    answer = answer_values(execution)
    return {
        "version": "maestro.reasoning_v2_scene.v1",
        "task_id": spec["task_id"],
        "problem_spec": spec,
        "world": world,
        "gold_program": program,
        "gold_answer": answer,
        "gold_trace": execution["trace"],
        "question": question,
        "question_origin": "deterministic_frozen_problem_spec",
        "public": initial_public_view(spec, world, question),
        "execution_assumptions": {
            "all_subagents_reliable": True,
            "source_information_scoped": True,
            "real_media_perception_performed": False,
            "goal_program_hidden_from_main_agent": True,
        },
    }
