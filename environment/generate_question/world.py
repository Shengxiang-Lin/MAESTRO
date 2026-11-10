"""Synthetic source World and executable Program compiler for Stage 1.

Every generated fact is fictional. Image/video/audio roles yield *structured
simulated annotations*, not analysis of original real media. Teacher entity
identity and gold program never appear in the initial public view.
"""
from __future__ import annotations

import random
from environment.schema import digest

from .program_ir import answer_values, evaluate_program
from .spec import spec_issues, VERSION as SPEC_VERSION

WORLD_VERSION = "maestro.source_observation_world.v4"


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


def _compile_approval_base(spec: dict) -> dict:
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
        "version": "maestro.reasoning_program.v4",
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


def initial_public_view(spec: dict, world: dict, question: str) -> dict:
    """Initial Main-Agent view; no original observations, answers, or program."""
    return {
        "task_id": spec["task_id"],
        "question": question,
        "available_source_descriptions": [{
            "source_id": sid,
            "title": item["title"],
            "modality_hint": item["modality"],
            **({"resource_url": item["resource_url"]}
               if "resource_url" in item else {}),
            **({} if spec.get("reasoning_pattern") == "macro_first_v2"
               else {"representation": item["representation"]}),
            **({"access_mode": item["access_policy"]["mode"]}
               if "access_policy" in item else {}),
        } for sid, item in sorted(world["sources"].items())],
        "max_concurrent_subagents": 10,
        "status": ("ready_for_investigation"
                   if spec.get("reasoning_pattern") == "macro_first_v2"
                   else "fictional_structured_multisource_world"),
    }


def build_scene(spec: dict, question: str) -> dict:
    """Full teacher scene; private gold execution and public projection."""
    program = compile_program(spec)
    world = generate_world(spec)
    execution = evaluate_program(program, sources=world["sources"])
    answer = answer_values(execution)
    return {
        "version": "maestro.generate_question.scene.v1",
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



def _finalize_current_program(program: dict, spec: dict) -> dict:
    """Make latest approval answers obey their explicit evidence contract."""
    program["version"] = "maestro.reasoning_program.v4"
    program["semantic_operator_pattern"] = "approval"
    if spec["family"] == "evidence_reconcile":
        terminal = program["output"]
        program["nodes"].append(
            _node("answer_with_evidence", "attach_evidence", terminal)
        )
        program["output"] = "answer_with_evidence"
    return program


def compile_program(spec: dict) -> dict:
    """Compile semantic branches into source-scoped executable programs."""
    if spec.get("reasoning_pattern") == "macro_first_v2":
        from .macro_planning import compile_macro_solvers
        return compile_macro_solvers(spec)
    if spec.get("reasoning_pattern") == "macro_first":
        from .macro_first import compile_macro_solvers
        return compile_macro_solvers(spec)
    if spec.get("reasoning_pattern") == "evidence_graph":
        from .graph_grammar import compile_evidence_program
        return compile_evidence_program(spec)
    if spec.get("reasoning_pattern") == "clue_chain":
        from .clue_chain import compile_clue_program
        return compile_clue_program(spec)
    if spec.get("reasoning_pattern") != "approval":
        from .pattern_world import compile_pattern_program
        return compile_pattern_program(spec)
    issues = spec_issues(spec)
    if issues:
        raise ValueError("invalid_spec:" + ",".join(issues))
    difficulty, family = spec["difficulty_target"], spec["family"]
    variant = int(digest(spec["seed"], spec["task_id"],
                         "program-variant")[:8], 16) % 4

    if difficulty == "easy" or variant == 0:
        program = _compile_approval_base(spec)
        return _finalize_current_program(program, spec)

    if variant == 1:
        # Apply active/approval filters only AFTER their two join boundaries.
        nodes = [
            _node("register_rows", "scan", source_id="register"),
            _node("glossary_rows", "scan", source_id="glossary"),
            _node("resolved_categories", "join",
                  "register_rows", "glossary_rows",
                  left_on="label_code", right_on="label_code"),
            _node("annotation_rows", "scan", source_id="annotations"),
            _node("joined_annotations", "join",
                  "resolved_categories", "annotation_rows",
                  left_on="category_name", right_on="category_name"),
            _node("active_entries", "filter_eq", "joined_annotations",
                  field="active", value=True),
            _node("approved_entries", "filter_eq", "active_entries",
                  field="approved", value=True),
        ]
    elif variant == 2:
        # Join filtered annotation labels to the glossary FIRST, then
        # bind the resulting category crosswalk to active entries.
        nodes = [
            _node("annotation_rows", "scan", source_id="annotations"),
            _node("approved_annotations", "filter_eq", "annotation_rows",
                  field="approved", value=True),
            _node("glossary_rows", "scan", source_id="glossary"),
            _node("approved_codebook", "join",
                  "glossary_rows", "approved_annotations",
                  left_on="category_name", right_on="category_name"),
            _node("register_rows", "scan", source_id="register"),
            _node("active_entries", "filter_eq", "register_rows",
                  field="active", value=True),
            _node("approved_entries", "join",
                  "active_entries", "approved_codebook",
                  left_on="label_code", right_on="label_code"),
        ]
    else:
        # Read and filter two source roots before reconciling them with
        # the already resolved register records.
        nodes = [
            _node("register_rows", "scan", source_id="register"),
            _node("active_entries", "filter_eq", "register_rows",
                  field="active", value=True),
            _node("annotation_rows", "scan", source_id="annotations"),
            _node("approved_annotations", "filter_eq", "annotation_rows",
                  field="approved", value=True),
            _node("glossary_rows", "scan", source_id="glossary"),
            _node("reviewed_codes", "join",
                  "approved_annotations", "glossary_rows",
                  left_on="category_name", right_on="category_name"),
            _node("approved_entries", "join",
                  "active_entries", "reviewed_codes",
                  left_on="label_code", right_on="label_code"),
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
        nodes += [
            _node("answer_entries", "project", final, fields=["entry_label"]),
            _node("distinct_answer", "unique", "answer_entries"),
        ]
        output = "distinct_answer"
    return _finalize_current_program({
        "version": "maestro.reasoning_program.v4",
        "nodes": nodes,
        "output": output,
        "authorized_source_ids": [
            source["source_id"] for source in spec["sources"]
        ],
        "frozen_problem_spec_id": spec["task_id"],
    }, spec)


def generate_world(spec: dict) -> dict:
    """Versioned synthetic sources; never rewrite old private World facts."""
    if spec.get("reasoning_pattern") == "macro_first_v2":
        from .macro_evidence_world import generate_world as generate_v2_world
        return generate_v2_world(spec)
    if spec.get("reasoning_pattern") == "macro_first":
        from .macro_first_world import generate_macro_first_world
        return generate_macro_first_world(spec)
    if spec.get("reasoning_pattern") == "evidence_graph":
        from .world_synthesizer import generate_evidence_world
        return generate_evidence_world(spec)
    if spec.get("reasoning_pattern") == "clue_chain":
        from .clue_chain import generate_clue_world
        return generate_clue_world(spec)
    if spec.get("reasoning_pattern") != "approval":
        from .pattern_world import generate_pattern_world
        return generate_pattern_world(spec)
    issues = spec_issues(spec)
    if issues:
        raise ValueError("invalid_spec:" + ",".join(issues))
    rng = random.Random(int(digest(spec["seed"], spec["task_id"],
                                   "fact-world-diverse-v3")[:15], 16))
    difficulty = spec["difficulty_target"]
    category_count = rng.randint(3, 6)
    entry_count = rng.randint(7, 12)
    codes = [f"{rng.choice('JKLMNPQRST')}-{i+1:02d}"
             for i in range(category_count)]
    kind = " ".join(spec["scenario"]["entity_kind"].replace("_", " ").split())
    if not kind:
        kind = "reference entries"
    categories = [
        f"{kind} group {chr(ord('A') + i)}"
        for i in range(category_count)
    ]
    indexed = list(range(category_count)) + [
        rng.randrange(category_count)
        for _ in range(entry_count - category_count)
    ]
    rng.shuffle(indexed)
    active = [rng.random() < rng.uniform(0.55, 0.80)
              for _ in range(entry_count)]
    review_count = rng.randint(1, max(1, category_count // 2))
    reviewed = set(rng.sample(list(range(category_count)), review_count))
    # Include an actual qualifying entry and an active negative witness.
    eligible = [i for i in range(entry_count)
                if active[i] and indexed[i] in reviewed]
    if not eligible:
        eligible_index = rng.randrange(entry_count)
        indexed[eligible_index] = min(reviewed)
        active[eligible_index] = True
    unreviewed_entries = [
        i for i in range(entry_count) if indexed[i] not in reviewed
    ]
    decoy = rng.choice(unreviewed_entries)
    active[decoy] = True
    eligible = [i for i in range(entry_count)
                if active[i] and indexed[i] in reviewed]
    # Make the single-source priority/time shortcuts demonstrably wrong:
    # the strongest raw record is an ACTIVE but unapproved decoy.
    priorities = rng.sample(range(100, 950), entry_count)
    positions = rng.sample(range(10, 950), entry_count)
    if difficulty != "easy":
        priorities[decoy] = 1001
        positions[decoy] = 1
    cleared = [False] * entry_count
    if difficulty == "hard":
        count = rng.randint(1, min(3, len(eligible)))
        for i in rng.sample(eligible, count):
            cleared[i] = True
        for i in range(entry_count):
            if i not in eligible and rng.random() < 0.2:
                cleared[i] = True
    entries = [{
        "entry_label": f"Entry-{i+1:02d}",
        "label_code": codes[indexed[i]],
        "active": active[i],
        "recorded_priority": priorities[i],
        "relative_step": positions[i],
    } for i in range(entry_count)]
    glossary = [{
        "label_code": code,
        "category_name": category,
    } for code, category in zip(codes, categories)]
    modality = next((role["modality"] for role in spec["sources"]
                     if role["source_id"] == "annotations"), "text")
    annotations = []
    for i, name in enumerate(categories, 1):
        display_field, display_value = _field_for_modality(modality, i)
        annotations.append({
            "category_name": name,
            "approved": (i - 1) in reviewed,
            display_field: display_value,
        })
    clearance = [{
        "entry_label": f"Entry-{i+1:02d}",
        "cleared": cleared[i],
    } for i in range(entry_count)]
    data = {
        "register": entries,
        "glossary": glossary,
        "annotations": annotations,
        "clearance": clearance,
    }
    registry = [{
        "hidden_entity_id": f"fictional-{i+1:03d}",
        "entry_label": item["entry_label"],
        "category_name": categories[indexed[i]],
        "identity_source": "synthetic_generation_only",
    } for i, item in enumerate(entries)]
    for role_data in data.values():
        rng.shuffle(role_data)  # avoid privileged first-row / F001 shortcuts
    sources = {
        role["source_id"]: _rows(
            role["source_id"], role["modality"], role["title"],
            role["purpose"], data[role["source_id"]]
        ) for role in spec["sources"]
    }
    result = {
        "version": WORLD_VERSION,
        "simulated_only": True,
        "sources": sources,
        "entities": registry,
        "generation_seed": spec["seed"],
        "value_sampling_contract": (
            "deterministic_relational_observations_after_frozen_spec"
        ),
    }
    return result
