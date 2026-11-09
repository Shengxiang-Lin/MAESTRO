"""Seeded Source–Entity–Observation pilot cases.

All subjects and observations are fictional. Image evidence means *simulated*
annotations, never image pixels or original MultiModalQA media perception.
The privileged program and entity registry are hidden from the public view.
"""
from __future__ import annotations

import random

from .program_ir import answer_values, evaluate_program


def _source(sid: str, modality: str, description: str,
            records: list[dict]) -> dict:
    if not records:
        raise ValueError("pilot sources must be nonempty")
    schema = list(records[0])
    if any(set(row) != set(schema) for row in records):
        raise ValueError("inconsistent_source_rows")
    return {
        "source_id": sid,
        "modality": modality,
        "description": description,
        "representation": "structured_simulated_observations_not_raw_media",
        "schema": schema,
        "observations": [{
            "observation_id": f"{sid}.O{i:03d}",
            "fact_refs": [f"{sid}:F{i:03d}"],
            "values": dict(row),
        } for i, row in enumerate(records, 1)],
    }


def build_linked_case(*, seed: int = 17, difficulty: str = "medium",
                      topic: str = "fictional roster audit") -> dict:
    """Create a relational case requiring real joins of independent sources.

    medium: roster -> code glossary -> simulated image review-board annotations.
    hard: adds a separate per-person clearance condition and join.
    """
    if difficulty not in {"medium", "hard"}:
        raise ValueError("pilot supports medium or hard difficulty only")
    codes = ["K7", "M2", "R4"]
    random.Random(seed).shuffle(codes)
    names = ("wide receiver", "cornerback", "tackle")
    positions = (0, 1, 0, 2)
    ledger = [
        {"player_label": f"Player-{i+1:02d}",
         "position_code": codes[pos], "active": i != 3}
        for i, pos in enumerate(positions)
    ]
    glossary = [
        {"position_code": codes[i], "position_name": name}
        for i, name in enumerate(names)
    ]
    board = [
        {"position_name": name, "approved": i == 0,
         "image_region": f"region-{i+1}"}
        for i, name in enumerate(names)
    ]
    sources = {
        "ledger": _source("ledger", "table",
                          "Simulated roster records with abbreviated positions", ledger),
        "codebook": _source("codebook", "text",
                            "Fictional position-code glossary", glossary),
        "board": _source("board", "image",
                         "Structured annotations for a fictional visual approval board",
                         board),
    }
    nodes = [
        {"id": "ledger_rows", "op": "scan", "inputs": [], "source_id": "ledger"},
        {"id": "active", "op": "filter_eq", "inputs": ["ledger_rows"],
         "field": "active", "value": True},
        {"id": "code_rows", "op": "scan", "inputs": [], "source_id": "codebook"},
        {"id": "resolved", "op": "join", "inputs": ["active", "code_rows"],
         "left_on": "position_code", "right_on": "position_code"},
        {"id": "board_rows", "op": "scan", "inputs": [], "source_id": "board"},
        {"id": "reviewed", "op": "filter_eq", "inputs": ["board_rows"],
         "field": "approved", "value": True},
        {"id": "matched", "op": "join", "inputs": ["resolved", "reviewed"],
         "left_on": "position_name", "right_on": "position_name"},
    ]
    final_node = "matched"
    if difficulty == "hard":
        sources["policy"] = _source(
            "policy", "text", "Separate fictional person-clearance notices",
            [{"player_label": f"Player-{i+1:02d}", "cleared": i == 0}
             for i in range(4)],
        )
        nodes.extend([
            {"id": "policy_rows", "op": "scan", "inputs": [], "source_id": "policy"},
            {"id": "cleared", "op": "filter_eq", "inputs": ["policy_rows"],
             "field": "cleared", "value": True},
            {"id": "final_match", "op": "join", "inputs": ["matched", "cleared"],
             "left_on": "player_label", "right_on": "player_label"},
        ])
        final_node = "final_match"
    nodes.extend([
        {"id": "answer", "op": "project", "inputs": [final_node],
         "fields": ["player_label"]},
        {"id": "unique_answer", "op": "unique", "inputs": ["answer"]},
    ])
    program = {
        "version": "maestro.reasoning_program.v1",
        "nodes": nodes,
        "output": "unique_answer",
        "source_universe": sorted(sources),
    }
    question = (
        f"In the {topic} records, which active players have positions "
        "approved in the simulated image review-board annotations after "
        "resolving the roster's abbreviated position codes?"
    )
    if difficulty == "hard":
        question = question[:-1] + (
            " Include only those whose separate clearance notices "
            "also mark them as cleared."
        )
    case = {
        "version": "maestro.source_entity_observation_pilot.v1",
        "seed": seed, "difficulty_target": difficulty,
        "synthetic_only": True, "topic": topic, "question": question,
        "world": {
            "entities": [{
                "entity_id": f"person-{i+1:02d}",
                "entity_type": "fictional_player",
                "label": f"Player-{i+1:02d}",
            } for i in range(4)],
            "sources": sources,
        },
        "gold_program": program,
    }
    case["gold_answer"] = answer_values(
        evaluate_program(program, sources=sources)
    )
    return case


def public_case(case: dict) -> dict:
    """Exclude hidden entities, source facts, program and gold answer."""
    return {
        "question": case["question"],
        "available_sources": [{
            "source_id": sid,
            "modality_hint": source["modality"],
            "description": source["description"],
        } for sid, source in sorted(case["world"]["sources"].items())],
        "synthetic_only": True,
    }
