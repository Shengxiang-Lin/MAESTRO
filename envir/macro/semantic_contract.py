"""Frozen, fact-minimal semantic contract for Question realization only.

The contract is derived from *already compiled* typed Macro TaskGraph fields,
its independent Goal Bank entry and the public source inventory. Original
MMQA questions, source observations, answers and private feedback are NEVER
included. A naturalizer may restate the goal; it cannot change the required
operation, candidate universe, evidence boundaries, or final WorkProduct.

The contract is NOT a proof that a generated DAG's execution semantics are
sufficient. That separate task belongs to the next Task/Trace/Answer stage.
"""
from __future__ import annotations

import re

from .goal_grammar import GOALS


# Machine-readable interpretation of existing eight Goal Bank operations.
# Do not hallucinate extra field meanings (e.g. score != prestige).
GOAL_SEMANTICS = {
    "lookup": ("select_positive_flag", ("candidate_key", "synthetic_attribute_flag"),
               (), "source-backed candidates with a positive recorded flag"),
    "composition": ("link_candidate_references", ("candidate_key", "synthetic_group_key"),
                    (), "source-linked candidate-reference relationships"),
    "comparison": ("rank_by_recorded_score", ("candidate_key", "synthetic_attribute_score"),
                   (), "candidate score ordering and highest-scoring candidate"),
    "intersection": ("match_candidate_groups", ("candidate_key", "synthetic_group_key"),
                     (), "groups recurring across source records"),
    "aggregation": ("count_positive_flags", ("candidate_key", "synthetic_attribute_flag"),
                    (), "positive candidate count and contributing sources"),
    "temporal": ("sort_by_relative_time_index", ("candidate_key", "synthetic_time_index"),
                 (), "relative candidate sequence and earliest observation"),
    "prioritization": ("prioritize_by_recorded_score",
                       ("candidate_key", "synthetic_attribute_score"),
                       (), "candidate ordering by recorded scores"),
    "feasibility": ("filter_flag_and_score", (
                    "candidate_key", "synthetic_attribute_flag",
                    "synthetic_attribute_score"),
                    ("synthetic_attribute_flag=true",
                     "synthetic_attribute_score>=50"),
                    "candidates meeting both explicit simulated eligibility criteria"),
}


def semantic_contract(scene: dict) -> dict:
    """Immutable question-side contract; caller fingerprints this exact dict."""
    graph = scene["macro_graph"]
    intent = scene["goal"]["intent"]
    if intent not in GOALS or intent not in GOAL_SEMANTICS:
        raise ValueError("unknown goal intent")
    final = graph["terminal_handoffs"][0]
    task = next(t for t in graph["tasks"] if t["task_id"] == final["from_task"])
    goal = GOALS[intent]
    operation, fields, constraints, expected_answer = GOAL_SEMANTICS[intent]
    if (scene["goal"]["desired_artifact_type"] != goal["artifact"]
            or task["outputs"][final["output_slot"]] != goal["artifact"]
            or task["operation_spec"]["question_operation"] != intent):
        raise ValueError("terminal task and goal semantics disagree")
    sources = scene["public"]["available_source_descriptions"]
    # Source ids and modality are authorization boundaries, not observations.
    allowed_sources = [
        {
            "source_id": row["source_id"],
            "modality": row["modality_hint"],
            "description": row["title"],
            "representation": row["representation"],
        }
        for row in sources
    ]
    topic = scene["scenario"]["topic"]
    contract = {
        "contract_version": "question_semantics_v1",
        "authorized_topic": topic,
        "authorized_entities": [topic],
        "goal_id": intent,
        "operation": operation,
        "target_universe": "candidates present in the authorized source observations",
        "allowed_fields": list(fields),
        "hard_constraints": list(constraints),
        "final_workproduct_type": goal["artifact"],
        "required_answer": expected_answer,
        "source_inventory": allowed_sources,
        "allowed_modalities": sorted({row["modality"] for row in allowed_sources}),
        "source_policy": (
            "Use only the listed source roles. A source modality is not "
            "evidence of what its media shows."
        ),
        "simulation_policy": (
            "Candidate flags/scores/time indices are synthetic records. "
            "Do not reinterpret them as real prestige, importance, awards, "
            "dates, rankings, identity, or other unprovided world facts."
        ),
        "immutable_solution_contract": {
            "task_count": len(graph["tasks"]),
            "task_recipe_ids": [
                task["recipe_id"] for task in graph["tasks"]
            ],
            "work_types": [
                task["work_type"] for task in graph["tasks"]
            ],
            "typed_prerequisites": [
                {"producer_recipe": next(
                    t["recipe_id"] for t in graph["tasks"] if t["task_id"] == edge["from"]
                 ),
                 "consumer_recipe": next(
                    t["recipe_id"] for t in graph["tasks"] if t["task_id"] == edge["to"]
                 ),
                 "workproduct_type": edge["type"]}
                for edge in graph["edges"]
            ],
        },
    }
    world_spec = scene.get("semantic_world_spec")
    if world_spec and world_spec.get("status") == "accepted":
        contract["synthetic_world_specification"] = {
            "scenario": world_spec["scenario"],
            "semantic_bindings": world_spec["semantic_bindings"],
            "user_need": world_spec["user_need"],
            "goal_contract": world_spec["goal_contract"],
            "source_cards": [{
                "source_id": source["source_id"],
                "candidate_label": source["candidate_label"],
                "source_role": source["source_role"],
            } for source in world_spec["source_bindings"]],
            "provenance": (
                "hypothetical_schema_defined_before_simulated_values;"
                "source_facts_generated_by_deterministic_code"
            ),
        }
    grounding = scene.get("scene_grounding")
    if grounding and grounding.get("status") == "accepted" and (
            grounding.get("version") == "semantic_scenario_contract_v1"):
        # v4 carries an explicit, typed semantic contract shared by all
        # downstream writing stages. Its question_intent was constructed by
        # Python, not supplied by the model.
        contract["synthetic_scenario_contract"] = {
            "scenario": grounding["scenario"],
            "semantic_bindings": grounding["semantic_bindings"],
            "question_intent": grounding["question_intent"],
            "user_need": grounding["user_need"],
            "source_cards": [{
                "source_id": card["source_id"],
                "candidate_key": card["candidate_key"],
                "candidate_label": card["candidate_label"],
                "source_role": card["source_role"],
            } for card in grounding["source_cards"]],
            "provenance": "source_bound_hypothetical_synthetic_only",
        }
    elif grounding and grounding.get("status") == "accepted":
        # LLM interpretation is *hypothetical synthetic* context. The
        # Critic may permit these labels, but it must not assert original
        # media/real-world verification or change the core operation.
        contract["synthetic_scene_interpretation"] = {
            "status": "accepted_hypothetical_labels_only",
            "scene_context": grounding["scene_context"],
            "candidate_kind": grounding["candidate_kind"],
            "field_meanings": grounding["field_meanings"],
            "user_need": grounding["user_need"],
            "source_cards": [{
                "source_id": card["source_id"],
                "candidate_label": card["candidate_label"],
                "source_role": card["source_role"],
                "candidate_key": card["candidate_key"],
            } for card in grounding["source_cards"]],
        }
    return contract


def semantic_surface_issues(question: str, contract: dict) -> list[str]:
    """High-precision guardrails, NOT a full semantic entailment test."""
    text = str(question)
    lowered = text.lower()
    errors = []
    unsupported_personal_sources = (
        r"\bmy\s+(?:account|profile|dashboard|records|files|"
        r"subscriptions|inbox|photos?|screenshots?|report)\b",
        r"\b(?:our|my)\s+(?:internal|company|personal|private)\s+"
        r"(?:records|data|reports|sources)\b",
        r"\b(?:attached|uploaded|enclosed)\s+(?:file|image|report|"
        r"document|screenshot|photo)\b",
    )
    if any(re.search(pattern, lowered) for pattern in unsupported_personal_sources):
        errors.append("invented_personal_or_attached_source")
    # All numerical constraints must be present in the approved topic or
    # explicitly in the Goal Bank contract. "50" can be used for feasibility,
    # but not newly invented "2018" or "top 3".
    allowed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b",
        str(contract["authorized_topic"]) + " " +
        " ".join(contract["hard_constraints"])))
    candidate_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    if candidate_numbers - allowed_numbers:
        errors.append("unsupported_number_or_threshold")
    # A source handle with modality=image does not license a user-facing
    # claim about the image's observed content or type. A genuine, explicitly
    # named source subtype must appear in authorized metadata.
    licensed_source_text = " ".join(
        [contract["authorized_topic"]] +
        [row["description"] for row in contract["source_inventory"]]
    ).lower()
    named_unlicensed_source_types = (
        "poster", "screenshot", "surveillance footage", "news article",
        "bank statement", "account summary", "medical record",
    )
    if any(
        re.search(r"\b" + re.escape(kind) + r"s?\b", lowered)
        and not re.search(r"\b" + re.escape(kind) + r"s?\b", licensed_source_text)
        for kind in named_unlicensed_source_types
    ):
        errors.append("unsupported_specific_source_type")
    if re.search(
        r"\b(?:image|photo|photograph|poster|video|screenshot|footage)\b"
        r".{0,60}\b(?:shows?|depicts?|features?|contains?|reveals?)\b"
        r"|\b(?:shows?|depicts?)\b.{0,60}"
        r"\b(?:image|photo|photograph|poster|video|screenshot)\b",
        lowered,
    ):
        errors.append("unverified_media_content_claim")
    # This is an explicit high-precision forbidden semantic substitution.
    # Never translate a synthetic score to an unrelated real-world criterion.
    if contract["operation"] in {"rank_by_recorded_score",
                                 "prioritize_by_recorded_score"} and re.search(
        r"\b(?:prestigious|prestige|famous|popular|important|"
        r"successful|best\s+award|most\s+significant)\b", lowered
    ) and not re.search(r"\b(?:recorded|given)\s+"
                        r"(?:attribute\s+)?scores?\b", lowered):
        errors.append("unlicensed_real_world_ranking_criterion")
    return sorted(set(errors))
