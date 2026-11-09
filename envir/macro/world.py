"""Deterministic synthetic source observations for MMQA-derived Macro Tasks.

This is a *simulator*, not a media decoder. Each observed original MMQA source
handle receives distinct, question-scoped candidate/attribute fields that can
be consumed by the internal work-product evaluator. Never call synthetic
attributes facts about the original photographs, tables or text documents.
"""
from __future__ import annotations

import re
from envir.schema import digest


def _focus(question: str) -> str:
    """Compact label for the attribute being tested, without inventing facts."""
    text = " ".join(str(question).split())
    for regex in (
        r"\b(?:have|has|wearing|showing|contains?|depicts?)\s+(.{4,90}?)(?:\?|$)",
        r"\b(?:which|what)\s+(.{4,90}?)(?:\?|$)",
    ):
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .?!")[:90]
    return text.rstrip("?")[-90:]


def _synthetic_facts(*, case_ref: str, source: dict, question: str,
                     domain: str, topic: str) -> list[dict]:
    doc_id = str(source.get("original_doc_id") or source["source_id"])
    modality = source["modality_hint"]
    # Scores, feature flags and candidate labels below are synthetic test
    # observations. They are NOT predictions about the real MMQA media.
    score = int(digest(case_ref, doc_id, "attribute_score")[:8], 16) % 100
    time_index = 1 + int(digest(case_ref, doc_id, "relative_time")[:8], 16) % 40
    group = int(digest(topic, doc_id, "group")[:8], 16) % 3
    candidate_id = "candidate-" + digest(doc_id, topic)[:10]
    criterion = _focus(question)
    source_key = f"{modality}:{doc_id}"
    attributes = [
        ("case_reference", case_ref,
         f"The simulated source is assigned to evidence scope {case_ref}."),
        ("original_source_handle", source_key,
         f"The original MMQA support handle is {source_key}."),
        ("source_modality", modality,
         f"This role represents {modality} evidence; its actual media content was not decoded."),
        ("candidate_key", candidate_id,
         f"Simulated candidate {candidate_id} was extracted from reference {source_key}."),
        ("question_criterion", criterion,
         f"The candidate is checked against the question criterion '{criterion}'."),
        ("synthetic_attribute_score", score,
         f"Simulation assigns candidate {candidate_id} attribute score {score}."),
        ("synthetic_attribute_flag", score % 3 != 0,
         f"Simulation marks the requested attribute for {candidate_id} as "
         f"{'present' if score % 3 != 0 else 'absent'}."),
        ("synthetic_time_index", time_index,
         f"Simulation assigns candidate {candidate_id} relative time index {time_index}."),
        ("synthetic_group_key", f"group-{group}",
         f"Simulation associates {candidate_id} with matching group {group}."),
        ("observation_provenance", "simulated_not_raw_perception",
         "All candidate attributes are deterministic synthetic observations, not original MMQA annotations."),
    ]
    return [
        {"fact_id": f"F{i+1}", "field": key, "value": value,
         "statement": sentence, "verification": "consistent_synthetic_record"}
        for i, (key, value, sentence) in enumerate(attributes)
    ]


def _semantic_facts(*, case_ref: str, source: dict,
                    spec: dict, values: dict) -> list[dict]:
    """Bind a reviewed meaning to deterministic numeric AND relational facts.

    A source contains one observation of an entity. Observation ID,
    entity identity, shared category and directed source references are
    distinct typed fields; none is inferred from actual media.
    """
    semantic = spec["semantic_bindings"]
    role = next(
        row for row in spec["source_bindings"]
        if row["source_id"] == source["source_id"])
    label = role["candidate_label"]
    origin = (
        f"{source['modality_hint']}:"
        f"{source.get('original_doc_id') or source['source_id']}"
    )
    values_by_field = [
        ("case_reference", case_ref,
         f"This fictional record is in case {case_ref}."),
        ("original_source_handle", origin,
         f"Provenance handle {origin} is not an original-media observation."),
        ("source_modality", source["modality_hint"],
         f"Source category {source['modality_hint']} is a handle, not decoded media."),
        ("candidate_key", values["candidate_key"],
         f"{semantic['candidate_kind']} '{label}' has synthetic observation "
         f"ID {values['candidate_key']}."),
        ("question_criterion", spec["user_need"],
         f"This fictional collection supports: {spec['user_need']}"),
        ("synthetic_attribute_score", values["score"],
         f"Fictional {label}: {semantic['score']} = {values['score']} (0..100)."),
        ("synthetic_attribute_flag", values["positive_flag"],
         f"Fictional {label}: {semantic['positive_flag']} = "
         f"{values['positive_flag']}; not actual media recognition."),
        ("synthetic_time_index", values["relative_time_index"],
         f"Fictional {label}: {semantic['relative_time_index']} = "
         f"{values['relative_time_index']}; not a calendar date."),
        ("synthetic_group_key", values["group_key"],
         f"Fictional {label}: {semantic['group_key']} = "
         f"{values['group_key']}; group membership is NOT identity."),
        ("synthetic_entity_key", values["entity_key"],
         f"Fictional {label}: entity key {values['entity_key']} is an "
         "explicit simulated identity association, not a shared category."),
        ("synthetic_reference_target_source_ids",
         values["reference_target_source_ids"],
         f"Fictional {label}: references the explicitly recorded source "
         f"handles {values['reference_target_source_ids']}; a reference "
         "does NOT imply identical entities."),
        ("synthetic_precedes_source_ids", values["precedes_source_ids"],
         f"Fictional {label}: precedes source records "
         f"{values['precedes_source_ids']} according to synthetic sequence."),
        ("observation_provenance", "semantic_first_relational_simulation_only",
         "All values and relations are generated deterministically AFTER "
         "semantic schema approval, never decoded from original MMQA media."),
    ]
    return [
        {"fact_id": f"F{i+1}", "field": field, "value": value,
         "statement": statement, "verification": "consistent_synthetic_record"}
        for i, (field, value, statement) in enumerate(values_by_field)
    ]


def build_clean_world(plan: dict, *, seed: int,
                      contexts: dict | None = None,
                      semantic_spec: dict | None = None) -> dict:
    domain = plan["scenario"]["domain"]
    case_ref = f"{domain[:3].upper()}-{seed % 100:02d}-{plan['task_id'][-7:]}"
    question = plan["question"]  # ONLY the newly composed TaskGraph-derived question
    topic = plan["scenario"]["topic"]
    semantic_roles, relation_values = {}, {}
    if semantic_spec is not None:
        from .semantic_world import (
            validate_spec, source_manifest_from_plan,
        )
        problems = validate_spec(plan, semantic_spec)
        if problems:
            raise ValueError("invalid_semantic_world_spec:" + ";".join(problems))
        semantic_roles = {
            row["source_id"]: row for row in semantic_spec["source_bindings"]
        }
        from .relational_world import relational_layout
        manifest = source_manifest_from_plan(plan)
        relation_values = relational_layout(
            spec=semantic_spec,
            case_ref=case_ref,
            source_ids=[row["source_id"] for row in manifest],
            goal_id=plan["goal"]["intent"],
        )
    sources, grants = {}, {}
    for task in plan["macro_graph"]["tasks"]:
        granted = []
        for requirement in task.get("source_requirements", []):
            modality = requirement["candidate_modalities"][0]
            doc_id = requirement.get("original_doc_id")
            role = requirement["role"]
            description = requirement["description"]
            source_id = "src-" + digest(plan["task_id"], role, description)
            if source_id not in sources:
                source = {
                    "source_id": source_id,
                    "role": role,
                    "title": description,
                    "modality_hint": modality,
                    "representation": "synthetic_text_record_not_raw_media",
                    "original_doc_id": doc_id,
                    "source_origin": requirement.get("source_origin"),
                }
                if semantic_spec is not None:
                    interpretation = semantic_roles[source_id]
                    source["title"] = interpretation["source_role"]
                    source["semantic_candidate_label"] = interpretation[
                        "candidate_label"]
                    source["semantic_source_role"] = interpretation[
                        "source_role"]
                    source["representation"] = (
                        "synthetic_semantic_metadata_not_raw_media")
                    source["facts"] = _semantic_facts(
                        case_ref=case_ref, source=source,
                        spec=semantic_spec,
                        values=relation_values[source_id])
                else:
                    source["facts"] = _synthetic_facts(
                        case_ref=case_ref, source=source, question=question,
                        domain=domain, topic=topic,
                    )
                # In semantic-first mode, importing actual MMQA snippets
                # would pollute the synthetic World with a separately
                # sampled original question's potentially unrelated facts.
                context = (
                    None if semantic_spec is not None else
                    (contexts or {}).get((modality, str(doc_id)))
                )
                if context is not None:
                    source["representation"] = (
                        "synthetic_observations_plus_original_text_excerpt"
                    )
                    source["context_provenance"] = context["provenance"]
                    source["facts"].append({
                        "fact_id": f"F{len(source['facts']) + 1}",
                        "field": "original_context_excerpt",
                        "value": context["text"],
                        "statement": (
                            f"Exact-source MMQA {modality} context excerpt "
                            f"for {doc_id}: {context['text']}"
                        ),
                        "verification": "original_mmqa_text_excerpt_not_pixels",
                    })
                sources[source_id] = source
            granted.append(source_id)
        grants[task["task_id"]] = sorted(set(granted))
    if not sources:
        raise ValueError("no supporting MMQA source handles")
    result = {
        "world_id": "world-" + digest(
            plan["task_id"], (
                semantic_spec["proposal_fingerprint"] if semantic_spec
                else "synthetic_v46"
            )),
        "case_reference": case_ref,
        "as_of": "synthetic_not_timestamped",
        "sources": sources,
        "source_grants": grants,
        "source_truth": "deterministic_simulated_only",
        "real_multimodal_files": False,
        "observation_policy": (
            "semantic_schema_first_synthetic_values_not_original_mmqa"
            if semantic_spec else "synthetic_question_scoped_not_original_mmqa_facts"
        ),
    }
    if semantic_spec:
        from .semantic_world import fingerprint
        from .relational_world import world_relations
        result["semantic_spec_fingerprint"] = fingerprint(semantic_spec)
        result["relations"] = world_relations(sources)
    return result
