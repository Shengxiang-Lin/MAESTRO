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


def _semantic_facts(*, case_ref: str, source: dict, topic: str,
                    spec: dict, ordinal: int, source_count: int) -> list[dict]:
    """Sample values after semantics are frozen, not before.

    Numeric values are deterministic synthetic measurements, NEVER inferred
    from original MMQA media. Fact IDs/field names stay executable by the
    existing EvidenceGraph evaluator.
    """
    sid = source["source_id"]
    semantic = spec["semantic_bindings"]
    role = next(x for x in spec["source_bindings"] if x["source_id"] == sid)
    label = role["candidate_label"]
    spec_hash = digest(spec["proposal_fingerprint"], case_ref, "values")
    score = int(digest(spec_hash, sid, "score")[:8], 16) % 101
    positive = int(digest(spec_hash, sid, "positive")[:8], 16) % 2 == 0
    # Distinct source ordinal positions are assigned by the caller using
    # a reproducible shuffled order; no real historical date is implied.
    time_index = ordinal + 1
    # Small, explicitly simulated cohort inventory; some outcomes have
    # group overlap and others do not. Empty intersections remain valid.
    bucket_count = max(2, (source_count + 1) // 2)
    group = f"group-{int(digest(spec_hash, sid, 'cohort')[:8], 16) % bucket_count}"
    candidate = "candidate-" + digest(spec_hash, sid, "candidate")[:10]
    origin = f"{source['modality_hint']}:{source.get('original_doc_id') or sid}"
    facts = [
        ("case_reference", case_ref,
         f"This fictional evidence source belongs to case {case_ref}."),
        ("original_source_handle", origin,
         f"Source handle {origin} is a provenance pointer, not observed media."),
        ("source_modality", source["modality_hint"],
         f"{source['modality_hint']} is a source modality; no raw pixels, audio or video were decoded."),
        ("candidate_key", candidate,
         f"{semantic['candidate_kind']} '{label}' is the synthetic entry corresponding to candidate key {candidate}."),
        ("question_criterion", spec["user_need"],
         f"The simulated Goal is: {spec['user_need']}"),
        ("synthetic_attribute_score", score,
         f"Fictional {label}: {semantic['score']} = {score} (synthetic 0..100 scale)."),
        ("synthetic_attribute_flag", positive,
         f"Fictional {label}: {semantic['positive_flag']} = {positive}; this is NOT an observed original-media feature."),
        ("synthetic_time_index", time_index,
         f"Fictional {label}: {semantic['relative_time_index']} = {time_index}; this is an ordinal, NOT a real event date."),
        ("synthetic_group_key", group,
         f"Fictional {label}: {semantic['group_key']} = {group}; this denotes a cohort, not entity identity."),
        ("observation_provenance", "synthetic_semantic_first_not_raw_perception",
         "The source attributes were sampled after semantic schema approval and are not original MMQA annotations."),
    ]
    return [
        {"fact_id": f"F{i+1}", "field": field, "value": value,
         "statement": statement, "verification": "consistent_synthetic_record"}
        for i, (field, value, statement) in enumerate(facts)
    ]


def build_clean_world(plan: dict, *, seed: int,
                      contexts: dict | None = None,
                      semantic_spec: dict | None = None) -> dict:
    domain = plan["scenario"]["domain"]
    case_ref = f"{domain[:3].upper()}-{seed % 100:02d}-{plan['task_id'][-7:]}"
    question = plan["question"]  # ONLY the newly composed TaskGraph-derived question
    topic = plan["scenario"]["topic"]
    semantic_roles, ordinals = {}, {}
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
        manifest = source_manifest_from_plan(plan)
        order = sorted(
            (row["source_id"] for row in manifest),
            key=lambda sid: digest(
                semantic_spec["proposal_fingerprint"], sid, "ordinal")
        )
        ordinals = {sid: i for i, sid in enumerate(order)}
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
                        case_ref=case_ref, source=source, topic=topic,
                        spec=semantic_spec,
                        ordinal=ordinals[source_id],
                        source_count=len(ordinals))
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
        from .scene_grounding import fingerprint
        result["semantic_spec_fingerprint"] = fingerprint(semantic_spec)
    return result
