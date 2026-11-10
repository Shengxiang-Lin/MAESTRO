"""Typed Semantic Evidence Contract for new Macro-first V2 scenes.

Compile exact predicates and modalities from the executable Gold Program and
source schemas. Static language checks are deliberately high precision: they
block unsupported NEW requirements but do not pretend to prove all natural
language entailment. Legacy and previously frozen V2 scenes are unaffected.
"""
from __future__ import annotations

import re

from envir.schema import digest

CONTRACT_VERSION = "maestro.semantic_evidence.v1"

_COLORS = ("amber", "blue", "green", "red", "violet",
           "crimson", "azure", "emerald")
_CLEARANCE = re.compile(
    r"\b(?:clearance|cleared|clearance[- ]verified|clearance[- ]approved|"
    r"pre[- ]?approved|eligible[- ]only)\b", re.I,
)
_APPROVAL = re.compile(
    r"\b(?:approved|certified|authorized|certification)\b", re.I,
)
_ACTIVE = re.compile(
    r"\b(?:active[- ]only|must be active|marked active|active entries only)\b",
    re.I,
)
_AUDIO = re.compile(
    r"\b(?:audio|spoken|speaking|voice[- ]?over|voice[- ]?note|"
    r"commentary|utterance|audible|narration)\b", re.I,
)
_VIDEO = re.compile(
    r"\b(?:video|footage|video clip|on[- ]screen|broadcast footage|"
    r"film clip|frame|replay)\b", re.I,
)
_IMAGE = re.compile(
    r"\b(?:photograph|photo|still image|pictured|image shows|image depicting)\b",
    re.I,
)
_TIME_AMOUNT = re.compile(
    r"\b(?P<value>\d+|one|two|half)[ -]?(?P<unit>"
    r"seconds?|secs?|minutes?|mins?)\b", re.I,
)
_TIME_EXACT = re.compile(
    r"\b(?:exactly|precisely|within|no more than|at most|"
    r"less than|more than)\b.{0,45}"
    r"(?:\d+|one|two|half)[ -]?(?:seconds?|secs?|minutes?|mins?)\b",
    re.I,
)
_TIME_AUDIO = re.compile(
    r"\b(?:audio|spoken|cue|commentary|utterance)\b"
    r"[^.?!]{0,95}\b(?:at|within|after|before|around|by|"
    r"into|during|mark|later)\b[^.?!]{0,40}"
    r"(?:\d+|one|two|half)[ -]?(?:seconds?|secs?|minutes?|mins?)\b",
    re.I,
)
_REVERSE_ORDER = (
    re.compile(
        r"\b(?:footage|video|frame|action|visual event)\b[^.?!]{0,100}"
        r"\b(?:after|later than|following)\b[^.?!]{0,75}"
        r"\b(?:commentary|audio|spoken|cue|utterance)\b", re.I,
    ),
    re.compile(
        r"\b(?:commentary|audio|spoken|cue|utterance)\b[^.?!]{0,100}"
        r"\b(?:before|preceding|earlier than)\b[^.?!]{0,75}"
        r"\b(?:footage|video|frame|action|visual event)\b", re.I,
    ),
)
_GOAL_MISMATCH = re.compile(
    r"\b(?:founding year|year of founding|release year|award year|"
    r"year of release|graduation year)\b", re.I,
)
_EMBLEM = re.compile(r"\b(?:emblem|symbol|colou?r|colored|colour)\b", re.I)


def has_semantic_contract(spec: dict) -> bool:
    return (
        spec.get("reasoning_pattern") == "macro_first_v2"
        and spec.get("semantic_evidence_contract_version") == CONTRACT_VERSION
    )


def compile_semantic_contract(
    spec: dict, *, program: dict | None = None,
    retrospective: bool = False,
) -> dict:
    """Build one reproducible, non-LLM semantic contract from frozen Solver.

    The plan provides the typed Goal; actual primitive predicates and
    available modality/source fields MUST come from the executable program
    and the frozen source schemas, not from the LLM's scenario narration.
    """
    if not has_semantic_contract(spec):
        if not (retrospective
                and spec.get("reasoning_pattern") == "macro_first_v2"):
            raise ValueError("semantic_evidence_contract_not_enabled")
    if program is None:
        from .macro_first_v2 import compile_macro_solvers
        program = compile_macro_solvers(spec)
    source_by_id = {x["source_id"]: x for x in spec["sources"]}
    schema_by_id = spec["macro_plan"]["source_schemas"]
    if set(source_by_id) != set(schema_by_id):
        raise ValueError("semantic_source_schema_mismatch")
    provenance = {}
    predicates = []
    for node in program["nodes"]:
        op = node["op"]
        inputs = node["inputs"]
        scopes = set()
        for node_id in inputs:
            if node_id not in provenance:
                raise ValueError("semantic_non_topological_node:" + node_id)
            scopes.update(provenance[node_id])
        if op in {"scan", "lookup_by_key"}:
            sid = node["source_id"]
            if sid not in source_by_id:
                raise ValueError("semantic_unknown_source:" + sid)
            scopes.add(sid)
        provenance[node["id"]] = sorted(scopes)
        if op in {"filter_eq", "filter_gte", "compare_fields"}:
            if op == "compare_fields":
                predicates.append({
                    "operator": node["comparison"],
                    "left_field": node["left_field"],
                    "right_field": node["right_field"],
                    "source_scopes": sorted(scopes),
                })
            else:
                predicates.append({
                    "operator": "eq" if op == "filter_eq" else "gte",
                    "field": node["field"],
                    "value": node["value"],
                    "source_scopes": sorted(scopes),
                })
    if not predicates:
        raise ValueError("semantic_program_has_no_predicates")
    operators = {row["operator"] for row in predicates}
    schemas = {
        sid: {
            "modality": source_by_id[sid]["modality"],
            "available_fields": sorted(fields),
            "access_mode": source_by_id[sid]["access_mode"],
        } for sid, fields in schema_by_id.items()
    }
    applied = {row["field"] for row in predicates if "field" in row}
    time_thresholds = [
        row["value"] // 1000 for row in predicates
        if row.get("field") == "frame_time_ms"
        and row["operator"] == "gte"
        and row["value"] % 1000 == 0
    ]
    temporal_ordering = any(
        row.get("left_field") == "footage_time_ms"
        and row.get("right_field") == "commentary_time_ms"
        and row["operator"] == "lte" for row in predicates
    )
    # A field existing in a source is not evidence that the Solver checks it.
    clue_labels = {
        "emblem_color": "visual_emblem_color",
        "emblem_shape": "visual_emblem_shape",
        "scene_action": "video_action",
        "spoken_cue": "audio_spoken_cue",
        "caption_motif": "written_caption_motif",
        "case_tag": "event_log_tag",
    }
    necessary_clues = {
        clue_labels[predicate["field"]]: predicate["value"]
        for predicate in predicates
        if predicate.get("operator") == "eq"
        and predicate.get("field") in clue_labels
    }
    if time_thresholds:
        necessary_clues["video_minimum_seconds_inclusive"] = time_thresholds[0]
    if temporal_ordering:
        necessary_clues["video_event_not_after_audio_commentary"] = True
    if any(
        p.get("left_field") == "reported_year_a"
        and p.get("right_field") == "reported_year_b"
        and p.get("operator") == "ne"
        for p in predicates
    ):
        necessary_clues["independent_report_year_disagreement"] = True
    contract = {
        "version": CONTRACT_VERSION,
        "goal": dict(spec["macro_plan"]["goal"]),
        "sources": schemas,
        "available_modalities": sorted({
            source["modality"] for source in source_by_id.values()
        }),
        "applied_predicates": predicates,
        "applied_filter_fields": sorted(applied),
        "necessary_question_clues": necessary_clues,
        "temporal_semantics": {
            "frame_minimum_seconds_inclusive": time_thresholds,
            "footage_not_after_commentary": temporal_ordering,
            "exact_time_difference_supported": False,
        },
        "prohibited_new_filters": [
            flag for flag, aliases in (
                ("clearance", {"cleared", "clearance"}),
                ("approval", {"approved", "certified", "authorized"}),
                ("active", {"active"}),
            ) if not (aliases & applied)
        ],
        "synthesis_note": (
            "Internal-only simulated evidence; user-facing wording must "
            "describe a normal investigation without inventing external facts."
        ),
        "program_digest": digest(program["nodes"], program["output"]),
    }
    if spec.get("macro_information_necessity_version"):
        contract["required_output_fields"] = [
            spec["macro_plan"]["goal"]["answer_field"]
        ]
        contract["semantic_goal_type"] = spec["macro_plan"]["goal"]["answer_type"]
        contract["requested_entities"] = []
        contract["output_policy"] = "one_typed_answer_field_no_additional_output"
        if spec.get("scene_realism_version"):
            contract["output_scope_version"] = "maestro.one_response_slot.v2"
    if (spec.get("question_fidelity_version")
            and spec.get("scene_realism_version")):
        contract["question_fidelity_version"] = spec["question_fidelity_version"]
        contract["temporal_predicate_types"] = (
            [{"type": "frame_timestamp_inclusive_lower_bound", "seconds": n}
             for n in time_thresholds]
            + ([{"type": "video_event_not_after_audio_commentary"}]
               if temporal_ordering else [])
        )
    return contract


def _normalized_time_ms(value: str, unit: str) -> int:
    numbers = {"one": 1, "two": 2, "half": 0.5}
    numeric = numbers.get(value.lower())
    if numeric is None:
        numeric = int(value)
    return int(numeric * (60000 if unit.lower().startswith("min") else 1000))


def _remove_topic(text: str, spec: dict) -> str:
    """Avoid treating words in proper names (e.g. Blue...) as new clues."""
    topic = spec.get("scenario", {}).get("topic", "")
    return re.sub(re.escape(topic), " TOPIC ", text, flags=re.I) if topic else text


_OUTPUT_QUESTION_PATTERNS = {
    "episode_title": (
        r"\bwhich episode\b", r"\bidentify (?:the )?episode\b",
        r"\bname (?:the )?episode\b",
    ),
    "song_title": (r"\bwhich song\b", r"\bname (?:the )?song\b"),
    "match_result": (
        r"\bwhat (?:was|is) the (?:final )?(?:score|match result)\b",
        r"\bgive (?:the )?(?:final )?score\b",
    ),
    "event_venue": (
        r"\bwhich venue\b", r"\bwhere (?:was|is) the event held\b",
    ),
    "screening_venue": (
        r"\bwhich (?:screening )?venue\b",
        r"\bwhere (?:was|is) the screening held\b",
    ),
    "performance_venue": (
        r"\bwhich (?:performance )?venue\b",
        r"\bwhere (?:was|is) the performance held\b",
    ),
    "production_year": (r"\bwhat (?:is|was) (?:the )?production year\b",),
    "publication_year": (r"\bwhat (?:is|was) (?:the )?publication year\b",),
}


_COORDINATED_OUTPUT = (
    # A first identity requested by 'which case/record/episode' and a
    # second explicitly queried fact are two output slots, even though
    # the identity can also be a legitimate intermediate matching key.
    r"\bwhich\s+(?:(?:specific|matching|identified)\s+)?"
    r"(?:case|record|entry|event|episode|song|match|game|item|artifact|"
    r"report|person|team|exhibit|venue)\b[^?!.]{0,220}"
    r"\b(?:and|plus|as well as)\s+(?:what|which|where|who)\b"
)
_REVERSE_COORDINATED_OUTPUT = (
    r"\b(?:what|which|where|who)\b[^?!.]{0,180}"
    r"\b(?:and|plus|as well as)\s+which\s+(?:case|record|"
    r"entry|event|episode|song|match|game|item|artifact|report|exhibit)\b"
)
_IMPERATIVE_COORDINATED_OUTPUT = (
    r"\b(?:identify|name|list|give|report|provide)\s+(?:the\s+)?"
    r"(?:case|record|entry|event|episode|song|match|game|item|exhibit)\b"
    r"[^?!.]{0,180}\band\s+"
    r"(?:give|report|provide|return|identify|name|state)\b"
)


def semantic_output_issues(text: str, contract: dict) -> list[str]:
    """Conservative check for clearly requested second output slots.

    Descriptive mentions of entities are NOT treated as outputs; a direct
    question or an imperative to name a second entity is.
    """
    required = contract.get("required_output_fields")
    if not required:
        return []
    expected = required[0]
    if contract.get("output_scope_version") == "maestro.one_response_slot.v2":
        if any(re.search(pattern, text, re.I) for pattern in (
            _COORDINATED_OUTPUT, _REVERSE_COORDINATED_OUTPUT,
            _IMPERATIVE_COORDINATED_OUTPUT,
        )):
            return ["semantic_additional_output_requested"]
    requested = set()
    for field, expressions in _OUTPUT_QUESTION_PATTERNS.items():
        if any(re.search(expression, text, re.I) for expression in expressions):
            requested.add(field)
    extra = requested - {expected}
    if extra:
        return ["semantic_additional_output_field_requested"]
    # Direct two-part requests that may not name a known ontology slot.
    if re.search(
        r"\b(?:also|as well as|along with)\s+"
        r"(?:identify|name|list|return|report|give|provide)\b",
        text, re.I,
    ):
        return ["semantic_additional_output_requested"]
    return []


def semantic_text_issues(text: str, spec: dict, contract: dict,
                         *, context: str = "question") -> list[str]:
    """High-confidence checks of unsupported claims and predicate drift.

    This complements an independent semantic Critic; no closed vocabulary can
    prove natural-language equivalence. All violations name their dimension.
    """
    if not isinstance(text, str):
        return ["semantic_not_text"]
    q = _remove_topic(text, spec).lower()
    issues = []
    fields = set(contract["applied_filter_fields"])
    modalities = set(contract["available_modalities"])
    time = contract["temporal_semantics"]
    mode = spec["macro_plan"]["pattern"]

    if context == "question":
        issues.extend(semantic_output_issues(q, contract))
        if contract.get("question_fidelity_version"):
            from .question_fidelity import question_fidelity_issues
            issues.extend(question_fidelity_issues(text, contract))

    if "clearance" in contract["prohibited_new_filters"] and _CLEARANCE.search(q):
        issues.append("semantic_unlicensed_clearance")
    if "approval" in contract["prohibited_new_filters"] and _APPROVAL.search(q):
        issues.append("semantic_unlicensed_approval")
    if "active" in contract["prohibited_new_filters"] and _ACTIVE.search(q):
        issues.append("semantic_unlicensed_active_filter")
    if _GOAL_MISMATCH.search(q):
        if not any(x in contract["goal"]["public_label"]
                   for x in ("founding", "release", "award", "graduation")):
            issues.append("semantic_goal_event_substitution")

    if "audio" not in modalities and _AUDIO.search(q):
        issues.append("semantic_unsupported_audio")
    if "video" not in modalities and _VIDEO.search(q):
        issues.append("semantic_unsupported_video")
    if "image" not in modalities and _IMAGE.search(q):
        issues.append("semantic_unsupported_image")
    if "emblem_color" not in fields:
        if re.search(r"\b(?:" + "|".join(_COLORS) + r")\b", q, re.I):
            issues.append("semantic_unapplied_color_clue")
    # Video footage has no emblem observations; an audio's spoken 'star'
    # does not make a visually observed star emblem valid.
    if "image" not in modalities and _EMBLEM.search(q):
        issues.append("semantic_unobserved_visual_emblem")
    elif ("image" in modalities and "emblem_shape" not in fields
          and mode in {"parallel_cues", "multimodal_event"}):
        # Avoid falsely attributing the independent audio/text cue to the
        # image itself. The LLM Critic handles subtler attribution variants.
        if (re.search(
            r"\b(?:image|photo|picture)\b[^.?!]{0,70}"
            r"\b(?:star|spiral|crescent|triangle|wave)\s+"
            r"(?:shaped\s+)?(?:emblem|symbol)\b", q,
        ) or re.search(
            r"\b(?:image|photo|picture)\b[^.?!]{0,70}"
            r"\b(?:emblem|symbol)\s+"
            r"(?:with|showing|shaped|depicting)\s+(?:a\s+)?"
            r"(?:star|spiral|crescent|triangle|wave)\b", q,
        )):
            issues.append("semantic_unapplied_visual_shape")

    amounts = [(_normalized_time_ms(m.group("value"), m.group("unit")), m)
               for m in _TIME_AMOUNT.finditer(q)]
    allowed_ms = {seconds * 1000 for seconds in
                  time["frame_minimum_seconds_inclusive"]}
    for millis, _match in amounts:
        if millis not in allowed_ms:
            issues.append("semantic_unsupported_exact_time")
    if amounts and not allowed_ms:
        issues.append("semantic_unsupported_numeric_time")
    if amounts and _TIME_EXACT.search(q):
        issues.append("semantic_unsupported_time_interval")
    if amounts and _TIME_AUDIO.search(q):
        issues.append("semantic_unsupported_audio_timestamp")
    if allowed_ms and amounts and re.search(
        r"\b(?:before|earlier than|no later than|strictly after)\b"
        r"[^.?!]{0,42}(?:\d+|one|two|half)[ -]?"
        r"(?:seconds?|secs?|minutes?|mins?)\b", q
    ):
        issues.append("semantic_reversed_frame_threshold")
    if ("audio" in modalities and "video" in modalities
            and not time["footage_not_after_commentary"]):
        # Parallel branches do not implement a cross-modal time comparison.
        if (re.search(
            r"\b(?:footage|video|frame|action)\b[^.?!]{0,45}"
            r"\b(?:happens|occurs|appears|takes place|is shown|"
            r"was shown|is seen)\s+(?:before|after|following)\s+"
            r"(?:the\s+)?(?:spoken|audio|commentary|cue)\b", q,
        ) or re.search(
            r"\b(?:footage|video|frame|action)\b[^.?!]{0,40}"
            r"\b(?:before|after)\s+(?:the\s+)?"
            r"(?:spoken|audio|commentary|cue)\b", q,
        ) or re.search(
            r"\b(?:spoken|audio|commentary|cue)\b[^.?!]{0,40}"
            r"\b(?:before|after)\s+(?:the\s+)?"
            r"(?:video|footage|frame|action)\b", q,
        )):
            issues.append("semantic_unlicensed_crossmodal_order")
    if time["footage_not_after_commentary"]:
        # "no later than" is the faithful rendering of <=, NOT the
        # reversed relation "later than". Mask negated comparison phrases
        # before applying the historical high-precision reversal guard.
        ordering_text = re.sub(
            r"\bno\s+later\s+than\b", "not_after", q
        )
        if any(pattern.search(ordering_text) for pattern in _REVERSE_ORDER):
            issues.append("semantic_reversed_multimodal_order")
        if re.search(r"\b(?:within|exactly|precisely|no more than|at most)\b"
                     r"[^.?!]{0,30}\b(?:time|seconds?|minutes?|interval|window)\b",
                     q):
            issues.append("semantic_unlicensed_temporal_window")
    # A source title may be descriptive but must not imply an unavailable
    # modality or a non-existent clearance/checking system.
    if context == "source_title":
        if "image" not in modalities and _IMAGE.search(q):
            issues.append("semantic_source_title_wrong_modality")
    return sorted(set(issues))


def semantic_source_title_issues(source_id: str, title: str,
                                 spec: dict, contract: dict) -> list[str]:
    if source_id not in contract["sources"]:
        return ["semantic_source_title_unknown_source"]
    issues = semantic_text_issues(title, spec, contract, context="source_title")
    modality = contract["sources"][source_id]["modality"]
    # Source-local mismatches should not pass merely because another source
    # of the correct modality is present elsewhere in the task.
    if modality != "audio" and re.search(
        r"\b(?:audio|spoken|voice[- ]?over|sound recording)\b", title, re.I
    ):
        issues.append("semantic_source_title_wrong_modality")
    if modality != "video" and re.search(
        r"\b(?:video|footage|film clip)\b", title, re.I
    ):
        issues.append("semantic_source_title_wrong_modality")
    if modality != "image" and re.search(
        r"\b(?:photo|photograph|still image)\b", title, re.I
    ):
        issues.append("semantic_source_title_wrong_modality")
    return sorted(set(issues))


def semantic_scenario_issues(spec: dict, contract: dict) -> list[str]:
    narrative = spec["scenario"]
    lines = [narrative.get(field, "") for field in (
        "setting", "purpose", "motivation", "entity_kind"
    )]
    issues = []
    for item in lines:
        issues.extend(semantic_text_issues(item, spec, contract,
                                           context="scenario"))
    for source in spec["sources"]:
        issues.extend(semantic_source_title_issues(
            source["source_id"], source["title"], spec, contract
        ))
    return sorted(set(issues))
