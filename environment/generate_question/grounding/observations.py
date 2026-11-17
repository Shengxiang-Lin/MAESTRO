"""Simulated perceptual descriptions grounded in executable source values.

Every claim in the readable annotation is a rephrasing of this very
observation's values; no new identity, event status, time threshold or
external historical fact may be introduced. This text improves the content
shown to Sub-Agents without being an alternative answer oracle.
"""
from __future__ import annotations


def _meaningful_value_fields(source_id: str, values: dict) -> list[tuple[str, object]]:
    skip = {"case_id", "project_id", "entity_final", "event_id",
            "route_key", "observation_id"}
    return [(key, val) for key, val in values.items() if key not in skip]


def _formatted_fields(items: list[tuple[str, object]]) -> str:
    labels = {
        "metric_a": "first measured count",
        "metric_b": "second measured count",
        "date_a": "first recorded date",
        "date_b": "second recorded date",
        "latitude_a": "site A latitude (degrees)",
        "longitude_a": "site A longitude (degrees)",
        "latitude_b": "site B latitude (degrees)",
        "longitude_b": "site B longitude (degrees)",
        "observation_year": "observation year",
        "cue": "identifying mark",
    }
    return "; ".join(
        labels.get(name, name.replace("_", " ")) + ": " + str(value)
        for name, value in items
    )


def annotate_source_observations(source: dict, role: dict, *,
                                 seed: int, ordinal: int) -> None:
    """Attach modality-specific, field-faithful annotations in-place.

    A simulated media observation is not a decoded binary asset. Exact
    values remain the authoritative IR inputs. The annotations must never
    name hidden candidate case IDs or reveal a correct answer.
    """
    modality, sid = role["modality"], role["source_id"]
    for observation in source["observations"]:
        values = observation["values"]
        fields = _meaningful_value_fields(sid, values)
        statement = _formatted_fields(fields)
        if modality == "image":
            if sid == "metric_a":
                description = (
                    "A photographed measurement sheet shows the following "
                    "readable entries: " + statement + "."
                )
                predicate = "visual_measurement_reading"
            else:
                description = (
                    "A photographed item contains a visible identifying "
                    "feature: " + statement + "."
                )
                predicate = "visual_feature_recognition"
        elif modality == "audio":
            if sid == "metric_b":
                description = (
                    'A speaker states the recorded measurements: "'
                    + statement + '."'
                )
                predicate = "spoken_measurement_extraction"
            else:
                description = (
                    'The recording contains this spoken identifier: "'
                    + statement + '."'
                )
                predicate = "spoken_identifier_extraction"
        elif modality == "video":
            description = (
                "A sequence of annotated footage contains the visible "
                "event cue: " + statement + "."
            )
            predicate = "video_event_recognition"
        elif modality == "table":
            description = (
                "A table row records " + (statement or
                    "its keyed reference to the case") + "."
            )
            predicate = "structured_record_reading"
        else:
            description = (
                "The written source states " + (statement or
                    "the indexed reference to the subject") + "."
            )
            predicate = "textual_evidence_extraction"
        observation["observed_content"] = description
        observation["observation_semantics"] = {
            "version": "maestro.simulated_observation.v1",
            "modality": modality,
            "predicate": predicate,
            "derived_from_value_fields": [name for name, _ in fields],
            "simulated_only": True,
        }
