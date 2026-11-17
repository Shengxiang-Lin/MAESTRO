"""Unified Stage-1 scene grounding: contextual source profiles and modality semantics.

All synthetic cue vocabularies, natural goal referents and readable source
observations belong to this single representation layer. Source-value truth
remains in the canonical World/Gold IR; descriptions cannot add facts.
"""
from __future__ import annotations



# --- CLUES ---

_PROFILES = {
    "biographical": (
        (("image", "portrait artifact photograph", "portrait emblem",
          ("laurel", "compass", "torch", "crown", "ribbon")),
         ("table", "historical biography index", "biography collection mark",
          ("folio", "memoir", "register", "chronicle", "volume"))),
        (("audio", "oral history recording", "spoken interview cue",
          ("journey", "festival", "crossing", "heritage", "portrait")),
         ("text", "historical profile index", "profile series",
          ("correspondence", "fieldnote", "memoir", "chronology", "recollection"))),
        (("video", "documentary footage", "documentary scene action",
          ("arrival", "farewell", "interview", "ceremony", "departure")),
         ("text", "curatorial notes", "document role",
          ("exhibit", "interview", "review", "chronicle", "correspondence"))),
    ),
    "academic": (
        (("image", "research poster photograph", "experimental symbol",
          ("wave", "grid", "helix", "prism", "spiral")),
         ("table", "laboratory reference sheet", "lab index code",
          ("alpha", "delta", "sigma", "kappa", "eta"))),
        (("audio", "seminar recording", "spoken research cue",
          ("control", "baseline", "replicate", "sample", "calibration")),
         ("text", "research correspondence", "study series",
          ("methods", "pilot", "protocol", "appendix", "fieldwork"))),
        (("video", "experiment footage", "experiment action",
          ("setup", "measurement", "mixing", "heating", "cooling")),
         ("text", "lab notebook register", "protocol category",
          ("trial", "review", "analysis", "replication", "setup"))),
    ),
    "institutional": (
        (("image", "facility sign photograph", "organization insignia",
          ("gear", "bridge", "tower", "arch", "star")),
         ("table", "dispatch and permit index", "case register section",
          ("ledger", "permit", "invoice", "registry", "docket"))),
        (("audio", "operations briefing", "spoken operational cue",
          ("transfer", "delivery", "inspection", "handover", "review")),
         ("text", "business report record", "operations series",
          ("staffing", "dispatch", "compliance", "planning", "operations"))),
        (("video", "facility activity footage", "observed site activity",
          ("arrival", "loading", "departure", "handoff", "inspection")),
         ("text", "organization activity report", "activity class",
          ("meeting", "shipment", "review", "inspection", "transfer"))),
    ),
    "geographical": (
        (("image", "survey site photograph", "visible geographic landmark",
          ("bridge", "harbor", "ridge", "valley", "tower")),
         ("table", "survey station list", "survey sheet code",
          ("coastal", "upland", "western", "central", "northern"))),
        (("audio", "field researcher recording", "spoken site descriptor",
          ("shoreline", "crossing", "summit", "estuary", "junction")),
         ("text", "field mapping summary", "map layer",
          ("routes", "terrain", "boundaries", "elevation", "watercourse"))),
        (("video", "site survey footage", "recorded field event",
          ("traverse", "crossing", "arrival", "turn", "departure")),
         ("text", "regional inspection notes", "station class",
          ("sector", "corridor", "waypoint", "sample", "transect"))),
    ),
}


def clue_profiles(domain: str) -> tuple:
    try:
        return _PROFILES[domain]
    except KeyError as exc:
        raise ValueError("unsupported_clue_domain:" + str(domain)) from exc


# --- GOAL_LANGUAGE ---

_FOCUS = {
    ("birthplace", "containedinplace"):
        "the administrative area containing the identified person's birthplace",
    ("birthplace", "location"):
        "the location recorded for the identified person's birthplace",
    ("deathplace", "location"):
        "the location recorded for the identified person's place of death",
    ("doctoraladvisor", "fieldofwork"):
        "the research field of the identified scholar's doctoral advisor",
    ("doctoraladvisor", "birthplace"):
        "the birthplace of the identified scholar's doctoral advisor",
    ("spouse", "birthplace"):
        "the birthplace of the identified person's spouse",
    ("memberof", "location"):
        "the location of the organization affiliated with the identified entity",
    ("location", "containedinplace"):
        "the administrative area containing the identified entity's location",
    ("containedinplace", "containedinplace"):
        "the larger administrative area containing the region of the identified place",
    ("neighbor", "location"):
        "the location of a neighboring place of the identified place",
    ("location", "neighbor"):
        "the place neighboring the identified entity's location",
    ("birthplace", "containedinplace", "containedinplace"):
        "the larger administrative area containing the region of the identified person's birthplace",
    ("birthplace", "location", "containedinplace"):
        "the administrative area containing the recorded location of the identified person's birthplace",
    ("doctoraladvisor", "birthplace", "location"):
        "the location recorded for the birthplace of the identified scholar's doctoral advisor",
    ("doctoraladvisor", "birthplace", "containedinplace"):
        "the administrative area containing the birthplace of the identified scholar's doctoral advisor",
    ("memberof", "location", "containedinplace"):
        "the administrative area containing the location of the identified entity's affiliated organization",
    ("location", "containedinplace", "containedinplace"):
        "the larger administrative area containing the region of the identified entity's location",
    ("spouse", "birthplace", "containedinplace"):
        "the administrative area containing the birthplace of the identified person's spouse",
    ("deathplace", "location", "containedinplace"):
        "the administrative area containing the recorded location of the identified person's place of death",
    ("nationality", "capital", "location"):
        "the location of the capital of the identified person's nationality",
    ("location", "neighbor", "location"):
        "the location recorded for the place neighboring the identified entity's location",
    ("deathplace", "location", "neighbor"):
        "the area neighboring the location of the identified person's place of death",
    ("location", "capital", "twinnedwith"):
        "the municipality twinned with the capital of the identified entity's location",
}


def focus_phrase(keys: tuple[str, ...]) -> str:
    try:
        return _FOCUS[keys]
    except KeyError as exc:
        # Fail closed, rather than silently dropping a semantic relation.
        raise ValueError("unsupported_natural_goal_focus:" + repr(keys)) from exc


def focus_coverage() -> frozenset[tuple[str, ...]]:
    return frozenset(_FOCUS)


# --- OBSERVATIONS ---

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


# --- MEASUREMENT_SOURCES ---

# Topic variants are *types of legitimate investigations*, not arbitrary
# labels pasted onto an unrelated person/place relationship.
_CONTEXT = {
    "biographical": (
        ("oral-history field study", "historical biography investigation",
         "cultural heritage research", "sports biography survey"),
        "biographical investigator",
        "a biographical investigation connecting person records and field observations",
        "identify the subject before comparing independently documented observations",
    ),
    "academic": (
        ("research experiment review", "academic correspondence study",
         "scientific publication investigation", "education history survey"),
        "research coordinator",
        "an academic investigation connecting scholar records to measured evidence",
        "resolve the researcher and the applicable observations before reporting",
    ),
    "institutional": (
        ("organization operations audit", "business affiliation review",
         "institutional transport survey", "civic services investigation"),
        "operations analyst",
        "an organizational investigation using membership and location evidence",
        "resolve the related organization before comparing its documented facts",
    ),
    "geographical": (
        ("regional geography study", "environmental field survey",
         "transport route investigation", "urban infrastructure review"),
        "field survey analyst",
        "a spatial investigation involving places and independent observations",
        "locate the area before interpreting comparative source measurements",
    ),
}


def scene_profile(domain: str, *, ordinal: int, goal_label: str,
                  output_kind: str) -> dict:
    """Plausible user intent, never a new source fact or solver requirement."""
    topics, role, setting, motivation = _CONTEXT[domain]
    selected_topic = topics[ordinal % len(topics)]
    purpose = (
        "identify the correct subject and derive the requested " + goal_label
        + " from separately documented evidence"
    )
    if output_kind == "evidence_based_explanation":
        purpose = (
            "compare two independently observed counts, explain which is "
            "larger and retain their reported evidence values"
        )
    return {
        "topic": selected_topic,
        "setting": setting,
        "user_role": role,
        "purpose": purpose,
        "motivation": motivation,
        "entity_kind": "investigation record",
    }


def measurement_source_profile(source_id: str, *, domain: str,
                               capabilities: list[str]) -> dict:
    """A source modality is meaningful because its fields enter the Gold DAG.

    A browse-only media observation is represented in the controlled World by
    one structured annotation per investigated case. It is *not* a claim that
    a real image or audio file was decoded. The Main/Stage-2 interface uses
    the corresponding simulated media-acquisition tool contract.
    """
    if domain not in _CONTEXT:
        raise ValueError("unknown_measurement_domain:" + domain)
    thematic = {
        "biographical": "biographical",
        "academic": "research",
        "institutional": "operational",
        "geographical": "regional survey",
    }[domain]
    properties = []
    for cap, label in (
        ("numeric", "counts"),
        ("temporal", "dates"),
        ("geographic", "site coordinates"),
        ("aggregate", "event tallies"),
    ):
        if cap in capabilities:
            properties.append(label)
    subject = ", ".join(properties) or "observations"
    if source_id == "metric_a":
        return {
            "modality": "image",
            "key_field": None,
            "title": "Photographed " + thematic + " measurement record",
            "purpose": ("Read first-source " + subject +
                        " from an annotated visual observation"),
        }
    if source_id == "metric_b":
        return {
            "modality": "audio",
            "key_field": None,
            "title": "Recorded " + thematic + " observation briefing",
            "purpose": ("Extract second-source " + subject +
                        " from a spoken observation transcript"),
        }
    if source_id == "observation_log":
        return {
            "modality": "table",
            "key_field": "entity_final",
            "title": "Dated " + thematic + " observation ledger",
            "purpose": ("Select distinct observation records and inspect "
                        "their recorded years"),
        }
    raise ValueError("unsupported_measurement_source:" + source_id)

__all__ = [
    'clue_profiles', 'focus_phrase', 'focus_coverage',
    'annotate_source_observations', 'scene_profile', 'measurement_source_profile',
]
