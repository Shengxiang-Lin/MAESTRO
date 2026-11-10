"""Schema-derived source titles and domain framing for NEW V2 scenes.

Source titles are not free-form Author outputs: a title must describe the
declared modality and source role, never a new observation or filter.
All descriptions refer to the synthetic task workspace, not real media.
"""
from __future__ import annotations

from copy import deepcopy

SCENE_REALISM_VERSION = "maestro.scene_realism.v1"

DOMAIN_FRAMES = {
    "sports": ("match and event review", "game or competition records"),
    "film": ("screening and episode identification", "production materials"),
    "music": ("performance and recording investigation", "performance materials"),
    "culture": ("exhibition and performance investigation", "exhibition materials"),
    "education": ("course and campus investigation", "academic materials"),
    "logistics": ("shipment and arrival investigation", "transport materials"),
    "civic": ("site and permit investigation", "civic working materials"),
    "environment": ("field survey and species investigation", "field observations"),
    "science": ("experiment and review investigation", "research records"),
    "business": ("filing and resolution investigation", "business records"),
    "book": ("edition and publication investigation", "edition records"),
    "history": ("artifact and historical collection investigation", "collection records"),
    "geography": ("site and geographical survey investigation", "survey records"),
    "general": ("source comparison and record investigation", "working materials"),
}

SOURCE_KIND = {
    "visual": {
        "image": "still-image annotations",
        "video": "video action and frame-time annotations",
    },
    "caption": {
        "audio": "spoken-commentary cue annotations",
        "text": "written caption annotations",
    },
    "footage": {"video": "video action timeline"},
    "commentary": {"audio": "spoken commentary timeline"},
    "context": {"text": "independent event log"},
    "report_a": {"text": "first reported-year account"},
    "report_b": {"table": "independent reported-year table"},
    "archive": {"table": "verified result table"},
}
 
def has_scene_realism(spec: dict) -> bool:
    return spec.get("scene_realism_version") == SCENE_REALISM_VERSION


def _domain_frame(spec: dict) -> tuple[str, str]:
    return DOMAIN_FRAMES.get(
        spec.get("macro_plan", {}).get("domain", "general"),
        DOMAIN_FRAMES["general"],
    )


def expected_source_titles(spec: dict) -> dict[str, str]:
    """Return source-ID-keyed titles determined by modality and source role."""
    topic = spec["scenario"]["topic"]
    frame, _ = _domain_frame(spec)
    titles = {}
    for source in spec["sources"]:
        sid, modality = source["source_id"], source["modality"]
        if sid.startswith("index_"):
            if modality != "text":
                raise ValueError("index_requires_text_source")
            desc = "keyed cross-reference index"
        else:
            desc = SOURCE_KIND.get(sid, {}).get(modality)
            if not desc:
                raise ValueError(f"invalid_source_modality_pair:{sid}:{modality}")
        titles[sid] = f"{topic} — {frame}: {desc}"
    return titles


def canonicalize_scenario_proposal(spec: dict, proposal: dict) -> dict:
    """Accept only the LLM's narrative; do not trust authored source titles.

    Deterministically overwrite every proposed title from the frozen schema.
    Unknown or partial titles cannot authorize new media types.
    """
    if not has_scene_realism(spec) or not isinstance(proposal, dict):
        return proposal
    chosen = deepcopy(proposal)
    titles = expected_source_titles(spec)
    chosen["source_titles"] = [
        {"source_id": sid, "title": titles[sid]}
        for sid in [row["source_id"] for row in spec["sources"]]
    ]
    return chosen


def scenario_domain_context(spec: dict) -> dict:
    frame, materials = _domain_frame(spec)
    return {
        "domain": spec["macro_plan"]["domain"],
        "activity": frame,
        "material_kind": materials,
        "goal_label": spec["macro_plan"]["goal"]["public_label"],
        "source_titles": expected_source_titles(spec),
        "source_modalities": {
            row["source_id"]: row["modality"] for row in spec["sources"]
        },
        "restriction": (
            "Use only these source types; do not invent new media, "
            "extra output targets, real historical events or clearance rules."
        ),
    }
