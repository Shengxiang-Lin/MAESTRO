"""Domain-specific simulated visual, spoken and event clues.

Each pair still discriminates candidates by the same frozen equality
operators. The values identify synthetic records, not actual media.
"""
from __future__ import annotations

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
