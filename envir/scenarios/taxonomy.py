"""Scenario packs: original simulated media situations, tied to MMQA motifs.

The packs supply different visual/temporal/acoustic affordances while their
fact roles stay consistent, so Tool Feedback can be checked exactly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenePack:
    domain: str
    setting: str
    asset: str
    location: str
    schedule: str
    registry: str
    characteristic: str
    scene_types: tuple[str, ...]
    units: tuple[str, ...]


PACKS = (
    ScenePack("logistics", "freight terminal", "delivery vehicle", "loading bay",
              "dispatch roster", "fleet registry", "visible vehicle ID",
              ("loading_bay_camera", "shipping_label_closeup", "vehicle_checkpoint"),
              ("North Quay", "Dock Seven", "South Gate", "East Annex")),
    ScenePack("culture", "museum exhibition", "artwork", "gallery",
              "exhibition catalog", "collection registry", "catalog label",
              ("gallery_wall", "exhibition_poster", "artifact_display"),
              ("East Gallery", "Hall B", "North Wing", "Archive Room")),
    ScenePack("sports", "athletics event", "competitor", "competition lane",
              "heat assignment sheet", "registration registry", "bib identifier",
              ("stadium_finish", "scoreboard", "sports_podium"),
              ("Lane Two", "Track Four", "Main Arena", "Warmup Court")),
    ScenePack("education", "campus ceremony", "participant", "session room",
              "event program", "attendee registry", "badge ID",
              ("conference_badge", "campus_noticeboard", "graduation_stage"),
              ("Room A", "Hall C", "Library Forum", "West Auditorium")),
    ScenePack("science", "engineering laboratory", "instrument", "test station",
              "experiment schedule", "calibration registry", "device serial",
              ("instrument_panel", "laboratory_bench", "control_console"),
              ("Station Delta", "Bench Three", "Test Bay A", "Calibration Suite")),
    ScenePack("commerce", "retail distribution center", "product batch", "storage zone",
              "inventory allocation table", "stock registry", "batch code",
              ("product_shelf", "barcode_label", "stockroom_display"),
              ("Rack Nine", "Cold Zone", "Dispatch Desk", "Upper Storage")),
    ScenePack("environment", "wildlife monitoring site", "tracked specimen", "observation site",
              "field assignment log", "survey registry", "tracking tag",
              ("field_camera", "habitat_marker", "wildlife_trap_monitor"),
              ("Sector One", "Riverside", "North Plot", "Forest Edge")),
    ScenePack("civic", "municipal maintenance center", "service asset", "maintenance sector",
              "work order table", "public asset registry", "asset tag",
              ("street_sign", "maintenance_site", "building_facade"),
              ("District Two", "Civic Square", "South Route", "Sector D")),
    ScenePack("general", "community event", "registered item", "activity area",
              "assignment board", "event registry", "printed reference ID",
              ("information_kiosk", "group_photo", "public_notice"),
              ("Area A", "Zone B", "Hall Two", "West Entrance")),
)
BY_DOMAIN = {pack.domain: pack for pack in PACKS}
