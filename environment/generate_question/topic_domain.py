"""Conservative domain cues from the original *topic context*, never answers.

The user's simulated World remains independent of original MMQA facts.
We use categorical vocabulary only to avoid obvious mismatches such as
a TV series assigned a geographic survey Goal. If the category is
ambiguous, fall back to existing sampled domain logic.
"""
from __future__ import annotations
import re

TOPIC_DOMAIN_VERSION = "maestro.topic_domain.v1"
_CUES = {
    "sports": (
        r"\b(?:football\s+club|soccer\s+team|basketball\s+team|"
        r"sports\s+team|stadium|professional\s+league|"
        r"football\s+match|soccer\s+match|final\s+score)\b",
    ),
    "film": (
        r"\b(?:tv\s+series|television\s+series|television\s+show|"
        r"tv\s+show|sitcom|episode|feature\s+film|"
        r"motion\s+picture|voice\s+actor|voice\s+actress|"
        r"film\s+cast|movie\s+cast)\b",
    ),
    "music": (
        r"\b(?:music\s+album|record\s+album|songwriter|"
        r"recording\s+artist|music\s+video|musician|singer)\b",
    ),
    "education": (
        r"\b(?:university|college\s+campus|school\s+district|"
        r"educational\s+institution|undergraduate\s+course)\b",
    ),
    "geography": (
        r"\b(?:mountain\s+range|river\s+basin|geographic\s+region|"
        r"geographical\s+region|lake\s+basin)\b",
    ),
    "environment": (
        r"\b(?:wildlife\s+species|bird\s+species|plant\s+species|"
        r"natural\s+habitat|ecosystem)\b",
    ),
    "logistics": (
        r"\b(?:shipping\s+container|delivery\s+service|"
        r"cargo\s+shipment|freight\s+carrier)\b",
    ),
}


def infer_topic_domain(seed: dict) -> str | None:
    """Only use category predicates, no dataset answer or event assertions."""
    question = seed.get("question")
    if not isinstance(question, str):
        return None
    labels = [
        label for label, patterns in _CUES.items()
        if any(re.search(pattern, question, re.I) for pattern in patterns)
    ]
    return labels[0] if len(labels) == 1 else None
