"""Explicitly authored macro-work semantics.

KQA is a prior over information structure, not a mandate to delegate every
KoPL operation. MMQA is a prior over source/capability roles, not media bytes.
Every package below describes a delegable work goal and a concrete handoff.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    domain: str
    setting: str
    issue: str
    material: str
    rule_source: str
    records: str
    affected: str
    decision_owner: str
    focus: str
    entry_modalities: tuple[str, ...]


CASES = (
    Case("logistics", "freight terminal", "a disputed restricted-zone vehicle entry",
         "gate surveillance and dispatch incident reports",
         "terminal access policies", "carrier permits and movement logs",
         "vehicle movements and deliveries", "terminal operations manager",
         "access authorization", ("video", "table", "search")),
    Case("culture", "municipal museum", "a late change to the exhibition opening schedule",
         "gallery posters and exhibition photographs",
         "gallery capacity and opening policies", "event catalog and booking records",
         "visitor reservations", "museum program coordinator",
         "updated exhibition schedule", ("image", "table", "search")),
    Case("sports", "regional tournament", "an eligibility concern affecting the final standings",
         "competition footage and event reports",
         "participant eligibility rules", "registration lists and result tables",
         "athlete standings", "competition organizer",
         "validity of recorded results", ("video", "table", "text")),
    Case("education", "academic symposium", "conflicting information about an invited lecture",
         "event posters and session recordings",
         "conference scheduling requirements", "speaker program and venue bookings",
         "session assignments", "conference coordinator",
         "the lecture's confirmed arrangements", ("image", "table", "search")),
    Case("science", "research laboratory", "a calibration discrepancy during an experiment",
         "instrument screenshots and experiment logs",
         "calibration and quality-control requirements",
         "instrument maintenance and experiment records",
         "measurement results", "laboratory lead",
         "experimental validity and corrective action", ("image", "text", "table")),
    Case("commerce", "retail distribution network", "a product-batch recall notice",
         "package labels and recall announcements",
         "recall handling rules", "inventory, shipment, and customer-order records",
         "open customer orders", "fulfilment manager",
         "recall impact and fulfilment response", ("image", "table", "search")),
    Case("environment", "protected wetland", "a newly reported wildlife disturbance",
         "field camera footage and ranger observation notes",
         "habitat access restrictions", "species sightings and restoration logs",
         "sensitive habitat zones", "conservation coordinator",
         "conservation risk and appropriate intervention", ("video", "text", "table")),
    Case("civic", "city maintenance department", "a service interruption following equipment damage",
         "inspection photographs and repair reports",
         "municipal safety and service standards",
         "work orders, asset records, and service schedules",
         "local service routes", "maintenance coordinator",
         "restoration priorities", ("image", "table", "search")),
    Case("general", "community event center", "a disruption to a scheduled public event",
         "site announcements and attendee reports",
         "event access and safety guidelines",
         "venue bookings and operational records",
         "attendee arrangements", "event coordinator",
         "revised event arrangements", ("audio", "text", "table")),
)

BY_DOMAIN = {case.domain: case for case in CASES}

# No domain has a designated graph. The graph motif is sampled independently.
GRAPH_MOTIFS = ("chain", "parallel_join", "fork_join", "mixed", "staged")

# Goal alternatives share the same evidence slots, but ask for different
# deliverables. They are not paraphrases of a single question.
INTENTS = ("impact", "compliance", "reconciliation", "mitigation", "prioritization")


def question_brief(case: Case, intent: str) -> dict:
    if intent == "impact":
        request = (f"Determine which {case.affected} are affected by {case.issue} "
                   f"at the {case.setting}, and explain the implications.")
        expected = "impact_assessment"
    elif intent == "compliance":
        request = (f"Assess whether the handling of {case.issue} at the "
                   f"{case.setting} meets applicable requirements; explain the evidence.")
        expected = "compliance_assessment"
    elif intent == "reconciliation":
        request = (f"Reconcile the information about {case.issue} at the "
                   f"{case.setting}, explain any discrepancies, and establish what is supported.")
        expected = "reconciled_findings"
    elif intent == "mitigation":
        request = (f"Investigate {case.issue} at the {case.setting} and recommend "
                   f"feasible next steps for the {case.decision_owner}.")
        expected = "action_recommendations"
    elif intent == "prioritization":
        request = (f"Which {case.affected} should the {case.decision_owner} "
                   f"prioritize in responding to {case.issue}, and why?")
        expected = "ranked_priorities"
    else:
        raise ValueError("unsupported goal intent: " + intent)
    return {
        "situation": f"{case.decision_owner} is reviewing {case.issue}.",
        "user_role": case.decision_owner,
        "overall_goal": request,
        "expected_deliverable": expected,
        "required_context": case.focus,
        "public_resources": [
            {"description": case.material, "access": "provided_or_discoverable"},
            {"description": case.records, "access": "discoverable"},
            {"description": case.rule_source, "access": "discoverable"},
        ],
    }


# Each work unit below is a *work package*, not an OCR/filter/search step.
# Inputs are typed artifact-level handoffs. Internal actions are execution
# hints for a FUTURE reliable Feedback builder, not Main-subtask nodes.
WORK_DEFINITIONS = {
    "A": {
        "work_type": "incident_investigation",
        "description": "Investigate {issue} using {material}; establish the relevant case, participants and timing.",
        "deliverable": "A documented case profile for {focus}.",
        "produces": {"case_profile": "CaseProfile"},
        "capabilities": ("MediaAnalysis", "Research"),
        "internal_actions": ("inspect relevant material", "identify the event or subject",
                             "extract the pertinent observations", "summarize the case"),
    },
    "B": {
        "work_type": "requirements_review",
        "description": "Review {rule_source} and establish the requirements relevant to {focus}.",
        "deliverable": "An applicable-rules brief with scope and exceptions.",
        "produces": {"rules": "RuleSet"},
        "capabilities": ("Research", "PolicyAnalysis"),
        "internal_actions": ("locate authoritative material", "read applicable sections",
                             "identify qualifying conditions", "summarize the rules"),
    },
    "C": {
        "work_type": "records_audit",
        "description": "Audit {records} in relation to {issue}; reconcile identities and relevant records.",
        "deliverable": "A structured record audit with matched entries and dates.",
        "produces": {"records": "RecordSet"},
        "capabilities": ("TableAnalysis", "Research"),
        "internal_actions": ("locate record sources", "match relevant identifiers",
                             "inspect conflicting entries", "compile record findings"),
    },
    "D": {
        "work_type": "impact_assessment",
        "description": "Assess the consequences for {affected} using the available case findings and criteria.",
        "deliverable": "A scoped assessment of the effect on {affected}.",
        "produces": {"assessment": "Assessment"},
        "capabilities": ("Analysis", "Research"),
        "internal_actions": ("collect relevant findings", "evaluate material conditions",
                             "compare affected cases", "prepare an assessment"),
    },
    "E": {
        "work_type": "response_planning",
        "description": "Investigate feasible responses to {issue} and prioritize practical next steps.",
        "deliverable": "A ranked set of feasible responses for {decision_owner}.",
        "produces": {"options": "Options"},
        "capabilities": ("Planning", "Research"),
        "internal_actions": ("identify possible responses", "check operational constraints",
                             "evaluate candidate responses", "prepare prioritized options"),
    },
}

# Macro edges are real typed result dependencies. Rows specify
# (producer, consumer, consumer slot, producer output slot). Each macro node
# has a single independently meaningful handoff.
DEPENDENCIES = {
    "chain": (
        ("A", "C", "case", "case_profile"),
        ("C", "D", "record_findings", "records"),
        ("D", "E", "impact_result", "assessment"),
    ),
    "parallel_join": (
        ("A", "D", "case", "case_profile"),
        ("B", "D", "requirements", "rules"),
        ("C", "D", "record_findings", "records"),
    ),
    "fork_join": (
        ("A", "B", "case", "case_profile"),
        ("A", "C", "case", "case_profile"),
        ("B", "D", "requirements", "rules"),
        ("C", "D", "record_findings", "records"),
    ),
    "mixed": (
        ("A", "C", "case", "case_profile"),
        ("B", "D", "requirements", "rules"),
        ("C", "D", "record_findings", "records"),
    ),
    "staged": (
        ("A", "C", "case", "case_profile"),
        ("C", "D", "record_findings", "records"),
        ("B", "E", "requirements", "rules"),
        ("D", "E", "impact_result", "assessment"),
    ),
}
NODE_SETS = {
    "chain": ("A", "C", "D", "E"),
    "parallel_join": ("A", "B", "C", "D"),
    "fork_join": ("A", "B", "C", "D"),
    "mixed": ("A", "B", "C", "D"),
    "staged": ("A", "B", "C", "D", "E"),
}


# Within-domain issue variation changes the concrete information need; it
# does not rewrite evidence into a supposedly factual answer.
CASE_VARIANTS = {
    "logistics": (
        "a disputed restricted-zone vehicle entry",
        "a carrier entering outside its assigned delivery window",
        "a gate movement whose recorded carrier does not match dispatch paperwork",
        "a service vehicle operating inside a temporarily closed loading area",
    ),
    "culture": (
        "a late change to the exhibition opening schedule",
        "a curator talk with conflicting publicity times",
        "an artwork loan postponing a scheduled display",
        "a gallery closure that affects published visitor bookings",
    ),
    "sports": (
        "an eligibility concern affecting the final standings",
        "a disputed athlete registration in a qualifying event",
        "a corrected results announcement after a timing discrepancy",
        "a match schedule amendment affecting eligible participants",
    ),
    "education": (
        "conflicting information about an invited lecture",
        "a symposium session moved after a venue conflict",
        "a workshop registration list inconsistent with the published program",
        "a last-minute change to an examination briefing",
    ),
    "science": (
        "a calibration discrepancy during an experiment",
        "a laboratory instrument operating outside its approved range",
        "a mismatch between experimental logs and calibration records",
        "an unexpected change in a shared laboratory booking",
    ),
    "commerce": (
        "a product-batch recall notice",
        "a packing-label discrepancy affecting shipped orders",
        "a stock-hold instruction issued after a supplier alert",
        "a delivery exception involving a temperature-sensitive batch",
    ),
    "environment": (
        "a newly reported wildlife disturbance",
        "an unexpected observation near a protected nesting area",
        "a survey finding that conflicts with the seasonal habitat inventory",
        "a maintenance activity close to a restricted conservation zone",
    ),
    "civic": (
        "a service interruption following equipment damage",
        "a repair backlog affecting an accessible transport route",
        "a street-safety inspection that conflicts with the maintenance log",
        "an urgent maintenance request for a public facility",
    ),
    "general": (
        "a disruption to a scheduled public event",
        "a venue change affecting registered attendees",
        "an access notice inconsistent with the booking schedule",
        "a revised event plan after an equipment failure",
    ),
}

ASSESSMENT_VARIANTS = {
    "impact": "Assess which {affected} are affected and identify the practical implications.",
    "compliance": "Assess the applicable requirements against the case and records and determine conformity.",
    "reconciliation": "Reconcile disagreements across the gathered observations and records.",
    "mitigation": "Assess operational constraints, consequences and options for resolving the situation.",
    "prioritization": "Prioritize affected cases by urgency and explain the criteria.",
}

RESPONSE_VARIANTS = {
    "impact": "Investigate actions that reduce the established impact and identify immediate next steps.",
    "compliance": "Investigate corrective actions that meet the relevant requirements.",
    "reconciliation": "Prepare options to resolve outstanding discrepancies and communicate a defensible account.",
    "mitigation": "Develop and compare practical mitigation options for the responsible coordinator.",
    "prioritization": "Develop a workable response sequence for the highest-priority affected cases.",
}
