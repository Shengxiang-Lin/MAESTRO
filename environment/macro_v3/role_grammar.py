"""Typed information-goal grammar for Macro-first graph sampling.

A role describes the *work product*, not a fixed tool sequence. Edges connect
named output ports to compatible input ports. Role ordering follows causal
handoffs, NOT a hard-coded Discovery < Identification < ... total order.
"""
from __future__ import annotations

VERSION = "maestro.macro_role_grammar.v1"

# Output type names are semantic information types, never Python/JSON types.
OUTPUTS = {
    "discovery": {"candidates": "CandidateSet", "evidence": "EvidenceRefs"},
    "identification": {"entity": "EntityRef"},
    "investigation": {"finding": "FactBundle"},
    "verification": {"verified": "VerifiedClaim"},
    "comparison": {"comparison": "ComparisonResult"},
    "synthesis": {"result": "FinalResult"},
}

REQUIRED_INPUTS = {
    "discovery": {},
    "identification": {"candidate_1": {"CandidateSet"}},
    "investigation": {"subject": {"EntityRef"}},
    "verification": {"finding": {"FactBundle"}},
    "comparison": {"left": {"FactBundle"}, "right": {"FactBundle"}},
    "synthesis": {"entity": {"EntityRef"}, "result_1": {
        "FactBundle", "VerifiedClaim", "ComparisonResult",
    }},
}

# Variable-arity slots; the checker also validates every actual port name.
OPTIONAL_INPUTS = {
    "discovery": {"prior_candidates": {"CandidateSet"}},
    "identification": {"candidate_*": {"CandidateSet"}},
    "investigation": {"verified_context": {"VerifiedClaim"}},
    "verification": {},
    "comparison": {},
    "synthesis": {"result_*": {
        "FactBundle", "VerifiedClaim", "ComparisonResult",
    }, "original_evidence": {"EvidenceRefs"}},
}


def allowed(role: str, name: str) -> set[str] | None:
    if role not in OUTPUTS:
        return None
    if name in REQUIRED_INPUTS[role]:
        return REQUIRED_INPUTS[role][name]
    option = OPTIONAL_INPUTS[role]
    if name in option:
        return option[name]
    if role == "identification" and name.startswith("candidate_") and name[10:].isdigit():
        return option["candidate_*"]
    if role == "synthesis" and name.startswith("result_") and name[7:].isdigit():
        return option["result_*"]
    return None


def new_node(mid: str, role: str, operation: str, inputs: dict | None = None) -> dict:
    if role not in OUTPUTS:
        raise ValueError("unknown_macro_role:" + role)
    inputs = inputs or {}
    return {
        "macro_id": mid, "role": role, "operation": operation,
        "input_ports": inputs,
        "output_ports": dict(OUTPUTS[role]),
        "depends_on": list(dict.fromkeys(
            item["from_macro"] for item in inputs.values()
        )),
    }
