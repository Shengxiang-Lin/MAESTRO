"""Deterministic, typed evidence operators. No LLM or hidden Gold access.

An assertion records what ONE source claims, not whether it is true. Independence
is a declared provenance property of the synthetic World, never inferred from
the number of Sub-Agent messages. Every operation is pure and JSON serializable.
"""
from __future__ import annotations

from collections import defaultdict

VERSION = "maestro.evidence_operators.v1"
WARRANT_POLICY = "independent_confirmation"


def extract_claim(source: dict, observation: dict, *, field: str,
                  attribute: str, valid_scope: str = "event") -> dict:
    """Extract a source-grounded assertion with a canonical Claim key."""
    sid = source.get("source_id")
    root = source.get("evidence_origin_id")
    values = observation.get("values", {})
    if (not isinstance(sid, str) or not sid or
            not isinstance(root, str) or not root):
        raise ValueError("claim_source_origin_required")
    if not isinstance(values, dict) or field not in values:
        raise ValueError("claim_source_field_missing")
    subject = values.get("case_id")
    if not isinstance(subject, str) or not subject:
        raise ValueError("claim_subject_missing")
    if not isinstance(attribute, str) or not attribute:
        raise ValueError("claim_attribute_missing")
    if not isinstance(valid_scope, str) or not valid_scope:
        raise ValueError("claim_valid_scope_missing")
    # Credentials and version history are source/observation facts; callers
    # cannot grant authority or invent a revision by a keyword argument.
    revision_seq = observation.get("revision_seq", 1)
    revision_of = observation.get("revision_of")
    authoritative = source.get("authority_designation") == "official_registry"
    if revision_of is not None and (
        not isinstance(revision_of, str) or not revision_of
    ):
        raise ValueError("claim_revision_reference_invalid")
    if type(revision_seq) is not int or revision_seq < 1:
        raise ValueError("claim_revision_seq_invalid")
    oid = observation.get("observation_id")
    refs = observation.get("fact_refs")
    if (not isinstance(oid, str) or not oid.startswith(sid + ".") or
            not isinstance(refs, list) or not refs or
            any(not isinstance(ref, str) or not ref.startswith(sid + ":")
                for ref in refs)):
        raise ValueError("claim_fact_provenance_invalid")
    value = values[field]
    if type(value) not in (str, int, bool, float):
        raise ValueError("claim_value_not_scalar")
    return {
        "claim_id": oid + "::" + field,
        "subject": subject, "attribute": attribute,
        "valid_scope": valid_scope, "value": value,
        "source_id": sid, "fact_refs": list(refs),
        "evidence_origin_id": root,
        "revision_seq": revision_seq,
        "revision_of": revision_of,
        "authoritative": bool(authoritative),
    }


def align_claims(a: dict, b: dict) -> dict:
    """Comparable only for the same entity, predicate, scope, and value type."""
    fields = ("subject", "attribute", "valid_scope")
    mismatched = [key for key in fields if a.get(key) != b.get(key)]
    if type(a.get("value")) is not type(b.get("value")):
        mismatched.append("value_type")
    return {
        "comparable": not mismatched,
        "mismatched": mismatched,
        "claim_key": [a.get(k) for k in fields] if not mismatched else None,
    }


def relate_claims(a: dict, b: dict) -> dict:
    """Infer a *semantic* relation, never an unsupported confidence score."""
    alignment = align_claims(a, b)
    if not alignment["comparable"]:
        return {"relation": "incomparable", "alignment": alignment}
    same_origin = a.get("evidence_origin_id") == b.get("evidence_origin_id")
    if not a.get("evidence_origin_id") or not b.get("evidence_origin_id"):
        raise ValueError("claim_independence_origin_required")
    # A signed revision must follow the previous claim in the SAME lineage.
    newer, older = None, None
    if (b.get("revision_of") == a.get("claim_id") and
            b.get("revision_seq", 0) > a.get("revision_seq", 0) and same_origin):
        newer, older = b, a
    elif (a.get("revision_of") == b.get("claim_id") and
          a.get("revision_seq", 0) > b.get("revision_seq", 0) and same_origin):
        newer, older = a, b
    if newer is not None:
        return {
            "relation": "supersedes", "alignment": alignment,
            "superseding_claim_id": newer["claim_id"],
            "superseded_claim_id": older["claim_id"],
        }
    if same_origin:
        relation = "dependent_duplicate"
    else:
        relation = ("corroborates" if a.get("value") == b.get("value")
                    else "contradicts")
    return {"relation": relation, "alignment": alignment}


def check_warrant(claims: list[dict], *,
                  policy: str = WARRANT_POLICY) -> dict:
    """Evaluate visible evidence, not the inaccessible true answer.

    independent_confirmation: at least two independent matching roots.
    authoritative_record: a designated authoritative claim, with no
    unresolved conflicting *other* authoritative claim.
    """
    if policy not in {"independent_confirmation", "authoritative_record"}:
        raise ValueError("unknown_warrant_policy")
    if not claims:
        return {
            "status": "missing", "sufficient_to_accept": False,
            "evidence_count": 0, "independent_origins": 0,
            "missing_evidence": ["first_claim"], "winning_claim_id": None,
        }
    keys = {(c.get("subject"), c.get("attribute"), c.get("valid_scope"),
             type(c.get("value")).__name__) for c in claims}
    if len(keys) != 1:
        raise ValueError("warrant_mixed_claim_keys")
    active = {c["claim_id"]: c for c in claims}
    for a in claims:
        for b in claims:
            if a is b:
                continue
            rel = relate_claims(a, b)
            if rel["relation"] == "supersedes":
                active.pop(rel["superseded_claim_id"], None)
    remaining = list(active.values())
    values = {str(type(c["value"])) + ":" + repr(c["value"]) for c in remaining}
    independent = len({c["evidence_origin_id"] for c in remaining})
    authorities = [c for c in remaining if c.get("authoritative")]
    if policy == "authoritative_record":
        accepted = bool(authorities) and len({
            str(type(c["value"])) + ":" + repr(c["value"])
            for c in authorities
        }) == 1
        status = "resolved" if accepted else (
            "disputed" if len(values) > 1 else "unverified"
        )
        winner = authorities[0]["claim_id"] if accepted else None
        missing = [] if accepted else ["authoritative_resolution"]
    elif len(values) > 1:
        accepted, status, winner, missing = (
            False, "disputed", None, ["independent_resolution"]
        )
    elif independent >= 2:
        accepted, status, winner, missing = (
            True, "corroborated", remaining[0]["claim_id"], []
        )
    else:
        accepted, status, winner, missing = (
            False, "unverified", None, ["independent_confirmation"]
        )
    return {
        "status": status, "sufficient_to_accept": accepted,
        "evidence_count": len(claims),
        "independent_origins": independent,
        "missing_evidence": missing, "winning_claim_id": winner,
    }


def resolve_claim(claims: list[dict], *,
                  policy: str = WARRANT_POLICY) -> dict:
    """Resolve only when evidence warrants it; otherwise abstain."""
    warrant = check_warrant(claims, policy=policy)
    winner = next((c for c in claims
                   if c["claim_id"] == warrant["winning_claim_id"]), None)
    return {
        "status": warrant["status"],
        "value": winner["value"] if winner is not None else None,
        "supporting_fact_refs": (
            sorted({ref for claim in claims
                    if winner is not None and claim["value"] == winner["value"]
                    for ref in claim["fact_refs"]})
            if winner is not None else []
        ),
        "sufficient_to_accept": warrant["sufficient_to_accept"],
    }


def evidence_states(claims: list[dict], *,
                    policy: str = WARRANT_POLICY) -> dict:
    """A Claim-keyed state from ONLY the provided evidence subset."""
    grouped = defaultdict(list)
    for claim in claims:
        key = (claim["subject"], claim["attribute"], claim["valid_scope"])
        grouped[key].append(claim)
    return {
        "|".join(key): check_warrant(group, policy=policy)
        for key, group in sorted(grouped.items())
    }
