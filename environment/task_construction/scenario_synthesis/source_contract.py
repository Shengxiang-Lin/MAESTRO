"""Public projection for optional certified source-semantic task contracts.

The construction pipeline may retain richer local source-semantic certificates,
but TaskDesign/P1.5 may expose only wording-safe resource semantics.  This helper
removes latent truth, proof ids, raw observations, oracle values, and validator-only
metadata before the contract crosses the LLM authoring boundary.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

_PRIVATE_KEYS = {
    "answer",
    "oracle_answer",
    "raw_answer",
    "fact_id",
    "fact_ids",
    "proof_id",
    "proof_ids",
    "source_ids",
    "critical_fact_ids",
    "decision_support_fact_ids",
    "indispensable",
    "proof_relevant",
    "hidden",
    "observation",
    "observations",
    "world",
    "program",
    "binding",
}


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            name = str(key)
            if name in _PRIVATE_KEYS or name.startswith("hidden_") or name.startswith("private_"):
                continue
            result[name] = _public(child)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_public(item) for item in value]
    return value


def public_source_contract(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the wording-safe public subset of a certified source contract."""
    if not isinstance(certificate, Mapping):
        raise TypeError("source semantic certificate must be a mapping")
    projected = _public(certificate)
    if not isinstance(projected, dict):
        raise TypeError("public source contract projection must be an object")
    projected.setdefault("contract_policy", "public_source_semantics_only")
    return projected


__all__ = ["public_source_contract"]
