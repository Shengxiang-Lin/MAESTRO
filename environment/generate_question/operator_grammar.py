"""Deterministic, typed relational-plan choices for Macro-first V2.

This is a deliberately bounded operator grammar: variants are *equivalent
query plans* for the same frozen Macro input/output contracts. It changes
where selective predicates execute or which Join input is on the left,
not what a question asks or which sources it can access.

The witness-level auditor separately compares every variant WorkProduct
(including provenance) with the legacy canonical solver. V2 Scenes without
the version marker retain their old programs byte-for-byte.
"""
from __future__ import annotations

from environment.schema import digest


VERSION = "maestro.typed_operator_grammar.v1"

# Each choice is authorized only where the Macro's canonical solver has
# independently typed predicates or a symmetric inner Join.
_JOIN_ROLES = frozenset({
    "candidate_fusion", "event_fusion", "dispute_resolution",
    "dispute_context_fusion", "goal_resolution",
})


def allowed_solver_variants(role: str, clues: dict) -> tuple[str, ...]:
    if role in _JOIN_ROLES:
        return ("canonical", "right_first")
    if role == "video_audio_alignment":
        # Placement of independently typed predicates relative to inner Join.
        return (
            "canonical",
            "pushdown_video_only",
            "pushdown_audio_only",
            "filter_after_join",
        )
    if role == "visual_identity":
        # Both image and video identity require two different predicates.
        return ("canonical", "reverse_filters")
    if role == "visual_candidates" and clues["visual_modality"] == "video":
        return ("canonical", "reverse_filters")
    return ("canonical",)


def choose_solver_variant(
    *, seed: int, ordinal: int, record_id: str, macro_id: str,
    role: str, clues: dict,
) -> str:
    """Select without consuming the World/Graph PRNG stream.

    Seeding by Macro ID and role keeps variant selection independent of
    evidence counts, entity values, source ordering and LLM naturalization.
    """
    variants = allowed_solver_variants(role, clues)
    index = int(digest(
        seed, ordinal, record_id, macro_id, role, VERSION
    )[:14], 16) % len(variants)
    return variants[index]


def validate_solver_variant(role: str, variant: str, clues: dict) -> None:
    if not isinstance(variant, str) or variant not in allowed_solver_variants(
        role, clues
    ):
        raise ValueError("invalid_typed_solver_variant:" + str(role))


def canonical_spec(spec: dict) -> dict:
    """Return an audit-only legacy solver spec without changing Source/World.

    A shallow copy is sufficient because only the Macro-plan task list and
    its per-task variant keys are replaced. It must pass ordinary Spec
    reproducibility checks, rather than bypassing the compiler contract.
    """
    plan = dict(spec["macro_plan"])
    plan["macro_tasks"] = [
        {k: value for k, value in task.items() if k != "solver_variant"}
        for task in plan["macro_tasks"]
    ]
    return {
        **{k: value for k, value in spec.items()
           if k != "operator_grammar_version"},
        "macro_plan": plan,
    }
