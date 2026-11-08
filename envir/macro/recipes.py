"""Typed Work-Product recipes for data-driven *backward* Macro Task planning.

This bank is a grammar, not a list of DAG templates. Multiple recipes can
produce the SAME WorkProduct, and their inputs are recursively expanded.
All recipes execute over documented, explicitly synthetic source observations;
source evidence and MMQA's composition annotations do not define topology.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    id: str
    output: str
    inputs: tuple[str, ...]
    work_type: str
    operation: str
    description: str
    deliverable: str


RECIPES = (
    Recipe("evidence_union", "EvidenceBundle", ("SourceRoots",),
           "evidence_consolidation", "merge_evidence",
           "Consolidate all available source findings into a traceable candidate evidence bundle.",
           "EvidenceBundle containing every original source handle and its observations."),
    Recipe("evidence_staged", "EvidenceBundle", ("PartialEvidence", "RemainingRoots"),
           "evidence_consolidation", "merge_evidence",
           "Join the initial source cross-check with the remaining independent sources.",
           "EvidenceBundle covering the full source inventory."),
    Recipe("partial_cross_check", "PartialEvidence", ("FirstTwoRoots",),
           "partial_source_cross_check", "merge_evidence",
           "Cross-check candidate references in an initial pair of source investigations.",
           "PartialEvidence identifying the first cross-source candidate links."),
    Recipe("reference_index", "ReferenceIndex", ("EvidenceBundle",),
           "reference_index_construction", "align_candidates",
           "Index document-to-candidate references across the collected evidence.",
           "ReferenceIndex mapping each candidate to its source handles."),
    Recipe("alignment_direct", "AlignedCandidates", ("EvidenceBundle",),
           "entity_reference_alignment", "align_candidates",
           "Reconcile source-specific candidate identities using the evidence bundle.",
           "AlignedCandidates with defensible source-to-candidate mappings."),
    Recipe("alignment_indexed", "AlignedCandidates", ("ReferenceIndex",),
           "entity_reference_alignment", "align_candidates",
           "Resolve candidate identities using the previously built reference index.",
           "AlignedCandidates with provenance-preserving reference links."),
    Recipe("review_direct", "EvidenceReview", ("EvidenceBundle",),
           "evidence_consistency_review", "verify_evidence",
           "Check completeness and scope of the consolidated source evidence.",
           "EvidenceReview with source coverage, consistency and limitations."),
    Recipe("review_linked", "EvidenceReview", ("AlignedCandidates", "EvidenceBundle"),
           "evidence_consistency_review", "verify_evidence",
           "Independently verify candidate linkage against the original evidence bundle.",
           "EvidenceReview testing source-reference consistency."),
    Recipe("review_indexed", "EvidenceReview", ("ReferenceIndex", "EvidenceBundle"),
           "evidence_consistency_review", "verify_evidence",
           "Validate candidate index coverage against source-scoped evidence.",
           "EvidenceReview with provenance checks for all indexed references."),
    Recipe("assessment_aligned", "GoalAssessment", ("AlignedCandidates",),
           "GOAL_ASSESSMENT", "GOAL_OP",
           "Apply the selected problem-solving operation to reconciled candidates.",
           "GoalAssessment with selected observations and original-source references."),
    Recipe("assessment_evidence", "GoalAssessment", ("EvidenceBundle",),
           "GOAL_ASSESSMENT", "GOAL_OP",
           "Compute the requested property or ordering over the source evidence.",
           "GoalAssessment directly grounded in full source evidence."),
    Recipe("assessment_reviewed", "GoalAssessment", ("AlignedCandidates", "EvidenceReview"),
           "GOAL_ASSESSMENT", "GOAL_OP",
           "Compute the requested assessment using aligned candidates and independently reviewed evidence.",
           "GoalAssessment supported by checked candidate links."),
    Recipe("assessment_crosschecked", "GoalAssessment", ("EvidenceBundle", "EvidenceReview"),
           "GOAL_ASSESSMENT", "GOAL_OP",
           "Evaluate candidates against all sources and a separate provenance review.",
           "GoalAssessment retaining source-level checks and computed outcomes."),
    Recipe("report_direct", "GoalReport", ("GoalAssessment", "EvidenceBundle"),
           "answer_evidence_synthesis", "answer_synthesis",
           "Present the requested outcome together with the evidence bundle used to compute it.",
           "A source-grounded final work product."),
    Recipe("report_verified", "GoalReport", ("GoalAssessment", "EvidenceReview"),
           "answer_evidence_synthesis", "answer_synthesis",
           "Present the requested outcome along with independently verified evidence coverage.",
           "A final work product with explicit evidence checks."),
    Recipe("report_reconciled", "GoalReport", ("GoalAssessment", "AlignedCandidates"),
           "answer_evidence_synthesis", "answer_synthesis",
           "Produce the requested result with traceable cross-source candidate mappings.",
           "A source-reconciled final work product."),
)

RECIPES_BY_OUTPUT = {
    output: tuple(r for r in RECIPES if r.output == output)
    for output in {r.output for r in RECIPES}
}


def recipe_manifest() -> list[dict]:
    return [
        {"recipe_id": r.id, "output": r.output, "inputs": list(r.inputs),
         "work_type": r.work_type, "operation": r.operation,
         "description": r.description, "deliverable": r.deliverable}
        for r in RECIPES
    ]
