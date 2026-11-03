"""Adapters for public structural/context priors used by the environment."""

from .kqa import analyze_programs, iter_kqa_records, record_to_ir, write_program_analysis
from .mmqa import analyze_source_motifs
from .mmqa_question_type_roles import (
    build_mmqa_semantic_bundle_bank,
    extract_mmqa_semantic_bundle,
)
from .profile_support import attach_entity_profile_support
from .taxonomy import TaxonomyIndex, build_taxonomy_index
from .yago import build_predicate_catalog, load_predicate_catalog, refresh_predicate_catalog
from .yago_profiles import build_entity_profile_index, load_entity_profile_index

__all__ = [
    "TaxonomyIndex",
    "analyze_programs",
    "analyze_source_motifs",
    "attach_entity_profile_support",
    "build_entity_profile_index",
    "build_mmqa_semantic_bundle_bank",
    "build_predicate_catalog",
    "build_taxonomy_index",
    "extract_mmqa_semantic_bundle",
    "iter_kqa_records",
    "load_entity_profile_index",
    "load_predicate_catalog",
    "record_to_ir",
    "refresh_predicate_catalog",
    "write_program_analysis",
]
