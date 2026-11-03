"""Semantic grounding primitives for synthetic world construction.

This package owns predicate-level meaning that must be fixed before evidence,
question, or trajectory generation. Downstream layers may render these semantics
but must not invent units, ranges, or literal roles on their own.
"""
from .literal_factory import SemanticLiteralFactory
from .profile import AttributeSemanticProfile, compile_attribute_profile
__all__ = ["AttributeSemanticProfile", "SemanticLiteralFactory", "compile_attribute_profile"]
