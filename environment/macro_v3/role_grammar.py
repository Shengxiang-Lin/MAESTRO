"""Compatibility exports for the frozen v3.2 typed role grammar.

Canonical implementation lives in environment.generate_question.
"""
from environment.generate_question.typed_role_grammar import (
    VERSION, OUTPUTS, REQUIRED_INPUTS, OPTIONAL_INPUTS, allowed, new_node,
)
