"""Terminal-family helpers for composed ProgramIR candidates."""
from __future__ import annotations
from ..ir import ProgramIR
_TERMINAL_FAMILY={"COUNT":"count","QUERY_NAME":"entity_lookup","QUERY_ATTR":"attribute_query","QUERY_RELATION":"relation_query","SELECT_BETWEEN":"comparison","SELECT_AMONG":"comparison","VERIFY_STRING":"verification","VERIFY_NUMBER":"verification","VERIFY_YEAR":"verification","VERIFY_DATE":"verification"}
def terminal_family(program:ProgramIR)->str:
    if not program.nodes:return "empty"
    op=program.node_map()[program.topological_order()[-1]].op;return _TERMINAL_FAMILY.get(op,op.lower())
def terminal_is_supported(program:ProgramIR)->bool:return terminal_family(program)!="empty"
__all__=["terminal_family","terminal_is_supported"]
