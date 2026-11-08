"""High-precision degeneracy checks for executable synthetic tasks.

Only detect fixed-result verification caused by an upstream equality filter
on the exact same entity attribute. No attempt at arbitrary symbolic
equivalence; the source-world oracle remains authoritative.
"""
from __future__ import annotations


def _comparison(value):
    token = str(value).strip().lower()
    if token.startswith("<cmp:") and token.endswith(">"):
        token = token[5:-1]
    aliases = {
        "=": "eq", "==": "eq", "eq": "eq", "equal": "eq",
        "!=": "ne", "<>": "ne", "ne": "ne", "not_equal": "ne",
    }
    return aliases.get(token, token)


def find_degenerate_verification(program):
    """Return a reason when a terminal verification is predetermined.

    Safe case: an upstream FILTER_X of a singleton source set requires
    attr == literal and QUERY_ATTR returns that same attribute, while the
    terminal VERIFY_X asks whether it differs from that literal (or equals).
    The returned answer is entailed by the question construction, not by
    independent evidence. Only identical public literals are compared.
    """
    nodes = program.node_map()
    if not program.nodes:
        return None
    terminal = nodes[program.topological_order()[-1]]
    if terminal.op not in {
        "VERIFY_STRING", "VERIFY_NUMBER", "VERIFY_YEAR", "VERIFY_DATE",
    } or len(terminal.dependencies) != 1 or len(terminal.inputs) < 2:
        return None
    query = nodes[terminal.dependencies[0]]
    if query.op != "QUERY_ATTR" or not query.dependencies or not query.inputs:
        return None
    predicate = str(query.inputs[0])
    target = str(terminal.inputs[1])
    verify_comp = _comparison(terminal.inputs[0])
    if verify_comp not in {"eq", "ne"}:
        return None
    pending = list(query.dependencies)
    seen = set()
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes[node_id]
        if (node.op in {"FILTER_STRING", "FILTER_NUMBER", "FILTER_YEAR", "FILTER_DATE"}
                and len(node.inputs) >= 3
                and node.inputs[0] == predicate
                and str(node.inputs[2]) == target
                and _comparison(node.inputs[1]) == "eq"):
            # WorldSampler allocates one value to each selected entity for
            # this predicate. Under this constraint, terminal validation
            # cannot add an independent information requirement.
            return ("verification_repeats_upstream_equal_filter"
                    if verify_comp == "eq"
                    else "verification_negates_upstream_equal_filter")
        pending.extend(node.dependencies)
    return None


__all__ = ["find_degenerate_verification"]
