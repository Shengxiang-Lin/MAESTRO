"""Small public semantic contract shared by freeze, Author and local validators."""
from __future__ import annotations
import re
from ..task_construction.rendering.naming import predicate_label,select_mode_phrase,comparison_phrase


def terminal_contract(record, draft_question):
    raw=record['program']['nodes']
    nodes={int(n['node_id']):n for n in raw}
    final=raw[-1]
    op=final['op']
    inputs=final.get('inputs',[])
    contract={'operation':op,'answer_cardinality':'set' if op=='QUERY_NAME' else 'scalar'}
    if op in {'QUERY_ATTR','SELECT_AMONG','SELECT_BETWEEN'}:
        contract['requested_attribute']=predicate_label(inputs[0])
    if op in {'SELECT_AMONG','SELECT_BETWEEN'}:
        contract['selection_mode']=select_mode_phrase(inputs[1])
        contract['candidate_pool']='union_of_operands' if op=='SELECT_BETWEEN' else 'all_matching_candidates'
    if op.startswith('VERIFY_'):
        contract['quantifier']='any_recorded_value'
        contract['comparison']=comparison_phrase(inputs[0])
        query=nodes[int(final['dependencies'][0])]
        contract['requested_attribute']=predicate_label(query['inputs'][0])
    # Only values already present in the public draft are protected here.
    # World facts, answers and hidden acquisition clues never enter this contract.
    contract['required_numbers']=sorted(set(re.findall(r'(?<!\w)-?\d+(?:\.\d+)?(?!\w)',draft_question)))
    contract['required_literals']=sorted({str(n['inputs'][2]) for n in raw
        if n['op'] in {'FILTER_STRING','FILTER_DATE'} and len(n.get('inputs',[]))>=3
        and str(n['inputs'][2]).casefold() in draft_question.casefold()})
    return contract


def contract_issues(payload,question):
    contract=payload.get('terminal_contract',{})
    if not isinstance(contract,dict):
        return ['invalid_terminal_contract']
    lower=question.casefold()
    issues=[]
    present=set(re.findall(r'(?<!\w)-?\d+(?:\.\d+)?(?!\w)',question))
    if set(contract.get('required_numbers',()))-present:
        issues.append('public_numeric_constraint_missing')
    if any(str(v).casefold() not in lower for v in contract.get('required_literals',())):
        issues.append('public_literal_constraint_missing')
    attribute=str(contract.get('requested_attribute') or '')
    if attribute and not set(re.findall(r'[a-z]+',attribute.lower())).issubset(set(re.findall(r'[a-z]+',lower))):
        issues.append('terminal_attribute_missing')
    if contract.get('operation')=='QUERY_NAME' and re.search(r'\b(?:exactly one|a single|the only)\b',lower):
        issues.append('invented_uniqueness')
    return issues
