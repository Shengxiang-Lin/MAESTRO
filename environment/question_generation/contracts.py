"""Small public semantic contract shared by freeze, Author and local validators."""
from __future__ import annotations
import re
from decimal import Decimal
from datetime import date, datetime
from ..task_construction.rendering.naming import predicate_label,select_mode_phrase,comparison_phrase


def normalize_value(value, operation):
    """Normalize by condition type; strings (including codes) stay literal."""
    text = str(value)
    if operation.endswith('_DATE') and re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        return {'type': 'date', 'value': date.fromisoformat(text).isoformat()}
    if operation.endswith(('_NUMBER', '_YEAR')):
        match = re.fullmatch(r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(.*)', text)
        if match:
            return {'type': 'year' if operation.endswith('_YEAR') else 'number',
                    'value': str(Decimal(match[1].replace(',', '')).normalize()),
                    'unit': match[2]}
    return {'type': 'string', 'value': text}


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
    # Tokenize the public draft as whole dates / numbers, including thousands
    # separators. Legacy frozen contracts split these into fragments; derive
    # the same protected values from their original public draft instead.
    required = typed_values(str(payload.get("draft_question") or ""), words=False)
    if not required:
        required = typed_values(" ".join(contract.get("required_numbers", ())))
    present = typed_values(question)
    if required-present:
        issues.append('public_numeric_constraint_missing')
    def literal_present(value):
        text = str(value)
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
            return ('date', text) in present
        return re.search(r'(?<![\w+])' + re.escape(text) + r'(?!\w)', question, re.I) is not None
    if any(not literal_present(v) for v in contract.get('required_literals',())):
        issues.append('public_literal_constraint_missing')
    attribute=str(contract.get('requested_attribute') or '')
    if attribute and not set(re.findall(r'[a-z]+',attribute.lower())).issubset(set(re.findall(r'[a-z]+',lower))):
        issues.append('terminal_attribute_missing')
    if contract.get('operation')=='QUERY_NAME' and re.search(r'\b(?:exactly one|a single|the only)\b',lower):
        issues.append('invented_uniqueness')
    return issues


_SMALL_NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
    range(21),
))
_VALUE_RE = re.compile(
    r"(?<!\w)\d{4}-\d{2}-\d{2}(?!\w)"
    r"|(?<!\w)(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}(?!\w)"
    r"|(?<!\w)\+\d+(?!\w)"
    r"|(?<![\w\d])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)"
    , re.I
)


def typed_values(text, *, words=True):
    """A cheap presence guard; ownership/equivalence still needs review."""
    values = set()
    for match in _VALUE_RE.finditer(text):
        token = match.group()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            values.add(("date", token))
        elif token.startswith('+'):
            values.add(("string", token))
        elif token[0].isalpha():
            try:
                values.add(("date", datetime.strptime(token.title(), '%B %d, %Y').date().isoformat()))
            except ValueError:
                values.add(("invalid_date", token))
        else:
            values.add(("number", Decimal(token.replace(",", ""))))
    for word in re.findall(r"\b[a-z]+\b", text.casefold()):
        if words and word in _SMALL_NUMBERS:
            values.add(("number", Decimal(_SMALL_NUMBERS[word])))
    return values
