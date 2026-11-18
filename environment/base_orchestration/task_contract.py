"""Current-round prose contracts compiled without source contents or future results."""
from __future__ import annotations

import json
import re

VERSION = 'stage3.local_task_contract.v1'


def compile_task_contract(target: dict, catalog: dict) -> dict:
    """Project local operators into public task obligations, never executable IDs.

    Private source rows, gold answers and future tasks are deliberately not inputs.
    Unsupported operators fail closed rather than receiving an incomplete fallback.
    """
    local_steps = {s["id"]: i for i, s in enumerate(target["operator_chain"], 1)}
    external_steps = {n: "earlier_returned_group_" + str(i)
                      for i, n in enumerate(target.get("external_inputs", []), 1)}
    scopes = {}
    schemas = {}
    for i, node in enumerate(target.get('external_inputs', [])):
        fields = target.get('input_schema', {}).get(f'input_{i}', [])
        scopes[node] = 'earlier returned ' + (', '.join(fields) or 'evidence') + ' records'
        schemas[node] = fields
    clauses, steps, filters = [], [], []
    supported = True
    source_names = {sid: f"{sid} ({catalog[sid]['title']})"
                    + (" at " + catalog[sid]["resource_url"] if catalog[sid].get("resource_url") else "")
                    for sid in target["source_ids"]}
    for number, original in enumerate(target['operator_chain'], 1):
        op = original['op']
        inputs = original.get('inputs', [])
        origins = [scopes.get(n, 'earlier returned records') for n in inputs]
        origin = origins[0] if origins else 'the available records'
        fields = schemas.get(inputs[0], []) if inputs else []
        step = {'step': number, 'operation': op, 'input_scopes': origins,
                'input_steps': [local_steps[n] if n in local_steps else external_steps[n] for n in inputs]}
        # Exclude original program IDs and executable links from the projection.
        step.update({k: v for k, v in original.items()
                     if k not in {'id', 'op', 'inputs', 'evidence_acquisition'}})
        sid = original.get('source_id')
        text = ''
        scope = origin
        if op == 'scan':
            scope = sid + ' records'
            text = f'Read {sid} and collect its available records.'
        elif op == 'lookup_by_key':
            key, source_key = original['input_key_field'], original['source_key_field']
            text = f'Use {key} from {origin} to query {sid} by {source_key}.'
            scope = sid + ' records'
        elif op in {'filter_eq', 'filter_gte'}:
            comparison = 'equal to' if op == 'filter_eq' else 'at least'
            value = json.dumps(original['value'], ensure_ascii=False)
            text = f'In {origin}, keep only records with {original["field"]} {comparison} {value}.'
            filters.append({'step': number, 'scope': origin, 'field': original['field'],
                            'comparison': comparison, 'value': original['value']})
        elif op == 'project':
            fields = list(original['fields'])
            rename = original.get('rename_fields', {})
            text = 'Retain ' + ', '.join(fields) + '.'
            if rename:
                text += ' Name ' + ', '.join(f'{k} as {v}' for k, v in rename.items()) + '.'
            fields = [rename.get(f, f) for f in fields]
        elif op == 'join':
            left, right = original['left_on'], original['right_on']
            text = f'Match {origins[0]} with {origins[1]} using {left} = {right}.'
            scope = 'the matched records'
            fields = list(dict.fromkeys(f for n in inputs for f in schemas.get(n, [])))
        elif op == 'unique':
            text = 'Remove duplicate records from this selection.'
        elif op == 'group_count':
            text = f'Count the selected records separately for each {original["field"]}.'
            fields = [original['field'], 'count']
        elif op == 'goal_compute':
            goal = original['goal_operation']
            args = ', '.join(original['input_fields'])
            operation = goal['op']
            meanings = {
                'temporal_difference': 'calculate the elapsed days between',
                'temporal_midpoint': 'calculate the midpoint date between',
                'geographic_distance': 'calculate the great-circle distance in kilometres between',
                'compare_select': f'select the {goal.get("direction", "higher")} value from',
                'verify': f'explain the {goal.get("relation", "comparison")} judgment using',
                'arithmetic': f'calculate {goal.get("function", "the specified arithmetic").replace("_", " ")} of',
            }
            if operation not in meanings:
                supported = False
            text = f'Using {origin}, {meanings.get(operation, operation)} {args}; store it as {original["output_field"]}.'
            if operation == 'geographic_distance':
                text += ' Round to the nearest kilometre.'
            if 'round_digits' in goal:
                text += f' Round to {goal["round_digits"]} decimal places.'
            fields = list(fields) + [original['output_field']]
        elif op == 'goal_format':
            text = 'Format the answer with ' + ', '.join(
                f'{key} from {value}' for key, value in original['bindings'].items()) + '.'
            if original['output_kind'] == 'date_time':
                text += ' Use YYYY-MM-DD.'
            fields = [original['output_field']]
        else:
            supported = False
        step['instruction'] = text
        steps.append(step)
        scopes[original['id']] = scope
        schemas[original['id']] = fields
        # Preserve operation order, including intermediate projection and deduplication.
        if op != 'scan':
            clauses.append(text)
    output = list(target['outputs']['typed_relation'])
    intro = ('Inspect ' + '; '.join(source_names.values()) + '. ' if source_names
             else 'Use the previously returned evidence. ')
    if not clauses:
        clauses = ['Collect the available records and preserve their field associations.']
    instruction = intro + ' '.join(clauses) + ' Return only ' + ', '.join(output) + ' with supporting observations.'
    return {'version': VERSION, 'steps': steps, 'filters': filters,
            'source_ids': list(target['source_ids']), 'output_fields': output,
            'reference_instruction': instruction, 'fallback_supported': supported}


def contract_prose_issues(instruction: str, contract: dict) -> list[str]:
    """Necessary lexical checks, not a claim of unrestricted NL equivalence."""
    issues = []
    for condition in contract['filters']:
        field = re.escape(condition['field']).replace('_', r'[ _-]')
        value = str(condition['value'])
        if (not re.search(r'(?<!\w)' + field + r'(?!\w)', instruction, re.I)
                or value.casefold() not in instruction.casefold()):
            issues.append('condition_missing:' + str(condition['step']))
    return issues


def critic_repair_guidance(review: dict, candidate: dict, assignments: list[dict]) -> list[str]:
    """Validate issue locations and resolve repairs from authoritative local facts.

    A reviewer cannot tell the author to remove a required field or alter a
    condition. Legacy prose-only judgments get a bounded, contract-led repair.
    """
    guidance = []
    tasks = candidate.get('params', {}).get('tasks', [])
    for issue in review.get('issues', []) if isinstance(review.get('issues', []), list) else []:
        if not isinstance(issue, dict):
            continue
        index, field = issue.get('task_index'), issue.get('field')
        if index is None and field == 'reasoning':
            text = candidate.get('reasoning', '')
            expected = 'Use only already returned evidence and describe this round as pending work.'
        elif type(index) is int and 0 <= index < len(tasks) and field in {'task_instruction', 'context'}:
            text = tasks[index].get(field, '')
            contract = assignments[index].get('semantic_contract', {})
            step = next((s for s in contract.get('steps', [])
                         if s['step'] == issue.get('contract_step')), None)
            expected = (step['instruction'] if step else assignments[index]['target_instruction'])
        else:
            continue
        quote = issue.get('quote', '')
        if not isinstance(quote, str):
            continue
        if quote and quote not in text:
            prefix = re.split(r'\.\.\.|…', quote, maxsplit=1)[0].strip()
            if len(prefix) < 10 or prefix not in text:
                continue
            quote = prefix
        if not quote and issue.get('kind') not in {'missing_condition', 'missing_requirement'}:
            continue
        guidance.append(f'Revise {field}' + (f' of task {index + 1}' if index is not None else '')
                        + (f' near {json.dumps(quote)}' if quote else '')
                        + '. Preserve this authoritative requirement: ' + expected)
    if not guidance:
        # Recover exact candidate locations from legacy reviews, never their advice.
        for quote in re.findall(r"['\"]([^'\"]{10,180})['\"]", str(review.get('reason', ''))):
            if quote in candidate.get('reasoning', ''):
                guidance.append('Recheck the reasoning claim ' + json.dumps(quote)
                                + ' against already returned facts; do not add counts of overlapping sets.')
    for item in assignments:
        witness = item.get('semantic_contract', {}).get('observed_identifier_join')
        if witness:
            guidance.append('The already returned identifier lists have exactly these shared identifiers: '
                            + json.dumps(witness['shared_identifiers'])
                            + '. Use their overlap for this join; do not add the input list counts.')
    if not guidance:
        # No private reviewer dialogue or unvalidated rewriting advice is copied.
        guidance.append('The previous prose was rejected. Revise its factual claims against the complete current-round references; do not infer this task from the final question alone.')
        guidance.extend(f'Task {i + 1}: preserve exactly {item["target_instruction"]}'
                        for i, item in enumerate(assignments))
    return guidance


def contract_fallback(assignments: list[dict], template: dict) -> dict | None:
    from copy import deepcopy
    if not all(a.get('semantic_contract', {}).get('fallback_supported') for a in assignments):
        return None
    action = deepcopy(template)
    if len(assignments) > 1:
        action['reasoning'] = ('The information currently available supports these independent investigations. '
                               'I will request their evidence together, then assess the returned findings.')
    else:
        action['reasoning'] = ('The information currently available supports the next investigation. '
                               'I will request the specified evidence before drawing further conclusions.')
    return action


def public_identifier_join(target: dict, delivered: dict) -> dict | None:
    """Witness only finite explicit public ID lists for a true two-parent join."""
    parents = target.get('depends_on', [])
    external = set(target.get('external_inputs', []))
    if len(parents) != 2 or not any(
        s['op'] == 'join' and s.get('left_on') == s.get('right_on') == 'case_id'
        and len(s.get('inputs', [])) == 2 and set(s['inputs']) <= external
        for s in target['operator_chain']):
        return None
    token = r'Dossier-[A-Za-z0-9]+'
    listing = token + r'(?:\s*(?:,\s*(?:and\s+)?|and\s+)' + token + r')*'
    pattern = re.compile(r'(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten) dossiers? match both clues:\s*)?(' + listing + r')[.;]*', re.I)
    numbers = {word: i for i, word in enumerate(['one','two','three','four','five','six','seven','eight','nine','ten'], 1)}
    groups = []
    for parent in parents:
        text = delivered[parent]['finish_result']['result'].strip()
        match = pattern.fullmatch(text)
        if not match:
            return None
        ids = re.findall(token, match.group(2), re.I)
        count = match.group(1)
        if (len(set(ids)) != len(ids) or (count and len(ids) !=
                (int(count) if count.isdigit() else numbers[count.casefold()]))):
            return None
        groups.append(ids)
    return {'input_identifier_sets': groups,
            'shared_identifiers': sorted(set(groups[0]) & set(groups[1])),
            'scope': 'Literal overlap of already returned public lists, not new source evidence or a future result.'}


def complete_prose_review(critic, system: str, payload: str, *, structured: bool):
    """Enforce located review issues on capable backends; retain mock/legacy APIs."""
    from inspect import signature
    if not structured or 'response_schema' not in signature(critic.complete_json).parameters:
        return critic.complete_json(system, payload)
    nullable_index = {'anyOf': [{'type': 'integer'}, {'type': 'null'}]}
    issue = {'type': 'object', 'additionalProperties': False,
             'properties': {'task_index': nullable_index, 'contract_step': nullable_index,
                 'field': {'type': 'string', 'enum': ['reasoning','task_instruction','context']},
                 'kind': {'type': 'string', 'enum': ['missing_condition','wrong_condition','wrong_lookup','extra_output','unsupported_claim','false_dependency','style']},
                 'quote': {'type': 'string', 'maxLength': 180}},
             'required': ['task_index','contract_step','field','kind','quote']}
    props = {'verdict': {'type': 'string', 'enum': ['pass','fail']},
             **{k: {'type': 'boolean'} for k in ['grounded','assignment_fidelity','o1_style']},
             'reason': {'type': 'string', 'maxLength': 320}, 'repair_hint': {'type': 'string', 'maxLength': 240},
             'issues': {'type': 'array', 'items': issue, 'maxItems': 3}}
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': props, 'required': list(props)}
    return critic.complete_json(system, payload, response_schema=schema)
