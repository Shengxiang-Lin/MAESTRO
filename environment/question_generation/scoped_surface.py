"""Reversible, bounded English surface forms for the public question contract."""
from __future__ import annotations

import re
from collections import Counter
from datetime import date
from decimal import Decimal

from .public_semantics import public_question_contract


class UnsupportedSurface(ValueError):
    """The contract needs a construction outside the supported grammar."""

class ScopedPlan(list):
    """English productions paired with their compiler-derived semantic trees."""

    def __init__(self, groups, meanings, query):
        super().__init__(groups)
        self.meanings = meanings
        self.query = query


def _required(value):
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedSurface('missing public operand')
    return value


def _value(value, operation, prose):
    value = _required(value)
    if not prose:
        return value
    if operation.endswith('_DATE') and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        parsed = date.fromisoformat(value)
        return f'{parsed:%B} {parsed.day}, {parsed.year}'
    if operation.endswith('_NUMBER'):
        match = re.fullmatch(r'(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(.*)', value)
        if match:
            if match[1].isdigit() and int(match[1]) <= 20 and not match[2]:
                from .contracts import _SMALL_NUMBERS
                return next(word for word, number in _SMALL_NUMBERS.items()
                            if number == int(match[1]))
            return format(Decimal(match[1].replace(',', '')), ',f') + match[2]
    return value


def _comparison(item, prose):
    comparison = _required(item.get('comparison'))
    return f"{comparison} {_value(item.get('public_value'), item['operation'], prose)}"


_NOUNS = {
    'Person': ('person', 'people'), 'Place': ('place', 'places'),
    'Country': ('country', 'countries'), 'Organization': ('organization', 'organizations'),
    'Administrative Area': ('administrative area', 'administrative areas'),
    'City In The United States': ('city in the United States', 'cities in the United States'),
    'Event': ('event', 'events'), 'Language': ('language', 'languages'),
    'Creative Work': ('creative work', 'creative works'), 'Book': ('book', 'books'),
    'Season sports': ('sports season', 'sports seasons'),
    'TVSeries': ('TV series', 'TV series'), 'Way': ('way', 'ways'),
    'Human Made Geographical Entity': ('human-made geographical entity', 'human-made geographical entities'),
}

def _noun(kind, *, singular=False):
    if kind in _NOUNS:
        one, many = _NOUNS[kind]
        return 'the ' + one if singular else many
    if kind:
        return f'the entity of type {kind}' if singular else f'{kind} entities'
    return 'entities'

def _join(items, connector):
    if len(items) < 2:
        raise UnsupportedSurface('set join needs two operands')
    if len(items) == 2:
        return f'{items[0]} {connector} {items[1]}'
    return ', '.join(items[:-1]) + f', {connector} {items[-1]}'


def _conditions(scope, operations, prose):
    kinds, attributes = [], []
    for condition in scope['conditions']:
        operation = operations[condition['step']]
        if operation == 'FILTER_TYPE':
            kinds.append(_required(condition.get('entity_type')))
        else:
            target = _value(condition.get('public_value'), operation, prose)
            comparison = _required(condition.get('comparison'))
            attribute = _required(condition.get('attribute'))
            article = 'an' if attribute[0].casefold() in 'aeiou' else 'a'
            connector = 'of' if comparison == 'equal to' else comparison
            attributes.append(f'{article} {attribute} {connector} {target}')
    if len(set(kinds)) > 1:
        raise UnsupportedSurface('multiple type constraints')
    return next(iter(kinds), None), ' and '.join(attributes)


def _relation(scope, source, noun, condition):
    relation = _required(scope.get('relation'))
    head = noun + (f' with {condition}' if condition else '')
    grouped = source.endswith(' group')
    source_members = f'members of {source}' if grouped else source
    target = f'a member of {source}' if grouped else source
    if scope['direction'] == 'forward':
        if relation == 'death place' and noun == 'entities':
            places = 'places' + (f' with {condition}' if condition else '')
            return f'{places} recorded as places of death for {source_members}'
        if relation == 'birth place' and noun in ('places', 'entities'):
            places = 'places' + (f' with {condition}' if condition else '')
            return f'{places} recorded as birthplaces of {source_members}'
        if relation == 'spouse':
            return f'{head} listed as spouses of {source_members}'
        if relation == 'location':
            return f'{head} recorded as locations of {source_members}'
        if relation == 'nationality':
            return f'{head} recorded as the nationality of {source_members}'
        if relation == 'member of':
            return (f'{head} with at least one member in {source}' if grouped
                    else f'{head} with at least one member among {source}')
        if relation == 'knows language':
            return f'{head} recorded as known by {source_members}'
        return f"{head} listed as targets of the '{relation}' relation from {source_members}"
    if relation == 'death place' and noun == 'people':
        return (f'{head} recorded as dying at one of {source}' if source=='the reference places'
                else f'{head} recorded as dying at {target}')
    if relation == 'spouse':
        return f'{head} with a recorded spouse among {source_members}'
    if relation == 'member of':
        return f'{head} listed as members of {target}'
    if relation == 'named after':
        return f'{head} named after {target}'
    if relation == 'terminus':
        return f'{head} with {target} as a recorded terminus'
    return f"{head} whose recorded '{relation}' points to {target}"


def _phrase(scope, refs, operations, clues, prose):
    kind, condition = _conditions(scope, operations, prose)
    definition = scope['definition']
    inputs = [refs[key] for key in scope['inputs']]
    suffix = f' with {condition}' if condition else ''
    if definition == 'FIND':
        start = scope['program_steps'][0]
        if start in clues:
            clue = clues[start]
            if not set(clue['identifying_node_ids']).issubset(scope['program_steps']):
                raise UnsupportedSurface('identification clue must stay with its anchor')
            from environment.task_construction.rendering.naming import type_label
            anchor_type = type_label(clue['entity_type'])
            if kind and kind != anchor_type:
                raise UnsupportedSurface('additional type differs from anchor type')
            noun = _noun(anchor_type, singular=True)
            values = [c for c in scope['conditions'] if c.get('attribute')]
            if anchor_type == 'Person' and len(values) == 1 and values[0].get('comparison') == 'equal to':
                item = values[0]
                phrase = {'birth date':'was born', 'death date':'died'}.get(item['attribute'])
                if phrase and operations[item['step']] == 'FILTER_DATE':
                    return f"{noun} who {phrase} on {_value(item['public_value'], 'FILTER_DATE', prose)}"
            return noun + suffix
        anchor = _required(scope.get('public_anchor'))
        if scope.get('type_anchor'):
            if kind and kind != scope['type_anchor']:
                raise UnsupportedSurface('different type filters on a type anchor')
            return f"{_noun(scope['type_anchor'])}{suffix} in the relevant records"
        if kind:
            raise UnsupportedSurface('explicit anchor has a type refinement')
        return anchor + (f' if it has {condition}' if condition else '')
    if definition == 'FIND_ALL':
        return f'{_noun(kind)}{suffix} in the relevant records'
    if definition == 'RELATE':
        return _relation(scope, inputs[0], _noun(kind), condition)
    if definition.startswith('FILTER_'):
        return f'{_noun(kind)}{suffix} drawn from {inputs[0]}'
    if definition == 'INTERSECT':
        return 'entities in both ' + _join(inputs, 'and') if len(inputs)==2 else 'entities common to '+_join(inputs,'and')
    if definition == 'UNION':
        return 'entities in at least one of '+_join(inputs, 'or')
    if definition in ('SELECT_AMONG', 'SELECT_BETWEEN'):
        pool = inputs[0] if len(inputs)==1 else _join(inputs, 'or')
        return f"entities with the {_required(scope.get('ranking'))} recorded {_required(scope.get('rank_attribute'))} among {pool}"
    raise UnsupportedSurface(f'unsupported intermediate scope {definition}')

_ORDINALS = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth')

def _output(scope, refs, by_id, prose):
    operation = scope['definition']
    if scope['conditions']:
        raise UnsupportedSurface('unsupported filtered output scope')
    inputs = [refs[key] for key in scope['inputs'] if key in refs]
    def kind_of(sid):
        item = by_id[sid]
        kinds = [c['entity_type'] for c in item['conditions'] if c.get('entity_type')]
        if kinds:
            return kinds[0]
        if item['definition'] == 'INTERSECT':
            return next((kind_of(key) for key in item['inputs'] if kind_of(key)), None)
        if item['definition'] == 'UNION':
            values = [kind_of(key) for key in item['inputs']]
            return values[0] if values and all(v == values[0] for v in values) else None
        if item['definition'].startswith('FILTER_'):
            return kind_of(item['inputs'][0])
        return item.get('type_anchor')
    noun = _noun(kind_of(scope['inputs'][0]))
    def joined_question(count=False):
        joined = by_id[scope['inputs'][0]]
        if joined['definition'] != 'INTERSECT' or len(joined['inputs']) != 2:
            return ()
        left, right = [refs[key] for key in joined['inputs']]
        if left.startswith(noun + ' with ') and right.startswith(noun + ' '):
            left, right = right, left
        if not right.startswith(noun + ' '):
            left, right = right, left
        if not right.startswith(noun + ' '):
            return ()
        tail = right[len(noun)+1:]
        if left.endswith(' group'):
            subject = f'{noun} in {left}'
        elif left.startswith(noun + ' '):
            subject = left
        else:
            return ()
        predicate = ('also have ' + tail[5:]) if tail.startswith('with ') else 'are also ' + tail
        prefix = 'How many distinct' if count else 'Which'
        return (f'{prefix} {subject} {predicate}?',)
    if operation == 'QUERY_NAME':
        if inputs[0].startswith('entities in both '):
            tail = inputs[0][len('entities in both '):]
            return joined_question() or (f'Which {noun} belong to both {tail}?', f'Name the {noun} in both {tail}.')
        if inputs[0].endswith(' group'):
            return (f'Which {noun} are in {inputs[0]}?', f'Name the {noun} in {inputs[0]}.')
        return (f'What are the names of {inputs[0]}?', f'Identify {inputs[0]} by name.')
    if operation == 'COUNT':
        source = inputs[0]
        if source.startswith('entities in both '):
            tail = source[len('entities in both '):]
            return joined_question(True) or (f'How many distinct {noun} belong to both {tail}?', f'Count the distinct {noun} in both {tail}.')
        first = (f'How many distinct {noun} are in {source}?' if source.endswith('group')
                 else f'How many {source} are there?')
        return (first, f'What is the number of {source}?')
    if operation == 'QUERY_ATTR':
        attr = _required(scope.get('output_attribute'))
        return (f'What {attr} values are recorded for {inputs[0]}?',
                f'List the recorded {attr} values for {inputs[0]}.')
    if operation == 'QUERY_RELATION':
        return (f'Which relations are recorded from {inputs[0]} to {inputs[1]}?',
                f'Name the relations whose subjects are in {inputs[0]} and whose targets are in {inputs[1]}.')
    if operation.startswith('VERIFY_'):
        value_scope = by_id[scope['inputs'][0]]
        if value_scope['definition'] != 'QUERY_ATTR':
            raise UnsupportedSurface('verification requires a recorded attribute')
        source = refs[value_scope['inputs'][0]]
        attr = _required(value_scope.get('output_attribute'))
        condition = _comparison({**scope, 'operation': operation}, prose)
        return (f'For {source}, is any recorded {attr} {condition}?',
                f'Is there a recorded {attr} {condition} for {source}?')
    if operation in ('SELECT_AMONG', 'SELECT_BETWEEN'):
        pool = inputs[0] if len(inputs)==1 else _join(inputs, 'or')
        rank = _required(scope.get('ranking'))
        attr = _required(scope.get('rank_attribute'))
        return (f'Among {pool}, which entities have the {rank} recorded {attr}? Include ties.',
                f'Which entities tie for or attain the {rank} recorded {attr} among {pool}?')
    raise UnsupportedSurface(f'unsupported question output {operation}')

def _compose(contract, clues, prose, split_ids=None):
    scopes = contract['scopes']
    by_id = {scope['id']: scope for scope in scopes}
    operations = {int(op['id'][1:]): op['operation'] for op in contract['obligations']}
    consumers = Counter(key for scope in scopes for key in scope['inputs'])
    terminal = by_id[contract['output']]
    refs, groups, split, sentence_scopes = {}, [], [], []
    for scope in scopes:
        sid = scope['id']
        if sid == contract['output']:
            groups.append(_output(scope, refs, by_id, prose))
            sentence_scopes.append(sid)
            break
        if scope['definition'] == 'QUERY_ATTR':
            # A terminal existential query inlines its attribute lookup.
            continue
        phrase = _phrase(scope, refs, operations, clues, prose)
        if split_ids is None:
            inline_join = (scope['definition']=='INTERSECT' and consumers[sid]==1 and
                           terminal['definition'] in ('QUERY_NAME','COUNT') and sid in terminal['inputs'])
            bind = (not inline_join and len(split) < len(_ORDINALS) and
                    (len(phrase.split()) > 23 or scope.get('relation')=='death place' and scope.get('direction')=='forward' or (consumers[sid] > 1 and len(phrase.split()) > 7)
                     or (scope.get('relation') == 'member of' and len(phrase.split()) > 12)
                     or (scope['definition']=='FIND' and ' if it has ' in phrase)))
        else:
            bind = sid in split_ids
        if bind:
            reference_places = (scope.get('relation')=='death place' and scope.get('direction')=='forward'
                                and 'the reference places' not in refs.values())
            generic_count = sum(refs[key].endswith(' group') for key in split)
            label = 'the reference places' if reference_places else f'the {_ORDINALS[generic_count]} group'
            if ' if it has ' in phrase and scope['definition']=='FIND':
                phrase += '. Otherwise, this group is empty'
            groups.append((f'Let {label} consist of {phrase}.',
                           f'Define {label} as {phrase}.'))
            sentence_scopes.append(sid)
            split.append(sid)
            refs[sid] = label
        else:
            refs[sid] = phrase
    return groups, split, sentence_scopes


def _semantic_tree(contract, sid, operations, clues):
    """Answer semantics only: no execution numbers, scheduling or evidence goals."""
    from .contracts import normalize_value
    scope = next(s for s in contract['scopes'] if s['id'] == sid)
    result = {'definition': scope['definition'], 'conditions': []}
    for condition in scope['conditions']:
        if condition.get('entity_type'):
            result['conditions'].append({'entity_type': condition['entity_type']})
        else:
            result['conditions'].append({
                'attribute': condition['attribute'], 'comparison': condition['comparison'],
                'value': normalize_value(condition['public_value'], operations[condition['step']]),
            })
    for key in ('relation', 'direction', 'ranking', 'rank_attribute', 'output_attribute'):
        if scope.get(key) is not None:
            result[key] = scope[key]
    if scope['definition'] == 'FIND':
        clue = clues.get(scope['program_steps'][0])
        if clue:
            result['anchor_type'] = clue['entity_type']
        else:
            result['anchor'] = scope.get('public_anchor')
        if scope.get('type_anchor'):
            result['type_anchor'] = scope['type_anchor']
    if scope['definition'] == 'FIND_ALL':
        result['range'] = 'bounded_public_task_evidence'
    if scope['definition'].startswith('VERIFY_'):
        result['comparison'] = scope['comparison']
        result['value'] = normalize_value(scope['public_value'], scope['definition'])
        result['quantifier'] = 'any_recorded_value'
    if scope['definition'] in ('SELECT_AMONG', 'SELECT_BETWEEN'):
        result['include_ties'] = True
    result['inputs'] = [_semantic_tree(contract, key, operations, clues) for key in scope['inputs']]
    return result

def build_scoped_plan(world, canonical, gold, reference):
    """Compile alternatives from a checked public projection and identity clues."""
    from environment.task_construction.rendering.naming import predicate_label, type_label
    contract = public_question_contract(world, canonical, gold)
    raw_nodes = {int(n['node_id']): n for n in world['program']['nodes']}
    # These labels are already public through the controlled renderer's lexical forms.
    relation_aliases = {('spouse','forward'):'spouses of', ('spouse','backward'):'married to',
                        ('member of','forward'):'belongs to', ('member of','backward'):'members of'}
    clues = {nid: clue for clue in reference['verified_identity_clues']
             for nid in clue['find_node_ids']}
    for scope in contract['scopes']:
        if scope['definition']=='RELATE' and scope.get('relation') is None:
            relation = predicate_label(raw_nodes[scope['program_steps'][0]]['inputs'][0])
            public_form = relation_aliases.get((relation, scope['direction']))
            if public_form and public_form in canonical.casefold():
                scope['relation'] = relation
        if scope['definition'] == 'FIND':
            token = raw_nodes[scope['program_steps'][0]]['inputs'][0]
            if token.startswith('type:'):
                kind = type_label(token.split(':', 1)[1])
                if not scope.get('public_anchor') or scope['public_anchor'].casefold() != kind.casefold():
                    raise UnsupportedSurface('missing public type anchor')
                scope['type_anchor'] = kind
    plain, split, sentence_scopes = _compose(contract, clues, False)
    prose, _, _ = _compose(contract, clues, True, set(split))
    groups = [tuple(dict.fromkeys([*p, *q])) for p, q in zip(prose, plain, strict=True)]
    if not groups or sum(max(map(len, group)) for group in groups) + len(groups) > 1200:
        raise UnsupportedSurface('scoped question exceeds supported length')
    operations = {int(op['id'][1:]): op['operation'] for op in contract['obligations']}
    meanings = [_semantic_tree(contract, sid, operations, clues) for sid in sentence_scopes]
    return ScopedPlan(groups, meanings,
                      _semantic_tree(contract, contract['output'], operations, clues))

def render_scoped(groups, choices):
    if (not isinstance(choices, list) or len(choices) != len(groups)
            or any(type(i) is not int or not 0 <= i < len(g) for g, i in zip(groups, choices))):
        raise ValueError('invalid scoped sentence choices')
    return ' '.join(group[index] for group, index in zip(groups, choices))

def read_scoped(question, groups):
    """Consume the actual question using only the contract's English productions."""
    pattern = ' '.join('(' + '|'.join(re.escape(t) for t in sorted(group, key=len, reverse=True)) + ')'
                       for group in groups)
    match = re.fullmatch(pattern, ' '.join(question.split()))
    if match is None:
        return None
    return [group.index(sentence) for group, sentence in zip(groups, match.groups())]


def read_scoped_semantics(question, plan):
    """Fully consume actual text, then apply the matched productions' meanings.

    This is a bounded grammar parser, not an open-ended English interpreter.
    Unknown text, extra restrictions and moved slots cannot inherit a meaning.
    """
    if not isinstance(plan, ScopedPlan):
        raise ValueError('semantic readback requires compiler-derived productions')
    choices = read_scoped(question, plan)
    if choices is None:
        return None
    return {
        'query': plan.meanings[-1],
        'sentences': [{'text': options[choice], 'meaning': meaning}
                      for options, choice, meaning in zip(plan, choices, plan.meanings, strict=True)],
    }

