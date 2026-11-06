"""Stage-3-only eligibility; frozen questions/programs are never repaired here."""
from __future__ import annotations
import hashlib,json,re
from difflib import SequenceMatcher
from pathlib import Path

def digest(value):
    text=value if isinstance(value,str) else json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()

def controlled_question_reference(gold, record, rendered):
    """Reconstruct supported renderer grammar from the executable input.

    Stored canonical/draft text is not a certificate. Check complete dependency
    coverage and the uses of every concealed identity before trusting a freshly
    rendered reference. Unsupported grammars still require semantic review.
    """
    from environment.task_construction.executor import SymbolicExecutor
    from environment.task_construction.rendering.pipeline import (
        program_from_dict, world_from_dict, render_task_record,
    )
    from environment.task_construction.rendering.query_grounding import build_public_query_plan

    seed=(getattr(rendered,'hidden_metadata',{}) or {}).get('rendering_seed')
    if type(seed) is not int:
        return None
    program=program_from_dict(record['program'])
    nodes=program.node_map()
    order=program.topological_order()
    if not order:
        return None
    arities={
        'FIND':(0,1),'FIND_ALL':(0,0),'RELATE':(1,2),
        'FILTER_TYPE':(1,1),'FILTER_STRING':(1,3),'FILTER_NUMBER':(1,3),
        'FILTER_YEAR':(1,3),'FILTER_DATE':(1,3),
        'QUERY_NAME':(1,0),'QUERY_ATTR':(1,1),'COUNT':(1,0),
        'QUERY_RELATION':(2,0),'SELECT_AMONG':(1,2),'SELECT_BETWEEN':(2,2),
        'VERIFY_STRING':(1,2),'VERIFY_NUMBER':(1,2),
        'VERIFY_YEAR':(1,2),'VERIFY_DATE':(1,2),
    }
    for node in nodes.values():
        if node.op in ('INTERSECT','UNION'):
            if len(node.dependencies)<2 or node.inputs:
                return None
        elif node.op not in arities or (
            len(node.dependencies),len(node.inputs)
        )!=arities[node.op]:
            return None
        if node.op=='RELATE' and node.inputs[1] not in ('forward','backward'):
            return None
    # The renderer describes only the terminal's ancestors. An unconsumed
    # branch must not be certified simply because its text was never rendered.
    needed=set()
    def visit(nid):
        if nid not in needed:
            needed.add(nid)
            for dep in nodes[nid].dependencies:
                visit(dep)
    visit(order[-1])
    if needed!=set(nodes):
        return None
    world=world_from_dict(record['world'])
    plan=build_public_query_plan(record,program,world)
    trace=SymbolicExecutor().execute_trace(program,world)
    for clue in plan.latent_entities:
        for find_id in clue.find_node_ids:
            users=[n for n in nodes.values() if find_id in n.dependencies]
            if not users:
                return None
            for user in users:
                if clue.rule=='unique_attribute_clue':
                    if (user.node_id not in clue.identifying_node_ids
                            or user.dependencies!=(find_id,)
                            or user.op not in (
                                'FILTER_STRING','FILTER_NUMBER','FILTER_YEAR','FILTER_DATE'
                            )):
                        return None
                elif clue.rule=='unique_intersection_clue':
                    visible=[d for d in user.dependencies
                             if d not in clue.find_node_ids]
                    if user.op!='INTERSECT' or not visible:
                        return None
                    result=set(trace[visible[0]])
                    for dep in visible[1:]:
                        result.intersection_update(trace[dep])
                    if result!={clue.entity_id} or result!=set(trace[user.node_id]):
                        return None
                else:
                    return None
    regenerated=render_task_record(record,seed=seed)
    if regenerated.task_id!=gold['task_id']:
        return None
    return {
        'question':regenerated.question,
        'question_sha256':digest(regenerated.question),
        'program_sha256':digest(record['program']),
        'rendering_seed':seed,
        'verified_node_ids':order,
        'verified_identity_clues':[c.to_dict() for c in plan.latent_entities],
        'contract':'controlled_renderer_dependency_and_clue_check_v1',
    }

def semantic_review_nodes(gold, record):
    """Use the full graph, eliminating only compiler-certified redundancies.

    Public delegation cuts are not a semantic graph: they retain original
    node dependencies even after projection. Rewire those dependencies here
    without modifying either the frozen program or the execution plan.
    """
    original={int(n['node_id']):n for n in record['program']['nodes']}
    public={int(op['node_id']):op for f in gold['fragments'] for op in f['operations']}
    replacements={}
    removed=set()
    for certificate in gold.get('public_projections',[]):
        if certificate.get('rule') not in (
            'unique_intersection_clue','witness_verified_intersection'
        ):
            continue
        hidden=certificate.get('concealed_find_node_id')
        visible=certificate.get('identifying_node_id')
        if (hidden not in original or visible not in original
                or hidden in public or original[hidden]['op']!='FIND'):
            continue
        joins=[nid for nid,n in original.items()
               if nid not in public and n['op']=='INTERSECT'
               and len(n['dependencies'])==2
               and set(n['dependencies'])=={hidden,visible}]
        if len(joins)!=1:
            continue
        join=joins[0]
        if {nid for nid,n in original.items() if hidden in n['dependencies']}!={join}:
            continue
        replacements[join]=visible
        removed.update((hidden,join))
    def resolve(nid):
        while nid in replacements:
            nid=replacements[nid]
        return nid
    return {
        nid:{**public.get(nid,raw),
             'dependencies':[resolve(int(d)) for d in raw['dependencies']]}
        for nid,raw in original.items() if nid not in removed
    }

def semantic_review_steps(nodes):
    """Give the reviewer explicit endpoint and branch semantics, not op guesses."""
    steps=[]
    for nid,op in sorted(nodes.items()):
        action,args,deps=op['op'],op['inputs'],op['dependencies']
        inputs=', '.join(f'S{d}' for d in deps)
        meaning=''
        if action=='FIND_ALL':
            meaning='All entities in the finite task evidence domain; discover them from task sources.'
        elif action=='FIND':
            meaning=f'Identify the entity/type matching {args[0]}.'
        elif action=='RELATE':
            backward=len(args)>1 and args[1]=='backward'
            meaning=(f'Return subjects x with x --{args[0]}--> y and y in {inputs}.'
                     if backward else
                     f'Return objects y with x --{args[0]}--> y and x in {inputs}.')
        elif action=='FILTER_TYPE':
            meaning=f'Keep entities in {inputs} of type {args[0]}.'
        elif action.startswith('FILTER_') and len(args)>=3:
            meaning=f'Keep entities x in {inputs} whose own {args[0]} {args[1]} {args[2]}.'
        elif action in ('INTERSECT','UNION'):
            meaning=f'{"Intersection" if action=="INTERSECT" else "Union"} of {inputs}.'
        elif action in ('SELECT_AMONG','SELECT_BETWEEN'):
            candidates=f'union of {inputs}' if action=='SELECT_BETWEEN' else inputs
            from environment.task_construction.executor import _select_minimum
            rank='minimum' if _select_minimum(args[1]) else 'maximum'
            meaning=f'Return entities attaining the {rank} recorded {args[0]} across the {candidates}; include ties. This is ranking, not a threshold filter.'
        elif action=='QUERY_ATTR':
            meaning=f'Return recorded {args[0]} values of entities in {inputs}.'
        elif action=='QUERY_NAME':
            meaning=f'Identify entities in {inputs} by name.'
        elif action=='COUNT':
            meaning=f'Count distinct entities in {inputs}.'
        elif action.startswith('VERIFY_'):
            meaning=f'Check whether any value in {inputs} {args[0]} {args[1]}.'
        elif action=='QUERY_RELATION':
            meaning=f'Return relation names from subjects in S{deps[0]} to objects in S{deps[1]}.'
        steps.append({'step':nid,'action':action,'arguments':args,
                      'input_steps':deps,'meaning':meaning})
    return steps

def semantic_review_expression(steps, *, max_chars=12000):
    """Expand dependency scopes, with a bound against repeated DAG branches."""
    by_id={step['step']:step for step in steps}
    expanded={}
    def expand(nid):
        if nid in expanded:
            return expanded[nid]
        step=by_id[nid]
        dependencies={d:expand(d) for d in step['input_steps']}
        text=re.sub(r'\bS(\d+)\b',
                    lambda match:'('+dependencies[int(match[1])]+')',
                    step['meaning'])
        if len(text)>max_chars:
            raise OverflowError
        expanded[nid]=text
        return text
    consumed={d for step in steps for d in step['input_steps']}
    try:
        roots=[expand(nid) for nid in by_id if nid not in consumed]
        result='\n'.join(roots)
        return result if len(result)<=max_chars else None
    except OverflowError:
        # Keep the numbered semantics for large DAGs; never truncate a branch.
        return None

def check_question(gold,record,role,*,review=True,rendered=None):
    question=gold['question'];fingerprint={'question_sha256':digest(question),'program_sha256':digest(record['program'])}
    audit_path = Path(__file__).with_name('question_reviews.json')
    # This manually curated file is optional and not committed in all setups.
    # Missing file is NOT an approval: require a reconstructed renderer
    # certificate or an actual LLM semantic review.
    audited = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else []
    if not isinstance(audited, list):
        raise ValueError("question_reviews.json must be a JSON array")
    for item in audited:
        if item['task_id']==gold['task_id'] and all(item[k]==v for k,v in fingerprint.items()):
            return {'eligible':item.get('eligible',False),'reviewed':True,'method':'audited_exact_input_review','issues':[] if item.get('eligible') else [item['reason']],'audit_reason':item['reason'],**fingerprint}
    if not review or getattr(role,'offline',False) or not getattr(role,'backend',None):
        return {'eligible':True,'reviewed':False,'method':'offline_structure_only','issues':[],**fingerprint}
    reference=controlled_question_reference(gold,record,rendered)
    if reference and question==reference['question']:
        return {'eligible':True,'reviewed':True,'decision':'eligible',
                'method':'verified_controlled_renderer','issues':[],
                'renderer_certificate':reference,**fingerprint}
    nodes=semantic_review_nodes(gold,record)
    steps=semantic_review_steps(nodes)
    # Raw SELECT '<' and RELATE direction flags tempted small reviewers to
    # substitute their own operator definitions despite the explicit meaning.
    # Audit the executor-derived semantics, retaining IDs for exact witnesses.
    review_steps=[{'step':s['step'],'input_steps':s['input_steps'],
                   'meaning':s['meaning']} for s in steps]
    expanded_meaning=semantic_review_expression(steps)
    def normalized(text):
        return re.sub(r'[^a-z0-9]+','',str(text).casefold())
    # Missing literal relation wording is a review trigger, not a rejection:
    # faithful paraphrases are allowed, but an approval must bind each such
    # hop to actual question evidence instead of overlooking it.
    relation_gaps=[{
        'step':nid,'relation':op['inputs'][0],
        'direction':op['inputs'][1] if len(op['inputs'])>1 else 'forward',
        'input_steps':op['dependencies'],
    } for nid,op in sorted(nodes.items()) if op['op']=='RELATE'
        and normalized(str(op['inputs'][0]).split(':')[-1]) not in normalized(question)]
    # A filter mentioning an attribute does not stand in for QUERY_ATTR.
    # If its name disappeared entirely, require explicit output evidence too.
    relation_gaps.extend({
        'step':nid,'attribute':op['inputs'][0],'operation':'QUERY_ATTR',
        'input_steps':op['dependencies'],
    } for nid,op in sorted(nodes.items()) if op['op']=='QUERY_ATTR'
        and normalized(str(op['inputs'][0]).split(':')[-1]) not in normalized(question))
    # Explain the controlled renderer's grammar once instead of presenting a
    # long list of hypothetical objections and duplicate recursive graphs.
    review_rules=(
        'Audit whether the accepted question computes canonical_steps. Do not solve or use real-world assumptions. '
        'Read each step meaning as authoritative operator semantics; report only an actual difference in the question. '
        'Controlled question grammar: FIND_ALL discovers all entities in the finite task evidence domain. '
        'Task sources/references/web search specify acquisition, not extra eligibility constraints. '
        'For a nested description D, "entities connected from D by R", "R targets of D", '
        '"entities reached through R from D" and "entities linked to D via R" mean forward R from D. '
        '"entities whose R points to D", "entities linked to D as the source of R", '
        '"entities for which D is the R target" and "entities connected into D by R" mean backward R into D. '
        'A suffix "D with/whose/after requiring attribute condition" filters D, not an inner anchor. '
        '"overlap", "meeting both sides" and "satisfy each constraint" mean intersection. '
        '"union"/"either side"/"or" combine complete branches; filters within one branch stay in that branch. '
        'Selecting "which entity has the smallest/largest attribute value" returns entities attaining the minimum/maximum '
        'over the input pool, NOT attribute values and NOT a threshold comparison. '
        'When the terminal is QUERY_NAME, identifying entities means returning their names. '
        'Redundant type filters and equivalent filter/intersection forms are allowed. '
        'Attributes may have multiple values; impossible conditions/empty results are valid. '
        'Certified redundant singleton checks are already eliminated; all other steps remain required. '
        'Check the entire question, including its ending, for every constraint and final queried attribute. '
        'A missing relation traversal or changed filter owner is a real mismatch; a mere type match cannot excuse it. '
        'For ambiguity choose uncertain; for equivalence choose eligible with issues=[]. '
        'For mismatch cite existing program_steps and a SHORT EXACT question_quote (no ellipses) plus '
        'program_claim, question_claim and reason; kind is missing_constraint/scope/direction/join/output. '
        'At most two issues; quotes <=240 characters, claims/reasons <=180 characters. '
        'Write issues FIRST (only actual differences; matching steps are not issues), relation_coverage next, '
        'then the final decision LAST. If all proposed objections were withdrawn, issues=[] and decision=eligible. '
        'relation_coverage=[] unless required hops are requested. '
        'No deliberation or rewritten questions.'
    )
    review_input={
            'accepted_question':question,
            'certified_public_projections':gold.get('public_projections',[]),
            'canonical_steps':review_steps,
            'fully_expanded_output_meaning':expanded_meaning}
    if reference:
        changes=[{'reference_quote':reference['question'][a:b],
                  'accepted_quote':question[c:d]}
                 for tag,a,b,c,d in SequenceMatcher(
                     None,reference['question'],question,autojunk=False
                 ).get_opcodes() if tag!='equal']
        review_input.update({
            'verified_renderer_reference':reference['question'],
            'changed_spans':changes,
        })
        review_rules=(
            'The renderer reference has been reconstructed from the frozen program '
            'and its dependency/clue proofs checked. Compare the accepted rewrite '
            'to that reference, concentrating on changed_spans. Do not invent a '
            'direction discrepancy in an unchanged span. A removed relation name '
            'requires an actual equivalent traversal in the rewrite; generic '
            '"associated with" is insufficient. A removed attribute query requires '
            'an explicit equivalent output request, not just an attribute filter. '
            'Replacing "connected into" with "connected to" can lose direction; '
            'if the rewrite does not establish the endpoint, return uncertain. '
            +review_rules
        )
    prompt=review_rules+'\n'+json.dumps(review_input,ensure_ascii=False,separators=(',',':'))
    cache_path=getattr(role,'eligibility_cache',None)
    if not isinstance(cache_path,(str,Path)):cache_path=None
    # Test doubles and partially constructed clients can expose Mock objects
    # for unknown attributes. Cache identity must be stable, JSON-native and
    # never depend on a Mock repr or memory address.
    def identity(field):
        value = getattr(role.backend, field, None)
        return value if isinstance(value, str) and value else "unknown"
    cache_key=digest({**fingerprint,'reviewer_model':identity('model'),
        'reviewer_endpoint':identity('base_url'),'review_contract':12,
        'renderer_reference_sha256':reference['question_sha256'] if reference else None})
    if cache_path and Path(cache_path).exists():
        for line in Path(cache_path).read_text().splitlines():
            item=json.loads(line)
            if item.get('cache_key')==cache_key:
                verdict=item['verdict'];break
        else:verdict=None
    else:verdict=None
    cache_hit=verdict is not None
    def invalid_reason(value, *, required_hops=()):
        """Reject unsupported review *witnesses*, not the underlying question.

        Mismatch must cite an existing step and an EXACT short question span.
        Malformed / contradictory reviews are quarantined, never auto-approved.
        """
        if not (isinstance(value,dict)
                and value.get('decision') in ('eligible','mismatch','uncertain')
                and isinstance(value.get('issues'),list)):
            return 'invalid_review_schema'
        if value['decision']=='eligible':
            if value['issues'] != []:
                return 'eligible_with_issues'
            if not required_hops:
                return None
            coverage=value.get('relation_coverage')
            if not isinstance(coverage,list):
                return 'missing_relation_coverage'
            used_quotes=set()
            for gap in required_hops:
                matching=[
                    c for c in coverage
                    if isinstance(c,dict) and c.get('step')==gap['step']
                    and isinstance(c.get('question_quote'),str)
                    and 0<len(c['question_quote'])<=240
                    and c['question_quote'] in question
                    and isinstance(c.get('interpretation'),str)
                    and 0<len(c['interpretation'].strip())<=180
                ]
                if not matching:
                    return 'missing_relation_coverage'
                quote=matching[0]['question_quote']
                if quote in used_quotes:
                    return 'reused_relation_coverage'
                used_quotes.add(quote)
            return None
        if value['decision']=='uncertain':
            return None
        if not 1<=len(value['issues'])<=2:
            return 'missing_mismatch_witness'
        for issue in value['issues']:
            if not isinstance(issue,dict):
                return 'invalid_mismatch_witness'
            if issue.get('kind') not in (
                'missing_constraint','scope','direction','join','output'
            ):
                return 'invalid_mismatch_kind'
            steps=issue.get('program_steps')
            if not (isinstance(steps,list) and 1<=len(steps)<=6
                    and all(type(n) is int and n in nodes for n in steps)):
                return 'invalid_mismatch_steps'
            quote=issue.get('question_quote')
            if not (isinstance(quote,str) and 0<len(quote)<=240
                    and quote in question):
                return 'invalid_mismatch_quote'
            if not all(isinstance(issue.get(k),str)
                       and 0<len(issue[k].strip())<=180
                       for k in ('program_claim','question_claim','reason')):
                return 'invalid_mismatch_explanation'
            # Observed v3 failure: "No discrepancy found" under mismatch.
            # This contradicts the review's own conclusion; request a
            # corrected witness, NEVER infer that the question is eligible.
            if re.search(
                r'\b(?:no|zero)\s+(?:(?:concrete|semantic|mathematical|actual|direction|scope|output)\s+)*'
                r'(?:discrepanc(?:y|ies)|differences?|mismatches?)\s*'
                r'(?:found|detected|exists?)?\b',
                issue['reason'], re.I,
            ):
                return 'contradictory_mismatch'
            if normalized(issue['program_claim'])==normalized(issue['question_claim']):
                return 'contradictory_mismatch'
        return None

    def valid_verdict(value, *, required_hops=()):
        return invalid_reason(value, required_hops=required_hops) is None
    audit=[]
    def call(text):
        from .backend import _LLMRole
        if isinstance(role,_LLMRole):
            result=role.check_question(text,audit=audit)
        else:
            result=(getattr(role,'check_question',None) or getattr(role,'review'))(text)
        audit.append({'stage':'semantic_verdict','verdict':result})
        return result
    try:
        if verdict is None:
            verdict=call(prompt)
            # One full-graph objection re-audit. For a malformed verdict,
            # avoid echoing its long/invalid JSON back into an already long
            # prompt; it can amplify truncation and false discrepancies.
            if not valid_verdict(verdict) or verdict['decision']=='mismatch':
                previous={'decision':verdict.get('decision'),
                          'issues':[
                              {k:item.get(k) for k in (
                                  'kind','program_steps','question_quote','reason'
                              )} for item in verdict.get('issues',[])[:2]
                              if isinstance(item,dict)
                          ]} if isinstance(verdict,dict) else {'decision':'invalid'}
                verdict=call(
                    'Independently re-audit the FULL question and numbered graph. '
                    'The previous verdict is only a hypothesis, not an authority. '
                    'Reject ONLY a concrete mathematical difference supported by '
                    'an exact short quote and existing step IDs; withdraw unsupported '
                    'objections. An invalid witness must not be treated as a match. '
                    'If ambiguity remains, return uncertain. Keep issue reasons <=180 '
                    'characters and relation_coverage=[] unless requested.\\n'
                    'Previous review: '+json.dumps(previous,ensure_ascii=False)+'\\n'+prompt
                )
        if (valid_verdict(verdict) and verdict['decision']=='eligible'
                and relation_gaps
                and not valid_verdict(verdict,required_hops=relation_gaps)):
            coverage=[]
            # Bound each required-hop answer: Qwen's previous long multi-hop
            # coverage output repeatedly exhausted its 2048-token budget.
            # Every gap still needs an exact, distinct question witness.
            for start in range(0,len(relation_gaps),2):
                batch=relation_gaps[start:start+2]
                focused_prompt=(
                    'Focused REQUIRED-HOP/output audit of EXACTLY the listed constraints. '
                    'Compare each direction, input owner, relation or queried attribute against the '
                    'full accepted question. A type name alone does not imply a '
                    'missing hop. Paraphrases are valid only if the quote actually '
                    'expresses that traversal/output query. For eligible, return issues=[] and '
                    'one short, DISTINCT, EXACT quote per required step in '
                    'relation_coverage. For missing/changed hops return mismatch '
                    'with a concrete step and <=240-character exact quote; for '
                    'ambiguity return uncertain. Do not use ellipses to fabricate '
                    'quotes. relation_coverage must be [] for mismatch/uncertain. '
                    +review_rules+'\n'
                    +json.dumps({
                        **review_input,
                        'required_hops':batch,
                    },ensure_ascii=False,separators=(',',':'))
                )
                focused=call(focused_prompt)
                if not valid_verdict(focused,required_hops=batch):
                    # A schema-valid but evidence-invalid response is still
                    # a review failure; retry once with the same strict task.
                    focused=call(
                        'Your last response had an invalid or incomplete '
                        'question/step witness. Retry the SAME audit from '
                        'the original question and numbered graph. Use only '
                        'exact <=240-character substrings, no ellipses or '
                        'long copied passages; do not auto-approve.\\n'
                        +focused_prompt
                    )
                focus_error=invalid_reason(focused,required_hops=batch)
                audit.append({'stage':'focused_hop_validation',
                              'required_steps':[gap['step'] for gap in batch],
                              'validation_error':focus_error})
                if focus_error:
                    verdict=focused
                    break
                if focused['decision']!='eligible':
                    verdict=focused
                    break
                coverage.extend(focused['relation_coverage'])
            else:
                verdict={'decision':'eligible','issues':[],
                         'relation_coverage':coverage}
    except Exception as exc:
        diagnostic=getattr(exc,'diagnostic',None)
        failure_kind=('review_output_truncated' if isinstance(diagnostic,dict)
                      and diagnostic.get('finish_reason')=='length'
                      else 'review_backend_error')
        return {'eligible':False,'reviewed':False,'decision':'review_error',
                'review_failure_kind':failure_kind,
                'method':'semantic_review_unavailable','issues':[str(exc)],
                'review_audit':audit,'backend_diagnostic':diagnostic,**fingerprint}
    valid=valid_verdict(verdict,required_hops=relation_gaps if isinstance(verdict,dict) and verdict.get('decision')=='eligible' else ())
    if valid and cache_path and not cache_hit:
        target=Path(cache_path);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('a',encoding='utf-8') as handle:handle.write(json.dumps({'cache_key':cache_key,'verdict':verdict},ensure_ascii=False)+'\n')
    if valid and verdict['decision']=='eligible' and verdict['issues']==[]:
        return {'eligible':True,'reviewed':True,'decision':'eligible','method':'llm_semantic_screen','cache_hit':cache_hit,'issues':[],'review_audit':audit,**fingerprint}
    issues=[]
    for issue in verdict.get('issues',[]) if isinstance(verdict,dict) else []:
        if isinstance(issue,dict) and issue.get('question_quote') and issue['question_quote'] in question:
            issues.append(issue)
    return {'eligible':False,'reviewed':bool(valid),
            'decision':verdict['decision'] if valid else 'review_error',
            'review_failure_kind':(
                'semantic_mismatch' if valid and verdict['decision']=='mismatch' else
                'semantic_uncertain' if valid and verdict['decision']=='uncertain' else
                invalid_reason(verdict,required_hops=relation_gaps if isinstance(verdict,dict) and verdict.get('decision')=='eligible' else ()) or 'invalid_review_schema'
            ),
            'method':'llm_semantic_screen','cache_hit':cache_hit,
            'issues':issues or ['equivalence_not_established'],
            'raw_verdict':verdict,'review_audit':audit,**fingerprint}
