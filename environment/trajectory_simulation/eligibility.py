"""Stage-3-only eligibility; frozen questions/programs are never repaired here."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

def digest(value):
    text=value if isinstance(value,str) else json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()

def check_question(gold,record,role,*,review=True):
    question=gold['question'];fingerprint={'question_sha256':digest(question),'program_sha256':digest(record['program'])}
    audit_path = Path(__file__).with_name('question_reviews.json')
    # This manually curated file is optional and not committed in all setups.
    # Missing file is NOT an approval: a real LLM semantic check is still
    # required before training_eligible can become true.
    audited = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else []
    if not isinstance(audited, list):
        raise ValueError("question_reviews.json must be a JSON array")
    for item in audited:
        if item['task_id']==gold['task_id'] and all(item[k]==v for k,v in fingerprint.items()):
            return {'eligible':item.get('eligible',False),'reviewed':True,'method':'audited_exact_input_review','issues':[] if item.get('eligible') else [item['reason']],'audit_reason':item['reason'],**fingerprint}
    if not review or getattr(role,'offline',False) or not getattr(role,'backend',None):
        return {'eligible':True,'reviewed':False,'method':'offline_structure_only','issues':[],**fingerprint}
    # A certified Stage-3 public projection may omit an internal FIND and
    # its redundant INTERSECT. Their original ProgramIR dependencies remain
    # referenced by the final operation. Semantic review must use the *full*
    # frozen graph, not infer the question's semantics from the pruned O1
    # delegation graph (which would KeyError on an omitted node).
    original_nodes={
        int(node['node_id']):node
        for node in record['program']['nodes']
    }
    public_nodes={
        int(op['node_id']):op
        for f in gold['fragments'] for op in f['operations']
    }
    nodes={
        nid:public_nodes.get(nid,raw)
        for nid,raw in original_nodes.items()
    }
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
    visited=set()
    def expression(nid):
        if nid in visited:return {'step_reference':nid}
        visited.add(nid)
        node=nodes[nid]
        return {'step':nid,'action':node['op'],'arguments':node['inputs'],
                'inputs':[expression(int(d)) for d in node['dependencies']]}
    expression_root=expression(max(nodes))
    prompt=('Check whether the accepted question describes EXACTLY the supplied canonical computation. '
        'Do not solve it, inspect answers, or apply real-world plausibility. Check the owner of each constraint, '
        'relation direction and all necessary hops, branch inputs, set joins and final operation. '
        'Compare mathematical meaning, not number of steps. Redundant type filters, filter order, '
        'FIND_ALL with no input steps enumerates ALL entities within the '
        'finite task evidence domain, not the entire Internet and not a '
        'complete initial document or pre-supplied list. A question may ask '
        'to discover those entities from relevant task sources; this is '
        'equivalent to the bounded FIND_ALL, not a narrower arbitrary subset. '
        'When the terminal is QUERY_NAME, phrases like "identify the entities" '
        'and "which entities" can mean returning those entities by name. '
        'Do NOT reject solely for not literally saying "names" unless '
        'the requested output is substantively different (count, attribute '
        'values, or a different entity owner). '
        'Read the numbered dependency graph literally: a filter before RELATE constrains '
        'the input entity; a filter after RELATE constrains the related entity. '
        'Repeated identical type filters do not change the set. '
        'and explicitly naming an already implied type are allowed. SELECT_AMONG and '
        'SELECT_BETWEEN return selected entities, not their ranking attribute values; '
        'QUERY_ATTR returns attribute values. Impossible numerical conditions '
        'are valid questions if the program asks that condition. Filtering people by any height > a, '
        'then checking their recorded heights for b, may be phrased as a height for a person whose '
        'height exceeds a that equals b; accept this even when b<a. Attributes may have multiple '
        'recorded values. Do not invent scalar uniqueness. '
        'Simplification is allowed only if the computation is unchanged. A person constraint cannot be moved '
        'to a place or vice versa. FALSE/empty outcomes do not make a question invalid. '
        'If ambiguity prevents establishing equivalence, choose uncertain. '
        'Do not replace a missing relation traversal or a moved filter owner '
        'with an implied semantics merely because the answer types match. '
        'An explicitly certified public projection may remove a redundant private singleton FIND '
        'and its INTERSECT from agent delegation; the accepted question need not expose the '
        'private singleton name when another public branch identifies the same entity. '
        'Do not excuse missing, redirected, or changed constraints beyond that exact redundancy. '
        'Reject only a concrete mathematical discrepancy, never stylistic wording. '
        'Before claiming a missing condition, check the ENTIRE accepted question, including its ending. '
        'Identical canonical wording is not proof of equivalence: verify the actual steps. '
        'Return JSON only: {"decision":"eligible"|"mismatch"|"uncertain","issues":'
        '[{"kind":"missing_constraint"|"scope"|"direction"|"join"|"output",'
        '"program_steps":[1],"question_quote":"exact span from accepted question",'
        '"program_claim":"what these steps compute",'
        '"question_claim":"what the quoted wording means or omits",'
        '"reason":"specific mathematical discrepancy"}]}. '
        'At most two issues; quotes <=240 characters, claims and reasons <=180 characters each. '
        'Do not put analysis or deliberation inside the JSON strings. '
        'For eligible, issues must be empty. '
        'For mismatch, at least one issue must cite existing step IDs and an exact question span. '
        'Always include relation_coverage: [] unless explicitly asked to certify required hops. '
        'Keep all explanation fields short; never echo the entire question as a quote. '
        'No answer values or regenerated questions.\n'+json.dumps({
            'accepted_question':question,'canonical_question':gold['canonical_question'],
            'certified_public_projections':gold.get('public_projections',[]),
            'canonical_expression':expression_root,
            'canonical_steps':[{
                'step':nid,'action':op['op'],'arguments':op['inputs'],
                'input_steps':op['dependencies'],
                'projected_from_public_delegation':nid not in public_nodes,
            } for nid,op in sorted(nodes.items())]},ensure_ascii=False,separators=(',',':')))
    cache_path=getattr(role,'eligibility_cache',None)
    if not isinstance(cache_path,(str,Path)):cache_path=None
    # Test doubles and partially constructed clients can expose Mock objects
    # for unknown attributes. Cache identity must be stable, JSON-native and
    # never depend on a Mock repr or memory address.
    def identity(field):
        value = getattr(role.backend, field, None)
        return value if isinstance(value, str) and value else "unknown"
    cache_key=digest({**fingerprint,'reviewer_model':identity('model'),
        'reviewer_endpoint':identity('base_url'),'review_contract':7})
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
                r'\b(?:no|zero)\s+(?:(?:concrete|semantic|mathematical|actual)\s+)?'
                r'(?:discrepanc(?:y|ies)|differences?|mismatches?)\s*'
                r'(?:found|detected|exists?)?\b',
                issue['reason'], re.I,
            ):
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
                    'Focused REQUIRED-HOP audit of EXACTLY the listed relations. '
                    'Compare each direction, input owner and relation against the '
                    'full accepted question. A type name alone does not imply a '
                    'missing hop. Paraphrases are valid only if the quote actually '
                    'expresses that traversal. For eligible, return issues=[] and '
                    'one short, DISTINCT, EXACT quote per required step in '
                    'relation_coverage. For missing/changed hops return mismatch '
                    'with a concrete step and <=240-character exact quote; for '
                    'ambiguity return uncertain. Do not use ellipses to fabricate '
                    'quotes. relation_coverage must be [] for mismatch/uncertain. '
                    'FIND_ALL is bounded to the task evidence domain and QUERY_NAME '
                    'may mean identifying entities. All explanations <=180 chars.\\n'
                    +json.dumps({
                        'accepted_question':question,
                        'required_hops':batch,
                        'canonical_steps':[{
                            'step':nid,'action':op['op'],'arguments':op['inputs'],
                            'input_steps':op['dependencies'],
                        } for nid,op in sorted(nodes.items())],
                        'certified_public_projections':gold.get('public_projections',[]),
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
                if not valid_verdict(focused,required_hops=batch):
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
