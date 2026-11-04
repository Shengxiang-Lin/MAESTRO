"""Deterministic summary guards plus an evidence-scoped LLM review prompt."""
from __future__ import annotations
import json
import re
from decimal import Decimal, InvalidOperation


def grouped_evidence(tasks,results):
    """Keep descriptor state, source titles and fact provenance together."""
    groups=[]
    inspected=set()
    discovered=set()
    for result in results:
        task=tasks[result["subtask_id"]]
        kind=task.get("kind","inspect")
        sid=task["resource_id"]
        observation=task.get("observation",{})
        (inspected if kind=="inspect" else discovered).add(sid)
        groups.append({
            "kind":kind,"subtask_id":task["subtask_id"],"resource_id":sid,
            "title":observation.get("title",""),"source_type":task.get("source_type",""),
            "claims":result.get("answer",[]),
        })
    return groups,{
        "inspected_resource_ids":sorted(inspected),
        "discovered_unread_resource_ids":sorted(discovered-inspected),
    }


def _numeric_tokens(text):
    values=set()
    for token in re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)*(?!\d|[.,]\d)',text):
        if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',token):
            token=token.replace(',','')
        try:values.add(Decimal(token))
        except InvalidOperation:values.add(token)
    return values


def _bound_review_evidence(claims):
    """Render each already-validated fact with explicit ownership and aliases."""
    observations=[]
    aliases={}
    for group in claims:
        if isinstance(group,dict) and isinstance(group.get("claims"),list):
            sid=group["resource_id"]
            title=group.get("title","")
            if title:aliases[title]=sid
            owner=f"{sid} (same resource title alias: {title})"
            if group.get("kind")=="discover":
                observations.append(f"DISCOVERED DESCRIPTOR ONLY for {owner}: "+json.dumps(group["claims"],ensure_ascii=False))
            else:
                for claim in group["claims"]:
                    observations.append(f"AUTHORITATIVE OBSERVED FACT from INSPECTED {owner}: "+json.dumps(claim,ensure_ascii=False))
        else:
            observations.append(json.dumps(group,ensure_ascii=False))
    return observations,aliases


def text_issues(text,claims=None,*,pending_ids=None,allowed_ids=None,question="",max_length=3000):
    if not isinstance(text,str) or not text.strip():return ['empty_or_non_string_summary']
    issues=[]
    if len(text)>max_length:issues.append('summary_not_compressed')
    for match in re.finditer(
        r'\b(?:read|observed|inspected|accessed)\b.{0,45}\b(?:unavailable|unseen|future)\b',text,re.I
    ):
        prefix=text[max(0,match.start()-30):match.start()]
        if not re.search(r"(?:\bnot|\bnever|\bdidn't|\bhaven't|\bhasn't)\s+(?:(?:been|yet|actually)\s+)*$",prefix,re.I):
            issues.append('claims_unavailable_observation')
            break
    if pending_ids and re.search(r'\b(?:all|every)\s+(?:required\s+)?(?:tasks?|subtasks?|work)\s+(?:(?:is|are|have been|has been)\s+)?(?:done|complete|completed|finished)\b',text,re.I):
        issues.append('premature_completion_in_summary')
    if claims is not None:
        evidence=json.dumps({"claims":claims,"public_question_constraints":question,"observed_resource_count":len(allowed_ids or ()),"pending_task_count":len(pending_ids or ())},ensure_ascii=False)
        # Numeric task/resource identifiers are not factual quantities.
        clean=re.sub(r'(?:source_|subtask:source_|discover:source_)\d+','',text)
        known_references={int(v) for v in re.findall(r'(?:source_|(?:Source|Reference)\s+)(\d+)',evidence,re.I)}
        known_references.update(int(v) for v in re.findall(r'source_(\d+)',json.dumps(list(allowed_ids or ()))))
        known_references.update(int(v) for v in re.findall(r'source_(\d+)',json.dumps(list(pending_ids or ()))))
        def resource_list(match):
            ids={int(v) for v in re.findall(r'\d+',match.group(1))}
            if ids.issubset(known_references):return 'known resources'
            issues.append('unknown_resource_reference')
            return match.group(0)
        clean=re.sub(r'\b(?:sources?|references?)\s+(\d{1,3}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)\d{1,3})*)\b',resource_list,clean,flags=re.I)
        supported=_numeric_tokens(evidence)
        if _numeric_tokens(clean)-supported:issues.append('unsupported_numeric_claim_in_summary')
    if allowed_ids is not None:
        mentioned=set(re.findall(r'source_\d+',text))
        pending_sources=set(re.findall(r'source_\d+',json.dumps(list(pending_ids or ()))))
        if mentioned-set(allowed_ids)-pending_sources:issues.append('unknown_source_in_summary')
    return sorted(set(issues))


def review_prompt(text,claims,*,question='',pending_ids=(),allowed_ids=(),context=None):
    constraint_values=sorted(str(value) for value in _numeric_tokens(question))
    observations,aliases=_bound_review_evidence(claims)
    pending_descriptions=[
        f"{sid} = {'DISCOVERY' if sid.startswith('discover:') else 'INSPECTION'} not yet executed"
        for sid in pending_ids
    ]
    if (context or {}).get("kind")=="main_final_summary":
        return (
            'Check ONLY factual statements in the final summary against authoritative_observations. '
            'The final answer has ALREADY been verified by deterministic execution. You are NOT '
            'an answer judge: do not re-solve the question or demand missing reasoning/conditions. '
            'Reject invented or contradicted subject/predicate/object values, wrong source '
            'attribution, or claims to read uninspected resources. Do not reject omissions: '
            'a compressed summary need not repeat every fact. Numeric comparisons may mention '
            'public_constraint_values (task thresholds), which are AUTHORITATIVE task inputs, '
            'not invented observations. Comparing an observed value against one is supported. '
            'Each observed fact has its resource owner written explicitly. resource_title_aliases '
            'is the ONLY authority for title-to-resource identity. A fact line title is an alias '
            'of THAT line resource, not a second resource. Never invent or infer an alias mapping '
            'from list order or title numbering. '
            'Use inspected_resource_ids for latest state; prior discovery does not undo inspection. '
            'Accept the verified_final_answer conclusion, while checking its stated supporting facts. '
            'Facts about the same entity may be JOINED across sources. Reject reasoning that '
            'dismisses a candidate solely because no single source contains all its attributes. '
            'Return JSON only: {"accept":true|false,"issues":[{"quote":"exact offending text from '
            'THIS summary","reason":"specific factual contradiction with supplied evidence"}]}. '
            'At most three issues.\n'+json.dumps({
                'execution_context':context,'summary':text,'authoritative_observations':observations,
                'resource_title_aliases':aliases,
                'public_constraint_values':constraint_values,
                'known_resource_ids':list(allowed_ids)},
                ensure_ascii=False,separators=(',',':')))
    return (
        'Check ONLY factual statements and execution state in this compressed summary. '
        'All authoritative_observations are already validated TRUE observed facts, '
        'including contextual/distractor facts irrelevant to the user query. NEVER filter '
        'these facts by a task objective, challenge their provenance, or demand external evidence. '
        'Reject contradicted or invented facts, wrong source attribution, reading uninspected '
        'resources, and claiming pending subtasks completed. Partial work is expected; do not '
        'require an original-question answer or repetition of every fact. Search/filter objectives '
        'may use public_constraint_values without asserting they were observed. '
        'Each fact explicitly states its resource owner. resource_title_aliases is the ONLY '
        'authority mapping a title to the SAME resource as its fact line, not a second resource. '
        'A descriptor establishes '
        'discovery ONLY. inspected_resource_ids (or the current inspect task) establish read '
        'contents. Past discovery does not undo later inspection. Source 01 means source_01. '
        'Never infer ownership from list order or title numbering. '
        'Pending IDs discover:source_* mean DISCOVERY, subtask:source_* mean INSPECTION. '
        'A source can have BOTH pending discovery and inspection; summarizing them as '
        '"discover and inspect" or "pending inspection" is correct. An empty discovered_unread '
        'list does not mean no pending discovery tasks: those resources are not discovered yet. '
        'Facts about the same public entity can be combined across sources. Scoped observations '
        'are complete, so an attribute may be described as absent from that source, not globally. '
        'Return JSON ONLY: {"accept":true|false,"issues":[{"quote":"exact offending text from '
        'THIS summary","reason":"specific evidence-based error"}]}. Quote actual text and '
        'explain each rejection. At most three issues; no repair or analysis.\n'+json.dumps({
            'execution_context':context or {},'summary':text,'authoritative_observations':observations,
            'resource_title_aliases':aliases,'pending_task_descriptions':pending_descriptions,
            'public_constraint_values':constraint_values,
            'pending_task_ids':list(pending_ids),'known_resource_ids':list(allowed_ids)},
            ensure_ascii=False,separators=(',',':')))


def semantic_review(role,text,claims,*,question='',pending_ids=(),allowed_ids=(),context=None):
    issues=text_issues(text,claims,pending_ids=pending_ids,allowed_ids=allowed_ids,question=question)
    if issues:return False,issues
    # The frozen programs join entity facts, never require document co-location.
    if (context or {}).get("kind")=="main_final_summary" and re.search(
        r'\bno single (?:source|document|report)\b[^.!?]*\b(?:so|therefore|thus|hence)\b'
        r'[^.!?]*\b(?:false|not met|not satisfied|cannot|no match)\b',text,re.I
    ):
        return False,["invalid_single_source_requirement_in_final_reasoning"]
    prompt=review_prompt(text,claims,question=question,
                         pending_ids=pending_ids,allowed_ids=allowed_ids,context=context)
    for _ in range(2):
        review=role.review(prompt)
        review_issues=review.get('issues',[])
        if review.get('accept') is True and review_issues==[]:return True,[]
        errors=[]
        for issue in (review_issues[:3] if isinstance(review_issues,list) else []):
            if isinstance(issue,dict):
                quote=issue.get('quote')
                reason=issue.get('reason')
                if isinstance(quote,str) and quote.strip() and quote in text and isinstance(reason,str) and reason.strip():
                    errors.append('summary_semantic_review:'+json.dumps(issue,ensure_ascii=False))
            elif getattr(role,'offline',False):
                errors.append('summary_semantic_review:'+str(issue))
        if errors:return False,errors
        prompt+='\nYour review was malformed: provide an exact quote from THIS summary and a specific reason for each rejection, or accept if no supported error exists.'
    return False,['summary_review_invalid_rejection']
