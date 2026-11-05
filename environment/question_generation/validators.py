"""Deterministic pre/postflight checks for LLM-authored questions."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any,Mapping,Sequence
@dataclass(frozen=True)
class QuestionValidation:
    accepted:bool;issues:tuple[str,...]
    def to_dict(self):return {"accepted":self.accepted,"issues":list(self.issues)}
_INTERNAL_RE=re.compile(r"\b(?:source_\d+|anchor_\d+|fact_?id|proof[_ -]?relevant|indispensable|capability_\w+|p1(?:\.5)?|taskdesign|orchestration[_ -]?ready)\b",re.I)
_PLACEHOLDER_RE=re.compile(r"\b(?:e\d{4}|f\d{5}|df\d{5}|lf\d{5}|value[_-]?\d+)\b",re.I)
def _content_words(text):return {token for token in re.findall(r"[A-Za-z]{4,}",str(text).lower()) if token not in {"what","which","that","with","from","this","these","those","does","have","into","among","following","recorded"}}
def preflight(payload:Mapping[str,Any],question:str)->QuestionValidation:
    issues=[];text=" ".join(str(question or "").split())
    if not text:issues.append("empty_question")
    if len(text)<15:issues.append("question_too_short")
    if len(text)>1200:issues.append("question_too_long")
    if _INTERNAL_RE.search(text):issues.append("internal_metadata_leak")
    if _PLACEHOLDER_RE.search(text) or re.search(r"<(?:SYMBOL|YEAR|NUMBER|DATE|ATTR|REL|ENTITY_OR_TYPE|STRING|BOOL|VALUE)[^>]*>",text,re.I):issues.append("synthetic_identifier_leak")
    if not (text.endswith("?") or text.endswith(".")):issues.append("missing_terminal_punctuation")
    contract=payload.get("task_design",{}) if isinstance(payload.get("task_design"),Mapping) else {};operation=str(contract.get("final_operation") or "").upper();lower=text.lower()
    if str(contract.get("entry_strategy") or "")=="web_search" and re.search(
        r"\b(?:supplied|provided|attached|uploaded|given)\s+"
        r"(?:(?:reference|source|input|document|spreadsheet|file|record)s?\b)"
        r"|\b(?:in|from)\s+the\s+(?:supplied|provided|attached)\s+"
        r"(?:file|document|records|spreadsheet|video|image)\b",
        lower,
    ):
        issues.append("web_search_not_presupplied_file")
    if operation=="COUNT" and not any(token in lower for token in ("how many","count","number of")):issues.append("count_intent_missing")
    if operation.startswith("VERIFY_") and not any(token in lower for token in ("whether","any ","does ","is ","are ","can ")):issues.append("verify_existential_intent_missing")
    if operation in {"SELECT_BETWEEN","SELECT_AMONG"} and not any(token in lower for token in ("largest","smallest","highest","lowest","maximum","minimum","most","least")):issues.append("selection_direction_missing")
    draft=str(payload.get("draft_question") or "")
    resources=payload.get("resources",())
    has_discoverable=(
        isinstance(resources,Sequence) and not isinstance(resources,(str,bytes))
        and any(
            isinstance(resource,Mapping)
            and resource.get("visibility")=="discoverable"
            for resource in resources
        )
    )
    # For the search/discovery roots, a closed-set list is a result of
    # inspecting task-scoped references, not an initial Main input. Reject
    # misleading claims about a *complete supplied list* only when the
    # task actually contains discoverable resources.
    discovery_root=any(phrase in draft.casefold() for phrase in (
        "the available entities", "the entities in the records",
        "all candidate entities",
        "entities identified from the relevant task evidence",
        "entities documented across the accessible task references",
        "candidate entities discoverable from the task-related sources",
    ))
    if discovery_root and has_discoverable and re.search(
        r"\b(?:all|the complete|a complete|the full|a full)\s+"
        r"(?:candidate\s+)?(?:list|set|records|entities)\s+"
        r"(?:was|were|is|are)?\s*(?:supplied|provided|given|available initially)\b"
        r"|\b(?:all|entire)\s+(?:supplied|provided|given)\s+records\b",
        text, re.I,
    ):
        issues.append("discovery_candidates_not_pre_supplied")
    if draft:
        draft_words=_content_words(draft);question_words=_content_words(text);overlap=len(draft_words&question_words)/len(draft_words) if draft_words else 1.
        if overlap<.12:issues.append("surface_drift_extreme")
    from .contracts import contract_issues
    issues.extend(contract_issues(payload,text))
    return QuestionValidation(not issues,tuple(dict.fromkeys(issues)))
def postflight(payload:Mapping[str,Any],question:str,*,oracle_answer:Any=None)->QuestionValidation:
    issues=list(preflight(payload,question).issues);lower=str(question).casefold()
    if oracle_answer is not None:
        values=[]
        def visit(value):
            if isinstance(value,Mapping):
                for item in value.values():visit(item)
            elif isinstance(value,(list,tuple,set)):
                for item in value:visit(item)
            elif value is not None:values.append(str(value))
        visit(oracle_answer)
        for value in values:
            surface=" ".join(value.split()).casefold()
            if len(surface)>=4 and surface in lower:issues.append("oracle_answer_leak");break
    resources=payload.get("resources",())
    if isinstance(resources,Sequence) and not isinstance(resources,(str,bytes)):
        for resource in resources:
            if isinstance(resource,Mapping) and str(resource.get("resource_id") or "").casefold() in lower and resource.get("resource_id"):issues.append("resource_id_leak");break
    return QuestionValidation(not issues,tuple(dict.fromkeys(issues)))
__all__=["QuestionValidation","postflight","preflight"]
