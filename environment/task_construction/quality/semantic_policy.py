"""Semantic plausibility policy for synthetic worlds."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date
from typing import Any,Mapping
from ..ir import ProgramIR,ProgramNode
from ..world import Fact,SyntheticWorld
from ..utils.determinism import stable_int as _stable_int
DEFAULT_WORLD_REFERENCE_DATE=date(2026,10,16)
_PLACEHOLDER_RE=re.compile(r"^(?:value|val|string|text|label|literal|token|item|name)[_-]?(?P<ordinal>\d+)(?:[_-]?(?:alt|variant|distractor)\d*)?$",re.IGNORECASE);_DATE_RE=re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$")
def _predicate_key(predicate):
    if not predicate:return ""
    text=str(predicate).split(":",1)[-1];text=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",text);return re.sub(r"[_-]+"," ",text).strip().lower()
def predicate_family(predicate):
    key=_predicate_key(predicate)
    if "birth" in key:return "birth"
    if "death" in key:return "death"
    if any(token in key for token in ("start date","start year","inception","founded","founding","established","formation")):return "start"
    if any(token in key for token in ("end date","end year","expiry","expiration","dissolution")):return "end"
    if "sport number" in key or "jersey number" in key or "shirt number" in key:return "sport_number"
    if "address" in key:return "address"
    if any(token in key for token in ("geo","region","location")):return "region"
    if "slogan" in key or "motto" in key:return "slogan"
    if any(token in key for token in ("code","identifier","icao","lei")):return "code"
    if "date" in key or "year" in key:return "temporal"
    return "generic"
def _namespace_placeholder(raw,predicate):
    match=_PLACEHOLDER_RE.match(raw.strip())
    if not match:return raw
    local=int(match.group("ordinal"));namespace=1000+(_stable_int(predicate)%8000)*100;return f"value_{namespace+local}"
def _synthetic_temporal_year(year,family,reference_year):
    if family=="end" or year<=reference_year:return year
    offset=year-2030;base={"birth":1980,"death":2010,"start":2000,"temporal":2015}.get(family,2015);return min(base+offset,reference_year)
def _normalize_literal(value,*,predicate,object_kind,reference_date):
    family=predicate_family(predicate)
    if object_kind=="year":
        try:year=int(value)
        except (TypeError,ValueError):return value
        return _synthetic_temporal_year(year,family,reference_date.year)
    if object_kind=="date":
        match=_DATE_RE.match(str(value))
        if not match:return value
        year=_synthetic_temporal_year(int(match.group("year")),family,reference_date.year);month=int(match.group("month"));day=int(match.group("day"))
        try:normalized=date(year,month,day)
        except ValueError:return value
        if family not in {"end"} and normalized>reference_date:normalized=date(reference_date.year,1,1)
        return normalized.isoformat()
    if object_kind=="string":
        raw=str(value)
        if family=="sport_number":
            try:number=int(raw)
            except ValueError:number=None
            if number is not None and 1<=number<=99:return str(number)
            return str(1+(_stable_int(predicate,raw)%98))
        return _namespace_placeholder(raw,predicate)
    return value
def _attribute_predicate_by_node(program):
    result={};nodes=program.node_map()
    for node_id in program.topological_order():
        node=nodes[node_id]
        if node.op=="QUERY_ATTR" and node.inputs:result[node_id]=str(node.inputs[0])
        elif len(node.dependencies)==1:result[node_id]=result.get(node.dependencies[0])
        else:result[node_id]=None
    return result
def normalize_program_world(program:ProgramIR,world:SyntheticWorld,*,reference_date:date=DEFAULT_WORLD_REFERENCE_DATE):
    normalized_world=world.copy();normalized_world.facts=[Fact(fact.fact_id,fact.subject,fact.predicate,fact.object if fact.object_kind=="entity" else _normalize_literal(fact.object,predicate=fact.predicate,object_kind=fact.object_kind,reference_date=reference_date),fact.object_kind,fact.source_id) for fact in world.facts];scalar_predicates=_attribute_predicate_by_node(program);nodes=[]
    for node in program.nodes:
        inputs=list(node.inputs)
        if node.op.startswith("FILTER_") and not node.op.startswith("FILTER_QUALIFIER") and len(inputs)>=3:
            kind={"FILTER_STRING":"string","FILTER_NUMBER":"number","FILTER_YEAR":"year","FILTER_DATE":"date"}.get(node.op)
            if kind is not None:inputs[2]=str(_normalize_literal(inputs[2],predicate=str(inputs[0]),object_kind=kind,reference_date=reference_date))
        elif node.op.startswith("VERIFY_") and len(inputs)>=2 and node.dependencies:
            predicate=scalar_predicates.get(node.dependencies[0]);kind={"VERIFY_STRING":"string","VERIFY_NUM":"number","VERIFY_NUMBER":"number","VERIFY_YEAR":"year","VERIFY_DATE":"date"}.get(node.op)
            if predicate and kind:inputs[1]=str(_normalize_literal(inputs[1],predicate=predicate,object_kind=kind,reference_date=reference_date))
        nodes.append(ProgramNode(node.node_id,node.op,node.dependencies,tuple(inputs),node.raw_function))
    metadata=dict(program.metadata);metadata["semantic_world_normalized"]=True;metadata["world_reference_date"]=reference_date.isoformat();normalized_program=ProgramIR(nodes,metadata=metadata);normalized_world.metadata["semantic_world_policy"]={"version":1,"reference_date":reference_date.isoformat(),"normalized":True};return normalized_program,normalized_world
def _as_temporal(value,kind):
    try:
        if kind=="year":return date(int(value),1,1)
        if kind=="date":return date.fromisoformat(str(value)[:10])
    except (TypeError,ValueError):return None
    return None
@dataclass(frozen=True)
class SemanticWorldReport:
    accepted:bool;reasons:tuple[str,...];details:Mapping[str,Any]
    def to_dict(self):return {"accepted":self.accepted,"reasons":list(self.reasons),"details":dict(self.details)}
@dataclass(frozen=True)
class SemanticWorldPolicy:
    reference_date:date=DEFAULT_WORLD_REFERENCE_DATE
    def assess(self,world:SyntheticWorld)->SemanticWorldReport:
        reasons=[];temporal_checked=0;literal_families={};by_subject={}
        for fact in world.facts:
            if fact.object_kind=="entity":continue
            family=predicate_family(fact.predicate);raw=str(fact.object);literal_families.setdefault(raw,set()).add(family);by_subject.setdefault(fact.subject,{}).setdefault(family,[]).append((fact.object_kind,raw));temporal=_as_temporal(fact.object,fact.object_kind)
            if temporal is not None:
                temporal_checked+=1
                if family in {"birth","death","start","temporal"} and temporal>self.reference_date:reasons.append(f"future_{family}:{fact.fact_id}")
            if family=="sport_number":
                try:number=int(raw)
                except ValueError:reasons.append(f"sport_number_not_integer:{fact.fact_id}")
                else:
                    if not 1<=number<=99:reasons.append(f"sport_number_out_of_range:{fact.fact_id}")
        for raw,families in literal_families.items():
            meaningful={family for family in families if family!="generic"}
            if len(meaningful)>1 and not raw.replace(".","",1).isdigit():reasons.append("cross_domain_literal_collision:"+raw+":"+",".join(sorted(meaningful)))
        for subject,families in by_subject.items():
            starts=[v for kind,value in families.get("start",[]) if (v:=_as_temporal(value,kind)) is not None];ends=[v for kind,value in families.get("end",[]) if (v:=_as_temporal(value,kind)) is not None];births=[v for kind,value in families.get("birth",[]) if (v:=_as_temporal(value,kind)) is not None];deaths=[v for kind,value in families.get("death",[]) if (v:=_as_temporal(value,kind)) is not None]
            if starts and ends and min(ends)<min(starts):reasons.append(f"end_before_start:{subject}")
            if births and deaths and min(deaths)<min(births):reasons.append(f"death_before_birth:{subject}")
        unique=tuple(dict.fromkeys(reasons));return SemanticWorldReport(not unique,unique,{"reference_date":self.reference_date.isoformat(),"temporal_facts_checked":temporal_checked,"literal_value_count":len(literal_families)})
