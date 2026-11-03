"""Predicate-level semantic profiles for synthetic attributes."""
from __future__ import annotations
import math,re
from collections import Counter
from dataclasses import asdict,dataclass
from typing import Any,Mapping,Sequence
from ..literals import parse_yago_literal
@dataclass(frozen=True)
class AttributeSemanticProfile:
    predicate:str;entity_type:str;value_kind:str;family:str;canonical_unit:str|None;minimum:float|None;maximum:float|None;integer_values:bool;confidence:float;safe:bool;evidence:str
    def to_dict(self)->dict[str,Any]:return asdict(self)
_FAMILY_KEYWORDS=(("percentage",("percentage","percent","unemployment rate","employment rate","literacy rate","birth rate","death rate","fertility rate","mortality rate","survival rate","growth rate","share","ratio")),("age",("age","minimum age","min age","maximum age","max age")),("height",("height","stature")),("mass",("mass","weight")),("duration",("duration","runtime","running time","incubation period","term length")),("population",("population",)),("count",("count","number of","attendance","capacity","circulation")),("money",("income","revenue","budget","price","cost","gdp","salary","worth")),("rating",("rating","score")),("index",("index",)),("temperature",("temperature",)),("distance",("distance","length","width","depth","elevation","altitude")),("speed",("speed","velocity")),("identifier",("identifier","code","isbn","issn","gtin","duns","lei","icao","iata","postal","telephone","calling code","registration number","catalog number")),("address",("address",)),("slogan",("slogan","motto")),("region",("region","geo","location","area")),("emblem",("logo","emblem","symbol","crest")),("color",("color","colour")))
_DEFAULT_NUMERIC_RANGES={"percentage":(1.,100.,False,"%"),"age":(18.,85.,True,"years"),"height":(150.,205.,False,"cm"),"mass":(45.,120.,False,"kg"),"duration":(15.,240.,False,"min"),"population":(10000.,5000000.,True,None),"count":(1.,50000.,True,None),"money":(1000.,500000.,False,None),"rating":(1.,10.,False,None),"index":(0.,100.,False,None),"temperature":(-10.,40.,False,"°C"),"distance":(1.,1000.,False,"km"),"speed":(10.,300.,False,"km/h")};_EMPIRICAL_RANGE_FAMILIES={"percentage","age","population","count","rating","index"};_EMPIRICAL_HARD_BOUNDS={"percentage":(0.,100.),"age":(0.,130.),"population":(0.,None),"count":(0.,None)};_UNIT_FAMILIES={"percent":("percentage","%"),"percentage":("percentage","%"),"kilogram":("mass","kg"),"gram":("mass","g"),"centimetre":("height","cm"),"centimeter":("height","cm"),"metre":("distance","m"),"meter":("distance","m"),"kilometre":("distance","km"),"kilometer":("distance","km"),"minute":("duration","min"),"second":("duration","s"),"hour":("duration","h"),"celsius":("temperature","°C")}
def _predicate_key(predicate:str)->str:
    text=str(predicate).split(":",1)[-1];text=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",text);return re.sub(r"[_-]+"," ",text).strip().lower()
def _phrase_matches(key:str,phrase:str)->bool:
    normalized=" ".join(str(phrase).strip().lower().split())
    if not normalized:return False
    pattern=r"(?<!\w)"+r"\s+".join(re.escape(token) for token in normalized.split())+r"(?!\w)";return re.search(pattern,key) is not None
def _lexical_family(predicate:str)->str|None:
    key=_predicate_key(predicate)
    for family,phrases in _FAMILY_KEYWORDS:
        if any(_phrase_matches(key,p) for p in phrases):return family
    return None
def _sample_literals(info:Mapping[str,Any])->list[Any]:
    result=[];raw=info.get("samples",())
    if not isinstance(raw,Sequence) or isinstance(raw,(str,bytes)):return result
    for sample in raw:
        if isinstance(sample,Mapping):
            parsed=parse_yago_literal(str(sample.get("object","")))
            if parsed is not None:result.append(parsed)
    return result
def _sample_unit_family(samples):
    units=Counter(str(i.unit).split(":",1)[-1].lower() for i in samples if getattr(i,"unit",None))
    if not units:return None,None
    name,_=units.most_common(1)[0];return _UNIT_FAMILIES.get(name,(None,None))
def _numeric_sample_range(samples,kind):
    values=[]
    for item in samples:
        if getattr(item,"kind",None)!=kind:continue
        value=getattr(item,"value",None)
        if isinstance(value,bool) or not isinstance(value,(int,float)):continue
        number=float(value)
        if math.isfinite(number):values.append(number)
    values.sort()
    if not values:return None,None,0
    lo=values[max(0,int(len(values)*.10)-1)] if len(values)>=6 else values[0];hi=values[min(len(values)-1,int(len(values)*.90))] if len(values)>=6 else values[-1]
    if lo==hi:span=max(1.,abs(lo)*.10);lo-=span;hi+=span
    return lo,hi,len(values)
def _bounded_empirical_range(family,lo,hi):
    if lo is None or hi is None:return lo,hi
    hard_lo,hard_hi=_EMPIRICAL_HARD_BOUNDS.get(family,(None,None))
    if hard_lo is not None:lo=max(float(hard_lo),lo);hi=max(float(hard_lo),hi)
    if hard_hi is not None:lo=min(float(hard_hi),lo);hi=min(float(hard_hi),hi)
    return (None,None) if hi<=lo else (lo,hi)
def _domain_support(info,entity_type):
    schema=info.get("schema",{});domains=[str(v) for v in (schema.get("domain",()) if isinstance(schema,Mapping) else ()) if not str(v).startswith(("xsd:","rdf:"))]
    if domains:return 1.,"schema-domain"
    empirical=info.get("empirical_subject_types",())
    if not isinstance(empirical,Sequence) or isinstance(empirical,(str,bytes)):return 0.,"no-domain-evidence"
    counts={}
    for row in empirical:
        if isinstance(row,Sequence) and not isinstance(row,(str,bytes)) and len(row)>=2:
            try:counts[str(row[0])]=int(row[1])
            except (TypeError,ValueError):pass
    support=counts.get(str(entity_type),0);total=sum(counts.values())
    if support<=0:return 0.,"empirical-domain-missing"
    share=support/total if total else 0.
    if support<3 or share<.10:return 0.,f"empirical-domain-weak:{support}/{total}"
    return min(.85,.45+.05*min(support,5)+.30*share),f"empirical-domain:{support}/{total}"
def compile_attribute_profile(predicate:str,info:Mapping[str,Any],*,entity_type:str,value_kind:str)->AttributeSemanticProfile:
    samples=_sample_literals(info);lexical_family=_lexical_family(predicate);unit_family,unit_surface=_sample_unit_family(samples);family=lexical_family or unit_family
    if value_kind=='number' and info.get('typed_literal_support',{}).get('version')==1:
        from .typed_profile import compile_typed_numeric_profile
        fields=compile_typed_numeric_profile(predicate,info,entity_type,family=family or 'generic_numeric');return AttributeSemanticProfile(predicate=str(predicate),entity_type=str(entity_type),value_kind=value_kind,**fields)
    domain_confidence,domain_evidence=_domain_support(info,entity_type);contradictory_unit=bool(lexical_family and unit_family and lexical_family!=unit_family and {lexical_family,unit_family} not in ({"height","distance"},));minimum=maximum=None;integer_values=False;canonical_unit=None;semantic_evidence=""
    if value_kind=="number":
        if family in _DEFAULT_NUMERIC_RANGES:
            minimum,maximum,integer_values,canonical_unit=_DEFAULT_NUMERIC_RANGES[family];semantic_evidence=f"recognized-family:{family}"
            if family in _EMPIRICAL_RANGE_FAMILIES:
                lo,hi,n=_numeric_sample_range(samples,"number")
                if n>=3:
                    lo,hi=_bounded_empirical_range(family,lo,hi)
                    if lo is not None and hi is not None:minimum,maximum=lo,hi;semantic_evidence+=f";empirical-range:{n}"
        else:
            lo,hi,n=_numeric_sample_range(samples,"number")
            if n>=3:
                minimum,maximum=lo,hi;integer_values=all(isinstance(getattr(i,"value",None),int) and not isinstance(getattr(i,"value",None),bool) for i in samples if getattr(i,"kind",None)=="number");canonical_unit=unit_surface;family=unit_family or "generic_numeric";semantic_evidence=f"empirical-numeric:{n}"
            else:family=family or "generic_numeric";semantic_evidence=f"insufficient-numeric-samples:{n}"
    elif value_kind=="year":
        lo,hi,n=_numeric_sample_range(samples,"year");minimum=lo if n>=2 else 1980.;maximum=hi if n>=2 else 2024.;integer_values=True;family="temporal";semantic_evidence=f"year:{n}"
    elif value_kind=="date":family="temporal";semantic_evidence="typed-date"
    elif value_kind=="bool":family="boolean";semantic_evidence="typed-boolean"
    elif value_kind=="string":
        n=sum(getattr(i,"kind",None)=="string" for i in samples);family=family or "generic_string";semantic_evidence=f"recognized-family:{family}" if lexical_family else f"empirical-string:{n}"
    else:family=family or "generic";semantic_evidence="unsupported-kind"
    if canonical_unit is None and unit_surface and family==unit_family:canonical_unit=unit_surface
    if value_kind=="number" and any(_phrase_matches(_predicate_key(predicate),name) for name in ("elevation","altitude")):
        unit_counts=Counter(str(s.unit) for s in samples if getattr(s,"unit",None))
        if unit_counts and unit_family=="distance" and unit_surface:
            dominant=unit_counts.most_common(1)[0][0];same=[s for s in samples if str(getattr(s,"unit",None))==dominant];lo,hi,n=_numeric_sample_range(same,"number")
            if n>=3:canonical_unit,minimum,maximum=unit_surface,lo,hi;semantic_evidence=f"observed-elevation-unit-range:{dominant}:{n}"
    if value_kind=="number":
        literal_support=family in _DEFAULT_NUMERIC_RANGES or semantic_evidence.startswith("empirical-numeric:") and not semantic_evidence.endswith(":0")
        if family=="generic_numeric":_,_,n=_numeric_sample_range(samples,"number");literal_support=n>=3
    elif value_kind=="string":literal_support=lexical_family is not None or sum(getattr(i,"kind",None)=="string" for i in samples)>=3
    else:literal_support=value_kind in {"year","date","bool"}
    safe=domain_confidence>=.45 and literal_support and not contradictory_unit;confidence=min(1.,domain_confidence+(.12 if lexical_family else 0)+(.08 if unit_family else 0));confidence=0. if contradictory_unit else confidence;evidence=f"{domain_evidence};{semantic_evidence}"+(f";unit-conflict:{lexical_family}!={unit_family}" if contradictory_unit else "")
    return AttributeSemanticProfile(str(predicate),str(entity_type),str(value_kind),str(family),canonical_unit,minimum,maximum,integer_values,round(confidence,4),bool(safe),evidence)
