"""Deterministic literal synthesis from bound semantic profiles."""
from __future__ import annotations
import re
from datetime import date
from typing import Any
from ..utils.determinism import stable_int as _stable_int
from .profile import _predicate_key
_GREATER={">","greater","<CMP:>>","<CMP:greater>",">=","<CMP:>=>"};_LESS={"<","less","<CMP:<>","<CMP:less>","<=","<CMP:<=>"};_NOT_EQUAL={"!=","<CMP:!=>"};_ALPHABET="ABCDEFGHIJKLMNOPQRSTUVWXYZ";_ALNUM="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_STREET_NAMES=("Cedar","Maple","Harbor","Willow","Oak","Pine","River","Hill");_STREET_SUFFIXES=("Avenue","Street","Road","Lane","Boulevard");_SLOGAN_PREFIXES=("Forward","Together","Building","Discover","Creating","Serving");_SLOGAN_SUFFIXES=("Tomorrow","Possibility","Communities","Progress","Ideas","Trust");_EMBLEM_ADJECTIVES=("silver","golden","blue","green","crimson","white");_EMBLEM_SYMBOLS=("falcon","oak","star","bridge","mountain","torch");_REGION_PREFIXES=("North","South","East","West","Central","Upper","Lower");_REGION_NAMES=("Harbor","Cedar","River","Highland","Meadow","Lake","Valley");_COLORS=("blue","green","red","white","black","silver","gold","orange");_GENERIC_TERMS=("Aurora","Cedar","Harbor","Meridian","Summit","Willow","Atlas","Nova")
def _field(binding:Any,name:str,default:Any=None)->Any:return getattr(binding,name,default)
def _base_n(value:int,alphabet:str,width:int)->str:
    base=len(alphabet);chars=[];number=max(0,int(value))
    for _ in range(width):chars.append(alphabet[number%base]);number//=base
    return "".join(reversed(chars))
def _gtin14(seed:int)->str:
    body=f"{seed%10**13:013d}";total=sum(int(char)*(3 if index%2==0 else 1) for index,char in enumerate(reversed(body)));return body+str((10-total%10)%10)
def _isbn13(seed:int)->str:
    body="978"+f"{seed%10**9:09d}";total=sum(int(char)*(1 if index%2==0 else 3) for index,char in enumerate(body));return body+str((10-total%10)%10)
def _issn(seed:int)->str:
    body=f"{seed%10**7:07d}";total=sum(int(char)*weight for char,weight in zip(body,range(8,1,-1)));remainder=(11-total%11)%11;check="X" if remainder==10 else str(remainder);return f"{body[:4]}-{body[4:]}{check}"
def _identifier_value(predicate:str,seed:int)->str:
    key=_predicate_key(predicate)
    if "duns" in key:return f"{seed%1_000_000_000:09d}"
    if "gtin" in key:return _gtin14(seed)
    if "isbn" in key:return _isbn13(seed)
    if "issn" in key:return _issn(seed)
    if "iata" in key:return _base_n(seed,_ALPHABET,3)
    if "icao" in key:return _base_n(seed,_ALPHABET,4)
    if "lei" in key:return _base_n(seed,_ALNUM,20)
    if "postal" in key:return f"{seed%100_000:05d}"
    if "telephone" in key or "calling code" in key:return f"+{1+seed%998}"
    if "code" in key:return _base_n(seed,_ALNUM,6)
    return f"ID-{_base_n(seed,_ALNUM,10)}"
def _natural_string_value(predicate:str,family:str,seed:int)->str:
    key=_predicate_key(predicate)
    if family=="identifier" or any(token in key for token in ("duns","gtin","isbn","issn","iata","icao","lei","postal")) or "calling code" in key or "telephone" in key:return _identifier_value(predicate,seed)
    if family=="address" or "address" in key:return f"{10+seed%890} {_STREET_NAMES[(seed//11)%len(_STREET_NAMES)]} {_STREET_SUFFIXES[(seed//23)%len(_STREET_SUFFIXES)]}"
    if family=="slogan" or "slogan" in key or "motto" in key:return f"{_SLOGAN_PREFIXES[(seed//7)%len(_SLOGAN_PREFIXES)]} {_SLOGAN_SUFFIXES[(seed//17)%len(_SLOGAN_SUFFIXES)]}"
    if family=="emblem" or any(token in key for token in ("logo","emblem","crest")):return f"{_EMBLEM_ADJECTIVES[(seed//5)%len(_EMBLEM_ADJECTIVES)]} {_EMBLEM_SYMBOLS[(seed//19)%len(_EMBLEM_SYMBOLS)]} emblem"
    if family=="region" or key in {"geo","region"}:return f"{_REGION_PREFIXES[(seed//13)%len(_REGION_PREFIXES)]} {_REGION_NAMES[(seed//29)%len(_REGION_NAMES)]} region"
    if family=="color" or "color" in key or "colour" in key:return _COLORS[seed%len(_COLORS)]
    if "chemical structure" in key or "molecular formula" in key or "chemical formula" in key:
        carbon=2+seed%18;hydrogen=4+(seed//7)%34;nitrogen=(seed//17)%5;oxygen=1+(seed//31)%8;formula=f"C{carbon}H{hydrogen}";formula+=f"N{nitrogen}" if nitrogen else "";return formula+f"O{oxygen}"
    term=_GENERIC_TERMS[seed%len(_GENERIC_TERMS)];label=re.sub(r"\s+"," ",key).strip().title() or "Attribute";return f"{term} {label}"
class SemanticLiteralFactory:
    def __init__(self,*,seed:int=17)->None:self.seed=int(seed)
    def _numeric_value(self,binding:Any,*,ordinal:int)->int|float:
        lower=_field(binding,"value_min");upper=_field(binding,"value_max")
        if lower is None or upper is None:lower,upper=10.0,100.0
        lower=float(lower);upper=float(upper)
        if upper<=lower:upper=lower+max(1.0,abs(lower)*0.1)
        rank=max(0,min(8,int(ordinal)));base_ratio=0.16+0.085*rank;jitter=(_stable_int(self.seed,_field(binding,"predicate"),_field(binding,"entity_type"))%17)/1000.0;ratio=min(0.90,base_ratio+jitter);value=lower+(upper-lower)*ratio
        if bool(_field(binding,"integer_values",False)):return int(round(value))
        if str(_field(binding,'profile_evidence','')).startswith('typed-observed:'):
            import math
            places=max(1,min(14,3-int(math.floor(math.log10(upper-lower)))));return max(lower,min(upper,round(value,places)))
        return round(value,1)
    def value(self,binding:Any,*,ordinal:int=1)->Any:
        kind=str(_field(binding,"value_kind","string"));predicate=str(_field(binding,"predicate","attribute"));entity_type=str(_field(binding,"entity_type","entity"));family=str(_field(binding,"semantic_family","generic"))
        if kind=="number":return self._numeric_value(binding,ordinal=ordinal)
        if kind=="year":return int(self._numeric_value(binding,ordinal=ordinal))
        if kind=="date":
            year=1995+(_stable_int(self.seed,predicate,entity_type,ordinal)%28);month=1+(_stable_int(predicate,ordinal,"month")%12);day=1+(_stable_int(predicate,ordinal,"day")%28);return date(year,month,day).isoformat()
        if kind=="bool":return bool(_stable_int(self.seed,predicate,ordinal)%2)
        token=_stable_int(self.seed,predicate,entity_type,ordinal)
        if kind=="string":return _natural_string_value(predicate,family,token)
        return f"synthetic-{token%100_000_000:08d}"
    def comparison_pair(self,binding:Any,comparator:str|None)->tuple[Any,Any]:
        token=str(comparator or "=");low=self.value(binding,ordinal=1);mid=self.value(binding,ordinal=4);high=self.value(binding,ordinal=7)
        if token in _GREATER:return low,high
        if token in _LESS:return high,low
        if token in _NOT_EQUAL:return low,high
        return mid,mid
    def selection_pair(self,binding:Any,*,minimum:bool)->tuple[Any,Any]:
        low=self.value(binding,ordinal=1);high=self.value(binding,ordinal=7);return (low,high) if minimum else (high,low)
