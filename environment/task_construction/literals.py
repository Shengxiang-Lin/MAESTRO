"""YAGO/RDF literal normalization used by typed binding and synthesis."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
LITERAL_RE=re.compile(r'^"(?P<lex>(?:\\.|[^"\\])*)"(?:(?P<lang>@[A-Za-z0-9-]+)|\^\^(?P<datatype>\S+))?$')
NUMERIC_XSD={"xsd:byte","xsd:short","xsd:int","xsd:integer","xsd:long","xsd:unsignedByte","xsd:unsignedShort","xsd:unsignedInt","xsd:unsignedLong","xsd:decimal","xsd:float","xsd:double","xsd:nonNegativeInteger","xsd:positiveInteger","xsd:nonPositiveInteger","xsd:negativeInteger"}; STRING_XSD={"xsd:string","rdf:langString"}; DATE_XSD={"xsd:date","xsd:dateTime"}; YEAR_XSD={"xsd:gYear"}; BOOL_XSD={"xsd:boolean"}
@dataclass(frozen=True)
class LiteralValue:
    raw:str; lexical:str; value:Any; kind:str; datatype:str|None=None; unit:str|None=None; language:str|None=None
    def comparable(self): return self.value.toordinal() if self.kind=="date" and isinstance(self.value,date) else self.value
def _unescape(text:str)->str:
    b=chr(92); return text.replace(b+b,b).replace(b+'"','"').replace(b+"n","\n").replace(b+"t","\t")
def _parse_number(text:str):
    try: value=float(text.replace(",",""))
    except ValueError: return None
    return int(value) if value.is_integer() and not any(ch in text.lower() for ch in (".","e")) else value
def parse_yago_literal(raw:str):
    text=str(raw).strip(); m=LITERAL_RE.match(text)
    if not m: return None
    lexical=_unescape(m.group("lex")); datatype=m.group("datatype"); lang_token=m.group("lang"); lang=lang_token[1:] if lang_token else None
    if datatype in BOOL_XSD: return LiteralValue(text,lexical,lexical.lower() in {"true","1"},"bool",datatype,language=lang)
    if datatype in YEAR_XSD:
        try: value=int(lexical)
        except ValueError: value=lexical
        return LiteralValue(text,lexical,value,"year",datatype,language=lang)
    if datatype in DATE_XSD:
        try: value=datetime.fromisoformat(lexical.replace("Z","+00:00")).date()
        except ValueError:
            try: value=date.fromisoformat(lexical[:10])
            except ValueError: value=lexical
        return LiteralValue(text,lexical,value,"date",datatype,language=lang)
    if datatype in NUMERIC_XSD:
        value=_parse_number(lexical); return LiteralValue(text,lexical,value if value is not None else lexical,"number",datatype,language=lang)
    numeric=_parse_number(lexical)
    if datatype and numeric is not None and not datatype.startswith(("xsd:","rdf:")): return LiteralValue(text,lexical,numeric,"number",datatype,unit=datatype,language=lang)
    return LiteralValue(text,lexical,lexical,"string",datatype,language=lang)
def literal_kind(raw:str):
    p=parse_yago_literal(raw); return p.kind if p else None
def datatype_kind(datatype):
    if datatype is None:return None
    if datatype in NUMERIC_XSD:return "number"
    if datatype in YEAR_XSD:return "year"
    if datatype in DATE_XSD:return "date"
    if datatype in BOOL_XSD:return "bool"
    if datatype in STRING_XSD:return "string"
    if datatype.startswith("yago:"):return "number"
    return None
def synthetic_value(kind:str,*,ordinal:int=0):
    if kind=="number":return 100+ordinal*10
    if kind=="year":return 2030+ordinal
    if kind=="date":return f"203{ordinal%10}-01-01"
    if kind=="bool":return bool(ordinal%2)
    return f"value_{ordinal}"
