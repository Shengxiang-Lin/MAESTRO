"""KQA-Pro adapters used only to extract reasoning-program structure."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from ..ir import ProgramIR, ProgramNode, abstract_scalar
from ..utils.progress import Progress

OPERATOR_MAP = {
    "Find": "FIND", "FindAll": "FIND_ALL", "Relate": "RELATE", "FilterConcept": "FILTER_TYPE",
    "FilterStr": "FILTER_STRING", "FilterNum": "FILTER_NUMBER", "FilterYear": "FILTER_YEAR", "FilterDate": "FILTER_DATE",
    "QFilterStr": "FILTER_QUALIFIER_STRING", "QFilterNum": "FILTER_QUALIFIER_NUMBER", "QFilterYear": "FILTER_QUALIFIER_YEAR", "QFilterDate": "FILTER_QUALIFIER_DATE",
    "And": "INTERSECT", "Or": "UNION", "Count": "COUNT", "What": "QUERY_NAME", "QueryName": "QUERY_NAME",
    "QueryAttr": "QUERY_ATTR", "QueryAttrUnderCondition": "QUERY_ATTR_CONDITIONAL", "QueryRelation": "QUERY_RELATION",
    "QueryAttrQualifier": "QUERY_ATTR_QUALIFIER", "QueryRelationQualifier": "QUERY_RELATION_QUALIFIER",
    "SelectBetween": "SELECT_BETWEEN", "SelectAmong": "SELECT_AMONG", "VerifyStr": "VERIFY_STRING",
    "VerifyNum": "VERIFY_NUMBER", "VerifyYear": "VERIFY_YEAR", "VerifyDate": "VERIFY_DATE",
}

def normalize_operator(function: str) -> str:
    return OPERATOR_MAP.get(function, f"KQA_{function.upper()}")

def _abstract_cmp(value: Any) -> str:
    token = abstract_scalar(value)
    return token if token.startswith("<CMP:") else "<CMP:=>"

def abstract_inputs(function: str, inputs: Sequence[Any]) -> tuple[str, ...]:
    values = list(inputs)
    if not values: return ()
    if function == "Relate":
        direction = str(values[1]).strip().lower() if len(values)>1 else "forward"
        return ("<REL>", direction if direction in {"forward","backward"} else "forward")
    if function in {"Find","FilterConcept"}: return ("<ENTITY_OR_TYPE>",)
    if function == "FilterStr":
        target=abstract_scalar(values[1]) if len(values)>1 else "<SYMBOL>"; return ("<ATTR>","<CMP:=>",target)
    if function in {"FilterNum","FilterYear","FilterDate"}:
        target=abstract_scalar(values[1]) if len(values)>1 else "<SYMBOL>"; comparator=_abstract_cmp(values[2]) if len(values)>2 else "<CMP:=>"; return ("<ATTR>",comparator,target)
    if function == "VerifyStr": return ("<CMP:=>", abstract_scalar(values[0]))
    if function in {"VerifyNum","VerifyYear","VerifyDate"}:
        target=abstract_scalar(values[0]); comparator=_abstract_cmp(values[1]) if len(values)>1 else "<CMP:=>"; return (comparator,target)
    if function in {"SelectBetween","SelectAmong"}:
        comparator=_abstract_cmp(values[1]) if len(values)>1 else "<CMP:greater>"; return ("<ATTR>",comparator)
    if function == "QueryAttr": return ("<ATTR>",)
    if function == "QueryRelation": return ()
    relation_like={"QueryRelationQualifier"}; attribute_like={"QFilterStr","QFilterNum","QFilterYear","QFilterDate","QueryAttrUnderCondition","QueryAttrQualifier"}; result=[]
    for index,value in enumerate(values):
        text=str(value).strip(); lowered=text.lower()
        if lowered in {"forward","backward"}: result.append(lowered); continue
        if lowered in {"=","==","!=",">","<",">=","<=","greater","less"}: result.append(f"<CMP:{lowered}>"); continue
        if function in relation_like and index==0: result.append("<REL>"); continue
        if function in attribute_like and index==0: result.append("<ATTR>"); continue
        result.append(abstract_scalar(value))
    return tuple(result)

def record_to_ir(record: Dict[str, Any], *, source: str="kqa_pro") -> ProgramIR:
    raw_program=record.get("program")
    if not isinstance(raw_program,list): raise ValueError("record has no KQA program list")
    nodes=[]
    for node_id,item in enumerate(raw_program):
        function=str(item.get("function","UNKNOWN")); deps=tuple(int(dep) for dep in item.get("dependencies",[])); inputs=abstract_inputs(function,item.get("inputs",[])); nodes.append(ProgramNode(node_id=node_id,op=normalize_operator(function),dependencies=deps,inputs=inputs,raw_function=function))
    return ProgramIR(nodes=nodes,metadata={"source":source,"question_type":record.get("question_type")})

def iter_kqa_records(paths: Sequence[str|Path], *, progress: bool=False) -> Iterator[Dict[str,Any]]:
    for path_like in paths:
        path=Path(path_like)
        with Progress(f"KQA loading {path.name}",unit="files",enabled=progress) as loading:
            with path.open("r",encoding="utf-8") as handle:data=json.load(handle)
            loading.advance()
        if not isinstance(data,list): raise ValueError(f"expected a JSON list in {path}")
        with Progress(f"KQA analyzing {path.name}",total=len(data),enabled=progress) as bar:
            for record in data:
                if isinstance(record,dict) and isinstance(record.get("program"),list): yield record
                bar.advance()

def analyze_programs(paths: Sequence[str|Path], *, top_k:int=200, progress:bool=False)->Dict[str,Any]:
    topology_counts=Counter(); operator_counts=Counter(); metric_sums=Counter(); metric_histograms={}; total=0
    for record in iter_kqa_records(paths,progress=progress):
        ir=record_to_ir(record); total+=1; topology_counts[ir.canonical_signature()]+=1; operator_counts.update(node.op for node in ir.nodes); metrics=ir.metrics()
        for key,value in metrics.items(): metric_sums[key]+=value; metric_histograms.setdefault(key,Counter())[value]+=1
    return {"records":total,"distinct_topologies":len(topology_counts),"operator_counts":dict(operator_counts.most_common()),"average_metrics":{k:(v/total if total else 0.0) for k,v in metric_sums.items()},"metric_histograms":{k:{str(b):c for b,c in sorted(h.items())} for k,h in metric_histograms.items()},"topologies":[{"signature":s,"count":c} for s,c in topology_counts.most_common(top_k)]}

def write_program_analysis(paths:Sequence[str|Path],output_path:str|Path,*,top_k:int=200,progress:bool=False)->Dict[str,Any]:
    result=analyze_programs(paths,top_k=top_k,progress=progress); output=Path(output_path); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); return result
