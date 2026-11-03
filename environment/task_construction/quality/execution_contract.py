"""Execution-contract audit for frozen symbolic tasks."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
from ..executor import SymbolicExecutor
from ..rendering.pipeline import program_from_dict,world_from_dict

def audit_execution_contract_jsonl(worlds_path,*,output_path=None,seed=17)->dict[str,Any]:
    source=Path(worlds_path);target=Path(output_path) if output_path is not None else None;writer=target.open("w",encoding="utf-8") if target is not None else None;total=accepted=0;reasons=Counter();executor=SymbolicExecutor()
    try:
        with source.open("r",encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():continue
                record=json.loads(line);program=program_from_dict(record["program"]);world=world_from_dict(record["world"]);issues=[]
                try:trace=executor.execute_trace(program,world);answer=trace[program.topological_order()[-1]]
                except Exception as exc:trace={};answer=None;issues.append("executor_failure:"+str(exc))
                binding=record.get("binding",{});node_types=binding.get("node_types",{}) if isinstance(binding,dict) else {}
                for node_id,value in node_types.items():
                    try:node=program.node_map()[int(node_id)]
                    except Exception:issues.append(f"missing_bound_node:{node_id}");continue
                    if node.op in {"FIND","FIND_ALL","RELATE","FILTER_TYPE","FILTER_STRING","FILTER_NUMBER","FILTER_YEAR","FILTER_DATE","INTERSECT","UNION","SELECT_BETWEEN","SELECT_AMONG"} and int(node_id) not in trace:issues.append(f"missing_trace_value:{node_id}")
                ok=not issues;total+=1;accepted+=int(ok);reasons.update(issue.split(":",1)[0] for issue in issues);payload={"candidate_id":record.get("candidate_id"),"accepted":ok,"issues":issues,"answer":answer,"seed":seed}
                if writer is not None:writer.write(json.dumps(payload,ensure_ascii=False,default=str)+"\n")
    finally:
        if writer is not None:writer.close()
    return {"input":str(source),"output":str(target) if target is not None else None,"audited":total,"accepted":accepted,"acceptance_rate":accepted/total if total else 0.,"rejection_reasons":dict(reasons.most_common()),"seed":seed}
__all__=["audit_execution_contract_jsonl"]
