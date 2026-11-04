"""Frozen public/private boundary validation before any LLM question-authoring call."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any,Mapping
class BoundaryError(RuntimeError):pass
_FORBIDDEN_KEYS={"answer","oracle_answer","raw_answer","facts","fact_ids","critical_fact_ids","proof_relevant","indispensable","minimal_proof","program","binding","world","hidden","source_roles","decision_support_fact_ids","anchor_registry"}
def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def _rows(path):
    with Path(path).open("r",encoding="utf-8") as handle:return [json.loads(line) for line in handle if line.strip()]
def _walk(value,path="$"):
    if isinstance(value,Mapping):
        for key,child in value.items():
            token=str(key)
            yield path+"."+token,token
            yield from _walk(child,path+"."+token)
    elif isinstance(value,list):
        for index,child in enumerate(value):yield from _walk(child,f"{path}[{index}]")
def _assert_public(row):
    leaked=[]
    for path,key in _walk(row):
        if key in _FORBIDDEN_KEYS:leaked.append(path)
    if leaked:raise BoundaryError("privileged field leaked into authoring projection: "+", ".join(leaked[:12]))
def validate_frozen_inputs(manifest_path,authoring_inputs_path,capabilities_path,*,worlds_path=None)->dict[str,Any]:
    manifest=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status")!="frozen":raise BoundaryError("manifest is not frozen")
    if manifest.get("semantic_world_policy_version")!=2:raise BoundaryError("unsupported semantic world policy")
    artifacts=manifest.get("artifacts",{})
    checks=(("authoring_inputs",authoring_inputs_path),("capabilities",capabilities_path))
    if worlds_path is not None:checks=(("worlds",worlds_path),*checks)
    for key,path in checks:
        expected=artifacts.get(key,{}).get("sha256") if isinstance(artifacts.get(key),Mapping) else None
        if not expected:raise BoundaryError(f"frozen artifact hash missing: {key}")
        if _sha(path)!=expected:raise BoundaryError(f"frozen artifact hash mismatch: {key}")
    authoring=_rows(authoring_inputs_path);capabilities=_rows(capabilities_path);cap_ids={str(row.get("task_id")) for row in capabilities};seen=set()
    for row in authoring:
        _assert_public(row);task_id=str(row.get("task_id") or "")
        if not task_id:raise BoundaryError("authoring input missing task_id")
        if task_id in seen:raise BoundaryError(f"duplicate authoring task_id: {task_id}")
        seen.add(task_id)
        if task_id not in cap_ids:raise BoundaryError(f"authoring task not present in frozen capabilities: {task_id}")
        if not isinstance(row.get("task_design"),Mapping):raise BoundaryError(f"{task_id}: missing public TaskDesign contract")
        if not isinstance(row.get("resources"),list):raise BoundaryError(f"{task_id}: missing public resources")
    frozen_count=int(manifest.get("frozen_count",0) or 0)
    if frozen_count<=0:raise BoundaryError("empty frozen batch")
    if len(authoring)!=frozen_count or len(capabilities)!=frozen_count:raise BoundaryError(f"authoring projection has {len(authoring)} rows; frozen manifest expects {frozen_count}")
    return {"status":"valid","manifest_version":manifest.get("version"),"semantic_world_policy_version":manifest.get("semantic_world_policy_version"),"frozen_count":frozen_count,"authoring_count":len(authoring),"capability_count":len(capabilities),"manifest":str(manifest_path),"authoring_inputs":str(authoring_inputs_path),"capabilities":str(capabilities_path),"worlds":str(worlds_path) if worlds_path is not None else None}
__all__=["BoundaryError","validate_frozen_inputs"]
