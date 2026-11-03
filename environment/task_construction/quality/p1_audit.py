"""Batch semantic-plausibility audit used before P1 freeze."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
from ..data_sources.taxonomy import TaxonomyIndex
from .observed_support import ObservedSupport
from ..rendering.pipeline import world_from_dict

def audit_plausibility_jsonl(worlds_path,yago_catalog_path,taxonomy_path,*,output_path=None)->dict[str,Any]:
    catalog=json.loads(Path(yago_catalog_path).read_text(encoding="utf-8"));taxonomy=TaxonomyIndex.from_json(taxonomy_path);support=ObservedSupport(catalog,taxonomy);source=Path(worlds_path);target=Path(output_path) if output_path is not None else None;writer=target.open("w",encoding="utf-8") if target is not None else None;total=accepted=0;reasons=Counter()
    try:
        with source.open("r",encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():continue
                record=json.loads(line);report=support.world(world_from_dict(record["world"]));total+=1;accepted+=int(bool(report.get("accepted")));reasons.update(report.get("reasons",()))
                if writer is not None:writer.write(json.dumps({"candidate_id":record.get("candidate_id"),**report},ensure_ascii=False)+"\n")
    finally:
        if writer is not None:writer.close()
    return {"input":str(source),"output":str(target) if target is not None else None,"audited":total,"accepted":accepted,"acceptance_rate":accepted/total if total else 0.,"rejection_reasons":dict(reasons.most_common())}
__all__=["audit_plausibility_jsonl"]
