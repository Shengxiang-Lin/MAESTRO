"""Attach compact subject-profile co-occurrence support to the YAGO catalog."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
def attach_entity_profile_support(catalog_path:str|Path,profile_index_path:str|Path)->dict[str,Any]:
    catalog_file=Path(catalog_path);profile_file=Path(profile_index_path);catalog=json.loads(catalog_file.read_text(encoding="utf-8"));profiles=json.loads(profile_file.read_text(encoding="utf-8"));support={"version":1,"policy":"subject_centric_joint_attribute_prior","profile_index":str(profile_file),"profile_count":int(profiles.get("profile_count",0) or 0),"type_profile_counts":dict(profiles.get("type_profile_counts",{})),"joint_attribute_support":dict(profiles.get("joint_attribute_support",{})),"identity_policy":"real_subject_ids_not_embedded_in_catalog"};catalog["entity_profile_support"]=support;catalog_file.write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding="utf-8");return {"version":1,"profile_count":support["profile_count"],"type_count":len(support["type_profile_counts"]),"joint_attribute_type_count":len(support["joint_attribute_support"]),"catalog":str(catalog_file),"profile_index":str(profile_file)}
__all__=["attach_entity_profile_support"]
