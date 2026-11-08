"""Canonical preparation of external structural/context priors.

This module is the only entry point that should parse KQA-Pro, YAGO and
MultiModalQA for the synthetic environment. Task-world generation consumes the
frozen artifacts produced here; it does not independently reinterpret source data.
"""
from __future__ import annotations
import argparse,json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm
from .data_sources import attach_entity_profile_support,build_entity_profile_index,build_mmqa_semantic_bundle_bank,build_predicate_catalog,build_taxonomy_index,write_program_analysis
from .evidence.scenario_prior import ScenarioEvidencePrior
@dataclass(frozen=True)
class SourcePreparationPaths:
    output_dir:Path;kqa_analysis:Path;yago_catalog:Path;yago_profiles:Path;yago_taxonomy:Path;mmqa_bundle_bank:Path;mmqa_scenario_prior:Path;summary:Path
    @classmethod
    def under(cls,output_dir:str|Path):
        root=Path(output_dir);return cls(root,root/"kqa_program_analysis.json",root/"yago_predicate_catalog.json",root/"yago_entity_profiles.json",root/"yago_taxonomy_index.json",root/"mmqa_semantic_bundles.jsonl",root/"mmqa_scenario_prior.json",root/"source_preparation_summary.json")
    def to_dict(self):return {key:str(value) for key,value in self.__dict__.items() if key!="output_dir"}
def _write_json(path:Path,payload:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
def _require(paths:list[Path])->None:
    missing=[path for path in paths if not path.is_file()]
    if missing:raise FileNotFoundError("missing source data:\n"+"\n".join(f"  - {path}" for path in missing))
def _scenario_prior_payload(prior:ScenarioEvidencePrior,*,bundle_summary:dict[str,Any])->dict[str,Any]:
    patterns=sorted(prior.patterns,key=lambda item:(-item.count,item.source_types));return {"version":1,"scenario_schema":"scenario_semantic_bundle.v1_projection","records":int(bundle_summary.get("scanned",0) or 0),"bundles":int(bundle_summary.get("bundles",0) or 0),"records_with_intermediate_steps":int(bundle_summary.get("records_with_intermediate_observations",0) or 0),"intermediate_step_count":int(bundle_summary.get("intermediate_observation_count",0) or 0),"nested_intermediate_records_seen":int(bundle_summary.get("nested_intermediate_records_seen",0) or 0),"scenario_patterns":[item.to_dict() for item in patterns[:400]],"prior_summary":prior.summary(),"perception_policy":"modality_conditioned_textual_observation_not_raw_media","note":"Derived from the already-frozen MMQA semantic bundle bank; original MMQA is parsed once. Scenario patterns provide artifact/discovery structure only, never synthetic truth."}
def compact_source_summary(report):
    """Keep provenance/stats, never inline the 100MB+ YAGO predicate catalog.

    Historical source-preparation summaries may already embed full catalogs.
    Trim them on reuse as well; never mutate the reusable source indexes.
    """
    report = dict(report or {})
    full = report.get("yago_catalog")
    if isinstance(full, dict):
        compact = {k: full[k] for k in (
            "source", "schema_source", "total_triples", "predicate_count",
            "samples_per_predicate", "schema_signature_count",
        ) if k in full}
        artifacts = report.get("artifacts") or {}
        if artifacts.get("yago_catalog"):
            compact["output"] = artifacts["yago_catalog"]
        report["yago_catalog"] = compact
    return report

def prepare_source_priors(*,kqa_train:str|Path,kqa_val:str|Path,yago_facts:str|Path,yago_schema:str|Path,yago_taxonomy:str|Path,mmqa_train:str|Path,mmqa_dev:str|Path,output_dir:str|Path,seed:int=17,yago_samples_per_predicate:int=128,yago_profile_limit:int=4096,yago_profile_literal_limit:int=24,mmqa_max_records:int|None=None,progress:bool=False)->dict[str,Any]:
    inputs=[Path(kqa_train),Path(kqa_val),Path(yago_facts),Path(yago_schema),Path(yago_taxonomy),Path(mmqa_train),Path(mmqa_dev)];_require(inputs);paths=SourcePreparationPaths.under(output_dir);paths.output_dir.mkdir(parents=True,exist_ok=True);result={"version":1,"policy":"frozen_source_priors_before_world_synthesis","seed":int(seed),"inputs":{"kqa_train":str(kqa_train),"kqa_val":str(kqa_val),"yago_facts":str(yago_facts),"yago_schema":str(yago_schema),"yago_taxonomy":str(yago_taxonomy),"mmqa_train":str(mmqa_train),"mmqa_dev":str(mmqa_dev)},"artifacts":paths.to_dict(),"dataset_roles":{"KQA-Pro":"reasoning_program_prior","YAGO":"typed_grounding_and_subject_centric_object_profile_prior","MultiModalQA":"artifact_organization_and_progressive_discovery_prior"},"perception_policy":"modality_conditioned_textual_observation_not_raw_media"}
    with tqdm(total=5,desc="Prepare source priors",unit="stage",disable=not progress,dynamic_ncols=True) as bar:
        result["kqa"]=write_program_analysis([kqa_train,kqa_val],paths.kqa_analysis,top_k=200,progress=progress);bar.update(1)
        result["yago_catalog"]=build_predicate_catalog(yago_facts,paths.yago_catalog,schema_path=yago_schema,samples_per_predicate=yago_samples_per_predicate,seed=seed,progress=progress);bar.update(1)
        result["yago_profiles"]=build_entity_profile_index(yago_facts,paths.yago_catalog,paths.yago_profiles,max_profiles=yago_profile_limit,max_literal_facts_per_profile=yago_profile_literal_limit,seed=seed,progress=progress);result["yago_profile_support"]=attach_entity_profile_support(paths.yago_catalog,paths.yago_profiles);bar.update(1)
        result["yago_taxonomy"]=build_taxonomy_index(yago_taxonomy,paths.yago_taxonomy,progress=progress);bar.update(1)
        bundle_summary=build_mmqa_semantic_bundle_bank([mmqa_train,mmqa_dev],paths.mmqa_bundle_bank,max_records=mmqa_max_records,progress=progress);result["mmqa_bundle_bank"]=bundle_summary;prior=ScenarioEvidencePrior.from_bundle_bank(paths.mmqa_bundle_bank);mmqa_prior=_scenario_prior_payload(prior,bundle_summary=bundle_summary);_write_json(paths.mmqa_scenario_prior,mmqa_prior);result["mmqa_scenario_prior"]=mmqa_prior;bar.update(1)
    result=compact_source_summary(result)
    _write_json(paths.summary,result);return result
def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Prepare frozen source priors for MAESTRO");parser.add_argument("--kqa-train",type=Path,required=True);parser.add_argument("--kqa-val",type=Path,required=True);parser.add_argument("--yago-facts",type=Path,required=True);parser.add_argument("--yago-schema",type=Path,required=True);parser.add_argument("--yago-taxonomy",type=Path,required=True);parser.add_argument("--mmqa-train",type=Path,required=True);parser.add_argument("--mmqa-dev",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--seed",type=int,default=17);parser.add_argument("--yago-profile-limit",type=int,default=4096);parser.add_argument("--mmqa-max-records",type=int);parser.add_argument("--no-progress",action="store_true");return parser
def main()->None:
    args=build_parser().parse_args();result=prepare_source_priors(kqa_train=args.kqa_train,kqa_val=args.kqa_val,yago_facts=args.yago_facts,yago_schema=args.yago_schema,yago_taxonomy=args.yago_taxonomy,mmqa_train=args.mmqa_train,mmqa_dev=args.mmqa_dev,output_dir=args.output_dir,seed=args.seed,yago_profile_limit=args.yago_profile_limit,mmqa_max_records=args.mmqa_max_records,progress=not args.no_progress);print(json.dumps({"status":"prepared","summary":str(SourcePreparationPaths.under(args.output_dir).summary),"artifacts":result["artifacts"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
