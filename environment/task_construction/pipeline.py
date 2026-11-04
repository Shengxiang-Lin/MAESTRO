"""Canonical programmatic construction pipeline for MAESTRO synthetic tasks."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from environment.config import EnvironmentConfigError, parse_with_config
from .capabilities import ground_capabilities_jsonl
from .deep_dependency.candidate_selection import DEFAULT_CANDIDATE_MULTIPLIER,generate_world_candidate_shortlist
from .deep_dependency.world_generation import generate_world_batch
from .freeze import FreezeError,freeze_batch
from .quality.execution_contract import audit_execution_contract_jsonl
from .quality.p1_audit import audit_plausibility_jsonl
from .rendering import render_jsonl
from .rendering.modality import audit_modality_jsonl
from .scenario import audit_scenarios_jsonl
from .source_preparation import SourcePreparationPaths,prepare_source_priors
from .task_design.io import plan_task_designs_jsonl
from .utils.io import write_json as _write_json
ENVIRONMENT_DIR=Path(__file__).resolve().parents[1];REPOSITORY_DIR=ENVIRONMENT_DIR.parent;WORLD_SOURCE_DIR=REPOSITORY_DIR/"data"/"world-source";OUTPUT_DIR=ENVIRONMENT_DIR/"data"
DEFAULT_KQA_TRAIN=WORLD_SOURCE_DIR/"KQA-Pro"/"train.json";DEFAULT_KQA_VAL=WORLD_SOURCE_DIR/"KQA-Pro"/"val.json";DEFAULT_YAGO_FACTS=WORLD_SOURCE_DIR/"YAGO-4.6"/"yago-facts.ttl";DEFAULT_YAGO_SCHEMA=WORLD_SOURCE_DIR/"YAGO-4.6"/"yago-schema.ttl";DEFAULT_YAGO_TAXONOMY=WORLD_SOURCE_DIR/"YAGO-4.6"/"yago-taxonomy.ttl"
def _prefer_existing(*paths):
    for path in paths:
        if path.is_file():return path
    return paths[0]
DEFAULT_MMQA_TRAIN=_prefer_existing(WORLD_SOURCE_DIR/"MultiModalQA"/"MMQA_train.jsonl.gz",WORLD_SOURCE_DIR/"MultiModalQA"/"MMQA_train.jsonl");DEFAULT_MMQA_DEV=_prefer_existing(WORLD_SOURCE_DIR/"MultiModalQA"/"MMQA_dev.jsonl.gz",WORLD_SOURCE_DIR/"MultiModalQA"/"MMQA_dev.jsonl")
SOURCE_PATHS=SourcePreparationPaths.under(OUTPUT_DIR);KQA_ANALYSIS=SOURCE_PATHS.kqa_analysis;YAGO_CATALOG=SOURCE_PATHS.yago_catalog;YAGO_ENTITY_PROFILES=SOURCE_PATHS.yago_profiles;YAGO_TAXONOMY_INDEX=SOURCE_PATHS.yago_taxonomy;MMQA_SEMANTIC_BUNDLES=SOURCE_PATHS.mmqa_bundle_bank;MMQA_SCENARIO_PRIOR=SOURCE_PATHS.mmqa_scenario_prior

def _read_json(path):
    if not Path(path).is_file():return {}
    try:payload=json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:return {}
    return dict(payload) if isinstance(payload,Mapping) else {}
def _require_inputs(paths):
    missing=[path for path in paths if not Path(path).is_file()]
    if missing:raise SystemExit("Missing required input file(s):\n"+"\n".join(f"  - {p}" for p in missing))
def _count_jsonl(path):
    with Path(path).open("r",encoding="utf-8") as handle:return sum(1 for line in handle if line.strip())
def _default_shortlist_target(target):return max(target+50,int(math.ceil(target*1.5)))
def _artifact_paths(target):return {"task_design":OUTPUT_DIR/f"task_designs_{target}.jsonl","scenario":OUTPUT_DIR/f"scenario_validity_{target}.jsonl","capability":OUTPUT_DIR/f"capability_worlds_{target}.jsonl","authoring":OUTPUT_DIR/f"authoring_inputs_{target}.jsonl","manifest":OUTPUT_DIR/f"frozen_manifest_{target}.json"}
def _require_prepared_sources():_require_inputs([KQA_ANALYSIS,YAGO_CATALOG,YAGO_ENTITY_PROFILES,YAGO_TAXONOMY_INDEX,MMQA_SEMANTIC_BUNDLES,MMQA_SCENARIO_PRIOR])
def _run_downstream(worlds_path,rendered_path,*,accepted,target,seed,progress):
    paths=_artifact_paths(target);result={};bar=tqdm(total=5,desc="Environment downstream",unit="stage",disable=not progress,dynamic_ncols=True)
    result["task_design"]=plan_task_designs_jsonl(worlds_path,paths["task_design"],max_records=accepted,taxonomy_index_path=YAGO_TAXONOMY_INDEX);bar.update(1)
    result["rendered"]=render_jsonl(worlds_path,rendered_path,seed=seed);bar.update(1)
    modality=audit_modality_jsonl(worlds_path,max_records=accepted);_write_json(OUTPUT_DIR/f"modality_audit_{target}.json",modality);result["modality"]=modality;bar.update(1)
    scenario=audit_scenarios_jsonl(worlds_path,output_path=paths["scenario"],max_records=accepted);_write_json(OUTPUT_DIR/f"scenario_validity_summary_{target}.json",scenario);result["scenario_validity"]=scenario;bar.update(1)
    capability=ground_capabilities_jsonl(worlds_path,paths["capability"],max_records=accepted,seed=seed,task_design_path=paths["task_design"]);_write_json(OUTPUT_DIR/f"capability_grounding_{target}.json",capability);result["capability_grounding"]=capability;result["semantic_plausibility"]=audit_plausibility_jsonl(worlds_path,YAGO_CATALOG,YAGO_TAXONOMY_INDEX,output_path=OUTPUT_DIR/f"semantic_plausibility_{target}.jsonl");result["execution_contract"]=audit_execution_contract_jsonl(worlds_path,output_path=OUTPUT_DIR/f"execution_contract_{target}.jsonl",seed=seed);bar.update(1);bar.close();return result
def build_parser():
    p=argparse.ArgumentParser(description="Build MAESTRO synthetic task worlds");p.add_argument("--target",type=int,default=100);p.add_argument("--shortlist-target",type=int);p.add_argument("--candidate-multiplier",type=float,default=DEFAULT_CANDIDATE_MULTIPLIER);p.add_argument("--seed",type=int,default=17);p.add_argument("--max-records",type=int);p.add_argument("--min-fragment-depth",type=int,default=2);p.add_argument("--max-fragment-depth",type=int,default=4);p.add_argument("--min-orchestration-depth",type=int,default=3);p.add_argument("--max-orchestration-depth",type=int,default=5);p.add_argument("--min-latent-anchors",type=int,default=2);p.add_argument("--min-parallel-width",type=int,default=2);p.add_argument("--kqa-train",type=Path,default=DEFAULT_KQA_TRAIN);p.add_argument("--kqa-val",type=Path,default=DEFAULT_KQA_VAL);p.add_argument("--yago-facts",type=Path,default=DEFAULT_YAGO_FACTS);p.add_argument("--yago-schema",type=Path,default=DEFAULT_YAGO_SCHEMA);p.add_argument("--yago-taxonomy",type=Path,default=DEFAULT_YAGO_TAXONOMY);p.add_argument("--mmqa-train",type=Path,default=DEFAULT_MMQA_TRAIN);p.add_argument("--mmqa-dev",type=Path,default=DEFAULT_MMQA_DEV);p.add_argument("--yago-profile-limit",type=int,default=4096);p.add_argument("--mmqa-max-records",type=int);p.add_argument("--reuse-source-priors",action="store_true");p.add_argument("--prepare-sources-only",action="store_true");p.add_argument("--resume-downstream",action="store_true");p.add_argument("--no-progress",action="store_true");return p
def main():
    
    try:
        args=parse_with_config(build_parser(),"task_construction")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    progress=not args.no_progress
    if args.target<=0:raise SystemExit("--target must be positive")
    shortlist=args.shortlist_target if args.shortlist_target is not None else _default_shortlist_target(args.target)
    if shortlist<args.target:raise SystemExit("--shortlist-target must be >= --target")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True);candidates=OUTPUT_DIR/f"semantic_candidates_{shortlist}_for_{args.target}.jsonl";worlds=OUTPUT_DIR/f"typed_worlds_{args.target}.jsonl";rendered=OUTPUT_DIR/f"rendered_worlds_{args.target}.jsonl";summary_path=OUTPUT_DIR/"build_summary.json";summary={"pipeline":{"mode":"validated_semantic_world_batch","target_worlds":args.target,"shortlist_target":shortlist,"candidate_multiplier":args.candidate_multiplier,"seed":args.seed,"semantic_binding":"graph_grounded_v2_plus_observed_support","split_policy":"group by topology/skeleton family when evaluating generalization"}}
    if args.resume_downstream:
        if not worlds.is_file():raise SystemExit(f"--resume-downstream requires {worlds}")
        _require_prepared_sources();accepted=_count_jsonl(worlds);summary["worlds"]={"accepted":accepted,"target":args.target,"target_filled":accepted>=args.target,"output":str(worlds),"resumed":True};summary.update(_run_downstream(worlds,rendered,accepted=accepted,target=args.target,seed=args.seed,progress=progress))
    else:
        if args.reuse_source_priors:_require_prepared_sources();summary["source_preparation"]=_read_json(SOURCE_PATHS.summary)
        else:
            _require_inputs([args.kqa_train,args.kqa_val,args.yago_facts,args.yago_schema,args.yago_taxonomy,args.mmqa_train,args.mmqa_dev]);summary["source_preparation"]=prepare_source_priors(kqa_train=args.kqa_train,kqa_val=args.kqa_val,yago_facts=args.yago_facts,yago_schema=args.yago_schema,yago_taxonomy=args.yago_taxonomy,mmqa_train=args.mmqa_train,mmqa_dev=args.mmqa_dev,output_dir=OUTPUT_DIR,seed=args.seed,yago_profile_limit=args.yago_profile_limit,mmqa_max_records=args.mmqa_max_records,progress=progress)
        if args.prepare_sources_only:_write_json(summary_path,summary);print(summary_path);return
        summary["candidate_selection"]=generate_world_candidate_shortlist([args.kqa_train,args.kqa_val],YAGO_CATALOG,YAGO_TAXONOMY_INDEX,candidates,shortlist_target=shortlist,world_target=args.target,candidate_multiplier=args.candidate_multiplier,seed=args.seed,max_records=args.max_records,min_fragment_depth=args.min_fragment_depth,max_fragment_depth=args.max_fragment_depth,min_orchestration_depth=args.min_orchestration_depth,max_orchestration_depth=args.max_orchestration_depth,min_latent_anchors=args.min_latent_anchors,min_parallel_width=args.min_parallel_width,progress=progress)
        if int(summary["candidate_selection"].get("accepted",0) or 0)<=0:
            _write_json(summary_path,summary)
            raise SystemExit(
                "Candidate shortlist is empty; no executable worlds can be generated. "
                "Inspect environment/data/build_summary.json -> candidate_selection."
                "rejection_reasons (for example, hardness or BindingError)."
            )
        summary["worlds"]=generate_world_batch(candidates,YAGO_CATALOG,YAGO_TAXONOMY_INDEX,worlds,target=args.target,min_depth=5,seed=args.seed,min_critical_sources=2,min_noncritical_sources=1,min_proof_sources=2,max_sources=6,mmqa_source_motifs_path=MMQA_SCENARIO_PRIOR,progress=progress);accepted=int(summary["worlds"].get("accepted",0) or 0)
        if accepted<=0:
            _write_json(summary_path,summary)
            raise SystemExit("World generation accepted no worlds. "
                             f"Rejection reasons: {summary['worlds'].get('rejection_reasons',{})}. "
                             f"See {summary['worlds'].get('rejected_output')} and {summary_path}")
        summary.update(_run_downstream(worlds,rendered,accepted=accepted,target=args.target,seed=args.seed,progress=progress))
    _write_json(summary_path,summary);accepted=int(summary["worlds"].get("accepted",0) or 0);paths=_artifact_paths(args.target)
    try:manifest=freeze_batch(summary,target=accepted,seed=args.seed,worlds_path=worlds,task_design_path=paths["task_design"],scenario_path=paths["scenario"],capability_path=paths["capability"],authoring_input_path=paths["authoring"],manifest_path=paths["manifest"])
    except FreezeError as exc:_write_json(summary_path,summary);raise SystemExit(str(exc)) from exc
    summary["freeze"]=manifest;_write_json(summary_path,summary);print(json.dumps({"status":"done","worlds":str(worlds),"manifest":str(paths["manifest"]),"authoring_inputs":str(paths["authoring"]),"summary":str(summary_path)},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
