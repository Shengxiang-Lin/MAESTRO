"""Canonical programmatic construction pipeline for MAESTRO synthetic tasks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any,Mapping
from tqdm.auto import tqdm
from environment.config import EnvironmentConfigError, parse_with_config
from .capabilities import ground_capabilities_jsonl
from .deep_dependency.candidate_selection import generate_candidate_pool
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
def _artifact_paths(pool_size):return {"task_design":OUTPUT_DIR/f"task_designs_{pool_size}.jsonl","scenario":OUTPUT_DIR/f"scenario_validity_{pool_size}.jsonl","capability":OUTPUT_DIR/f"capability_worlds_{pool_size}.jsonl","authoring":OUTPUT_DIR/f"authoring_inputs_{pool_size}.jsonl","manifest":OUTPUT_DIR/f"frozen_manifest_{pool_size}.json"}
def _require_prepared_sources():_require_inputs([KQA_ANALYSIS,YAGO_CATALOG,YAGO_ENTITY_PROFILES,YAGO_TAXONOMY_INDEX,MMQA_SEMANTIC_BUNDLES,MMQA_SCENARIO_PRIOR])
def _run_downstream(worlds_path,rendered_path,*,accepted,pool_size,seed,progress):
    paths=_artifact_paths(pool_size);result={};bar=tqdm(total=5,desc="Environment downstream",unit="stage",disable=not progress,dynamic_ncols=True)
    result["task_design"]=plan_task_designs_jsonl(worlds_path,paths["task_design"],max_records=accepted,taxonomy_index_path=YAGO_TAXONOMY_INDEX);bar.update(1)
    result["rendered"]=render_jsonl(worlds_path,rendered_path,seed=seed);bar.update(1)
    modality=audit_modality_jsonl(worlds_path,max_records=accepted);_write_json(OUTPUT_DIR/f"modality_audit_{pool_size}.json",modality);result["modality"]=modality;bar.update(1)
    scenario=audit_scenarios_jsonl(worlds_path,output_path=paths["scenario"],max_records=accepted);_write_json(OUTPUT_DIR/f"scenario_validity_summary_{pool_size}.json",scenario);result["scenario_validity"]=scenario;bar.update(1)
    capability=ground_capabilities_jsonl(worlds_path,paths["capability"],max_records=accepted,seed=seed,task_design_path=paths["task_design"]);_write_json(OUTPUT_DIR/f"capability_grounding_{pool_size}.json",capability);result["capability_grounding"]=capability;result["semantic_plausibility"]=audit_plausibility_jsonl(worlds_path,YAGO_CATALOG,YAGO_TAXONOMY_INDEX,output_path=OUTPUT_DIR/f"semantic_plausibility_{pool_size}.jsonl");result["execution_contract"]=audit_execution_contract_jsonl(worlds_path,output_path=OUTPUT_DIR/f"execution_contract_{pool_size}.jsonl",seed=seed);bar.update(1);bar.close();return result
def build_parser():
    p=argparse.ArgumentParser(description="Build all valid MAESTRO worlds from a fixed semantic candidate pool")
    p.add_argument("--candidate-pool-size",type=int,default=2000,help="number of validated semantic candidates to collect and evaluate")
    p.add_argument("--construction-mode",choices=("program_composition","semantic_anchored","semantic_anchored_v2"),
                   default="program_composition",
                   help="Original KQA/YAGO composition or YAGO subject-profile-anchored binding")
    p.add_argument("--anchor-tries",type=int,default=6,
                   help="Binding attempts per proposal in semantic_anchored mode")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--max-records",type=int)
    p.add_argument("--min-fragment-depth",type=int,default=2)
    p.add_argument("--max-fragment-depth",type=int,default=4)
    p.add_argument("--min-orchestration-depth",type=int,default=3)
    p.add_argument("--max-orchestration-depth",type=int,default=5)
    p.add_argument("--min-latent-anchors",type=int,default=2)
    p.add_argument("--min-parallel-width",type=int,default=2)
    p.add_argument("--kqa-train",type=Path,default=DEFAULT_KQA_TRAIN)
    p.add_argument("--kqa-val",type=Path,default=DEFAULT_KQA_VAL)
    p.add_argument("--yago-facts",type=Path,default=DEFAULT_YAGO_FACTS)
    p.add_argument("--yago-schema",type=Path,default=DEFAULT_YAGO_SCHEMA)
    p.add_argument("--yago-taxonomy",type=Path,default=DEFAULT_YAGO_TAXONOMY)
    p.add_argument("--mmqa-train",type=Path,default=DEFAULT_MMQA_TRAIN)
    p.add_argument("--mmqa-dev",type=Path,default=DEFAULT_MMQA_DEV)
    p.add_argument("--yago-profile-limit",type=int,default=4096)
    p.add_argument("--mmqa-max-records",type=int)
    p.add_argument("--reuse-source-priors",action="store_true")
    p.add_argument("--prepare-sources-only",action="store_true")
    p.add_argument("--resume-downstream",action="store_true")
    p.add_argument("--no-progress",action="store_true")
    p.add_argument("--candidate-max-attempts",type=int)
    p.add_argument("--world-attempts-per-candidate",type=int,default=3)
    p.add_argument("--candidate-workers",type=int)
    p.add_argument("--candidate-proposal-batch-size",type=int)
    p.add_argument("--restart-candidates",action="store_true")
    p.add_argument("--reuse-candidate-pool",action="store_true",
                   help="reuse the existing 2000 validated ProgramIR candidates, "
                        "but regenerate Worlds, evidence, TaskDesign and Freeze")
    p.add_argument("--web-search-weight",type=float,default=45)
    p.add_argument("--hybrid-weight",type=float,default=40)
    p.add_argument("--provided-media-weight",type=float,default=15)
    return p

def main():
    try:
        args=parse_with_config(build_parser(),"task_construction")
    except EnvironmentConfigError as exc:
        raise SystemExit(f"Environment configuration error: {exc}") from exc
    pool_size=args.candidate_pool_size
    if pool_size<1:
        raise SystemExit("--candidate-pool-size must be positive")
    if args.world_attempts_per_candidate<1:
        raise SystemExit("--world-attempts-per-candidate must be positive")
    if args.anchor_tries<1:
        raise SystemExit("--anchor-tries must be positive")
    if args.candidate_max_attempts is not None and args.candidate_max_attempts<1:
        raise SystemExit("--candidate-max-attempts must be positive")
    scenario_weights=(args.web_search_weight,args.hybrid_weight,args.provided_media_weight)
    if any(x<0 for x in scenario_weights) or not sum(scenario_weights):
        raise SystemExit("scenario weights must be nonnegative and sum to a positive value")
    if args.reuse_candidate_pool and (args.restart_candidates or args.resume_downstream):
        raise SystemExit("--reuse-candidate-pool cannot be combined with "
                         "--restart-candidates or --resume-downstream")
    progress=not args.no_progress
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    candidates=OUTPUT_DIR/f"semantic_candidates_pool_{pool_size}.jsonl"
    worlds=OUTPUT_DIR/f"typed_worlds_{pool_size}.jsonl"
    rendered=OUTPUT_DIR/f"rendered_worlds_{pool_size}.jsonl"
    summary_path=OUTPUT_DIR/(
        f"build_summary_{args.construction_mode}_{pool_size}.json"
        if args.construction_mode in ("semantic_anchored","semantic_anchored_v2") else "build_summary.json"
    )
    summary={"pipeline":{
        "mode":"fixed_candidate_pool",
        "construction_mode":args.construction_mode,
        "candidate_pool_size":pool_size,
        "seed":args.seed,
        "semantic_binding":"graph_grounded_v2_plus_observed_support",
        "split_policy":"group by topology/skeleton family when evaluating generalization",
    }}
    if args.resume_downstream:
        if not worlds.is_file():
            raise SystemExit(f"--resume-downstream requires {worlds}")
        _require_prepared_sources()
        accepted=_count_jsonl(worlds)
        summary["worlds"]={
            "accepted":accepted,
            "candidate_pool_size":pool_size,
            "output":str(worlds),
            "resumed":True,
        }
        summary.update(_run_downstream(worlds,rendered,accepted=accepted,pool_size=pool_size,
                                       seed=args.seed,progress=progress))
    else:
        if args.reuse_source_priors:
            _require_prepared_sources()
            summary["source_preparation"]=_read_json(SOURCE_PATHS.summary)
        else:
            _require_inputs([args.kqa_train,args.kqa_val,args.yago_facts,args.yago_schema,
                             args.yago_taxonomy,args.mmqa_train,args.mmqa_dev])
            summary["source_preparation"]=prepare_source_priors(
                kqa_train=args.kqa_train,kqa_val=args.kqa_val,
                yago_facts=args.yago_facts,yago_schema=args.yago_schema,
                yago_taxonomy=args.yago_taxonomy,
                mmqa_train=args.mmqa_train,mmqa_dev=args.mmqa_dev,
                output_dir=OUTPUT_DIR,seed=args.seed,
                yago_profile_limit=args.yago_profile_limit,
                mmqa_max_records=args.mmqa_max_records,progress=progress)
        if args.prepare_sources_only:
            _write_json(summary_path,summary)
            print(summary_path)
            return
        from .deep_dependency.parallel_candidates import CandidateCheckpointError
        if args.reuse_candidate_pool:
            if not candidates.is_file():
                raise SystemExit(f"--reuse-candidate-pool requires existing {candidates}")
            checkpoint=candidates.with_name(
                candidates.stem+".candidate-state.json"
            )
            if not checkpoint.is_file():
                raise SystemExit(
                    f"Cannot safely reuse candidate pool without {checkpoint}"
                )
            state=_read_json(checkpoint)
            settings=state.get("signature",{}).get("settings",{})
            if (settings.get("seed") != args.seed
                    or settings.get("pool_size") != pool_size
                    or settings.get("construction_mode","program_composition") != args.construction_mode
                    or (args.construction_mode in ("semantic_anchored","semantic_anchored_v2") and settings.get("anchor_tries") != args.anchor_tries)
                    or state.get("status") not in {"complete","budget_exhausted"}):
                raise SystemExit(
                    "Reusable candidate checkpoint has incompatible seed, "
                    "capacity or incomplete status"
                )
            number=_count_jsonl(candidates)
            if number<=0 or number>pool_size or state.get("accepted") != number:
                raise SystemExit(
                    f"Invalid reusable candidate pool count {number}, "
                    f"checkpoint={state.get('accepted')}, capacity={pool_size}"
                )
            summary["candidate_selection"]={
                "validated_candidates":number,
                "pool_filled":number==pool_size,
                "reused_existing_candidates":True,
                "candidate_pool_size":pool_size,
                "output":str(candidates),
            }
        else:
            try:
                summary["candidate_selection"]=generate_candidate_pool(
                [args.kqa_train,args.kqa_val],YAGO_CATALOG,YAGO_TAXONOMY_INDEX,
                candidates,pool_size=pool_size,seed=args.seed,max_records=args.max_records,
                min_fragment_depth=args.min_fragment_depth,
                max_fragment_depth=args.max_fragment_depth,
                min_orchestration_depth=args.min_orchestration_depth,
                max_orchestration_depth=args.max_orchestration_depth,
                min_latent_anchors=args.min_latent_anchors,
                min_parallel_width=args.min_parallel_width,
                progress=progress,max_attempts=args.candidate_max_attempts,
                candidate_workers=args.candidate_workers,
                candidate_proposal_batch_size=args.candidate_proposal_batch_size,
                    restart_candidates=args.restart_candidates,
                construction_mode=args.construction_mode,
                anchor_profiles_path=YAGO_ENTITY_PROFILES if args.construction_mode in ("semantic_anchored","semantic_anchored_v2") else None,
                anchor_tries=args.anchor_tries)
            except CandidateCheckpointError as exc:
                raise SystemExit(f"Candidate checkpoint error: {exc}") from exc
        if summary["candidate_selection"]["validated_candidates"]<=0:
            _write_json(summary_path,summary)
            raise SystemExit("No validated semantic candidates; inspect "
                             f"{summary_path} -> candidate_selection.rejection_reasons")
        summary["worlds"]=generate_world_batch(
            candidates,YAGO_CATALOG,YAGO_TAXONOMY_INDEX,worlds,
            min_depth=5,seed=args.seed,min_critical_sources=2,min_noncritical_sources=1,
            min_proof_sources=2,max_sources=6,
            mmqa_source_motifs_path=MMQA_SCENARIO_PRIOR,
            progress=progress,world_attempts_per_candidate=args.world_attempts_per_candidate,
            acquisition_scenario_weights=scenario_weights)
        accepted=int(summary["worlds"]["accepted"])
        if accepted<=0:
            _write_json(summary_path,summary)
            raise SystemExit("World generation accepted no worlds. "
                             f"Rejection reasons: {summary['worlds'].get('rejection_reasons',{})}. "
                             f"See {summary['worlds'].get('rejected_output')} and {summary_path}")
        summary.update(_run_downstream(worlds,rendered,accepted=accepted,pool_size=pool_size,
                                       seed=args.seed,progress=progress))
    _write_json(summary_path,summary)
    accepted=int(summary["worlds"].get("accepted",0) or 0)
    paths=_artifact_paths(pool_size)
    try:
        manifest=freeze_batch(
            summary,accepted_count=accepted,seed=args.seed,worlds_path=worlds,
            task_design_path=paths["task_design"],scenario_path=paths["scenario"],
            capability_path=paths["capability"],authoring_input_path=paths["authoring"],
            manifest_path=paths["manifest"])
    except FreezeError as exc:
        _write_json(summary_path,summary)
        raise SystemExit(str(exc)) from exc
    summary["freeze"]=manifest
    _write_json(summary_path,summary)
    print(json.dumps({
        "status":"done",
        "candidate_pool_size":pool_size,
        "accepted_worlds":accepted,
        "worlds":str(worlds),
        "manifest":str(paths["manifest"]),
        "authoring_inputs":str(paths["authoring"]),
        "summary":str(summary_path),
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
