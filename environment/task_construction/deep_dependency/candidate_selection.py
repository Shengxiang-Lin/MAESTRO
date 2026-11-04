"""Generate and select graph-grounded deep ProgramIR candidates."""
from __future__ import annotations
import json,math,random
from collections import Counter
from pathlib import Path
from typing import Any,Iterable,Mapping,Sequence
from tqdm.auto import tqdm
from ..binding import BindingError,GraphGroundedProgramBinder
from ..data_sources.taxonomy import TaxonomyIndex
from ..ir import ProgramIR
from ..data_sources.kqa import record_to_ir
from ..quality.diversity import BatchDiversityTracker
from ..quality.hardness import complexity_profile,pure_topology_signature
from .composer import compose_deep_program
from .terminal_families import terminal_family
DEFAULT_CANDIDATE_MULTIPLIER=8.0

def _program_from_kqa(item:Mapping[str,Any])->ProgramIR|None:
    """Use the canonical KQA adapter, not raw KQA function names.

    KQA records say Find/Relate/FilterNum/And/Count, whereas the IR,
    composer, binder, and executor use FIND/RELATE/FILTER_NUMBER/
    INTERSECT/COUNT. Keeping raw labels silently produced empty
    compositions and a zero-candidate shortlist.
    """
    raw=item.get("program")
    if not isinstance(raw,list) or not raw:return None
    try:return record_to_ir(dict(item))
    except (ValueError,TypeError,KeyError):return None

def _iter_kqa_programs(paths:Sequence[str|Path],*,max_records:int|None=None)->list[ProgramIR]:
    programs=[];seen=0
    for path in paths:
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        iterable=payload if isinstance(payload,list) else payload.get("data",()) if isinstance(payload,Mapping) else ()
        for item in iterable:
            if max_records is not None and seen>=max_records:return programs
            seen+=1
            if isinstance(item,Mapping):
                program=_program_from_kqa(item)
                if program is not None and len(program.nodes)>=2:programs.append(program)
    return programs

def _binding_predicates(binding:Mapping[str,Any])->list[str]:
    result=[]
    for key in ("relation_bindings","attribute_bindings"):
        for item in binding.get(key,{}).values():
            if isinstance(item,Mapping) and item.get("predicate"):result.append(str(item["predicate"]))
    return result

def _binding_type_pairs(binding:Mapping[str,Any])->list[str]:
    result=[]
    for item in binding.get("relation_bindings",{}).values():
        if isinstance(item,Mapping):result.append(f"relation:{item.get('subject_type')}->{item.get('object_type')}")
    for item in binding.get("attribute_bindings",{}).values():
        if isinstance(item,Mapping):result.append(f"attribute:{item.get('entity_type')}->{item.get('value_kind')}")
    return result

def generate_world_candidate_shortlist(kqa_paths:Sequence[str|Path],yago_catalog_path:str|Path,taxonomy_path:str|Path,output_path:str|Path,*,shortlist_target:int,world_target:int,candidate_multiplier:float=DEFAULT_CANDIDATE_MULTIPLIER,seed:int=17,max_records:int|None=None,min_fragment_depth:int=2,max_fragment_depth:int=4,min_orchestration_depth:int=3,max_orchestration_depth:int=5,min_latent_anchors:int=2,min_parallel_width:int=2,progress:bool=False)->dict[str,Any]:
    programs=_iter_kqa_programs(kqa_paths,max_records=max_records)
    if not programs:raise ValueError("no usable KQA-Pro programs")
    catalog=json.loads(Path(yago_catalog_path).read_text(encoding="utf-8"));taxonomy=TaxonomyIndex.from_json(taxonomy_path);rng=random.Random(seed);reservoir_target=max(shortlist_target,int(math.ceil(shortlist_target*candidate_multiplier)));tracker=BatchDiversityTracker();accepted=[];rejections=Counter();rejection_examples=[];attempts=0;iterator=range(reservoir_target)
    if progress:iterator=tqdm(iterator,desc="Semantic candidate reservoir",unit="candidate")
    for _ in iterator:
        attempts+=1
        try:
            composed=compose_deep_program(programs,min_fragment_depth=min_fragment_depth,max_fragment_depth=max_fragment_depth,min_orchestration_depth=min_orchestration_depth,max_orchestration_depth=max_orchestration_depth,min_parallel_width=min_parallel_width,seed=rng.randrange(1<<31))
            profile=complexity_profile(composed,min_depth=max(3,min_fragment_depth+1))
            if not profile.qualifies:rejections["hardness"]+=1;continue
            binder=GraphGroundedProgramBinder(catalog,taxonomy,seed=rng.randrange(1<<31),candidate_limit=32,max_states=4096);plan=binder.bind(composed);binding=plan.to_dict();topology=pure_topology_signature(plan.program);predicates=_binding_predicates(binding);type_pairs=_binding_type_pairs(binding);family=terminal_family(plan.program);ok,reasons=tracker.would_accept(topology=topology,predicates=predicates,type_pairs=type_pairs,terminal_family=family)
            if not ok:rejections.update(reasons);continue
            row={"candidate_id":f"candidate-{len(accepted)+1:06d}","program":plan.program.to_dict(),"binding":binding,"source_program_signature":plan.program.canonical_signature(),"reasoning_motif":profile.label,"hardness":profile.to_dict(),"topology_signature":topology,"terminal_family":family,"selection":{"world_target":world_target,"shortlist_target":shortlist_target,"candidate_multiplier":candidate_multiplier}}
            accepted.append(row);tracker.add(topology=topology,predicates=predicates,type_pairs=type_pairs,terminal_family=family)
            if len(accepted)>=shortlist_target:break
        except (BindingError,ValueError,RuntimeError) as exc:
            detail=str(exc)
            category=(detail.split(":",1)[0].strip() or type(exc).__name__)
            key=f"{type(exc).__name__}:{category}"
            rejections[key]+=1
            if len(rejection_examples)<20:
                rejection_examples.append({"attempt":attempts,"category":key,"detail":detail[:240]})
        if progress and attempts%50==0 and hasattr(iterator,"set_postfix"):
            main_reason=rejections.most_common(1)
            iterator.set_postfix(
                accepted=len(accepted),
                dominant_rejection=main_reason[0][0][:45] if main_reason else "none",
            )
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8") as writer:
        for row in accepted:writer.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {"output":str(target),"requested_shortlist":shortlist_target,"world_target":world_target,"generated_attempts":attempts,"accepted":len(accepted),"acceptance_rate":len(accepted)/attempts if attempts else 0.0,"reservoir_target":reservoir_target,"rejection_reasons":dict(rejections.most_common()),"rejection_examples":rejection_examples,"diversity":tracker.summary(),"policy":"deep_program_composition_then_graph_grounded_semantic_binding_then_diversity_selection"}
__all__=["DEFAULT_CANDIDATE_MULTIPLIER","generate_world_candidate_shortlist"]
