"""Generate and select graph-grounded deep ProgramIR candidates."""
from __future__ import annotations
import json,os
from pathlib import Path
from typing import Any,Mapping,Sequence
from ..binding import BindingError,GraphGroundedProgramBinder
from ..data_sources.taxonomy import TaxonomyIndex
from ..ir import ProgramIR
from ..data_sources.kqa import record_to_ir
from ..quality.hardness import complexity_profile
from .composer import compose_deep_program,prepare_composition_pools
from .terminal_families import terminal_family

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

def _generate_candidate_pool(
    kqa_paths, yago_catalog_path, taxonomy_path, output_path, *,
    pool_size, seed=17, max_records=None,
    min_fragment_depth=2, max_fragment_depth=4,
    min_orchestration_depth=3, max_orchestration_depth=5,
    min_latent_anchors=2, min_parallel_width=2, progress=False,
    max_attempts=None, candidate_workers=None,
    candidate_proposal_batch_size=None, restart_candidates=False,
):
    """Build a fixed number of *validated* semantic candidates, with checkpoints."""
    if pool_size<1:
        raise ValueError("pool_size must be positive")
    if max_attempts is not None and max_attempts<1:
        raise ValueError("max_attempts must be positive")
    programs = _iter_kqa_programs(kqa_paths, max_records=max_records)
    unique = {p.canonical_signature():p for p in programs}
    pools = prepare_composition_pools(tuple(unique.values()),
        min_depth=min_fragment_depth, max_depth=max_fragment_depth)
    if not all(pools):
        raise ValueError('no usable dependency-closed producer or terminal pools')
    catalog = json.loads(Path(yago_catalog_path).read_text(encoding='utf-8'))
    taxonomy = TaxonomyIndex.from_json(taxonomy_path)
    from .parallel_candidates import (input_signature,prepare_worker_context,
        run_batches,_worker_init)
    workers=min(8,os.cpu_count() or 1) if candidate_workers is None else candidate_workers
    batch_size=2*workers if candidate_proposal_batch_size is None else candidate_proposal_batch_size
    if workers<1 or batch_size<1:
        raise ValueError('candidate workers and proposal batch size must be positive')
    attempts_limit=max_attempts if max_attempts is not None else max(1000,pool_size*40)
    settings=dict(seed=seed,pool_size=pool_size,
        workers=workers,batch_size=batch_size,max_records=max_records,
        min_fragment_depth=min_fragment_depth,max_fragment_depth=max_fragment_depth,
        min_orchestration_depth=min_orchestration_depth,max_orchestration_depth=max_orchestration_depth,
        min_latent_anchors=min_latent_anchors,min_parallel_width=min_parallel_width)
    signature=input_signature([*kqa_paths,yago_catalog_path,taxonomy_path],settings)
    prepare_worker_context(catalog,taxonomy,seed)

    def make_proposal(attempt,rng,predicate_usage):
        try:
            motifs=('chain',) if min_parallel_width==1 else ('chain','fork_join','mixed_dag')
            composed=compose_deep_program((),prepared_pools=pools,
                min_fragment_depth=min_fragment_depth,max_fragment_depth=max_fragment_depth,
                min_orchestration_depth=min_orchestration_depth,max_orchestration_depth=max_orchestration_depth,
                min_latent_anchors=min_latent_anchors,min_parallel_width=min_parallel_width,
                motif=motifs[attempt%len(motifs)],seed=rng.randrange(1<<31))
            profile=complexity_profile(composed,min_depth=5)
            if not profile.qualifies:
                return {'attempt':attempt,'error':'hardness'}
            return {'attempt':attempt,'program':composed.to_dict(),'binding_seed':rng.randrange(1<<31),
                'predicate_usage':predicate_usage,'reasoning_motif':profile.label,'hardness':profile.to_dict()}
        except (BindingError,ValueError,RuntimeError) as exc:
            return {'attempt':attempt,'error':type(exc).__name__+':'+str(exc).split(':',1)[0],
                    'detail':str(exc)[:240]}

    candidates,parallel=run_batches(output_path,signature=signature,target=pool_size,
        attempts_limit=attempts_limit,seed=seed,workers=workers,batch_size=batch_size,
        make_proposal=make_proposal,progress=progress,restart=restart_candidates,
        initializer=_worker_init,initargs=(str(yago_catalog_path),str(taxonomy_path),seed))
    target=Path(output_path)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8") as writer:
        for row in candidates:
            writer.write(json.dumps(row,ensure_ascii=False)+"\n")
    return {
        "output":str(target),
        "candidate_pool_size":pool_size,
        "validated_candidates":len(candidates),
        "pool_filled":len(candidates)>=pool_size,
        "generated_attempts":parallel["attempts"],
        "attempts_limit":attempts_limit,
        "rejection_reasons":parallel["rejections"],
        "rejection_examples":parallel["examples"],
        "pools":{
            "source_programs":len(unique),
            "producer_fragments":len(pools[0]),
            "terminal_programs":len(pools[1]),
        },
        "parallel_generation":parallel,
    }


def generate_candidate_pool(*args,**kwargs):
    from .parallel_candidates import _exclusive_lock
    output=Path(args[3] if len(args)>3 else kwargs['output_path'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with _exclusive_lock(output.with_name(output.stem+'.candidate-output.lock')):
        return _generate_candidate_pool(*args,**kwargs)


__all__ = ["generate_candidate_pool"]
