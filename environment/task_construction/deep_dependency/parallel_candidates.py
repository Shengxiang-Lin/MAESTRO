"""Ordered parallel binding with an append-only, batch-committed checkpoint."""
from __future__ import annotations
import hashlib
import json
import multiprocessing as mp
import os
import random
import signal
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tqdm.auto import tqdm

_CONTEXT = None
_LOCK_HANDLES = []
SCHEMA = 'maestro_candidate_batches.v1'


class CandidateCheckpointError(RuntimeError):
    pass


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def input_signature(paths,settings):
    inputs=[{'path':str(Path(p).resolve()),'sha256':_digest(p)} for p in paths]
    root=Path(__file__).resolve().parents[1]
    code=[Path(__file__),root/'deep_dependency/candidate_selection.py',
          root/'deep_dependency/composer.py',root/'deep_dependency/terminal_families.py',
          root/'ir.py',root/'literals.py',root/'data_sources/kqa.py',root/'data_sources/taxonomy.py']
    for directory in ('binding','semantic','quality','utils'):
        code.extend(sorted((root/directory).glob('*.py')))
    return {'schema':SCHEMA,'settings':settings,'inputs':inputs,
            'implementation':hashlib.sha256(''.join(_digest(p) for p in code).encode()).hexdigest()}


def prepare_worker_context(catalog,taxonomy,seed,anchor_profiles_path=None):
    """Build reusable read-mostly indexes before Linux fork."""
    global _CONTEXT
    from ..binding.predicate_binder import PredicateBinder
    from ..binding.graph_grounding import GraphWitnessIndex
    predicates=PredicateBinder(catalog,taxonomy,seed=seed)
    predicates._relation_options()
    for kind in (None,'string','number','year','date'):
        tuple(predicates._attribute_specs(kind))
    GraphWitnessIndex(catalog,taxonomy)
    anchor_bank=None
    if anchor_profiles_path is not None:
        from ..binding.semantic_anchors import SemanticAnchorBank
        anchor_bank=SemanticAnchorBank.from_file(anchor_profiles_path,catalog,taxonomy)
    _CONTEXT=(catalog,taxonomy,predicates,anchor_bank)


def _worker_init(catalog_path,taxonomy_path,seed,anchor_profiles_path=None):
    global _CONTEXT
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if _CONTEXT is None:
        from ..data_sources.taxonomy import TaxonomyIndex
        prepare_worker_context(json.loads(Path(catalog_path).read_text()),
                               TaxonomyIndex.from_json(taxonomy_path),seed,anchor_profiles_path)


def bind_proposal(proposal):
    if 'error' in proposal:return proposal
    from ..binding import BindingError,GraphGroundedProgramBinder
    from ..ir import ProgramIR,ProgramNode
    from ..quality.hardness import pure_topology_signature
    from .terminal_families import terminal_family
    catalog,taxonomy,predicates,anchor_bank=_CONTEXT
    raw=proposal['program']
    program=ProgramIR([ProgramNode(int(n['node_id']),n['op'],tuple(n['dependencies']),
        tuple(n['inputs']),n.get('raw_function')) for n in raw['nodes']],metadata=raw['metadata'])
    anchored=proposal.get('construction_mode')=='semantic_anchored'
    if anchored and anchor_bank is None:
        return {'attempt':proposal['attempt'],'error':'semantic_anchor_bank_missing'}
    attempts=proposal.get('anchor_tries',6) if anchored else 1
    last_error='semantic_anchor_joint_support_missing'
    for index in range(attempts):
        binding_seed=proposal['binding_seed']+index*10007
        predicates.rng=random.Random(binding_seed)
        anchor=anchor_bank.sample(program,rng=random.Random(binding_seed)) if anchored else None
        try:
            binder=GraphGroundedProgramBinder(catalog,taxonomy,seed=binding_seed,
                candidate_limit=16,max_states=128,predicate_usage=proposal['predicate_usage'],
                predicate_binder=predicates,
                preferred_predicates=anchor['predicates'] if anchor else None)
            plan=binder.bind(program)
            evidence=anchor_bank.evaluate(plan,anchor) if anchored else None
            if anchored and evidence is None:
                last_error='semantic_anchor_joint_support_missing'
                continue
            row={'program':plan.program.to_dict(),
                'binding':plan.to_dict(),'source_program_signature':plan.program.canonical_signature(),
                'reasoning_motif':proposal['reasoning_motif'],'hardness':proposal['hardness'],
                'topology_signature':pure_topology_signature(plan.program),
                'terminal_family':terminal_family(plan.program)}
            if evidence is not None:
                row['semantic_anchor']=evidence
            return {'attempt':proposal['attempt'],'row':row}
        except (BindingError,ValueError,RuntimeError) as exc:
            last_error=type(exc).__name__+':'+str(exc).split(':',1)[0]
    return {'attempt':proposal['attempt'],'error':last_error,
            'detail':f'No supported subject-centered joint predicate binding after {attempts} attempts'}


@contextmanager
def _exclusive_lock(path):
    handle=Path(path).open('a+b')
    _LOCK_HANDLES.append(handle)
    try:
        if os.name=='posix':
            import fcntl
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise CandidateCheckpointError('another process owns this candidate checkpoint') from exc
        else:
            import msvcrt
            handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:raise CandidateCheckpointError('another process owns this candidate checkpoint') from exc
        yield
    finally:
        _LOCK_HANDLES.remove(handle)
        handle.close()


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value,list) else value


def _write_checkpoint(path,state):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(state,handle,ensure_ascii=False,sort_keys=True)
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    if os.name=='posix':
        fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def run_batches(output_path,*,signature,target,attempts_limit,seed,workers,batch_size,
                make_proposal,progress=False,restart=False,worker_function=bind_proposal,
                initializer=None,initargs=()):
    """Commit whole batches in proposal order; interrupted batches are replayed."""
    output=Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    state_path=output.with_name(output.stem+'.candidate-state.json')
    partial=output.with_name(output.stem+'.candidate-progress.jsonl')
    with _exclusive_lock(state_path.with_suffix('.lock')):
        return _run_locked(state_path,partial,signature,target,attempts_limit,seed,
            workers,batch_size,make_proposal,progress,restart,worker_function,initializer,initargs)


def _run_locked(state_path,partial,signature,target,attempts_limit,seed,workers,batch_size,
                make_proposal,progress,restart,worker_function,initializer,initargs):
    if min(target,attempts_limit,workers,batch_size)<1:
        raise ValueError('target, attempt limit, workers and batch size must be positive')
    if restart:
        state_path.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    resumed=state_path.is_file()
    rng=random.Random(seed)
    digest=hashlib.sha256()
    rows=[]
    if resumed:
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('signature')!=signature:
            raise CandidateCheckpointError('candidate checkpoint inputs/configuration/code changed; use --restart-candidates for a fresh run')
        expected_bytes=int(state['committed_bytes'])
        if not partial.is_file() or partial.stat().st_size<expected_bytes:
            raise CandidateCheckpointError('candidate progress is missing or shorter than its checkpoint')
        with partial.open('rb') as handle:
            prefix=handle.read(expected_bytes)
        digest.update(prefix)
        if digest.hexdigest()!=state['committed_sha256']:
            raise CandidateCheckpointError('committed candidate progress hash mismatch')
        rows=[json.loads(line) for line in prefix.splitlines() if line.strip()]
        if len(rows)!=state['accepted']:
            raise CandidateCheckpointError('candidate progress count mismatch')
        if int(state['attempts'])>attempts_limit:
            raise CandidateCheckpointError('attempt limit is below already committed attempts')
        rng.setstate(_tuple_tree(state['rng_state']))
        # Discard only the uncommitted tail left by an interrupted append.
        with partial.open('r+b') as handle:handle.truncate(expected_bytes)
    else:
        if partial.exists():
            raise CandidateCheckpointError('candidate progress exists without a checkpoint; use --restart-candidates')
        partial.touch()
        state={'schema':SCHEMA,'signature':signature,'attempts':0,'accepted':0,
            'committed_bytes':0,'committed_sha256':digest.hexdigest(),
            'rng_state':rng.getstate(),'rejections':{},'examples':[],
            'committed_batches':0,'speculative_discarded':0,'status':'running'}
        _write_checkpoint(state_path,state)
    signatures={r['source_program_signature'] for r in rows}
    usage=Counter()
    for row in rows:usage.update(_predicates(row))
    rejections=Counter(state['rejections'])
    examples=list(state['examples'])
    executor=None
    aborted=True
    bar=_NullProgress()
    try:
        if len(rows)<target and state['attempts']<attempts_limit and workers>1:
            context=mp.get_context('fork' if os.name=='posix' else 'spawn')
            if os.name=='posix':
                _stop_progress_monitor()
            executor=ProcessPoolExecutor(max_workers=workers,mp_context=context,
                initializer=_process_initializer,initargs=(initializer,initargs))
            # Start the processes before creating a tqdm monitoring thread.
            executor.submit(_worker_ready).result()
        if progress:
            bar=tqdm(total=target,initial=len(rows),desc='Validated candidate reservoir')
            bar.set_postfix(accepted=len(rows),attempts=state['attempts'],workers=workers)
            if resumed:bar.write(f'Resuming {len(rows)}/{target} validated candidates at attempt {state["attempts"]}')
        while len(rows)<target and state['attempts']<attempts_limit:
            start=state['attempts']
            size=min(batch_size,attempts_limit-start)
            jobs=[make_proposal(start+i+1,rng,dict(usage)) for i in range(size)]
            results=list(executor.map(worker_function,jobs)) if executor else [worker_function(j) for j in jobs]
            new=[]
            for offset,result in enumerate(results):
                if len(rows)>=target:
                    state['speculative_discarded']+=len(results)-offset
                    break
                error=result.get('error')
                row=result.get('row')
                if not error and row['source_program_signature'] in signatures:
                    error='duplicate'
                if error:
                    rejections[error]+=1
                    if len(examples)<20:
                        examples.append({'attempt':jobs[offset]['attempt'],'category':error,
                                         'detail':result.get('detail',error)})
                    continue
                row['candidate_id']=f'candidate-{len(rows)+1:06d}'
                signatures.add(row['source_program_signature'])
                rows.append(row);new.append(row);usage.update(_predicates(row));bar.update(1)
            data=b''.join((json.dumps(row,ensure_ascii=False)+'\n').encode('utf-8') for row in new)
            with partial.open('ab') as handle:
                handle.write(data);handle.flush();os.fsync(handle.fileno())
            digest.update(data)
            state.update(attempts=start+size,accepted=len(rows),rng_state=rng.getstate(),
                committed_bytes=state['committed_bytes']+len(data),committed_sha256=digest.hexdigest(),
                rejections=dict(rejections),examples=examples,committed_batches=state['committed_batches']+1,
                status='complete' if len(rows)>=target else 'running')
            _write_checkpoint(state_path,state)
            bar.set_postfix(accepted=len(rows),attempts=state['attempts'],workers=workers)
        if len(rows)<target:
            state['status']='budget_exhausted'
            _write_checkpoint(state_path,state)
        aborted=False
    finally:
        bar.close()
        if executor is not None:
            if aborted:
                for process in list((executor._processes or {}).values()):process.terminate()
            executor.shutdown(wait=True,cancel_futures=aborted)
    return rows,{**{k:state[k] for k in ('attempts','rejections','examples','committed_batches','status','speculative_discarded')},
        'workers':workers,'proposal_batch_size':batch_size,'resumed':resumed,
        'checkpoint':str(state_path),'progress_output':str(partial)}


def _predicates(row):
    return {str(item['predicate']) for section in ('relation_bindings','attribute_bindings')
        for item in row.get('binding',{}).get(section,{}).values() if item.get('predicate')}


class _NullProgress:
    def update(self,*args,**kwargs):pass
    def set_postfix(self,*args,**kwargs):pass
    def close(self):pass


def _stop_progress_monitor():
    from tqdm.std import tqdm as base_tqdm
    for cls in (tqdm,base_tqdm):
        monitor=getattr(cls,'monitor',None)
        if monitor is not None:
            monitor.exit()
            cls.monitor=None


def _process_initializer(initializer,initargs):
    # Workers must not keep the parent's advisory checkpoint locks alive.
    for handle in _LOCK_HANDLES:handle.close()
    _LOCK_HANDLES.clear()
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if initializer is not None:initializer(*initargs)


def _worker_ready():
    return os.getpid()
