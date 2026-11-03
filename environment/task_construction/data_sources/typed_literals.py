"""Offline numeric literal/type join over the existing YAGO reservoir."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time
from .yago import TRIPLE_RE,TYPE_PREDICATES
from ..literals import parse_yago_literal,datatype_kind
from .taxonomy import TaxonomyIndex
def numeric_subjects(catalog):
    result=set()
    for info in catalog['predicates'].values():
        if info.get('kind')!='attribute':continue
        for sample in info.get('samples',[]):
            parsed=parse_yago_literal(sample['object'])
            if parsed is not None and parsed.kind=='number':result.add(sample['subject'])
    return result
def compile_typed_literal_support(catalog,types,*,taxonomy=None):
    rows_count=predicates=untyped=0
    for predicate,info in catalog['predicates'].items():
        if info.get('kind')!='attribute':continue
        numeric=[];seen=set();has_numeric=False
        for sample in info.get('samples',[]):
            literal=parse_yago_literal(sample['object'])
            if literal is None or literal.kind!='number':continue
            has_numeric=True;key=(sample['subject'],sample['object'])
            if key in seen:continue
            seen.add(key);direct=sorted(types.get(sample['subject'],()))
            if not direct:untyped+=1;continue
            supported=set(direct)
            if taxonomy is not None:
                for t in direct:supported.update(taxonomy.ancestors(t))
            numeric.append({'subject':sample['subject'],'subject_types':direct,'supported_types':sorted(supported),'raw':sample['object'],'value':literal.value,'unit':literal.unit,'datatype':literal.datatype})
        if has_numeric or any(datatype_kind(t)=='number' for t in info.get('schema',{}).get('range',[])):
            info['typed_literal_support']={'version':1,'samples':sorted(numeric,key=lambda r:(r['subject'],r['raw'])),'policy':'real_subject_type_join;deduplicated_subject_literal_pairs;no_parent_to_unobserved_subtype_inheritance'};rows_count+=len(numeric);predicates+=1
    catalog['typed_literal_support']={'version':1,'numeric_predicates':predicates,'numeric_rows':rows_count,'untyped_numeric_rows':untyped,'subject_type_join':'rdf:type_on_existing_reservoir_subjects','ancestor_expansion':taxonomy is not None};return catalog
def augment_file(input_path,output_path,taxonomy_path,*,facts_path=None):
    target=Path(output_path)
    if target.exists():raise FileExistsError('refuse to overwrite a catalog pinned by an earlier experiment')
    catalog=json.loads(Path(input_path).read_text(encoding='utf-8'));facts=Path(facts_path or catalog['source']);wanted={s.encode('utf-8') for s in numeric_subjects(catalog)};types=defaultdict(set);hasher=sha256();scanned=0;last=time.monotonic();print(json.dumps({'numeric_subjects':len(wanted),'facts_bytes':facts.stat().st_size}),flush=True)
    with facts.open('rb') as reader:
        for raw in reader:
            hasher.update(raw);scanned+=len(raw);parts=raw.split(None,2)
            if len(parts)!=3 or parts[0] not in wanted or parts[1] not in {b'rdf:type',b'a'}:
                if time.monotonic()-last>30:print(json.dumps({'bytes_scanned':scanned,'typed_subjects':len(types)}),flush=True);last=time.monotonic()
                continue
            match=TRIPLE_RE.match(raw.decode('utf-8'))
            if match and match.group(2) in TYPE_PREDICATES:types[match.group(1)].add(match.group(3))
    compile_typed_literal_support(catalog,types,taxonomy=TaxonomyIndex.from_json(taxonomy_path));catalog['typed_literal_support'].update(facts_source=str(facts.resolve()),facts_sha256=hasher.hexdigest(),input_catalog_sha256=sha256(Path(input_path).read_bytes()).hexdigest(),taxonomy_sha256=sha256(Path(taxonomy_path).read_bytes()).hexdigest());target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(catalog,ensure_ascii=False,separators=(',',':')),encoding='utf-8');tmp.replace(target);print(json.dumps(catalog['typed_literal_support'],ensure_ascii=False),flush=True);return catalog['typed_literal_support']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--taxonomy',required=True);p.add_argument('--facts');a=p.parse_args();augment_file(a.input,a.output,a.taxonomy,facts_path=a.facts)
if __name__=='__main__':main()
