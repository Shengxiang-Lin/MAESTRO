"""Deterministic observed-edge and shared-entity motif priors."""
from __future__ import annotations
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
import json
from typing import Any,Mapping
from ..binding.semantic_support import MIN_EMPIRICAL_TYPE_MASS,empirical_type_counts
class ObservedSupport:
    def __init__(self,catalog:Mapping[str,Any],taxonomy):
        self.catalog=catalog;self.taxonomy=taxonomy;self._edges={};self._motifs={};self._motif_rows=None;self._port_groups=defaultdict(list);self._port_mass={};self._typed_numeric_profiles={}
    def _supports(self,observed,actual):return bool(actual and self.taxonomy.is_subtype(observed,actual))
    @staticmethod
    def _decision(mass,support,*,source):
        status="supported" if support else "unsupported" if mass>=MIN_EMPIRICAL_TYPE_MASS else "unknown";return {"status":status,"accepted":status!="unsupported","observed_mass":mass,"supporting_mass":support,"source":source,"evidence_strength":"multiple_observations" if support>=MIN_EMPIRICAL_TYPE_MASS else "weak_positive" if support else status}
    def edge(self,predicate,subject_type,object_type=None):
        key=(predicate,subject_type,object_type)
        if key in self._edges:return dict(self._edges[key])
        info=self.catalog.get("predicates",{}).get(predicate,{});rows=info.get("observed_edge_types")
        if rows is not None:
            mass=sum(int(r[2]) for r in rows);support=sum(int(r[2]) for r in rows if any(self._supports(t,subject_type) for t in r[0]) and (object_type is None or any(self._supports(t,object_type) for t in r[1])));result=self._decision(mass,support,source="observed_joint_edges");exposure=sum(int(r[2]) for r in rows if any(self._supports(t,subject_type) for t in r[0]));result.update(typed_subject_exposure=exposure,conditional_support_ratio=support/exposure if exposure else None)
        elif object_type is None:
            counts=empirical_type_counts(info,"subject");result=self._decision(sum(n for _,n in counts),sum(n for t,n in counts if self._supports(t,subject_type)),source="legacy_subject_marginal")
        else:
            sides=[empirical_type_counts(info,side) for side in ("subject","object")];unsupported=any(sum(n for _,n in rows)>=MIN_EMPIRICAL_TYPE_MASS and not any(self._supports(t,actual) for t,_ in rows) for rows,actual in zip(sides,(subject_type,object_type)));result={"status":"unsupported" if unsupported else "unknown","accepted":not unsupported,"observed_mass":0,"supporting_mass":0,"source":"legacy_marginals_no_joint_evidence"}
        self._edges[key]=result;return dict(result)
    def motif(self,center_type,ports):
        if self._motif_rows is None:
            self._motif_rows=defaultdict(list)
            for row in self.catalog.get("observed_support",{}).get("motifs",()):self._motif_rows[tuple(row["ports"])].append(row)
            for pair,rows in self._motif_rows.items():
                for port in pair:self._port_groups[port].append(rows)
        ports=tuple(sorted(ports));key=(center_type,ports)
        if key not in self._motifs:
            rows=self._motif_rows.get(ports,());mass=sum(int(row["count"]) for row in rows);support=sum(int(row["count"]) for row in rows if any(self._supports(t,center_type) for t in row["center_types"]));exposure=[]
            for port in ports:
                port_key=(center_type,port)
                if port_key not in self._port_mass:self._port_mass[port_key]=max((sum(int(row["count"]) for row in group if any(self._supports(t,center_type) for t in row["center_types"])) for group in self._port_groups[port]),default=0)
                exposure.append(self._port_mass[port_key])
            result=self._decision(mass,support,source="observed_shared_entity_motif");result["typed_port_exposure_lower_bounds"]=exposure
            if result["status"]=="unsupported" and min(exposure)<MIN_EMPIRICAL_TYPE_MASS:result.update(status="unknown",accepted=True,evidence_strength="unknown",decision="sparse_typed_motif_exposure")
            self._motifs[key]=result
        return dict(self._motifs[key])
    def world(self,world):
        edges=[];incident=defaultdict(set);slots=defaultdict(set)
        for fact in sorted(world.facts,key=lambda f:f.fact_id):
            if fact.predicate.startswith("o3:"):continue
            subject_type=world.entities.get(fact.subject);value=json.dumps(fact.object,sort_keys=True,ensure_ascii=False)
            if fact.object_kind=="number":
                try:value=str(Decimal(str(fact.object)).normalize())
                except InvalidOperation:pass
            slots[(fact.subject,fact.predicate)].add((fact.object_kind,value));object_type=world.entities.get(str(fact.object)) if fact.object_kind=="entity" else None;edges.append({"fact_id":fact.fact_id,"predicate":fact.predicate,"subject_type":subject_type,"object_type":object_type,**self.edge(fact.predicate,subject_type,object_type)});incident[fact.subject].add("out|"+fact.predicate)
            if fact.object_kind=="entity":incident[str(fact.object)].add("in|"+fact.predicate)
        motifs=[]
        for entity,ports in sorted(incident.items()):
            ordered=sorted(ports)
            for i,left in enumerate(ordered):
                for right in ordered[i+1:]:motifs.append({"entity_id":entity,"center_type":world.entities[entity],"ports":[left,right],**self.motif(world.entities[entity],(left,right))})
        reasons=[];cardinality=[]
        for (subject,predicate),values in sorted(slots.items()):
            maximum=self.catalog.get("predicates",{}).get(predicate,{}).get("schema",{}).get("max_count")
            if isinstance(maximum,int) and not isinstance(maximum,bool) and len(values)>maximum:cardinality.append({"entity_id":subject,"predicate":predicate,"distinct_value_count":len(values),"schema_max_count":maximum})
        if cardinality:reasons.append("schema_max_count_exceeded")
        if any(r["status"]=="unsupported" for r in edges):reasons.append("observed_edge_unsupported")
        if any(r["status"]=="unsupported" for r in motifs):reasons.append("observed_motif_unsupported")
        report={"version":2,"accepted":not reasons,"reasons":reasons,"cardinality_violations":cardinality,"physical_profile_unit_violations":[],"prior_version":self.catalog.get("observed_support",{}).get("version"),"edge_status_counts":dict(Counter(r["status"] for r in edges)),"motif_status_counts":dict(Counter(r["status"] for r in motifs)),"edge_strength_counts":dict(Counter(r.get("evidence_strength",r["status"]) for r in edges)),"motif_strength_counts":dict(Counter(r.get("evidence_strength",r["status"]) for r in motifs)),"edges":edges,"motifs":motifs}
        if self.catalog.get("typed_literal_support",{}).get("version")==1:
            from ..semantic.typed_profile import typed_fact_audit
            typed=typed_fact_audit(world,self.catalog,self._typed_numeric_profiles);report["typed_numeric_profiles"]=typed
            if not typed["accepted"]:report["reasons"]=list(report["reasons"])+sorted({v["reason"] for v in typed["violations"]});report["accepted"]=False
        return report
