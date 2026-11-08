"""YAGO subject-centered anchors for semantically coherent candidate binding.

The profile bank contains REAL co-occurring predicates from the same YAGO
subject. It is only teacher-side provenance: entity identifiers, literal
values, and original names never enter generated public tasks.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .type_solver import TypeConstraintSolver


class SemanticAnchorBank:
    """Sample real YAGO subject profiles and check joint binding support."""

    def __init__(self, profile_index, catalog, taxonomy, *, min_count=25):
        self.taxonomy = taxonomy
        allowed = {
            predicate
            for predicate, info in catalog.get("predicates", {}).items()
            if int(info.get("count", 0) or 0) >= min_count
            and info.get("kind") in {"relation", "attribute"}
        }
        self.profiles = []
        for row in profile_index.get("profiles", ()):
            predicates = tuple(sorted(set(row.get("observed_predicates", ())) & allowed))
            types = tuple(sorted(set(row.get("entity_types", ())) - {"*", "schema:Thing"}))
            if len(predicates) < 2 or not types:
                continue
            source = str(row.get("source_subject", ""))
            self.profiles.append({
                "id": hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
                "types": types,
                "predicates": predicates,
                "relation_count": sum(
                    catalog["predicates"][p].get("kind") == "relation"
                    for p in predicates
                ),
                "attribute_count": sum(
                    catalog["predicates"][p].get("kind") == "attribute"
                    for p in predicates
                ),
            })
        if not self.profiles:
            raise ValueError(
                "semantic_anchored requires YAGO profiles with at least two "
                "observed, eligible predicates; rebuild source priors or "
                "increase --yago-profile-limit"
            )

    @classmethod
    def from_file(cls, path, catalog, taxonomy):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), catalog, taxonomy)

    def sample(self, program, *, rng: random.Random, usage=None, balanced=False):
        """Choose a compatible motif before typed binding (no gold answers)."""
        ops = {node.op for node in program.nodes}
        has_relation = "RELATE" in ops
        has_attribute = any(
            op.startswith("FILTER_") and op != "FILTER_TYPE"
            or op in {"QUERY_ATTR", "SELECT_AMONG", "SELECT_BETWEEN"}
            for op in ops
        )
        eligible = [
            p for p in self.profiles
            if (not has_relation or p["relation_count"] > 0)
            and (not has_attribute or p["attribute_count"] > 0)
        ]
        # Some structurally unusual KoPL programs have no matching dual-kind
        # real subject. Never silently relax the final joint-witness check.
        eligible = eligible or self.profiles
        if not balanced:
            return rng.choice(eligible)
        # V2 balances anchor types and repeated predicates BEFORE World
        # Generation instead of wasting most samples on late caps.
        usage = usage or {}
        best = None
        for candidate in eligible:
            predicate_cost = sum(usage.get(p, 0) for p in candidate["predicates"])
            predicate_cost /= max(1, len(candidate["predicates"]))
            type_cost = min(
                usage.get("anchor_type:" + t, 0) for t in candidate["types"]
            )
            score = (type_cost * 4.0 + predicate_cost, rng.random())
            if best is None or score < best[0]:
                best = (score, candidate)
        return best[1]

    def can_anchor(self, program):
        """Fast feasibility test: one scope must own at least two predicate slots."""
        solver = TypeConstraintSolver.for_program(program, self.taxonomy)
        slots = defaultdict(set)
        for node in program.nodes:
            if node.op == "RELATE" and node.dependencies:
                direction = node.inputs[1] if len(node.inputs) > 1 else "forward"
                owner = node.dependencies[0] if direction == "forward" else node.node_id
                slots[solver.find(owner)].add(node.node_id)
            elif node.op in {
                "FILTER_STRING", "FILTER_NUMBER", "FILTER_YEAR", "FILTER_DATE",
                "QUERY_ATTR", "SELECT_BETWEEN", "SELECT_AMONG",
            }:
                if node.op == "QUERY_ATTR":
                    owners = node.dependencies[:1]
                elif node.op == "SELECT_BETWEEN":
                    owners = node.dependencies
                else:
                    owners = (node.node_id,)
                for owner in owners:
                    slots[solver.find(owner)].add(node.node_id)
        return any(len(items) >= 2 for items in slots.values())

    def evaluate(self, plan, anchor):
        """Require two chosen predicates on ONE logical entity scope.

        Relation predicates are attributed to their subject, not to the
        relation's object. A backward RELATE therefore belongs to its output
        scope. Attribute predicates belong to the filtered / queried entity.
        """
        program = plan.program
        solver = TypeConstraintSolver.for_program(program, self.taxonomy)
        predicates_by_root = defaultdict(set)
        nodes = program.node_map()
        for node_id, binding in plan.relation_bindings.items():
            node = nodes[node_id]
            direction = node.inputs[1] if len(node.inputs) > 1 else binding.direction
            subject_node = node.dependencies[0] if direction == "forward" else node_id
            predicates_by_root[solver.find(subject_node)].add(binding.predicate)
        for node_id, binding in plan.attribute_bindings.items():
            node = nodes[node_id]
            if node.op == "QUERY_ATTR":
                owners = node.dependencies[:1]
            elif node.op == "SELECT_BETWEEN":
                owners = node.dependencies
            else:
                owners = (node_id,)
            for owner in owners:
                predicates_by_root[solver.find(owner)].add(binding.predicate)

        members = defaultdict(list)
        for node_id in nodes:
            members[solver.find(node_id)].append(node_id)
        selected = set(anchor["predicates"])
        best = (0, "", ())
        for root, predicates in predicates_by_root.items():
            kinds = {plan.node_types.get(n) for n in members[root]}
            compatible = any(
                kind and self.taxonomy.compatible(kind, t)
                for kind in kinds for t in anchor["types"]
            )
            if not compatible:
                continue
            matches = tuple(sorted(predicates & selected))
            if len(matches) > best[0]:
                best = (len(matches), next(iter(sorted(k for k in kinds if k)), ""), matches)
        if best[0] < 2:
            return None
        unique = {
            binding.predicate
            for binding in (*plan.relation_bindings.values(),
                            *plan.attribute_bindings.values())
        }
        coverage = best[0] / max(1, len(unique))
        return {
            "policy": "yago_subject_joint_predicates_v1",
            "anchor_id": anchor["id"],
            "anchor_type": best[1],
            "matched_predicates": list(best[2]),
            "matched_count": best[0],
            "global_predicate_count": len(unique),
            "global_coverage": round(coverage, 5),
            "profile_predicate_count": len(selected),
            "evidence": "observed predicates of one real YAGO subject",
        }
