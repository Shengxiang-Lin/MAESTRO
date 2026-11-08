"""Goal-first selection from grounded subtasks; answer-free question surface.

Use KQA operator distributions as motifs, not as a fake executable proof.
Dependent edges bind real entity IDs or exact normalized dataset labels.
"""
from __future__ import annotations

import json
import random
from collections import Counter

from envir.schema import TaskGraph, digest, display, normalize_label
from envir.store import EvidenceStore
from envir.subtasks import SubtaskFactory, unique_entity_labels


class TaskGraphComposer:
    def __init__(self, store: EvidenceStore, *, seed: int = 17):
        self.store = store
        self.factory = SubtaskFactory(store)
        self.rng = random.Random(seed)
        self.labels = unique_entity_labels(store)
        self.allowed = {
            r["pattern"] for r in store.db.execute(
                "SELECT DISTINCT pattern FROM skeletons"
            )
        }
        self.rejections = Counter()
        self._candidate_cache = {}

    def _optional_candidates(self, nodes: list[dict], *, limit: int = 2
                             ) -> tuple[str, ...]:
        """Select cached, replayed subtasks disconnected from required facts.

        Optional candidates are NEVER required Gold dependencies. They can
        later be surfaced as choices/distractors in a Main Agent runtime.
        """
        used_ids = {n["subtask_id"] for n in nodes}
        used_sources = {s for n in nodes for s in n.get("evidence_ids", ())}
        used_entities = {
            v for n in nodes
            for v in (n.get("inputs", {}).get("subject_id"),
                      n.get("answer", {}).get("entity_id")) if v
        }
        options = []
        for row in self.store.db.execute(
            "SELECT data FROM subtasks WHERE kind IN "
            "('search_relation','search_attribute') ORDER BY subtask_id LIMIT 150"
        ):
            candidate = json.loads(row["data"])
            if candidate["subtask_id"] in used_ids:
                continue
            if set(candidate["evidence_ids"]) & used_sources:
                continue
            if (candidate.get("inputs", {}).get("subject_id") in used_entities
                    or candidate.get("answer", {}).get("entity_id") in used_entities):
                continue
            if not self.factory.replay(candidate):
                continue
            options.append(candidate["subtask_id"])
            if len(options) >= limit:
                break
        return tuple(options)

    def _pack(self, pattern: str, nodes: list[dict], edges: list[dict],
              question: str, brief: dict, optional_ids: tuple[str, ...] = ()):
        if not optional_ids:
            optional_ids = self._optional_candidates(nodes)
        final = nodes[-1]
        graph = TaskGraph(
            task_id="envir-" + digest(pattern, [n["subtask_id"] for n in nodes], question),
            pattern=pattern,
            nodes=tuple(nodes),
            dependencies=tuple(edges),
            final_node=final["subtask_id"],
            question_brief=brief,
            question=question,
            gold_answer=final["answer"],
            optional_candidate_ids=optional_ids,
            validation={"status": "pending_replay"},
        )
        from envir.validation import check_graph
        issues = check_graph(graph.to_dict(), self.store)
        if issues:
            self.rejections.update(issues)
            return None
        result = graph.to_dict()
        has_labeled_input = any(
            node.get("verification") == "dataset_labeled" for node in nodes
        )
        result["validation"] = {
            "status": "mixed_provenance" if has_labeled_input else "verified",
            "all_answers_source_replay_verified": not has_labeled_input,
            "checks": ["source_replay_where_available", "dependency_binding",
                       "acyclic", "answer_composition"],
        }
        return result

    @staticmethod
    def _brief(goal: str, anchors: list[str], constraints: list[str],
               answer_type: str) -> dict:
        return {
            "final_goal": goal, "public_anchors": anchors,
            "necessary_constraints": constraints, "answer_type": answer_type,
            "private_answers_excluded": True,
        }

    def _rows(self, sql: str, *, limit: int = 1000):
        # Expensive connected-graph joins should run ONCE per composition
        # pass, not once per attempted question.
        key = (sql, limit)
        if key not in self._candidate_cache:
            self._candidate_cache[key] = self.store.db.execute(
                sql + " LIMIT ?", (limit,)
            ).fetchall()
        return list(self._candidate_cache[key])

    def chain(self) -> dict | None:
        if "chain" not in self.allowed:
            self.rejections["missing_kqa_chain_skeleton"] += 1
            return None
        # A's observed YAGO output is B's actual observed subject.
        candidates = self._rows("""
            SELECT a.subject AS start, a.predicate AS p1,
                   a.object AS middle, b.predicate AS p2
            FROM facts a
            JOIN facts b ON a.object=b.subject
            WHERE a.object_kind='entity'
              AND a.source_id IS NOT NULL AND b.source_id IS NOT NULL
              AND a.subject<>a.object AND a.predicate<>b.predicate
            ORDER BY a.subject,a.predicate,b.predicate
        """, limit=2500)
        self.rng.shuffle(candidates)
        for row in candidates:
            first = self.factory.yago(row["start"], row["p1"])
            second = self.factory.yago(row["middle"], row["p2"])
            if not first or not second:
                continue
            if first["answer"]["entity_id"] != second["inputs"]["subject_id"]:
                continue
            a, b = display(row["start"]), display(row["p1"])
            c = display(row["p2"])
            question = f"What is the {c} of the {b} of {a}?"
            # Explicitly tie the role, not just the string output.
            edges = [{"from": first["subtask_id"], "to": second["subtask_id"],
                      "input_field": "subject_id", "binding": "entity_id"}]
            return self._pack(
                "chain", [first, second], edges, question,
                self._brief(question, [a], [f"{b} of {a}", f"{c} of the result"],
                            second["output_type"]),
            )
        self.rejections["no_replayable_yago_chain"] += 1
        return None

    def parallel(self) -> dict | None:
        if "parallel" not in self.allowed:
            self.rejections["missing_kqa_parallel_skeleton"] += 1
            return None
        # Keep both public referents related by a real YAGO link, while
        # the answer needs two independent pages. This is a parallel
        # information request, NOT an unverified logical intersection.
        candidates = self._rows("""
            SELECT link.subject AS left_subject, link.object AS right_subject,
                   l.predicate AS lp, r.predicate AS rp,
                   link.predicate AS relation
            FROM facts link
            JOIN facts l ON l.subject=link.subject
            JOIN facts r ON r.subject=link.object
            WHERE link.object_kind='entity'
              AND l.object_kind='literal' AND r.object_kind='literal'
              AND l.source_id IS NOT NULL AND r.source_id IS NOT NULL
              AND l.predicate <> link.predicate
              AND link.subject <> link.object
            ORDER BY link.subject,link.object,l.predicate,r.predicate
        """, limit=3000)
        self.rng.shuffle(candidates)
        for row in candidates:
            left = self.factory.yago(row["left_subject"], row["lp"])
            right = self.factory.yago(row["right_subject"], row["rp"])
            if not left or not right:
                continue
            if set(left["evidence_ids"]) & set(right["evidence_ids"]):
                continue
            left_name, right_name = display(row["left_subject"]), display(row["right_subject"])
            left_prop, right_prop = display(row["lp"]), display(row["rp"])
            question = (
                f"For {left_name} and {right_name}, what are the "
                f"{left_prop} and {right_prop}, respectively?"
            )
            final = {
                "subtask_id": "combine-" + digest(left["subtask_id"], right["subtask_id"]),
                "template_id": "combine_two_results",
                "kind": "combine", "goal": question,
                "inputs": {
                    "left": "$" + left["subtask_id"] + ".answer",
                    "right": "$" + right["subtask_id"] + ".answer",
                },
                "output_type": "pair",
                "answer": {"value": {
                    "left": left["answer"]["value"],
                    "right": right["answer"]["value"]},
                    "type": "pair", "entity_id": None},
                "evidence_ids": sorted(set(left["evidence_ids"] + right["evidence_ids"])),
                "tool_events": [],
                "summary": "Combined the two independently retrieved properties.",
                "summary_trace": ["Joined the verified results from two public source pages."],
                "verification": "derived_from_verified_subtasks",
                "provenance": {"policy": "ordered_pair_without_extra_world_claim"},
            }
            edges = [{"from": left["subtask_id"], "to": final["subtask_id"],
                      "input_field": "left", "binding": "answer"},
                     {"from": right["subtask_id"], "to": final["subtask_id"],
                      "input_field": "right", "binding": "answer"}]
            return self._pack(
                "parallel", [left, right, final], edges, question,
                self._brief(question, [left_name, right_name],
                            [f"{left_prop} of {left_name}",
                             f"{right_prop} of {right_name}"], "pair"),
            )
        self.rejections["no_replayable_yago_parallel"] += 1
        return None

    def media_to_search(self) -> dict | None:
        if "chain" not in self.allowed:
            self.rejections["missing_kqa_chain_skeleton"] += 1
            return None
        records = self.store.db.execute(
            "SELECT data FROM subtasks WHERE kind='mmqa_qa' ORDER BY subtask_id"
        ).fetchall()
        rows = list(records)
        self.rng.shuffle(rows)
        for row in rows:
            media = json.loads(row["data"])
            # Strict one-to-one match; do not treat type compatibility as
            # proof of cross-dataset entity identity.
            subject = self.labels.get(normalize_label(media["answer"]["value"]))
            if not subject:
                continue
            facts = self.store.get_facts(subject)
            self.rng.shuffle(facts)
            for fact in facts:
                target = self.factory.yago(subject, fact["predicate"])
                if not target:
                    continue
                media_refs = [self.store.source(s) for s in media["evidence_ids"]]
                if not any(s and s["modality"] in {"image", "table"} for s in media_refs):
                    continue
                # Do not render the dataset's private answer in the user question.
                original = media["goal"].rstrip(" ?.")
                property_name = display(fact["predicate"])
                question = (
                    f"Using the supplied material, first determine: {original.lower()}; "
                    f"then what is that entity's {property_name}?"
                )
                edges = [{"from": media["subtask_id"], "to": target["subtask_id"],
                          "input_field": "subject_id", "binding": "unique_exact_label",
                          "resolved_entity_id": subject}]
                return self._pack(
                    "media_to_search", [media, target], edges, question,
                    self._brief(question, ["supplied MultiModalQA material"],
                                [original, f"{property_name} of the identified entity"],
                                target["output_type"]),
                )
        self.rejections["no_exact_mmqa_yago_bridge"] += 1
        return None

    def build(self, *, target: int = 20, max_attempts: int = 200,
              patterns: tuple[str, ...] = ("chain", "parallel", "media_to_search")
              ) -> list[dict]:
        if target < 1 or max_attempts < 1:
            raise ValueError("target and max_attempts must be positive")
        results = []
        signatures = set()
        for index in range(max_attempts):
            if len(results) >= target:
                break
            method = patterns[index % len(patterns)]
            if method not in {"chain", "parallel", "media_to_search"}:
                raise ValueError(f"unsupported pattern {method}")
            graph = getattr(self, method)()
            if graph is None:
                continue
            signature = tuple(n["subtask_id"] for n in graph["nodes"])
            if signature in signatures:
                self.rejections["duplicate_graph"] += 1
                continue
            signatures.add(signature)
            results.append(graph)
        return results
