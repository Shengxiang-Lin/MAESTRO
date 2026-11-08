"""Independent answer/source/dependency checks for task graphs."""
from __future__ import annotations

from collections import defaultdict, deque

from envir.schema import display, normalize_label
from envir.store import EvidenceStore
from envir.subtasks import SubtaskFactory


def check_graph(graph: dict, store: EvidenceStore) -> list[str]:
    issues = []
    nodes = graph.get("nodes") or []
    ids = [n.get("subtask_id") for n in nodes]
    if not nodes or not all(ids) or len(ids) != len(set(ids)):
        return ["invalid_nodes"]
    by_id = {n["subtask_id"]: n for n in nodes}
    root = graph.get("final_node")
    if root not in by_id:
        issues.append("missing_final_node")
    incoming = defaultdict(list)
    children = defaultdict(list)
    indegree = dict.fromkeys(ids, 0)
    for edge in graph.get("dependencies", []):
        parent, child = edge.get("from"), edge.get("to")
        if parent not in by_id or child not in by_id or parent == child:
            issues.append("invalid_dependency")
            continue
        incoming[child].append((parent, edge))
        children[parent].append(child)
        indegree[child] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    ordered = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(ordered) != len(nodes):
        issues.append("cyclic_dependencies")
    if root in by_id:
        ancestors = {root}
        todo = [root]
        while todo:
            current = todo.pop()
            for parent, _ in incoming[current]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    todo.append(parent)
        if ancestors != set(ids):
            issues.append("unused_required_subtasks")
        if graph.get("gold_answer") != by_id[root].get("answer"):
            issues.append("final_answer_drift")
    factory = SubtaskFactory(store)
    for node in nodes:
        kind = node.get("kind")
        evidence_ids = node.get("evidence_ids") or []
        for source_id in evidence_ids:
            if store.source(source_id) is None:
                issues.append("missing_public_source")
        if kind in {"search_relation", "search_attribute"}:
            if node.get("verification") != "source_replay_verified" or not factory.replay(node):
                issues.append("yago_source_replay_failed")
        elif kind == "mmqa_qa":
            if (node.get("verification") != "dataset_labeled"
                    or not evidence_ids or not node.get("answer", {}).get("value")):
                issues.append("mmqa_label_or_source_missing")
        elif kind == "combine":
            parents = {edge["input_field"]: by_id[p]["answer"]["value"]
                       for p, edge in incoming.get(node["subtask_id"], [])
                       if edge.get("binding") == "answer"}
            if (set(parents) != {"left", "right"}
                    or node.get("answer", {}).get("value") != parents
                    or len(set(evidence_ids)) < 2):
                issues.append("combine_inputs_not_replayed")
            if node.get("verification") != "derived_from_verified_subtasks":
                issues.append("combine_verification_mismatch")
        else:
            issues.append("unsupported_subtask_kind")
    for edge in graph.get("dependencies", []):
        parent, child = edge.get("from"), edge.get("to")
        if parent not in by_id or child not in by_id:
            continue
        up, down = by_id[parent], by_id[child]
        binding = edge.get("binding")
        field = edge.get("input_field")
        if binding == "entity_id":
            if (field != "subject_id" or not up["answer"].get("entity_id")
                    or up["answer"]["entity_id"] != down["inputs"].get(field)):
                issues.append("entity_binding_mismatch")
        elif binding == "unique_exact_label":
            subject = down["inputs"].get(field)
            if (field != "subject_id" or not subject
                    or normalize_label(up["answer"].get("value"))
                    != normalize_label(display(subject))
                    or subject != edge.get("resolved_entity_id")):
                issues.append("cross_dataset_entity_alignment_mismatch")
        elif binding == "answer":
            if down.get("kind") != "combine":
                issues.append("unexpected_answer_binding")
        else:
            issues.append("unsupported_dependency_binding")
    if not graph.get("question") or not graph.get("question_brief", {}).get(
        "private_answers_excluded", False
    ):
        issues.append("invalid_public_question")
    if graph.get("pattern") == "parallel":
        base = [n for n in nodes if n.get("kind") != "combine"]
        if len(base) != 2 or set(base[0]["evidence_ids"]) & set(base[1]["evidence_ids"]):
            issues.append("parallel_sources_not_independent")
    return sorted(set(issues))


def corpus_statistics(graphs: list[dict]) -> dict:
    from collections import Counter
    patterns = Counter(g["pattern"] for g in graphs)
    sources = Counter(
        src for graph in graphs for node in graph["nodes"]
        for src in node.get("evidence_ids", ())
    )
    return {"graphs": len(graphs), "patterns": dict(patterns),
            "distinct_evidence_sources": len(sources),
            "repeated_sources": sum(value > 1 for value in sources.values()),
            "max_source_reuse": max(sources.values(), default=0)}
