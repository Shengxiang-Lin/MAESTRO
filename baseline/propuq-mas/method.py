"""PropUQ-MAS-inspired propagated uncertainty over a visible message DAG.

Local risk u_v and uptake alpha_{parent,v} can be supplied from a separate
estimator or inferred from visible history via an LLM; no oracle DAG is used.
The propagation equation is the published noisy-OR family:
r_v = 1 - (1-u_v)*product(1-alpha_pv*r_p).
"""
from __future__ import annotations

import json
import math


def propagate(nodes: list[dict], local: dict, edges: list[dict]) -> dict:
    from reliability_core import clip
    index = {n["id"]: i for i, n in enumerate(nodes)}
    parents = {n["id"]: [] for n in nodes}
    for e in edges:
        a, b = e.get("parent"), e.get("child")
        if a in index and b in index and index[a] < index[b]:
            parents[b].append((a, clip(e.get("uptake", 0))))
    risks = {}
    for node in nodes:
        rid = node["id"]
        u = clip(local[rid])
        survival = 1 - u
        for ancestor, alpha in parents[rid]:
            survival *= 1 - alpha * risks[ancestor]
        risks[rid] = clip(1 - survival)
    return risks


async def review(*, reports: list[dict], llm, options: dict) -> dict:
    from reliability_core import ask_json, clip
    raw_local = options.get("local_uncertainty")
    raw_edges = options.get("edges")
    # External estimates have a fully explicit interface. A costed LLM
    # estimator is used only if either component is not supplied.
    if raw_local is None or raw_edges is None:
        prompt = (
            "Estimate uncertainty in the following already-delivered "
            "Sub-Agent reports without knowledge of ground truth. "
            "Each local risk is in [0,1] and based on status, contradictions, "
            "source insufficiency and stated uncertainty. For any dependency "
            "edge, cite an earlier report clearly used by a later one; do "
            "NOT infer dependency merely from agreement. Uptake is the "
            "probability the child relies on parent. Respond ONLY JSON: "
            '{"local":{"r1.1":0.2},"edges":[{"parent":"r1.1",'
            '"child":"r2.1","uptake":0.8}]}.\n'
            + json.dumps(reports, ensure_ascii=False)[:int(
                options.get("max_chars", 22000))])
        estimate = await ask_json(llm, prompt)
        if raw_local is None:
            raw_local = estimate.get("local", {})
        if raw_edges is None:
            raw_edges = estimate.get("edges", [])
    local = {}
    for report in reports:
        rid = report["id"]
        if rid not in raw_local:
            raise ValueError(f"No local uncertainty for {rid}")
        local[rid] = clip(raw_local[rid])
        if report["status"] != "done":
            local[rid] = max(local[rid], 0.95)
    index = {r["id"]: i for i, r in enumerate(reports)}
    edges = []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        p, c = e.get("parent"), e.get("child")
        if p in index and c in index and index[p] < index[c]:
            edges.append({"parent": p, "child": c,
                          "uptake": clip(e.get("uptake", 0))})
    result = propagate(reports, local, edges)
    threshold = float(options.get("threshold", 0.55))
    return {
        "evidence": [{"report_id": rid, "risk": risk,
                      "reason": f"local={local[rid]:.3f}; propagation-aware={risk:.3f}"}
                     for rid, risk in result.items()],
        "suggestions": [
            {"report_id": rid, "action": "verify",
             "reason": "High uncertainty after parent-message risk propagation"}
            for rid, risk in result.items() if risk >= threshold],
        "issues": [],
        "details": {"local_uncertainty": local, "edges": edges,
                    "propagated_uncertainty": result,
                    "estimator": "provided" if
                    options.get("local_uncertainty") is not None and
                    options.get("edges") is not None else "LLM-visible-prefix"}
    }
