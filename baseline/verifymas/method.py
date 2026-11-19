"""VerifyMAS-inspired online prefix-level hypothesis verification.

The published VerifyMAS targets full failed-trajectory attribution.
This adapter deliberately prohibits access to future execution and treats
agent attribution as provisional hypotheses, not verified causality.
"""
from __future__ import annotations

import json


async def review(*, reports: list[dict], llm, options: dict) -> dict:
    from reliability_core import ask_json, clip
    source = json.dumps(reports, ensure_ascii=False)[:int(
        options.get("max_chars", 22000))]
    hypotheses = await ask_json(llm, (
        "Generate concrete, testable *error hypotheses* about already "
        "returned Sub-Agent reports. Do not inspect imagined future history "
        "or private truth. Candidate types: missing result, incomplete "
        "retrieval, contradictory summary, wrong entity, unsupported claim, "
        "cross-report conflict, incorrect computation. Every hypothesis "
        "must cite a report_id. Respond ONLY JSON: "
        '{"hypotheses":[{"report_id":"r1.1","error_type":"...",'
        '"claim":"The report contains the suspected error ..."}]}.\n'
        "History:\n" + source))
    ids = {r["id"] for r in reports}
    candidates = [
        {"report_id": h["report_id"],
         "error_type": str(h.get("error_type", "unknown")),
         "claim": str(h.get("claim", ""))[:800]}
        for h in hypotheses.get("hypotheses", [])
        if isinstance(h, dict) and h.get("report_id") in ids and h.get("claim")
    ][:int(options.get("max_hypotheses", 8))]
    if not candidates:
        return {"evidence": [
            {"report_id": r["id"], "risk": 0.5,
             "reason": "No testable error hypothesis; reliability unknown"}
            for r in reports], "suggestions": [], "issues": [],
            "details": {"hypotheses": []}}
    result = await ask_json(llm, (
        "Verify the following ERROR hypotheses based ONLY on history "
        "already delivered. label must be entailed (visible evidence "
        "supports ERROR), contradicted (visible evidence refutes ERROR) "
        "or neutral (insufficient visible evidence). Do not infer truth "
        "from missing evidence. Include exact candidate index. Respond "
        'ONLY JSON: {"verdicts":[{"index":0,"label":"neutral",'
        '"reason":"..."}]}.\n'
        + json.dumps({"history": reports, "hypotheses": candidates},
                     ensure_ascii=False)[:int(options.get("max_chars", 22000))]))
    decisions = {}
    for v in result.get("verdicts", []):
        if not isinstance(v, dict) or not isinstance(v.get("index"), int):
            continue
        i = v["index"]
        if 0 <= i < len(candidates) and v.get("label") in {
                "entailed", "contradicted", "neutral"}:
            decisions[i] = v
    risk = {r["id"]: 0.15 if r["status"] == "done" else 0.95 for r in reports}
    issues, suggestions, audit = [], [], []
    for i, item in enumerate(candidates):
        v = decisions.get(i, {"label": "neutral", "reason": "No verdict"})
        label = v["label"]
        rid = item["report_id"]
        if label == "entailed":
            risk[rid] = max(risk[rid], 0.9)
            issues.append(item["claim"])
            suggestions.append({"report_id": rid, "action": "verify",
                                "reason": str(v.get("reason", ""))[:300]})
        elif label == "neutral":
            risk[rid] = max(risk[rid], 0.5)
        audit.append({"candidate": item, "verdict": v})
    return {"evidence": [
        {"report_id": rid, "risk": score,
         "reason": "Error-hypothesis verification on visible prefix"}
        for rid, score in risk.items()],
        "suggestions": suggestions, "issues": issues,
        "details": {"hypotheses": audit}}
