"""Omni-Decision-inspired Critic + typed Evidence Ledger, no planner training.

A stateless ledger is reconstructed from already delivered reports each round.
Critic extraction uses a cost-metered LLM; the reducer never trusts raw
report strings or private simulation facts as confirmed ground truth.
"""
from __future__ import annotations

import json
from collections import defaultdict


async def review(*, reports: list[dict], llm, options: dict) -> dict:
    from reliability_core import ask_json
    allowed_ids = {r["id"] for r in reports}
    prompt = (
        "You are an evidence Critic for an ongoing multi-agent investigation. "
        "Review ONLY the delivered reports. Extract claims, unknowns and "
        "internal inconsistencies. A claim being reported does NOT prove truth. "
        "Give subject, attribute, scope, value for every grounded claim and "
        "cite the report_id. Use status=reported or disputed, not 'verified' "
        "without a separate visible independent source. "
        "Reply with ONLY JSON: "
        '{"claims":[{"report_id":"r1.1","subject":"...","attribute":"...",'
        '"scope":"...","value":"...","status":"reported"}],'
        '"open_needs":["..."],"issues":["..."]}.\n'
        "Delivered history:\n" +
        json.dumps(reports, ensure_ascii=False)[:int(options.get("max_chars", 22000))]
    )
    verdict = await ask_json(llm, prompt)
    ledger: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for claim in verdict.get("claims", []):
        if not isinstance(claim, dict) or claim.get("report_id") not in allowed_ids:
            continue
        key = tuple(str(claim.get(k, "")).strip() for k in
                    ("subject", "attribute", "scope"))
        value = str(claim.get("value", "")).strip()
        if not all(key) or not value:
            continue
        ledger[key].append({"report_id": claim["report_id"], "value": value,
                            "status": claim.get("status", "reported")})
    risk = {r["id"]: 0.1 if r["status"] == "done" else 0.95 for r in reports}
    issues = [str(s)[:250] for s in verdict.get("issues", [])[:15]]
    conflicted = []
    ledger_json = []
    for key, claims in ledger.items():
        values = {x["value"].strip().casefold() for x in claims}
        conflict = len(values) > 1
        if conflict:
            for claim in claims:
                risk[claim["report_id"]] = max(risk[claim["report_id"]], 0.85)
            conflicted.append(" / ".join(key))
        if any(x["status"] == "disputed" for x in claims):
            for x in claims:
                risk[x["report_id"]] = max(risk[x["report_id"]], 0.65)
        ledger_json.append({"subject": key[0], "attribute": key[1],
                            "scope": key[2], "claims": claims,
                            "state": "conflict" if conflict else "unverified"})
    suggestions = []
    for rid, score in risk.items():
        if score >= 0.8:
            suggestions.append({"report_id": rid,
                                "action": "verify" if score < 0.95 else "retry",
                                "reason": "Conflicting/unavailable evidence; select independent source"})
    return {"evidence": [
        {"report_id": rid, "risk": score,
         "reason": "Ledger conflict or unresolved observation" if score >= 0.65
                   else "Reported claim, not independently verified"}
        for rid, score in risk.items()],
        "suggestions": suggestions,
        "issues": issues + ["Unresolved: " + x for x in conflicted],
        "details": {"ledger": ledger_json, "open_needs": verdict.get("open_needs", [])}}
