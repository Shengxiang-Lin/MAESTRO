"""HASSUM-inspired semantic-uncertainty routing.

For a faithful worker-level comparison, provide independently sampled worker
answers for each report via options.samples_by_report. When missing, this
adapter explicitly uses LLM *text-only* resampling (not original media
re-execution), which must be reported separately in experimental tables.
"""
from __future__ import annotations

import json
import math


def semantic_entropy(groups: list[list[int]], n: int) -> float:
    if n < 2:
        return 0.0
    h = -sum((len(g)/n)*math.log(len(g)/n) for g in groups if g)
    return h / math.log(n)


async def _sample(llm, report: dict) -> str:
    from reliability_core import ask_json
    prompt = (
        "Provide ONE independent concise candidate answer to this Sub-Agent "
        "task, from the supplied TEXT ONLY. No access to original media or "
        "tools is implied; do not claim to inspect them. You may say unknown. "
        "Return ONLY JSON {\"answer\":\"...\"}.\nTask: " +
        report["task"][:900] + "\nVisible report: " + report["result"][:1300])
    result = await ask_json(llm, prompt)
    return str(result["answer"])


async def _relation(llm, a: str, b: str) -> tuple[bool, float]:
    from reliability_core import ask_json, clip
    verdict = await ask_json(llm, (
        "For two proposed answers to the SAME question, are they semantically "
        "equivalent in facts (not merely similar wording)? Return ONLY JSON "
        '{"equivalent":true,"similarity":0.9} (similarity 0..1).\n'
        + json.dumps({"a": a, "b": b}, ensure_ascii=False)))
    return (verdict.get("equivalent") is True, clip(verdict["similarity"]))


async def _groups_and_density(llm, samples: list[str]):
    groups: list[list[int]] = []
    pair_scores = []
    for i, answer in enumerate(samples):
        placed = False
        for group in groups:
            match, score = await _relation(llm, answer, samples[group[0]])
            pair_scores.append(score)
            if match:
                group.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    # Score of sampled answer against the original-worker answer is handled
    # separately; mean semantic similarity is an online density proxy.
    density = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    return groups, density


async def review(*, reports: list[dict], llm, options: dict) -> dict:
    from reliability_core import clip
    sampled = options.get("samples_by_report", {})
    k = int(options.get("num_samples", 4))
    if not 2 <= k <= 16:
        raise ValueError("num_samples must be between 2 and 16")
    weight = clip(options.get("entropy_weight", 0.6))
    evidence, suggestions, meta = [], [], {}
    for report in reports:
        if report["status"] != "done":
            evidence.append({"report_id": report["id"], "risk": 0.95,
                             "reason": "Incomplete worker result"})
            suggestions.append({"report_id": report["id"], "action": "retry",
                                "reason": "No completed answer"})
            continue
        answers = sampled.get(report["id"])
        origin = "worker-samples"
        if answers is None:
            answers = [await _sample(llm, report) for _ in range(k)]
            origin = "text-only-resampling"
        if not isinstance(answers, list) or len(answers) < 2:
            raise ValueError("HASSUM requires >=2 candidate answers per report")
        answers = [str(x) for x in answers]
        groups, density = await _groups_and_density(llm, answers)
        entropy = semantic_entropy(groups, len(answers))
        risk = clip(weight * entropy + (1 - weight) * (1 - density))
        evidence.append({"report_id": report["id"], "risk": risk,
                         "reason": f"semantic_entropy={entropy:.3f},density={density:.3f}"})
        if risk >= float(options.get("threshold", 0.5)):
            suggestions.append({"report_id": report["id"], "action": "verify",
                                "reason": "High semantic uncertainty"})
        meta[report["id"]] = {"entropy": entropy, "density_proxy": density,
                              "sample_origin": origin, "group_sizes":
                              [len(g) for g in groups], "sample_count": len(answers)}
    return {"evidence": evidence, "suggestions": suggestions,
            "issues": [], "details": meta}
