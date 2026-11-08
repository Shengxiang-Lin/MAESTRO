"""V4 optional LLM Question Author -> local Critic -> bounded revision -> accept.

Adapted from environment/question_generation/{realization,question_review}.py,
but unlike Stage 2's KoPL-specific question review, this module checks that a
natural user Question is still implied by the *same immutable Macro Task
work goals, deliverable and dependencies*.

Only a sanitized Task/Question contract is passed to either LLM. The actual
DAG IDs, reference waves, KoPL programs, private evidence, Feedback and world
facts remain teacher-private. No claim of formal semantic equivalence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from environment.question_generation.backend import BackendError
from .v4_taskfirst import author_question, task_level_brief


AUTHOR_SYSTEM = """You are the Author for a realistic multi-source, potentially
multimodal benchmark question, in the style of a natural request from a
person, not a compiler trace or an instruction to an agent scheduler.

The immutable TASK CONTRACT was composed before this Question. Preserve its
whole situation, scope, main user intent, required final deliverable, and
enough clues so that the SAME underlying independent/dependent macro work
goals remain relevant and sufficient to address the user request.

Write a genuinely natural, self-contained user Question. Vary its rhetorical
shape with the supplied style hint (direct question, brief scenario-based
request, comparison or investigation). Do not use the same canned opening
across records. Questions may be one or two coherent sentences; do not force
an enumerated source checklist. A user does NOT name planned subtasks, specify
their dependency order or tell an assistant how to delegate.

The macro work descriptions are SEMANTIC DESIGN INPUTS, not passages to copy
and not additional goals to invent. Do not add or remove requirements, change
entity owners, reverse comparisons, invent dates/numbers, claim evidence was
already observed, invent source contents, or claim media is attached. Mention
media only if the public resource modality supports it.

CRITICAL: Generic types must remain generic. If the case says "laboratory
instrument", never choose "spectrometer", "microscope" or another subtype.
If it says "vehicle" do not invent a truck, forklift or driver. If it says
"records" do not invent surveillance cameras, gate logs, staff testimony,
experiment logs or individual witnesses. If the case is not explicitly
ongoing, do not say "currently", "still", "now" or "ongoing". Do not turn
ranking RESPONSES TO reservations into ranking the reservations themselves.
Do not prescribe "first investigate ... then rank"; express the overall
information need, NOT the internal agent workflow.

Avoid copied boilerplate, particularly "documented findings", "Explain
which documented findings support your assessment", and repetitious
"Based on X, Y and Z". The user may ask for evidence but should sound
like a person, not a report specification. Preserve support requirements
naturally ("What supports the recommendation?", "Why?", "using the relevant
records", etc.). Prefer the supplied style_seed_question for readable
phrasing, treating the original reference as the semantic authority.

If given a previous_draft and reviewer_feedback, make the smallest necessary
changes to fix the specific flaws, preserving all unaffected meaning.
Return JSON {"question":"..."} and no other text."""

CRITIC_SYSTEM = """You are an independent local Question Critic for a
Task-first multi-agent orchestration dataset.

Compare the proposed natural Question with the reference AND immutable
task-level semantic contract. PASS only if: (1) the same overall situation,
scope, answer/work-product obligation, constraints and permissible sources are
preserved; (2) it still naturally motivates the SAME macro work goals and
their semantic handoffs, without adding or eliminating essential work; and
(3) it reads like a real user's request rather than a templated plan.

The contract's supporting work goals need not be explicitly enumerated in the
Question; implicit necessity is fine. Do NOT require the original sentence
order or exact words. Do not demand meaningless extra details. Do not invent
hidden evidence or infer facts not in the contract.

Before any PASS, perform a candidate-only SEMANTIC DELTA REVIEW:
(a) list any NEW specific entity, instrument, source or record type. If a
generic "laboratory instrument" becomes "spectrometer", that is unsupported
specificity even when plausible; it must be REVISED, not celebrated as clarity.
(b) identify temporal/ongoing-state additions ("currently", "still", "now").
(c) compare the OBJECT receiving each action. Ranking response priorities
is not ranking reservation requests. (d) detect instructions that expose
internal work ordering ("first investigate, then rank"). (e) compare data
sources, deliverables, negative conditions and constraints.
Every candidate-only factual detail needs clear textual support from the
original reference or the immutable public contract; don't treat common
domain knowledge, guesses, or possible evidence as authorization.

Reject forced template phrases, duplicated evidence requests and robotic
operator lists as poor naturalness. Do not reject harmless paraphrases,
pronoun-based location references ("our museum"), or requests that rely on
named records without literally saying "documented findings".

When problems exist, report up to THREE local differences with an exact
minimal fix. If unable to determine equivalence, choose uncertain; do not
rubber-stamp. Cite the exact unsupported candidate phrase in candidate_excerpt.

Return strict JSON:
{"verdict":"pass|revise|uncertain",
 "naturalness":"acceptable|needs_polish",
 "summary":"...",
 "differences":[{"reference_excerpt":"...", "candidate_excerpt":"...",
                 "problem":"...", "minimal_fix":"..."}]}
PASS must have naturalness=acceptable and differences=[]."""

AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}
DIFF_SCHEMA = {
    "type": "object",
    "properties": {key: {"type": "string"} for key in (
        "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"
    )},
    "required": [
        "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"
    ],
    "additionalProperties": False,
}
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["pass", "revise", "uncertain"]},
        "naturalness": {"enum": ["acceptable", "needs_polish"]},
        "summary": {"type": "string"},
        "differences": {"type": "array", "items": DIFF_SCHEMA},
    },
    "required": ["verdict", "naturalness", "summary", "differences"],
    "additionalProperties": False,
}

STOP = {
    "with", "this", "that", "their", "have", "been", "which", "about", "from",
    "some", "given", "reported", "issue", "around", "after", "before",
    "potential", "affecting", "associated", "there", "where", "into",
    "relating", "review", "records", "current", "available", "relevant",
}
STEMS = {
    "impact": ("impact", "affect", "consequen", "effect", "implication"),
    "compliance": ("complian", "require", "rule", "policy", "eligib",
                   "authoriz", "permit", "satisf", "violat"),
    "reconciliation": ("reconcil", "discrepan", "conflict", "inconsisten",
                       "differen", "match", "account", "align"),
    "mitigation": ("mitigat", "respond", "response", "remed", "recommend",
                   "action", "address", "solution"),
    "prioritization": ("priorit", "urgent", "rank", "first", "order",
                       "sequence"),
    "comparison": ("compar", "differen", "relative", "versus", "better",
                   "which", "more", "less"),
    "feasibility": ("feasib", "possible", "practic", "implement",
                    "workable", "constraint"),
    "timeline": ("timeline", "chronolog", "sequence", "when", "time",
                 "order", "occur"),
}
STYLE_HINTS = {
    "composition": [
        "A compact cross-source question whose final goal emerges from several clues",
        "An authentic contextual request with a natural reason to consult several sources",
        "A direct investigation request; all relationships are implicit, not an enumerated workflow",
    ],
    "intersection": [
        "A natural request to reconcile overlapping conditions and narrow the relevant cases",
        "A short context plus a focused question about qualifying cases and supporting evidence",
        "A fact-finding request that relies on multiple records without listing each agent's steps",
    ],
    "comparison": [
        "A practical comparative decision question framed around the same alternatives",
        "A natural contrast-oriented question with a defensible evidence requirement",
        "An applied assessment asking which interpretation or course is better supported",
    ],
    "aggregation": [
        "A concise synthesis question about the scope or cumulative effects of the situation",
        "A contextual request that asks for a consolidated assessment from several sources",
    ],
    "lookup": [
        "A direct real-world information request with precise scope and final deliverable",
        "A two-sentence investigation context followed by one focused request",
    ],
}


def _fingerprint(value) -> str:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def reference_question(scene: dict) -> str:
    """Reconstruct the immutable baseline from Task goals, NOT from a candidate."""
    meta = scene["task_plan_provenance"]
    brief = task_level_brief(scene)
    return author_question(brief, ordinal=meta["ordinal"], seed=meta["seed"])


def public_task_contract(scene: dict) -> dict:
    """Author/critic input projection; NO Task IDs, graph topology or Evidence."""
    brief = task_level_brief(scene)
    tasks = scene["macro_graph"]["tasks"]
    by_id = {t["task_id"]: t for t in tasks}
    goal_descriptions = []
    for task in tasks:
        item = {
            "work_goal": task["goal"],
            "deliverable": task["deliverable"],
            "work_type": task["work_type"],
        }
        if item not in goal_descriptions:
            goal_descriptions.append(item)
    handoffs = []
    for edge in scene["macro_graph"]["edges"]:
        link = {
            "prerequisite_goal": by_id[edge["from"]]["goal"],
            "downstream_goal": by_id[edge["to"]]["goal"],
            "information_needed": edge["type"],
        }
        if link not in handoffs:
            handoffs.append(link)
    return {
        "scenario": {
            "domain": brief["domain"],
            "setting": brief["setting"],
            "issue": brief["issue"],
            "user_role": brief["user_role"],
        },
        "main_request": {
            "intent": brief["intent"],
            "final_work_goal": brief["final_work_goal"],
            "final_deliverable": brief["final_deliverable"],
            "resource_categories": brief["public_resource_categories"],
        },
        "supporting_work_goals": goal_descriptions,
        "semantic_handoffs": handoffs,
        "public_modalities": sorted({
            s["modality_hint"] for s in
            scene.get("public", {}).get("available_source_descriptions", [])
        }),
    }


def style_hint(scene: dict) -> str:
    comp = scene["source_priors"]["MultiModalQA"].get("composition_family", "lookup")
    choices = STYLE_HINTS.get(comp, STYLE_HINTS["lookup"])
    seed = int(_fingerprint([scene["task_id"], "question_style"])[:12], 16)
    return choices[seed % len(choices)]



# High-precision additions that materially narrow otherwise generic
# synthetic cases. These are NOT a replacement for the semantic Critic.
# Only unsupported *candidate-only* terms are rejected.
CONCRETE_DETAILS = {
    "spectrometer", "spectrophotometer", "oscilloscope", "microscope",
    "centrifuge", "chromatograph", "thermometer", "mass-spectrometer",
    "forklift", "semi-truck", "tanker", "van", "lorry",
    "gate log", "gate logs", "maintenance log", "maintenance logs",
    "experiment log", "experiment logs", "surveillance camera",
    "surveillance cameras", "witness accounts", "witness statements",
    "inspection photographs", "event posters", "session recordings",
}
CURRENT_STATE_TERMS = (
    "currently", "presently", "right now", "still", "ongoing",
    "at the moment", "now occupying", "now operating",
)
SETTING_ALIASES = {
    "freight terminal": ("terminal",),
    "municipal museum": ("museum",),
    "regional tournament": ("tournament", "competition"),
    "academic symposium": ("symposium", "conference"),
    "research laboratory": ("laboratory", "lab"),
    "retail distribution network": ("distribution", "retail", "network"),
    "protected wetland": ("wetland", "habitat"),
    "city maintenance department": ("maintenance", "department"),
    "community event center": ("community", "event center", "venue"),
}


def polished_reference_question(reference: str) -> str:
    """One narrow, provably non-substantive cleanup for old V4 baselines.

    V4's old terminal work goal sometimes already asked to explain supporting
    findings, and the fixed suffix asked the SAME thing again. Remove only
    the exact duplicate clause, leaving the final support requirement.
    Never paraphrase arbitrary facts or remove the primary task request.
    """
    return re.sub(
        r";\s*explain which documented findings support the conclusion"
        r"(?=\.\s*Explain which documented findings support your assessment\.)",
        "", reference, flags=re.I,
    )


def _term_present(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])",
                     text, re.I) is not None


def _semantic_safety_issues(scene: dict, question: str, reference: str) -> list[str]:
    errors = []
    contract_text = json.dumps(public_task_contract(scene), ensure_ascii=False)
    allowed = contract_text + " " + reference
    # Inference from generic source roles is not a citation for an invented
    # specific source or instrument. This catches concrete regressions such
    # as "spectrometer" and "gate logs" not named in the Task contract.
    for term in CONCRETE_DETAILS:
        if _term_present(question, term) and not _term_present(allowed, term):
            errors.append("unsupported_concrete_detail")
            break
    if (any(_term_present(question, word) for word in CURRENT_STATE_TERMS)
            and not any(_term_present(allowed, word)
                        for word in CURRENT_STATE_TERMS)):
        errors.append("invented_current_state")
    # Require two explicit DELEGATION verbs, not mere event chronology.
    # "Which incident happened first and what happened then?" is valid.
    if re.search(
        r"\b(?:first|initially|start by)\b.{0,100}?"
        r"\b(?:investigat\w*|analy[sz]\w*|examin\w*|collect\w*|"
        r"gather\w*|check\w*|review\w*|verify\w*|retriev\w*)\b"
        r".{0,170}?\b(?:then|next|afterwards|finally)\b",
        question, re.I,
    ):
        errors.append("internal_workflow_order_exposed")
    # Response prioritization is not prioritization of the reservations.
    goal = scene["question_brief"]["final_work_goal"].lower()
    if "rank response priorities" in goal:
        if (re.search(r"\brank\s+(?:the\s+)?(?:visitor\s+)?"
                      r"(?:reservation\s+)?requests\b", question, re.I)
                and not re.search(
                    r"\b(?:response|handling|action|measure|intervention|"
                    r"follow.up)\w*\b", question, re.I)):
            errors.append("ranking_target_changed")
    return sorted(set(errors))


def surface_issues(scene: dict, candidate: str, *, baseline: str | None = None) -> list[str]:
    """Only high-precision public-surface checks; Critic handles semantics."""
    question = " ".join(str(candidate or "").split())
    brief = task_level_brief(scene)
    ref = baseline or reference_question(scene)
    errors = []
    if not (55 <= len(question) <= 1000):
        errors.append("question_length")
    if "\n" in str(candidate):
        errors.append("multiline_question")
    if re.search(r"\b(?:S\d+|T\d+|E\d+|subtask|sub-agent|main-agent|KoPL"
                 r"|first agent|second agent|delegate|task graph|DAG)\b", question, re.I):
        errors.append("private_orchestration_plan_leak")
    if re.search(r"\b(?:attached|uploaded|as shown in|I watched|I observed)\b",
                 question, re.I):
        errors.append("unverified_attachment_or_observation")
    if re.search(r"\b\d+(?:\.\d+)?%?\b", question):
        ref_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", ref))
        cand_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", question))
        if cand_numbers - ref_numbers:
            errors.append("invented_numeric_constraint")
    # Allow semantic aliases such as "our museum" in place of "municipal
    # museum"; the Critic judges whether the same site is implied. Reject
    # absent site anchors and explicit moves to a DIFFERENT site.
    setting = brief["setting"].casefold()
    anchors = SETTING_ALIASES.get(setting, (setting.split()[-1],))
    if not any(_term_present(question, label) for label in anchors):
        errors.append("changed_or_missing_setting")
    if any(re.search(
        r"\b(?:another|different|other)\s+(?:\w+\s+){0,2}?" +
        re.escape(label) + r"\b", question, re.I
    ) for label in anchors):
        errors.append("changed_setting")
    subject_words = {
        word for word in re.findall(r"[a-z]{4,}", brief["issue"].lower())
        if word not in STOP
    }
    used_words = set(re.findall(r"[a-z]{4,}", question.lower()))
    # Exact overlap of two content words was over-rejecting legitimate
    # paraphrases (e.g. "time discrepancy" for "conflicting times").
    if subject_words and not (subject_words & used_words):
        errors.append("missing_case_subject")
    # Lexical intent keywords and the phrase "documented findings" are NOT
    # a sufficient semantic validator. The Critic inspects the full goal
    # contract; these checks formerly rejected natural but faithful wording.
    errors.extend(_semantic_safety_issues(scene, question, ref))
    if "documented findings" in question.casefold():
        errors.append("formulaic_evidence_phrasing")
    declared = set(public_task_contract(scene)["public_modalities"])
    for word, permitted in (
        ("video", {"video"}), ("footage", {"video"}), ("audio", {"audio"}),
        ("image", {"image"}), ("photograph", {"image"}),
    ):
        if re.search(r"\b" + word + r"s?\b", question, re.I) and not (
            permitted & declared
        ):
            errors.append("invented_media_type")
            break
    return sorted(set(errors))


def _validate_review(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("critic_review_not_object")
    verdict = value.get("verdict")
    naturalness = value.get("naturalness")
    differences = value.get("differences")
    summary = value.get("summary")
    if (verdict not in {"pass", "revise", "uncertain"}
            or naturalness not in {"acceptable", "needs_polish"}
            or not isinstance(differences, list)
            or len(differences) > 3 or not isinstance(summary, str)):
        raise ValueError("critic_review_invalid")
    for diff in differences:
        if (not isinstance(diff, dict) or
                any(not isinstance(diff.get(k), str) for k in (
                    "reference_excerpt", "candidate_excerpt",
                    "problem", "minimal_fix"
                )) or not diff["problem"].strip() or
                not diff["minimal_fix"].strip()):
            raise ValueError("critic_difference_invalid")
    if verdict == "pass" and (naturalness != "acceptable" or differences):
        raise ValueError("critic_pass_inconsistent")
    if verdict == "revise" and not differences and naturalness != "needs_polish":
        raise ValueError("critic_revision_without_issue")
    return {
        "verdict": verdict,
        "naturalness": naturalness,
        "summary": summary[:600],
        "differences": [
            {key: diff[key][:400] for key in (
                "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"
            )} for diff in differences
        ],
    }


def _simple_feedback(issues: list[str]) -> dict:
    return {
        "verdict": "revise",
        "naturalness": "needs_polish",
        "summary": "Correct these concrete preservation/format issues: " +
        ", ".join(issues),
        "differences": [
            {"reference_excerpt": "", "candidate_excerpt": "",
             "problem": issue,
             "minimal_fix": "Repair only this mismatch using the immutable task contract."}
            for issue in issues[:3]
        ],
    }


def naturalize_one(scene: dict, *, author, critic,
                   max_revisions: int = 2) -> tuple[str, dict]:
    """Bounded Author/Critic/repair with safe deterministic fallback.

    A model PASS is a screening judgment, NOT a formal semantic proof.
    """
    if not 0 <= max_revisions <= 8:
        raise ValueError("max_revisions must be between 0 and 8")
    contract = public_task_contract(scene)
    initial = reference_question(scene)
    attempts = []
    latest = None
    feedback = None
    last_failure = "not_attempted"
    author_calls = critic_calls = 0
    accepted = None
    accepted_review = None
    for revision in range(max_revisions + 1):
        row = {"round": revision + 1, "author_calls": 1, "critic_calls": 0}
        author_calls += 1
        payload = {
            "reference_question": initial,
            # The reference remains immutable; this style-only seed removes
            # the legacy duplicate evidence request where present.
            "style_seed_question": polished_reference_question(initial),
            "immutable_task_contract": contract,
            "style_hint": style_hint(scene),
            "instruction": (
                "Rewrite the public user need, not the subtask schedule. "
                "Avoid copied 'documented findings' boilerplate. "
                "Never invent specific sources or turn generic objects into named subtypes."
            ),
        }
        if latest is not None:
            payload["previous_draft"] = latest
            payload["reviewer_feedback"] = feedback
            payload["instruction"] = "Fix only the reported local differences."
        try:
            reply = author.complete_json(
                AUTHOR_SYSTEM, json.dumps(payload, ensure_ascii=False),
                response_schema=AUTHOR_SCHEMA,
            )
            candidate = reply.get("question")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("author_question_missing")
            candidate = " ".join(candidate.split())
        except (BackendError, ValueError, KeyError, TypeError) as exc:
            last_failure = "author_error"
            row["failure"] = last_failure
            row["detail"] = str(exc)[:180]
            attempts.append(row)
            break
        latest = candidate
        row["candidate"] = candidate
        if re.sub(r"\W+", "", candidate.casefold()) == re.sub(
            r"\W+", "", initial.casefold()
        ):
            last_failure = "not_naturalized"
            row["failure"] = last_failure
            feedback = _simple_feedback([last_failure])
            attempts.append(row)
            continue
        issues = surface_issues(scene, candidate, baseline=initial)
        row["deterministic_guards"] = issues
        if issues:
            last_failure = "surface_guard"
            row["failure"] = last_failure
            feedback = _simple_feedback(issues)
            attempts.append(row)
            continue
        review_payload = {
            "reference_question": initial,
            "candidate_question": candidate,
            "immutable_task_contract": contract,
            "semantic_delta_checklist": [
                "Any novel specific entity, instrument, event state or source name?",
                "Any changed target of rank, compare, assess or other action?",
                "Any explicit private subtask execution order?",
                "Any new constraints, data assertions, or omitted work products?",
            ],
        }
        try:
            row["critic_calls"] += 1
            critic_calls += 1
            judgment = _validate_review(critic.complete_json(
                CRITIC_SYSTEM,
                json.dumps(review_payload, ensure_ascii=False),
                response_schema=CRITIC_SCHEMA,
            ))
        except (BackendError, ValueError, KeyError, TypeError) as exc:
            last_failure = "critic_protocol_error"
            row["failure"] = last_failure
            row["detail"] = str(exc)[:180]
            # One FORMAT repair on the SAME candidate, not a new rewrite.
            try:
                row["critic_calls"] += 1
                critic_calls += 1
                review_payload["format_repair_instruction"] = (
                    "Return the required fields exactly; if unsure choose uncertain."
                )
                judgment = _validate_review(critic.complete_json(
                    CRITIC_SYSTEM, json.dumps(review_payload, ensure_ascii=False),
                    response_schema=None,
                ))
            except (BackendError, ValueError, KeyError, TypeError) as fix_exc:
                row["protocol_retry_failure"] = str(fix_exc)[:180]
                attempts.append(row)
                break
        row["critic_review"] = judgment
        if judgment["verdict"] == "pass":
            accepted = candidate
            accepted_review = judgment
            row["accepted"] = True
            attempts.append(row)
            break
        last_failure = (
            "critic_uncertain" if judgment["verdict"] == "uncertain"
            else "critic_requests_revision"
        )
        row["failure"] = last_failure
        feedback = judgment
        attempts.append(row)
        if judgment["verdict"] == "uncertain":
            break

    safe_reference = polished_reference_question(initial)
    result = accepted if accepted is not None else safe_reference
    status = "accepted" if accepted is not None else "fallback"
    metadata = {
        "review_policy_version": "v4.2_fidelity_naturalness",
        "safe_reference_polish_applied": (
            accepted is None and result != initial
        ),
        "pipeline": "v4_task_contract_author_local_critic_repair",
        "status": status,
        "reason": "independent_critic_pass" if accepted is not None else last_failure,
        "reference_question": initial,
        "final_question": result,
        "contract_sha256": _fingerprint(contract),
        "macro_graph_sha256": _fingerprint(scene["macro_graph"]),
        "reference_sha256": _fingerprint(initial),
        "accepted_review": accepted_review,
        "audit": attempts,
        "revision_rounds": len(attempts) - 1 if attempts else 0,
        "author_calls": author_calls,
        "critic_calls": critic_calls,
        "semantic_equivalence_certified": False,
    }
    return result, metadata


def naturalize_scenes(scenes: list[dict], *, author, critic,
                      max_revisions: int = 2, workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    def process(scene):
        return naturalize_one(
            scene, author=author, critic=critic, max_revisions=max_revisions
        )

    # map preserves input order, while individual Author -> Critic -> Repair
    # loops execute sequentially per scene.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        produced = list(pool.map(process, scenes))
    counts = Counter()
    failure_reasons = Counter()
    # Accepting 100 independently natural-looking questions can still
    # collapse distinct records into identical prose. Keep the earliest
    # accepted candidate and revert later duplicates to the frozen
    # deterministic reference, preserving the original Task semantics.
    previously_seen = set()
    for scene, (question, info) in zip(scenes, produced):
        normalized = re.sub(r"\W+", " ", question.casefold()).strip()
        if info["status"] == "accepted" and normalized in previously_seen:
            question = polished_reference_question(info["reference_question"])
            info["safe_reference_polish_applied"] = (
                question != info["reference_question"]
            )
            info["status"] = "fallback"
            info["reason"] = "duplicate_naturalized_question"
            info["final_question"] = question
            info["accepted_review"] = None
            for row in info["audit"]:
                if row.get("accepted"):
                    row["accepted"] = False
                    row["failure"] = "duplicate_naturalized_question"
        previously_seen.add(re.sub(r"\W+", " ", question.casefold()).strip())
        scene["question"] = question
        scene["public"]["question"] = question
        scene["question_authoring"] = info
        counts[info["status"]] += 1
        if info["status"] == "fallback":
            failure_reasons[info["reason"]] += 1
        counts["author_calls"] += info["author_calls"]
        counts["critic_calls"] += info["critic_calls"]
        if info["status"] == "accepted" and info["revision_rounds"] > 0:
            counts["accepted_after_revision"] += 1
    return {
        "mode": "optional_llm_author_critic",
        "accepted": counts["accepted"],
        "fallback": counts["fallback"],
        "accepted_after_revision": counts["accepted_after_revision"],
        "author_calls": counts["author_calls"],
        "critic_calls": counts["critic_calls"],
        "failure_reasons": dict(failure_reasons),
        "max_revisions": max_revisions,
        "workers": workers,
        "model_review_not_formal_proof": True,
    }


def validate_naturalized_question(scene: dict) -> list[str]:
    """Replay-verifiable invariants, not a fresh LLM Critic call."""
    meta = scene.get("question_authoring")
    if meta is None:
        return []
    errors = []
    original = reference_question(scene)
    contract = public_task_contract(scene)
    if meta.get("reference_question") != original:
        errors.append("review_reference_drift")
    if meta.get("reference_sha256") != _fingerprint(original):
        errors.append("review_reference_fingerprint_mismatch")
    if meta.get("contract_sha256") != _fingerprint(contract):
        errors.append("review_task_contract_changed")
    if meta.get("macro_graph_sha256") != _fingerprint(scene["macro_graph"]):
        errors.append("review_macro_graph_changed")
    if meta.get("final_question") != scene["question"]:
        errors.append("review_final_question_changed")
    if meta.get("semantic_equivalence_certified") is not False:
        errors.append("review_overclaims_semantic_certification")
    status = meta.get("status")
    if status == "accepted":
        review = meta.get("accepted_review")
        try:
            validated = _validate_review(review)
            if validated["verdict"] != "pass":
                errors.append("review_not_passed")
        except (ValueError, TypeError):
            errors.append("review_invalid_pass_record")
        # Previously published (pre-v4.2) records retain their old schema
        # and cannot retroactively claim stronger semantic screening. New
        # naturalizations are checked against the current stricter guard.
        if (meta.get("review_policy_version") == "v4.2_fidelity_naturalness"
                and surface_issues(scene, scene["question"], baseline=original)):
            errors.append("review_surface_mismatch")
        if scene["question"] == original:
            errors.append("review_not_naturalized")
        if not any(
            a.get("accepted") and a.get("candidate") == scene["question"]
            and a.get("critic_review") == review
            for a in meta.get("audit", [])
        ):
            errors.append("review_audit_missing_approved_candidate")
    elif status == "fallback":
        expected = (
            polished_reference_question(original)
            if meta.get("review_policy_version") == "v4.2_fidelity_naturalness"
            else original
        )
        if scene["question"] != expected or meta.get("accepted_review") is not None:
            errors.append("review_fallback_changed_question")
        if meta.get("review_policy_version") == "v4.2_fidelity_naturalness":
            if meta.get("safe_reference_polish_applied") is not (expected != original):
                errors.append("review_fallback_polish_metadata_mismatch")
    else:
        errors.append("review_unknown_status")
    return errors
