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


AUTHOR_SYSTEM = """You are the QUESTION AUTHOR for a realistic,
task-first multi-agent benchmark. You have the fixed Macro TaskGraph, but a
real user knows only their problem, not the internal Task decomposition.

OUTPUT GOAL: In natural language, state ONE overarching user information
need that would be appropriately answered by the provided Macro Tasks.
Usually use a short situation cue plus a direct question, or one natural
first-person request. Around 25-65 words is often enough, but clarity and
faithful scope matter more than the count.

WHAT YOU MUST PRESERVE: the scenario, affected target, final decision or
output, and substantive constraints. The same existing Macro TaskGraph
must remain a reasonable and sufficient way to answer the question.

WHAT YOU MUST NEVER DO:
- Do not turn the Macro TaskGraph into a to-do list: no "review the footage,
  then cross-reference records, next assess compliance, finally recommend".
  These are INTERNAL execution steps, not things the user should specify.
- Do not promote intermediate Tasks into extra required final deliverables:
  an investigation that internally needs a timeline does not mean the user
  requests a separately delivered detailed timeline.
- Do not invent a source merely because a source CATEGORY sounds like it:
  "primary event material" does NOT authorize "surveillance footage",
  "operational records" does NOT authorize "dispatch logs", and
  "independent reports" does NOT authorize "witness statements".
  Public modality labels likewise do not establish any observed contents.
- Do not turn a generic object into a specific subtype (instrument ->
  spectrometer), or add current states (currently, still, ongoing), people,
  times, numeric thresholds, evidence claims, restrictions or conclusions.
- Do not change the object of an operation. Ranking corrective responses
  is not ranking booking requests.
- Do not mention internal Task numbers, agents, dependency edges, or
  execution order. Task data informs YOUR interpretation, not USER prose.

STYLE:
- Vary naturally among question forms, short background + question, and
  practical first-person requests. Do not force "please" at the beginning.
- Do not repeat "documented findings", "which specific findings support"
  or lists of evidence categories. A concise request for justification or
  evidence is enough when the final goal requires support.
- Treat reference_question as the SEMANTIC baseline, not prose to copy.
  The Macro Tasks are for reasoning about the goal, not instructions to
  transcribe. Do not force every supporting Task to be mentioned.

Examples, illustrating form only and NOT supplying task facts:
BAD: "Please review footage, cross-reference logs, then assess the policy,
and finally rank actions based on the documented findings."
GOOD: "What should we prioritize to address this incident, considering the
relevant rules and available capacity, and why?"

If reviewer_feedback is provided, make ONLY the requested local changes
to previous_draft; keep its unaffected user goal and context.
Return JSON {"question":"..."} only."""

CRITIC_SYSTEM = """You are an independent and SKEPTICAL Macro Task-aware
CRITIC. In the last V4.3 pilot, formulaic multi-step procedural requests and
invented specific source names often received an undeserved PASS. Do not
rubber-stamp a question just because the task decomposition could execute it.

You receive a reference question, the candidate question, and a direct
semantic view of the SAME frozen Macro TaskGraph. Evaluate FOUR independent
dimensions BEFORE deciding the verdict:

1. GOAL ALIGNMENT: Does the user's main objective, target of any operation,
   scope, and required FINAL deliverable stay the same? Supporting Macro
   Tasks need not appear in the question, and their existence does not
   justify adding each intermediate output as a new user deliverable.
2. TASK SUFFICIENCY: Are the provided Macro Tasks still collectively
   reasonable and sufficient to answer everything the new user is asking?
   Do not demand the question explicitly name every task.
3. USER FRAMING: Is this what a normal person would ask, or did the Author
   write the AGENT'S WORKFLOW? Lists of imperative analysis steps such as
   "review X, cross-reference Y, then assess Z, finally rank" are
   workflow_instructions, even if the steps exactly match the Macro TaskGraph.
   A natural user can ask for an evidence-based decision without prescribing
   intermediate subagent operations. Robotic and repeated report boilerplate
   also needs revision.
4. UNSUPPORTED ADDITIONS: Compare wording to the reference and Macro Task
   semantic view. Are there NEW facts, named data sources, entities, timing
   states or constraints? Source category "primary event material" does not
   warrant "surveillance footage"; "operational records" does not warrant
   "dispatch logs"; "independent reports" does not warrant "witness
   statements". A public video modality does not guarantee footage exists.
   Plausibility and task feasibility are NOT support for invented details.

Think about the user's requested END, not the internal HOW. Allow concise
paraphrases and "our museum" when the referent remains clear. Do not
require exact source names, fixed intent verbs, or "documented findings".

Fill the diagnostic axes below honestly. PASS only when ALL are true:
goal_alignment=same, task_sufficiency=sufficient,
user_framing=user_goal, unsupported_additions=[], naturalness=acceptable,
differences=[]. A polished to-do list is NOT user_goal.

For a problem that can be repaired, choose revise, name the actual offending
candidate excerpt, and propose at most three minimal changes. If the
sufficiency or semantic match cannot be judged, choose uncertain rather
than claiming a definite match. Do NOT pursue a target PASS rate.

Return strict JSON only:
{"goal_alignment":"same|changed|uncertain",
 "task_sufficiency":"sufficient|insufficient|uncertain",
 "user_framing":"user_goal|workflow_instructions|robotic",
 "unsupported_additions":["exact candidate-only phrase, if any"],
 "verdict":"pass|revise|uncertain",
 "naturalness":"acceptable|needs_polish",
 "summary":"specific reason explaining which final goal is preserved or changed",
 "differences":[{"reference_excerpt":"...", "candidate_excerpt":"...",
                 "problem":"...", "minimal_fix":"..."}]}
All arrays may be empty when appropriate. Make each cited excerpt real;
avoid generic praise like "professional restatement" without comparing
the goal, user request and specific Task work.
"""

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

# V4.4 uses structured critical axes so "natural professional restatement"
# cannot quietly count as PASS when the same model detects extra workflow
# instructions, unsupported source claims, or a new final deliverable.
CRITIC_SCHEMA_V44 = {
    "type": "object",
    "properties": {
        "goal_alignment": {"enum": ["same", "changed", "uncertain"]},
        "task_sufficiency": {"enum": ["sufficient", "insufficient", "uncertain"]},
        "user_framing": {"enum": ["user_goal", "workflow_instructions", "robotic"]},
        "unsupported_additions": {
            "type": "array", "items": {"type": "string"},
        },
        **CRITIC_SCHEMA["properties"],
    },
    "required": [
        "goal_alignment", "task_sufficiency", "user_framing",
        "unsupported_additions", "verdict", "naturalness",
        "summary", "differences",
    ],
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
        "A concise direct question with just enough real-world context",
        "A short first-person user concern followed by the key decision question",
        "An applied question about the overall outcome, with no procedural checklist",
    ],
    "intersection": [
        "A straightforward user question about the issue to resolve",
        "A compact situation description plus a focused question",
        "A first-person concern with an implicit need to reconcile several facts",
    ],
    "comparison": [
        "A direct comparison question without listing the steps of analysis",
        "A practical choice framed as a short natural question",
        "A user decision request emphasizing the outcome, not investigation steps",
    ],
    "aggregation": [
        "A succinct synthesis request, expressed as a real user need",
        "A single compact question about the overall result",
    ],
    "lookup": [
        "A direct fact-finding question with a clear target",
        "A natural context sentence followed by one focused question",
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


def macro_task_view(scene: dict) -> dict:
    """Direct SEMANTIC view of the Macro TaskGraph for Author and Critic.

    This is not an immutable word-by-word writing contract. It contains all
    legitimate Macro Task goals, typed deliverables, and the prerequisite
    structure. Internal evidence, source values, private trace and scheduling
    waves are intentionally absent. Position numbers are local to this
    model-facing view, not user-facing Task IDs.
    """
    brief = task_level_brief(scene)
    tasks = scene["macro_graph"]["tasks"]
    order = {task["task_id"]: index + 1 for index, task in enumerate(tasks)}
    return {
        "scenario": {
            "domain": brief["domain"],
            "setting": brief["setting"],
            "issue": brief["issue"],
            "user_role": brief["user_role"],
        },
        "overall_user_need": {
            "intent": brief["intent"],
            "final_work_goal": brief["final_work_goal"],
            "final_deliverable": brief["final_deliverable"],
        },
        "macro_tasks": [
            {
                "number": order[task["task_id"]],
                "goal": task["goal"],
                "work_type": task["work_type"],
                "deliverable": task["deliverable"],
                "prerequisite_task_numbers": [
                    order[previous] for previous in task["depends_on"]
                ],
            }
            for task in tasks
        ],
        "public_resource_categories": brief["public_resource_categories"],
        "public_modalities": sorted({
            source["modality_hint"] for source in
            scene["public"]["available_source_descriptions"]
        }),
    }


def minimal_question_guards(candidate: str) -> list[str]:
    """Only basic format/privacy invariants, NEVER lexical equivalence.

    Factual, entity, target and scope changes are assessed by the LLM Critic
    from the full semantic Macro Task view. This checker intentionally does
    not require fixed setting words, intent stems, 'documented findings',
    source names or special operation verbs.
    """
    raw = str(candidate or "")
    question = raw.strip()
    errors = []
    if not 20 <= len(question) <= 1200:
        errors.append("question_length")
    if "\n" in raw or "\r" in raw:
        errors.append("multiline_question")
    if re.search(
        r"\b(?:S\d+|T\d+|E\d+|subtasks?|sub.agents?|main.agent|"
        r"KoPL|DAG|task\s+graph)\b", question, re.I,
    ):
        errors.append("private_task_structure_leak")
    if re.match(r"^\s*[\{\[]", raw) or re.search(
        r"(?i)\b(?:system|assistant|developer)\s*:\s*", raw
    ):
        errors.append("non_user_question_format")
    return sorted(set(errors))


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


def _validate_review(value: dict, *, policy: str | None = None) -> dict:
    """Validate critic protocol. V4.1-V4.3 audits keep their original format."""
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
    output = {
        "verdict": verdict,
        "naturalness": naturalness,
        "summary": summary[:600],
        "differences": [
            {key: diff[key][:400] for key in (
                "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"
            )} for diff in differences
        ],
    }
    if policy in {"v4.4_goal_first_critic", "v4.5_graph_primary_benchmark_style"}:
        alignment = value.get("goal_alignment")
        sufficiency = value.get("task_sufficiency")
        framing = value.get("user_framing")
        additions = value.get("unsupported_additions")
        if (alignment not in {"same", "changed", "uncertain"}
                or sufficiency not in {"sufficient", "insufficient", "uncertain"}
                or framing not in {"user_goal", "workflow_instructions", "robotic"}
                or not isinstance(additions, list) or len(additions) > 5
                or any(not isinstance(x, str) or not x.strip()
                       for x in additions)):
            raise ValueError("critic_v44_diagnostic_axes_invalid")
        if verdict == "pass" and (
            alignment != "same" or sufficiency != "sufficient"
            or framing != "user_goal" or additions
        ):
            raise ValueError("critic_v44_pass_contradicts_diagnostics")
        if verdict == "revise" and (alignment == "uncertain" or
                                    sufficiency == "uncertain"):
            # Doubt must not be transformed into a confident repair decision.
            raise ValueError("critic_v44_uncertain_as_revision")
        if verdict == "uncertain" and alignment == "same" and (
            sufficiency == "sufficient" and framing == "user_goal"
            and not additions and naturalness == "acceptable"
            and not differences
        ):
            raise ValueError("critic_v44_uncertain_without_issue")
        output.update({
            "goal_alignment": alignment,
            "task_sufficiency": sufficiency,
            "user_framing": framing,
            "unsupported_additions": additions[:5],
        })
    return output


def _simple_feedback(issues: list[str]) -> dict:
    return {
        "verdict": "revise",
        "naturalness": "needs_polish",
        "summary": "Correct these concrete preservation/format issues: " +
        ", ".join(issues),
        "differences": [
            {"reference_excerpt": "", "candidate_excerpt": "",
             "problem": issue,
             "minimal_fix": "Repair only this mismatch, preserving the original Macro Task goals."}
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
    task_view = macro_task_view(scene)
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
            "macro_task_graph": task_view,
            "optional_reference_hint": initial,
            "semantic_authority": "macro_task_graph",
            "style_hint": style_hint(scene),
            "instruction": (
                "Write an MMQA/OmniGAIA-style final question supported by "
                "the Macro TaskGraph. The old reference is only optional "
                "writing guidance; hide intermediate work and tool order."
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
        issues = minimal_question_guards(candidate)
        row["deterministic_guards"] = issues
        if issues:
            last_failure = "surface_guard"
            row["failure"] = last_failure
            feedback = _simple_feedback(issues)
            attempts.append(row)
            continue
        review_payload = {
            "candidate_question": candidate,
            "macro_task_graph": task_view,
            "review_focus": (
                "Evaluate the candidate ONLY against the graph's terminal "
                "goal and prerequisite work, including style, scope and "
                "unsupported specificity. Ignore old reference wording."
            ),
        }
        try:
            row["critic_calls"] += 1
            critic_calls += 1
            judgment = _validate_review(critic.complete_json(
                CRITIC_SYSTEM,
                json.dumps(review_payload, ensure_ascii=False),
                response_schema=CRITIC_SCHEMA_V44,
            ), policy="v4.5_graph_primary_benchmark_style")
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
                ), policy="v4.5_graph_primary_benchmark_style")
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
        "review_policy_version": "v4.5_graph_primary_benchmark_style",
        "safe_reference_polish_applied": (
            accepted is None and result != initial
        ),
        "pipeline": "v4_benchmark_style_graph_primary_author_critic_repair",
        "status": status,
        "reason": "independent_critic_pass" if accepted is not None else last_failure,
        "reference_question": initial,
        "final_question": result,
        "macro_task_view_sha256": _fingerprint(task_view),
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
    policy = meta.get("review_policy_version")
    if meta.get("reference_question") != original:
        errors.append("review_reference_drift")
    if meta.get("reference_sha256") != _fingerprint(original):
        errors.append("review_reference_fingerprint_mismatch")
    if policy in {"v4.3_macro_task_critic", "v4.4_goal_first_critic", "v4.5_graph_primary_benchmark_style"}:
        if meta.get("macro_task_view_sha256") != _fingerprint(macro_task_view(scene)):
            errors.append("review_macro_task_view_changed")
        expected_pipeline = {
            "v4.3_macro_task_critic": "v4_macro_task_aware_author_critic_repair",
            "v4.4_goal_first_critic": "v4_goal_first_author_diagnostic_critic_repair",
            "v4.5_graph_primary_benchmark_style": "v4_benchmark_style_graph_primary_author_critic_repair",
        }[policy]
        if meta.get("pipeline") != expected_pipeline:
            errors.append("review_wrong_pipeline")
    else:
        # Backward compatibility: V4.1/V4.2 metadata was originally bound
        # to the old task contract. Don't re-certify it under V4.3.
        if meta.get("contract_sha256") != _fingerprint(public_task_contract(scene)):
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
            validated = _validate_review(review, policy=policy)
            if validated["verdict"] != "pass":
                errors.append("review_not_passed")
        except (ValueError, TypeError):
            errors.append("review_invalid_pass_record")
        # Previously published (pre-v4.2) records retain their old schema
        # and cannot retroactively claim stronger semantic screening. New
        # naturalizations are checked against the current stricter guard.
        if (policy == "v4.2_fidelity_naturalness"
                and surface_issues(scene, scene["question"], baseline=original)):
            errors.append("review_surface_mismatch")
        if (policy in {"v4.3_macro_task_critic", "v4.4_goal_first_critic"}
                and minimal_question_guards(scene["question"])):
            errors.append("review_surface_mismatch")
        if scene["question"] == original and policy != "v4.5_graph_primary_benchmark_style":
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
            if policy in {"v4.2_fidelity_naturalness", "v4.3_macro_task_critic", "v4.4_goal_first_critic", "v4.5_graph_primary_benchmark_style"}
            else original
        )
        if scene["question"] != expected or meta.get("accepted_review") is not None:
            errors.append("review_fallback_changed_question")
        if policy in {"v4.2_fidelity_naturalness", "v4.3_macro_task_critic"}:
            if meta.get("safe_reference_polish_applied") is not (expected != original):
                errors.append("review_fallback_polish_metadata_mismatch")
    else:
        errors.append("review_unknown_status")
    return errors
