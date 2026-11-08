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
from .question import author_question, task_level_brief
from .semantic_contract import semantic_contract, semantic_surface_issues


AUTHOR_SYSTEM = """You are a QUESTION REALIZER, not a problem designer.

Turn the FROZEN TaskGraph-defined information need into ONE fluent,
natural user question. The supplied semantic_contract is authoritative.
Do not create a new task, scene, evidence source, or solution strategy.

HARD INVARIANTS (higher priority than fluent writing):
1. EXACT OBJECTIVE: preserve the candidate universe, operator, comparison
   attribute, conditions, threshold, scope and required output specified
   in the semantic_contract. Do not turn synthetic scores into fame,
   prestige, importance or quality, nor time indices into calendar dates.
2. SAME SOLUTION: the existing Macro TaskGraph must remain an adequate
   solution route without any new prerequisite, operator, field, source or
   WorkProduct. Do not change the meaning so that required work is bypassed.
   You are NOT tasked with repairing or redesigning the original graph.
3. SOURCE CLOSED WORLD: use only authorized entities and source roles.
   Do not invent a specific account, user, report, website, observation,
   media content, source subtype, real-world event, name, date or number.
   An 'image' label does NOT tell you what an image depicts.
4. ANSWER EQUIVALENCE: the result requested by the rewritten Question must
   be the same result defined by the terminal WorkProduct, not a new,
   easier, narrower, broader, or more interesting deliverable.
5. NO PRIVATE ORCHESTRATION: do not describe agents, Task IDs, graph
   structure, individual delegations, the sequence of handoffs, or tools.

NATURAL EXPRESSION:
- You MAY change syntax, shorten boilerplate, vary question openings, use
  already licensed entity references, and keep multi-step work implicit.
- You MAY NOT make the request 'realistic' by inventing story details or
  substituting a synthetic measurement with a real-world fact.
- The deterministic reference is an optional writing hint only; the
  semantic_contract and TaskGraph are the authority.
- If fidelity and fluency conflict, choose a simpler faithful question.

Examples are not new facts; do not transfer their content to the scene.
Before returning, check the objective, source boundary, numerical conditions,
and whether the SAME TaskGraph can answer exactly this question.

Return strict JSON only: {"question":"..."}"""

CRITIC_SYSTEM = """You are a STRICT, independent SEMANTIC EQUIVALENCE CRITIC,
not a reviewer optimizing naturalness or approval rate.

Compare candidate_question against the frozen semantic_contract AND existing
macro_task_graph. The contract is the specification; graph is the existing
solution route. Do not infer unseen source content or reward creativity.

CHECK ALL FIVE independent invariants:
1. objective_equivalent: exactly the same operation, target universe,
   criterion, constraints, scope and requested output?
2. solution_preserved: does the same typed Macro TaskGraph remain an
   adequate route without any NEW Task, field, observation, source,
   inference, operation or output? Do not repair the graph.
3. required_work_preserved: does the rewrite avoid simplifying away
   the intended necessary input and reasoning requirements? Assess at
   the graph-contract level, not by claiming execution ablation proof.
4. source_faithful: no unsupported entity, source type, personal account,
   source subtype, image/video/audio content, event, date, threshold,
   relationship, social significance, claim or external source?
5. answer_equivalent: the terminal result required by the contract would
   fully answer this Question, and vice versa?

Also judge naturalness: one focused user information need, no workflow
instructions, no robotic report boilerplate.

Reject an invented context even if it sounds harmless (e.g. 'my account
summary' is not licensed by generic image+table metadata).
Reject substitutions like 'most prestigious' for 'highest recorded score'.
Do NOT assume pixels were read, actual dates were established, or
additional sources exist. If evidence is missing or uncertain, return
'uncertain', not an unjustified PASS.

DECISION: 'pass' is legal ONLY when all five checks = 'yes',
user_framing='user_goal', naturalness='acceptable',
unsupported_additions=[], and differences=[].
For a clear, minimally repairable issue return 'revise' and quote the
candidate-only phrase with a minimal correction. For any unverifiable
semantic relation return 'uncertain'. Never target a PASS rate.
Do not claim formal equivalence or that each node is provably indispensable.

Return JSON with EXACT fields:
{"objective_equivalent":"yes|no|uncertain",
 "solution_preserved":"yes|no|uncertain",
 "required_work_preserved":"yes|no|uncertain",
 "source_faithful":"yes|no|uncertain",
 "answer_equivalent":"yes|no|uncertain",
 "user_framing":"user_goal|workflow_instructions|robotic",
 "unsupported_additions":["exact unsupported phrase, if any"],
 "verdict":"pass|revise|uncertain",
 "naturalness":"acceptable|needs_polish",
 "summary":"specific contract-based rationale",
 "differences":[{"reference_excerpt":"","candidate_excerpt":"...",
                 "problem":"...", "minimal_fix":"..."}]}"""

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


# Strict, versioned Question-only semantic review. Legacy audits keep their
# original schema; this does not modify downstream Task/Trace/Answer work.
SEMANTIC_REVIEW_POLICY = "taskfirst_semantic_contract_v1"
SEMANTIC_REVIEW_PIPELINE = "taskfirst_question_semantic_author_critic_v1"
SEMANTIC_AXES = (
    "objective_equivalent",
    "solution_preserved",
    "required_work_preserved",
    "source_faithful",
    "answer_equivalent",
)
CRITIC_SCHEMA_SEMANTIC = {
    "type": "object",
    "properties": {
        **{name: {"enum": ["yes", "no", "uncertain"]}
           for name in SEMANTIC_AXES},
        "user_framing": {"enum": [
            "user_goal", "workflow_instructions", "robotic",
        ]},
        "unsupported_additions": {
            "type": "array", "items": {"type": "string"},
        },
        **CRITIC_SCHEMA["properties"],
    },
    "required": [
        *SEMANTIC_AXES, "user_framing", "unsupported_additions",
        "verdict", "naturalness", "summary", "differences",
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
    intent = scene["goal"]["intent"]
    choices = STYLE_HINTS.get(intent, STYLE_HINTS["lookup"])
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
    if policy == SEMANTIC_REVIEW_POLICY:
        axes = {key: value.get(key) for key in SEMANTIC_AXES}
        framing = value.get("user_framing")
        additions = value.get("unsupported_additions")
        if (any(val not in {"yes", "no", "uncertain"} for val in axes.values())
                or framing not in {"user_goal", "workflow_instructions", "robotic"}
                or not isinstance(additions, list) or len(additions) > 5
                or any(not isinstance(item, str) or not item.strip()
                       for item in additions)):
            raise ValueError("semantic_review_axes_invalid")
        all_yes = all(val == "yes" for val in axes.values())
        if verdict == "pass" and (
            not all_yes or framing != "user_goal" or additions
            or naturalness != "acceptable" or differences
        ):
            raise ValueError("semantic_pass_contradicts_contract_checks")
        if verdict == "revise" and any(val == "uncertain" for val in axes.values()):
            raise ValueError("uncertainty_cannot_be_relabelled_revision")
        if verdict == "uncertain" and all_yes and framing == "user_goal" and (
            not additions and naturalness == "acceptable" and not differences
        ):
            raise ValueError("unexplained_semantic_uncertainty")
        output.update({
            **axes,
            "user_framing": framing,
            "unsupported_additions": list(additions),
        })
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
    contract = semantic_contract(scene)
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
            "semantic_contract": contract,
            "optional_reference_hint": initial,
            "semantic_authority": "frozen_semantic_contract_and_macro_task_graph",
            "style_hint": style_hint(scene),
            "instruction": (
                "Realize exactly the frozen semantic_contract as one natural "
                "user Question, with the same TaskGraph solution path and no "
                "new facts or sources. Hide the internal workflow."
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
        issues.extend(semantic_surface_issues(candidate, contract))
        from envir.schema import normalize_label
        source_original = scene.get("source_priors", {}).get(
            "MultiModalQA", {}).get("original_question", "")
        if source_original and normalize_label(candidate) == normalize_label(source_original):
            issues.append("candidate_copies_original_mmqa_question")
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
            "semantic_contract": contract,
            "review_focus": (
                "Five-way bidirectional problem equivalence: exact operation, "
                "same existing solution path, no loss of required work, "
                "closed source inventory, same final result. Strictly reject "
                "new contextual claims or unsupported source content."
            ),
        }
        try:
            row["critic_calls"] += 1
            critic_calls += 1
            judgment = _validate_review(critic.complete_json(
                CRITIC_SYSTEM,
                json.dumps(review_payload, ensure_ascii=False),
                response_schema=CRITIC_SCHEMA_SEMANTIC,
            ), policy=SEMANTIC_REVIEW_POLICY)
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
                ), policy=SEMANTIC_REVIEW_POLICY)
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
        "review_policy_version": SEMANTIC_REVIEW_POLICY,
        "semantic_contract_sha256": _fingerprint(contract),
        "safe_reference_polish_applied": (
            accepted is None and result != initial
        ),
        "pipeline": SEMANTIC_REVIEW_PIPELINE,
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
    from envir.schema import normalize_label
    source_original = scene.get("source_priors", {}).get(
        "MultiModalQA", {}).get("original_question", "")
    if source_original and normalize_label(scene["question"]) == normalize_label(source_original):
        errors.append("naturalized_question_copies_original_mmqa_question")
    policy = meta.get("review_policy_version")
    if meta.get("reference_question") != original:
        errors.append("review_reference_drift")
    if meta.get("reference_sha256") != _fingerprint(original):
        errors.append("review_reference_fingerprint_mismatch")
    if policy == SEMANTIC_REVIEW_POLICY:
        try:
            if meta.get("semantic_contract_sha256") != _fingerprint(semantic_contract(scene)):
                errors.append("semantic_contract_fingerprint_mismatch")
        except (ValueError, KeyError, TypeError):
            errors.append("semantic_contract_unavailable")
        if meta.get("macro_task_view_sha256") != _fingerprint(macro_task_view(scene)):
            errors.append("review_macro_task_view_changed")
        if meta.get("pipeline") != SEMANTIC_REVIEW_PIPELINE:
            errors.append("review_wrong_pipeline")
    elif policy in {"v4.3_macro_task_critic", "v4.4_goal_first_critic", "v4.5_graph_primary_benchmark_style"}:
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
        if policy == SEMANTIC_REVIEW_POLICY and (
            minimal_question_guards(scene["question"]) or
            semantic_surface_issues(scene["question"], semantic_contract(scene))
        ):
            errors.append("review_semantic_surface_guard_failed")
        if scene["question"] == original and policy not in {
            "v4.5_graph_primary_benchmark_style", SEMANTIC_REVIEW_POLICY,
        }:
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
            if policy in {"v4.2_fidelity_naturalness", "v4.3_macro_task_critic", "v4.4_goal_first_critic", "v4.5_graph_primary_benchmark_style", SEMANTIC_REVIEW_POLICY}
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
