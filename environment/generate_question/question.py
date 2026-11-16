"""MacroGraph-grounded Question authoring from frozen source-scoped specifications.

LLM rewriting is optional. An independent Critic and deterministic guards
can reject additions; the original executable problem remains unchanged.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .spec import spec_issues, VERSION
from .intent import intent_contract, style_contract
from .macro_contract import macro_question_contract
from .surface_quality import (
    question_language_warnings, normalize_question_punctuation,
    topic_mentioned, is_explicit_user_request,
)


def reference_question(spec: dict) -> str:
    errors = spec_issues(spec)
    if errors:
        raise ValueError("invalid_spec:" + ",".join(errors))
    return _intent_reference_question(spec)


def _one_output_v2_question(spec: dict) -> str:
    """Natural deterministic fallback, with one (and only one) answer slot."""
    plan = spec["macro_plan"]
    clues = plan["clues"]
    topic = spec["scenario"]["topic"]
    goal = plan["goal"]["public_label"]
    resolution = plan.get("goal_resolution", {}).get("kind", "direct_lookup")
    goal_question = (
        "which of the two independent sources has the higher count, first or second?"
        if resolution == "comparison" else
        "what is the completion rate in whole percent (rounded down) "
        "from the independently recorded completed and total counts?"
        if resolution == "calculation" else
        f"what is the confirmed {goal}?"
    )
    domain = plan["domain"]
    item = {
        "sports": "match incident", "film": "episode or screening",
        "music": "performance", "culture": "exhibition item",
        "education": "academic item", "logistics": "shipment event",
        "civic": "site record", "environment": "survey observation",
        "science": "experiment observation", "business": "filing",
    }.get(domain, "item")
    if plan["pattern"] in {
        "disputed_reports", "corroborated_dispute",
    }:
        if plan["pattern"] == "corroborated_dispute":
            return (
                f"In the {topic} materials, two independent reports "
                f"disagree about the reported year for some {item} records. "
                f"An independent event log also marks the matching "
                f"record as {clues['case_tag']}. For the record satisfying "
                f"both conditions, {goal_question}"
            )
        return (
            f"Two independent {topic} records disagree about the year "
            f"reported for one {item}. For the disputed {item}, "
            f"{goal_question}"
        )
    if plan["pattern"] == "multimodal_event":
        return (
            f"In the {topic} materials, a video shows a "
            f"{clues['scene_action']} action no later than the spoken "
            f"{clues['spoken_cue']} cue. An independent event log marks "
            f"the matching observation as {clues['case_tag']}. "
            f"For that {item}, {goal_question}"
        )
    if clues["visual_modality"] == "video":
        visual = (
            f"a video frame with a {clues['scene_action']} action at or "
            f"after the {clues['time_threshold_ms']//1000}-second mark"
        )
    else:
        visual = f"an image with a {clues['emblem_color']} emblem"
        if plan["pattern"] == "serial_identity":
            visual += f" and a {clues['emblem_shape']} motif"
    if plan["pattern"] == "parallel_cues":
        second = (
            f"spoken commentary with a {clues['spoken_cue']} cue"
            if clues["caption_modality"] == "audio" else
            f"written notes describing a {clues['emblem_shape']} motif"
        )
        return (
            f"In the {topic} materials, {visual} and {second} "
            f"identify the same {item}. For that item, "
            f"{goal_question}"
        )
    return (
        f"In the {topic} materials, {visual} identifies one {item}. "
        f"For the matching item, {goal_question}"
    )


def _intent_reference_question(spec: dict) -> str:
    """Deterministic fidelity fallback, not the Author's primary template."""
    intent = intent_contract(spec)
    context, kind, topic = intent["setting"], intent["entity_kind"], intent["topic"]
    if (spec.get("reasoning_pattern") == "task_plan"
            and spec["macro_plan"].get("semantic_grounding_policy") ==
            "goal_constrained_yago_grounding_v1"):
        # The Goal is already selected and frozen by the witnessed relation
        # grammar. Do not turn individual RDF predicates into a chain of
        # awkward nested noun phrases, and do not let the Question invent
        # new constraints or historical claims.
        plan = spec["macro_plan"]
        goal = plan["goal"]["semantic"]
        topic = spec["scenario"]["topic"]
        clues = "; ".join(
            f"{info['attribute']} '{info['value']}' in {info['title']}"
            for _, info in sorted(plan["clues"].items())
        )
        condition = plan["condition"]
        if condition:
            qualifying = (
                "the completed archive session" if condition["field"] ==
                "event_status" else "the archive session recorded in or after 2025"
            )
            if plan["mode"] == "verify":
                instruction = (
                    f"Using {qualifying}, reconcile the two independent "
                    "candidate-reference reports with the documented genuine "
                    "YAGO secondary relationship to identify the real-world "
                    "entity referenced by the dossier."
                )
            else:
                instruction = (
                    f"Use {qualifying} to locate the dossier's documented "
                    "real-world entity reference."
                )
        else:
            instruction = (
                "Use the matching dossier's documented real-world entity "
                "reference."
            )
        return (
            f"The fictional {topic} has an unresolved dossier identified by "
            f"these simulated source clues: {clues}. {instruction} "
            f"{goal['question']}"
        )
    if spec.get("reasoning_pattern") == "task_plan" and (
        spec["macro_plan"].get("question_sampling_policy") ==
        "grounded_relation_goal_question_v1"
    ):
        # Express the *actual RDF relation chain* as a user-level goal
        # instead of printing an internal predicate sequence or Macro IDs.
        # This is an optional current generation policy; historical frozen
        # task_plan reference questions remain unchanged for strict Replay.
        from .knowledge_macro import predicate_label
        plan = spec["macro_plan"]
        target = plan["world_binding"]["target_case"]
        path = plan["world_binding"]["candidate_paths"][target]
        nouns = {
            "neighbor": "neighboring place",
            "location": "location",
            "birth place": "birthplace",
            "contained in place": "containing region",
            "capital": "capital",
            "twinned with": "twin municipality",
            "member of": "member organization",
            "influenced by": "influencing entity",
            "student of": "teacher",
            "spouse": "spouse",
            "performer": "performer",
            "nationality": "nationality",
            "works at": "workplace",
            "founder": "founder",
        }
        current = "the project's referenced real-world entity"
        for item in path:
            relation_name = predicate_label(item["predicate"]).lower()
            noun = nouns.get(
                relation_name, relation_name + " relation target"
            )
            current = "the " + noun + " of " + current
        clues = [
            f"{info['attribute']} '{info['value']}' in the "
            f"{info['title']}"
            for _, info in sorted(plan["clues"].items())
        ]
        clue_text = "; ".join(clues)
        topic = spec["scenario"]["topic"]
        condition = plan["condition"]
        selector = (
            "the completed project meeting" if condition and
            condition["field"] == "event_status"
            else "the project meeting recorded in or after 2025"
            if condition else None
        )
        if plan["mode"] == "trace":
            return (
                f"The {topic} includes several clues for one project: "
                f"{clue_text}. Once the project is identified, "
                f"what is {current}?"
            )
        if plan["mode"] == "event":
            return (
                f"An unidentified project in the {topic} is described "
                f"by {clue_text}. For {selector}, "
                f"what is {current}?"
            )
        return (
            f"In the {topic}, the project is identified by {clue_text}. "
            f"After checking its two independent reference reports "
            f"against the information from {selector} and a genuine "
            f"YAGO relationship, what is {current}?"
        )
    if spec.get("reasoning_pattern") == "task_plan":
        from .knowledge_macro import predicate_label
        plan = spec["macro_plan"]
        binding = plan["world_binding"]
        route = [
            predicate_label(item["predicate"])
            for item in binding["candidate_paths"][binding["target_case"]]
        ]
        clues = "; ".join(
            f"{row['attribute']} '{row['value']}' from {row['title']}"
            for _, row in sorted(plan["clues"].items())
        )
        criterion = plan["condition"]
        rule = (
            "the project's completed event" if criterion and
            criterion["field"] == "event_status" else
            "the project's event recorded in or after 2025"
            if criterion else None
        )
        sequence = ", followed by ".join(route)
        title = spec["scenario"]["topic"]
        if plan["mode"] == "trace":
            return (
                f"In the {title}, simulated catalog records describe an "
                f"unidentified project using these clues: {clues}. "
                f"For that project's real-world reference, what entity "
                f"is reached by the documented {sequence} relationship trail?"
            )
        if plan["mode"] == "event":
            return (
                f"In the {title}, identify the fictional project described "
                f"by {clues}. Using {rule}, what real-world entity is "
                f"reached by following its reference through {sequence} "
                f"in the documented YAGO relationships?"
            )
        return (
            f"For the {title}, simulated records provide these clues: "
            f"{clues}. Identify the project, reconcile its two candidate "
            f"reference records with {rule} and the documented YAGO "
            f"secondary relation, and follow {sequence}. "
            f"Which related real-world entity is confirmed?"
        )
    if spec.get("reasoning_pattern") == "task_graph_first_v2":
        from environment.schema import digest
        plan = spec["macro_plan"]
        event = spec["task_first_v2_relation_sketch"]["event_relation"]
        clues = [
            f"{r['attribute']} '{r['value']}' in {r['title']}"
            for _, r in sorted(plan["clues"].items())
        ]
        eligibility = (
            "the completed event (not a scheduled one)"
            if event["field"] == "event_status" else
            "the event recorded in or after 2025 (not an older session)"
        )
        topic = spec["scenario"]["topic"]
        statements = [
            (
                f"In {topic}, an unidentified fictional project is "
                f"described by simulated clues: {'; '.join(clues)}. "
                f"Determine the project, select {eligibility}, compare "
                f"the two simulated candidate-reference sources against "
                f"the event's recorded relations and genuine YAGO facts, "
                f"and identify the single entity reached by the documented "
                f"real-world relationship trail. Which entity is it?"
            ),
            (
                f"Please resolve an archival investigation within {topic}. "
                f"The observed clues are {'; '.join(clues)}. "
                f"For the matching fictional project, use {eligibility} "
                f"to establish the relevant evidence, reconcile competing "
                f"reference records with genuine YAGO secondary relation "
                f"evidence. Which final linked real-world entity is identified?"
            ),
        ]
        choice = int(digest(
            spec["seed"], spec["ordinal"], "task-first-v2-question"
        )[:10], 16) % len(statements)
        return statements[choice]
    if spec.get("reasoning_pattern") == "task_graph_first_v1":
        from environment.schema import digest
        plan = spec["macro_plan"]
        clues = [
            f"{item['attribute']} '{item['value']}' in the {item['title']}"
            for _, item in sorted(plan["clues"].items())
        ]
        # A user asks an information question, NOT an internal DAG
        # execution recipe. The World sources supply the actual relations.
        topic = spec["scenario"]["topic"]
        options = [
            (
                f"In {topic}, several simulated records identify an "
                f"unresolved project: {'; '.join(clues)}. "
                f"The project's completed event and two independent "
                f"candidate directories describe competing possible "
                f"connections to real entities. Using the event records "
                f"and genuine YAGO relationship evidence, which entity "
                f"lies at the end of the documented relation trail?"
            ),
            (
                f"Please determine the final linked real-world entity for "
                f"a project in {topic}. Its identifying observations are "
                f"{'; '.join(clues)}. Select the completed event rather "
                f"than a scheduled follow-up, reconcile the candidate "
                f"reference records with the actual YAGO secondary "
                f"relationship, and report the entity reached by the "
                f"documented knowledge-graph trail. Which entity is it?"
            ),
            (
                f"The {topic} contains these identifying hints: "
                f"{'; '.join(clues)}. An unidentified project's event "
                f"records include both completed and scheduled sessions, "
                f"and its reference documents admit more than one candidate. "
                f"After establishing which real link is supported by the "
                f"completed session and YAGO evidence, what is the "
                f"final related entity's name?"
            ),
        ]
        slot = int(digest(
            spec["seed"], spec["ordinal"], "graph-first-question-v1"
        )[:8], 16) % len(options)
        return options[slot]
    if spec.get("reasoning_pattern") == "typed_knowledge_v4":
        from environment.schema import digest
        from .knowledge_macro import predicate_label
        plan = spec["macro_plan"]
        clue_phrases = [
            f"{info['attribute']} '{info['value']}' in the {info['title']}"
            for _, info in sorted(plan["clues"].items())
        ]
        steps = [
            predicate_label(f["predicate"]) for f in plan["real_facts"]
        ]
        path_phrase = " and then ".join(steps)
        context = spec["scenario"]["topic"]
        variants = [
            (
                f"Among {context}, identify the fictional project described "
                f"by the following simulated observations: "
                f"{'; '.join(clue_phrases)}. Its two reference directories "
                f"disagree on candidate linked real entities. Inspect the "
                f"project's fictional event, organizer and venue record, "
                f"cross-check the relevant secondary relation in YAGO, "
                f"and trace the confirmed entity via {path_phrase}. "
                f"What is the final related entity's name?"
            ),
            (
                f"A project within {context} is described in simulated "
                f"materials by {'; '.join(clue_phrases)}. "
                f"Which related real-world entity is reached after matching "
                f"the project, reconciling the two project reference "
                f"directories, checking its fictional event's eligible "
                f"relationships against a real YAGO fork, and following "
                f"the documented {path_phrase} relationship trail?"
            ),
            (
                f"Find the real linked entity for an unidentified project "
                f"in {context}. The simulated image, audio, video and "
                f"catalog evidence includes {'; '.join(clue_phrases)}. "
                f"The project's event records provide only a partial "
                f"constraint on the two reference directories. Verify "
                f"the corresponding secondary real-world relationship "
                f"in YAGO before investigating {path_phrase}. "
                f"What name do you obtain?"
            ),
        ]
        slot = int(digest(spec["seed"], spec["ordinal"],
                          "causal-hybrid-question-v4")[:8], 16) % len(variants)
        return variants[slot]
    if spec.get("reasoning_pattern") == "typed_knowledge_v3":
        from environment.schema import digest
        from .knowledge_macro import predicate_label
        plan = spec["macro_plan"]
        clues = [
            f"{item['attribute']} '{item['value']}' in the {item['title']}"
            for _, item in sorted(plan["clues"].items())
        ]
        route = ", then ".join(
            predicate_label(item["predicate"]) for item in plan["real_facts"]
        )
        topic = spec["scenario"]["topic"]
        choices = [
            (
                f"For {topic}, simulated records identify an unresolved "
                f"project using these observations: {'; '.join(clues)}. "
                f"Two separate simulated project reference directories "
                f"list possible real entity links. Reconcile both directories "
                f"and then follow documented YAGO relationships {route}. "
                f"What is the name of the related entity?"
            ),
            (
                f"A researcher working with {topic} has these records: "
                f"{'; '.join(clues)}. Identify the fictional project, compare "
                f"two separate simulated reference directories to establish "
                f"the actual entity it documents, and then follow the YAGO "
                f"relation sequence {route}. Which related entity is reached?"
            ),
            (
                f"Within {topic}, simulated media and catalog observations "
                f"give the following identifying evidence: {'; '.join(clues)}. "
                f"Match the project, cross-check candidate entity references "
                f"in two simulated source directories, and investigate the "
                f"documented YAGO trail {route}. What is the related entity name?"
            ),
        ]
        slot = int(digest(spec["seed"], spec["ordinal"],
                          "hybrid-question-style-v3")[:8], 16) % len(choices)
        return choices[slot]
    if spec.get("reasoning_pattern") in {"typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4"}:
        plan = spec["macro_plan"]
        clues = plan["clues"]
        source_phrases = [
            f"the {info['title']} shows {info['attribute']} {info['value']}"
            for _, info in sorted(clues.items())
        ]
        route = " followed by ".join(
            __import__("re").sub(r"(?<=[a-z])(?=[A-Z])", " ",
                               item["predicate"].rsplit("/", 1)[-1])
            for item in plan["real_facts"]
        )
        return (
            "In an archival reference collection, several independent "
            "catalog materials describe an unidentified dossier: "
            + "; ".join(source_phrases) + ". "
            + "Identify the dossier matching all these evidence clues "
            + "and determine the related entity reached by following "
            + route + ". What is the name of that related entity?"
        )
    if (spec.get("reasoning_pattern") == "macro_first_v2"
            and spec.get("scene_realism_version")):
        return _one_output_v2_question(spec)
    if spec["reasoning_pattern"] == "macro_first_v2":
        plan = spec["macro_plan"]
        clues = plan["clues"]
        goal = plan["goal"]["public_label"]
        mode = plan["pattern"]
        if mode == "disputed_reports":
            return (
                f"Two independent {topic} event records disagree about the "
                f"year reported for one case. What is its confirmed {goal}?"
            )
        if mode == "multimodal_event":
            return (
                f"In the {topic} coverage, footage shows a "
                f"{clues['scene_action']} action followed by commentary "
                f"mentioning a {clues['spoken_cue']} cue. The event log "
                f"tags the corresponding case as {clues['case_tag']}. "
                f"What is its confirmed {goal}?"
            )
        visual = (
            f"a frame showing a {clues['scene_action']} action at or after the "
            f"{clues['time_threshold_ms']//1000}-second mark"
            if clues["visual_modality"] == "video" else
            f"an image with a {clues['emblem_color']} emblem"
        )
        if mode == "parallel_cues":
            second = (
                f"audio commentary mentioning a {clues['spoken_cue']} cue"
                if clues["caption_modality"] == "audio" else
                f"written notes describing a {clues['emblem_shape']} motif"
            )
            return (
                f"While reviewing {topic} materials, {visual} and {second} "
                f"point to the same item. What is the confirmed {goal} "
                "for that item?"
            )
        if clues["visual_modality"] == "image":
            visual += f" and a {clues['emblem_shape']} motif"
        return (
            f"Among the {topic} materials, which confirmed {goal} belongs "
            f"to the item identified by {visual}?"
        )
    if spec["reasoning_pattern"] != "approval":
        raise ValueError("unsupported_stage1_reasoning_pattern")
    outputs = {
        "reconcile": "Which entries meet these conditions?",
        "eligibility": "Which entries meet these conditions?",
        "evidence_reconcile": (
            "Which entries qualify, and what source fact references support each entry?"
        ),
        "coverage_count": "How many distinct entries qualify?",
        "category_distribution": "How many qualifying entries occur in each resolved category?",
        "priority": "Which qualifying entry has the highest recorded priority?",
        "chronology": "Which qualifying entry has the earliest relative step?",
    }
    constraints = "Consider only active entries from the register."
    beginning = [
        f"For a fictional {topic} reference collection,",
        f"I'm examining fictional {topic} records.",
        f"Please help check the fictional {topic} collection.",
        f"The fictional {topic} register needs a cross-source review.",
        f"Regarding the fictional {topic} record set,",
    ][int(spec["ordinal"]) % 5]
    return f"{beginning} {constraints} {outputs[spec['family']]}"


AUTHOR_SYSTEM = """You write ONE realistic natural-language user question from a
FROZEN fictional benchmark ProblemSpec. Do not add, delete or change any
condition, information source or output. The source roles are simulations, not
actual observations of the original MultiModalQA media.

Start with the information the user needs or a concrete clue already available
in the fictional scenario. Name a supplied material when it helps the user
identify the clue or understand a comparison. Express the link in ordinary
language, e.g. a code in the event log identifies the session whose timing
should be checked. Let the deliverable follow from the user need.
Use a short context sentence when it helps. Ask for the result once.
Treat the topic label as the name of the fictional collection if it is an
ordinary phrase. Source titles can be shortened without changing their roles.
A role introduction or a recital of every operator adds little context.
No raw IDs, backend field names, TaskGraph vocabulary, step-by-step tool
instructions, or hallucinated facts about a real person/event/visual scene.
All relevant conditions must be clear to someone who will solve it:
active records, code-to-category translation and category approval when
those sources exist, separate clearance when provided, and the exact final
answer type (count, list, highest recorded priority, or earliest relative
ordering). No fabricated additional filters. A hypothetical problem about
a real topic does NOT assert historical facts. NEVER add calendar years,
seasons, dated cycles or concrete thresholds that the original topic
and frozen intent do not provide. Do not invent an additional eligibility
criterion such as a funding tier or certification that is absent from
the required predicates. Even "no archived records" is an extra filter
unless the frozen MacroGraph explicitly filters an archived field.
The FINAL ANSWER uses the terminal output schema: entry_label is an
entry display label, NOT label_code, route_code, abbreviated category
labels or codes. Never ask to list abbreviated labels if only
entry_label is projected. Specifically, do not add "non-archived" or
"exclude archived" unless the contract contains an archived field. A
time window is NOT independently "approved" merely because it exists;
a policy group meeting a threshold is NOT "approved" unless the graph
contains an explicit approval filter. Describe any synthetic source
field in readable language ("measured value", "audio interval start step"),
not raw names such as measured_value, minimum_required or diagram_step_start.

The frozen user intent specifies every answer-defining relationship and
deliverable. Distinguish active status from the reported status being compared.
An independent approval applies to the category, not the glossary translation. Use the material titles as readable source handles. Keep
simulated observations clearly fictional and avoid inventing a new clue.

For evidence_graph tasks, describe the user need and the starting evidence
clues without leaking the full retrieval algorithm. For a conflict task,
identify the discrepancy but never invent factual years. For visual clues,
retain BOTH stated features. The typed source graph is an internal contract,
not a list of step-by-step instructions that must appear in the Question.

For latest-version requests, you receive a STRUCTURED FROZEN USER INTENT:
constraints, typed output, source roles and a style preference. Write an
original user question from that contract. You may vary clause order,
interrogative structure and user motivation, but every mandatory condition
must remain unambiguous. Do NOT start with "As a" or "As an". Do not merely
copy or minimally edit a reference question.

For historical requests and explicit repair attempts, a FROZEN REFERENCE
QUESTION may also be supplied. It was authored from the fictional
contract, NOT the original MultiModalQA question. Preserve its meaning.
Preserve the word "active" and exactly the cross-source relation required
by the structured intent: approval, temporal window, binary disagreement,
or per-group threshold. Do NOT substitute one operation for another.
Preserve distinct entry-specific clearance for hard cases.
Temporal boundaries are INCLUSIVE: a step equal to the start or end
qualifies. NEVER say "strictly inside" or "strictly within" when also
describing a closed or inclusive interval. Use "within the inclusive
interval" or "between the endpoints, inclusive".
Preserve counting, category grouping, priority MAXIMUM, or chronological
EARLIEST directions when applicable.

Return ONE complete, grammatical English user REQUEST (55-1200 characters).
It may be a direct question ending with '?' or an unambiguous, complete
imperative such as "Please list ...", "Identify ..." or "For ... identify ...".
Never return only a declarative description or an incomplete fragment.
When the original topic has a proper name, keep that exact proper name,
rather than inventing an unrelated company/organization or dropping it. Do not return a summary, classification, fragment, table,
answer, explanation, or commentary. If unsure, closely paraphrase the frozen
reference question without inventing facts. End with '?'.

Output JSON only: {"question":"..."}."""

CRITIC_SYSTEM = """Review the candidate user question against an immutable
fictional ProblemSpec. Never infer facts from the original MultiModalQA
question, answer, image, video or audio. Decide whether the user can infer
the same required output and all constraints, with no added evidence claims.
A statement that a simulated source has hypothetical record fields is
permitted; asserting the content of ORIGINAL media is not.

Output JSON only:
{"verdict":"pass|revise|reject",
 "all_constraints_preserved":true,
 "no_unlicensed_facts":true,
 "same_answer_scope":true,
 "natural":true,
 "reason":"one sentence"}
When a compiled Macro Task contract is supplied, check it rather than
trusting the prose Goal alone. Confirm that the Question asks for the
terminal result of the stated operator chain, not a broader inventory,
all evidence, or an unrelated computation.
Check substantive semantics independently: (a) required source mapping,
(b) the exact comparator, including inclusive temporal endpoints,
(c) hard entry-specific clearance if required, (d) exact output type,
(e) no invented calendar year or new classification/eligibility rule.
The candidate must be grounded in the frozen topic/conditions, not in
unsupported elaborations supplied by an earlier scene paraphrase.
A candidate saying "strictly inside" while also claiming inclusive
boundaries MUST be rejected. Reject added predicates such as "approved
policy group" in a threshold-only task, "approved session window" in a
temporal-only task, or excluding "archived" items when the Program only
filters active=True. A plausible fictional setting does NOT license
adding unsupported eligibility criteria. In particular distinguish
the final requested output field: entry_label denotes an entry name,
label_code/route_code denote source-specific abbreviations. Reject a
candidate that asks to LIST abbreviated codes when the terminal
MacroGraph projects entry_label. A well-formed imperative user request (e.g.
"Please identify ...") is just as valid as an interrogative question.
For chronology, the compiled sort uses ascending relative_step: "earliest"
or "first" is correct; "latest" or "most recent" is WRONG. Never hallucinate
a reversed direction. Surface fluency alone is insufficient.
For macro_first tasks, the frozen Macro DAG is an internal reference;
check the initial distinguishing clues, the exact authoritative year
deliverable and, for conflict tasks, the dispute condition. Do not
require the question to list Macro operations, retrieved keys or the
intermediate source traversal.
For evidence_graph tasks, check all frozen starting clues, multi-source
identification or conflict conditions and the accession-year deliverable.
Do NOT require a user question to name every index or archive lookup.
Only pass if all four booleans are true."""


def guard_question(question: str, spec: dict,
                   macro_graph: dict | None = None) -> list[str]:
    """Conservative high-precision textual checks before/after the LLM Critic."""
    if not isinstance(question, str):
        return ["question_not_string"]
    q = question.strip()
    errors = []
    if len(q) < 55 or len(q) > 1200:
        errors.append("question_length_out_of_bounds")
    if not is_explicit_user_request(q):
        errors.append("question_not_actionable_request")
    errors.extend(question_language_warnings(q, spec, macro_graph))
    if not topic_mentioned(q, spec["scenario"]["topic"]):
        errors.append("topic_missing")
    pattern = spec.get("reasoning_pattern", "approval")
    if pattern == "task_plan" and (
        spec["macro_plan"].get("semantic_grounding_policy") ==
        "goal_constrained_yago_grounding_v1"
    ):
        # Deterministic high-precision Goal-to-question check. The
        # Author/Critic may still paraphrase, but only accepted proposals
        # passing the existing semantic/question checks are retained.
        goal = spec["macro_plan"]["goal"]["semantic"]
        if goal["question"].casefold() not in q.casefold():
            errors.append("task_plan_semantic_goal_question_missing")
    if pattern == "task_plan":
        plan = spec["macro_plan"]
        for info in plan["clues"].values():
            if info["value"].casefold() not in q.casefold():
                errors.append("task_plan_required_clue_missing")
        if plan["condition"]:
            if plan["condition"]["field"] == "event_status":
                if not re.search(r"\bcompleted\b", q, re.I):
                    errors.append("task_plan_required_status_missing")
            elif not re.search(r"\b2025\b", q):
                errors.append("task_plan_required_year_missing")
        if not re.search(r"\b(?:related|relation|entity)\b", q, re.I):
            errors.append("task_plan_final_goal_missing")
    if pattern == "task_graph_first_v2":
        relation = spec["task_first_v2_relation_sketch"]["event_relation"]
        if relation["field"] == "event_status":
            if not re.search(r"\bcompleted\b", q, flags=re.I):
                errors.append("graph_first_v2_completed_event_missing")
        else:
            if not re.search(r"\b(?:2025|in or after 2025)\b",
                             q, flags=re.I):
                errors.append("graph_first_v2_year_threshold_missing")
        if not re.search(r"\b(?:YAGO|real.world)\b", q, flags=re.I):
            errors.append("graph_first_v2_real_relation_missing")
        if not re.search(r"\b(?:reference|candidate|directory)\b",
                         q, flags=re.I):
            errors.append("graph_first_v2_reference_evidence_missing")
    if pattern == "task_graph_first_v1":
        if not re.search(r"\bcompleted\b", q, flags=re.I):
            errors.append("graph_first_required_event_status_missing")
        if not re.search(r"\b(?:YAGO|knowledge.graph|real.world)\b",
                         q, flags=re.I):
            errors.append("graph_first_real_knowledge_source_missing")
        if not re.search(r"\b(?:reference|candidate|directory)\b",
                         q, flags=re.I):
            errors.append("graph_first_competing_sources_missing")
    if pattern == "typed_knowledge_v4":
        if not re.search(r"\b(?:event|organizer|venue)\b", q, flags=re.I):
            errors.append("causal_hybrid_event_relation_missing")
        if not re.search(r"\b(?:YAGO|secondary|real-world)\b", q, flags=re.I):
            errors.append("causal_hybrid_real_fork_missing")
    if pattern == "typed_knowledge_v3":
        if not (re.search(r"\b(?:two|both|separate|independent)\b",
                          q, flags=re.I)
                and re.search(r"\b(?:reference|directory|directories)\b",
                              q, flags=re.I)):
            errors.append("hybrid_missing_two_source_reference_goal")
    if pattern in {"typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4"}:
        plan = spec["macro_plan"]
        for info in plan["clues"].values():
            if info["value"].casefold() not in q.casefold():
                errors.append("knowledge_required_clue_missing")
        if not re.search(r"\b(?:entity|related|relation|name)\b", q, re.I):
            errors.append("knowledge_goal_not_expressed")
    if pattern == "macro_first_v2":
        plan = spec["macro_plan"]
        mode, clues = plan["pattern"], plan["clues"]
        from .macro_planning import GOAL_PHRASES
        label = plan["goal"]["public_label"]
        goal_field = plan["goal"]["answer_field"]
        mentions_goal = any(
            phrase in q.casefold()
            for phrase in GOAL_PHRASES.get(goal_field, (label,))
        )
        if goal_field == "higher_count_source" and not mentions_goal:
            # "Which of the two independently recorded counts is the higher
            # one?" is a valid single-output comparison paraphrase.
            mentions_goal = bool(
                re.search(r"\b(?:higher|larger|greater)\b", q, re.I)
                and re.search(r"\b(?:counts?|totals?)\b", q, re.I)
                and re.search(r"\b(?:two|both|first|second)\b", q, re.I)
                and re.search(r"\b(?:which|determine|identify|compare)\b",
                              q, re.I)
            )
        if not mentions_goal:
            errors.append("macro_first_v2_goal_semantics_missing")
        if mode in {"disputed_reports", "corroborated_dispute"} and not re.search(
            r"\b(disagree\w*|disput\w*|differ\w*|conflict\w*|"
            r"discrepanc\w*|mismatch\w*)\b", q, re.I
        ):
            errors.append("macro_first_v2_disagreement_missing")
        if mode == "corroborated_dispute" and clues["case_tag"] not in q:
            errors.append("macro_first_v2_context_tag_missing")
        if mode not in {"disputed_reports", "corroborated_dispute"}:
            required = ([clues["scene_action"], clues["spoken_cue"],
                         clues["case_tag"]]
                        if mode == "multimodal_event"
                        else [clues["scene_action"]] if
                        clues["visual_modality"] == "video"
                        else [clues["emblem_color"]])
            if mode == "serial_identity" and clues["visual_modality"] == "image":
                required.append(clues["emblem_shape"])
            if mode == "parallel_cues":
                required.append(clues["emblem_shape"])
            def has_required_clue(cue: str) -> bool:
                # "triangular emblem" is a legitimate paraphrase for a
                # triangle motif; require the cue rather than exact stems.
                if spec.get("scene_realism_version") and cue == "triangle":
                    return bool(re.search(r"\b(?:triangle|triangular)\b", q, re.I))
                return cue.casefold() in q.casefold()
            if not all(has_required_clue(cue) for cue in required):
                errors.append("macro_first_v2_clue_missing")
            if clues["visual_modality"] == "video" and mode != "multimodal_event":
                if str(clues["time_threshold_ms"] // 1000) not in q:
                    errors.append("macro_first_v2_video_time_missing")
            if mode == "multimodal_event" and not re.search(
                r"\b(before|after|follow\w*|later|preced\w*)\b", q, re.I
            ):
                errors.append("macro_first_v2_temporal_order_missing")
        # Topic names are verbatim, user-visible material titles. A topic
        # containing "Fictional" must not trigger the metafiction guard.
        topic_masked = re.sub(
            re.escape(spec["scenario"]["topic"]), "__topic__", q,
            flags=re.I,
        )
        if re.search(r"\b(fictional|simulated|synthetic|hypothetical|imaginary)\b",
                     topic_masked, re.I):
            errors.append("macro_first_v2_meta_fiction_framing")
        if spec.get("semantic_evidence_contract_version"):
            from .semantic_evidence import (
                compile_semantic_contract, semantic_text_issues
            )
            errors.extend(semantic_text_issues(
                q, spec, compile_semantic_contract(spec),
            ))
        if spec.get("question_semantic_lock_version"):
            from .question_semantic_lock import (
                question_semantic_lock_issues,
            )
            errors.extend(question_semantic_lock_issues(q, spec))
    elif pattern == "approval":
        if not re.search(r"\bactive\b", q, flags=re.I):
            errors.append("active_condition_missing")
    family = spec["family"]
    if family in {"coverage_count", "category_distribution"} and not re.search(r"\b(how many|count|number of)\b", q, re.I):
        errors.append("count_answer_scope_missing")
    if family == "priority" and not re.search(r"\b(highest|largest|top|maximum|greatest)\b", q, re.I):
        errors.append("priority_direction_missing")
    if family == "chronology":
        if not re.search(
            r"\b(earliest|first|lowest|minimum|oldest)\b", q, re.I
        ):
            errors.append("relative_ordinal_direction_missing")
        if re.search(
            r"\b(latest|most recent|newest|last|most current)\b", q, re.I
        ):
            errors.append("reversed_chronological_direction")
    if family == "evidence_reconcile":
        if not re.search(r"\b(evidence|source|citation|reference|support)\w*", q, re.I):
            errors.append("evidence_output_scope_missing")
    fields = {"entry_label", "evidence_refs"}
    if macro_graph is not None:
        fields.update(
            field for task in macro_graph["tasks"]
            for field in task["outputs"]["typed_relation"]
            if "_" in field
        )
    implementation_terms = r"TaskGraph|WorkProduct|candidate-\d+|source_id|fact_refs|synthetic_attribute"
    terms = implementation_terms + "|" + "|".join(map(re.escape, sorted(fields)))
    if re.search(r"\b(?:" + terms + r")\b", q, re.I):
        errors.append("internal_implementation_leak")
    if re.search(r"\b(actual photograph shows|original image depicts|original video shows)\b", q, re.I):
        errors.append("unsupported_real_media_assertion")
    return sorted(set(errors))


def naturalize_one(spec: dict, *, author, critic, max_revisions: int = 2,
                   macro_graph: dict | None = None) -> tuple[str, dict]:
    """Model cannot edit World, Program, TaskGraph, or Gold Answer."""
    initial = reference_question(spec)
    instruction = {
        "intent": intent_contract(spec),
        "style": style_contract(spec),
        "frozen_scenario": spec["scenario"],
    }
    graph_contract = (
        macro_question_contract(spec, macro_graph)
        if macro_graph is not None else None
    )
    if graph_contract is not None:
        # Never reveal private World values or Gold answers; the Author sees
        # only source roles, operations, dependencies and final answer type.
        instruction["compiled_macro_task_contract"] = {
            key: graph_contract[key] for key in (
                "semantic_pattern", "answer_kind", "output_fields",
                "required_source_roles", "required_operations",
                "comparison_contract", "selection_contract",
                "macro_task_count", "macro_dependency_depth",
                "no_extra_question_conditions",
            )
        }
        instruction["macro_dependency_plan"] = [
            {
                "step": i,
                "depends_on": [
                    next(j for j, upstream in enumerate(macro_graph["tasks"])
                         if upstream["task_id"] == parent)
                    for parent in task["depends_on"]
                ],
                "source_roles": task["source_ids"],
                "operations": [
                    part["op"] for part in task.get(
                        "operator_chain", [{"op": task["operation"]}]
                    )
                ],
            }
            for i, task in enumerate(macro_graph["tasks"])
        ]
    if spec.get("semantic_evidence_contract_version"):
        from .semantic_evidence import compile_semantic_contract
        instruction["semantic_evidence_contract"] = compile_semantic_contract(spec)
    if max_revisions < 0 or max_revisions > 8:
        raise ValueError("revisions must be 0..8")
    failures = []
    for attempt in range(max_revisions + 1):
        author_contract = {
            "user_intent": {
                key: value for key, value in instruction["intent"].items()
                if key not in {"sources", "difficulty", "reasoning_pattern"}
            },
            "materials": [
                {"title": source["title"], "modality": source["modality"],
                 "purpose": source["purpose"]}
                for source in spec["sources"]
            ],
            "style": instruction["style"],
            **({
                "semantic_evidence_contract":
                    instruction["semantic_evidence_contract"],
            } if "semantic_evidence_contract" in instruction else {}),
        }
        if spec.get("question_semantic_lock_version"):
            from .question_semantic_lock import question_author_requirements
            author_contract["mandatory_question_semantics"] = (
                question_author_requirements(spec)
            )
        if attempt > 0:
            author_contract["repair_reference_question"] = initial
            prior = failures[-1]
            prior_errors = prior.get("errors", [])
            author_contract["repair_priority"] = [
                "Keep every mandatory semantic requirement EXACT while "
                "changing natural sentence structure.",
                "Request ONLY the final typed output; identity matching "
                "must be a subordinate clause, not a second answer.",
                "Do not copy the deterministic reference question.",
            ]
            if spec.get("question_semantic_lock_version"):
                if spec["macro_plan"]["goal_resolution"]["kind"] == "calculation":
                    author_contract["repair_priority"].append(
                        "Explicitly state rounded DOWN to a WHOLE percent."
                    )
                if spec["macro_plan"]["pattern"] in {
                    "corroborated_dispute", "multimodal_event",
                }:
                    author_contract["repair_priority"].append(
                        "Preserve the exact independent event-log tag: "
                        + spec["macro_plan"]["clues"]["case_tag"]
                    )
                if ("too_similar_to_reference_question" in prior_errors
                        or "semantic_additional_output_requested" in prior_errors):
                    author_contract["repair_priority"].append(
                        "Use a different single-output sentence pattern: "
                        "'What [single Goal] applies to the record matching "
                        "[all evidence predicates]?'"
                    )
            if "topic_missing" in failures[-1].get("errors", ()):
                author_contract["topic_repair"] = (
                    ("Mention the exact topic naturally in the supplied "
                     "working materials: "
                     if spec.get("scene_realism_version") else
                     "Mention the exact topic as the fictional collection title: ")
                    + spec["scenario"]["topic"]
                )
        user_text = json.dumps({
            "contract": author_contract,
            "previous_failures": [{
                "errors": entry.get("errors", []),
                "critic_reason": entry.get("decision", {}).get("reason"),
            } for entry in failures[-2:]],
            "constraint": "Preserve all conditions and the exact answer contract. "
                          "Prefer a direct natural question, avoid As-a role openings. "
                          "Return a self-contained natural user request: either "
                          "a question ending in '?' or an explicit imperative "
                          "with identify/list/determine/return. Preserve the "
                          "exact grounded topic label and temporal direction.",
        }, ensure_ascii=False)
        try:
            from .prompt_templates import (
                QUESTION_AUTHOR, QUESTION_CRITIC, QUESTION_AUTHOR_SINGLE_SLOT,
            )
            v2 = spec.get("reasoning_pattern") == "macro_first_v2"
            system_prompt = (
                QUESTION_AUTHOR_SINGLE_SLOT if spec.get("scene_realism_version")
                else QUESTION_AUTHOR if v2 else AUTHOR_SYSTEM
            )
            candidate = author.complete_json(system_prompt, user_text)
            raw = candidate.get("question")
            q = (normalize_question_punctuation(raw)
                 if isinstance(raw, str) else raw)
            surface = guard_question(q, spec, macro_graph)
            if (isinstance(q, str)
                    and SequenceMatcher(None, q.strip().casefold(),
                                        initial.casefold()).ratio() > 0.96):
                surface.append("too_similar_to_reference_question")
            if surface:
                failures.append({
                    "stage": "guard", "errors": surface,
                    "candidate_length": len(q) if isinstance(q, str) else None,
                    "candidate_has_question_mark": (
                        "?" in q if isinstance(q, str) else False
                    ),
                    "candidate_preview": (
                        q[:320] if isinstance(q, str) else repr(q)[:120]
                    ),
                    "punctuation_normalized": (
                        isinstance(raw, str) and q != raw.strip()
                    ),
                })
                continue
            critic_prompt = QUESTION_CRITIC if v2 else CRITIC_SYSTEM
            decision = critic.complete_json(critic_prompt, json.dumps({
                "frozen_contract": {
                    **instruction,
                    "frozen_reference_question": initial,
                    "frozen_program_output": spec["requirements"]["answer_kind"],
                },
                "candidate_question": q,
            }, ensure_ascii=False))
            axes = ("all_constraints_preserved", "no_unlicensed_facts",
                    "same_answer_scope", "natural")
            if (decision.get("verdict") == "pass" and
                    all(decision.get(axis) is True for axis in axes)):
                return q.strip(), {
                    "status": "accepted", "attempts": attempt + 1,
                    "punctuation_normalized": (
                        isinstance(raw, str) and q != raw.strip()
                    ),
                    "review": decision,
                    "semantic_equivalence_proven": False,
                }
            failures.append({
                "stage": "critic", "decision": decision,
                "candidate_preview": q[:320],
            })
        except Exception as exc:
            # Do not invent acceptance; preserve the executable reference.
            failures.append({"stage": "backend",
                             "error_type": type(exc).__name__,
                             "error": str(exc)[:240]})
    return initial, {
        "status": "fallback", "attempts": max_revisions + 1,
        "failures": failures,
        "semantic_equivalence_proven": False,
    }
