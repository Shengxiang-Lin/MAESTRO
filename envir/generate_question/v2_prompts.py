"""V2-specific Author/Critic prompts.

The benchmark World is synthetic for reproducibility, while a user-facing
task is phrased like a real information investigation. These prompts do
NOT assert any named real-world event or publication actually occurred.
"""

SCENARIO_AUTHOR = """Write a credible information-work scenario and source
titles for the FROZEN benchmark contract. The provided materials exist
inside the controlled task workspace, not necessarily in public reality.

The external user-facing scene must NOT say fictional, hypothetical,
simulated, imaginary or synthetic. The dataset's internal construction
method is not part of a user's question. Treat any real-world named topic
as a subject or collection label, NEVER as proof of any real event, date,
record release or organization action.

Keep the exact topic; all source IDs, modalities and source roles;
the exact typed Goal and its meaning; the video-action, audio-spoken
cue and time-order conditions; and all required identity constraints.
Never substitute a founding, award, release or match year for a different
year target. Do not add unsupported hard numbers, dates, historical facts,
eligibility requirements, place names, or outcomes.

The semantic_evidence_contract in frozen_contract is authoritative. Only
necessary_question_clues and applied_predicates are answer-defining.
Random clues absent from that contract must NOT be reintroduced. The
presence of an index, final archive or Hard difficulty NEVER licenses
clearance, approval, certification or active-only requirements. Do not
name a clearance/approval register unless the executed predicates truly
use that field. A video action-time threshold is NOT a time difference
between footage and audio, nor a timestamp of the spoken cue.

Make each source title fit its modality and purpose, e.g. footage review,
commentary transcript, event log, case inventory, index or result register.
Give the user a concrete motivation for cross-referencing the materials,
rather than mentioning a database or solver.

Return JSON only:
{"scenario":{"setting":"natural scene mentioning exact topic",
"user_role":"...","purpose":"...","motivation":"...",
"entity_kind":"short plural noun"},
"source_titles":[{"source_id":"exact id","title":"..."}]}
"""

SCENARIO_CRITIC = """Judge the candidate against the frozen contract, not
against facts on the public web. The source observations are intentionally
constructed for this benchmark. A normal working scenario about those
provided materials is permitted and is NOT an unsupported assertion that
a corresponding historical event occurred.

The compiled semantic_evidence_contract identifies source modalities,
exact available fields, executed comparisons and absent filters. Treat it
as decisive if a source title or setting introduces a clearance registry,
audio absent from all sources, a fabricated emblem in video, or a temporal
window that no Solver actually checks.

Reject only when: the named topic is replaced, a real-world historical
event or public source is invented, the typed Goal's meaning changes,
the modality/source contract changes, a necessary temporal or cross-source
condition is lost, or an extra rule is introduced. Prefer topic-coherent
working materials to arbitrary fictional archive stories. Reject
meta labels (fictional/simulated/synthetic/hypothetical/imaginary) in the
user-facing narrative.

Return JSON only:
{"verdict":"pass|revise|reject","topic_coherent":true,
"same_relation_semantics":true,"source_faithful":true,
"no_unlicensed_facts":true,"reason":"one sentence"}
All four booleans must be true for pass.
"""

QUESTION_AUTHOR = """Write one realistic, self-contained user question from
the supplied frozen Goal, scenario and source clues. Return JSON only:
{"question":"..."}

The controlled benchmark World is synthetic but the QUESTION must read as
an ordinary real-world information request. Do not use fictional,
simulated, hypothetical, imaginary or synthetic. Do not make assertions
about historical events, named people's actions or actual media files.
You can describe the supplied working materials and ask what they show.

The semantic_evidence_contract supplied with the user contract is the
binding specification: use its necessary_question_clues, actual source
modalities, applied_predicates and temporal_semantics, and NEVER assume
other sampled plan features were used by the Gold Solver. A hard task
does NOT imply clearance; an archive does NOT imply approved status.
If footage has only action and timestamp fields, do not describe a
colored emblem visible in the footage. If no audio source exists, do
not ask for an audio cue. If video is filtered at or after 45 seconds,
that does not imply speech is heard 45 seconds later.
For a footage<=commentary constraint, express relative ordering only,
never an exact elapsed time or arbitrary one-minute window.

Preserve the exact type and MEANING of the requested field: e.g. match
result is not a match year; venue is not a team name; review status is
not a founding year. Express the answer field using natural words.
For image tasks preserve the visual color and motif. For video preserve
the action and timing threshold. For video + audio preserve the action,
spoken cue, footage-before-commentary time order and independent log tag.
For conflicts preserve the mismatch in reported years even if the final
answer is non-year. Do not require the user to enumerate Macro Task steps.

Keep the exact topic. Do not use internal IDs, field names, source keys,
program operators, task graphs or trace terminology. Do not add irrelevant
facts, deadlines, dates, claims or source content. Write a direct question
or a clear imperative ending with a question mark.
"""

QUESTION_CRITIC = """Independently verify that the candidate question has
the same requested outcome and all source-defining conditions as the
frozen contract. The underlying working materials are synthetic by
design; describing their hypothetical contents is not a claim about
real public historical events. The question must NOT tell the user that
the benchmark is fictional or simulated.

Use the frozen semantic_evidence_contract to reject ALL additional
clearance, approval, audio, image, video, emblem color/shape, and
numeric time claims that are absent from actual executed predicates.
A timestamp on a video is not an audio timestamp, and a relative
before/after comparison does not authorize an exact 45-second gap.
Judge what the question says, not what a plausible source title suggests.

Check that the topical subject remains exact, answer type and semantics
are unchanged, image/video/audio cues and relevant time ordering remain,
disagreements remain disagreements, and no unsupported fact or extra
filter is asserted. A readable paraphrase is allowed. Intermediate
lookup keys, internal Macro IDs and database operators need not appear.

Return JSON only:
{"verdict":"pass|revise|reject",
"all_constraints_preserved":true,
"no_unlicensed_facts":true,
"same_answer_scope":true,
"natural":true,
"reason":"one sentence"}
All four booleans must be true for pass.
"""


SCENARIO_AUTHOR_SCHEMA_LOCKED = """Write ONLY the natural scenario narrative for a
specific domain-grounded information task. You are NOT allowed to name,
rename, invent, or select source files. All source titles, modalities,
roles and field schemas are ALREADY FROZEN by the environment and will be
provided to the Critic unchanged. Output exactly:
{"scenario":{"setting":"...","user_role":"...","purpose":"...",
"motivation":"...","entity_kind":"..."}}

DOMAIN COHERENCE: Use frozen_contract.domain_context.domain and
domain_context.activity, material_kind, source_modalities and goal_label.
The main activity, user role and requested Goal must fit that domain:
a football match-result request should involve a competition or match,
not municipal vehicle depots; an exhibition-year request should concern
an exhibition, not a fabricated match result. The topic is a proper name,
not evidence that any claimed historical match or exhibition took place.

Describe only a plausible task using supplied working evidence: why
the user needs the exact domain-specific final Goal, which distinct
working material roles are relevant, and why more than one source is
needed. Do not introduce a second requested output field, alternate
historical events, named real-world actions, dates, numeric values, false
clearance statuses, or media content not present in the frozen contract.
The internal world is synthetic, but DO NOT use words fictional, simulated,
synthetic, hypothetical, or imaginary in the scenario.

Use precise words for a video timeline, audio utterances, visual image
observations, written notes and keyed indexes as applicable, without
claiming to have actually watched media or opened a public website.
No source_titles field is needed; any such field will be discarded.
"""


QUESTION_AUTHOR_SINGLE_SLOT = QUESTION_AUTHOR + """
ONE-OUTPUT REQUIREMENT (V3): The Gold Answer provides exactly one
typed field. If identity has to be found internally, phrase identity
matching as a subordinate condition, not as a second answer requested
from the user. Prefer 'For the item matching [clues], what is the
confirmed [Goal]?' or 'What is the confirmed [Goal] of the event
identified by [clues]?'.

NEVER ask 'which case ... and what is the status?', 'which episode ...
and what production year?', 'name the item and report its year', or
any coordinated output list. The subject can be *internally* identified,
but the question must request ONLY the final typed Goal. Keep the true
source modalities and clues, and avoid 'case' or 'archive' boilerplate
unless appropriate to the selected domain. A domain-specific scene is
not a license to invent historical facts or new source requirements.
Keep the exact required source conditions while varying surface syntax.
"""

QUESTION_AUTHOR_SINGLE_SLOT += """
STRICT QUESTION PREDICATE FIDELITY (new V2):
- A frame_time_ms >= 60000 predicate means the relevant frame/action is
  seen AT OR AFTER the 60-second timestamp. It does NOT say that the
  action lasted for 60 seconds, started exactly at 60 seconds, or first
  appeared at 60 seconds. Keep inclusive inequality semantics explicit.
- footage_time_ms <= commentary_time_ms means the video event occurs NO
  LATER THAN the spoken audio cue (simultaneity is allowed). Do not add
  a fixed elapsed interval or claim the event happened strictly before.
- Identity matching is a condition, not a second returned answer.
  Do not ask which clip, which game, or which record AND what final goal.
  Phrase instead: 'For the video clip matching [evidence], what is the
  confirmed [single Goal]?'
- Never change a source observation into a duration, first occurrence,
  date, real media action, or an unverified additional filter.
"""
