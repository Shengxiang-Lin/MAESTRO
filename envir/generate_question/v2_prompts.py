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
