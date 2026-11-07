# Stage 2: natural question generation

Stage 1 freezes an executable ProgramIR, a synthetic World and the canonical
task. **Stage 2 does not change the task or the Gold.** It converts the
canonical task into a natural question while preserving the exact public
answer-defining semantics.

There is **one** Stage-2 generation strategy. The old \`--surface-mode\`
choice, bounded-grammar publication fallback, graph-review pipeline,
recursive full-program readback, and their legacy interfaces have been
removed.

## Active sequence

\`\`\`text
frozen Stage 1 task + Controlled Renderer
                   |
       compile Gold (private; no LLM)
                   |
     construct deterministic Scoped question
        (private public-only reference)
                   |
     Author produces complete natural draft
                   |
   Critic compares reference vs actual draft
         |                       |
       pass               revise / uncertain
         |                       |
         |             Author EDITS prior draft
         |             with concrete Critic feedback
         |                       |
         +------ repeat (up to 2 revisions)
                   |
       independent FINAL Judge
     (reference + final text only)
                   |
     +-------------+--------------+
     |                            |
  approved                  not approved /
  publication               uncertain / bad JSON /
     |                      leak / poor naturalness
  accepted                        |
  question                    REJECTED
\`\`\`

The deterministic Scoped question is a **reference only**. It is never
substituted for an unsuccessful natural-language draft. Even a semantically
valid but mechanical copy is rejected. Only \`naturalized_certified\`
questions appear in \`realized_capability_worlds.jsonl\`.

## What each model sees and produces

* **Author**: receives the publicly rendered Scoped reference, which contains
  no private World facts, hidden entity IDs or Gold answers. First call
  returns \`{"question":"..."}\`. A revision call also receives the actual
  previous candidate and the Critic's concrete findings, so that it edits
  the prior draft instead of starting from the same reference.
* **Critic**: receives only the reference question and the current natural
  candidate. It emits \`pass\`, \`revise\` or \`uncertain\` with a concise list
  of \`kind\`, \`severity\`, \`explanation\` and \`suggestion\` for each
  identified problem. For a real omitted date, owner mismatch or reversed
  relation, it must specify which clause needs repair. Natural paraphrase
  and sentence reordering are not reasons to reject.
* **Final Judge**: independently receives the reference and **final candidate
  only**. It does not see prior revisions, Critic votes or private World.
  It judges (i) answer-set equivalence and (ii) substantial improvement in
  naturalness. A pass requires:
  \`verdict=pass\`, \`semantic_equivalence=equivalent\`,
  \`naturalness=improved\` and \`issues=[]\`.
* **Deterministic safety checks**: before calling reviewers, reject or ask
  Author to repair malformed questions, answer leakage, internal IDs,
  false claims of pre-supplied discovery rosters, and other public-boundary
  violations. Type-aware date/number/code differences are presented as
  *advisory* to Critic/Judge, not an automatic exact-string veto.

The two editorial roles can use the same local Qwen3.5-9B checkpoint, but
different prompts and separate inference calls. The Final Judge is
**context-independent**, not a mathematically independent verifier. Its
approval is an auditable model assessment, not a formal proof of semantic
equivalence. For scientific reporting, manually sample approved questions
and test targeted adversarial edits to relations, scopes, quantifiers and
critical clues.

## Publication and rejection

Every frozen task produces exactly one accepted or rejected result. No
fallback question can be counted as a naturalization success.

| Result | Location | Meaning |
| --- | --- | --- |
| \`naturalized_certified\` | \`realized_capability_worlds.jsonl\` | Correct boundary checks, Critic pass and independent Final Judge pass |
| \`rejected_semantic\` | \`question_rejected.jsonl\` | A material mismatch remains |
| \`rejected_uncertain\` | rejected JSONL | Model cannot reliably establish equivalence |
| \`rejected_quality\` | rejected JSONL | Insufficient natural language improvement |
| \`rejected_validation_error\` | rejected JSONL | Model/protocol or public-boundary failure |
| \`rejected_reference\` | rejected JSONL | Program/Controlled Renderer reference unusable |

Detailed per-attempt drafts, Critic feedback, judge decisions and error
causes are recorded in \`question_review_audit.jsonl\`. Because this is a
teacher-side diagnostic, do not expose it to the Main Agent. Stream writes
are atomic, input ordered and resumable.

## Run: local Qwen3.5-9B

\`environment/config.yaml\` already selects \`local_qwen35_9b\` for both
\`question_author\` and \`question_critic\` roles, so no API key or profile
switch is necessary. Start with the SAME first ten frozen Stage-1 tasks:

\`\`\`bash
cd ~/code/MAESTRO
git switch main
git pull origin main

python environment/generate_questions.py \
  --max-records 10 \
  --max-revisions 2 \
  --output-dir environment/data/questions_editorial_v16_10 \
  --question-workers 4 \
  --flush-every 1
\`\`\`

Then inspect \`review_statistics\` and \`rejection_reasons\`, and compare
each accepted question with its Scoped reference in the review audit.
\`accepted\` now counts **only approved free naturalizations**.

\`\`\`bash
python -m unittest discover -s environment/tests -p "test_*.py"
\`\`\`

Stage 3 remains a separate consumer: it reconstructs the canonical task,
compiles Gold from the unchanged ProgramIR, rechecks the Stage-2 certificate
and performs its own public-executability and evidence validation. The new
certificate schema is \`maestro.stage2_surface_cert.v5\` (contract 22).
Previously generated v17/v18 Scoped/Free artifacts are **not supported by
this new validator**; keep their original code/checkpoints for comparison,
or regenerate questions in a new directory.

## Code ownership

* \`realization.py\` — per-task Author/Critic/revision/Final-Judge state.
* \`question_review.py\` — small model prompts and response contracts.
* \`scoped_surface.py\` — deterministic public reference compiler.
* \`public_semantics.py\` — public-only condition/scope projection.
* \`verification.py\` — publication certificate and Stage-3 fingerprint
  and reference replay.
* \`validators.py\`, \`contracts.py\` — public boundary and typed literals.
* \`pipeline.py\`, \`streaming.py\` — CLI, durable ordered outputs, resume,
  summaries and model configuration.

There are no legacy graph or recursive semantic readback publication paths.

## v15 compact risk review (historical)

A frozen 10-task evaluation of v14 yielded 0 accepted questions, primarily
because of REVIEW PROTOCOL errors, not confirmed semantic drift:

* 11 Critic responses did not contain a parseable JSON object.
* 12 "preserved" and 2 "changed/uncertain" evidence quotes failed exact
  substring matching.
* 1 Final Judge omitted the required checklist response.
* The v14 JSON Schema mistakenly required the nested \`checks\` field in
  each individual check, while failing to require it at the final top level.
  That implementation bug has been corrected.

v15 replaces the 9–20 item output form with **at most four review decisions**:
set logic, condition ownership, directed relations, and final answer. Every
original public semantic obligation remains listed in the INPUT under its
category. The model returns one \`preserved/changed/uncertain\` status and a
reason for each category, not 20 byte-identical candidate quotations.
A category is preserved only if ALL its listed requirements survive.
The same local model can still make a false-positive semantic judgment:
grouping is a pragmatic model-facing review format, **not a proof**.

The deterministic safety checks still reject changed public typed
numbers/dates/codes, explicit group intersection-to-union flips, output
boundary violations, and trivial or boilerplate naturalizations. Critic
can request targeted revisions. A malformed Critic/Final response now
retries ONCE on the **same candidate** using simpler JSON-object mode.
If both reviews fail, the task is rejected as \`rejected_validation_error\`;
it does not consume another Author revision just because the Reviewer
failed to format JSON. The private audit retains the backend's finish
reason, token usage, and a truncated model response tail when available.

Run with local Qwen3.5-9B on the same first 10 frozen tasks:

\`\`\`bash
cd ~/code/MAESTRO
git switch main
git pull origin main
python environment/generate_questions.py \
  --max-records 10 --max-revisions 2 \
  --output-dir environment/data/questions_editorial_v15_10 \
  --question-workers 4 --flush-every 1
\`\`\`

Use a NEW output directory. The new Stage 2 certificate is
\`maestro.stage2_surface_cert.v4\` / review contract 21. Stage 3 verifies
the final-review status, grouped risk decisions, typed/public boundaries,
and hashes, without invoking the Reviewer or redoing Gold compilation.
Legacy v13/v14 accepted datasets need their original code or regeneration.

This update should lower PROTOCOL failure rates, but a higher accepted
fraction by itself is not success. Before scaling to 50 or 1120 tasks,
independently assess relation ownership, AND/OR, filter omission,
scope, and genuinely improved naturalness on approved examples and
controlled negative pairs. The local CI runs mocked backends, not an
actual Qwen inference experiment.

## v16 pinned-atomic risk review (previous)

A local Qwen3.5-9B v15 run accepted 4/10 frozen questions; there were no
observed review protocol errors. Audit nevertheless found an accepted
named-after/name-equality substitution and an inconsistent negative verdict.
The 40% figure measures reviewer acceptance, not proven semantic precision.

v16 retains at most four grouped reviewer decisions, but every public semantic
obligation now has a stable atomic requirement ID. Filter obligations name
the publicly reconstructed relation-owner route. When a reviewer returns
changed or uncertain, it must identify one or more problematic atomic IDs
in `problem_ids`; preserved requires an empty list. Unknown, repeated and
missing IDs fail closed. Some clearly self-contradictory negative explanations
are treated as a protocol error and retried on the same candidate.

A narrow public-text guard additionally flags a named-after relation replaced
with name equality, including when the draft retains other named-after
clauses. It requests an Author repair; it is not a general-purpose relation
parser or formal equivalence proof.

The v16 certificate is `maestro.stage2_surface_cert.v5` (review contract 22,
method `llm_pinned_atomic_risk_review_v4`). Old v15 artifacts cannot be
silently reused for Stage 3 and must be regenerated with matching code.

Run on the same ten frozen tasks, with a NEW output directory:

```bash
python environment/generate_questions.py --max-records 10 --max-revisions 2 \
  --output-dir environment/data/questions_editorial_v16_10 \
  --question-workers 4 --flush-every 1
```

CI uses scripted reviewers, not a real Qwen inference run. Manually audit
accepted questions and compare real-model acceptance, semantic precision,
false rejects and call counts before scaling. Counterfactual-world execution
and general independent atomic re-review are **not implemented** in v16.

## v17 candidate-centric semantic probes (previous)

A real local Qwen3.5-9B v16 10-task run accepted 7/10 questions but
three rejected tasks failed Critic protocol validation; the audit only
reported `review_group_results_incomplete`. Independent manual inspection of
the accepted questions found candidate-side scope changes, including:
- Group 2 defined with "either" when the public contract requires BOTH branches.
- A final OR branch accidentally restricted to Group 2, although that branch
  does not require Group-2 membership.
- "The entity itself belongs to Group 4" replaced by "a member of the entity
  belongs to Group 4."

v17 retains the four-group Critic/Final-Judge contract. Before review,
conservative public-text guards reject explicit known scope reversals.
For candidates that pass Critic, v17 additionally invokes selective
**candidate-only semantic probes**. The
probe model sees the candidate text and short multiple-choice questions,
not the reference, expected answers, private World or Gold. The compiler
constructs expected answers from the public ScopedPlan and asks about (a)
labeled group intersection/union, (b) member-vs-entity quantifier scope, and
(c) independent final union branch scope. At most five probes are issued in
one compact model request for a candidate, and only when supported surface
patterns remain explicit in the natural candidate. Models may still misread
these probes; the result is an additional fallible diagnostic, not formal
equivalence verification.

Probe mismatches or UNCLEAR readings request targeted Author revisions.
Malformed/incomplete probe JSON gets one same-candidate retry and is rejected
as validation error if still unusable. Stage 3 independently reconstructs
which probes were required and compares the saved answer/expected pairs.
This new publication record is `maestro.stage2_surface_cert.v6`, review
contract `23`. **Do not reuse v16 certificate files.**

`question_review_audit.jsonl` now retains expected/received risk-group IDs,
received group count and response field names on malformed Critic or Judge
output, without logging any private Gold or World facts. The final summary
reports `candidate_probe_calls` separately from Critic/Final-Judge calls.

Run the SAME 10 frozen tasks in a new output directory:

```bash
python environment/generate_questions.py \
  --max-records 10 \
  --max-revisions 2 \
  --output-dir environment/data/questions_editorial_v17_10 \
  --question-workers 4 --flush-every 1
```

Do not scale up solely on acceptance percentage: manually compare final
accepted questions against their Scoped references, check false positives
and false negatives, and inspect protocol diagnostics. Automated unit tests
use scripted probe readers, not an actual local Qwen checkpoint.

## v18 candidate-side condition ownership and final-pool OR (previous)

Manual inspection of the real local-Qwen v17 10-task run found one clear
accepted semantic error: in the second group of task 3 the condition
population < 1,765 was moved from the SELECTED entity to a related
administrative area, while the AND/OR probe still said AND. The reviewer
also confused SELECT_BETWEEN's final OR with intermediate AND joins in
task 5. Task 8 had review_problem_ids_invalid without logging which IDs
were invalid.

v18 adds **public-plan condition-owner probes** for conservative cases:
an INTERSECT has a branch containing a top-level numeric/date filter on
the selected entity and another branch with a location relation. The
candidate-only reader is asked whether the specified filter belongs to
SELECTED or RELATED. Expected labels come only from the public ScopedPlan,
not from World answers. An explicit 'entities located in both' rewrite
of such a set is additionally detected as a narrow deterministic owner
scope contradiction. Ambiguous/unrecognized paraphrases still require
independent manual audit; this is not a complete English parser.

SELECT_BETWEEN now has a separate set_logic requirement explicitly
declaring the FINAL ranking pool a UNION of independent branches, even
when one branch contains an INTERSECTION. Reviewer instructions warn
against conflating the two. Critic feedback that claims a numeric value
was omitted when that value is visibly present is marked unverified and
redirected to checking its true owner; no failed check is silently passed.
On malformed reviewer output, the audit now includes a bounded per-group
record of invalid or unknown public C-IDs and the type of checks.

The current certificate is maestro.stage2_surface_cert.v7, review
contract 24, method candidate_owner_semantic_probe_review_v6. Issuance
and Stage-3 replay validate the probe answers against public expectations.
Previous v17 certified data must be regenerated with matching code.

Run the first ten frozen tasks in a **new** output directory:

```bash
python environment/generate_questions.py \
  --max-records 10 --max-revisions 2 \
  --output-dir environment/data/questions_editorial_v18_10 \
  --question-workers 4 --flush-every 1
```

The repository CI tests use scripted reviewer responses, not actual
Qwen3.5-9B. Check every accepted output and the 3/5/8 task regressions
before increasing to 50 questions, and do not use 70% reviewer
acceptance as a proxy for semantic precision.

## v19 label-free coverage, group identity and conditional semantics (previous)

On the real ten-question v18 Qwen3.5-9B run (8/10 accepted), audits showed
three material gaps: task 3 removed the literal "second group" label, so
the condition-owner probe did not run; task 5 moved population 1,203 from
a derived subset into the definition of Group 1, and task 9 replaced
"if IATA code BQG, otherwise empty" with "assuming IATA code BQG".

v19 compiles **condition-owner** probes from public relation/attribute
semantics whether or not the candidate retains explicit "Nth group"
labels. These are candidate-only reading questions; the reference and
expected answer never enter the model call. For the high-confidence
"entities exist in administrative areas meeting both" construction, a
public numeric-filter anchor enables a conservative deterministic guard
against moving the selected entity's population condition to a related
administrative area.

Two new targeted probe families are **group_identity**, which asks
whether a downstream population filter has been moved into the broader
first group's definition, and **conditional_scope**, which distinguishes
a live IF/empty-set branch from merely assuming its condition true.
Narrow high-confidence deterministic guards cover the exact v18 forms.
All probes and answers are bound to the Stage-2 publication certificate;
Stage 3 recomputes the public probes and rejects altered records.

The Critic and Final Judge still return up to four grouped risk votes.
Their output C-IDs are now **optional on changed/uncertain votes**: a
negative group without accurately mapped atom IDs still vetoes the draft,
instead of dropping the whole task as a protocol error. Malformed C-IDs
on negative votes are removed (not reinterpreted or treated as evidence)
and noted in the audit. A complete dictionary keyed by R1/R2/... can
be normalized safely to a list; missing groups still fail closed.
All-preserved votes retain their strict pass conditions.

The summary additionally reports `candidates_with_probes`,
`accepted_without_probes`, and `compiled_probe_kinds`; zero probes
are **not** evidence of semantic equivalence. Probe matching is
conservative and does not guarantee full coverage of arbitrary natural
paraphrases. Manually audit certified outputs, especially those without
any candidate-side probes. Local unit tests do not establish model-level
precision.

Certificate: `maestro.stage2_surface_cert.v8`, review contract `25`,
method `candidate_scope_coverage_review_v7`. Existing v18 outputs
cannot be replayed with v19's Stage-3 verifier: regenerate into a
NEW output directory.

```bash
python environment/generate_questions.py \
  --max-records 10 --max-revisions 2 \
  --output-dir environment/data/questions_editorial_v19_10 \
  --question-workers 4 --flush-every 1
```

## v20 coverage-aware semantic certification (historical; 50-task evaluation)

v19 (real local Qwen3.5-9B on the same 10 frozen tasks) accepted 6/10.
Five of its six accepted drafts had NO candidate-only probe call, and the
one probed acceptance missed a directed `location` hop. Consequently,
per-candidate groups, typed literal preservation, Critic agreement and
a `naturalized_certified` label cannot be treated as proof of semantic
equivalence.

v20 adds an independent mandatory **public semantic obligation manifest**
in `coverage_review.py`. Compiling from the frozen PUBLIC `ScopedPlan`
does not depend on whether Author retains mechanical group labels:

- Every distinct supported `RELATE` node gets a candidate-only direction /
  relation-meaning probe: `FORWARD`, `BACKWARD`, `DIRECT`, `OTHER`,
  `UNCLEAR`. These are intended to catch reference relations replaced
  by direct attributes or weaker name/reference associations.
- Every distinct `INTERSECT`/`UNION` node gets a candidate-only
  `AND`/`OR`/`UNCLEAR` probe.
- Every task gets a final answer-type probe (`COUNT`, `ENTITIES`,
  `ATTRIBUTE`, `RELATIONS`, `RANK`, `VERIFY`, or `UNCLEAR`).
- Existing high-confidence public condition-owner, conditional, member
  scope and group-identity probes remain in place.

Review questions include only public, short clause anchors and candidate
text, **not** the full reference, Gold, private World or expected answer
label. The compiler holds the expected answer separately. A mismatched
or `UNCLEAR` response means the candidate is not certified; Author may
revise against feedback within the existing revision budget.

To avoid silent truncation, all required obligations must fit the bounded
compiler budget (24 total). Large/deep unsupported structures fail closed
as `rejected_coverage_unverifiable` rather than dropping checks.
Requests are batched 4 probes at a time, retaining the existing
same-candidate bounded protocol retry. Every accepted candidate must have
a nonempty, **complete** model-checked coverage manifest; no zero-probe
acceptance is allowed. This is model-evaluated *coverage* rather than
formal proof of semantics.

The Stage-2 certificate is now `maestro.stage2_surface_cert.v9` /
review contract `26`; Stage 3 deterministically rebuilds the same
public obligations, checks the manifest hash and every saved answer, and
refuses missing/modified coverage. The resume signature also fingerprints
`coverage_review.py`, `semantic_probes.py` and `semantic_checks.py`.
Older v19 certified outputs are not compatible with v20 Stage-3 replay.

**Evaluate on the FIRST 50 frozen questions, in a new output directory,
not the old first-10 test directory:**

```bash
cd ~/code/MAESTRO
git switch main
git pull origin main
python environment/generate_questions.py \
  --max-records 50 --max-revisions 2 \
  --output-dir environment/data/questions_coverage_v20_50 \
  --question-workers 4 --flush-every 1
```

The summary includes `accepted_with_complete_coverage`,
`compiled_probe_kinds`, `verified_probe_kinds` and
`candidate_probe_calls`. Collect the JSONL audit/accepted/rejected files
and manually inspect **semantic precision, false positives, false
negatives, validation errors, probe coverage and inference cost**. It is
acceptable for the acceptance rate to fall if actual false positives are
caught; a model saying `FORWARD` is NOT ground-truth proof of a relation.
The 50 tasks are an evaluation slice, not held-out once they are used to
tune future versions. No local Qwen inference was run in repository CI.

## v23 Evidence–Semantics Decoupling (current)

The v22 local Qwen3.5-9B 50-task audit (4 accepted, 421 calls) exposed
two separate failure modes. In task 11, the reader repeatedly fabricated
English evidence quotations not found in the natural candidate and the
pipeline silently collapsed otherwise explicit semantic answers to
`UNCLEAR`. In accepted task 29, a recheck changed a logo-owner label
to the public expected A but the quoted owner was still "organizations",
matching the B-side organization in the SAME directed relation.

v23 preserves the candidate-only reader's **raw_answer** separately from
the final normalized `answer`. Every evidence-backed probe stores
`evidence_status` and `evidence_reason`. These distinguish
`verified`, `semantic_mismatch`, `model_uncertain`, and
`evidence_invalid` (e.g. `evidence_not_verbatim`,
`owner_quote_invalid`). All are derived from public candidate text;
no private World/Gold or teacher expected answer is sent to the model.

For an initial raw label matching the teacher-side expected label whose
quote fails mechanical grounding, Stage 2 performs **one evidence-only
repair** (batched) supplying the model its **own previous answer** plus
candidate text, but NOT the reference or expected label. Substantive
mismatches and model uncertainty still get one independent focused
semantic reread. If evidence remains ungrounded, the candidate is
quarantined as `rejected_evidence_unverified`, not labelled a semantic
failure or handed to Author for blind rewrite.

A new stable `edge_id` connects each public relation-path obligation to
its local condition-owner obligations. The cross-check catches an explicit
contradiction when an owner noun phrase matches ONLY the opposite
A/B endpoint in the same model-extracted relation quote. Conflicts are
quarantined as `rejected_evidence_conflict`. This is a conservative
negative check, **not** an automatic positive proof for same-type
endpoints or ambiguous phrasings.

The Stage-2 certificate advances to
`maestro.stage2_surface_cert.v11` / review contract 28, requiring
replayable raw-vs-normalized evidence diagnostics and endpoint coherence.
Stage 3 independently checks them against the frozen public obligations
and exact candidate text. Summary metrics now separate evidence-repair
model calls, original/final grounding states, evidence failure reasons,
and role conflicts.

Do NOT overwrite v22 data. For a new **holdout 51–100** evaluation use:

```bash
cd ~/code/MAESTRO
git switch main && git pull origin main
python environment/generate_questions.py \
  --start-record 50 --max-records 50 --max-revisions 2 \
  --output-dir environment/data/questions_coverage_v23_51_100 \
  --question-workers 4 --flush-every 1
```

Compare the frozen diagnostic first 50 separately if a before/after
comparison is needed, without tuning on the holdout until analysis.
Scripted CI tests do not measure real Qwen-level semantic precision.
