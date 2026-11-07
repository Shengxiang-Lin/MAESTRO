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

## v15 compact risk review (local Qwen3.5-9B)

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

## v16 pinned-atomic risk review (current)

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
