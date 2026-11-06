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
  --output-dir environment/data/questions_editorial_v13_10 \
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
certificate schema is \`maestro.stage2_surface_cert.v2\` (contract 19).
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
