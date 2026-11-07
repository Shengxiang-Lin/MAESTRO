# Stage 2 — Certified Subtask Goals → Natural Main Questions

Stage 2 is now **two independently resumable jobs**. Stage 1 freezes
ProgramIR, rendered resources, the canonical question and the Gold fragment
DAG. Neither Stage 2 job edits task semantics or generates Gold answers.

## Stage 2A — independently generate and verify Subtask Goals

\`\`\`text
Frozen ProgramIR + compact Gold DAG
    └─ compile_delegation_outline (deterministic PUBLIC projection)
        └─ each fragment concurrently/independently:
            ├─ Local Goal Author (current public steps + direct dependencies)
            ├─ deterministic checks (numbers, comparisons, foreign branch literals)
            ├─ Local Goal Critic (answer equivalence, relation direction, owners, AND/OR)
            └─ bounded local repair, max --max-goal-revisions
        └─ require ALL fragments PASS; otherwise quarantine whole task
            └─ deterministic DAG assembly + goal_certificate
\`\`\`

The writer and Critic **do not** see the full mechanical reference,
other siblings' authored Goals, private world facts, Gold answers or
future observed results. The LLM returns only \`{"goal": "..."}\`;
fragment IDs, \`depends_on\` and \`waves\` are determined by the compiler.
The Local Critic returns only a verdict, concrete semantic issues, and a
summary. It checks whether answer-defining conditions are preserved,
allowing natural compression of multiple operations. This is an audited
model judgement, NOT proof. Deterministic guards still check obvious
numerical/branch/comparator violations. An invalid Critic JSON response
is retried on the SAME Goal at most once, without regenerating Goal prose.

This is a bounded **automatic fidelity review**, not a formal mathematical
proof of natural-language equivalence. A reviewer may miss semantic
changes; ambiguous or repeatedly failed tasks are quarantined, not promoted.

Run from the MAESTRO root with the existing configured local vLLM:

\`\`\`bash
python environment/generate_goals.py \
  --max-records 50 \
  --max-goal-revisions 2 \
  --output-dir environment/data/stage2a_goals_v2_50 \
  --question-workers 4 \
  --blueprint-workers 2 \
  --flush-every 1
\`\`\`

Stage 2A outputs are separate and checkpointed:

| File | Purpose |
| --- | --- |
| \`certified_blueprints.jsonl\` | Only tasks whose **every** Goal passed, with Blueprint, per-fragment reviews and \`goal_certificate\` |
| \`rejected_blueprints.jsonl\` | Failed tasks and isolation reason; never forwarded |
| \`goal_review_audit.jsonl\` | Every writer/reviewer attempt, local feedback and per-task provenance |
| \`goal_generation_summary.json\` | Task yield, reviewer and author calls, failure reasons |
| \`goal_progress_state.json\` | Crash-safe ordered checkpoint; resumable with identical inputs/code/model settings |

The earlier 50-task Stage-2A pilot used an incompatible v1 reviewer
protocol. Keep its artifacts for analysis and use a new output directory
for this v2 experiment.

To intentionally overwrite an earlier Stage-2A run, add
\`--restart-goals\`. To preserve previous experiments, prefer a new
output directory. A failed Question in Stage 2B does **not** invalidate
the completed Stage-2A Goal dataset.

## Stage 2B — author and review only certified Questions

\`\`\`text
Stage 2A certified_blueprints.jsonl
    └─ join frozen task IDs / discard non-certified originals
        └─ independently rebuild compact Gold DAG and replay 2A certificate
            └─ benchmark Question Author (Goal Blueprint + original contract)
                └─ existing deterministic guards + Question Critic / local repair
                    └─ Stage 2B certified Question, including signed 2A review
                        └─ Stage 3: verified source-based Sub-Agent reports
                                      → just-in-time Main delegations
\`\`\`

\`\`\`bash
python environment/generate_questions.py \
  --max-records 50 \
  --blueprints environment/data/stage2a_goals_v2_50/certified_blueprints.jsonl \
  --max-revisions 2 \
  --output-dir environment/data/stage2b_questions_50 \
  --question-workers 4 \
  --flush-every 1
\`\`\`

**Stage 2B requires \`--blueprints\`.** Of the selected 50 originals, only
the Stage-2A certified task IDs are submitted; the summary includes
\`input_candidates\`, \`skipped_uncertified\`, and
\`certified_input_count\`. A tampered or stale Goal certificate is checked
against the frozen ProgramIR **before any Question Author call**. Each
accepted Question carries the source \`delegation_blueprint\`,
\`goal_reviews\`, \`goal_certificate\` and its own Stage-2B
\`surface_certificate\`. Stage 2B has **zero Goal Author/Goal Critic calls**;
it can be rerun against the same frozen certified Goal dataset to improve
Question prompts without regenerating Goals.

Stage-2B outputs and checkpoints retain the existing names:
\`realized_capability_worlds.jsonl\`,
\`question_rejected.jsonl\`,
\`question_review_audit.jsonl\`,
\`question_generation_summary.json\`,
and \`question_progress_state.json\`.

### Contract versions and Stage 3

- Goal Blueprint: \`maestro.delegation_blueprint.v2\`.
- Stage-2A Goal certificate: \`maestro.stage2a.certified_blueprint.v2\`.
- Stage-2B Question certificate for these Goals:
  \`maestro.stage2_surface_cert.certified_goals.v3\` (review contract 32).
- Existing earlier v2 Question certificates continue to replay for
  compatibility. **New Stage-2B CLI tasks can only publish v3** certificates.
- Stage 3 recompiles the original compact DAG and verifies both
  certificates; its Gold answers, public source replay and Main state
  visibility boundaries are unchanged.

A Goal review certificate represents an audited model judgement, not a
guarantee of perfect semantic fidelity. Continue measuring Stage-2A
fidelity and Stage-2B false-pass/false-reject rates independently on
unseen frozen tasks. No one-to-one natural-language operator mirroring
is required of the final Question; it should remain a coherent user request.

## Performance and troubleshooting

The two stages intentionally have distinct model-call budgets. Stage 2A
uses one Goal writer and up to one local Critic per attempt per Fragment;
\`--max-goal-revisions 2\` allows up to three local attempts for a failed
Fragment. Stage 2B uses Author and Question Critic only.

\`--question-workers\` is task-level parallelism; in Stage 2A,
\`--blueprint-workers\` is additional Fragment-level parallelism. Their
product bounds in-flight Goal requests; use smaller values if local vLLM
is memory constrained. Keep separate output directories for both stages
and use the **same frozen input files and seed** throughout Stage 2A/2B.

The existing Stage-3 commands are documented in
\`../trajectory_simulation/README.md\`.

Run all no-network tests:

\`\`\`bash
python -m unittest discover -s environment/tests -v
\`\`\`
