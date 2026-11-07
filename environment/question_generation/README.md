# Stage 2 — Shared Delegation Blueprint → Natural Benchmark Question

Stage 2 prepares the **same Compact Gold DAG** that Stage 3 will execute,
projects a public delegation outline, writes one short natural-language intent
for each immutable fragment, and uses that shared blueprint to author a
natural benchmark-style Main Question.

## One common plan, three language surfaces

```text
Stage 1: frozen ProgramIR, sources, composition_links
  │
  ├─ compile_gold_solution(schedule="compact")   [deterministic]
  │   └─ fixed fragment IDs, dependency edges, ready waves, outputs
  │
  ├─ compile_delegation_outline                  [PUBLIC projection]
  │   └─ no answer, source IDs, operations' raw private inputs or witness facts
  │
  ├─ naturalize_delegation_blueprint             [one LLM call]
  │   └─ one reusable 1–2 sentence goal per fixed fragment
  │
  ├─ Question Author                            [one LLM call per draft]
  │   └─ goal-first, natural benchmark request; mechanical reference is
  │      literal-spelling backup, not a template
  │
  ├─ cheap deterministic guards + one Critic    [one LLM call per draft]
  │   └─ pass / localized revision / uncertain
  │
  └─ certificate v30
      └─ binds question, original program, public reference and blueprint
          │
          ▼
Stage 3 Phase 1: independently replay certificate and same compact DAG;
                generate Sub-Agent summary/trace_summary from verified
                public evidence, assisted by the fragment's BLUEPRINT goal.
Stage 3 Phase 2: at EACH actual round, use only ready fragment goal(s) and
                completed, public earlier findings to write Main delegations.
```

A **blueprint is teacher-private**. Main never receives the full plan, future
fragment goals or Gold outputs as input. The blueprint's natural task goals
are writing hints, *never ground-truth facts*. Gold answers, intermediate
execution values, source/witness IDs, evidence statements and actual parallel
scheduling all remain controlled by the original compiler and replay.

The blueprint JSON contains `version`, `outline` and `goals`. Immutable
`outline.tasks` include `fragment_id`, `wave`, `depends_on`,
`input_slots`, `output_kind`, `public_steps` and `capabilities`. The
LLM generates **only** the text inside `goals[].goal`; a mismatch in fragment
count/order/ID or unsafe text fails closed. The Stage-2 certificate
(`maestro.stage2_surface_cert.blueprint.v1`, review contract 30) binds
the blueprint digest and the question. Stage 3 recompiles the outline from
the canonical frozen ProgramIR and refuses mismatches or tampering.

**No candidate-only semantic probes, A/B endpoint parser, extra final judge
or artificial edit-distance requirement** is used. The Critic checks genuine
relation, ownership, set-scope, output and user-facing language differences.
This is a bounded model editorial process, not a formal equivalence proof.

## Run a small pilot

```bash
git switch main && git pull origin main
python environment/generate_questions.py \
  --start-record 100 --max-records 50 --max-revisions 2 \
  --output-dir environment/data/questions_blueprint_101_150 \
  --question-workers 4 --flush-every 1
```

Use a **new** output directory; old v25 certs and streaming checkpoints
are not compatible with the blueprint protocol. Generated artifacts include
`realized_capability_worlds.jsonl` (question, blueprint, certificate),
`question_rejected.jsonl`, `question_review_audit.jsonl`, and
`question_generation_summary.json`. `review_statistics.author_calls`
now includes **one extra Author call per blueprint attempt**.

## Quality and limitations

Automatic tests check structure, public/private boundary, replay integrity
and state isolation. They cannot prove that local Qwen naturalizes every
question correctly. Evaluate semantic fidelity, task-style naturalness,
resource reachability, stage-3 bank acceptance, and total model calls.
The model may be more creative than v25 and therefore needs empirical
evaluation on previously unoptimized frozen tasks before scaling to 1120.

Stage-3 commands and phase boundaries are documented in
`../trajectory_simulation/README.md`.
