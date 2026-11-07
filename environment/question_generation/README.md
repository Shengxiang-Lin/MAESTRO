# Stage 2: Natural Question Editing

Stage 2 converts a **compiler-checked, mechanically worded** task question
into readable natural language while trying to preserve its meaning. This is
an **Author → Critic → local repair** pipeline, not a formal NLP semantic parser.

## Current workflow

```text
frozen Stage-1 ProgramIR + checked canonical question
    │
    ├─ compile/verify deterministic public Scoped Reference
    │
    ▼
Author (local Qwen): conservative naturalization
    │
    ├─ cheap deterministic guards:
    │  • nontrivial lexical edit (punctuation-only does not count)
    │  • required public values, dates and codes retained
    │  • clearly reversed AND/OR or "named after" → same-name change
    │  • public boundary / hidden answer leak protection
    │
    ▼
Critic (local Qwen): compare reference vs candidate
    │
    ├─ pass + readable → publish and certify
    ├─ revise → at most 3 LOCAL text differences + minimal patch instructions
    │              │
    │              └── Author patches the PREVIOUS draft (up to N revisions)
    └─ uncertain / invalid review → quarantine, never pretend it is equivalent
```

**The model receives only public, checked English questions.** The reference
is rebuilt from the frozen task with a verified public projection; private
Gold answers and internal evidence are not sent to the Author or Critic.
The deterministic checks catch only high-confidence errors; the Critic is
responsible for contextual ownership and relation semantics.

This replaces the mandatory relation/owner A/B probe system, evidence
extraction, coverage manifest and independent final judge from v17–v23.
Those two old modules and their version-specific tests were **deleted**,
not merely disabled. There is no per-node model interrogation anymore.

## How the revision feedback looks

```json
{
  "verdict": "revise",
  "naturalness": "acceptable",
  "summary": "The population constraint moved to the neighboring place.",
  "differences": [
    {
      "reference_excerpt": "entities with population below 1770",
      "candidate_excerpt": "neighbors with population below 1770",
      "problem": "Population originally describes the result entity.",
      "minimal_fix": "Attach population below 1770 to the result entities; leave the neighbor clause unchanged."
    }
  ]
}
```

The Critic must distinguish genuine semantic changes from different
wording of the same AND/OR expression. It may pass modest improvements
instead of demanding extreme shortening or restructuring. An
`uncertain` review is kept separate from a confirmed semantic error.

## Reproducible runs

Update the checkout and run the previously debugged 50 tasks in a **new**
directory for a comparable diagnostic run:

```bash
git switch main && git pull origin main
python environment/generate_questions.py \
  --max-records 50 --max-revisions 2 \
  --output-dir environment/data/questions_simple_50 \
  --question-workers 4 --flush-every 1
```

A separate second slice can use `--start-record 50 --max-records 50`
and a different output directory. Do not reuse v23 checkpoint/certificate
files: the review contract is now
`maestro.stage2_surface_cert.simple.v1` (contract 29), method
`simple_author_critic_local_repair_v1`.

Output files:
- `realized_capability_worlds.jsonl`: accepted natural questions only
- `question_rejected.jsonl`: refused tasks and reasons
- `question_review_audit.jsonl`: reference, attempted drafts and local feedback
- `question_generation_summary.json`: acceptance, revisions, Author/Critic
  calls, protocol failures and frozen evaluation slices

## Interpretation and limitations

**No LLM Critic is a perfect equivalence checker.** This simplification
removes a large number of false rejections caused by asking a small model
to construct an artificial A/B semantic parse. It can also let some
subtle scope errors slip past. Treat `naturalized_certified` as
*accepted by a specific, reproducible editorial review protocol*, not as
a mathematical guarantee.

Assess **human-verified semantic precision**, **valid-naturalization yield**,
false positives, false negatives, and total model calls. Keep reference
generation failures (`rejected_reference`) separate from editing quality.
Stage 3 verifies the bound publication certificate and performs its own
evidence/public-executability checks; it does not re-run the Critic.
