# Stage 2 — LLM Question Generation

This stage is the **unchanged P1.5 authoring pipeline** from the pinned `orchestra-o2` environment, separated from trajectory generation.

It performs:

```text
frozen P1 manifest + public authoring projection
        ↓
frozen-boundary validation
        ↓
LLM question author
        ↓
deterministic preflight
        ↓
semantic critic / repair
        ↓
deterministic postflight
        ↓
validated natural-language question
```

The Stage-1 freeze invariant remains authoritative: the LLM may rewrite wording, but may not add facts, resources, proof requirements, hidden criteria, or redesign resource visibility.

Run the same arguments used by the original `environment.llm_pipeline`; MAESTRO automatically adds `--questions-only`:

```bash
python environment/run_stage2.py \
  --manifest environment/data/p1_frozen_manifest_100.json \
  --authoring-inputs environment/data/p15_authoring_inputs_100.jsonl \
  --worlds environment/data/typed_worlds_100.jsonl \
  --capabilities environment/data/capability_worlds_100.jsonl \
  --output-dir environment/data/stage2_questions_100 \
  --provider openai-compatible \
  --base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --model "$P15_MODEL"
```

The actual author, critic, retry and deterministic validation logic is executed from the pinned upstream implementation without modification.
