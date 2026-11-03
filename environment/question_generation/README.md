# Question Generation

Natural-language realization over a frozen programmatic task batch.

```text
frozen manifest + public authoring projection
        ↓
public/private boundary validation
        ↓
LLM author
        ↓
deterministic preflight
        ↓
semantic critic / wording-only repair
        ↓
deterministic postflight
        ↓
validated natural-language question
```

The LLM may change wording only. It may not add facts, resources, proof requirements, hidden criteria, answer hints, or change acquisition visibility.

Example:

```bash
python environment/generate_questions.py \
  --manifest environment/data/frozen_manifest_100.json \
  --authoring-inputs environment/data/authoring_inputs_100.jsonl \
  --worlds environment/data/typed_worlds_100.jsonl \
  --capabilities environment/data/capability_worlds_100.jsonl \
  --output-dir environment/data/questions_100 \
  --provider openai-compatible \
  --base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --model "$QUESTION_MODEL"
```
