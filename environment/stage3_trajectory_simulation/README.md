# Stage 3 — LLM Trajectory Simulation

Stage 3 is intentionally separated from P1.5 question generation.

The current executable baseline reuses the pinned Orchestra-O3 P2 clean-trajectory implementation:

- compile the hidden execution DAG/frontiers programmatically;
- ask a Teacher LLM to synthesize round-by-round orchestration actions;
- validate the trajectory against the environment-authoritative evidence frontier.

Example:

```bash
python environment/run_stage3.py synthesize \
  --worlds environment/data/typed_worlds_100.jsonl \
  --realized environment/data/stage2_questions_100/realized_capability_worlds.jsonl \
  --output environment/data/stage3_trajectories_100/clean_trajectories.jsonl \
  --provider openai-compatible \
  --base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --model "$TRAJECTORY_MODEL"
```

If the first positional command is omitted, `synthesize` is assumed. `plan` remains available for no-LLM execution-DAG compilation.

For MAESTRO, this folder is the future home of controlled sub-agent fallibility, recovery branches, macro labels and counterfactual action values. Those extensions should build on the frozen Stage-1/2 task semantics rather than modifying them.
