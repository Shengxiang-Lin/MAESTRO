# Trajectory Simulation

Dependency-aware orchestration trajectories over frozen tasks and generated questions.

The current implementation first compiles environment-authoritative acquisition frontiers, then asks a Teacher LLM to decide how to organize the work at each frontier. Evidence itself always comes from the frozen environment; the Teacher is not allowed to invent facts or future resources.

Plan only:

```bash
python environment/simulate_trajectories.py plan \
  --worlds environment/data/typed_worlds_100.jsonl \
  --realized environment/data/questions_100/realized_capability_worlds.jsonl \
  --output environment/data/trajectory_plans_100.jsonl
```

Synthesize clean trajectories:

```bash
python environment/simulate_trajectories.py synthesize \
  --worlds environment/data/typed_worlds_100.jsonl \
  --realized environment/data/questions_100/realized_capability_worlds.jsonl \
  --output environment/data/trajectories_100.jsonl \
  --provider openai-compatible \
  --base-url "$OPENAI_COMPATIBLE_BASE_URL" \
  --model "$TRAJECTORY_MODEL"
```

This package is also the extension point for controlled natural sub-agent fallibility, recovery branches, macro-action labels, and counterfactual action-value targets used by the trainable MAESTRO Main.
