# MAESTRO Synthetic Environment

The environment is organized by function rather than pipeline stage numbers:

```text
environment/
├── task_construction/          # deterministic/programmatic task-world construction
├── question_generation/        # LLM question authoring over frozen task semantics
├── trajectory_simulation/      # LLM trajectory synthesis and recovery data
├── construct_tasks.py
├── generate_questions.py
└── simulate_trajectories.py
```

`task_construction/` and `question_generation/` are migrated from the production implementation in `orchestra-o2/orchestra-o3/environment` at commit `28a3b520cbfb71223e72aa90657b0893e45db683`. The migration preserves the production algorithms and freeze/boundary invariants while reorganizing modules under functional names.

The deterministic construction path remains:

```text
KQA-Pro + YAGO + MultiModalQA priors
        -> ProgramIR composition
        -> Semantic Binding v2
        -> executable synthetic worlds + evidence
        -> TaskDesign
        -> P0 scenario validation
        -> P1 capability grounding
        -> frozen P1 manifest
```

Question generation remains a wording-only operation over the frozen P1 contract:

```text
frozen public projection
        -> boundary validation
        -> LLM author
        -> deterministic preflight
        -> semantic critic / repair
        -> deterministic postflight
        -> validated natural-language question
```

Trajectory simulation is separate from question generation. Its initial implementation mirrors the existing clean-trajectory planner/Teacher pipeline and is the extension point for MAESTRO-specific unreliable-feedback, recovery, macro-label and counterfactual-value generation.

Generated artifacts live under `environment/data/`. Source datasets are expected under `data/world-source/` by default, matching the previous implementation.
