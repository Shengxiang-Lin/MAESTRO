# Stage 1 — canonical V1.2

Run from the repository root:

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
```

The single supported generation configuration enables typed equivalent
operator variants, diverse numeric Goal witnesses and exact one-output
Question semantics for Medium/Hard Macro-first V2 scenes. Easy remains
a one-source approval baseline.

The persisted `reasoning_pattern: macro_first_v2` is a **data-format
identifier**, not a reference to the retired v1 source generator.
Stage 2 and Stage 4 depend on this frozen Scene and its FactRefs.

Key modules:

- `macro_planning.py`, `operator_grammar.py`: frozen Macro DAG and
  bounded typed query plans, checked for per-Macro witness equivalence
- `macro_evidence_world.py`, `numeric_goal_world.py`: synthetic
  sources, numeric Gold and source counterfactual mutation
- `question_semantic_lock.py`, `semantic_evidence.py`:
  Question/Program consistency checks
- `world.py`, `compiler.py`, `quality.py`: execution, replay,
  provenance and difficulty diagnostics
- `stages.py`, `pipeline.py`: independent Author/Critic reviews,
  ordered journals and quality reports.

`generation.target` is a candidate count, not an accepted quota.
The versioned output directory must not resume an older pilot checkpoint.
