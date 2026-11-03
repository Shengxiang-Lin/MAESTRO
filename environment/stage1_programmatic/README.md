# Stage 1 — Programmatic Composition

This stage is the **unchanged deterministic Stage-1/P0/P1 production path** from the pinned `orchestra-o2` implementation.

The logical submodules are:

1. **source priors** — KQA-Pro, YAGO and MultiModalQA preprocessing;
2. **program composition + semantic binding** — deep ProgramIR composition and Semantic Binding v2;
3. **world + evidence synthesis** — typed worlds, evidence allocation and hardening;
4. **task design + rendering** — frozen task motif/clue/visibility design and deterministic surfaces;
5. **P0/P1 validation + freeze** — scenario validity, capability grounding, orchestration-readiness audits and `p1_frozen_manifest_*`.

The public MAESTRO entry is:

```bash
python environment/run_stage1.py --target 100
```

All arguments are forwarded verbatim to the pinned upstream `python -m environment.build_all`. Therefore existing flags such as `--shortlist-target`, `--candidate-multiplier`, `--resume-downstream`, and explicit KQA/YAGO/MMQA paths retain their current behavior.

Generated artifacts are redirected to `MAESTRO/environment/data/` through a runtime symlink; the upstream source code itself remains untouched.
