# MAESTRO Environment

The data-generation environment is organized as three explicit stages:

```text
environment/
├── stage1_programmatic/             # ① programmatic composition + P0/P1 freeze
├── stage2_question_generation/      # ② LLM question realization (P1.5 only)
├── stage3_trajectory_simulation/    # ③ LLM trajectory simulation (P2)
├── run_stage1.py
├── run_stage2.py
├── run_stage3.py
└── data/                            # generated artifacts (runtime)
```

## Why the old implementation is pinned instead of copied

The current `orchestra-o2/orchestra-o3/environment` implementation contains many tightly coupled modules. Stage 1 and Stage 2 are already implemented and heavily validated. Rewriting them during the MAESTRO migration would make it difficult to tell whether a changed dataset came from the intended reorganization or from accidental logic drift.

Therefore MAESTRO pins the exact upstream implementation as the private git submodule `vendor/orchestra-o2` at commit:

```text
28a3b520cbfb71223e72aa90657b0893e45db683
```

The three MAESTRO stage runners are a clean facade over that frozen code. Stage 1 and Stage 2 forward to the original production modules; only the former combined Stage-2/Stage-3 entry is split so that P1.5 always runs with `--questions-only` and P2 is invoked separately.

Initialize once after cloning:

```bash
git submodule update --init --recursive vendor/orchestra-o2
```

## Stage 1 — programmatic composition

Equivalent to the current upstream:

```bash
PYTHONHASHSEED=0 python -m environment.build_all --target 100
```

through the MAESTRO entry:

```bash
python environment/run_stage1.py --target 100
```

This retains the complete existing sequence: frozen source priors → ProgramIR composition → Semantic Binding v2 → world/evidence synthesis → TaskDesign → rendering/audits → P0 → P1 → frozen manifest.

## Stage 2 — LLM-generated questions

Run P1.5 only:

```bash
python environment/run_stage2.py ...
```

The runner uses the original author + semantic critic + deterministic pre/postflight implementation and always forces `--questions-only`.

## Stage 3 — LLM-simulated trajectories

Run the existing clean trajectory Teacher synthesis independently:

```bash
python environment/run_stage3.py synthesize ...
```

This is the extension point for MAESTRO's later controlled-unreliability and recovery trajectory generator.

## Data placement

The original deterministic build hard-codes its output under its package `environment/data`. At runtime MAESTRO creates a symlink from the pinned package's `environment/data` to this repository's `environment/data`, so the upstream implementation stays unchanged while generated artifacts live with MAESTRO.

If `MAESTRO/data/world-source/` exists, it is similarly exposed to the pinned runtime at the path expected by the original code. Otherwise pass the original Stage-1 source-path flags explicitly.

## Reproducibility contract

- Stage 1 and Stage 2 algorithm code is pinned to the upstream commit above.
- Stage 2 cannot invoke Stage 3 accidentally.
- Stage 3 cannot mutate Stage-1 frozen task semantics.
- `baseline/` is not involved in environment generation.
