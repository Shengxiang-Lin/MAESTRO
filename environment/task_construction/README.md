# Task Construction

Deterministic/programmatic construction of synthetic orchestration tasks.

```text
source priors (KQA-Pro + YAGO + MultiModalQA)
        ↓
deep ProgramIR composition
        ↓
graph-grounded semantic binding
        ↓
executable symbolic world
        ↓
hardening + heterogeneous evidence allocation
        ↓
TaskDesign freeze
        ↓
P0 scenario validity
        ↓
P1 capability / progressive-discovery grounding
        ↓
frozen public authoring projection
```

The symbolic executor is authoritative for task truth. LLMs are not used in this package.

Run from the repository root:

```bash
python environment/construct_tasks.py --reuse-source-priors
```

Important outputs live under `environment/data/`, including `typed_worlds_*.jsonl`, `task_designs_*.jsonl`, `capability_worlds_*.jsonl`, `authoring_inputs_*.jsonl`, and `frozen_manifest_*.json`.

The construction policy follows the former `orchestra-o3/environment` production path, but the code is reorganized by responsibility rather than by historical P0/P1 naming.

The only production size control is `task_construction.candidate_pool_size`
(default: 2000) in `environment/config.yaml`. The program collects up to that
many semantically valid candidates and checks each one against the unchanged
world-quality and diversity gates. It freezes every passing world, without a
fixed accepted-world quota. `frozen_manifest_2000.json` records the actual
`frozen_count` (not necessarily 2000). `--reuse-source-priors` skips
the expensive source scan only after those priors have been prepared.
