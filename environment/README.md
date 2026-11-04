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


## Configuration: environment/config.yaml

All three entrypoints read `environment/config.yaml` by default. It
contains task-construction parameters, frozen question-generation inputs,
trajectory outputs, generation settings, and model profiles. Repository-relative
paths resolve from MAESTRO's root. `{pool_size}` in paths is expanded from
`task_construction.candidate_pool_size`. Explicit CLI flags override the YAML defaults.
Use `--config path/to/another.yaml` to select another configuration.

Install the configuration dependency:

```bash
pip install -r environment/requirements.txt
```

### Remote API

The `dashscope_qwen` profile uses an OpenAI-compatible API. Export its
API key instead of putting secrets into YAML:

```bash
export DASHSCOPE_API_KEY="YOUR_API_KEY"
```

Choose this profile for an individual role:

```yaml
question_generation:
  author_profile: dashscope_qwen
  critic_profile: dashscope_qwen
trajectory_simulation:
  teacher_profile: dashscope_qwen
```

### Local vLLM (default: Qwen3.5-9B)

By default, Question Author, Semantic Critic, and Trajectory Teacher all use
the **same** `local_qwen35_9b` profile in `environment/config.yaml`:
model `local-qwen3.5-9b` at `http://127.0.0.1:8807/v1`. The model profile
corresponds to `config/model_config.yaml`'s `local-qwen3.5-9b` service.

Use a Qwen3.5-compatible version of vLLM (at least v0.17.0 according to the
vLLM model recipe). Start one service **before** running question/trajectory
generation. The model path may be a local downloaded checkpoint directory or
the Hugging Face model ID:

```bash
export QWEN35_9B_MODEL_PATH="/path/to/Qwen3.5-9B"  # or Qwen/Qwen3.5-9B

# GPU 1 in the repository's shared config; change to GPU 0 if applicable.
CUDA_VISIBLE_DEVICES=1 vllm serve "$QWEN35_9B_MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8807 \
  --served-model-name local-qwen3.5-9b \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The reduced context size and text-only mode are practical starting points
for 24 GB GPUs, not a guarantee against out-of-memory errors. Adjust memory
settings and concurrency as needed. Disabling thinking is particularly
useful here because the Author/Critic/Teacher produce structured JSON.
Qwen3.5-9B full BF16 requires substantial VRAM; on smaller GPUs consider
a supported quantized checkpoint or more hardware.

Verify that the server advertises the exact model name:

```bash
curl -s http://127.0.0.1:8807/v1/models
```

Test one structured response before a large data-generation batch:

```bash
curl -s http://127.0.0.1:8807/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-qwen3.5-9b","messages":[{"role":"user","content":"Return a JSON object with the field ok set to true."}],"response_format":{"type":"json_object"},"max_tokens":128,"temperature":0}'
```

The environment client accesses a **running** vLLM HTTP service; it does not
launch or download model weights automatically. To run it on another machine,
expose the server securely and change `models.local_qwen35_9b.base_url`.
You can use different profiles per role or switch to `dashscope_qwen` by
changing the three `author_profile`, `critic_profile` and
`teacher_profile` settings. The `offline` (passthrough) profile remains
available for no-network checks.

### Commands

Run from the repository root. All paths and generation settings come from
the YAML file unless explicitly overridden:

```bash
python environment/construct_tasks.py
python environment/generate_questions.py
python environment/simulate_trajectories.py plan
python environment/simulate_trajectories.py synthesize
```

No-network checks:

```bash
python environment/generate_questions.py --provider passthrough
python environment/simulate_trajectories.py synthesize --provider passthrough
```

Legacy `--model`, `--critic-model`, `--base-url` and `--api-key-env`
flags remain available for CLI overrides; roles can have fully independent
endpoints through YAML profiles. Use `--no-json-mode` if the server does not
support JSON response formatting.

The root `config/model_config.yaml` remains MAESTRO's shared orchestration
model registry. `environment/config.yaml` is scoped to synthetic data
generation and does not alter `baseline/` or Main/Sub-Agent settings.

### Troubleshooting: no executable worlds / missing frozen manifest

If `construct_tasks.py` reports `World generation accepted no worlds`,
then the downstream P0/P1 freeze did not finish: `frozen_manifest_{pool_size}.json`
will not exist, so question generation cannot start. Examine the actual
candidate and world rejection reasons:

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path("environment/data/build_summary.json")
d=json.loads(p.read_text())
for section in ("candidate_selection", "worlds"):
    x=d.get(section, {})
    count = x.get("validated_candidates") if section=="candidate_selection" else x.get("accepted")
    print(section, "accepted=", count,
          "rejections=", x.get("rejection_reasons"))
PY
```

For an already completed source-prior build, reuse the expensive KQA/YAGO/MMQA
artifacts without repeating the YAGO scan:

```bash
python environment/construct_tasks.py --reuse-source-priors
```

This only skips source-prior preparation. It reuses a compatible candidate
checkpoint (or creates a new one), evaluates every candidate, and then attempts P0/P1 freeze. If the
new run still rejects all worlds, review
`environment/data/typed_worlds_{pool_size}.jsonl.quality_rejected.jsonl`
for the concrete validation reason rather than proceeding to questions.

## Fixed validated candidate pool

Task construction has **one** mode: choose a candidate pool capacity and
evaluate every validated semantic candidate; retain however many pass.

```yaml
task_construction:
  candidate_pool_size: 2000
  seed: 17
```

The construction flow is:

1. ProgramIR composition + graph-grounded semantic binding collects **up to
   2000 validated candidates**. The bounded proposal attempt budget means
   this count may underfill; rejected raw proposals do not count toward the
   pool. There is **no 8x multiplier, second shortlist, or world quota**.
2. World Generation tests every collected candidate, with up to three
   deterministic world-synthesis attempts each. All existing evidence,
   semantic-support, symbolic execution, TaskDesign, scenario, capability,
   rendering and relative diversity checks remain enabled.
3. P0/P1 downstream audits and Freeze run over the worlds that actually pass.
   `frozen_manifest_2000.json` contains their **actual** `frozen_count`.
4. The LLM Author + Semantic Critic then evaluate each frozen task.
   `questions_2000/question_generation_summary.json` reports the actual
   accepted-question count; this may be smaller than `frozen_count`.

The `2000` filename suffix identifies the **configured pool capacity**, not
the number of valid worlds/questions. Inspect these metrics:

- `build_summary.json`: `candidate_selection.validated_candidates`,
  `candidate_selection.pool_filled`, `worlds.candidates_tested`,
  `worlds.accepted`, `worlds.rejection_reasons`.
- `frozen_manifest_2000.json`: `candidate_pool_size`, `frozen_count`.
- `questions_2000/question_generation_summary.json`: `requested`,
  `accepted`, `rejected`.

Run from the repository root after preparing source priors:

```bash
python environment/construct_tasks.py --reuse-source-priors
python environment/generate_questions.py
python environment/simulate_trajectories.py synthesize
```

To change pool size, edit `task_construction.candidate_pool_size` in
`environment/config.yaml` before running all three stages (or pass the same
custom `--config` file to every command). Avoid changing only the construction
CLI flag: question/trajectory paths expand `{pool_size}` from the YAML pool size.

To change the number of available proposals without relaxing quality
checks, use `--candidate-max-attempts N`. The candidate generator supports
`--candidate-workers N`, `--candidate-proposal-batch-size N`, and
`--restart-candidates`, plus `--world-attempts-per-candidate N` for world
synthesis retries. Source priors are read from `environment/data/` and can
be reused, avoiding repeated KQA/YAGO/MMQA scans.

### Parallel candidate checkpoint and recovery

Parallel binding defaults to `min(8, os.cpu_count())` worker processes. The
parent schedules proposals and commits accepted candidates in deterministic
proposal order, independent of completion order. Worker binding uses the
existing graph-grounded semantic validity checks. The checkpoint is updated
after each complete batch.

Files are placed beside `semantic_candidates_pool_2000.jsonl`:

- `*.candidate-progress.jsonl`: validated candidates from committed batches.
- `*.candidate-state.json`: attempt count, RNG state, committed bytes and hash.
- `*.candidate-state.lock` / `*.candidate-output.lock`: advisory process locks.

Rerun the same construction command to reuse a compatible checkpoint. Source
contents, relevant code and candidate-generation settings are hashed and
verified. If you modify those inputs or change pool size, start a fresh
checkpoint with `--restart-candidates`; the expensive source priors can still
be reused. World Generation, Freeze and question generation are separate
stages, not part of the candidate checkpoint.

The retained semantic candidate pool is `semantic_candidates_pool_2000.jsonl`.
A second duplicate `*.reservoir.jsonl` file is no longer generated.

Run no-network regression tests:

```bash
python -m unittest discover -s environment/tests -v
```

The quality thresholds have **not** been relaxed. An accepted count below
the pool size means the remaining candidates failed one or more enabled
checks; it is not a reason to bypass quality validation.
