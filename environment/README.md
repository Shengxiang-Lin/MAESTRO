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
paths resolve from MAESTRO's root. `{target}` in paths is expanded from
`task_construction.target`. Explicit CLI flags override the YAML defaults.
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

### Local vLLM

The `local_qwen` profile connects to an **already running** local vLLM
server, using the same OpenAI-compatible client. Example server launch:

```bash
vllm serve "$QWEN3_8B_MODEL_PATH" \
  --host 127.0.0.1 --port 8803 \
  --served-model-name qwen3-8b-local
```

Match the YAML `model` with the server's served model name and set
`base_url` to its `/v1` endpoint. Author, Critic and Teacher may use
different backends; for example:

```yaml
question_generation:
  author_profile: dashscope_qwen
  critic_profile: local_qwen
trajectory_simulation:
  teacher_profile: local_qwen
```

The environment HTTP client does **not** automatically launch vLLM. Manage
the server separately or use the repository's shared model infrastructure.
Use the `offline` (passthrough) profile for no-network wiring checks.

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
