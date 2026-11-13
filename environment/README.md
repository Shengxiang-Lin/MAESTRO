# MAESTRO environment

The supported dataset workflow is **Stage 1 → Stage 2 → Stage 4**.
It does not require a separate Stage-3 Main LLM run or a
`base_episodes.jsonl` input. Stage 4 derives a clean deterministic O1
prefix from accepted Stage-1 scenes and Stage-2 reviewed reports.

## One command

From the repository root:

```bash
python -m environment run \
  --questions-config environment/configs/questions.yaml \
  --reports-config environment/configs/subagent_reports.yaml \
  --untrusted-config environment/configs/untrusted_orchestration.yaml \
  --mode llm
```

Stage outputs and resumable journals stay in the locations configured by
their YAML files. Stage 1 creates the frozen Scene/World/Question bank;
Stage 2 materializes executable Macro tasks and naturalizes Sub-Agent
reports; Stage 4 samples one genuinely executable untrusted fault, runs
causal replay, reviews natural-language reports and Main reasoning, and
publishes only completely recovered and independently reviewed SFT turns.

## Run from existing Stage 1 and Stage 2 artifacts

```bash
python -m environment generate-questions \
  --config environment/configs/questions.yaml

python -m environment naturalize-subtasks \
  --config environment/configs/subagent_reports.yaml

python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --mode llm \
  --output-dir environment/data/maestro_untrusted_rounds_v10_4_2
```

The two `input` paths in the Stage-4 YAML must point to the actual
`questions/scenes.jsonl` and reviewed `subtasks.jsonl` created by
the first two stages. The single `run` command wires these paths
automatically.

The Stage-4 template mode supports fast local diagnostics without an
LLM backend:

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --mode template \
  --max-scenes 2 \
  --output-dir environment/data/smoke_stage4
```

## Module boundaries

- `generate_question/`: Stage-1 Scene, World, evidence sources and task DAG
- `implement_subtask/`: Stage-2 source-grounded Macro execution and reports
- `untrusted_orchestration/trusted_prefix.py`: **internal deterministic**
  clean Main history builder from Stage-1/2 inputs
- `untrusted_orchestration/faults.py`, `atomic_faults.py`,
  `scenario_generator.py`: Stage-4 fault and causal scenario compiler
- `untrusted_orchestration/state_machine.py`,
  `evidence_planning.py`: Stage-4 executor, evidence state and rechecks
- `untrusted_orchestration/report_author.py` and `report_contract.py`:
  Stage-4 Sub-Agent naturalization and factual delivery contract
- `untrusted_orchestration/main_decision_contract.py`,
  `reasoning_fidelity.py`, `offline_pipeline.py`: Stage-4
  Main-visible facts, guarded reasoning and reviewed-turn materialization
- `untrusted_orchestration/publication_guard.py`: independent final
  release check, including reviewer and internal-planning prose leaks
- `base_orchestration/trajectory_builder.py`, `action_author.py`,
  `o1_compat.py`: shared deterministic O1 format and action validators
  reused by the Stage-4 bootstrap and reviewers

Older standalone trusted-Main research code is not part of the public
`python -m environment` workflow. It is not invoked to generate Stage 4
training data.

## Trust and publication invariants

Only Main-visible reports and granted sources can justify a verification
decision; Gold/private fault annotations never enter the Main prompt.
Single-fault scenarios can propagate through multiple dependent tasks.
A synchronous delegated batch finishes before Main acts again. If a
recheck changes an upstream result, its dependent WorkProducts become
stale and must be recomputed. A critic cannot automatically override a
verified visible discrepancy, and cannot auto-pass its own rereview.
Reasoning should be naturally generated, not padded with compiler
templates or private scheduler details.

The current Scene bank may have zero independent same-claim cross-source
witnesses; this is a Stage-1 evidence design limitation for future work,
not permission to synthesize private corroborating evidence in Stage 4.
