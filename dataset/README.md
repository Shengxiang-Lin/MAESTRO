# MAESTRO → Orchestra-o1 verl GRPO data

This adapter **does not generate new trajectories**. It exports one existing
Main-Agent decision per row in the structure expected by the original
Orchestra-o1 `compute_score(data_source, solution_str, ground_truth, extra_info)`
reward interface. The policy sees only the versioned O1 Main-Agent prompt;
the gold answer and expert action are **reward-only metadata**.

## Installation

From the repository root:

```bash
pip install pandas pyarrow
# Optional prompt token validation:
# pip install transformers
```

## Export Stage 3 (trusted) data

```bash
python -m dataset.prepare_da_grpo \
  --scenes environment/data/maestro_current_100/questions/scenes.jsonl \
  --base-turns environment/data/maestro_base_orchestration_100/main_turns.jsonl \
  --output-dir dataset/data/da-grpo \
  --dry-run
python -m dataset.prepare_da_grpo \
  --scenes environment/data/maestro_current_100/questions/scenes.jsonl \
  --base-turns environment/data/maestro_base_orchestration_100/main_turns.jsonl \
  --output-dir dataset/data/da-grpo
```

## Optionally include accepted Stage 4 recovery turns

```bash
python -m dataset.prepare_da_grpo \
  --untrusted-turns environment/data/maestro_untrusted_orchestration_100/main_turns.jsonl \
  --output-dir dataset/data/da-grpo
```

To check Qwen's prompt token budget, add
`--tokenizer /path/to/Qwen3-8B --max-prompt-tokens 24576`.
By default the converter retains all *published positive* turns (including
explicitly labeled template completions). Add `--require-llm-expert` to keep all LLM-reviewed regular, completion, and
contract-frozen decisions, excluding deterministic fallback labels.

The exporter tolerates only the verified documentation-only rename in the O1
renderer from MAESTRO PR #137; other renderer hash mismatches remain errors.

Input generation directories must include their `manifest.json` file; the
converter verifies the Scene hashes and uses each run's original
max-attempts/model-alias settings. Stage 4 keeps the branch identity in
`sample_id`. No private Macro/fault/Oracle fields enter the policy prompt.

## Output

- `train.parquet`: columns `data_source`, `prompt`, `reward_model`,
  `extra_info`; `prompt` is an OpenAI-style message list.
- `expert_decisions.jsonl`: an expert-steps index for parity with the
  original O1 launcher precheck. The actual reward uses the Parquet
  `extra_info` reference directly.
- `summary.json`: row, episode, mode and source counts.

The exporter deliberately does **not** add an assistant expert reply to
`prompt` and does **not** roll out Sub-Agents: it reconstructs the same
single-decision observation used in Orchestra-o1's offline GRPO.

Train with [../baseline/da-grpo/README.md](../baseline/da-grpo/README.md).

## Current DA-SFT entrypoint

DA-SFT now consumes the canonical `environment.data_export` adapter directly.
Use `environment/data/unified/adapters/da-sft/{train,eval,test}.parquet` with
the frozen `splits.json`; no DA-GRPO-to-SFT join is needed. See
[`baseline/da-sft/README.md`](../baseline/da-sft/README.md).
