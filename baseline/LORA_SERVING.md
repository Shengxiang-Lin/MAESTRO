# DA-SFT / DA-GRPO: online PEFT LoRA inference with Qwen3 Thinking

This is the **existing Orchestra-o1 Main-Agent runner**, not a new
inference backend. It routes OpenAI chat requests to a managed vLLM service
with the ORIGINAL Qwen3-8B base and the requested method's PEFT adapter.

## 1. Training outputs

The following files must exist after SFT/GRPO training:

- `base_models/Qwen3-8B/` (the same base as training)
- `baseline/da-sft/output/final_adapter/adapter_config.json`
- `baseline/da-sft/output/final_adapter/adapter_model.safetensors`
- `baseline/da-grpo/output/final_adapter/adapter_config.json`
- `baseline/da-grpo/output/final_adapter/adapter_model.safetensors`

Run a method only after **that method's adapter** is saved. For alternate
experiments, use `--adapter` for single-decision testing or `ADAPTER_PATH`
when launching multi-round benchmarks. An adapter-only folder is NOT a
standalone vLLM `--model`: the base is loaded through `--model`, and
the adapter through `--enable-lora --lora-modules name=path`.

## 2. Offline held-out single-decision tests (10 examples by default)

From repository root, in an environment with vLLM and the project
dependencies installed:

```bash
python baseline/evaluate_lora.py --method da-sft --dry-run
python baseline/evaluate_lora.py --method da-grpo --dry-run

python baseline/evaluate_lora.py --method da-sft --gpu 2 --max-samples 10
python baseline/evaluate_lora.py --method da-grpo --gpu 2 --max-samples 10
```

The adapter selection is independent for each method; both read their
respective `baseline/<method>/data/test.parquet` held-out split.

Outputs:
- `baseline/da-sft/output/eval_lora_test.json`
- `baseline/da-grpo/output/eval_lora_test.json`

Metrics: parseable JSON, valid Orchestra-o1 action, binary action agreement.
**This is one-step imitation / action agreement, NOT task success.**
An expert may have followed one valid path among several.

For full held-out single-turn data, pass `--max-samples 0`.
`--base-model /path`, `--adapter /path`, `--test-parquet /path`,
`--gpu N` and `--port N` support experiment overrides.

The two services use ports 8810 (SFT) / 8811 (GRPO), and default to
GPU 2 as on the reference workstation. A 24-GB GPU generally cannot run
two 8B BF16 base servers concurrently. Run comparisons **sequentially**.
Newly launched servers are stopped on normal program exit; an already
running compatible vLLM endpoint is reused but not killed.

## 3. Real multi-round Orchestra-o1 / OmniGAIA tests

The launchers set local model and adapter paths automatically:

```bash
bash baseline/orchestra-o1/bench_qwen/run_da_sft_lora.sh --tasks 5
bash baseline/orchestra-o1/bench_qwen/run_da_grpo_lora.sh --tasks 5
```

`--tasks` selects specific benchmark task IDs (not dataset row numbers);
omit it to run the fixed pretest-60 split. Inspect
`data/OmniGAIA/splits/pretest_60_ids.txt` for valid IDs.

For the fixed final-100 split:

```bash
CONFIG_PATH=baseline/orchestra-o1/config/omnigaia/final/orchestra_o1_da_sft_lora.yaml \
  bash baseline/orchestra-o1/bench_qwen/run_da_sft_lora.sh

CONFIG_PATH=baseline/orchestra-o1/config/omnigaia/final/orchestra_o1_da_grpo_lora.yaml \
  bash baseline/orchestra-o1/bench_qwen/run_da_grpo_lora.sh
```

The launchers also accept `MODEL_PATH` and `ADAPTER_PATH`.
For remote Sub-Agent/tool/Judge calls, configure `DASHSCOPE_API_KEY`
(or the normal project remote-API credentials) as you already do for
Orchestra-o1.

GPU, context, concurrent sequences, LoRA rank and ports live in
`config/model_config.yaml`; a benchmark `models:` override can change
`gpus` without changing the global registry.

## 4. Qwen3 Thinking and JSON

Both named DA services use:

```yaml
reasoning_parser: qwen3
max_completion_tokens: 4096
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

The Qwen3 template enables the thought segment. vLLM's reasoning parser
returns the thought in `message.reasoning_content` and the **final
action JSON** in `message.content`. Orchestra-o1 consumes ONLY final
content; the private thought is not substituted for the decision's own
`reasoning` JSON field. A response that exhausts the thinking budget and
never produces final content is reported as an error.

The default `max_model_len=16384` includes both visible history and the
thinking/completion budget. For long O1 histories, reduce completion or
prompt budgets as appropriate; do not silently truncate expert evidence.

The server validates that the adapter is a saved PEFT checkpoint, its
LoRA rank <= configured `max_lora_rank=32`, and (for local known paths)
its recorded base model matches the served base. It will refuse to reuse
a same-named service whose reported LoRA root is a different path.
If you **retrain and overwrite** an adapter at the same path, restart its
running vLLM service so the updated weights are loaded.

### Why not merge LoRA into the base model?

A merged full checkpoint is unnecessary: online LoRA leaves the original
BF16 Qwen3-8B weights unchanged and makes SFT/GRPO model selection
explicit. Existing merged-checkpoint `qwen3-8b-grpo-local` config is
preserved for backward compatibility; use `qwen3-8b-da-grpo-lora` for
the new adapter-only deployment.
