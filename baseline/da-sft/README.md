# DA-SFT — Unsloth + TRL SFTTrainer (MAESTRO Main-Agent)

Offline supervised imitation of **one original Orchestra-o1 Main-Agent turn**.
Unlike DA-GRPO, training consumes the original complete `turn["output"]` JSON
(action + reasoning + params). A GRPO Parquet alone is **insufficient** to
reconstruct the full SFT label: `expert_params` does not contain reasoning.

No change is made to `baseline/da-grpo`, its Judge, or Stage 3/4 generation.

## 1. Prepare data

From the MAESTRO repository root, first create the existing DA-GRPO Parquet
with `python -m dataset.prepare_da_grpo` (see `dataset/README.md`).
Then join the original source turns back to the identical O1 prompts:

```bash
python -m dataset.prepare_da_sft --dry-run
python -m dataset.prepare_da_sft
```

If GRPO data also contains Stage 4, you **must** supply the exact Stage 4
turn file used for its export:

```bash
python -m dataset.prepare_da_sft \
  --untrusted-turns environment/data/maestro_untrusted_orchestration_100/main_turns.jsonl
```

For custom runs, pass `--grpo-parquet`, `--base-turns`, `--scenes`,
`--untrusted-turns`, `--output-dir`, and `--eval-ratio` as needed.

The conversion enforces generation-manifest checks and byte-for-byte prompt
agreement with `o1_prompt_from_turn`; it verifies the GRPO expert metadata
against the original decision and fails on duplicates/drift. The default label
filter accepts `llm_reviewed`, `llm_reviewed_completion`, and
`llm_reviewed_contract_frozen`. Use `--include-fallback` for a separate
ablation involving deterministic/template actions.

Outputs: `dataset/data/da-sft/{train.parquet,eval.parquet,metadata.jsonl,summary.json}`.
Only `prompt` (one user message) and `completion` (one assistant JSON message)
are stored in the training Parquets. The JSONL sidecar records sample IDs,
source/mode, Scene, and action. Train/eval are **grouped by original Scene ID**,
so Stage 3 rounds and every Stage 4 recovery branch of the same Scene cannot
leak across splits. Each split must contain at least one Scene.

## 2. Install and validate

Set up a fresh environment (Python 3.11+ recommended) to avoid changing the
active DA-GRPO / vLLM installation.

```bash
python -m pip install -r baseline/da-sft/requirements.txt
python -m unittest discover -s dataset -p 'test_prepare_da_sft.py'
python -m unittest discover -s baseline/da-sft -p 'test_sft.py'
python baseline/da-sft/train.py --dry-run
python baseline/da-sft/train.py --check
```

`--check` uses only the tokenizer/CPU: it rejects samples for which the
original prompt exceeds `max_prompt_tokens` or the **full formatted dialogue**
exceeds `max_seq_length`. There is no silent prompt/answer truncation.

## 3. Single-GPU BF16 LoRA

```bash
python baseline/da-sft/train.py --max-steps 1 \
  --output-dir baseline/da-sft/output_smoke
bash baseline/da-sft/run.sh
```

For more aggressive memory saving, test QLoRA separately:

```bash
python baseline/da-sft/train.py --load-in-4bit \
  --output-dir baseline/da-sft/output_qlora
```

Continue a **DA-SFT TRL** checkpoint (not an older GRPO checkpoint):

```bash
python baseline/da-sft/train.py \
  --resume baseline/da-sft/output/checkpoint-100
```

The default is Qwen3-8B / BF16 LoRA r=32, 16K total context, 12K prompt
budget, gradient checkpointing, 8-bit AdamW, micro-batch 1, accumulation 8,
two epochs. No vLLM, FlashInfer sampler, KL, Judge calls or GRPO rollout
buffer is created. We keep packing **off** initially to validate token
boundaries and the actual completion masks. Enabling packing later requires
attention-kernel and completion-mask checks.

The dataset uses TRL's **conversational prompt-completion** contract with
`completion_only_loss=True`: user history is context, not supervised text.
At startup `check_mask` inspects the actual processed TRL dataset and
fails if the prompt is not masked or if no answer tokens are supervised.
Qwen3 uses the model tokenizer's original chat template and
`<|im_end|>` as the assistant EOS. The example labels contain the complete
JSON only; they do **not** provide a separate chain-of-thought. Keep inference
chat-template thinking settings consistent in controlled comparisons with
GRPO; do not assume the model learned an explicit thinking trace.

## 4. Evaluating against DA-GRPO

Compare held-out Scene groups, parseable O1 JSON, valid actions, matching
tool sets, decision quality (the same 4D Judge used for GRPO), completion
length and inference latency. **Training loss alone is not an orchestration
metric.** The SFT LoRA adapter can later be used to warm-start GRPO as an
additional ablation, but optimizer checkpoints are not interchangeable.

No end-to-end GPU run can be verified without your local Qwen3-8B weights,
datasets and RTX 6000D environment.
