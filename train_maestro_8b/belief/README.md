# Stage 1 Pilot — Evidence Belief Pretraining (Qwen3-8B + LoRA)

This is an **experimental, GPU-trainable Stage1** of MAESTRO. It does NOT
replace the existing frozen-backbone `risk/` experiment and does NOT change
the proven Stage2 DA/Unsloth/vLLM/TRL training path by default.

## Objective

Train **the SAME Qwen3-8B LoRA projections** that Stage2 can optionally
reuse, using observed evidence states rather than hidden fault correctness.

* Input: exact Main-visible Orchestra-o1 question and accumulated prior
  Sub-Agent result/summary. The same Qwen3 user chat-template wraps the text.
* Labels: from accepted Stage4 `claim_state_supervision.jsonl` only;
  **unverified / corroborated / disputed / partial**. Conflicting and
  intra-report-conflicting claims share the `disputed` class.
* Extra lightweight objective: when a *previously observed claim group*
  changes status because of the latest batch, label that update. We **do
  not** label first observation as a belief revision.
* Unknown/unparsed/superseded reports have `belief_label=-100` and
  are excluded. Reports with incompatible class labels are also excluded.
* The entire Stage1 task is **non-generative**. It does not train Action SFT
  yet; this pilot isolates whether implicit evidence understanding can be
  learned. Stage2 still generates normal O1 Actions.
* Backbone base weights frozen, LoRA + one small shared belief Head trainable;
  optimizer backpropagates through the live LoRA hidden states.
* The Head combines EACH report's span representation with the FINAL
  full-history token representation so an earlier report can be re-evaluated
  when new evidence arrives.

The old Risk Head was trained on cached frozen features; that is NOT
what happens in this pilot.

## 1. Export exact Stage4-aligned data

Run from MAESTRO repository root with existing Stage1/Stage4 artifacts:

```bash
bash train_maestro_8b/run/pipeline.sh belief-export
cat environment/data/stage4/private/evidence_belief/summary.json
```

Equivalent direct command:

```bash
python -m train_maestro_8b.data.export_belief \
  --scenes environment/data/stage1/questions/scenes.jsonl \
  --episodes environment/data/stage4/episodes.jsonl \
  --states environment/data/stage4/private/claim_state_supervision.jsonl \
  --output_dir environment/data/stage4/private/evidence_belief
```

Files: `train.jsonl`, `val.jsonl`, `summary.json`. Prompt/report offsets
are re-used from `train_maestro_8b.data.export_risk.render_samples`.
Decisions are joined to **exact** candidate branch+round, and the original
risk exporter’s masked labels prevent unsupported conclusions. Same Scene
and all of its Stage4 counterfactual branches go into the SAME split.
No attack labels, gold values or future reports are copied into the
training prompt.

Audit `status_*` counts, `update_supervised` and `unknown_masked`.
If there is no corroborated data, the pilot cannot establish that the
model learned independent-source validation: gather more such examples
before making that research claim.

## 2. Train a small GPU pilot

Install `torch`, `transformers`, `peft`, `accelerate`,
`bitsandbytes`, `safetensors` (versions compatible with current
Stage2 environment).

```bash
BELIEF_MAX_STEPS=20 BELIEF_EPOCHS=2 \
  bash train_maestro_8b/run/pipeline.sh belief-train
```

Or direct, with all options:

```bash
python -m train_maestro_8b.belief.train \
  --model base_models/Qwen3-8B \
  --train_file environment/data/stage4/private/evidence_belief/train.jsonl \
  --val_file environment/data/stage4/private/evidence_belief/val.jsonl \
  --output_dir workspace/maestro_8b/checkpoints/belief-stage1-pilot \
  --max_length 8192 --epochs 2 --max_steps 20 \
  --batch_size 1 --grad_accum 4 --lr 5e-5 --head_lr 2e-4 \
  --lora_rank 32 --lora_alpha 64 --load_in_4bit
```

Default is NF4 to conserve GPU memory; `--bf16_full` optionally uses
BF16 backbone if GPU headroom permits. **No prompt truncation**: data
longer than `max_length` is counted and skipped. The loss includes
class-balanced report-state CE and 0.3× masked status-update CE.
A missing/zero LoRA gradient stops training. Start with max_length 8192;
raise to 12288 only if OOM headroom and coverage justify it.

Artifacts:

```text
workspace/maestro_8b/checkpoints/belief-stage1-pilot/
  adapter/adapter_model.safetensors
  adapter/adapter_config.json
  belief_head.pt
  manifest.json
```

The manifest records class labels, train/eval sample counts, validation
loss, state accuracy, macro F1 over present classes, update accuracy,
training steps, model and adapter settings. No oracle truth or raw
prompts are saved into the checkpoint.

## 3. Optional Stage2 transfer

**Do not set this until the Stage1 pilot successfully trains.**

```bash
STAGE1_BELIEF_ADAPTER=workspace/maestro_8b/checkpoints/belief-stage1-pilot/adapter \
  bash train_maestro_8b/run/pipeline.sh grpo
```

The existing Stage2 `train_maestro_8b/policy/da_backend.py` creates
its normal Unsloth LoRA, then checks and loads the Stage1 adapter tensors
into it. Stage2 **does not** use the Head, predicted risk scores, edited
KV or a different Action output. Model base/rank/alpha, tensor names and
shapes must match, and weight values are checked after loading.
Use a new Stage2 output directory; never resume an unrelated optimizer
checkpoint with a different initialization.

Leave `STAGE1_BELIEF_ADAPTER` empty for the pure Global GRPO control.

## Limits of this first version

* Four classes aggregate claim-level evidence into report-level labels.
  Multiple incompatible statuses per report are masked. It is NOT an
  inference of whether any hidden claim is true.
* Belief-update loss currently uses only status transitions visible from
  the latest delivered report batch, not a full memory model.
* It does not yet generate matched counterfactual pairs or include Action
  SFT; adding those before a basic positive state/transfer result would
  unnecessarily complicate attribution.
* CPU regression tests validate exports, offsets, masking and tiny
  Qwen3+LoRA gradients. **Real 8B GPU training and Stage2 warm-start
  still require user-side smoke tests.**
