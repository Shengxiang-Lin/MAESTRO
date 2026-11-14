# MAESTRO 8B — two-stage latent evidence-risk orchestration

This is the proposed trainable Main-Agent implementation **outside baseline/**.
It preserves the existing Stage-4 observable-claim data export and O1 public
Main prompt. No private Gold, injected-fault identity, hidden dependency
diagnostic, or teacher action is appended to the model-visible prompt.

## Architecture

Stage 1: **Frozen full Qwen3-8B -> window features -> Risk Head only.**

- Each Main decision sees the original question and all earlier Sub-Agent
  Result, Summary and Trace summary text (up to 16K tokens, no silent cut).
- Each returned report is split into 128-token windows with 96-token stride;
  windows never cross report boundaries.
- A single frozen 8B forward gives local window vectors and the final
  full-context vector. The trainable Risk Head outputs per-window warning
  logits; report-wise top-k pooling supports masked weak-label BCE.
- **No LoRA in Stage 1.** The saved artifact is risk_head.pt and risk_config.json.
- Risk = observable conflict, missing/partial return, or similar visible
  warning. Risk=0 does not mean truth; unparseable or ambiguous feedback is
  masked. MIL scores are not verified fine-grained fact localizations.
- Feature extraction can be performed once and sharded to BF16 tensors to
  train the head repeatedly without rerunning the full 8B.

Stage 2: **Frozen base + frozen Risk Head -> steered KV -> LoRA-only decoder.**

- Run the **same original full 8B** with adapters disabled over the Main
  prompt; collect final-layer states and a frozen prefix DynamicCache.
- Reuse Stage-1 Risk Head. Map window scores to source-token spans; keep
  non-report tokens at zero risk.
- Adjust only the cached **Values** in the selected last N attention
  layers with a bounded risk threshold/strength. No cache Keys or original
  prompt text are modified. This implementation tests positive steering;
  it must be compared with no-risk/random-risk and negative steering.
- A fixed newline Decision Anchor is fed **after** risk steering with LoRA
  enabled, ensuring even the first generated action token uses the risk
  signal. Later generated tokens continue with a private mutable cache.
- The original Qwen weights, Risk Head and frozen prefill remain fixed.
  **Only the PEFT LoRA parameters are updated** by the Stage-2 policy loss.
- Sampling and differentiable action-token log probability calculations
  share RiskAwarePolicy, including the same anchor, cache treatment, and
  temperature, to avoid off-policy errors.

The implementation currently uses standard Hugging Face DynamicCache. Static
KV, offloaded/quantized/paged caches and vLLM are not supported. The policy
suffix path keeps gradients only for Anchor and Action tokens, so the long
frozen prefix is not replayed during LoRA backward.

## Step 0 — export observational risk examples

    python -m train_maestro_8b.export_reliability \
      --output_dir environment/data/stage4/private/window_risk

Inputs should already exist from the canonical Stage1-4 generator.
The exporter intentionally only uses globally recovered and language-reviewed
Stage-4 branches, because these are the branches released in the accepted
claim-state supervision file. Reports that do not yield dependable labels
are masked. Check summary.json for positive, negative and mask coverage.

## Stage 1 — frozen risk learning (recommended cached workflow)

Run feature extraction separately for training and validation, with the exact
same frozen model/tokenizer and window hyperparameters:

    python -m train_maestro_8b.extract_features \
      --model /path/to/Qwen3-8B \
      --input environment/data/stage4/private/window_risk/train.jsonl \
      --output_dir workspace/cache/risk-train \
      --max_length 16384 --window_size 128 --window_stride 96

    python -m train_maestro_8b.extract_features \
      --model /path/to/Qwen3-8B \
      --input environment/data/stage4/private/window_risk/val.jsonl \
      --output_dir workspace/cache/risk-val

Train only the small MLP head:

    accelerate launch -m train_maestro_8b.train_reliability \
      --model /path/to/Qwen3-8B \
      --features_dir workspace/cache/risk-train \
      --eval_features_dir workspace/cache/risk-val \
      --output_dir workspace/checkpoints/maestro-risk-head \
      --epochs 3 --batch_size 8

The online no-cache alternative is:

    accelerate launch -m train_maestro_8b.train_reliability \
      --model /path/to/Qwen3-8B \
      --train_file environment/data/stage4/private/window_risk/train.jsonl \
      --eval_file environment/data/stage4/private/window_risk/val.jsonl \
      --output_dir workspace/checkpoints/maestro-risk-head \
      --load_in_4bit --batch_size 1

The cached approach needs disk space (about ~1 MiB per 128-window context at
BF16/4096 dimensions), but repeated epochs of Risk Head training are fast.
Train/val should be split by scene/world, not individual history prefix.

## Stage 2 — risk-steered LoRA policy

Join the risk examples with accepted Stage-4 teacher decisions:

    python -m train_maestro_8b.export_grpo \
      --risk_dir environment/data/stage4/private/window_risk \
      --main_turns environment/data/stage4/main_turns.jsonl \
      --output_dir environment/data/stage4/private/risk_grpo

Run a small *offline plumbing* pilot:

    python -m train_maestro_8b.train_grpo \
      --model /path/to/Qwen3-8B \
      --risk_checkpoint workspace/checkpoints/maestro-risk-head \
      --train_file environment/data/stage4/private/risk_grpo/train.jsonl \
      --output_dir workspace/checkpoints/maestro-risk-policy \
      --load_in_4bit --max_length 16384 \
      --steer_last_n 8 --steer_strength 0.1 \
      --sft_warmup 1 --num_generations 4 --max_prompts 20

An SFT warmup, when enabled, updates **only LoRA**, never the Risk Head.
The default offline reward is an intentionally limited JSON/action-matching
*teacher proxy*, **NOT** a simulator recovery reward. An actual online
recovery RL run requires an action-executing environment evaluator:

    --reward_callable your_package.reward_module:score

The callable receives (candidate_text, record) and returns a finite float.
The current Stage-4 offline trajectory producer does not provide a general
step() method for arbitrary natural-language Main assignments; implementing
that online adapter remains separate work. Do not report offline
teacher-proxy rewards as recovery success.

## Important consistency / performance conditions

- The Stage-1 and Stage-2 base model *must* be identical. Use the same
  tokenizer, prompt format and Risk Head window/stride configuration.
- The longest prompt is encoded **once** for each decision, but a GRPO group
  requires a private *clone* of the steered prefix KV for each candidate.
  16K KV copies consume significant memory; the reference trainer evaluates
  candidates sequentially.
- The decision anchor is consumed with LoRA after steering, not before.
- For sampling and logprob loss, only the Anchor+Action suffix is forwarded
  through LoRA; the prefix never recomputes under LoRA.
- No vLLM or arbitrary-action online rollout support is claimed.
- Validate that Risk Head learns content conflicts rather than only
  detecting words like "partial", and compare no-risk/random/true-risk
  steering before attributing policy gains to risk representations.

## Tests

    python -m compileall -q train_maestro_8b
    python -m unittest train_maestro_8b.test_reliability \
      train_maestro_8b.test_runtime_alignment -v
    python -m unittest train_maestro_8b.test_risk_model -v
    python -m unittest train_maestro_8b.test_risk_policy -v

The last test instantiates a tiny Qwen3 on CPU, requiring Transformers and
PEFT; it tests shared frozen-prefix caching, exact sampling/teacher-forced
log-prob agreement, and that only LoRA receives Stage-2 gradients.

Full Qwen3-8B GPU throughput, cache-steering gain, long-history temporal
robustness and real Environment GRPO improvements are NOT established until
evaluated on your own training machine.
