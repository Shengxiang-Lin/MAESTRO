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
KV, offloaded/quantized/paged caches and vLLM are not supported. Stage 2's
primary CLI uses Unsloth QLoRA plus TRL GRPOConfig/HF Trainer scheduling,
with a custom risk-KV rollout and streaming loss (NOT stock GRPOTrainer's
rollout engine). The policy suffix path keeps gradients only for Anchor
and Action tokens, so the long frozen prefix is not replayed during LoRA backward.

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
      --load_in_4bit \
      --max_length 16384 --window_size 128 --window_stride 96

    python -m train_maestro_8b.extract_features \
      --model /path/to/Qwen3-8B \
      --input environment/data/stage4/private/window_risk/val.jsonl \
      --output_dir workspace/cache/risk-val \
      --load_in_4bit

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

**Stage2 NF4 parity:** when using Unsloth QLoRA, extract Stage1 frozen
features with --load_in_4bit as above. Newly saved Risk Head checkpoints
record feature_model_quantization (nf4/none/unknown). Unsloth Stage2 fails
closed on non-NF4/unknown feature provenance. Old Stage1 checkpoints must
be retrained from verifiably NF4-extracted features before this backend is used.

## Stage 2 runtime: Unsloth + TRL (default)

The default `python -m train_maestro_8b.train_grpo` now launches the
**Unsloth FastLanguageModel + TRL GRPOConfig / HF Trainer** integration.
This is a deliberately customized GRPO training_step, not the unmodified
`trl.GRPOTrainer`, because both rollout and new-logprob evaluation must
consume the SAME risk-steered prefix cache. The previous plain PyTorch
reference is retained in `train_grpo_reference.py`.

Install the CUDA training backend on the Qwen3 machine, in an isolated
environment with a compatible CUDA / Torch combination:

    python -m pip install -r train_maestro_8b/requirements-unsloth.txt

Official reference: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide

**Do not turn on vLLM / fast_inference** in this version: a separate vLLM
text-generation backend will bypass the custom DynamicCache. Unsloth's
FastLanguageModel accelerated QLoRA and TRL's Trainer lifecycle are used,
but **vLLM's speedups are not claimed**. The trainer supports ONE CUDA
process; rollout completions are evaluated sequentially to limit KV memory.

Before training, validate the GPU/Unsloth implementation with:

    python -m train_maestro_8b.train_grpo \
      --model /path/to/Qwen3-8B \
      --risk_checkpoint workspace/checkpoints/maestro-risk-head \
      --train_file environment/data/stage4/private/risk_grpo/train.jsonl \
      --output_dir workspace/checkpoints/maestro-risk-policy \
      --backend unsloth --preflight_only

This probe checks that Risk-Aware Prefill yields a valid DynamicCache,
rollout logprobs match teacher-forced replay and gradients affect only
LoRA weights. If Unsloth's patched forward fails to preserve this contract,
**do not disable the probe and continue training**; use --backend hf as a
controlled fallback or fix the underlying version compatibility.

TRL GRPOConfig uses gradient accumulation divisible by the number of
completions. For example, `--num_generations 4 --gradient_accumulation_steps 4`
means 4 completions for EACH of four prompt microbatches per update.
The streaming backward keeps only one action-completion graph alive at
a time. The default reward is the existing deterministic decision-quality
reward, not a teacher string-matching score.

## Stage 2 — risk-steered LoRA policy

Join risk examples with exact-candidate, PRIVATE Stage-4 decision-time
reward states (the Teacher Action is retained only for optional SFT warmup):

    python -m train_maestro_8b.export_grpo \
      --risk_dir environment/data/stage4/private/window_risk \
      --main_turns environment/data/stage4/main_turns.jsonl \
      --decision_states environment/data/stage4/private/decision_reward_states.jsonl \
      --output_dir environment/data/stage4/private/risk_grpo

**Important:** decision_reward_states.jsonl is generated by the updated
Stage-4 trajectory builder *before* each Main action, then independently
replayed after Stage-4B/4C review so the reward scorer uses the FINAL
natural-language evidence actually delivered to Main. For **older Stage4
journals**, use the deterministic audited-attempt replay instead of
expensive language-model regeneration:

    python -m train_maestro_8b.backfill_reward_states \
      --stage4 environment/data/stage4 \
      --scenes environment/data/stage1/questions/scenes.jsonl \
      --max_parallel_tasks 4 --max_retries 1 --model_alias model_1

This reads accepted Stage4 journals and previously delivered reports.
It DOES NOT call an LLM, regenerate evidence, or infer unchosen tasks from
Teacher text. It replays stored attempt versions through the *same* state
machine and validates the recorded Teacher dispatches. If the old journal
does not contain sufficient information, it fails closed and writes
private/reward_backfill_failures.jsonl; the previous sidecar is preserved.
Use exactly the original Stage4 scenes, parallel cap and retry budget.
For very large journals the replay is CPU/disk-bound, not an LLM run.

Cases requiring a private, Main-unobservable fault diagnosis or
an unreleased future-evidence event not announced to Main are marked
unscorable and excluded from RL.

Before running GRPO, inspect whether Teacher actions themselves
can be bound deterministically and what their achievable rewards are:

    python -m train_maestro_8b.inspect_rewards \
      --train_file environment/data/stage4/private/risk_grpo/train.jsonl \
      --output_jsonl workspace/reward_reference_audit.jsonl

A high invalid-reference fraction indicates scope/field binding drift or a
data-export mismatch; investigate it before wasting GPU time.

Run a small *offline plumbing* pilot:

    python -m train_maestro_8b.train_grpo \
      --model /path/to/Qwen3-8B \
      --risk_checkpoint workspace/checkpoints/maestro-risk-head \
      --train_file environment/data/stage4/private/risk_grpo/train.jsonl \
      --output_dir workspace/checkpoints/maestro-risk-policy \
      --backend unsloth --load_in_4bit --max_length 16384 \
      --steer_last_n 8 --steer_strength 0.1 \
      --num_generations 4 --gradient_accumulation_steps 4 \
      --max_prompts 20 --max_steps 2 \
      --reward_debug_jsonl workspace/checkpoints/maestro-reward-debug.jsonl

An SFT warmup, when enabled, updates **only LoRA**, never the Risk Head.
The **default GRPO reward is now a deterministic current-round
orchestration quality score**, not Teacher action text similarity:

    R = 0.35 * evidence_coverage
      + 0.25 * evidence_priority
      + 0.25 * legal_parallel_occupancy
      + 0.15 * requested_field_focus

Unsupported/ambiguous evidence-to-action bindings, duplicate assignments,
wrong tool/source scope, over-capacity batches, same-batch dependencies and
premature completion receive -1. Successful legal actions score 0..1.
A legal supported completion receives 1 for the STOP decision, **not**
for final answer correctness. The reward uses the frozen DAG and the
current pre-action state. It never compares natural-language instructions
to Teacher instructions or calls an LLM judge, and does not require
executing arbitrary candidate actions in future simulation rounds.

Binding is deliberately conservative: each task instruction must explicitly
name the required source scope (when any) and target evidence fields and
match the authorized tools/model. Semantically equivalent paraphrases
of typed fields are supported (e.g. project_id / project id). If multiple
goals remain indistinguishable, it fails closed rather than guesses a
private Macro task ID. Monitor invalid-action and zero-advantage ratios.

The weights are V1 research hyperparameters, not proven optimal. Parallel
efficiency uses a combinatorially computed feasible maximum within the
current round, not a naive max_parallel_tasks=4 denominator. A task is
legal only when its prerequisites and in-round repair constraints hold.
Current source/check status must be Main-observable; hidden fault causes
are never rewarded as if the policy had seen them.

Add --reward_debug_jsonl PATH to log actual candidate actions, bound
internal task IDs, per-component rewards, and failure reasons for audits.
The legacy text imitation proxy is available only when EXPLICITLY selected:

    --reward_callable train_maestro_8b.rewards:teacher_proxy_reward

External research reward implementations remain supported:

    --reward_callable your_package.reward_module:score

The callable receives (candidate_text, record) and returns a finite float.
None of these offline local-decision rewards should be reported as final
end-to-end environment recovery success: evaluating that requires a
separate action-executing online simulator.

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
    python -m unittest train_maestro_8b.test_orchestration_reward \
      train_maestro_8b.test_candidate_alignment -v
    python -m unittest train_maestro_8b.test_grpo_trainer -v
    # With TRL / Transformers / PEFT installed on CPU:
    python -m unittest train_maestro_8b.test_trl_integration -v

The last test instantiates a tiny Qwen3 on CPU, requiring Transformers and
PEFT; it tests shared frozen-prefix caching, exact sampling/teacher-forced
log-prob agreement, and that only LoRA receives Stage-2 gradients.

Full Qwen3-8B GPU throughput, cache-steering gain, long-history temporal
robustness and real Environment GRPO improvements are NOT established until
evaluated on your own training machine.
