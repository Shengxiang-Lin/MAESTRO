"""Primary Stage-2 CLI: Unsloth + TRL with MAESTRO Risk-KV rollouts.

The old single-GPU pure PyTorch training loop is retained separately in
train_grpo_reference.py for baselines, but is NOT the default CLI.
"""
from .train_grpo_unsloth import main

if __name__ == "__main__":
    main()
