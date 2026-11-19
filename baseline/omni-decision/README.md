# Omni-Decision: Critic + Evidence Ledger online adapter

Paper: Omni-Decision: Evidence-Ledger Planning for Omni-Modal Agents,
https://arxiv.org/abs/2607.11433 (use latest revised version).

Current implementation: a metered LLM Critic extracts typed evidence claims
from Main-visible Sub-Agent outputs. A deterministic reducer groups identical
(subject,attribute,scope) assertions, flags conflicting values and open
needs, and appends an advisory ledger risk note to the unchanged O1 Planner.

This is the **zero-training Critic+Ledger component**, NOT the full
Omni-Decision Planner with State-SFT / Decision-RL. Independent source
confirmation is not silently assumed; agreement alone is not proof.
Ledger is rebuilt from the currently delivered history each round.

    python baseline/reliability_benchmark.py --method omni-decision --max-tasks 3

Extra cost: one Critic call per Main turn after first return.
Options: max_chars (default 22000), applied to reviewed history.
