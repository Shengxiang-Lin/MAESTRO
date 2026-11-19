# VerifyMAS: online-prefix hypothesis verification adapter

Paper: VerifyMAS: Hypothesis Verification for Failure Attribution in
LLM Multi-Agent Systems, https://arxiv.org/abs/2605.17467
Official code: https://github.com/mala-lab/VerifyMAS

Original method diagnoses complete failed trajectories, with zero-shot and
SFT verifier variants. This adaptation uses its **hypothesis-first** design
on ONLY the currently visible O1 prefix: first generate typed error
hypotheses tied to report IDs, then check each against already delivered
evidence using entailed/contradicted/neutral judgments. An entailed *error
hypothesis* triggers advisory verification, not automatic attribution.

No paper-specific fine-tuned checkpoint is loaded. This is a training-free
online adaptation using the same Main LLM (two additional calls per round).

    python baseline/reliability_benchmark.py --method verifymas \
      --max-tasks 3 --options '{"max_hypotheses":8}'
