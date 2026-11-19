# Five reliability baselines for online Orchestra-o1

All five methods have independent implementations under baseline/:
- a-trust (attention-based message trust; small trained classifiers)
- omni-decision (zero-training Critic + Evidence Ledger component)
- propuq-mas (propagation-aware UQ)
- hassum (semantic entropy/density uncertainty routing)
- verifymas (online-prefix error-hypothesis verifier)

These are **paper-inspired online adaptations**, not full replications of
the authors' training, datasets or benchmark claims. See each README.

Run from repository root with the existing O1 benchmark and config:

    python baseline/reliability_benchmark.py --method omni-decision \
        --config baseline/orchestra-o1/config/omnigaia/pretest/orchestra_o1_omnigaia.yaml \
        --max-tasks 3 --max-concurrency 1

Use --method a-trust, omni-decision, propuq-mas, hassum or verifymas.
Method options are a JSON object, e.g. --options '{"threshold":0.65}'.
Note: a-trust requires local white-box weights and trained classifiers.

    python -m unittest discover -s baseline -p 'test_reliability_baselines.py'

The original orchestra-o1/ and da-grpo/ and da-sft/ sources are unchanged.
Reviewer inputs comprise only Main-delivered task_entries. Every extra call
uses the Main model backend and contributes to its logged token/cost usage.
The same original O1 action schema and Sub-Agent runtime are retained.
Output: logs/reliability_baselines/METHOD/TIMESTAMP/{manifest.json,
results.csv,trajectories/,review/}. No faults or private oracle labels
are exposed to the runtime model.

Fairness: report task success, recovery, unnecessary verifications, added
LLM calls, total input/output tokens, cost and latency. HASSUM uses
text-only resampling unless actual worker samples are supplied. The original
VerifyMAS is **offline attribution**, not a trained online planner. The
Omni-Decision baseline here implements **Critic+Ledger only**, not State-SFT
or its RL planner. A-Trust depends on actual white-box attention; do not
replace missing attention with fabricated risk scores.
