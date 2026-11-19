"""Fit six A-Trust-style logistic classifiers on white-box attention features.

Input JSONL rows:
 {"task_entries":[...], "report_id":"r1.1",
  "labels":{"factuality":1,"logical_consistency":1,"relevance":1,
            "completeness":1,"clarity":1,"source_support":1}}
All labels must be independently curated/visible-history grounded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from method import DIMENSIONS, attention_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--attention-model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from reliability_core import public_reports
    from sklearn.linear_model import LogisticRegression
    import joblib
    import numpy as np
    data, labels = [], {k: [] for k in DIMENSIONS}
    for number, line in enumerate(args.train.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        reports = public_reports(obj["task_entries"])
        matching = [r for r in reports if r["id"] == obj["report_id"]]
        if len(matching) != 1:
            raise ValueError(f"row {number}: missing/duplicate report_id")
        target = obj["labels"]
        if set(target) != set(DIMENSIONS):
            raise ValueError(f"row {number}: must label all six dimensions")
        for k in DIMENSIONS:
            if target[k] not in (0, 1):
                raise ValueError(f"row {number}: binary label required for {k}")
            labels[k].append(int(target[k]))
        data.append(attention_features(matching[0], reports, args.attention_model))
    if len(data) < 10:
        raise ValueError("At least 10 labeled samples required")
    X = np.stack(data)
    args.output.mkdir(parents=True, exist_ok=True)
    # Fit on this single, fixed feature extraction template (see method.py).
    for name in DIMENSIONS:
        y = np.asarray(labels[name])
        if set(y) != {0, 1}:
            raise ValueError(f"Both labels required for {name}")
        model = LogisticRegression(class_weight="balanced", max_iter=2000,
                                   random_state=17)
        model.fit(X, y)
        joblib.dump(model, args.output / (name + ".joblib"))
    joblib.dump({"attention_model": args.attention_model,
                 "dimensions": list(DIMENSIONS),
                 "feature_count": int(X.shape[1]), "rows": len(X)},
                args.output / "metadata.joblib")
    print(json.dumps({"rows": len(X), "feature_count": int(X.shape[1]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
