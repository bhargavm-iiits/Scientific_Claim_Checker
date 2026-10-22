"""Five-fold development-only checks of a numeric verifier ensemble.

Only predictions/retrieval scores enter features; all gold fields are targets.
This reports out-of-fold scores, not resubstitution accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_jsonl, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("SUPPORT", "CONTRADICT", "NEI")


def features(llm, nli, retrieval):
    result = [float(llm["verdict"] == label) for label in LABELS]
    decisions = llm["doc_verdicts"]
    for hit in retrieval[:3]:
        result.extend(float(decisions.get(hit["doc_id"], "NEI") == label) for label in LABELS)
        result.append(hit["score"])
        for view in ("abstract", "selected"):
            contexts = [r for r in nli["contexts"] if r["doc_id"] == hit["doc_id"] and r["view"] == view]
            result.extend(max((c["scores"][label] for c in contexts), default=float(label == "NEI"))
                          for label in LABELS)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    llm = {r["claim_id"]: r for r in read_jsonl(ROOT / "artifacts/runs/dev_v2_frozen/predictions.jsonl")
           if r["variant"] == "full_v2_nothink"}
    nli = {r["claim_id"]: r for r in read_jsonl(args.folder / "predictions.jsonl")}
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    ids = load_split(load_config(), "dev_tune")["claim_ids"]
    if set(nli) != set(ids) or not set(ids) <= set(llm):
        raise ValueError("Complete dev_tune predictions required")
    x = np.array([features(llm[c], nli[c], retrieval[c]) for c in ids])
    y = np.array([nli[c]["gold"] for c in ids])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for c in (0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
        pred = cross_val_predict(model, x, y, cv=cv)
        results.append({"model": f"logistic_C{c}", **verdict_metrics(list(y), list(pred))})
    for depth in (1, 2):
        model = HistGradientBoostingClassifier(max_depth=depth, max_iter=75, l2_regularization=2.0, random_state=42)
        pred = cross_val_predict(model, x, y, cv=cv)
        results.append({"model": f"hist_depth{depth}", **verdict_metrics(list(y), list(pred))})
    write_json(args.folder / "ensemble_cross_validation.json", results)
    print(json.dumps([{k: r[k] for k in ("model", "accuracy", "macro_f1")} for r in results], indent=2))


if __name__ == "__main__":
    main()
