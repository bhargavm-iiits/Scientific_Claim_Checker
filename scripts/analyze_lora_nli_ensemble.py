"""Development-only numeric ensemble diagnostic with out-of-fold predictions."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("SUPPORT", "CONTRADICT", "NEI")


def features(llm, nli, hits):
    values = [float(llm["verdict"] == label) for label in LABELS]
    documents = {d["doc_id"]: d for d in llm["documents"]}
    for hit in hits[:3]:
        decision = documents.get(hit["doc_id"], {}).get("verdict", "ERROR")
        values.extend(float(decision == label) for label in LABELS)
        values.append(hit["score"])
        contexts = [c for c in nli["contexts"] if c["doc_id"] == hit["doc_id"] and c["view"] == "abstract"]
        values.extend(max((c["scores"][label] for c in contexts), default=float(label == "NEI")) for label in LABELS)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lora_folder", type=Path)
    p.add_argument("--nli-folder", type=Path, default=ROOT / "artifacts/improvement/nli_scifact_train_v2_epoch3_dev120")
    args = p.parse_args()
    ids = load_split(load_config(), "dev_tune")["claim_ids"]
    llm = {r["claim_id"]: r for r in read_jsonl(args.lora_folder / "predictions.jsonl")}
    nli = {r["claim_id"]: r for r in read_jsonl(args.nli_folder / "predictions.jsonl")}
    if set(llm) != set(nli) or set(llm) != set(ids):
        raise ValueError("Exactly the complete frozen development population is required")
    hits = {r["claim_id"]: r["hits"] for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    x = np.asarray([features(llm[c], nli[c], hits[c]) for c in ids])
    y = np.asarray([nli[c]["gold"] for c in ids])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for c in (0.1, 1.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
        predictions = cross_val_predict(model, x, y, cv=cv)
        results.append({"model": f"logistic_C{c}", **verdict_metrics(list(y), list(predictions))})
    report = {"population": "dev_tune120", "method": "Five-fold out-of-fold; diagnostic on reused development data, not validation accuracy",
              "lora_predictions_sha256": sha256_file(args.lora_folder / "predictions.jsonl"),
              "nli_predictions_sha256": sha256_file(args.nli_folder / "predictions.jsonl"), "results": results,
              "oracle_choice_of_two_components_accuracy": sum(llm[c]["verdict"] == nli[c]["gold"] or nli[c]["verdict"] == nli[c]["gold"] for c in ids) / len(ids)}
    write_json(args.lora_folder / "nli_ensemble_diagnostic.json", report)
    print([{key: r[key] for key in ("model", "accuracy", "macro_f1")} for r in results])
    print("Non-deployable component-choice ceiling:", report["oracle_choice_of_two_components_accuracy"])


if __name__ == "__main__":
    main()
