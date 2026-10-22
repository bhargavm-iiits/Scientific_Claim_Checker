"""Check retrieval-relevance gating using internal development predictions only."""
from pathlib import Path

from analyze_verifier import decide

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_jsonl, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "artifacts/improvement/nli_scifact_train_v2_epoch3_dev120"
    rows = read_jsonl(folder / "predictions.jsonl")
    expected = set(load_split(load_config(), "dev_tune")["claim_ids"])
    assert {r["claim_id"] for r in rows} == expected
    retrieval = {r["claim_id"]: {h["doc_id"]: h["score"] for h in r["hits"]}
                 for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    results = []
    for cutoff in (None, -5.0, -2.0, 0.0, 2.0, 4.0):
        predictions = []
        for row in rows:
            filtered = {**row, "contexts": [c for c in row["contexts"]
                                            if cutoff is None or retrieval[row["claim_id"]][c["doc_id"]] >= cutoff]}
            predictions.append(decide(filtered, view="abstract", k=3, threshold=0.5))
        result = verdict_metrics([r["gold"] for r in rows], predictions)
        results.append({"minimum_retrieval_score": cutoff, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    write_json(folder / "development_relevance_gate.json", results)
    print(results)


if __name__ == "__main__":
    main()
