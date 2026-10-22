"""Aggregate the unchanged300-claim baseline; never construct training examples."""
from collections import Counter
from pathlib import Path

from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    folders = [ROOT / "artifacts/runs" / name for name in ("nli_titlefamily_recheck_v1", "nli_titlefamily_holdout_v1")]
    rows = [r for folder in folders for r in read_jsonl(folder / "predictions.jsonl")]
    assert len(rows) == len({r["claim_id"] for r in rows}) == 300
    metrics = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
    assert metrics == read_json(ROOT / "artifacts/improvement/final_title_verifier_results.json")["pooled_validation_300"]
    errors, availability = Counter(), Counter()
    for row in rows:
        decisive = row["gold_label"] != "NEI"
        evidence = set(row["evidence_doc_ids"])
        examined = set(row["examined_doc_ids"] or [])
        retrieved = row["retrieved_doc_ids"] or []
        if decisive:
            availability["decisive_claims"] += 1
            for name, ids in (("retrieved_top3", retrieved[:3]), ("retrieved_top10", retrieved[:10]), ("examined_after_gate", examined)):
                availability[name] += bool(evidence & set(ids))
        if row["correct"]:
            continue
        if not decisive:
            errors["gold_NEI_predicted_decisive_or_error"] += 1
        elif not evidence & set(retrieved[:3]):
            errors["decisive_error_missing_annotated_evidence_in_top3"] += 1
        elif not evidence & examined:
            errors["decisive_error_annotated_evidence_removed_by_gate"] += 1
        else:
            errors["decisive_error_despite_examining_annotated_evidence"] += 1
    report = {"metrics": metrics, "error_categories": dict(errors), "annotated_evidence_availability": dict(availability),
              "strict_95percent_accuracy_requires_correct": 286, "net_additional_correct_needed": 286 - metrics["correct"],
              "prediction_hashes": {folder.name: sha256_file(folder / "predictions.jsonl") for folder in folders},
              "limitations": "Reused validation diagnostic, not a blind test. Missing annotated evidence does not prove absence of other valid evidence. No labels or cases are exported for training."}
    write_json(ROOT / "artifacts/improvement/scifact95_baseline_audit.json", report)
    print({"correct": metrics["correct"], "errors": dict(errors), "evidence_availability": dict(availability)}, flush=True)


if __name__ == "__main__":
    main()
